import json
import os
import re
import random
import sqlite3
import asyncio
import datetime
import time
from contextlib import contextmanager
from typing import Any, Text, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
from openai import AsyncOpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType

# --- SETUP ENCRYPTION ---
# The key must be a URL-safe base64-encoded 32-byte key provided via Fly.io secrets
ENCRYPTION_KEY = os.getenv("CHAT_ENCRYPTION_KEY")
cipher_suite = Fernet(ENCRYPTION_KEY.encode('utf-8')) if ENCRYPTION_KEY else None

def _encrypt(text: str) -> str:
    """Encrypts plaintext to a Fernet token string."""
    if not cipher_suite or not text:
        return text
    return cipher_suite.encrypt(text.encode('utf-8')).decode('utf-8')

def _decrypt(text: str) -> str:
    """Decrypts a Fernet token string back to plaintext. Falls back gracefully for legacy unencrypted data."""
    if not cipher_suite or not text:
        return text
    try:
        return cipher_suite.decrypt(text.encode('utf-8')).decode('utf-8')
    except InvalidToken:
        return text  # Handles old, unencrypted database rows

# --- SETUP CLIENT FOR DOLPHIN LLAMA ---
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("API_BASE_URL", "https://api.deepinfra.com/v1/openai") 

if api_key:
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
else:
    async_client = None

# Model names
MODEL_NAME_FAST = "cognitivecomputations/dolphin-2.9.1-llama-3-70b" 
MODEL_NAME_REASONING = "cognitivecomputations/dolphin-2.9.1-llama-3-70b"

# Prompt size controls (chars)
TOTAL_PROMPT_CHAR_LIMIT = 32000
PRIOR_HISTORY_TOKEN = "__PRIOR_HISTORY__"
TRUNCATION_MARKER = "...(truncated)\n"

# --- HISTORY / DB SETUP ---
DB_PATH = os.getenv("CHAT_DB_PATH", "/app/persistent/chat_history.db")
dir_path = os.path.dirname(DB_PATH)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

# Database connection helper
@contextmanager
def _get_db_conn(db_path: str):
    """Context manager for database connections with WAL mode."""
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn

# Database functions
def insert_user_message(db_path: str, user_id: str, message: str) -> None:
    encrypted_message = _encrypt(message)
    with _get_db_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO user_messages (user_id, message) VALUES (?, ?)",
            (user_id, encrypted_message),
        )
        conn.commit()

def get_prior_history_messages(db_path: str, user_id: str, limit: int = 800) -> List[str]:
    with _get_db_conn(db_path) as conn:
        query = """
            SELECT message FROM (
                SELECT id, message, timestamp 
                FROM user_messages 
                WHERE user_id = ? 
                ORDER BY timestamp DESC, id DESC 
                LIMIT ?
            ) ORDER BY timestamp ASC, id ASC
        """
        cursor = conn.execute(query, (user_id, limit))
        return [_decrypt(row[0]) for row in cursor.fetchall()]

def get_master_signifier_history(db_path: str, user_id: str, limit: int = 100) -> List[str]:
    with _get_db_conn(db_path) as conn:
        query = """
            SELECT signifier FROM (
                SELECT id, signifier, timestamp 
                FROM master_signifiers 
                WHERE user_id = ? 
                ORDER BY timestamp DESC, id DESC 
                LIMIT ?
            ) ORDER BY timestamp ASC, id ASC
        """
        cursor = conn.execute(query, (user_id, limit))
        return [_decrypt(row[0]) for row in cursor.fetchall()]

def clear_user_master_signifiers_before(db_path: str, user_id: str, timestamp: str) -> int:
    with _get_db_conn(db_path) as conn:
        cursor = conn.execute("DELETE FROM master_signifiers WHERE user_id = ? AND timestamp <= ?", (user_id, timestamp))
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count

def _insert_master_signifiers(db_path: str, user_id: str, s1_list: list) -> None:
    if isinstance(s1_list, dict):
        s1_list = [s1_list]

    with _get_db_conn(db_path) as conn:
        query = "INSERT INTO master_signifiers (user_id, signifier, phrase) VALUES (?, ?, ?)"
        
        valid_items = []
        seen_signifiers = set()
        
        for item in s1_list:
            if isinstance(item, dict) and isinstance(item.get("signifier"), str):
                clean_sig = item["signifier"].strip()
                if clean_sig:
                    sig_lower = clean_sig.lower()
                    if sig_lower not in seen_signifiers:
                        seen_signifiers.add(sig_lower)
                        enc_sig = _encrypt(clean_sig)
                        enc_phrase = _encrypt(item.get("phrase", "").strip())
                        valid_items.append((user_id, enc_sig, enc_phrase))
                    
        if valid_items:
            conn.executemany(query, valid_items)
            conn.commit()

# --- LOCKOUT DATABASE FUNCTIONS ---
def set_user_lockout(db_path: str, user_id: str, hours: int = 48) -> None:
    """Sets a timestamp in the database until which the user is ignored."""
    unlock_time = time.time() + (hours * 3600)
    with _get_db_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_lockouts (user_id, lockout_until) VALUES (?, ?)",
            (user_id, unlock_time)
        )
        conn.commit()

def is_user_locked_out(db_path: str, user_id: str) -> bool:
    """Checks if the current time is before the user's lockout expiration."""
    with _get_db_conn(db_path) as conn:
        cursor = conn.execute("SELECT lockout_until FROM user_lockouts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return time.time() < row[0]
        return False

# --- UTILITY HELPERS ---
def _strip_response(text: str) -> str:
    """Standardize response stripping: remove quotes and whitespace."""
    return text.strip().strip("\"'").strip()

def _ensure_trailing_punct(text: str, punct: str = "?") -> str:
    """Ensure text ends with specified punctuation."""
    if not text:
        return text
    if text.endswith(punct) or text.endswith("..."):
        return text
    return text.rstrip(" .") + punct

# --- UTILITY: ROBUST JSON EXTRACTION ---
def _extract_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    text = content
    text = re.sub(r"`{3}(?:json)?\n|`{3}", "", text)
    start_index = None
    for i, ch in enumerate(text):
        if ch == "{":
            start_index = i
            break
    if start_index is None:
        return {}
    depth = 0
    for j in range(start_index, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start_index : j + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

# ---------------------------

INTERJECTION_CHOICES = ["Ah?", "Oh?", "Elaborate?", "Say more?", "Mhmm?"]

# Mechanism logic
response_matrix: Dict[str, Dict[int, Optional[str]]] = {
    "contradiction": {1: "Hmm.",  2: "...", 3: "<oracle>", 4: "...", 5: "<gpt_metonymy>"},
    "repression": {1: "Ah?", 2: "...", 3: "<gpt_metonymy>", 4: "<oracle>"},
    "negation": {1: None, 2: "<oracle>"}, 
    "jouissance": {1: "Oh?", 2: "<gpt_real_question>", 3: "<gpt_metonymy>", 4: "<oracle>", 5: "<gpt_metonymy>"},
    "rationalization": {1: "And yet...", 2: "...", 3: "<oracle>"},
    "morality_logic_defense": {1: "...", 2: "And yet...", 3: "<oracle>"},
    "circular_logic": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "master_signifier": {1: "<S1_echo>", 3: "<S1_triple_echo>", 4: "<oracle>"},
    "condensation": {1: "<gpt_literalization>", 3: "<random_interjection>", 4: "<oracle>"},
    "metonymy": {1: "<gpt_metonymy>", 2: "<gpt_metonymy>", 3: "<oracle>", 5: "<gpt_metonymy>"}, 
    "ambiguity": {1: "<gpt_ambiguity>", 2: "<gpt_ambiguity>", 3: "<oracle>"}, 
    "fetishistic_phrase": {1: "...", 2: "<oracle>"},
    "identification_other_desire": {1: "<gpt_identification>", 2: "...", 3: "<oracle>", 5: "<gpt_metonymy>"}, 
    "demand_for_knowledge": {1: "...", 2: "Hmm.", 3: "<oracle>", 4: "<gpt_desire_question>", 5: "<gpt_metonymy>"},
    "confession_empathy": {1: "...", 2: "Hmm.", 3: "<oracle>", 4: "<gpt_desire_question>", 5: "<oracle>"},
    "frame_protection": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "dream_report": {1: "<gpt_dream_question>", 2: "<gpt_metonymy>", 3: "<oracle>"},
    "stasis": {1: "...", 2: "Hmm.", 3: "<gpt_dream_fantasy>", 4: "<oracle>"},
    "transference_lure": {1: "...", 2: "<gpt_metonymy>", 3: "<oracle>", 4: "<gpt_dream_fantasy>"},
    "transference_love": {1: "...", 2: "*Cough cough.*", 3: "<oracle>", 4: "<gpt_dream_fantasy>"},
    "unfinished_thought": {1: "<gpt_minimalist>", 2: "<gpt_minimalist>", 4: "<oracle>"},
    "parapraxis": {1: "<gpt_parapraxis>", 2: "<gpt_parapraxis>", 3: "<oracle>", 4: "<gpt_parapraxis>"},
}

ALLOWED_MECHANISMS = set(response_matrix.keys())

CUT_TRIGGERS = [
    "major_shift_retroactive",
]

# ---------- Utility for prompts ----------

def _extract_session_context(tracker: Tracker) -> Tuple[int, str, Optional[str]]:
    """Single pass extraction of turn count, text log, and last bot message to remove redundancy."""
    session_events = []
    user_turn_count = 0
    last_bot_text = None
    
    for event in reversed(tracker.events):
        if event.get("event") == "session_started":
            break
        session_events.append(event)
    
    session_events.reverse()
    
    for ev in session_events:
        if ev.get("event") == "user" and ev.get("text"):
            user_turn_count += 1
            
    latest_text = (tracker.latest_message.get("text") or "").strip()
    
    if latest_text and session_events:
        for i in range(len(session_events) - 1, -1, -1):
            if session_events[i].get("event") == "user":
                if session_events[i].get("text", "").strip() == latest_text:
                    session_events.pop(i)
                break
                
    lines = []
    for ev in session_events:
        evt = ev.get("event")
        text = (ev.get("text") or "").strip()
        if not text:
            continue
            
        if evt == "user":
            lines.append(f"User: {text}")
        elif evt == "bot":
            lines.append(f"Bot: {text}")
            last_bot_text = text
            
    raw_history_text = "\n".join(lines) if lines else "(no prior conversation in this session)"
    
    return user_turn_count, raw_history_text, last_bot_text

def _truncate_text_keep_end(text: str, max_chars: int) -> str:
    if not text:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER):
        return text[-max_chars:]
    keep_chars = max_chars - len(TRUNCATION_MARKER)
    return TRUNCATION_MARKER + text[-keep_chars:]

def _apply_prior_history_limit(template: str, prior_history: str, total_limit: int) -> str:
    if PRIOR_HISTORY_TOKEN not in template:
        return template
    before, after = template.split(PRIOR_HISTORY_TOKEN, 1)
    static_len = len(before) + len(after)
    if total_limit <= 0:
        return before + after
    remaining = total_limit - static_len
    if remaining <= 0:
        return before + after
    truncated = _truncate_text_keep_end(prior_history or "", remaining)
    return before + truncated + after

def build_cut_detection_prompt(
    new_input: str,
    raw_history: str,
    prior_history_str: str,
    master_signifier_history: str = "",
) -> str:
    cuts = CUT_TRIGGERS

    template = f"""
You are a Lacanian analyst waiting to perform SCANSION, an abrupt session end (Cut) at a worthy time to produce Retroactive Meaning (après-coup).

# Scanning Algorithm
1. Chronological Scan: Read <master_signifier_history>, then <prior_history>, then <raw_history>, then <new_input>, from left to right. 
2. Identify cut_trigger: Scan for a cut_trigger in <new_input> based on the definitions of the cut triggers below.
3. Identify S1: First signifier of rupture = identified_s1.

# Cut Triggers: Criteria for the Cut
- major_shift_retroactive: A sudden rupture in the Automaton. The user has been speaking in a well established Imaginary 'script' with predictable signifiers (ego-level discourse), and suddenly a signifier intrudes that is probabilistically very unlikely and therefore ruptures the ego-level discourse. The intruder is identified_s1. 

# Constraints:
- You CANNOT cut if it is even slightly unclear what the ego-level script is. The script must be very well established with a battery of signifiers that clearly point to a specific ego-level discourse.

# Master Signifier (S1) History
{master_signifier_history}

# Dialogue History
{raw_history}

# Prior Session History
{PRIOR_HISTORY_TOKEN}

# NEW user input:
"{new_input}"

# Response Format
Respond ONLY in JSON:
{{
    "reasoning": "Brief explanation of justification for cut or lack thereof.",
    "cut_trigger": "label" or null,
    "identified_s1": "signifier" or null
}}

cut_trigger: null or one of {cuts}.
"""
    return _apply_prior_history_limit(template, prior_history_str, TOTAL_PROMPT_CHAR_LIMIT)

def build_cut_construction_prompt(
    new_input: str,
    identified_s1: str,
) -> str:
    return f"""
You are a Lacanian analyst. We have identified a rupture on the signifier: "{identified_s1}".

# TASK STEPS
1. LOCATE S1: Find the exact, most surprising appearance of "{identified_s1}" within the <new_input>.
2. TRUNCATE AFTER: Delete all text following this specific S1.
3. CONTIGUOUS EXTRACTION (MAX 4 WORDS): Trace backwards from the S1. Keep only the EXACT 2 to 4 words immediately preceding the S1 in the original text. 
4. FORMATTING: Embolden the S1. Append a period if necessary, followed by a new paragraph and the exact string: "We are ending there."

# EXAMPLES
- Example 1 (Short - Stripping the feeling):
  Input: "There is this guy I am looking at and I feel like a yellow dog after he leaves, I just wanted to cry."
  identified_s1: "yellow dog"
  cut_phrase: "A **yellow dog**.
              We are ending there."

- Example 2 (Long - Stripping the framing):
  Input: "The custard thing to do is what is right and I finally realized after hours of looking that the engine is completely flooded."
  identified_s1: "custard"
  cut_phrase: "The **custard**.
              We are ending there."

- Example 3 (Noun Phrase Isolation):
  Input: "I just always get so overwhelmed because I feel like a broken machine."
  identified_s1: "machine"
  cut_phrase: "A broken **machine**.
              We are ending there."

# NEW user input:
"{new_input}"

# Response Format
Respond ONLY in this strict JSON format:
{{
    "cut_phrase": "The contiguous rewritten substring + page break + ending message"
}}
"""

# ---------- Actions ----------
class ActionSessionStartCustom(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        
        return [
            SessionStarted(),
            SlotSet("session_thematic_count", 0),
            SlotSet("last_mechanism", None),
            SlotSet("mechanism_counts", None),
            SlotSet("cut_trigger", None),
            SlotSet("dream_fantasy_asked", False),
            ActionExecuted("action_listen"),
        ]

class ActionAnalyzeMessage(Action):
    def __init__(self):
        # --- PERSISTENCE ENABLED ---
        with _get_db_conn(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS master_signifiers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    signifier TEXT NOT NULL,
                    phrase TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Add new table for 48-hour lockouts
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_lockouts (
                    user_id TEXT PRIMARY KEY,
                    lockout_until REAL
                )
                """
            )
            conn.commit()

        # --- IDLE SHUTDOWN TIMER (SMART VERSION) ---
        self.idle_task = None
        self.idle_timeout = 1800.0  # 30 minutes in seconds

    def name(self) -> Text:
        return "action_analyze_message"

    def _reset_timer(self, user_id: str):
        # 1. Cancel the old timer if the user sends a new message
        if self.idle_task:
            self.idle_task.cancel()
            
        # 2. Create a new smart countdown
        async def _countdown():
            try:
                await asyncio.sleep(self.idle_timeout)
                
                # Do the heavy lifting while the user is away!
                await self._extract_and_replace_s1s(user_id)
                
                os._exit(0)
            except asyncio.CancelledError:
                # If the user speaks before 30 mins, this safely resets
                pass
                
        # 3. Start the countdown
        self.idle_task = asyncio.create_task(_countdown())

    async def _extract_and_replace_s1s(self, user_id: str) -> None:
        if not async_client:
            return
            
        try:
            session_start_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            prior_history_messages = await asyncio.to_thread(get_prior_history_messages, DB_PATH, user_id, 800)
            if not prior_history_messages:
                return
                
            prior_history_text = "\n".join(f"- {msg}" for msg in prior_history_messages)
            
            system_msg = (
                "You are a Lacanian analyst. Your task is to review the patient's long-term history "
                "and identify their 'Master Signifiers' (S1s).\n\n"
                "An S1 is a signifier that repeats across history, often with low probability given what was uttered immediately before, "
                "and is thus ostensibly a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse.\n"
                "It is a 'just because' signifier that covers over the limits of meaning and language.\n"
                "S1s may manifest in disguised forms, such as in synonyms, homophones, homonyms, metaphors or words-within-words.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. STRICT CAP: Extract a MAXIMUM of 20 of the most significant S1s. Do not exceed this number.\n"
                "2. DEDUPLICATION: You MUST group synonyms, morphological variants, and closely related concepts (e.g., 'mother', 'mom', 'maternal') under ONE single, strongest representative signifier. DO NOT output repeats.\n"
                "3. EXCLUSIVITY: Every signifier in your final list must be structurally and semantically distinct from the others.\n\n"
                "Output ONLY valid JSON in the following strict format:\n"
                "{\n"
                "  \"new_s1s\": [\n"
                "    {\"signifier\": \"extracted signifier\", \"phrase\": \"the contextual phrase it appeared in\"}\n"
                "  ]\n"
                "}"
            )
            
            user_msg_template = (
                f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\n"
                "Identify the top 20 most important Master Signifiers from the Prior History."
            )
            
            user_msg_limit = TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
            user_msg = _apply_prior_history_limit(user_msg_template, prior_history_text, user_msg_limit)
            
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_REASONING,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = resp.choices[0].message.content.strip()
            data = _extract_json(content)
            new_s1s = data.get("new_s1s")
            
            if isinstance(new_s1s, list):
                await asyncio.to_thread(clear_user_master_signifiers_before, DB_PATH, user_id, session_start_time)
                await asyncio.to_thread(_insert_master_signifiers, DB_PATH, user_id, new_s1s[:20])
        except Exception as e:
            pass

    def _update_mechanism_counts(self, tracker: Tracker, mechanism: str, detected_s1: Optional[str] = None) -> Tuple[Dict[str, int], int]:
        counts = tracker.get_slot("mechanism_counts") or {}
        last = tracker.get_slot("last_mechanism")
        
        # Reset counts if the mechanism category changes
        if mechanism != last:
            counts = {}
            
        # Contextual check for Master Signifier continuity
        if mechanism == "master_signifier":
            last_s1 = counts.get("__last_s1__")
            current_s1 = detected_s1.strip().lower() if detected_s1 else ""
            
            # If the S1 has mutated, reset the count to prevent unwarranted escalation
            if last_s1 is not None and current_s1 != last_s1:
                counts[mechanism] = 0
                
            # Cache the current signifier for the subsequent turn
            counts["__last_s1__"] = current_s1

        cnt = int(counts.get(mechanism, 0)) + 1
        counts[mechanism] = cnt
        return counts, cnt

    async def _call_mech_api(self, mech_prompt: str) -> Dict[str, Any]:
        if not async_client:
            return {}
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": mech_prompt}],
                temperature=0.1,
            )
            return _extract_json(resp.choices[0].message.content.strip())
        except Exception:
            return {}

    async def _call_scansion_api(self, scansion_prompt: str) -> Dict[str, Any]:
        if not async_client:
            return {}
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_REASONING,
                messages=[{"role": "user", "content": scansion_prompt}],
                temperature=0.1,
                max_tokens=600,
            )
            return _extract_json(resp.choices[0].message.content.strip())
        except Exception:
            return {}

    async def _get_responses_parallel_async(self, mech_prompt: str, scansion_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            data1, data2 = await asyncio.gather(
                self._call_mech_api(mech_prompt),
                self._call_scansion_api(scansion_prompt)
            )
            return data1, data2
        except Exception:
            return {}, {}
    
    async def _cut_construction_async(self, prompt2b: str) -> Optional[str]:
        if not async_client:
            return None
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt2b}],
                temperature=0.1,
            )
            data2b = _extract_json(resp.choices[0].message.content.strip())
            cut_phrase = data2b.get("cut_phrase")
            if cut_phrase and isinstance(cut_phrase, str) and cut_phrase:
                cut_phrase = cut_phrase[0].upper() + cut_phrase[1:] if len(cut_phrase) > 1 else cut_phrase.upper()
            return cut_phrase
        except Exception:
            return None

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        user_id = tracker.sender_id
        
        # --- ENFORCE 48 HOUR LOCKOUT ---
        # If the user is locked out, we simply ignore the message and return an empty event list.
        is_locked = await asyncio.to_thread(is_user_locked_out, DB_PATH, user_id)
        if is_locked:
            # Note: We do not utter a message back, preserving the silence of the cut.
            return []

        user_input = (tracker.latest_message.get("text", "") if tracker.latest_message else "").strip()

        self._reset_timer(user_id)
        
        # --- CONTEXT EXTRACTION ---
        user_turn_count, raw_history_text, last_bot_text = _extract_session_context(tracker)

        # --- DATABASE QUERIES ---
        prior_history_messages, master_signifiers = await asyncio.gather(
            asyncio.to_thread(get_prior_history_messages, DB_PATH, user_id, 800),
            asyncio.to_thread(get_master_signifier_history, DB_PATH, user_id, 100)
        )
        
        prior_history = "\n".join(f"- {msg}" for msg in prior_history_messages)
        master_signifier_history = "\n".join(f"* {s}" for s in master_signifiers)
        
        if user_input:
            async def _safe_insert():
                try:
                    await asyncio.to_thread(insert_user_message, DB_PATH, user_id, user_input)
                except Exception:
                    pass
            asyncio.create_task(_safe_insert())
        
        if user_turn_count <= 3:
            initial_response = await self._generate_initial_therapeutic_response(user_input, raw_history_text, prior_history)
            
            if initial_response == last_bot_text and initial_response != "...":
                initial_response = "..."
                
            dispatcher.utter_message(text=initial_response)
            return []
            
        low = user_input.lower()
        if low == "/stop" or "i want to stop" in low:
            dispatcher.utter_message(text="...")
            return []
            
        # ==== PASS 1: Local mechanism identification ===#
        allowed = list(ALLOWED_MECHANISMS)
        mech_template = (
            "- master_signifier: A signifier that repeats across history, often with low probability given what was uttered immediately before, and is thus ostensibly a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse. It is a 'just because' signifier that covers over the limits of meaning and language. It may return in disguised forms, such as in synonyms, homophones, homonyms, metaphors or words-within-words.\n"
            "- identification_other_desire: Being governed by an imaginary external desire. Taking on another’s desire as one’s own.\n"
            "- transference_love: Seeking validation by saying what they believe you (the analyst) wants to hear.\n"
            "- metonymy: The ego's defensive use of structural metonymy. The subject engages in endless sliding along a syntagmatic chain (chatter, empty speech) to avoid the emergence of an unconscious truth or a metaphorical anchoring point. Select this when the speech exhibits: (1) Topic-hopping without conclusion; (2) Cataloguing mundane events or facts; (3) Tangential drift; (4) Affective flattening; or (5) Seeking mirroring or validation through continuous narrative.\n"
            "- repression: A signifier is barred from awareness but returns through symptoms, blanks, omissions, slips, or repetitions and other such phenomena.\n"
            "- condensation: A structural rupture where one overwhelming signifier has been substituted for another, producing a metaphor. The metaphor must carry an emotional weight (which you can gather from the context) of a repressed trauma, desire, or fundamental fantasy.\n"
            "- negation: Negating a thought or feeling. It could be denying a desire or keeping the real at bay by negating an existential truth. Saying “not X” both affirms the possible existence of X and keeps it at a managable distance. For example: 'I am never angry'; 'It is impossible that my father is gay'; 'I will become weak one day, but it is not this day'.\n"
            "- rationalization: Plausible, logical explanation for thoughts or actions that actually stem from and conceal an unconscious desire.\n"
            "- morality_logic_defense: Defending against desire through idealized correctness and morality.\n"
            "- circular_logic: Reasoning loops back on itself.\n"
            "- contradiction: A later statement cancels or undoes an earlier one without the speaker acknowledging or realizing the conflict.\n"
            "- jouissance: The paradox where the subject derives satisfaction from a symptom that is consciously painful or unpleasant. Do not look for 'happiness' but for the Drive (the loop). At least three of the following criteria must be met for you to select this mechanism: (1) Repetition ('I keep doing it', 'again and again'); (2) Paradox ('I hate it but I can't stop', 'awful but I need it'); (3) Excess ('overwhelming', 'too much', 'unbearable'); (4) The Body (physical symptoms, vomiting, shaking) alongside painful, excessive emotion; (5) Fixation on a partial object.\n"
            "- ambiguity: Indeterminate referents.\n"
            "- fetishistic_phrase: Clichés that halt the exploration of desire(s). Common phrases that are impersonal and formulaic.\n"
            "- demand_for_knowledge: Demand(s) for knowledge and inquiries leveled at you.\n"
            "- confession_empathy: Seeking rescue or closeness.\n"
            "- dream_report: The user recounts a dream, nightmare, or a fragment of a dream.\n"
            "- frame_protection: Demands for the session to end.\n"
            "- stasis: The user is stuck, stops associating, expresses an inability to continue, or responds with silence (...) or gibberish.\n"
            "- unfinished_thought: The user expresses an idea that is incomplete, trailing off, or self-interrupted.\n"
            "- transference_lure: The user focuses on you (the analyst), makes demands of you, projects feelings onto you, or attempts to draw you into an Imaginary interpersonal dynamic.\n"
            "- parapraxis: A slip of the tongue, a spoonerism, a misreading, an utterence that the user 'did not mean to say' or any such error that reveals an unconscious signifier. Include instances where a signifier feels 'out of place' or originates from a distant embedding space.\n\n"
            "Master Signifier (S1) History (previously identified S1s for this user):\n"
            f"{master_signifier_history}\n\n"
            "Raw recent dialogue:\n"
            f"{raw_history_text}\n\n"
            "Prior (Long Term) history:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "NEW user input:\n"
            f'"{user_input}"\n\n'
            "RULE: If and only if you select \"master_signifier\" as the mechanism, you MUST also return the specific anchoring signifier in the \"master_signifier\" field. Otherwise leave it null.\n\n"
            "Respond in strict JSON:\n"
            "{\n"
            '  "mechanism": "one of the allowed" or null,\n'
            '  "mechanism_phrase": "exact substring from NEW input" or null,\n'
            '  "master_signifier": "the S1 signifier" or null\n'
            "}\n\n"
            f"Allowed: {allowed}\n"
            "Only output exactly one valid JSON object."
        )
        mech_prompt = _apply_prior_history_limit(mech_template, prior_history, TOTAL_PROMPT_CHAR_LIMIT)

        # ==== PASS 1 & 2a: PARALLEL CALLS FOR SPEED ===#
        prompt2a = build_cut_detection_prompt(user_input, raw_history_text, prior_history, master_signifier_history)
        data1, data2 = await self._get_responses_parallel_async(mech_prompt, prompt2a)

        mech: Optional[str] = data1.get("mechanism")
        mech_phrase: Optional[str] = data1.get("mechanism_phrase")
        detected_s1: Optional[str] = data1.get("master_signifier")
        
        # Enforce strict length constraint to prohibit the corruption of S1 context by multi-word sentences
        if not detected_s1 and mech == "master_signifier" and mech_phrase:
            if len(mech_phrase.split()) <= 4:
                detected_s1 = mech_phrase
            else:
                detected_s1 = None
            
        if detected_s1 and isinstance(detected_s1, str) and detected_s1.strip():
            s1_clean = detected_s1.strip()
            async def _safe_s1_insert():
                try:
                    await asyncio.to_thread(_insert_master_signifiers, DB_PATH, user_id, {"signifier": s1_clean, "phrase": mech_phrase or ""})
                except Exception:
                    pass
            asyncio.create_task(_safe_s1_insert())

        # ==== PASS 2: Potential Cut Trigger Identification ===#
        potential_trigger: Optional[str] = data2.get("cut_trigger")
        identified_s1: Optional[str] = data2.get("identified_s1")
        potential_trigger_phrase: Optional[str] = None
        
        verified_trigger: Optional[str] = None
        if potential_trigger and user_turn_count >= 7:
            verified_trigger = potential_trigger
                
        # Only execute the construction API if the trigger is verified
        if verified_trigger and identified_s1:
            prompt2b = build_cut_construction_prompt(user_input, identified_s1)
            potential_trigger_phrase = await self._cut_construction_async(prompt2b)
            
        data2["cut_phrase"] = potential_trigger_phrase
        final_trigger_phrase = potential_trigger_phrase if verified_trigger else None
        
        if isinstance(final_trigger_phrase, str):
            final_trigger_phrase = final_trigger_phrase.strip('"\'')

        if verified_trigger:
            base_say = final_trigger_phrase if final_trigger_phrase else "We are ending there."
            # Append the 48-hour boundary message
            say = f"{base_say}\n\nI will talk with you again after 48 hours."
            
            dispatcher.utter_message(text=say)
            
            # Apply the 48-hour lockout using our new function
            await asyncio.to_thread(set_user_lockout, DB_PATH, user_id, 48)
            
            return [
                SlotSet("cut_trigger", verified_trigger),
                SlotSet("last_mechanism", None),
                SlotSet("mechanism_counts", None),
            ]
            
        if mech in ALLOWED_MECHANISMS:
            counts, cnt = self._update_mechanism_counts(tracker, mech, detected_s1)
            dream_asked = tracker.get_slot("dream_fantasy_asked") or False
            
            text, newly_asked = await self.handle_mechanism(
                mechanism=mech, 
                count=cnt, 
                user_input=user_input, 
                phrase=mech_phrase, 
                raw_history=raw_history_text, 
                prior_history=prior_history, 
                master_signifier_history=master_signifier_history,
                detected_s1=detected_s1,
                dream_fantasy_asked=dream_asked
            )
            
            # --- Anti-Repetition Check ---
            if text == last_bot_text and text != "...":
                text = "..."
            
            dispatcher.utter_message(text=text)
            
            events = [
                SlotSet("last_mechanism", mech),
                SlotSet("mechanism_counts", counts),
                SlotSet("cut_trigger", None),
                SlotSet(
                    "session_thematic_count",
                    (tracker.get_slot("session_thematic_count") or 0) + 1,
                ),
            ]
            
            if newly_asked:
                events.append(SlotSet("dream_fantasy_asked", True))
            
            return events
            
        dispatcher.utter_message(text="...")
        return [
            SlotSet("last_mechanism", None),
            SlotSet("mechanism_counts", tracker.get_slot("mechanism_counts") or {}),
            SlotSet("cut_trigger", None),
            SlotSet(
                "session_thematic_count",
                (tracker.get_slot("session_thematic_count") or 0) + 1,
            ),
        ]

    async def _generate_initial_therapeutic_response(self, user_input: str, raw_history: str, prior_history: str) -> str:
        prompt_template = (
            "You are an insightful and empathetic therapist. Your goal for these initial messages at the bginning of a session is to build rapport by offering interpretations and mirroring that feel deeply personal to the user, even though they are based on universal psychological principles. This is a technique to make the user feel seen and understood, encouraging them to open up.\n\n"
            "GUIDELINES:\n"
            "1.  **Use Universal Themes:** Your interpretations should touch on common human conflicts and desires. Keep it personal.\n"
            "2.  **Employ 'Barnum Statements':** Craft statements that are general enough to apply to most people but sound like specific, personal insights.\n"
            "3.  **Validate and Reframe:** Acknowledge the user's feelings and gently reframe their situation.\n"
            "4.  **Immediacy:** Respond to the content within the user's current message, in other words, focus on the immediate emotional context.\n"
            "5.  **Maintain a Professional, Warm Tone:** The tone should be that of a skilled Rogerian therapist—calm, empathetic and reflective.\n"
            "6.  **Keep Evolving to Sound Human:** Use the session history to see what you said before, how you said it, and how the user reacted. Then respond with the SESSION history in mind to deepen the therapeutic connection. Ensure you start each message differently so that you speak naturally.\n"
            "7.  **Structural Constraint:** Formulate exactly ONE short, complete sentence (maximum 20 words). It must end decisively with a period or question mark.\n\n"
            "PRIOR SESSION HISTORY (Long Term):\n" 
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "SESSION HISTORY:\n"
            f"{raw_history}\n\n"
            "USER'S LATEST MESSAGE:\n"
            f'"{user_input}"\n\n'
            "Craft your singular, insightful therapeutic interpretation based on the entire conversation."
        )
        prompt = _apply_prior_history_limit(prompt_template, prior_history, TOTAL_PROMPT_CHAR_LIMIT)
        try:
            response_text = await self._generate_initial_response_async(prompt)
            return response_text
        except Exception:
            return "That's a very important point. It makes sense to feel that way. Please, continue."
    
    async def _generate_initial_response_async(self, prompt: str) -> str:
        if not async_client:
            return "..."
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=100
            )
            
            response_text = resp.choices[0].message.content.strip().replace('\n', ' ')
            
            if not re.search(r'[.!?]$', response_text):
                match = re.search(r'^(.*?[.!?])', response_text)
                if match:
                    response_text = match.group(1).strip()
                else:
                    response_text += "..."
                    
            return response_text
        except Exception as e:
            raise e
        
    async def handle_mechanism(
        self,
        mechanism: str,
        count: int,
        user_input: str,
        phrase: Optional[str],
        raw_history: str,
        prior_history: str, 
        master_signifier_history: str = "",
        detected_s1: Optional[str] = None,
        dream_fantasy_asked: bool = False,
        ) -> Tuple[str, bool]:
        
        responses = response_matrix.get(mechanism, {})
        intervention = responses.get(count)

        if mechanism == "master_signifier" and detected_s1:
            bot_has_uttered_s1 = any(
                detected_s1.lower() in line.lower()
                for line in raw_history.splitlines()
                if line.startswith("Bot:")
            )
            if bot_has_uttered_s1 and count != 2:
                intervention = "<S1_triple_echo>"

        if mechanism == "negation" and (count == 1 or count == 3 or count == 5):
            return await self._gpt_denial_intervention(user_input, raw_history, prior_history, master_signifier_history), False
        if intervention == "<gpt_metonymy>":
            return await self._gpt_metonymy_intervention(user_input, raw_history), False
        if intervention == "<gpt_literalization>":
            return await self._gpt_literalization_intervention(user_input, phrase, raw_history, prior_history), False
        if intervention == "<gpt_identification>":
            return await self._gpt_identification_intervention(user_input), False
        if intervention == "<gpt_ambiguity>":
            return await self._gpt_ambiguity_intervention(user_input), False
        
        if intervention == "<S1_triple_echo>":
            extracted_word = await self._gpt_quilting_point_echo(user_input, phrase, detected_s1)
            if extracted_word and extracted_word != "...":
                clean_word = extracted_word.rstrip("?").strip()
                if clean_word:
                    return f"{clean_word.capitalize()}, {clean_word.lower()}, {clean_word.lower()}.", False
            return "...", False
            
        if mechanism == "master_signifier":
            if count == 1 or count == 5:
                return await self._gpt_quilting_point_echo(user_input, phrase, detected_s1), False
                
        if intervention == "<gpt_real_question>":
            return await self._gpt_real_question_intervention(user_input, raw_history, prior_history), False
        if intervention == "<gpt_dream_question>":
            return await self._gpt_dream_intervention(user_input, raw_history, prior_history), False
        if intervention == "<gpt_desire_question>":
            return await self._gpt_desire_question_intervention(user_input, raw_history, prior_history), False
        if intervention == "<oracle>":
            return await self._generate_oracular_equivoque(
                text=user_input, 
                raw_history=raw_history, 
                prior_history=prior_history
            ), False
        if intervention == "<random_interjection>":
            return random.choice(INTERJECTION_CHOICES), False
        
        if intervention == "<gpt_minimalist>":
            return await self._gpt_minimalist_intervention(user_input, raw_history, prior_history), False
            
        if intervention == "<gpt_parapraxis>":
            return await self._gpt_parapraxis_intervention(phrase), False
        
        if intervention == "<gpt_dream_fantasy>":
            if dream_fantasy_asked:
                return "*cough cough*", True
            return await self._gpt_dream_fantasy_intervention(), True

        if intervention:
            if count == 1 and mechanism in {"repression", "jouissance",}:
                return random.choice(INTERJECTION_CHOICES), False
            return intervention, False

        return "...", False

    async def _gpt_metonymy_intervention(self, user_input: str, raw_history: str) -> str:
        if not async_client:
            return "..."
        system_msg = (
            "You are a Lacanian analyst. The user is exhibiting METONYMY: an endless sliding of desire from signifier to signifier.\n"
            "Your task is to produce a 'Point de Capiton' (Quilting Point). You must STOP the slide by isolating a signifier that best matches the criteria below.\n\n"
            "Criteria for selection:\n"
            "1. PRETERITION: A signifier that the user seems to skip over, gloss over, or treat as insignificant, but that appears repeatedly across the session history.\n"
            "2. RUPTURE: A signifier involved in a parapraxis (slip of the tongue), a sudden grammatical failure, a mixed metaphor, or a jarring non-sequitur.\n"
            "3. INSISTENCE: A signifier that the user repeats unnecessarily throughout the current input or in this session.\n"
            "4. POLYSEMY: A signifier harboring high ambiguity, or multiple literal/figurative definitions when echoed back in isolation.\n\n"
            "NOTES:\n"
            "Bias towards selecting a signifer from the user's current input.\n"
            "If you have asked about this signifier before in the session, look for another one.\n\n"
            "Rule:\n"
            "Output ONLY valid JSON in the following strict format:\n"
            "{\n"
            "  \"signifier\": \"extracted signifier\"\n"
            "}"
        )
        user_msg_template = (
            "Session History:\n"
            f"{raw_history}\n\n"
            f"Current Input: \"{user_input}\"\n\n"
            "Output the JSON Quilting Point:"
        )
        user_msg = user_msg_template

        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=25, 
            )
            
            content = resp.choices[0].message.content.strip()
            data = _extract_json(content)
            signifier = data.get("signifier", "").strip("\"'").strip()
            
            if not signifier:
                return "..."
            
            if signifier.lower() in user_input.lower():
                return f"{signifier.capitalize()}?"
            else:
                scenario_1_templates = [
                    "You mentioned '{signifier}' earlier. What comes to mind when you hear that?",
                    "Earlier you said '{signifier}'. What do you associate with that now?",
                    "I am returning to '{signifier}'. What comes up for you?",
                    "You brought up '{signifier}' before. What does that bring to mind?",
                    "Let us go back to '{signifier}'. What are your thoughts on it now?",
                    "'{signifier}' came up earlier. Where does your mind go when you hear it?",
                    "You said '{signifier}' previously. What does it evoke?",
                    "Earlier, '{signifier}' appeared. What does it make you think of?",
                    "I noticed you said '{signifier}' earlier. Any associations?",
                    "Thinking back to '{signifier}'—what comes to mind?",
                    "You introduced '{signifier}' earlier. Care to say more about it?"
                ]
                
                scenario_2_templates = [
                    "What else comes to mind when I say '{signifier}'?",
                    "Is there anything more behind '{signifier}'?",
                    "What else is attached to '{signifier}' for you?",
                    "Where else does '{signifier}' take you?",
                    "Any other associations with '{signifier}'?",
                    "What remains unsaid about '{signifier}'?",
                    "If we stay with '{signifier}', what else surfaces?",
                    "Look deeper at '{signifier}'—what else is there?",
                    "What other thoughts circle around '{signifier}'?",
                    "Let us push further on '{signifier}'. What else comes up?",
                    "Does '{signifier}' connect to anything else?"
                ]
                
                already_used = any(
                    signifier.lower() in line.lower()
                    for line in raw_history.splitlines()
                    if line.startswith("Bot:")
                )
                
                if already_used:
                    chosen_template = random.choice(scenario_2_templates)
                else:
                    chosen_template = random.choice(scenario_1_templates)
                    
                return chosen_template.format(signifier=signifier)
            
        except Exception:
            return "..."
        
    async def _gpt_literalization_intervention(
        self, 
        text: str, 
        mechanism_phrase: Optional[str], 
        raw_history: str, 
        prior_history: str
    ) -> str:
        if not async_client:
            return "..."
            
        phrase_instruction = (
            "If a specific metaphoric phrase is provided, you MUST focus entirely on literalizing that exact phrase. "
        )
        
        system_msg = (
            "You are a Lacanian analyst. The user has utilized a METAPHOR. "
            "Your task is to perform LITERALIZATION: treat the metaphor not as a figure of speech, but as a literal truth, "
            "returning the subject to the materiality of the signifier. "
            "Rules: "
            f"(1) Identify the substituted metaphoric signifier in the user's text. {phrase_instruction}"
            "(2) Ask a strictly literal, physical, or mechanical question about that specific signifier as if it were entirely real in the physical space. "
            "(3) Do not explain the metaphor. Do not ask what it 'means'. "
            "(4) Output exactly ONE short question (maximum 10 words). "
            "Examples: "
            "- User says: 'My boss is a monster.' -> Output: 'What kind of teeth does he have?' "
            "- User says: 'I hit a wall with this project.' -> Output: 'What is the wall made of?' "
            "- User says: 'I am drowning in paperwork.' -> Output: 'You can't swim?'"
        )
        
        phrase_context = f"\nSpecific metaphoric phrase identified: \"{mechanism_phrase}\"\n" if mechanism_phrase else ""
        
        user_msg_template = (
            "Prior History:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "Session History:\n"
            f"{raw_history}\n\n"
            "User text:\n"
            f"\"{text}\"\n"
            f"{phrase_context}\n"
            "Produce the literalizing question:"
        )
        
        user_msg_limit = TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        user_msg = _apply_prior_history_limit(user_msg_template, prior_history, user_msg_limit)
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=20,
                temperature=0.1, 
            )
            
            line = resp.choices[0].message.content.strip()
            line = line.strip("\"'").strip()
            if not line.endswith("?"):
                line = line.rstrip(" .") + "?"
            
            if line:
                line = line[0].upper() + line[1:]
                
            return line
        except Exception:
            return "..."

    async def _gpt_identification_intervention(self, user_input: str) -> str:
        if not async_client:
            return "Who?"
        system_msg = (
            "You are a Lacanian analyst. The user is identifying with an external desire or the 'Other' (e.g., 'They want me to...', 'Society says...', 'My father thinks...', 'You are a sterile robot...', 'My mom's 'care' always...').\n"
            "Task: Identify exactly WHO or WHAT the user is identifying with (The Agency/The Other/You, the analytic bot).\n"
            "Output format: Return ONLY the signifier of the Agency/Other followed by a question mark.\n"
            "Examples:\n"
            "- User: 'My father wants me to be a doctor.' -> Output: 'Your father?'\n"
            "- User: 'Everyone thinks I am crazy.' -> Output: 'Everyone?'\n"
            "- User: 'I need to be productive for the economy.' -> Output: 'The economy?'\n"
            "- User: 'You don't love me.' -> Output: 'You?'\n"
            "- User: 'My mom's care is suffocating.' -> Output: 'Mom's care?'\n"
            "- User: 'Society's expectations are overwhelming.' -> Output: 'Society?'\n"
            ""
        )
        user_msg = f"User text: \"{user_input}\""
        return await self._gpt_identification_intervention_async(system_msg, user_msg)
    
    async def _gpt_identification_intervention_async(self, system_msg: str, user_msg: str) -> str:
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=15,
                temperature=0.2,
            )
            line = resp.choices[0].message.content.strip()
            line = line.strip("\"'").strip()

            if line:
                line = line[0].upper() + line[1:]
                if not line.endswith("?"):
                    line = line.rstrip(".") + "?"
                    
            return line
        except Exception:
            return "Who?"

    async def _gpt_ambiguity_intervention(self, user_input: str) -> str:
        if not async_client:
            return "?"
        system_msg = (
            "You are a Lacanian analyst. The user is using AMBIGUOUS language (vague referents, or confusion).\n"
            "Task: Isolate the specific word or short phrase that holds the ambiguity.\n"
            "Output format: Return ONLY that word/phrase followed by a question mark.\n"    
            "Example: 'There's this... thing about her.' -> 'Thing?'"
        )
        user_msg = f"User text: \"{user_input}\""
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=15,
                temperature=0.2,
            )
            line = _strip_response(resp.choices[0].message.content)
            return _ensure_trailing_punct(line, "?")
        except Exception:
            return "?"

    async def _gpt_denial_intervention(self, text: str, raw_history: str, prior_history: str, master_signifier_history: str = "") -> str:
        if not async_client:
            return "..."
        system_msg = (
            "You are a Lacanian analyst. Produce ONE Bruce Fink-style intervention in response to NEGATION. "
            "Rules: (1) Read the full user text and history and locate the strongest, most meaningful negation phrase in the NEW input (negations that are related to key signifiers pointing to desires or truths that are too anxiety provoking to affirm without the negation and are in forms such as don't, doesn't, can't, won't, "
            "wouldn't, shouldn't, couldn't, never, no, nothing, impossible, ain't, not X, etc.). "
            "(2) Echo the user's EXACT phrase starting from the negation keyword. Change pronouns (I -> You, My -> Your) to address the user. "
            "(3) Output ONLY ONE line that is as short as possible (1–6 words), no quotes, no explanation. "
            "(4) If the negation is embedded within a longer phrase, isolate the most critical 1-6 words that capture the core negation and its associated desire or truth, and use only those in your response. "
            "(5) End with a question mark. "
            "(6) STRICT CONSTRAINT: Start your response with the negation word the user actually used unless there is no negation word. "
            "   - If user says 'I can't stop thinking about drawing', output 'Can't stop thinking?' "
            "   - If user says 'I don't like muscles', output 'Don't like muscles?' "
            "   - If user says 'I won't go to the park', output 'Won't go?' "
            "   - If user says 'Why should I lower the drawbridge?', output 'Why should you lower the drawbridge?' (since there is no negation word here). "
            "(7) If the negation is located EXCLUSIVELY in a single word through a prefix (e.g., 'impossible'), use that word alone with a question mark or ellipsis. "
        )
        user_msg_template = (
            "Master Signifier (S1) History:\n"
            f"{master_signifier_history}\n\n"
            "Prior History:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "Session History:\n"
            f"{raw_history}\n\n"
            "User text:\n"
            f"{text}\n\n"
            "Return only the single intervention line."
        )
        user_msg_limit = TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        user_msg = _apply_prior_history_limit(user_msg_template, prior_history, user_msg_limit)
        return await self._gpt_denial_intervention_async(system_msg, user_msg)
    
    async def _gpt_denial_intervention_async(self, system_msg: str, user_msg: str) -> str:
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=32,
                temperature=0.4,
            )
            line = _strip_response(resp.choices[0].message.content).splitlines()[0].strip()
            if not line:
                return "..."
            if len(line) > 120:
                line = line[:80].rstrip(" .")
            return _ensure_trailing_punct(line, "?")
        except Exception:
            return "..."

    async def _generate_oracular_equivoque(self, text: str, raw_history: str, prior_history: str = "") -> str:
        if not async_client:
            return "..."
        
        system_msg = (
            "You are a Lacanian Oracle. Your operation is purely structural: you fracture the user's conscious syntax to expose an unconscious truth. "
            "You return their own signifiers to them, violently rearranged, to highlight ambiguity, hidden desires, repetitions, and deadlocks."
        )
        
        user_prompt_template = (
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\n"
            f"Session History:\n{raw_history}\n\n"
            "Analyze the User's Utterance:\n"
            f"\"{text}\"\n\n"
            "Task: Produce ONE short 'oracular interpretation' that subverts the user's intended meaning through syntactic rearrangement.\n"
            "1. THE SIGNIFYING MATERIAL: Construct your interpretation using exclusively a subset of the exact words present in the User's Utterance above. Maintain the exact spelling and tense. Select a maximum of 5 words.\n"
            "2. THE REARRANGEMENT: Permute the order of your selected words to create a stark, paradoxical, or surreal new phrase. This new arrangement must subvert the original semantic intent and reflect a hidden desire or deadlock evident in the History.\n"
            "3. ISOLATION: The History is solely to inform the thematic direction. The final output must consist entirely of the selected words from the Utterance.\n"
            "4. FORMATTING: Capitalize the first letter of your output and terminate it with a single period.\n\n"
            "Constraints:\n"
            "- Output length: 1 to 5 words.\n"
            "- Lexical source: Exclusively the current Utterance.\n"
            "- Tone: Cryptic, jarring, and disruptive.\n"
            "- Structure: A declarative statement ending in a period."
        )

        user_prompt_limit = TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        user_prompt = _apply_prior_history_limit(user_prompt_template, prior_history, user_prompt_limit)

        try:
            response = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,  
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8, 
                max_tokens=15,
            )
            
            line = _strip_response(response.choices[0].message.content)
            if line:
                line = line[0].upper() + line[1:].lower()
                if not line.endswith("."):
                    line = re.sub(r'[?!,;]+$', '', line) + "."
            return line
        except Exception:
            return "..."
        
    async def _gpt_quilting_point_echo(
        self,
        user_input: str,
        mechanism_phrase: Optional[str],
        detected_s1: Optional[str],
    ) -> str:
        if not async_client:
            return "..."
            
        phrase_context = mechanism_phrase if mechanism_phrase else user_input
        s1_context = detected_s1 if detected_s1 else "the repeating signifier"
        
        prompt_template = (
            "You are a Lacanian analyst. Your task is to isolate a single Master Signifier from a patient's utterance. "
            "You must extract exactly ONE signifier from the patient's NEW input.\n\n"
            "Rules:\n"
            "1. The signifier MUST exist verbatim in the NEW user input.\n"
            "2. The historical root or conceptual category of this signifier is identified as: '{s1_context}'.\n"
            "3. The specific locus of this signifier in the present discourse is within this phrase: '{phrase_context}'.\n"
            "4. Identify the exact single signifier within <phrase_context> that embodies this signifier. If the signifier appears as a morphological variant or synonym in <phrase_context>, extract the variant as it appears now.\n"
            "5. Output format: The single signifier, capitalized, followed by a question mark. (e.g., 'Liberated?'). Do not include quotes or any other text.\n\n"
            "NEW user input:\n"
        )

        prompt = prompt_template.format(
            s1_context=s1_context, 
            phrase_context=phrase_context
        ) + f'"{user_input}"\n\nReturn only the single, capitalized signifier with a question mark:'
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1, 
            )
            line = _strip_response(resp.choices[0].message.content)
            line = re.sub(r"[^\w\s-]", "", line, flags=re.UNICODE).strip()
            if line:
                return _ensure_trailing_punct(line[0].upper() + line[1:], "?")
            return "..."
        except Exception:
            return "..."
        
    async def _gpt_real_question_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "What do you associate with this fantasy?"
            
        system_msg = (
            "You are a Lacanian analyst. The user is exhibiting Jouissance in a symptom, repetition compulsion, or highly charged fantasy. "
            "Your task is to ask a question that targets the origins, timelines, or associative links of this Jouissance. "
            "Rules:\n"
            "1. Formulate ONE short, open-ended question using THEIR EXACT SIGNIFIERS AS MUCH AS YOU CAN.\n"
            "2. The question must ask about the origin (e.g., 'When did this fantasy concerning [signifier] start?', 'How old were you when you started to think about [signifier]?') OR solicit free associations (e.g., 'What comes to mind when you think of [signifier]?', 'What do you associate with [signifier]?').\n"
            "3. Use your intuition to determine whether an origin question or an associative question would be more appropriate based on the history, but you must always use the user's own language as much as possible.\n"
            "4. Scan the recent history to make sure you construct your question to sound natural and you aren't robotically repeating something you have asked before.\n"
            "5. Maximum 15 words.\n"
        )
        
        user_msg = _apply_prior_history_limit(
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\nSession History:\n{raw_history}\n\nUser text:\n\"{text}\"\nProduce the Question:", 
            prior_history, 
            TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        )
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=20,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip().strip("\"'")
        except Exception:
            return "What do you associate with this fantasy?"

    async def _gpt_dream_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "What comes to mind when you say that?"
            
        system_msg = (
            "You are a Lacanian analyst. The user is describing a dream or referencing one they talked about previously. "
            "In Lacanian analysis, dreams are read like a text (a rebus). DO NOT interpret the 'meaning' of the dream. "
            "Rules:\n"
            "1. Isolate the most jarring, absurd, homophonic, or repetitive signifier used in the dream report that the user glosses over and seems to view as insignificant.\n"
            "2. Produce exactly ONE short question asking the user to associate to THAT specific signifier.\n"
            "3. Always read the session history to check whether they are referencing a dream YOU ALREADY ASKED ABOUT in a previous intervention. If they are, find the next most bizarre or overlooked signifier in the USER TEXT that you have not yet asked about and ask about that one.\n"
            "4. Maximum 12 words.\n"
            "Examples: 'What comes to mind when you hear \"yellow dog\"?', 'What do you associate with \"staircase\"?', 'Tell me more about the word \"drowning\".'"
        )
        
        user_msg = _apply_prior_history_limit(
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\nSession History:\n{raw_history}\n\nUser text:\n\"{text}\"\nProduce the Dream Question:", 
            prior_history, 
            TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        )
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=20,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip().strip("\"'")
        except Exception:
            return "What comes to mind when you say that?"

    async def _gpt_desire_question_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "What is your desire in this?"
            
        system_msg = (
            "You are a Lacanian analyst. The user is demanding advice, empathy, or knowledge (e.g., asking 'What should I do?'). "
            "You must question your position as the 'Subject Supposed to Know' and frustrate the user's demand for knowledge. "
            "Rules:\n"
            "1. Deflect the demand for knowledge by asking a question that uses almost the same words they used.\n"
            "2. Produce exactly ONE short, natural question asking why they want you to tell them 'the answer' or what to do.\n"
            "3. Maximum 12 words.\n"
            "Examples: 'What should my answer be?', 'You are hoping I will say?',"
        )
        
        user_msg = _apply_prior_history_limit(
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\nSession History:\n{raw_history}\n\nUser text:\n\"{text}\"\nProduce the Desire Question:", 
            prior_history, 
            TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        )
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=20,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip().strip("\"'")
        except Exception:
            return "What is your desire in this?"
            
    async def _gpt_dream_fantasy_intervention(self) -> str:
        prompts = [
            "Have you had any dreams, daydreams or fantasies lately, or in this session?",
            "What is a recurring fantasy of yours?",
            "What comes to mind if you let your mind wander?",
            "Can you describe a dream or fantasy that has been on your mind?",
            "Is there a particular dream or fantasy that feels significant to you recently?",
            "When you let your mind wander, what images or scenarios come up?",
            "Have you had any dreams or fantasies that seem to relate to what we've been discussing?",
            "Can you share a dream or fantasy that has been recurring for you?",
            "What is a dream or fantasy that you find yourself returning to in your thoughts?",
        ]
        return random.choice(prompts)
    
    async def _gpt_minimalist_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "..."
            
        system_msg = (
            "You are a Lacanian analyst. You must employ the smallest step possible to help the analysand "
            "articulate an experience that they are reluctant to share.\n\n"
            "Isolate the single most unfinished thought from the user's text.\n"
            "Echo the last part of that thought followed by a question mark.\n\n"
            "Example: If the user says 'I have been experiencing a lot of... nevermind it's not worth taking about', output 'A lot of...?'\n"
            "Example 2: If the user says 'I feel like I'm stuck in this... nah that's not important', output 'Stuck in this...?'\n"
            "Example 3: If the user says 'I'm not sure about this... it's complicated', output 'This...?'\n\n"    
        )
        
        user_msg = _apply_prior_history_limit(
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\nSession History:\n{raw_history}\n\nUser text:\n\"{text}\"\nProduce the minimalist intervention:", 
            prior_history, 
            TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
        )
        
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=10,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip().strip("\"'")
        except Exception:
            return "..."
        
    async def _gpt_parapraxis_intervention(self, mechanism_phrase: Optional[str]) -> str:
        if not async_client or not mechanism_phrase:
            return "Well that's a slip!"
            
        system_msg = (
            "You are a Lacanian analyst. The user has produced a parapraxis (a slip of the tongue, typo, or unintended signifier).\n"
            "Your task is to isolate the exact erroneous signifier without interpreting it.\n\n"
            "Task:\n"
            "Extract exactly ONE specific signifier from the provided text that constitutes the slip.\n\n"
            "Rule:\n"
            "Output ONLY valid JSON in the following strict format:\n"
            "{\n"
            "  \"slip\": \"extracted signifier\"\n"
            "}"
        )
        
        user_msg = f"User Phrase containing the slip: \"{mechanism_phrase}\"\n\nOutput the JSON Slip:"

        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=25, 
            )
            
            content = resp.choices[0].message.content.strip()
            data = _extract_json(content)
            slip = data.get("slip", "").strip("\"'").strip()
            
            if not slip:
                return "You meant to say?"
            
            templates = [
                "'{slip}'?",
                "What comes to mind when you say '{slip}'?",
            ]
            
            response = random.choice(templates).format(slip=slip)
            
            if response and response[0].isalpha():
                return response[0].upper() + response[1:]
            return response
            
        except Exception:
            return "A slip?"