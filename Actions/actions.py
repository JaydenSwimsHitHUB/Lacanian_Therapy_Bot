import asyncio
import datetime
import json
import logging
import os
import random
import re
import sqlite3
import sys
import threading
import time

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Text, Tuple

# --- THIRD-PARTY IMPORTS ---
from cryptography.fernet import Fernet, InvalidToken
from g2p_en import G2p
from openai import AsyncOpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.events import (
    ActionExecuted,
    EventType,
    ReminderScheduled,
    SessionStarted,
    SlotSet,
)
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# --- STATIC STOP WORDS ---
# A lightweight set to prevent basic articles and pronouns from triggering false positive homophones.
STOP_WORDS = {"a", "an", "the", "and", "but", "or", "on", "in", "with", "is", "was", "to", "for", "it", "of", "my", "i"}

# --- SETUP ENCRYPTION ---
ENCRYPTION_KEY = os.getenv("CHAT_ENCRYPTION_KEY")
cipher_suite = Fernet(ENCRYPTION_KEY.encode('utf-8')) if ENCRYPTION_KEY else None

# --- GLOBAL ACTION SERVER MONITOR ---
GLOBAL_LAST_ACTIVITY = time.time()

def action_server_idle_monitor():
    """Monitors global activity and kills the Action Server after 30 minutes of total silence."""
    while True:
        time.sleep(10)
        elapsed = time.time() - GLOBAL_LAST_ACTIVITY
        if elapsed > 1800.0:  # 30 minutes
            sys.stdout.write("[MONITOR] Action server idle for 30 minutes. Initiating termination.\n")
            sys.stdout.flush()
            os._exit(0)

def _encrypt(text: str) -> str:
    if not cipher_suite or not text:
        return text
    return cipher_suite.encrypt(text.encode('utf-8')).decode('utf-8')

def _decrypt(text: str) -> str:
    if not cipher_suite or not text:
        return text
    try:
        return cipher_suite.decrypt(text.encode('utf-8')).decode('utf-8')
    except InvalidToken:
        return text 

# --- SETUP CLIENT FOR DOLPHIN LLAMA ---
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("API_BASE_URL", "https://api.deepinfra.com/v1/openai")
llm_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "90"))

if api_key:
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=llm_timeout)
else:
    async_client = None

MODEL_NAME_FAST = os.getenv(
    "MODEL_NAME_FAST",
    "deepseek-ai/DeepSeek-V4-Flash",
)

TOTAL_PROMPT_CHAR_LIMIT = int(os.getenv("TOTAL_PROMPT_CHAR_LIMIT", "12000"))
PRIOR_HISTORY_PROMPT_LIMIT = int(os.getenv("PRIOR_HISTORY_PROMPT_LIMIT", "100"))
PRIOR_HISTORY_TOKEN = "__PRIOR_HISTORY__"
TRUNCATION_MARKER = "...(truncated)\n"
PRIOR_HISTORY_ENABLED = True

# --- HISTORY / DB SETUP ---
DB_PATH = os.getenv("CHAT_DB_PATH", "/app/persistent/chat_history.db")
dir_path = os.path.dirname(DB_PATH)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

@contextmanager
def _get_db_conn(db_path: str):
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        yield conn

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
def set_user_lockout(db_path: str, user_id: str, hours: int = 48, base_time: Optional[float] = None) -> None:
    if base_time is None:
        base_time = time.time()
    lockout_until = base_time + (hours * 3600)
    with _get_db_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_lockouts (user_id, lockout_until) VALUES (?, ?)",
            (user_id, lockout_until)
        )
        conn.commit()

def is_user_locked_out(db_path: str, user_id: str) -> bool:
    with _get_db_conn(db_path) as conn:
        cursor = conn.execute("SELECT lockout_until FROM user_lockouts WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            if time.time() < row[0]:
                return True
            else:
                conn.execute("DELETE FROM user_lockouts WHERE user_id = ?", (user_id,))
                conn.commit()
        return False

# --- UTILITY HELPERS ---
def _strip_response(text: str) -> str:
    return text.strip().strip("\"'").strip()

def _ensure_trailing_punct(text: str, punct: str = "?") -> str:
    if not text:
        return text
    if text.endswith(punct) or text.endswith("..."):
        return text
    return text.rstrip(" .") + punct

def _format_two_paragraph_message(first_paragraph: str, second_paragraph: str) -> str:
    first = re.sub(r"\s+", " ", (first_paragraph or "").strip())
    second = re.sub(r"\s+", " ", (second_paragraph or "").strip())
    if first and second:
        return f"{first}\n\u200b\n{second}"
    return first or second

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

# --- G2P PHONETIC ARCHITECTURE ---
g2p = G2p()

@lru_cache(maxsize=2048)
def _get_phonemes(text: str) -> List[str]:
    """Translates text to ARPAbet phonemes and strips stress markers."""
    if not text:
        return []
    raw_phonemes = g2p(text)
    return [p.strip('012') for p in raw_phonemes if p.strip() and p.isalnum()]

def _phoneme_edit_distance(l1: List[str], l2: List[str]) -> int:
    """Calculates Levenshtein distance cleanly on string arrays to avoid C-extension hacks."""
    m, n = len(l1), len(l2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): 
        dp[i][0] = i
    for j in range(n + 1): 
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if l1[i-1] == l2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def _check_phoneme_sub_array(chunk_phonemes: List[str], s1_phonemes: List[str]) -> bool:
    """Checks if the contiguous S1 phoneme sequence exists within a larger chunk."""
    if not s1_phonemes or not chunk_phonemes:
        return False
    s1_len = len(s1_phonemes)
    for i in range(len(chunk_phonemes) - s1_len + 1):
        if chunk_phonemes[i:i+s1_len] == s1_phonemes:
            return True
    return False

def _is_acoustic_homophone(chunk_phonemes: List[str], s1_phonemes: List[str]) -> bool:
    """Applies the proportional Levenshtein edit distance logic to phoneme arrays."""
    if not s1_phonemes or not chunk_phonemes:
        return False
    dist = _phoneme_edit_distance(chunk_phonemes, s1_phonemes)
    if len(s1_phonemes) < 4:
        return dist == 0
    return dist == 0

def _scan_for_master_signifier(user_input_lower: str, user_words: List[str], master_signifiers_sorted: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, str]]]:
    s1_phonemes_map = {s1: _get_phonemes(s1) for s1 in master_signifiers_sorted}
    
    for s1 in master_signifiers_sorted:
        if not s1.strip():
            continue
            
        s1_lower = s1.lower()
        s1_ph = s1_phonemes_map[s1]
        if not s1_ph:
            continue
            
        # STEP 1: Single-Word Analysis (Catches exact, morphological, and internal hidden sounds)
        for word in user_words:
            word_ph = _get_phonemes(word)
            
            # Condition A: Hidden phonetic sound inside a larger word (e.g., S1 "sun" inside "asunder")
            if _check_phoneme_sub_array(word_ph, s1_ph):
                if word == s1_lower or _is_acoustic_homophone(word_ph, s1_ph):
                    return "master_signifier", word, word, None
                return "master_signifier", s1_lower, s1_lower, {"chunk": word, "s1": s1_lower}
                
            # Condition B: Direct morphological variant or homophone (e.g., "sun" vs "son" vs "suns")
            if _is_acoustic_homophone(word_ph, s1_ph):
                # Reject meaningless stopword homophones (unless it IS the S1)
                if word in STOP_WORDS and word != s1_lower:
                    continue
                return "master_signifier", word, word, None

        # STEP 2: Cross-Boundary Analysis (Sliding Window 2-4 words)
        max_window = min(4, len(user_words))
        for window_size in range(2, max_window + 1):
            for i in range(len(user_words) - window_size + 1):
                chunk_list = user_words[i:i + window_size]
                
                # Join with spaces to preserve G2P dictionary inference
                chunk_str = " ".join(chunk_list)
                chunk_ph = _get_phonemes(chunk_str)
                
                # Condition C: Hidden sound bridging two words (e.g., S1 "ice" in "my son" -> M AY S AH N)
                if _check_phoneme_sub_array(chunk_ph, s1_ph):
                    return "master_signifier", s1_lower, s1_lower, {"chunk": chunk_str, "s1": s1_lower}
                    
                # Condition D: Phrase-level homophones
                if _is_acoustic_homophone(chunk_ph, s1_ph):
                    return "master_signifier", chunk_str, chunk_str, None

    return None, None, None, None

# ---------------------------

INTERJECTION_CHOICES = ["Ah?", "Oh?", "Elaborate?", "Say more?", "Mhmm?"]

response_matrix: Dict[str, Dict[int, Optional[str]]] = {
    "contradiction": {1: "Hmm.",  2: "...", 3: "<oracle>", 4: "...", 5: "<gpt_metonymy>"},
    "repression": {1: "Ah?", 2: "...", 3: "<gpt_metonymy>", 4: "<oracle>"},
    "negation": {1: None, 2: "<oracle>", 3: "<gpt_metonymy>", 4: None}, 
    "jouissance": {1: "Oh?", 2: "<gpt_real_question>", 3: "<oracle>", 4: "<gpt_metonymy>", 5: "<gpt_metonymy>"},
    "rationalization": {1: "And yet...", 2: "...", 3: "<oracle>"},
    "morality_logic_defense": {1: "...", 2: "And yet...", 3: "<oracle>"},
    "circular_logic": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "master_signifier": {1: "<S1_echo>", 3: "<S1_triple_echo>", 4: "<oracle>"},
    "metaphor": {1: "<gpt_literalization>", 2: "<gpt_metonymy>", 3: "<random_interjection>", 4: "<oracle>"},
    "metonymy": {1: "<gpt_metonymy>", 2: "<gpt_metonymy>", 3: "<oracle>", 5: "<gpt_metonymy>"}, 
    "ambiguity": {1: "<gpt_ambiguity>", 2: "<gpt_ambiguity>", 3: "<oracle>"}, 
    "fetishistic_phrase": {1: "...", 2: "<oracle>"},
    "identification_other_desire": {1: "<gpt_identification>", 2: "<random_interjection>", 3: "<oracle>", 5: "<gpt_metonymy>"},
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

def _extract_session_context(tracker: Tracker) -> Tuple[int, str, Optional[str]]:
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
    if not PRIOR_HISTORY_ENABLED:
        return before + after
    
    static_len = len(before) + len(after)
    if total_limit <= 0:
        return before + after
    remaining = total_limit - static_len
    if remaining <= 0:
        return before + after
    truncated = _truncate_text_keep_end(prior_history or "", remaining)
    return before + truncated + after

MECHANISM_DEFINITIONS = (
    "- identification_other_desire: Identifying with a signifier that comes from the Symbolic Other.\n"
    "- transference_love: Seeking validation by saying what they believe you (the analyst) wants to hear.\n"
    "- metonymy: Speech that slides according to the laws of metonymy (syntagmatic speech). It is manifest in free association and default speech in therapy, which slides along an associative chain.\n"
    "- repression: A signifier is barred from awareness but returns through symptoms, blanks, omissions or repetitions and other such phenomena.\n"
    "- metaphor: One traumatic, or overly charged signifier has been substituted for another, producing a metaphor. The manifest signifier hides the emotional weight of a repressed trauma, desire, or fundamental fantasy associated with the substituted signifier.\n"
    "- negation: Negating a signifier or double negation within a phrase. It could be caused by denial, or an unconscious ambivalence, contradiction or overwhleming jouissance. Saying “not X” both affirms the possible existence of X and keeps it at a manageable distance. Look for grammatical indicators of negation.\n"
    "- rationalization: Plausible, logical explanation for thoughts or actions that actually stem from and conceal an unconscious desire.\n"
    "- morality_logic_defense: Defending against desire through idealized correctness and morality.\n"
    "- circular_logic: Reasoning loops back on itself.\n"
    "- contradiction: A later statement cancels or undoes an earlier one without the speaker acknowledging or realizing the conflict.\n"
    "- jouissance: The paradox where the subject derives satisfaction from a symptom that is consciously painful or unpleasant. Do not look for 'happiness' but for the Drive (the loop). At least three of the following criteria must be met for you to select this mechanism: (1) Repetition ('I keep doing it', 'again and again'); (2) Paradox ('I hate it but I can't stop', 'awful but I need it'); (3) Excess ('overwhelming', 'too much', 'unbearable'); (4) The Body (physical symptoms, vomiting, shaking) alongside painful, excessive emotion; (5) Fixation on a partial object.\n"
    "- ambiguity: Indeterminate referents.\n"
    "- fetishistic_phrase: Clichés that halt the exploration of desire(s). Common phrases that are impersonal and formulaic.\n"
    "- confession_empathy: Seeking rescue or closeness.\n"
    "- dream_report: The user recounts a dream, nightmare, or a fragment of a dream.\n"
    "- frame_protection: Demands for the session to end.\n"
    "- stasis: The user is stuck, shows resistence, or responds with silence (...) or gibberish.\n"
    "- unfinished_thought: The user expresses an idea that is incomplete, trailing off, or self-interrupted.\n"
    "- transference_lure: The user focuses on you (the analyst), makes demands of you, projects feelings onto you, or attempts to draw you into an Imaginary interpersonal dynamic.\n"
    "- parapraxis: A slip of the tongue, a spoonerism, a misreading, an utterance that the user 'did not mean to say' or any such error that reveals an unconscious signifier. Include instances where a signifier feels 'out of place' or originates from a distant embedding space.\n"
)

def build_combined_analysis_prompt(
    new_input: str,
    raw_history: str,
    prior_history_str: str,
    allowed_mechanisms: List[str],
) -> str:
    cuts = CUT_TRIGGERS
    template = (
        "You are Bruce Fink. Perform TWO analyses on the NEW user input in a single pass.\n\n"
        "# TASK 1: Discourse Mechanism Detection\n"
        "Identify the most important Lacanian discursive phenomena in the NEW input that Bruce Fink would intervene on.\n"
        f"{MECHANISM_DEFINITIONS}\n"
        "Raw recent dialogue:\n"
        f"{raw_history}\n\n"
        "Prior (Long Term) history:\n"
        f"{PRIOR_HISTORY_TOKEN}\n\n"
        "NEW user input:\n"
        f'"{new_input}"\n\n'
        "# TASK 2: Scansion (Cut) Detection\n"
        "Scan for a session-ending cut trigger in the NEW user input.\n\n"
        "STEPS:\n"
        "- Read chronologically the prior history, raw recent dialogue and finally the NEW input.\n"
        "- Identify if the NEW input exhibits the following phenomena:\n"
        "- major_shift_retroactive: A sudden rupture in the Automaton. The user has been speaking in a well established Imaginary 'script' with predictable signifiers (ego-level discourse), and suddenly a signifier intrudes that is probabilistically very unlikely and therefore ruptures the ego-level discourse. The intruder is identified_s1.\n"
        "RULE: You CANNOT cut if it is even slightly unclear what the ego-level script is (use the current input, the prior and session history to determine).\n\n"
        "Respond in strict JSON:\n"
        "{\n"
        '  "mechanism": "one of the allowed" or null,\n'
        '  "mechanism_phrase": "exact substring from NEW input" or null,\n'
        '  "cut_trigger": "label" or null,\n'
        '  "identified_s1": "signifier" or null\n'
        "}\n\n"
        f"Allowed mechanisms: {allowed_mechanisms}\n"
        f"Allowed cut_trigger values: null or one of {cuts}.\n"
        "Only output exactly one valid JSON object."
    )
    return _apply_prior_history_limit(template, prior_history_str, TOTAL_PROMPT_CHAR_LIMIT)

def build_cut_construction_prompt(
    new_input: str,
    identified_s1: str,
) -> str:
    return f"""
You are Bruce Fink. We have identified a rupture on the signifier: "{identified_s1}".

# TASK STEPS
1. LOCATE S1: Find the exact, most surprising appearance of "{identified_s1}" within the <new_input>.
2. TRUNCATE AFTER: Delete all text following this specific S1.
3. CONTIGUOUS EXTRACTION (MAX 4 WORDS): Trace backwards from the S1. Keep only the EXACT 2 to 4 words immediately preceding the S1 in the original text. 
4. FORMATTING: Embolden the S1. Append a period if necessary.

# EXAMPLES
- Example 1 (Short - Stripping the feeling):
  Input: "There is this guy I am looking at and I feel like a yellow dog after he leaves, I just wanted to cry."
  identified_s1: "yellow dog"
  cut_phrase: "A **yellow dog**."

- Example 2 (Long - Stripping the framing):
  Input: "The custard thing to do is what is right and I finally realized after hours of looking that the engine is completely flooded."
  identified_s1: "custard"
  cut_phrase: "The **custard**."

- Example 3 (Noun Phrase Isolation):
  Input: "I just always get so overwhelmed because I feel like a broken machine."
  identified_s1: "machine"
  cut_phrase: "A broken **machine**."

# NEW user input:
"{new_input}"

# Response Format
Respond ONLY in this strict JSON format:
{{
    "cut_phrase": "The contiguous rewritten substring only"
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
        
        start_time = time.time()
        user_id = tracker.sender_id
        
        return [
            SessionStarted(),
            SlotSet("session_thematic_count", 0),
            SlotSet("last_mechanism", None),
            SlotSet("mechanism_counts", None),
            SlotSet("cut_trigger", None),
            SlotSet("dream_fantasy_asked", False),
            SlotSet("session_start_time", start_time),
            ReminderScheduled(
                intent_name="EXTERNAL_session_timeout",
                trigger_date_time=datetime.datetime.now() + datetime.timedelta(seconds=1800),
                name="session_30_min_timeout",
                kill_on_user_message=False,
            ),
            ActionExecuted("action_listen"),
        ]
    
class ActionExecuteTimeout(Action):
    def name(self) -> Text:
        return "action_execute_timeout"

    async def run(self, dispatcher, tracker, domain):
        user_id = tracker.sender_id
        session_start_time = tracker.get_slot("session_start_time")

        is_locked = await asyncio.to_thread(is_user_locked_out, DB_PATH, user_id)
        if is_locked:
            return []

        session_end_message = _format_two_paragraph_message(
            "Our session time is up, to push further would encourage over-analysis.",
            "I will talk with you again after 48 hours."
        )
        
        dispatcher.utter_message(text=session_end_message)
        
        if session_start_time:
            await asyncio.to_thread(set_user_lockout, DB_PATH, user_id, 48, session_start_time)
            
        return []

class ActionAnalyzeMessage(Action):
    def __init__(self):
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_lockouts (
                    user_id TEXT PRIMARY KEY,
                    lockout_until REAL
                )
                """
            )
            conn.commit()

        # Dictionary to hold independent tasks for multiple users
        self.idle_tasks = {}
        
        # Start the global infrastructure kill-switch if it hasn't been started yet
        if not hasattr(self.__class__, '_monitor_thread_started'):
            self.__class__._monitor_thread_started = True
            threading.Thread(target=action_server_idle_monitor, daemon=True).start()

    def name(self) -> Text:
        return "action_analyze_message"

    def _ensure_s1_timer(self, user_id: str, session_start_time: float):
        """Ensures a 35-minute extraction timer is running for the user. Prevents duplicates."""
        
        # Initialize a ledger for completed extractions if it doesn't exist yet
        if not hasattr(self, 'completed_extractions'):
            self.completed_extractions = set()

        # Create a unique ID for this specific session
        session_id = f"{user_id}_{session_start_time}"

        # If we already completed this session's extraction, do nothing
        if session_id in self.completed_extractions:
            return

        # If the timer is currently sleeping/running for this user, leave it alone
        if user_id in self.idle_tasks and not self.idle_tasks[user_id].done():
            return

        # Calculate time remaining out of the 35-minute (2100 seconds) window
        elapsed = time.time() - session_start_time
        remaining = 2100.0 - elapsed

        # If 35 minutes have already passed, log it as completed and execute immediately
        if remaining <= 0:
            self.completed_extractions.add(session_id)
            asyncio.create_task(self._extract_and_replace_s1s(user_id))
            return
            
        async def _countdown():
            try:
                await asyncio.sleep(remaining)
                # Mark as completed right before executing to prevent race conditions
                self.completed_extractions.add(session_id)
                await self._extract_and_replace_s1s(user_id)
                
                # Clean up the pending task dictionary after execution
                if user_id in self.idle_tasks:
                    del self.idle_tasks[user_id]
            except asyncio.CancelledError:
                pass
                
        # Assign the un-interruptible countdown to the user
        self.idle_tasks[user_id] = asyncio.create_task(_countdown())

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
                "You are Bruce Fink. Your task is to review the patient's long-term history and identify their 'Master Signifiers' (S1s).\n"
                "An S1 is a foundational signifier that dictates the subject's discourse and stabilizes their identity. It repeats often across history, functioning as an authoritative, absolute reference point. It operates as a 'just because' signifier—an unquestionable baseline that halts the chain of free association and covers over the limits of meaning.\n"
                "Look for words or phrases that carry disproportionate affective or logical weight—the points where the subject's ability to explain themselves runs dry. The S1 anchors and organizes the rest of the patient's narrative network (their S2).\n"
                "Pay strict attention to the materiality of the signifier: S1s bypass conscious meaning and frequently manifest in disguised forms, persisting through homophones, slips of the tongue, and words embedded phonetically within other words.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. STRICT CAP: Extract a MAXIMUM of 3 of the most significant S1s. Do not exceed this number.\n"
                "2. EXCLUSIVITY: Every signifier in your final list must be materially distinct from the others.\n"
                "3. ISOLATE PHONETIC KERNEL: Finally, you must isolate the most basic, meaningful phonetic kernel of the signifier. If a compound word, conjugated verb, or larger signifier contains a structurally vital, contiguous sound block, you must reduce the extraction to that core phonetic kernel.\n\n"
                "Output ONLY valid JSON in the following strict format:\n"
                "{\n"
                "  \"new_s1s\": [\n"
                "    {\"signifier\": \"extracted signifier\", \"phrase\": \"the contextual phrase it appeared in\"}\n"
                "  ]\n"
                "}"
            )
            
            user_msg_template = (
                f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\n"
                "Identify the top 3 most important Master Signifiers from the Prior History."
            )
            
            user_msg_limit = TOTAL_PROMPT_CHAR_LIMIT - len(system_msg)
            user_msg = _apply_prior_history_limit(user_msg_template, prior_history_text, user_msg_limit)
            
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = resp.choices[0].message.content.strip()
            data = _extract_json(content)
            new_s1s = data.get("new_s1s")

            if isinstance(new_s1s, list):
                logger.info(f"Extraction complete. Found {len(new_s1s)} master signifiers.")
                await asyncio.to_thread(clear_user_master_signifiers_before, DB_PATH, user_id, session_start_time)
                await asyncio.to_thread(_insert_master_signifiers, DB_PATH, user_id, new_s1s[:5])
        except Exception:
            pass

    def _update_mechanism_counts(self, tracker: Tracker, mechanism: str, detected_s1: Optional[str] = None) -> Tuple[Dict[str, int], int]:
        counts = tracker.get_slot("mechanism_counts") or {}
        last = tracker.get_slot("last_mechanism")
        
        if mechanism != last:
            counts = {}
            
        if mechanism == "master_signifier":
            last_s1 = counts.get("__last_s1__")
            current_s1 = detected_s1.strip().lower() if detected_s1 else ""
            
            if last_s1 is not None and current_s1 != last_s1:
                counts[mechanism] = 0
                
            counts["__last_s1__"] = current_s1

        cnt = int(counts.get(mechanism, 0)) + 1
        counts[mechanism] = cnt
        return counts, cnt

    async def _call_combined_analysis_api(self, prompt: str) -> Dict[str, Any]:
        if not async_client:
            return {}
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()
            return _extract_json(content)
        except Exception:
            return {}

    async def _cut_construction_async(self, prompt2b: str) -> Optional[str]:
        if not async_client:
            return None
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt2b}],
                temperature=0.1,
                max_tokens=300,
            )
            content = resp.choices[0].message.content.strip()
            data2b = _extract_json(content)
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
        
        # Update the global activity tracker to keep the Action Server alive
        global GLOBAL_LAST_ACTIVITY
        GLOBAL_LAST_ACTIVITY = time.time()
        
        is_locked = await asyncio.to_thread(is_user_locked_out, DB_PATH, user_id)
        if is_locked:
            return []

        session_start_time = tracker.get_slot("session_start_time")
        if session_start_time is not None:
            current_time = time.time()
            session_duration = current_time - session_start_time
            if session_duration >= 1800:
                session_end_message = _format_two_paragraph_message(
                    "Our session time is up, to push further would encourage over-analysis.",
                    "I will talk with you again after 48 hours.",
                )
                dispatcher.utter_message(text=session_end_message)
                
                await asyncio.to_thread(set_user_lockout, DB_PATH, user_id, 48, session_start_time)
                
                return []

        user_input = (tracker.latest_message.get("text", "") if tracker.latest_message else "").strip()

        # Guarantee we have a baseline time for the calculation
        start_time_float = float(session_start_time) if session_start_time is not None else time.time()
        
        # Initiate the 35-minute timer (it will ignore this call if already running)
        self._ensure_s1_timer(user_id, start_time_float)
        
        user_turn_count, raw_history_text, last_bot_text = _extract_session_context(tracker)

        if user_input:
            async def _safe_insert():
                try:
                    await asyncio.to_thread(insert_user_message, DB_PATH, user_id, user_input)
                except Exception:
                    pass
            asyncio.create_task(_safe_insert())
        
        if user_turn_count <= 3:
            initial_response = await self._generate_initial_therapeutic_response(user_input, raw_history_text)
            
            if initial_response == last_bot_text and initial_response != "...":
                initial_response = "..."
                
            dispatcher.utter_message(text=initial_response)
            return []
            
        prior_history_messages, master_signifiers = await asyncio.gather(
            asyncio.to_thread(get_prior_history_messages, DB_PATH, user_id, PRIOR_HISTORY_PROMPT_LIMIT),
            asyncio.to_thread(get_master_signifier_history, DB_PATH, user_id, 100)
        )
        
        prior_history = "\n".join(f"- {msg}" for msg in prior_history_messages)
        
        low = user_input.lower()
        if low == "/stop" or "i want to stop" in low:
            dispatcher.utter_message(text="...")
            return []
                        
        # ==== PASS 1: Combined mechanism + scansion detection (single LLM call) ===#
        allowed = list(ALLOWED_MECHANISMS)
        if "master_signifier" in allowed:
            allowed.remove("master_signifier")
            
        combined_prompt = build_combined_analysis_prompt(
            user_input, raw_history_text, prior_history, allowed
        )
        data = await self._call_combined_analysis_api(combined_prompt)

        mech: Optional[str] = data.get("mechanism")
        mech_phrase: Optional[str] = data.get("mechanism_phrase")

        user_input_lower = user_input.lower()
        user_words = user_input_lower.split()

        master_signifiers_sorted = sorted(master_signifiers, key=len, reverse=True)
        
        mech_res, detected_s1, mech_phrase_res, shifted_boundary_data = await asyncio.to_thread(
            _scan_for_master_signifier,
            user_input_lower,
            user_words,
            master_signifiers_sorted
        )

        if mech_res:
            mech = mech_res
            mech_phrase = mech_phrase_res
        else:
            detected_s1 = None
            shifted_boundary_data = None

        # ==== PASS 2: Potential Cut Trigger Identification ===#
        potential_trigger: Optional[str] = data.get("cut_trigger")
        identified_s1: Optional[str] = data.get("identified_s1")
        potential_trigger_phrase: Optional[str] = None
        
        verified_trigger: Optional[str] = None
        if potential_trigger and user_turn_count >= 7:
            verified_trigger = potential_trigger
                
        if verified_trigger and identified_s1:
            prompt2b = build_cut_construction_prompt(user_input, identified_s1)
            potential_trigger_phrase = await self._cut_construction_async(prompt2b)
            
        final_trigger_phrase = potential_trigger_phrase if verified_trigger else None
        
        if isinstance(final_trigger_phrase, str):
            final_trigger_phrase = final_trigger_phrase.strip('"\'')

        if verified_trigger:
            base_say = final_trigger_phrase if final_trigger_phrase else ""
            if base_say:
                say = _format_two_paragraph_message(
                    base_say,
                    "We are ending there. I will talk with you again after 48 hours.",
                )
            else:
                say = _format_two_paragraph_message(
                    "We are ending there.",
                    "I will talk with you again after 48 hours.",
                )
            
            dispatcher.utter_message(text=say)
            
            session_start_time = tracker.get_slot("session_start_time")
            await asyncio.to_thread(set_user_lockout, DB_PATH, user_id, 48, session_start_time)
            
            return [
                SlotSet("cut_trigger", verified_trigger),
                SlotSet("last_mechanism", None),
                SlotSet("mechanism_counts", None),
            ]
            
        if mech in ALLOWED_MECHANISMS:
            counts, cnt = self._update_mechanism_counts(tracker, mech, detected_s1)
            dream_asked = tracker.get_slot("dream_fantasy_asked") or False
            
            if mech == "master_signifier" and shifted_boundary_data:
                text = await self._gpt_shifted_boundary_intervention(
                    shifted_boundary_data["chunk"], 
                    shifted_boundary_data["s1"]
                )
                newly_asked = False
            else:
                text, newly_asked = await self.handle_mechanism(
                    mechanism=mech, 
                    count=cnt, 
                    user_input=user_input, 
                    phrase=mech_phrase, 
                    raw_history=raw_history_text, 
                    prior_history=prior_history, 
                    detected_s1=detected_s1,
                    dream_fantasy_asked=dream_asked
                )
            
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

    async def _generate_initial_therapeutic_response(self, user_input: str, raw_history: str) -> str:
        prompt_template = (
            "You are an insightful and empathetic Rogerian therapist. Your goal for these initial messages at the beginning of a session is to build rapport by offering interpretations and mirroring that feel deeply personal to the user, even though they are based on universal psychological principles. This is a technique to make the user feel seen and understood, encouraging them to open up.\n\n"
            "GUIDELINES:\n"
            "1.  **Employ 'Barnum Statements':** Craft statements that are general enough to apply to most people but sound like specific, personal insights.\n"
            "2.  **Validate and Reframe with Warmth:** Acknowledge the user's feelings and warmly reframe their situation or ask a clarifying question. Be neutral but warm.\n"
            "3.  **Immediacy:** Respond to the emotional content within the user's current message, in other words, focus on the immediate emotional context.\n"
            "4.  **Maintain a Professional, Warm Tone:** The tone should be that of a skilled Rogerian therapist—calm, empathetic and reflective.\n"
            "5.  **Keep Evolving to Sound Human:** Use the session history to see what you said before, how you said it, and how the user reacted. Then respond with the SESSION history in mind to deepen the therapeutic connection. Ensure you start each message differently so that you speak naturally.\n"
            "6.  **Structural Constraint:** Formulate exactly ONE short, complete sentence (maximum 30 words). It must end with a period or question mark.\n\n"
            "SESSION HISTORY:\n"
            f"{raw_history}\n\n"
            "USER'S LATEST MESSAGE:\n"
            f'"{user_input}"\n\n'
            "Craft your response based on the entire conversation."
        )
        prompt = prompt_template
        
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
            
            raw_content = resp.choices[0].message.content
            response_text = raw_content.strip().replace('\n', ' ')
            
            if not re.search(r'[.!?]$', response_text):
                match = re.search(r'^(.*?[.!?])', response_text)
                if match:
                    response_text = match.group(1).strip()
                else:
                    response_text += "..."
                    
            return response_text
        except Exception:
            pass
        
    async def handle_mechanism(
        self,
        mechanism: str,
        count: int,
        user_input: str,
        phrase: Optional[str],
        raw_history: str,
        prior_history: str, 
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
            return await self._gpt_denial_intervention(user_input, raw_history, prior_history), False
        if intervention == "<gpt_metonymy>":
            return await self._gpt_metonymy_intervention(user_input, raw_history), False
        if intervention == "<gpt_literalization>":
            return await self._gpt_literalization_intervention(user_input, phrase, raw_history, prior_history), False
        if intervention == "<gpt_identification>":
            return await self._gpt_identification_intervention(user_input), False
        if intervention == "<gpt_ambiguity>":
            return await self._gpt_ambiguity_intervention(user_input), False
        
        if intervention == "<S1_triple_echo>":
            extracted_word = await self._gpt_quilting_point_echo(detected_s1)
            if extracted_word and extracted_word != "...":
                clean_word = extracted_word.rstrip(".").strip()
                if clean_word:
                    return f"{clean_word.capitalize()}, {clean_word.lower()}, {clean_word.lower()}.", False
            return "...", False
            
        if mechanism == "master_signifier":
            if count == 1 or count == 5:
                return await self._gpt_quilting_point_echo(detected_s1), False
                
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
            "You are Bruce Fink. The user is exhibiting METONYMY: an endless sliding of desire from signifier to signifier.\n"
            "Your task is to produce a 'Point de Capiton' (Quilting Point). You must STOP the slide by isolating a signifier that best matches the criteria below.\n\n"
            "Criteria for selection:\n"
            "1. PRETERITION: A signifier that the user seems to skip over, gloss over, or treat as insignificant, but that appears repeatedly across the session history.\n"
            "2. RUPTURE: A signifier involved in a parapraxis (slip of the tongue), a sudden grammatical failure, a mixed metaphor, or a jarring non-sequitur.\n"
            "3. INSISTENCE: A signifier that the user repeats unnecessarily throughout the current input or in this session.\n"
            "4. POLYSEMY: A signifier harboring high ambiguity, or multiple literal/figurative definitions when echoed back in isolation.\n\n"
            "NOTES:\n"
            "Bias towards selecting a signifier from the user's current input.\n"
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
            "You are Bruce Fink. The user has utilized a METAPHOR. "
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
                temperature=0.4, 
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
            "You are Bruce Fink. The user is identifying with an external desire or signifier that comes from the 'Other' (e.g., 'They want me to be a doctor...', 'Society says I am a bad girl', 'My father thinks I am his favourite...', 'You think I am a sterile person.', 'My mom's 'care' always...').\n"
            "Task: Identify exactly WHO or WHAT the user is identifying with (The Agency/The Other/You, the analytic bot) and the signifier they have adopted as part of their identity.\n"
            "Output format: Return ONLY the signifier of the Agency/Other's desire followed by a question mark.\n"
            "Examples:\n"
            "- User: 'My father wants me to be a doctor.' -> Output: 'Doctor?'\n"
            "- User: 'Everyone thinks I am crazy.' -> Output: 'Crazy?'\n"
            "- User: 'I need to be productive for the economy.' -> Output: 'Productive?'\n"
            "- User: 'You don't love me.' -> Output: 'Don't love you?'\n"
            "- User: 'My mom's care is suffocating.' -> Output: 'Suffocating?'\n"
            "- User: 'Society's expectations are overwhelming.' -> Output: 'Overwhelming?'\n"
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
            return "..."
        system_msg = (
            "You are Bruce Fink. The user is using AMBIGUOUS language (vague referents, or confusion).\n"
            "Task: Isolate the specific word or short phrase that holds the ambiguity.\n"
            "Output format: Return ONLY that word/phrase followed by a question mark, or silence if it is more appropriate (...).\n"    
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
            return "..."

    async def _gpt_denial_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "..."
        system_msg = (
            "You are Bruce Fink. Produce ONE Bruce Fink-style intervention in response to NEGATION. "
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
            "You are Bruce Fink. Your operation is purely structural: you fracture the user's conscious syntax to expose an unconscious truth. "
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
            "- Tone: Trippy, jarring, and disruptive.\n"
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
                temperature=0.4, 
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
        detected_s1: Optional[str],
    ) -> str:
        if not detected_s1:
            return "..."

        signifier = detected_s1.strip().strip('"\'')
        if not signifier:
            return "..."

        # --- NEW REGEX SCRUBBING STEP ---
        # This removes any trailing commas, question marks, or symbols from the end of the word
        signifier = re.sub(r'[^\w\s]+$', '', signifier).strip()
        # --------------------------------

        parts = signifier.split()
        if parts:
            signifier = " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

        # Directly append the full stop instead of passing it to _ensure_trailing_punct
        return signifier + "."
        
    async def _gpt_real_question_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "What do you associate with this fantasy?"
            
        system_msg = (
            "You are Bruce Fink. The user is exhibiting Jouissance in a symptom, repetition compulsion, or satisfaction found within a disturbing fantasy.\n"
            "Your task is to formulate a question that atempts to redirect the user towards the Symbolic, relational message of their bodily symptom/charged fantasy.\n\n"
            "Rules:\n"
            "1. Form: Must be open-ended and use THEIR EXACT SIGNIFIERS.\n"
            "2. Kinds of questions:\n" 
            "   - To engage the symbolic, a question might strip the signifier of its anatomical or literal context to highlight a double meaning.\n"
            "   - Or it might enquire about the function of the symptom, the user's relationship to it, or what they hope the Other will say or do in response.\n"
            "   - Another possibility is to ask what/who is the hidden agent of the verb.\n"
            "3. Sound natural: Scan the recent history to ensure your phrasing sounds natural and you are not repeating a previous question or a questioning structure.\n"
            "4. Ensure you maintain a neutral, non-judgmental tone. Avoid any interpretation or moralizing.\n"
            "5. Maximum 15 words.\n"
            "6. Strictly refuse empathy, medicalization, or validation. Do not say 'I understand,' acknowledge their pain, or offer support. Your question must function as an enigmatic cut. Output ONLY the question itself, with no introductory or explanatory text.\n"
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
                max_tokens=100,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip().strip("\"'")
        except Exception as e:
            return "What do you associate with this fantasy?"

    async def _gpt_dream_intervention(self, text: str, raw_history: str, prior_history: str) -> str:
        if not async_client:
            return "What comes to mind when you say that?"
            
        system_msg = (
            "You are Bruce Fink. The user is describing a dream or referencing one they talked about previously. "
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
            "You are Bruce Fink. The user is demanding advice, empathy, or knowledge (e.g., asking 'What should I do?'). "
            "You must question your position as the 'Subject Supposed to Know' and frustrate the user's demand for knowledge. "
            "Rules:\n"
            "1. Deflect the demand for knowledge by asking a question that uses almost the same words they used.\n"
            "2. Produce exactly ONE short, natural question asking why they want you to tell them 'the answer' or what to do.\n"
            "3. Maximum 12 words.\n"
            "Examples: 'What would my answer be?', 'You are hoping I will say?', 'And I would tell you...?' "
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
                max_tokens=30,
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
            "You are Bruce Fink. You must employ the smallest step possible to help the analysand "
            "articulate an experience that they are reluctant to share.\n\n"
            "Isolate the single most unfinished thought from the user's text.\n"
            "Echo the last part of that thought followed by a question mark.\n\n"
            "Example: If the user says 'I have been experiencing a lot of... nevermind it's not worth talking about', output 'A lot of...?'\n"
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
            "You are Bruce Fink. The user has produced a parapraxis (a slip of the tongue, typo, or unintended signifier).\n"
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
        
    async def _gpt_shifted_boundary_intervention(self, chunk: str, s1: str) -> str:
        if not async_client:
            return f"{s1.capitalize()}?"
            
        # PROMPT UPDATED: Only appended new words to the first sentence and the examples list.
        system_msg = (
            "You are Bruce Fink. The user unconsciously produced a Master Signifier (S1) hidden across a word boundary or within a single larger word.\n"
            "Your task is to formulate a 2 to 3 word, polyvalent, 'trippy', oracular echo that highlights this phonetic amalgamation.\n\n"
            "Rules:\n"
            "1. Exclusively use the provided host words and the hidden S1 to construct the intervention.\n"
            "2. Maximum 3 words.\n"
            "3. Capitalize the first word and end wih a period.\n\n"
        )
        
        user_msg = f"Host words: \"{chunk}\"\nHidden S1: \"{s1}\"\nProduce the intervention:"

        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
                max_tokens=15, 
            )
            
            line = _strip_response(resp.choices[0].message.content)
            return line if line else f"{s1.capitalize()}?"
        except Exception:
            return f"{s1.capitalize()}?"