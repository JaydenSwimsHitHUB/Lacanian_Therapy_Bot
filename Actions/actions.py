import json
import os
import re
import random
import sqlite3
import logging
import asyncio
import sys
import threading  # <--- NEW: For the synchronous timer
from typing import Any, Text, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType

# Quiet noisy loggers (optional)
for _name in ["rasa", "rasa.core", "rasa_sdk", "urllib3"]:
    logging.getLogger(_name).setLevel(logging.WARNING)

# --- SETUP CLIENT FOR DEEPSEEK R1 / COMPATIBLE PROVIDER ---
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("API_BASE_URL", "https://api.deepinfra.com/v1/openai") 

if not api_key:
    logging.warning("OPENAI_API_KEY environment variable not set")

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

# Database functions
def insert_user_message(conn: sqlite3.Connection, user_id: str, message: str) -> None:
    conn.execute(
        "INSERT INTO user_messages (user_id, message) VALUES (?, ?)",
        (user_id, message),
    )
    conn.commit()

# --- UTILITY: ROBUST JSON EXTRACTION ---
def _extract_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    text = content
    text = re.sub(r"```(?:json)?\n|```", "", text)
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
    "contradiction": {1: "Hmm.",  2: "<oracle>"},
    "repression": {1: "...", 2: "Ah?", 3: "<oracle>"},
    "denial": {1: None, 2: "<oracle>"}, 
    "jouissance": {1: "<oracle>", 2: "Oh?", 4: "<oracle>"},
    "rationalization": {1: "And yet...", 2: "...", 3: "<oracle>"},
    "morality_logic_defense": {1: "...", 2: "And yet...", 3: "<oracle>"},
    "circular_logic": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "master_signifier": {1: "<S1_echo>", 2: "<oracle>"},
    "metaphor": {1: "...", 2: "Mhmm?", 3: "<oracle>"},
    "metonymy": {1: "<gpt_metonymy>", 2: "<gpt_metonymy>", 3: "<oracle>", 4: "<gpt_metonymy>"},
    "ambiguity": {1: "<gpt_ambiguity>", 2: "<oracle>"}, 
    "fetishistic_phrase": {1: "...", 2: "<oracle>"},
    "identification_other_desire": {1: "<gpt_identification>", 2: "<oracle>"}, 
    "demand_for_knowledge": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "confession_empathy": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "frame_protection": {1: "...", 2: "Hmm.", 3: "<oracle>"},
}

ALLOWED_MECHANISMS = set(response_matrix.keys())

CUT_TRIGGERS = [
    "marked_signifier_collapse",
    "major_shift_retroactive",
    "point_of_maximal_ambiguity",
    "return_of_repressed_master_signifier",
    "omission_missing_signifier",
]


# ---------- Utility for prompts ----------

def _get_session_history_text(tracker: Tracker) -> str:
    session_events: List[Dict[Text, Any]] = []
    for event in reversed(tracker.events):
        if event.get("event") == "session_started":
            break
        session_events.append(event)
    session_events.reverse()

    latest_text = (tracker.latest_message.get("text") or "").strip()
    
    if latest_text and session_events:
        for i in range(len(session_events) - 1, -1, -1):
            if session_events[i].get("event") == "user":
                if session_events[i].get("text", "").strip() == latest_text:
                    session_events.pop(i)
                break

    lines: List[str] = []
    for ev in session_events:
        evt = ev.get("event")
        if evt == "user":
            text = ev.get("text", "")
            if text:
                lines.append(f"User: {text}")
        elif evt == "bot":
            text = ev.get("text", "")
            if text:
                lines.append(f"Bot: {text}")
    
    return "\n".join(lines) if lines else "(no prior conversation in this session)"


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


def _get_session_user_turn_count(tracker: Tracker) -> int:
    count = 0
    for event in reversed(tracker.events): 
        if event.get("event") == "session_started":
            break
        if event.get("event") == "user" and event.get("text"):
            count += 1
    return count


def build_cut_detection_prompt(
    new_input: str,
    tracker: Tracker,
    prior_history_str: str,
    master_signifier_history: str = "",
) -> str:
    raw_history = _get_session_history_text(tracker)
    cuts = CUT_TRIGGERS

    template = f"""
You are a Lacanian analyst waiting to perform SCANSION, an abrupt session end (Cut) at a worthy time to produce Retroactive Meaning (après-coup).

# Scanning Algorithm
1. Chronological Scan: Read <prior_history>, then <master_signifier_history>, then <raw_history>, then <new_input>, from left to right. 
2. Identify cut_trigger: Scan for a cut_trigger in <new_input> based on the definitions of the cut triggers below.
3. Identify S1: First signifier of rupture = identified_s1.

# Cut Triggers: Criteria for the Cut
- major_shift_retroactive: A sudden rupture in the Automaton. The user has been speaking in a well established Imaginary 'script' with predictable signifiers (ego-level discourse), and suddenly a signifier intrudes that is probabilistically very unlikely and therefore ruptures the ego-level discourse. The intruder is identified_s1. 

# Constraints:
- You CANNOT cut if it is even slightly unclear what the ego-level script is. The script must be very well established with mula battery of signifiers that clearly point to a specific ego-level discourse.

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
# CRITICAL RULES
1. IDENTIFY S1: Take the `identified_s1` as S1.
2. CONSTRUCT CUT: In `cut_phrase`, rewrite the user's NEW input *up to and including* that S1.
   Embolden the S1.
   Do NOT include any text after the S1.
   Follow it immediately with a new paragraph and the exact string "We are ending there."
   
# EXAMPLE
   - Example Input: "I felt so yellow I mean blue after he left, I just wanted to cry."
   - identified_s1: "yellow"
   - cut_phrase: "I felt so **yellow**.
                  We are ending there."

# NEW user input:
"{new_input}"

# Response Format
Respond ONLY in this strict JSON format:
{{
    "cut_phrase": "The rewritten input + page break + ending message"
}}
"""

# ---------- Actions ----------
class ActionSessionStartCustom(Action):
    def name(self) -> Text:
        return "action_session_start_custom"

    def run(
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
            ActionExecuted("action_listen"),
        ]


class ActionAnalyzeMessage(Action):
    def __init__(self):
        # --- PERSISTENCE ENABLED ---
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
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
        self.conn.commit()

        # --- IDLE SHUTDOWN TIMER (THREADING BASED) ---
        self.shutdown_timer = None
        self.idle_timeout = 1800.0  # 30 minutes in seconds
        
        # Start the timer immediately on initialization
        self._reset_timer()

    def name(self) -> Text:
        return "action_analyze_message"

    def _shutdown_server(self):
        """The callback function that runs when the timer expires."""
        logging.warning(f"Machine idle for {self.idle_timeout}s. Initiating shutdown via os._exit(0).")
        # Force immediate exit. Fly.io will see the process die.
        os._exit(0)

    def _reset_timer(self):
        """Cancels any existing timer and starts a new one in a separate thread."""
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        
        # Create a new Timer that calls _shutdown_server after idle_timeout
        self.shutdown_timer = threading.Timer(self.idle_timeout, self._shutdown_server)
        self.shutdown_timer.daemon = True # Daemon thread exits if main program exits
        self.shutdown_timer.start()

    def _update_mechanism_counts(self, tracker: Tracker, mechanism: str) -> Tuple[Dict[str, int], int]:
        counts = tracker.get_slot("mechanism_counts") or {}
        last = tracker.get_slot("last_mechanism")
        if mechanism != last:
            counts = {}
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
        except Exception as e:
            logging.error(f"Mechanism API call failed: {e}")
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
        except Exception as e:
            logging.error(f"Scansion API call failed: {e}")
            return {}

    async def _get_responses_parallel_async(self, mech_prompt: str, scansion_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            data1, data2 = await asyncio.gather(
                self._call_mech_api(mech_prompt),
                self._call_scansion_api(scansion_prompt)
            )
            return data1, data2
        except Exception as e:
            logging.error(f"Async parallel calls failed: {e}")
            return {}, {}

    async def _fallback_responses_async(self, mech_prompt: str, scansion_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not async_client:
            return {}, {}
        try:
            data1, data2 = await asyncio.gather(
                async_client.chat.completions.create(
                    model=MODEL_NAME_FAST,
                    messages=[{"role": "user", "content": mech_prompt}],
                    temperature=0.1,
                ),
                async_client.chat.completions.create(
                    model=MODEL_NAME_FAST,
                    messages=[{"role": "user", "content": scansion_prompt}],
                    temperature=0.4,
                    max_tokens=600,
                )
            )
            return _extract_json(data1.choices[0].message.content.strip()), _extract_json(data2.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"Fallback async calls failed: {e}")
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
            return data2b.get("cut_phrase")
        except Exception as e:
            logging.error(f"Cut construction failed: {e}")
            return None

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        
        # --- TIMER RESET ---
        # We received a message, so reset the 30-minute idle timer using threading
        self._reset_timer()

        user_id = tracker.sender_id
        user_input = (tracker.latest_message.get("text", "") if tracker.latest_message else "").strip()

        # Retrieve prior history from the database for this user
        def get_prior_history_messages(conn, user_id):
            cursor = conn.execute(
                "SELECT message FROM user_messages WHERE user_id = ? ORDER BY timestamp ASC, id ASC",
                (user_id,)
            )
            return [row[0] for row in cursor.fetchall()]

        prior_history_messages = get_prior_history_messages(self.conn, user_id)
        prior_history = "\n".join(f"- {msg}" for msg in prior_history_messages)
        
        def get_master_signifier_history(conn, user_id):
            cursor = conn.execute(
                "SELECT signifier FROM master_signifiers WHERE user_id = ? ORDER BY timestamp ASC, id ASC",
                (user_id,)
            )
            return [row[0] for row in cursor.fetchall()]
        master_signifiers = get_master_signifier_history(self.conn, user_id)
        master_signifier_history = "\n".join(f"* {s}" for s in master_signifiers)


        # --- MODIFIED: Add therapist logic for the first 3 turns ---
        user_turn_count = _get_session_user_turn_count(tracker)

        if user_turn_count <= 3:
            initial_response = await self._generate_initial_therapeutic_response(user_input, tracker, prior_history)
            dispatcher.utter_message(text=initial_response)

            if user_input:
                insert_user_message(self.conn, user_id, user_input)
            return []
        
        low = user_input.lower()
        if low == "/stop" or "i want to stop" in low:
            dispatcher.utter_message(text="...")
            return []

        # ==== PASS 1: Local mechanism identification ===#
        raw_history_mech = _get_session_history_text(tracker)
        allowed = list(ALLOWED_MECHANISMS)

        mech_template = (
            "You are a Lacanian analyst. Identify the single most structuring mechanism in the user's NEW input by examining the NEW message and the signifying chain (The current session's history as well as the prior sessions' history).\n\n"
            "Use the NEW message and the signifying chain (message history) to help you decide.\n"
            "=== Structural Mechanisms: Definitions & Diagnostic Cues ===\n"
            "- MASTER SIGNIFIER: A signifier that repeats and supposes to be a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse. It is a 'just because' signifier that stabilizes the subject's discourse and covers over the limits of meaning. It may often arise in metaphorical or disguised forms. Identify a master signifier in the new input by examining the entire history.\n"
            "- IDENTIFICATION WITH OTHER’S DESIRE: Being governed by an imaginary external desire. Taking on another’s desire as one’s own.\n"
            "- METONYMY: The default mode of speech where meaning slides along a syntagmatic chain without anchoring. Select this when NONE of the other mechanisms are clearly present AND the speech exhibits the characteristics of: (1) Topic-hopping: jumping between loosely associated subjects without concluding any ('...and then... and then... and also...'). (2) Listing/Cataloguing: enumerating details, events, or complaints without arriving at a point. (3) Tangential drift: starting on one topic but sliding sideways into adjacent ones via loose associations. (4) Avoidance through narration: telling stories or recounting events as a way of not confronting what is at stake. (5) Surface-level description: staying at the level of facts and events without affect or subjective engagement. This is the 'unmarked' mechanism — ordinary speech that has not yet been punctuated by any rupture.\n"
            "- REPRESSION: A signifier is barred from awareness but returns through symptoms, blanks, omissions, slips, or repetitions and other such phenomena.\n"
            "- METAPHOR: An arrest of metonymy where a signifier is substituted from distant embedding spaces. This rupture creates vector discontinuities and syntactic shifts—such as a noun appearing where an adjective is expected—signaling a formal displacement of meaning. Such anomalies often leverage phonemic echoes to bridge manifest utterances with repressed unconscious material.\n"
            "- DENIAL: A negation of a thought or feeling. Stating an unconscious truth while refusing it consciously. Saying “not X” both utters X and keeps it repressed.\n"
            "- RATIONALIZATION: Plausible, logical explanation for thoughts or actions that actually stem from and conceal an unconscious desire.\n"
            "- MORALITY/LOGIC DEFENSE: Defending against desire through idealized correctness and morality.\n"
            "- CIRCULAR LOGIC: Reasoning loops back on itself.\n"
            "- CONTRADICTION: A later statement cancels or undoes an earlier one without the speaker acknowledging or realizing the conflict.\n"
            "- JOUISSANCE: The paradox where the subject derives satisfaction from a symptom that is consciously painful or unpleasant. Do not look for 'happiness' but for the Drive (the loop). At least three of the following criteria must be met for you to select this mechanism: (1) Repetition ('I keep doing it', 'again and again'); (2) Paradox ('I hate it but I can't stop', 'awful but I need it'); (3) Excess ('overwhelming', 'too much', 'unbearable'); (4) The Body (physical symptoms, vomiting, shaking) alongside painful, excessive emotion; (5) Fixation on a partial object.\n"
            "- AMBIGUITY: Indeterminate referents and unfinished ideas.\n"
            "- FETISHISTIC PHRASES: Clichés that halt the exploration of desire(s). Common phrases that are impersonal and formulaic.\n"
            "- DEMAND FOR KNOWLEDGE: Demand(s) for knowledge and inquiries leveled at you.\n"
            "- CONFESSION/EMPATHIC APPEAL: Seeking rescue or closeness.\n"
            "- FRAME PROTECTION: Demands for the session to end.\n\n"
            "Master Signifier (S1) History (previously identified S1s for this user):\n"
            f"{master_signifier_history}\n\n"
            "Raw recent dialogue:\n"
            f"{raw_history_mech}\n\n"
            "Prior (Long Term) history:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "NEW user input:\n"
            f"\"{user_input}\"\n\n"
            "RULE: If and only if you select \"master_signifier\" as the mechanism, you MUST also return the specific anchoring signifier in the \"master_signifier\" field. Otherwise leave it null.\n\n"
            "Respond in strict JSON:\n"
            "{\n"
            '  "mechanism": "one of the allowed" or null,\n'
            '  "mechanism_phrase": "exact substring from NEW input" or null,\n'
            '  "master_signifier": "the S1 signifier" or null\n'
            "}\n\n"
            f"Allowed: {allowed}\n"
            "No commentary."
        )
        mech_prompt = _apply_prior_history_limit(mech_template, prior_history, TOTAL_PROMPT_CHAR_LIMIT)
        
        # ==== PASS 1 & 2a: PARALLEL CALLS FOR SPEED ===#
        
        # --- PASS 2a: Detection (With Long Term History) ---
        prompt2a = build_cut_detection_prompt(user_input, tracker, prior_history, master_signifier_history)

        data1, data2 = await self._get_responses_parallel_async(mech_prompt, prompt2a)
        
        if not data1 and not data2:
            logging.warning("Both API calls failed, using fallback responses")
            data1, data2 = await self._fallback_responses_async(mech_prompt, prompt2a)
        
        mech: Optional[str] = data1.get("mechanism")
        mech_phrase: Optional[str] = data1.get("mechanism_phrase")
        detected_s1: Optional[str] = data1.get("master_signifier")

        if not detected_s1 and mech == "master_signifier" and mech_phrase:
            detected_s1 = mech_phrase
            
        if detected_s1 and isinstance(detected_s1, str) and detected_s1.strip():
            s1_clean = detected_s1.strip()
            cursor = self.conn.execute(
                "SELECT 1 FROM master_signifiers WHERE user_id = ? AND signifier = ? LIMIT 1",
                (user_id, s1_clean)
            )
            if cursor.fetchone() is None:
                self.conn.execute(
                    "INSERT INTO master_signifiers (user_id, signifier, phrase) VALUES (?, ?, ?)",
                    (user_id, s1_clean, mech_phrase or "")
                )
                self.conn.commit()

        # ==== PASS 2: Potential Cut Trigger Identification ===#
        potential_trigger: Optional[str] = data2.get("cut_trigger")
        identified_s1: Optional[str] = data2.get("identified_s1")
        potential_trigger_phrase: Optional[str] = None

        if potential_trigger and potential_trigger in CUT_TRIGGERS and identified_s1:
            prompt2b = build_cut_construction_prompt(user_input, identified_s1)
            potential_trigger_phrase = await self._cut_construction_async(prompt2b)
        
        data2["cut_phrase"] = potential_trigger_phrase

        # ==== PASS 3: Cut Verification ===#
        verified_trigger: Optional[str] = None
        if potential_trigger and potential_trigger in CUT_TRIGGERS:
            user_turn_count = _get_session_user_turn_count(tracker)
            if user_turn_count < 7:
                verified_trigger = None 
            else:
                verified_trigger = potential_trigger

        final_trigger_phrase = potential_trigger_phrase if verified_trigger else None
        if isinstance(final_trigger_phrase, str):
            final_trigger_phrase = final_trigger_phrase.strip('"\'')

        logging.info("DETECTIONS: mechanism=%s, cut_trigger=%s, verified=%s, s1=%s",
                     mech or "", potential_trigger or "", verified_trigger or "", detected_s1 or "")

        if user_input:
            insert_user_message(self.conn, user_id, user_input)

        if verified_trigger:
            say = final_trigger_phrase if final_trigger_phrase else "We are ending there."
            dispatcher.utter_message(text=say)
            return [
                SlotSet("cut_trigger", verified_trigger),
                SlotSet("last_mechanism", None),
                SlotSet("mechanism_counts", None),
            ]

        if mech in ALLOWED_MECHANISMS:
            counts, cnt = self._update_mechanism_counts(tracker, mech)
            text = await self.handle_mechanism(mech, cnt, user_input, mech_phrase, tracker, prior_history, master_signifier_history)
            dispatcher.utter_message(text=text)
            return [
                SlotSet("last_mechanism", mech),
                SlotSet("mechanism_counts", counts),
                SlotSet("cut_trigger", None),
                SlotSet(
                    "session_thematic_count",
                    (tracker.get_slot("session_thematic_count") or 0) + 1,
                ),
            ]

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

    async def _generate_initial_therapeutic_response(self, user_input: str, tracker: Tracker, prior_history: str) -> str:
        history = _get_session_history_text(tracker)
        prompt_template = (
            "You are an insightful and empathetic therapist. Your goal for these initial sessions is to build rapport and hope by offering interpretations and mirroring that feel deeply personal to the user, even though they are based on universal psychological principles. This is a technique to make the user feel seen and understood, encouraging them to open up.\n\n"
            "GUIDELINES:\n"
            "1.  **Use Universal Themes:** Your interpretations should touch on common human conflicts and desires. Do NOT point out that these desires and conflicts are universally human. Keep it personal. \n"
            "2.  **Employ 'Barnum Statements':** Craft statements that are general enough to apply to most people but sound like specific, personal insights.\n"
            "3.  **Validate and Reframe:** Acknowledge the user's feelings and gently reframe their situation.\n"
            "4.  **Maintain a Professional, Warm Tone:** The tone should be that of a skilled therapist—calm, reassuring, and thoughtful. Avoid clichés and overly sentimental language.\n"
            "5.  **Keep Evolving to Sound Human:** Use the session history to see what you said before, how you said it, and how the user reacted. Then respond with the history in mind to deepen the therapeutic connection. Ensure you do not sound robotic and predictible.\n"
            "6.  **Keep it Brief and Open-Ended:** Respond as briefly as possible. Between 1-2 sentences. The response should invite further reflection from the user without asking a direct question.\n"
            "PRIOR SESSION HISTORY (Long Term):\n" 
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "SESSION HISTORY:\n"
            f"{history}\n\n"
            "USER'S LATEST MESSAGE:\n"
            f'"{user_input}"\n\n'
            "Craft your insightful, universally applicable therapeutic interpretation based on the entire conversation."
        )
        prompt = _apply_prior_history_limit(prompt_template, prior_history, TOTAL_PROMPT_CHAR_LIMIT)
        try:
            response_text = await self._generate_initial_response_async(prompt)
            return response_text
        except Exception as e:
            logging.error(f"Error generating initial therapeutic response: {e}")
            return "That's a very important point. It makes sense to feel that way. Please, continue."
    
    async def _generate_initial_response_async(self, prompt: str) -> str:
        if not async_client:
            return "..."
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            response_text = resp.choices[0].message.content.strip().replace('\n', ' ')
            return response_text
        except Exception as e:
            raise e

    async def handle_mechanism(
        self,
        mechanism: str,
        count: int,
        user_input: str,
        phrase: Optional[str],
        tracker: Tracker,
        prior_history: str, 
        master_signifier_history: str = "",
        ) -> str:
        
        responses = response_matrix.get(mechanism, {})
        intervention = responses.get(count)

        if mechanism == "denial" and (count == 1 or count == 3):
            return await self._gpt_denial_intervention(user_input, tracker, prior_history, master_signifier_history)
        if intervention == "<gpt_metonymy>":
            return await self._gpt_metonymy_intervention(user_input, tracker, prior_history, master_signifier_history)
        if intervention == "<gpt_identification>":
            return await self._gpt_identification_intervention(user_input, tracker)
        if intervention == "<gpt_ambiguity>":
            return await self._gpt_ambiguity_intervention(user_input, tracker)
        if mechanism == "master_signifier" and (count == 1 or count == 3):
            return await self._gpt_quilting_point_echo(user_input, tracker, prior_history)
        if intervention == "<oracle>":
            return await self._generate_oracular_equivoque(user_input, tracker)
        if intervention:
            if count == 2 and mechanism in {"repression", "jouissance", "metaphor"}:
                return random.choice(INTERJECTION_CHOICES)
            return intervention

        return "..."

    async def _gpt_metonymy_intervention(self, user_input: str, tracker: Tracker, prior_history: str, master_signifier_history: str = "") -> str:
        if not async_client:
            return "..."
        history = _get_session_history_text(tracker)
        system_msg = (
            "You are a Lacanian analyst. The user is exhibiting METONYMY: an endless sliding of meaning (displacement, running from topic to topic, inability to conclude).\n"
            "Your task is to produce a 'Point de Capiton' (Quilting Point). You must STOP the sliding by isolating a single signifier or short phrase that anchors the nonsense.\n"
            "Rules:\n"
            "1. Read the history to identify what the user is sliding AWAY from.\n"
            "2. Select one specific word/phrase from the User's text (current or recent) that acts as the anchor.\n"
            "3. Return ONLY that word/phrase with a period to denote finality/anchoring.\n"
            "4. Do NOT explain or ask a question. Be authoritative but cryptic."
        )
        user_msg_template = (
            "Master Signifier (S1) History:\n"
            f"{master_signifier_history}\n\n"
            "Prior History:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "History:\n"
            f"{history}\n\n"
            f"Current Input: \"{user_input}\"\n\n"
            "Output the Quilting Point:"
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
                temperature=0.2,
                max_tokens=15,
            )
            line = resp.choices[0].message.content.strip()
            line = re.sub(r'^[\"\'\“\”]+|[\"\'\“\”]+$', "", line).strip()
            if line:
                line = line[0].upper() + line[1:]
            return line
        except Exception:
            return "..."        
    
    async def _gpt_identification_intervention(self, user_input: str, tracker: Tracker) -> str:
        if not async_client:
            return "Who?"
        system_msg = (
            "You are a Lacanian analyst. The user is identifying with an external desire or the 'Other' (e.g., 'They want me to...', 'Society says...', 'My father thinks...').\n"
            "Task: Identify exactly WHO or WHAT the user is identifying with (The Agency/The Other).\n"
            "Output format: Return ONLY the name of the Agency/Other followed by a question mark.\n"
            "Examples:\n"
            "- User: 'My father wants me to be a doctor.' -> Output: 'Your father?'\n"
            "- User: 'Everyone thinks I am crazy.' -> Output: 'Everyone?'\n"
            "- User: 'I need to be productive for the economy.' -> Output: 'The economy?'"
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
            line = re.sub(r'^[\"\'\“\”]+|[\"\'\“\”]+$', "", line).strip()
            if not line.endswith("?"):
                line += "?"
            return line
        except Exception:
            return "Who?"

    async def _gpt_ambiguity_intervention(self, user_input: str, tracker: Tracker) -> str:
        if not async_client:
            return "?"
        system_msg = (
            "You are a Lacanian analyst. The user is using AMBIGUOUS language (vague referents, or confusion).\n"
            "Task: Isolate the specific word or short phrase that holds the ambiguity.\n"
            "Output format: Return ONLY that word/phrase followed by a question mark.\n"    
            "Example: 'There's this... thing about her.' -> 'Thing?'"
        )
        user_msg = f"User text: \"{user_input}\""
        return await self._gpt_ambiguity_intervention_async(system_msg, user_msg)
    
    async def _gpt_ambiguity_intervention_async(self, system_msg: str, user_msg: str) -> str:
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
            line = re.sub(r'^[\"\'\“\”]+|[\"\'\“\”]+$', "", line).strip()
            if not line.endswith("?"):
                line += "?"
            return line
        except Exception:
            return "?"

    async def _gpt_denial_intervention(self, text: str, tracker: Tracker, prior_history: str, master_signifier_history: str = "") -> str:
        if not async_client:
            return "..."
        history = _get_session_history_text(tracker)
        system_msg = (
            "You are a Lacanian analyst. Produce ONE Bruce Fink-style intervention in response to DENIAL. "
            "Rules: (1) Read the full user text and history and locate the strongest, most meaningful denial phrase in the NEW input (negations that are related to key denied signifiers and are in forms such as don't, can't, won't, "
            "wouldn't, shouldn't, couldn't, never, no, nothing, impossible, ain't, not X). "
            "(2) Echo the user's EXACT wording starting from the negation keyword. Change pronouns (I -> You, My -> Your) to address the user. "
            "(3) Output ONLY ONE short line (1–12 words), no quotes, no explanation. "
            "(4) End with a question mark or ellipsis. "
            "(5) STRICT CONSTRAINT: Start your response with the negation word the user actually used. "
            "   - If user says 'I don't like it', output 'Don't like it?' "
            "   - If user says 'I won't go', output 'Won't go?' "
            "   - If user says 'Why should I lower the drawbridge?', output 'Why should you lower the drawbridge?' (since there is no 'don't/won't' here). "
            "   - DO NOT transform 'Why should I' into 'Don't want to'. Only use 'Don't' if the user said 'Don't'. "
            "(6) If the denial is located EXCLUSIVELY in a single word through a prefix (e.g., 'impossible'), use that word alone with a question mark or ellipsis. "
            "(7) If no clear denial is present, output just '...'."
        )
        user_msg_template = (
            "Master Signifier (S1) History:\n"
            f"{master_signifier_history}\n\n"
            "Prior History:\n"
            f"{PRIOR_HISTORY_TOKEN}\n\n"
            "Session History:\n"
            f"{history}\n\n"
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
            line = resp.choices[0].message.content.strip()
            line = re.sub(r'^[\"\'\“\”]+|[\"\'\“\”]+$', "", line).strip()
            line = line.splitlines()[0].strip()
            if not line:
                return "..."
            if len(line) > 120:
                return (line[:80].rstrip(" .") + "?")
            if not (line.endswith("?") or line.endswith("...")):
                line = line.rstrip(" .") + "?"
            return line
        except Exception:
            return "..."

    @staticmethod
    def _format_citation(text: str) -> str:
        if not text:
            return "..."
        text = text[0].upper() + text[1:]
        if not text.endswith(('.', '!', '?', '…')):
            text += '.'
        return text

    async def _generate_oracular_equivoque(self, text: str, tracker: Tracker) -> str:
        if not async_client:
            return "..."
        prompt = (
            "SYSTEM: You are a Lacanian psychoanalyst performing a 'phonetic decoupage.' "
            "You treat speech as 'lalangue'—a chain of signifiers where the unconscious "
            "speaks through homophonic puns (l'équivoque).\n\n"
            "UTTERANCE:\n"
            f"\"{text}\"\n\n"
            "TASK: Extract one hidden, repressed signifier through the following steps:\n\n"
            "STEP 1: THE SYMPTOM: Identify the ONE short phrase (1-8 words) from the utterance "
            "that seems most ambiguous.\n\n"
            "STEP 2: THE CHAIN: Convert ONLY that phrase into an unbroken IPA string.\n\n"
            "STEP 3: THE ÉQUIVOQUE: Re-segment the IPA string by moving the word boundaries to "
            "reveal just a few NEW words. This must result in a 1-8 word English phrase that sounds very similar "
            "to the original but means something different. It may use some of the original words.\n\n"
            "STEP 4: THE EDIT: If the new phrase does not make ANY sense, do the following:\n"
            "- Remove words from the beginning or end, one at a time, and check if the phrase is surprising, playful and polyvalent and contains at least one changed word from the original.\n"
            "- If you cannot make such a phrase, use the single most changed word that is closest in sound to the original.\n\n"
            "- If the final product is still not surprising, polyvalent, and provocative, return silence (...).\n\n"
            "WORK THROUGH EACH STEP LOUDLY.\n\n"
            "CITATION: <Your final 1-8 word phrase.>. No bold, no quotes, etc., just the phrase with proper capitalization and a period at the end."
        )
        try:
            response = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()
            logging.info("[ORACULAR LLM OUTPUT] Raw output and working:\n%s", raw)
            citation_match = re.search(r"CITATION:\s*(.+)", raw, re.IGNORECASE)
            if citation_match:
                line = citation_match.group(1).strip()
                line = re.sub(r'^["\']|["\']$', '', line).strip()
                return self._format_citation(line) if line else "..."
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if lines:
                line = lines[-1]
                line = re.sub(r'^["\']|["\']$', '', line).strip()
                line = re.sub(r'^(?:citation|result|output|final)[:\s]*', '', line, flags=re.IGNORECASE).strip()
                if len(line) > 80:
                    first_sentence = line.split('.')[0].strip()
                    if first_sentence and len(first_sentence) <= 80:
                        return self._format_citation(first_sentence)
                return self._format_citation(line) if line else "..."
            return "..."
        except Exception:
            logging.exception("[ORACULAR LLM OUTPUT] Exception during oracular LLM call:")
            return "..."
    
    async def _gpt_quilting_point_echo(
        self,
        user_input: str,
        tracker: Tracker,
        prior_history: str,
    ) -> str:
        if not async_client:
            return "..."
        history = _get_session_history_text(tracker)
        prompt_template = (
            "You are a Lacanian analyst. Select the Master Signifier (S1) from the NEW user input by examinning it in the context of the full discourse history. "
            "This is a signifier that supposes to be a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse. It is a 'just because' signifier that stops the endless sliding of meaning.\n\n"
            "Rules:\n"
            "- Extract the text verbatim.\n"
            "- Output format: Signifier... (First word capitalized, no quotes).\n\n"
            f"Prior History:\n{PRIOR_HISTORY_TOKEN}\n\n" 
            f"Recent dialogue:\n{history}\n\n"
            f"NEW user input:\n\"{user_input}\"\n\n"
            "Return only the Signifier?"
        )
        prompt = _apply_prior_history_limit(prompt_template, prior_history, TOTAL_PROMPT_CHAR_LIMIT)
        try:
            resp = await async_client.chat.completions.create(
                model=MODEL_NAME_FAST,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.3,
            )
            line = resp.choices[0].message.content.strip()
            line = re.sub(r"[^a-zA-Z0-9 ]", "", line).strip()
            if line:
                words = line.split()
                if words:
                    words[0] = words[0].capitalize()
                    line = " ".join([words[0]] + [w.lower() for w in words[1:]])
                return line + "?"
            return "..."
        except Exception:
            return "..."