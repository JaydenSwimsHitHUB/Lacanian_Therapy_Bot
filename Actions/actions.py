import json
import os
import re
import random
import sqlite3
import logging
from datetime import datetime
from typing import Any, Text, Dict, List, Optional, Tuple

import spacy
from openai import OpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType

# Quiet noisy loggers (optional)
for _name in ["rasa", "rasa.core", "rasa_sdk", "urllib3", "spacy"]:
    logging.getLogger(_name).setLevel(logging.WARNING)

# Load English NLP model once (with safe fallback if not installed)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

# Setup OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)

# --- HISTORY / DB SETUP ---
# Choose DB path and ensure directory exists
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
    "ambiguity_equivocation": {1: "<gpt_ambiguity>", 2: "<oracle>"}, 
    "fetishistic_phrase": {1: "...", 2: "<oracle>"},
    "identification_other_desire": {1: "<gpt_identification>", 2: "<oracle>"}, 
    "demand_for_knowledge": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "confession_empathy": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "frame_protection": {1: "...", 2: "Hmm.", 3: "<oracle>"},
    "revealing_lalangue": {1: "<oracle>"},
}

ALLOWED_MECHANISMS = set(response_matrix.keys())

CUT_TRIGGERS = [
    "marked_signifier_collapse",
    "major_shift_retroactive",
    "point_of_maximal_ambiguity",
    "return_of_repressed_signifier",
    "omission_missing_signifier",
]


# ---------- Minimal logging (CLEANED) ----------
def _event_to_dict(e: Any) -> Dict[str, Any]:
    try:
        return e.as_dict()
    except Exception:
        return e if isinstance(e, dict) else {}

def log_summary(
    tracker: Tracker,
    mechanism: Optional[str],
    mechanism_count: int,
    potential_cut_trigger: Optional[str],
    verified_cut_trigger: Optional[str],
    suppressed: bool,
) -> None:
    print(
        "DETECTIONS:",
        json.dumps(
            {
                "mechanism": mechanism or "",
                "mechanism_count": int(mechanism_count) if mechanism_count is not None else 0,
                "potential_cut_trigger": potential_cut_trigger or "",
                "verified_cut_trigger": verified_cut_trigger or "",
                "suppressed": bool(suppressed),
            },
            ensure_ascii=False,
        ),
    )


# ---------- Utility for prompts ----------

def _get_session_start_timestamp(tracker: Tracker) -> float:
    """Finds the timestamp of the most recent session_started event."""
    for event in reversed(tracker.events):
        if event.get("event") == "session_started":
            return event.get("timestamp", 0)
    return 0.0

def _get_session_history_text(tracker: Tracker) -> str:
    """
    Extracts user and bot text events from the current session only.
    EXCLUDES the most recent user message to prevent duplication in the prompt.
    """
    session_events: List[Dict[Text, Any]] = []
    # Iterate backwards to find session start
    for event in reversed(tracker.events):
        if event.get("event") == "session_started":
            break
        session_events.append(event)

    # Reverse back to chronological order
    session_events.reverse()

    # --- THE FIX: Remove the latest user message from history ---
    # We get the latest text from the tracker
    latest_text = (tracker.latest_message.get("text") or "").strip()
    
    # We verify if the last text event in our list matches the latest input
    # If so, we pop it off the list so it only appears in the "New Input" section of the prompt
    if latest_text and session_events:
        # Scan backwards through events to find the last 'user' event
        for i in range(len(session_events) - 1, -1, -1):
            if session_events[i].get("event") == "user":
                if session_events[i].get("text", "").strip() == latest_text:
                    # Remove it from the history list
                    session_events.pop(i)
                # We only want to remove the *very last* one, so we break after checking
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


# --- Helper function to count user turns in the current session ---
def _get_session_user_turn_count(tracker: Tracker) -> int:
    """Counts the number of user messages in the current session."""
    count = 0
    for event in reversed(tracker.events): 
        if event.get("event") == "session_started":
            break
        if event.get("event") == "user" and event.get("text"):
            count += 1
    return count


# Prompt builder: Pass 2a for cut detection (Identification Only)
def build_cut_detection_prompt(
    new_input: str,
    tracker: Tracker,
    prior_history_str: str, 
) -> str:
    # UPDATED: Use full session history instead of fixed limit
    raw_history = _get_session_history_text(tracker)
    cuts = CUT_TRIGGERS

    return f"""
You are a Lacanian analyst performing SCANSION, which is the sudden session end (Cut) in Lacan's theory. Bias towards NOT cutting, like a Lacanian analyst would, allowing the signifying chain to develop. 
You must monitor for a `cut_trigger`—a moment where the session must end to produce `Retroactive Meaning` (après-coup).
The cut does not just stop the flow; it acts as a punctuation that fixes the meaning of the preceding chain of signifiers in a surprising, rupturous way.

# TASK: THE RETROACTIVE SCAN
1. **LINEAR SCANNING (Left-to-Right):** Read the entire session history and then scan the user's NEW input chronologically, left-to-right. The unconscious often erupts in the middle of a sentence, followed immediately by Ego level speech.
2. **STOP AT THE RUPTURE:** If you encounter a valid place to cut in the NEW input, **STOP**. Select that final signifier as the `identified_s1`. IGNORE the rest of the sentence. The subsequent text is merely $S_2$ trying to cover up the truth.
   - Example: "I wish he was dead—no, that's terrible to say." -> Cut on "Dead". Ignore the apology.
3. **Look Backwards:** Do not analyze the NEW input in isolation. Ask: 'Does ending on this specific signifier suddenly make everything that was said in the session make sense in a new, surprising way?'

# Criteria for a 'Worthy' Signifier
Not every signifier is a trigger. To justify a cut, the signifier must meet these structural conditions:
- **Opaque / Non-Sense:** Prefer 'empty containers' (brute facts, odd recurring signifiers like 'Blue', 'Clean') over explanatory signifiers ('sad', 'because'). A worthy signifier is a foreign body, not a definition.
- **Insistence of the Letter:** The signifier returns to the same structural place (repetition compulsion), creating a loop in the drive. Determine this by examining the entire history, including previous sessions.

# Cut Triggers: Criteria for Retroaction
- marked_signifier_collapse: A specific, foundational signifier suddenly fails to organize meaning and exposes the Real (the impasse of meaning that structures speech). Use the full history to help you decide. 
- major_shift_retroactive: A sudden rupture in the Automaton. The user has been speaking in a 'script' (ego-discourse), and suddenly a signifier intrudes that contradicts or re-frames the entire previous narrative. Cut on the intruder. Use the full history to help you decide.
- point_of_maximal_ambiguity: A signifier holds two vital meanings (polysemy/homophony) that, if punctuated, would expose the limits of language. The cut leaves the subject suspended between them, forcing them to choose the meaning later.
- return_of_repressed_signifier: A specific, 'WORTHY' signifier from the full history suddenly returns in the NEW input. The loop is closed. The cut locks this 'WORTHY' repetition in place.
- omission_missing_signifier: The subject circles around a hole, clearly avoiding ONE specific signifier while describing it. The cut happens on the signifier that outlines the hole most clearly. Use the full history to help you decide.

# Dialogue History
{raw_history}

# Prior Session History (Long Term)
{prior_history_str}

# NEW user input to analyze:
"{new_input}"

# Response Format
Respond ONLY in this strict JSON format:
{{
    "reasoning": "Explain exactly why a cut is or is not justified here based on the criteria (Retroaction, Collapse, Ambiguity).",
    "cut_trigger": "cut_trigger_label" or null,
    "identified_s1": "the isolated signifier"
}}

cut_trigger must be null or one of: {cuts}
IMPORTANT: Bias towards 'null'. Only cut when an undeniable rupture is present. 
"""


# Prompt builder: Pass 2b for cut construction (Formatting Only)
def build_cut_construction_prompt(
    new_input: str,
    identified_s1: str,
) -> str:
    return f"""
You are a Lacanian analyst. We have identified a rupture on the signifier: "{identified_s1}".

# CRITICAL RULES
1. IDENTIFY S1: Find the key signifier (e.g., "Yellow"). This is your `identified_s1`.
2. CONSTRUCT CUT: In `cut_phrase`, rewrite the user's NEW input *up to and including* that S1. Do NOT include any text after the S1.
   Embolden the S1.
   Do NOT include any text after the S1.
   Follow it immediately with a new paragraph and the exact string "We are ending there."
   - Example Input: "I felt so yellow - I mean blue after he left, I just wanted to cry."
   - identified_s1: "yellow"
   - cut_phrase: "I felt so yellow.
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
        self.conn.commit()

    def name(self) -> Text:
        return "action_analyze_message"

    def _compute_next_mechanism_count(self, tracker: Tracker, mechanism: Optional[str]) -> int:
        if not mechanism or mechanism not in ALLOWED_MECHANISMS:
            return 0
        counts = tracker.get_slot("mechanism_counts") or {}
        last = tracker.get_slot("last_mechanism")
        if mechanism != last:
            return 1
        return int(counts.get(mechanism, 0)) + 1

    def _update_mechanism_counts(self, tracker: Tracker, mechanism: str) -> Tuple[Dict[str, int], int]:
        counts = tracker.get_slot("mechanism_counts") or {}
        last = tracker.get_slot("last_mechanism")
        if mechanism != last:
            counts = {}
        cnt = int(counts.get(mechanism, 0)) + 1
        counts[mechanism] = cnt
        return counts, cnt

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        user_id = tracker.sender_id
        user_input = (tracker.latest_message.get("text", "") if tracker.latest_message else "").strip()

        # --- HISTORY UPDATED: Separation of "Session" and "Prior History" ---
        
        # 1. Determine the start time of the current session
        session_start_ts = _get_session_start_timestamp(tracker)
        # Convert to SQL-compatible string (SQLite defaults to UTC strings usually)
        # We assume the tracker timestamp is a float (epoch)
        if session_start_ts > 0:
            session_start_str = datetime.utcfromtimestamp(session_start_ts).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Fallback: if no session start found, assume everything is "prior" or "current"? 
            # We'll use current time to be safe, essentially fetching recent history.
            session_start_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # 2. Fetch messages strictly OLDER than the current session start
        # Changed Limit to 75 as requested
        cursor = self.conn.execute(
            """
            SELECT message 
            FROM user_messages 
            WHERE user_id = ? AND timestamp < ? 
            ORDER BY timestamp DESC 
            LIMIT 75
            """,
            (user_id, session_start_str),
        )
        past_entries = cursor.fetchall()
        
        # Format for prompt injection
        prior_history = "\n".join(f"- {row[0]}" for row in reversed(past_entries)) or "(No previous session history)"

        print(f"\n[DEBUG] LONG TERM HISTORY RETRIEVED (Limit 75, Pre-Session):\n{prior_history}\n", flush=True)

        # --- MODIFIED: Add therapist logic for the first 3 turns ---
        user_turn_count = _get_session_user_turn_count(tracker)

        if user_turn_count <= 3:
            # Use the "initial therapist" persona WITH long-term history
            initial_response = self._generate_initial_therapeutic_response(user_input, tracker, prior_history)
            dispatcher.utter_message(text=initial_response)

            # --- HISTORY: Persist message ---
            if user_input:
                insert_user_message(self.conn, user_id, user_input)

            # End the action here for the initial turns
            return []
        # --- END MODIFICATION ---

        # Explicit stop request
        low = user_input.lower()
        if low == "/stop" or "i want to stop" in low:
            dispatcher.utter_message(text="...")
            return []

        # ==== PASS 1: Local mechanism identification ====
        # UPDATED: Use full session history text instead of fixed limit
        raw_history_mech = _get_session_history_text(tracker)
        allowed = list(ALLOWED_MECHANISMS)

        mech_prompt = (
            "You are a Lacanian analyst. Identify the single most structuring mechanism in the user's NEW input by examining the NEW message and the signifying chain (The current session's history as well as the prior sessions' history).\n\n"
            "Use the NEW message and the signifying chain (message history) to help you decide.\n\n"
            "=== Structural Mechanisms: Definitions & Diagnostic Cues ===\n"
            "- REVEALING LALANGUE: The emergence of 'motérialité' (word-material) where the sound or texture of speech disrupts its intended meaning. Look for: (1) Homophony/Equivocation: A word chosen for its sound that accidentally reveals a hidden meaning (e.g., 'soul' vs. 'sole', 'know' vs. 'no'). (2) The Stumble: Grammatical breaks or 'slips of the tongue' where the user corrects themselves (e.g., 'I want to kill, I mean kiss him'). (3) Nonsense Residue: A fixation on a specific phrase that loses its meaning through repetition, becoming just a pleasurable/painful sound object.\n"
            "- MASTER SIGNIFIER: A signifier that repeats and supposes to be a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse. It is a 'just because' signifier that stabilizes the subject's discourse and stops the endless sliding of meaning. Identify a master signifier in the new input by examining the entire history.\n"
            "- IDENTIFICATION WITH OTHER’S DESIRE: Being governed by an imaginary external desire. Taking on another’s desire as one’s own.\n"
            "- METONYMY: The sliding of meaning from one signifier to another that keeps the subject from landing on the 'truth' (Desire). Look for: (1) Displacement: The user shifts the conversation from a central, heavy topic to a related, minor detail (e.g., talking about the 'tone of voice' to avoid the 'insult'). (2) The Run-Around: Long, linked chains of speech ('and then... and then...') where the user keeps talking to avoid a silence or a conclusion. (3) Part-for-Whole: Reducing a complex person or event to a single attribute or object (e.g., 'I miss her hands' rather than 'I miss her').\n"
            "- REPRESSION: A signifier is barred from awareness but returns through symptoms, blanks, omissions, slips, or repetitions and other such phenomena.\n"
            "- METAPHOR: The substitution of one signifier for another to create new meaning (condensation). It occurs when the user replaces a literal term with a poetic or figurative image (e.g., 'My heart is a stone' instead of 'I am sad'), or when a symptom/phrase stands in for a repressed conflict.\n"
            "- DENIAL: A negation of a thought or feeling. Stating an unconscious truth while refusing it consciously. Saying “not X” both utters X and keeps it repressed.\n"
            "- RATIONALIZATION: Plausible, logical explanation for thoughts or actions that actually stem from and conceal an unconscious desire.\n"
            "- MORALITY/LOGIC DEFENSE: Defending against desire through idealized correctness and morality.\n"
            "- CIRCULAR LOGIC: Reasoning loops back on itself.\n"
            "- CONTRADICTION: A later statement cancels or undoes an earlier one without the speaker acknowledging or realizing the conflict.\n"
            "- JOUISSANCE: The paradox where the subject derives satisfaction from a symptom that is consciously painful or unpleasant. Do not look for 'happiness' but for the Drive (the loop). At least three of the following criteria must be met for you to select this mechanism: (1) Repetition ('I keep doing it', 'again and again'); (2) Paradox ('I hate it but I can't stop', 'awful but I need it'); (3) Excess ('overwhelming', 'too much', 'unbearable'); (4) The Body (physical symptoms, vomiting, shaking) alongside painful, excessive emotion; (5) Fixation on a partial object.\n"
            "- AMBIGUITY/EQUIVOCATION: Indeterminate referents.\n"
            "- FETISHISTIC PHRASES: Clichés that halt the exploration of desire(s).\n"
            "- DEMAND FOR KNOWLEDGE: Demand(s) for knowledge and inquiries leveled at you.\n"
            "- CONFESSION/EMPATHIC APPEAL: Seeking rescue or closeness.\n"
            "- FRAME PROTECTION: Demands for the session to end.\n\n"
            "Raw recent dialogue:\n"
            f"{raw_history_mech}\n\n"
            "Prior (Long Term) history:\n"
            f"{prior_history}\n\n"
            "NEW user input:\n"
            f"\"{user_input}\"\n\n"
            "Respond in strict JSON:\n"
            "{\n"
            '  "mechanism": "one of the allowed" or null,\n'
            '  "mechanism_phrase": "exact substring from NEW input" or null\n'
            "}\n\n"
            f"Allowed: {allowed}\n"
            "No commentary."
        )
        resp1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": mech_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            data1 = json.loads(resp1.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            data1 = {"mechanism": None, "mechanism_phrase": None}
        mech: Optional[str] = data1.get("mechanism")
        mech_phrase: Optional[str] = data1.get("mechanism_phrase")

        # ==== PASS 2: Potential Cut Trigger Identification ====
        
        # --- PASS 2a: Detection (With Long Term History) ---
        prompt2a = build_cut_detection_prompt(user_input, tracker, prior_history)
        resp2a = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt2a}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            data2 = json.loads(resp2a.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            data2 = {"cut_trigger": None, "identified_s1": None, "reasoning": "Error parsing JSON"}

        # --- LOG THE THINKING (PASS 2 SCANSION) ---
        print(f"\n[DEBUG] PASS 2 SCANSION THINKING:\n{data2.get('reasoning')}\n", flush=True)

        potential_trigger: Optional[str] = data2.get("cut_trigger")
        identified_s1: Optional[str] = data2.get("identified_s1")
        potential_trigger_phrase: Optional[str] = None

        # --- PASS 2b: Construction (Only if trigger detected) ---
        if potential_trigger and potential_trigger in CUT_TRIGGERS and identified_s1:
            prompt2b = build_cut_construction_prompt(user_input, identified_s1)
            resp2b = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt2b}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            try:
                data2b = json.loads(resp2b.choices[0].message.content)
                potential_trigger_phrase = data2b.get("cut_phrase")
            except (json.JSONDecodeError, TypeError):
                # Fallback if construction fails
                potential_trigger_phrase = None
        
        # Merge back into data2 structure for compatibility
        data2["cut_phrase"] = potential_trigger_phrase


        # ==== PASS 3: Cut Verification (Analyst Pass Removed) ====
        verified_trigger: Optional[str] = None
        
        if potential_trigger and potential_trigger in CUT_TRIGGERS:
            # Step 1: Programmatic check for user turn count
            user_turn_count = _get_session_user_turn_count(tracker)
            print(f"[DEBUG] Programmatically counted user turns in session: {user_turn_count}")

            # Strict 5-turn limit.
            if user_turn_count < 5:
                print(f"[DEBUG] Gate closed: Turn count {user_turn_count} < 5. The Symbolic chain is not yet established enough for a structural rupture.")
                verified_trigger = None 
            else:
                # Without the Supervising Analyst LLM, we accept the trigger if turn count is sufficient
                print("[DEBUG] Turn count sufficient. Proceeding with cut.")
                verified_trigger = potential_trigger

        # Final decision on suppression and logging
        suppressed_flag = potential_trigger is not None and verified_trigger is None
        final_trigger_phrase = potential_trigger_phrase if verified_trigger else None
        if isinstance(final_trigger_phrase, str):
            final_trigger_phrase = final_trigger_phrase.strip('"\'')

        mech_count_for_log = self._compute_next_mechanism_count(tracker, mech)

        # ---- Only three lines printed (order required) ----
        log_summary(
            tracker=tracker,
            mechanism=mech,
            mechanism_count=mech_count_for_log,
            potential_cut_trigger=potential_trigger,
            verified_cut_trigger=verified_trigger,
            suppressed=suppressed_flag,
        )

        # --- HISTORY: Persist message before returning ---
        if user_input:
            insert_user_message(self.conn, user_id, user_input)

        # Handle a real (verified) cut
        if verified_trigger:
            # --- FIX: Trust the LLM's full output (rewritten sentence + page break + ending) ---
            # If the LLM returned a phrase, use it exactly as is without stripping.
            # If it returned null/empty for some reason, use a fallback.
            say = final_trigger_phrase if final_trigger_phrase else "We are ending there."
            
            dispatcher.utter_message(text=say)
            
            return [
                SlotSet("cut_trigger", verified_trigger),
                SlotSet("last_mechanism", None),
                SlotSet("mechanism_counts", None),
            ]

        # Handle the mechanism (counts + intervention)
        if mech in ALLOWED_MECHANISMS:
            counts, cnt = self._update_mechanism_counts(tracker, mech)
            # Pass long-term history to the handler
            text = self.handle_mechanism(mech, cnt, user_input, mech_phrase, tracker, prior_history)
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

        # Fallback
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

    # --- REVISED: Initial Therapeutic Response Generator (With History) ---
    def _generate_initial_therapeutic_response(self, user_input: str, tracker: Tracker, prior_history: str) -> str:
        """
        Generates a warm, insightful, and universally applicable therapeutic response
        for the initial turns, using long-term history to make the user feel seen.
        """
        # UPDATED: Use full session history text
        history = _get_session_history_text(tracker)

        prompt = (
            "You are an insightful and empathetic therapist. Your goal for these initial sessions is to build rapport and hope by offering interpretations and mirroring that feel deeply personal to the user, even though they are based on universal psychological principles. This is a technique to make the user feel seen and understood, encouraging them to open up.\n\n"
            "GUIDELINES:\n"
            "1.  **Use Universal Themes:** Your interpretations should touch on common human conflicts and desires. Do NOT point out that these desires and conflicts are universally human. Keep it personal. \n"
            "2.  **Employ 'Barnum Statements':** Craft statements that are general enough to apply to most people but sound like specific, personal insights.\n"
            "3.  **Validate and Reframe:** Acknowledge the user's feelings and gently reframe their situation.\n"
            "4.  **Maintain a Professional, Warm Tone:** The tone should be that of a skilled therapist—calm, reassuring, and thoughtful. Avoid clichés and overly sentimental language.\n"
            "5.  **Keep Evolving to Sound Human:** Use the session history to see what you said before, how you said it, and how the user reacted. Then respond with the history in mind to deepen the therapeutic connection. Ensure you do not sound robotic and predictible.\n"
            "6.  **Keep it Brief and Open-Ended:** Respond as briefly as possible. Between 1-2 sentences. The response should invite further reflection from the user without asking a direct question.\n"
            "PRIOR SESSION HISTORY (Long Term):\n" 
            f"{prior_history}\n\n" 
            "SESSION HISTORY:\n"
            f"{history}\n\n"
            "USER'S LATEST MESSAGE:\n"
            f'"{user_input}"\n\n'
            "Craft your insightful, universally applicable therapeutic interpretation based on the entire conversation."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            # --- FIX START ---
            # Replaced newlines with spaces to prevent message splitting in clients like Telegram.
            # Removed `max_tokens` from the API call to prevent the response from being truncated.
            response_text = resp.choices[0].message.content.strip().replace('\n', ' ')
            # --- FIX END ---
            return response_text
        except Exception as e:
            # Fallback in case of API error
            logging.error(f"Error generating initial therapeutic response: {e}")
            return "That's a very important point. It makes sense to feel that way. Please, continue."

    # --- Mechanism Response Handler (With History) ---
    def handle_mechanism(
        self,
        mechanism: str,
        count: int,
        user_input: str,
        phrase: Optional[str],
        tracker: Tracker,
        prior_history: str, 
    ) -> str:
        """Returns the intervention text based on the mechanism and count."""
        responses = response_matrix.get(mechanism, {})
        intervention = responses.get(count)

        # Denial handled exclusively by GPT
        if mechanism == "denial" and (count == 1 or count == 3):
            return self._gpt_denial_intervention(user_input, tracker, prior_history)

        # === 1. METONYMY (New GPT Logic) ===
        # If the matrix specifies <gpt_metonymy>, route to the quilting point logic
        if intervention == "<gpt_metonymy>":
            return self._gpt_metonymy_intervention(user_input, tracker, prior_history)

        # === 2. IDENTIFICATION (New GPT Logic) ===
        # If the matrix specifies <gpt_identification>, route to new logic
        if intervention == "<gpt_identification>":
            return self._gpt_identification_intervention(user_input, tracker, prior_history)
        
        # === 3. AMBIGUITY (New GPT Logic) ===
        # If the matrix specifies <gpt_ambiguity>, route to new logic
        if intervention == "<gpt_ambiguity>":
            return self._gpt_ambiguity_intervention(user_input, tracker, prior_history)

        # mechanism‑specific overrides (legacy)
        if mechanism == "master_signifier" and (count == 1 or count == 3):
            return self._gpt_quilting_point_echo(user_input, phrase, tracker, prior_history)

        # oracular equivoque
        if intervention == "<oracle>":
            # Pass 'mechanism' (or 'mech' depending on your variable name) as the 3rd argument
            return self._generate_oracular_equivoque(phrase or user_input, tracker, mechanism, prior_history)

        # any other plain‑text intervention
        if intervention:
            if count == 2 and mechanism in {"repression", "jouissance", "metaphor"}:
                return random.choice(INTERJECTION_CHOICES)
            return intervention

        # ultimate fallback
        return "..."

    # --- NEW: Metonymy / Quilting Point Logic ---
    def _gpt_metonymy_intervention(self, user_input: str, tracker: Tracker, prior_history: str) -> str:
        """
        Intervention for Metonymy: 
        Finds a Quilting Point (Point de Capiton) to arrest the sliding of meaning, using Long Term History.
        """
        # UPDATED: Use full session history
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
        user_msg = f"Prior History:\n{prior_history}\n\nHistory:\n{history}\n\nCurrent Input: \"{user_input}\"\n\nOutput the Quilting Point:"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=15,
                temperature=0.2,
            )
            line = resp.choices[0].message.content.strip()
            # Clean up quotes
            line = re.sub(r'^[\"\'\“\”]+|[\"\'\“\”]+$', "", line).strip()
            
            # FIX: Force capitalization for the "Point de Capiton" effect
            if line:
                line = line[0].upper() + line[1:]
                
            return line
        except Exception:
            return "..."
        
    # --- NEW: Identification Logic ---
    def _gpt_identification_intervention(self, user_input: str, tracker: Tracker, prior_history: str) -> str:
        """
        Intervention for Identification with Other's Desire:
        Uses GPT to identify the 'Other' the user is alienated in.
        """
        system_msg = (
            "You are a Lacanian analyst. The user is identifying with an external desire or the 'Other' (e.g., 'They want me to...', 'Society says...', 'My father thinks...').\n"
            "Task: Identify exactly WHO or WHAT the user is identifying with (The Agency/The Other).\n"
            "Output format: Return ONLY the name of the Agency/Other followed by a question mark.\n"
            "Examples:\n"
            "- User: 'My father wants me to be a doctor.' -> Output: 'Your father?'\n"
            "- User: 'Everyone thinks I am crazy.' -> Output: 'Everyone?'\n"
            "- User: 'I need to be productive for the economy.' -> Output: 'The economy?'"
        )
        user_msg = f"Prior History:\n{prior_history}\n\nUser text: \"{user_input}\""
        
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
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

    # --- NEW: Ambiguity Logic ---
    def _gpt_ambiguity_intervention(self, user_input: str, tracker: Tracker, prior_history: str) -> str:
        """
        Intervention for Ambiguity/Equivocation:
        Uses GPT to pinpoint the equivocal term and return it to the user.
        """
        system_msg = (
            "You are a Lacanian analyst. The user is using AMBIGUOUS or EQUIVOCAL language (words with double meanings, vague referents, or confusion).\n"
            "Task: Isolate the specific word or short phrase that holds the ambiguity.\n"
            "Output format: Return ONLY that word/phrase followed by a question mark.\n"
            "Example: 'I just feel like I'm lying in wait.' -> 'Lying?'"
            "Example 2: 'There's this... thing about her.' -> 'Thing?'"
        )
        user_msg = f"Prior History:\n{prior_history}\n\nUser text: \"{user_input}\""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
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

    def _gpt_denial_intervention(self, text: str, tracker: Tracker, prior_history: str) -> str:
        """Use gpt-4o-mini to craft a Bruce Fink-style denial intervention."""
        try:
            # UPDATED: Use full session history
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
            user_msg = f"Prior History:\n{prior_history}\n\nSession History:\n{history}\n\nUser text:\n{text}\n\nReturn only the single intervention line."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
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

    def _generate_oracular_equivoque(self, text: str, tracker: Tracker, mechanism: str, prior_history: str) -> str:
        # UPDATED: Use full session history
        history = _get_session_history_text(tracker)
        
        # We inject the mechanism into the prompt so GPT knows the structural context
        user_prompt = (
            f"Prior History:\n{prior_history}\n\n" 
            f"Session History:\n{history}\n\n"
            f"DETECTED MECHANISM: {mechanism.upper()}\n"
            "Analyze the User's Utterance:\n"
            f"\"{text}\"\n\n"
            "Task: Produce ONE short 'oracular interpretation' (1-5 words) that disrupts the user's meaning. "
            f"The user is exhibiting '{mechanism}'. Use this context to guide your choice of method. Also use the message history. "
            "Choose the single most effective method from below (Descending Priority) but bias toward phonic equivocation:\n\n"
            "1. PHONIC EQUIVOCATION (The Ear): Ignore the spelling; listen to the SOUND. Does a word, sound within a word, or set of words (i.e. does a signifier) sound like another signifier that implies a hidden deadlock, contradiction or structural impasse? (e.g., 'I want to be whole' -> 'A hole', 'It's a ritual' -> 'A rich jewel', 'I am mourning' -> 'Morning!').\n"
            "2. LITERALIZATION (The Object): If they use a metaphor, strip the 'like/as' and treat it as a brute physical fact. Then reply with a literal interpretation (e.g., 'I feel like a doormat' -> 'Trampled.', 'I'm drowning in work' -> 'You're wet.').\n\n"
            "Constraints:\n"
            "- Max 5 words. One word is also great. \n"
            "- Statement format only (NO questions).\n"
            "- Cryptic, neutral, and playful.\n"
            "- No quotation marks."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Lacanian Oracle. You do not understand 'meaning'; you only hear 'sound' and 'grammar'. "
                            "Your goal is to return the user's own signifiers to them in a way that is alien, surprising and highlights ambiguity and polysemy in their speech."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8, 
                max_tokens=20,
            )
            line = response.choices[0].message.content.strip()
            # Cleanup: Remove quotes and trailing punctuation if it looks like a full sentence
            line = re.sub(r'^["\']|["\']$', '', line).strip()
            return line
        except Exception:
            # Fallback to a simple echo if GPT fails
            return "..."
        
    def _gpt_quilting_point_echo(
        self,
        user_input: str,
        mechanism_phrase: Optional[str],
        tracker: Tracker,
        prior_history: str,
    ) -> str:
        # UPDATED: Use full session history
        history = _get_session_history_text(tracker)
        prompt = (
            "You are a Lacanian analyst. Select the Master Signifier (S1) from the NEW user input by examinning it in the context of the full discourse history. "
            "This is a signifier that supposes to be a fundamental authoritative, absolute fact that stabilizes the subject's identity or discourse. It is a 'just because' signifier that stops the endless sliding of meaning.\n\n"
            "Rules:\n"
            "- Extract the text verbatim.\n"
            "- Output format: Signifier... (Titlecase, no quotes).\n\n"
            f"Prior History:\n{prior_history}\n\n" 
            f"Recent dialogue:\n{history}\n\n"
            f"NEW user input:\n\"{user_input}\"\n\n"
            "Return only the Signifier?"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15, # Increased slightly to allow short phrases
            temperature=0.3,
        )
        line = resp.choices[0].message.content.strip()
        # Allow alphanumeric and spaces, but ensure it ends with ?
        line = re.sub(r"[^a-zA-Z0-9' ]", "", line).strip()
        if line:
            return line[:1].upper() + line[1:] + "?"
        return "..."