"""
gemma_brain.py
The "Gemma 4 (AI Brain)" box: takes the raw patient transcript (Tamil/Telugu),
and in one reasoning loop handles:
    - Translation Engine   (Tamil/Telugu -> English, with medical-term care)
    - Medical Understanding (symptom/intent parsing)
    - Function Calling      (invokes healthcare_tools as needed)
    - Response Generation   (final English text for the doctor + optional
                             short ack back to the patient)

Supports two interchangeable backends:
    GEMMA_BACKEND=ollama   -> local Gemma via Ollama's /api/chat
    GEMMA_BACKEND=google   -> Gemma via Google AI Studio (Gemini-compatible API)

Function calling is implemented via a strict JSON-tool-call protocol in the
system prompt rather than relying on a specific provider's native tool-calling
schema, so it works identically across backends.
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from config import (
    EMERGENCY_KEYWORDS_EN,
    GEMMA_BACKEND,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from healthcare_tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

logger = logging.getLogger("gemma_brain")

MAX_TOOL_HOPS = 4  # safety cap so a looping model can't hang the call forever

SYSTEM_PROMPT = """You are the clinical triage AI brain inside a patient-doctor \
voice translation system. A patient has spoken in Tamil or Telugu; you receive \
the transcript in the original language.

Your job, in order:
1. Translate the patient's statement into clear, medically precise English. \
   Preserve symptom details, duration, and severity exactly — never soften or \
   omit clinical details.
2. Understand the medical intent (symptom report, follow-up question, \
   appointment request, emergency, etc).
3. Call the available tools when useful, to extract symptoms, check for \
   emergencies, route to a department/doctor, or book an appointment. \
   ALWAYS call detect_emergency first for any symptom-bearing statement.
4. Produce a final structured result for the doctor.

TOOLS AVAILABLE (call by emitting ONLY a JSON object, nothing else, in this \
exact shape when you want to call a tool):
{"tool_call": {"name": "<tool_name>", "arguments": {...}}}

When you are done calling tools and are ready to give the final answer, \
respond ONLY with this JSON shape (no markdown fences, no prose outside it):
{
  "final": {
    "english_translation": "<faithful English translation of the patient's statement>",
    "doctor_summary": "<concise clinical summary for the doctor, in English>",
    "is_emergency": <true|false>,
    "department": "<department or null>",
    "patient_ack_message": "<short reassuring message to relay back to the patient, in English; will be translated back to the patient's language before playback>"
  }
}

Available tools:
""" + json.dumps(TOOL_SCHEMAS, indent=2)

# Used only by translate_simple() for doctor -> patient replies: a plain,
# fast translation with no clinical reasoning or tool-calling involved.
TRANSLATE_SYSTEM_PROMPT = """You are a translation engine inside a doctor-patient \
voice call. Translate the given text faithfully from the source language to the \
target language, preserving medical meaning and tone (e.g. instructions, \
reassurance). Respond ONLY with this JSON shape, no markdown fences, no prose \
outside it:
{"translation": "<translated text>"}"""

# Used only by translate_simple_stream() -- the low-latency voice path. This
# deliberately has NO JSON wrapper (unlike TRANSLATE_SYSTEM_PROMPT above):
# partial JSON fragments aren't safely speakable mid-stream, but plain text
# chunks are, so streamed responses here go straight into TTS as they arrive.
STREAM_TRANSLATE_SYSTEM_PROMPT = """You are a translation engine inside a doctor-patient \
voice call. Translate the given text faithfully from the source language to the \
target language, preserving medical meaning and tone (e.g. instructions, \
reassurance). Respond with ONLY the translated text -- no JSON, no quotes, no \
labels, no explanation, nothing in the source language."""

# Used only by analyze_clinical() -- the background, non-blocking clinical
# dashboard path (extracted symptoms + emergency flag). Scoped to exactly
# these two tools, and asks for both to be called in a single turn (a JSON
# array of tool calls) instead of the sequential one-hop-at-a-time protocol
# SYSTEM_PROMPT uses, since there's no need for multi-turn reasoning here and
# a single round-trip is meaningfully faster.
_ANALYZE_TOOL_NAMES = {"extract_symptoms", "detect_emergency", "extract_patient_details"}
ANALYZE_TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS if t["name"] in _ANALYZE_TOOL_NAMES]

ANALYZE_CLINICAL_SYSTEM_PROMPT = """You analyze a single patient statement (already \
translated to English) for a doctor-facing clinical dashboard. Call ALL THREE \
tools below, in a single response, by returning a JSON array of tool calls. Do \
not ask follow-up questions and do not call any other tool. Respond ONLY with \
this JSON shape, no markdown fences, no prose outside it:
{"tool_calls": [
  {"name": "extract_symptoms", "arguments": {"symptoms": [...], "duration": "...", "severity": "low"|"medium"|"high"}},
  {"name": "detect_emergency", "arguments": {"emergency": true|false, "level": "LOW"|"MEDIUM"|"HIGH", "reason": "..."}},
  {"name": "extract_patient_details", "arguments": {"name": "...", "age": "...", "gender": "..."}}
]}

Tool argument guidance:
- extract_symptoms: list every distinct symptom mentioned, in plain English. \
"duration" is how long the patient has had these symptoms in their own words \
(e.g. "3 days"), or "unknown" if not mentioned. "severity" is your best \
clinical read of urgency based on wording, not just symptom count.
- detect_emergency: check specifically for chest pain, breathing difficulty, \
severe bleeding, unconsciousness, stroke symptoms (face drooping, slurred \
speech, one-sided weakness), and severe allergic reaction, plus any other \
clearly life-threatening presentation. "reason" is one short sentence \
explaining the call either way.
- extract_patient_details: only fill in name/age/gender if the patient \
actually stated them IN THIS statement (e.g. "My name is Priya" or "I'm 45 \
years old") -- this is a single utterance, not the whole conversation, so \
most of the time none of these will be mentioned. Leave a field as an empty \
string rather than guessing or inferring it.

Available tools:
""" + json.dumps(ANALYZE_TOOL_SCHEMAS, indent=2)

# Used only by explain_terms() for the chat UI's "What does this mean?" action:
# plain-language definitions of medical terms, never a diagnosis or advice.
EXPLAIN_TERMS_SYSTEM_PROMPT = """You identify medical or clinical terms in the \
given text and explain each one in plain, non-technical language, in the \
target language. Do not diagnose, give medical advice, or speculate about the \
patient's condition — only explain what each term means. If there are no \
medical terms worth explaining, return an empty list. Respond ONLY with this \
JSON shape, no markdown fences, no prose outside it:
{"terms": [{"term": "<term as it appears in the text>", "explanation": "<plain-language explanation in the target language>"}]}"""


class GemmaBrainResult:
    def __init__(
        self,
        english_translation: str,
        doctor_summary: str,
        is_emergency: bool,
        department: Optional[str],
        patient_ack_message: str,
        tool_trace: List[Dict[str, Any]],
    ):
        self.english_translation = english_translation
        self.doctor_summary = doctor_summary
        self.is_emergency = is_emergency
        self.department = department
        self.patient_ack_message = patient_ack_message
        self.tool_trace = tool_trace


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from a model reply,
    tolerating stray markdown fences some models add despite instructions."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


class GemmaBrain:
    def __init__(self, backend: str = GEMMA_BACKEND):
        self.backend = backend

    async def _call_model(self, messages: List[Dict[str, str]]) -> str:
        if self.backend == "ollama":
            return await self._call_ollama(messages)
        elif self.backend == "google":
            return await self._call_google(messages)
        raise ValueError(f"Unknown GEMMA_BACKEND: {self.backend}")

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama call failed [{resp.status}]: {text}")
                data = await resp.json()
        return data["message"]["content"]

    async def _call_google(self, messages: List[Dict[str, str]]) -> str:
        # Google AI Studio serves Gemma through the same generateContent
        # REST shape as Gemini. Auth uses the x-goog-api-key header (the
        # current recommended method — the older ?key= query param still
        # works but Google is phasing out unrestricted query-param keys).
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"

        # Gemma on this endpoint doesn't support a separate system role, so
        # fold the system prompt into the first user turn instead.
        contents = []
        for m in messages:
            if m["role"] == "system":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
                continue
            role = "user" if m["role"] in ("user", "tool") else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        # Ask the API to enforce JSON at the transport level. Prompt-only JSON
        # instructions are unreliable with Gemma 4's visible reasoning output.
        payload = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }

        headers = {"x-goog-api-key": GOOGLE_API_KEY, "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Google Gemma call failed [{resp.status}]: {text}")
                data = await resp.json()

        # Even with thinkingLevel "minimal", this model still emits a leading
        # `{"thought": true}` part with empty text before the real answer —
        # parts[0] alone is always "". Concatenate every non-thought part.
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    async def _call_google_stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Same request shape as _call_google(), but hits Gemini's SSE
        streaming endpoint and yields text pieces as they're generated —
        used only by translate_simple_stream() for the low-latency voice
        path. No responseMimeType/JSON enforcement here on purpose: the
        streamed output is meant to be plain speakable text, not JSON."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:streamGenerateContent?alt=sse"

        contents = []
        for m in messages:
            if m["role"] == "system":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
                continue
            role = "user" if m["role"] in ("user", "tool") else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "contents": contents,
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "minimal"}},
        }
        headers = {"x-goog-api-key": GOOGLE_API_KEY, "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Google Gemma stream call failed [{resp.status}]: {text}")
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if not payload_str:
                        continue
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    for cand in chunk.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            if part.get("thought"):
                                continue
                            piece = part.get("text", "")
                            if piece:
                                yield piece

    async def process_utterance(self, patient_transcript: str, source_language: str) -> GemmaBrainResult:
        """
        Runs the full translate -> understand -> tool-call -> respond loop
        for one patient utterance.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Patient language: {source_language}\n"
                    f"Patient transcript (original language): {patient_transcript}"
                ),
            },
        ]

        tool_trace: List[Dict[str, Any]] = []

        for hop in range(MAX_TOOL_HOPS):
            reply_text = await self._call_model(messages)
            parsed = _extract_json_object(reply_text)

            if not parsed:
                logger.warning("Model reply was not valid JSON, retrying with a nudge: %r", reply_text[:500])
                messages.append({"role": "assistant", "content": reply_text})
                messages.append(
                    {"role": "user", "content": "Respond with ONLY the JSON object described in the instructions."}
                )
                continue

            if "tool_call" in parsed:
                call = parsed["tool_call"]
                name = call.get("name")
                args = call.get("arguments", {})
                fn = TOOL_IMPLEMENTATIONS.get(name)

                if not fn:
                    tool_result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        tool_result = fn(**args)
                    except Exception as e:  # noqa: BLE001
                        tool_result = {"error": str(e)}

                tool_trace.append({"tool": name, "arguments": args, "result": tool_result})

                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append(
                    {"role": "user", "content": f"Tool result for {name}: {json.dumps(tool_result)}"}
                )
                continue

            if "final" in parsed:
                f = parsed["final"]
                return GemmaBrainResult(
                    english_translation=f.get("english_translation", ""),
                    doctor_summary=f.get("doctor_summary", ""),
                    is_emergency=bool(f.get("is_emergency", False)),
                    department=f.get("department"),
                    patient_ack_message=f.get("patient_ack_message", ""),
                    tool_trace=tool_trace,
                )

            # Unexpected shape — bail out safely
            logger.warning("Gemma reply had neither 'tool_call' nor 'final' key: %r", parsed)
            break

        # Safety fallback if the model never converges within MAX_TOOL_HOPS —
        # never silently drop a patient utterance, especially given emergency risk.
        logger.error("Gemma brain did not converge to a final answer; using safe fallback.")
        return GemmaBrainResult(
            english_translation=patient_transcript,
            doctor_summary="[AUTOMATIC TRANSLATION UNAVAILABLE — raw transcript shown, please verify with patient directly]",
            is_emergency=True,  # fail safe: flag for human review rather than silently pass
            department=None,
            patient_ack_message="One moment, connecting you with the doctor.",
            tool_trace=tool_trace,
        )

    async def translate_simple(self, text: str, source_language: str, target_language: str) -> str:
        """
        Lightweight one-shot translation for doctor -> patient replies. Unlike
        process_utterance(), this does no clinical reasoning or tool-calling —
        it's just a fast translation, so it fails open (returns the original
        text) rather than blocking playback if the model call breaks.
        """
        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source language: {source_language}\n"
                    f"Target language: {target_language}\n"
                    f"Text: {text}"
                ),
            },
        ]
        try:
            reply_text = await self._call_model(messages)
            parsed = _extract_json_object(reply_text)
            if parsed and "translation" in parsed:
                return parsed["translation"]
            logger.warning("translate_simple got an unexpected reply, using original text: %r", reply_text[:500])
        except Exception:
            logger.exception("translate_simple call failed, using original text")
        return text

    async def translate_simple_stream(
        self, text: str, source_language: str, target_language: str
    ) -> AsyncIterator[str]:
        """Low-latency counterpart to translate_simple(): yields the
        translation as plain text chunks as Gemma generates them, so the
        caller (the streaming TTS path) can start speaking before the whole
        sentence exists. Only the "google" backend actually streams token by
        token; "ollama" falls back to one non-streamed chunk rather than
        silently doing nothing, since this deployment defaults to
        GEMMA_BACKEND=google and Ollama's streaming API isn't wired up here."""
        if self.backend != "google":
            yield await self.translate_simple(text, source_language, target_language)
            return

        messages = [
            {"role": "system", "content": STREAM_TRANSLATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source language: {source_language}\n"
                    f"Target language: {target_language}\n"
                    f"Text: {text}"
                ),
            },
        ]
        yielded_any = False
        try:
            async for piece in self._call_google_stream(messages):
                yielded_any = True
                yield piece
        except Exception:
            logger.exception("translate_simple_stream failed")
            if not yielded_any:
                # Nothing went out yet, so it's safe to fail open with one
                # full non-streamed attempt. If we'd already yielded partial
                # text, retrying here would produce duplicated/garbled
                # speech, so we deliberately just stop instead.
                try:
                    yield await self.translate_simple(text, source_language, target_language)
                except Exception:
                    logger.exception("translate_simple fallback also failed")

    async def explain_terms(self, text: str, target_language: str) -> list:
        """
        On-demand medical-terminology explainer for the chat UI's "What does
        this mean?" action. Fails open to an empty list — a missing
        explanation is a minor UX gap, not something worth surfacing an error
        for, and never blocks the underlying translation/chat flow.
        """
        messages = [
            {"role": "system", "content": EXPLAIN_TERMS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Target language: {target_language}\nText: {text}",
            },
        ]
        try:
            reply_text = await self._call_model(messages)
            parsed = _extract_json_object(reply_text)
            if parsed and isinstance(parsed.get("terms"), list):
                return parsed["terms"]
            logger.warning("explain_terms got an unexpected reply: %r", reply_text[:500])
        except Exception:
            logger.exception("explain_terms call failed")
        return []

    async def analyze_clinical(self, patient_text_english: str) -> Dict[str, Any]:
        """
        Runs the extract_symptoms + detect_emergency + extract_patient_details
        tools (all three, in one round-trip -- see
        ANALYZE_CLINICAL_SYSTEM_PROMPT) for the doctor dashboard. Deliberately
        separate from process_utterance()/translate_simple_stream(): this is
        slower (a full tool-calling round-trip) and is meant to run in the
        background without gating the doctor hearing the spoken translation,
        which is latency-critical.

        Returns {"symptoms": {...}, "emergency": {...}, "patient_details": {...}}
        — see the matching healthcare_tools functions for the exact shape of
        each. On top of whatever Gemma decides, this cross-checks
        EMERGENCY_KEYWORDS_EN against the raw text and forces
        emergency=True/level=HIGH if a hard-trigger phrase is present, so a
        bad/missed tool call can never silently suppress an emergency flag.
        """
        messages = [
            {"role": "system", "content": ANALYZE_CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient statement (English): {patient_text_english}"},
        ]

        tool_calls: List[Dict[str, Any]] = []
        try:
            reply_text = await self._call_model(messages)
            parsed = _extract_json_object(reply_text)
            if parsed and isinstance(parsed.get("tool_calls"), list):
                tool_calls = parsed["tool_calls"]
            else:
                logger.warning("analyze_clinical got an unexpected reply: %r", reply_text[:500])
        except Exception:
            logger.exception("analyze_clinical model call failed")

        symptoms_result = None
        emergency_result = None
        patient_details_result = None
        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments") or {}
            fn = TOOL_IMPLEMENTATIONS.get(name)
            if not fn:
                continue
            try:
                result = fn(**args)
            except Exception:
                logger.exception("Tool %s failed with arguments %r", name, args)
                continue
            if name == "extract_symptoms":
                symptoms_result = result
            elif name == "detect_emergency":
                emergency_result = result
            elif name == "extract_patient_details":
                patient_details_result = result

        if symptoms_result is None:
            symptoms_result = TOOL_IMPLEMENTATIONS["extract_symptoms"]()
        if emergency_result is None:
            emergency_result = TOOL_IMPLEMENTATIONS["detect_emergency"]()
        if patient_details_result is None:
            patient_details_result = TOOL_IMPLEMENTATIONS["extract_patient_details"]()

        text_lower = patient_text_english.lower()
        keyword_hits = [kw for kw in EMERGENCY_KEYWORDS_EN if kw in text_lower]
        if keyword_hits and not emergency_result["emergency"]:
            emergency_result = TOOL_IMPLEMENTATIONS["detect_emergency"](
                emergency=True,
                level="HIGH",
                reason=f"Deterministic safety check matched: {', '.join(keyword_hits)}.",
            )

        return {
            "symptoms": symptoms_result,
            "emergency": emergency_result,
            "patient_details": patient_details_result,
        }
