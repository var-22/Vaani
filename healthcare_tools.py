"""
healthcare_tools.py
The "Healthcare Tools" box in the diagram. Each function is exposed to
Gemma as a callable tool (JSON-schema function-calling). Gemma decides
which tool(s) to invoke based on the translated/understood patient text;
this module just implements the deterministic business logic behind
each tool so results are auditable and don't depend on the LLM alone.

Replace the in-memory stubs (DOCTOR_DIRECTORY, book_appointment storage)
with real DB / EHR / scheduling-system calls in production.
"""

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("healthcare_tools")

# ---------------------------------------------------------------------------
# Stub data — swap for real hospital directory / scheduling API
# ---------------------------------------------------------------------------
DOCTOR_DIRECTORY = {
    "cardiology": [{"name": "Dr. Priya Raman", "id": "doc_101", "next_slot": "Today 4:30 PM"}],
    "general_medicine": [{"name": "Dr. Arjun Nair", "id": "doc_102", "next_slot": "Today 3:00 PM"}],
    "pediatrics": [{"name": "Dr. Kavya Menon", "id": "doc_103", "next_slot": "Tomorrow 10:00 AM"}],
    "orthopedics": [{"name": "Dr. Suresh Kumar", "id": "doc_104", "next_slot": "Tomorrow 11:00 AM"}],
    "emergency": [{"name": "ER On-Call Physician", "id": "doc_ER", "next_slot": "Immediate"}],
}

SYMPTOM_TO_DEPARTMENT = {
    "chest pain": "cardiology", "palpitations": "cardiology", "heart": "cardiology",
    "fracture": "orthopedics", "joint pain": "orthopedics", "back pain": "orthopedics",
    "fever": "general_medicine", "cold": "general_medicine", "cough": "general_medicine",
    "child": "pediatrics", "infant": "pediatrics",
}

_appointments_db: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Tool 1: Symptom Extraction
# ---------------------------------------------------------------------------
EMERGENCY_ACTION_BY_LEVEL = {
    "HIGH": "Immediate medical attention required.",
    "MEDIUM": "Prompt medical review recommended.",
    "LOW": "Continue normal consultation.",
}


def extract_symptoms(
    symptoms: Optional[List[str]] = None,
    duration: str = "unknown",
    severity: str = "low",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Records structured symptom data. Gemma does the actual extraction from
    the (already-translated) patient utterance and calls this tool with the
    symptoms/duration/severity it has already reasoned out -- this function
    just validates and normalizes the shape rather than re-deriving it, so
    the doctor dashboard always gets a consistent, well-formed record.
    """
    severity = severity if severity in ("low", "medium", "high") else "low"
    return {
        "symptoms": [s for s in (symptoms or []) if s],
        "duration": duration or "unknown",
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Tool 2: Emergency Detection
# ---------------------------------------------------------------------------
def detect_emergency(
    emergency: bool = False,
    level: str = "LOW",
    reason: str = "",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Records Gemma's emergency assessment. The deterministic
    EMERGENCY_KEYWORDS_EN keyword net is intentionally NOT applied inside
    this function (it has no access to the original patient text, only
    Gemma's already-reasoned arguments) -- callers cross-check it themselves
    and can force emergency=True/level=HIGH on top of whatever this returns,
    preserving the existing "never silently miss an emergency" fail-safe.
    """
    level = (level or "LOW").upper()
    if level not in ("LOW", "MEDIUM", "HIGH"):
        level = "HIGH" if emergency else "LOW"
    return {
        "emergency": bool(emergency),
        "level": level,
        "reason": reason or (
            "Emergency indicators present." if emergency else "No emergency indicators identified."
        ),
        "action": EMERGENCY_ACTION_BY_LEVEL[level],
    }


# ---------------------------------------------------------------------------
# Tool 3: Patient Detail Extraction
# ---------------------------------------------------------------------------
def extract_patient_details(
    name: str = "",
    age: str = "",
    gender: str = "",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Records any patient identity/demographic details Gemma has picked out of
    the conversation (e.g. the patient introducing themselves by name, or
    mentioning their age). Most utterances won't mention any of this --
    Gemma is expected to leave a field as an empty string when nothing was
    said, and the caller only publishes/displays whichever fields are
    actually non-empty rather than treating a blank as "unknown".
    """
    return {
        "name": (name or "").strip(),
        "age": (age or "").strip(),
        "gender": (gender or "").strip(),
    }


# ---------------------------------------------------------------------------
# Tool 4: Doctor Routing
# ---------------------------------------------------------------------------
def route_doctor(symptom_keywords: List[str], is_emergency: bool = False) -> Dict[str, Any]:
    """
    Maps extracted symptoms (or an emergency flag) to a department and
    surfaces the next available doctor.
    """
    if is_emergency:
        dept = "emergency"
    else:
        dept = "general_medicine"
        for kw in symptom_keywords:
            if kw in SYMPTOM_TO_DEPARTMENT:
                dept = SYMPTOM_TO_DEPARTMENT[kw]
                break

    candidates = DOCTOR_DIRECTORY.get(dept, DOCTOR_DIRECTORY["general_medicine"])
    return {"department": dept, "candidates": candidates}


# ---------------------------------------------------------------------------
# Tool 5: Appointment Booking
# ---------------------------------------------------------------------------
def book_appointment(doctor_id: str, patient_name: str, reason: str) -> Dict[str, Any]:
    appt_id = str(uuid.uuid4())[:8]
    record = {
        "appointment_id": appt_id,
        "doctor_id": doctor_id,
        "patient_name": patient_name,
        "reason": reason,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "CONFIRMED",
    }
    _appointments_db[appt_id] = record
    logger.info(f"Booked appointment {appt_id} for {patient_name} with {doctor_id}")
    return record


# ---------------------------------------------------------------------------
# Tool schema exposed to Gemma for function calling
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "extract_symptoms",
        "description": "Record the symptoms you've identified in the patient's (English-translated) statement, how long they've had them, and how severe you judge them to be.",
        "parameters": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Each distinct symptom mentioned, in plain English (e.g. 'fever', 'headache').",
                },
                "duration": {
                    "type": "string",
                    "description": "How long the patient has had these symptoms, in their own words (e.g. '3 days'), or 'unknown' if not mentioned.",
                },
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["symptoms", "duration", "severity"],
        },
    },
    {
        "name": "detect_emergency",
        "description": "Record whether the patient's statement describes a medical emergency requiring immediate attention. Check specifically for chest pain, breathing difficulty, severe bleeding, unconsciousness, stroke symptoms, and severe allergic reaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "emergency": {"type": "boolean"},
                "level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                "reason": {"type": "string", "description": "One short sentence explaining the call either way."},
            },
            "required": ["emergency", "level", "reason"],
        },
    },
    {
        "name": "extract_patient_details",
        "description": "Record any patient identity/demographic details mentioned anywhere in the conversation so far, such as their name, age, or gender. Leave a field as an empty string if it was not mentioned -- do not guess or invent a value.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Patient's name, if they said it. Empty string if not mentioned."},
                "age": {"type": "string", "description": "Patient's age, if mentioned (e.g. '45' or '45 years'). Empty string if not mentioned."},
                "gender": {"type": "string", "description": "Patient's gender, if mentioned. Empty string if not mentioned."},
            },
            "required": ["name", "age", "gender"],
        },
    },
    {
        "name": "route_doctor",
        "description": "Given symptom keywords (and emergency flag), recommend a department and the next available doctor.",
        "parameters": {
            "type": "object",
            "properties": {
                "symptom_keywords": {"type": "array", "items": {"type": "string"}},
                "is_emergency": {"type": "boolean"},
            },
            "required": ["symptom_keywords"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment for the patient with a specific doctor once they confirm they want to proceed.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctor_id": {"type": "string"},
                "patient_name": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["doctor_id", "patient_name", "reason"],
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "extract_symptoms": extract_symptoms,
    "detect_emergency": detect_emergency,
    "extract_patient_details": extract_patient_details,
    "route_doctor": route_doctor,
    "book_appointment": book_appointment,
}
