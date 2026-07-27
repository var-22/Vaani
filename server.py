"""
server.py
Small FastAPI backend that the two web UIs (patient.html / doctor.html) talk to:

    GET  /token?room=<room>&identity=<identity>&role=patient|doctor
         -> issues a LiveKit access token so the browser can join the room

    POST /book-appointment  {doctor_id, patient_name, reason}
         -> calls the same book_appointment() tool the Gemma brain uses,
            so a doctor tapping "Book" in the UI hits the identical
            business logic as an automatic booking

    POST /explain-terms  {text, target_language}
         -> on-demand medical-terminology explainer for the chat UI's
            "What does this mean?" action (chat-patient.html / chat-doctor.html)

    GET  /health
         -> simple liveness check

Run with:
    pip install fastapi uvicorn livekit-api
    export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    uvicorn server:app --reload --port 8080

Then open frontend/patient.html?room=consult-1 and frontend/doctor.html?room=consult-1
(served via any static file server, or file:// during local testing) pointed
at this backend's origin.
"""

import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel

from config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL, SUPPORTED_PATIENT_LANGUAGES
from gemma_brain import GemmaBrain
from healthcare_tools import book_appointment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="MedVoice Agent Backend")
brain = GemmaBrain()

# Loosen for local hackathon dev; restrict to your actual frontend origin
# before deploying anywhere real, since PHI is in play.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def public_config():
    """Return only browser-safe connection settings (never API credentials)."""
    return {"livekit_url": LIVEKIT_URL}


@app.get("/token")
def get_token(
    room: str = Query(..., description="LiveKit room name, e.g. consult-1"),
    identity: str = Query(..., description="Unique participant identity"),
    role: str = Query("patient", description="'patient' or 'doctor' — controls display name"),
    language: Optional[str] = Query(
        None, description="Patient's chosen language code, e.g. 'ta-IN' or 'te-IN'"
    ),
):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(500, "LIVEKIT_API_KEY / LIVEKIT_API_SECRET not configured on server")

    # Convention agent.py relies on: identities are prefixed by role so the
    # agent knows which track is the patient's mic.
    prefixed_identity = identity if identity.startswith(role) else f"{role}-{identity}"

    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    # Embed the patient's chosen language in their participant metadata so
    # agent.py can honor it per-session instead of assuming one language for
    # every patient (previously hardcoded to Tamil regardless of UI choice).
    metadata = {}
    if role == "patient" and language in SUPPORTED_PATIENT_LANGUAGES.values():
        metadata["language"] = language

    token_builder = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(prefixed_identity)
        .with_name(role.capitalize())
        .with_grants(grants)
    )
    if metadata:
        token_builder = token_builder.with_metadata(json.dumps(metadata))
    token = token_builder.to_jwt()
    return {"token": token, "identity": prefixed_identity, "room": room}


class BookAppointmentRequest(BaseModel):
    doctor_id: str
    patient_name: str
    reason: str


@app.post("/book-appointment")
def api_book_appointment(req: BookAppointmentRequest):
    try:
        record = book_appointment(
            doctor_id=req.doctor_id, patient_name=req.patient_name, reason=req.reason
        )
        return record
    except Exception as e:  # noqa: BLE001
        logger.exception("Booking failed")
        raise HTTPException(500, str(e))


class ExplainTermsRequest(BaseModel):
    text: str
    target_language: str


@app.post("/explain-terms")
async def api_explain_terms(req: ExplainTermsRequest):
    try:
        terms = await brain.explain_terms(req.text, req.target_language)
        return {"terms": terms}
    except Exception as e:  # noqa: BLE001
        logger.exception("Explain-terms failed")
        raise HTTPException(500, str(e))
