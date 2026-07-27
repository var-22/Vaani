# MedVoice — Real-Time AI Voice Interpreter for Doctor–Patient Consultations

**Built for: Build with Gemma 4 – AI Durg (Kaggle Hackathon)**

MedVoice is a real-time voice translation system that sits inside a doctor–patient
video/voice call and acts as a live medical interpreter: the patient speaks in
their own language, the doctor hears it in English (spoken *and* transcribed)
within roughly 1–2 seconds, and Gemma 4 simultaneously extracts structured
clinical signal — symptoms, duration, severity, and emergency risk — so nothing
important gets lost to a language barrier.

 

---

## 1. Problem Statement

Language barriers between doctors and patients create major challenges in
healthcare communication. When doctors provide medical services to people in
rural areas, urban communities, or different states, they may face difficulty
understanding patients who speak different regional languages.

Without proper translation, patients may not be able to clearly explain their
symptoms, pain, or medical history, and doctors may misunderstand the
patient's condition or assume incorrect information. This can lead to wrong
interpretation of symptoms, incorrect medical guidance, delays in treatment,
and reduced quality of healthcare.

To overcome this challenge, an AI-powered real-time voice translation system
is proposed. The system acts as a virtual medical interpreter that translates
doctor-patient conversations instantly between different languages while
maintaining medical accuracy. It helps doctors and patients communicate
effectively, improves healthcare accessibility, and enables better medical
support regardless of language differences.

## 2. Our Solution

MedVoice is a single shared-screen "consultation kiosk" (one tablet/laptop in
the room, or two browser tabs) where:

- The **patient** speaks naturally in **Tamil, Telugu, or English**. The mic
  listens continuously — there is no "press to talk" button, no round-trip
  delay from tapping — voice activity detection (VAD) decides where each
  sentence starts and ends.
- The **doctor** hears an English translation spoken back **and** sees it as
  live text, within roughly 1–2 seconds of the patient finishing a sentence.
- The doctor can reply in **English, Tamil, or Telugu**; the same pipeline
  translates and speaks it back to the patient in their chosen language.
- In the background, **Gemma 4** independently extracts symptoms (with
  duration and severity) and screens for medical emergencies (chest pain,
  breathing difficulty, severe bleeding, stroke symptoms, severe allergic
  reaction, etc.), surfacing both directly in the doctor's view — without
  ever slowing down the actual conversation.

## 3. Why Gemma 4

**Model used**: [`gemma-4-26b-a4b-it`](https://aistudio.google.com) — the
instruction-tuned variant, served through **Google AI Studio's**
Gemini-compatible `generateContent` / `streamGenerateContent` API
(configurable via the `GOOGLE_MODEL` env var; `gemma-4-31b-it` is a drop-in
alternative for higher quality at the cost of latency).

Gemma 4 is the **reasoning core** of MedVoice — everything downstream of raw
speech-to-text goes through it. Concretely, Gemma 4 is used for:

1. **Real-time streaming translation** — the doctor⇄patient reply is
   generated token-by-token via `streamGenerateContent` and piped directly
   into text-to-speech as it's produced, so spoken playback starts before
   Gemma has even finished the sentence. This is the single biggest lever
   behind the "1–2 second" response feel.
2. **Structured medical function-calling** — Gemma is given three tools,
   `extract_symptoms`, `detect_emergency`, and `extract_patient_details`, and
   reasons out their arguments directly from the patient's (translated)
   statement:
   ```json
   {"tool_calls": [
     {"name": "extract_symptoms", "arguments": {"symptoms": ["fever", "headache"], "duration": "3 days", "severity": "medium"}},
     {"name": "detect_emergency", "arguments": {"emergency": false, "level": "LOW", "reason": "No red-flag symptoms mentioned."}},
     {"name": "extract_patient_details", "arguments": {"name": "Varsha", "age": "", "gender": ""}}
   ]}
   ```
   This isn't keyword matching — Gemma is doing the actual clinical language
   understanding (what counts as a symptom, how severe it sounds, whether
   it's an emergency, whether the patient stated their name/age) and the
   Python side just validates/normalizes the structured output and applies
   one deterministic keyword safety-net on top of the emergency call (so a
   missed or hallucinated tool call can never silently suppress an emergency
   flag).
3. **Provider-agnostic tool-calling protocol** — rather than depending on one
   vendor's native function-calling schema, `gemma_brain.py` implements its
   own strict JSON-object protocol in the system prompt
   (`{"tool_call": ...}` / `{"tool_calls": [...]}` / `{"final": ...}`), so
   the exact same code path works whether Gemma 4 is served through Google
   AI Studio (the default here) or run locally via Ollama
   (`GEMMA_BACKEND=ollama`) — no local GPU required to try the project, but
   local inference is a drop-in swap.
4. **A full clinical-triage mode** (`GemmaBrain.process_utterance`) — a
   multi-hop tool-calling loop that also routes the patient to a department
   (`route_doctor`) and can book an appointment (`book_appointment`),
   available as a deeper analysis path beyond the live-call fast path.

### Why the hosted API instead of running the open-weights model directly

`GEMMA_BACKEND=ollama` is fully implemented and works (see `gemma_brain.py`'s
`_call_ollama`) — this isn't a case of the open-weights model being
unavailable to us. Google AI Studio's API is the *default* deployment choice
for this project specifically because of what a real-time voice pipeline
needs:

- **Token-level streaming is load-bearing, not a nice-to-have.** The whole
  "spoken reply starts before the sentence finishes generating" latency
  design depends on `streamGenerateContent`'s SSE token stream. The local
  Ollama path in this codebase currently calls Gemma non-streamed
  (`stream: false`), so it works as a correctness fallback but doesn't get
  the streaming latency benefit — getting equivalent low-latency streaming
  out of a self-hosted model is real additional engineering, not a config
  flag.
- **Consistent inference speed regardless of reviewer hardware.** A 26B-parameter
  model run locally is only as fast as the GPU it's on; a hackathon judge
  running this on a laptop with no GPU (or a weaker one than ours) would see
  wildly different — likely unusable — latency for a "real-time" demo. The
  hosted API gives the same response time for everyone who runs this
  repository, which matters for a project whose entire pitch is speed.
- **Reproducibility for judges without a multi-GB download or GPU
  requirement.** Setup is `pip install -r requirements.txt` plus a free
  Google AI Studio API key — no pulling model weights, no VRAM requirements,
  no quantization choices that could silently change output quality. This
  also matches the competition's own guidance that commercial auxiliary APIs
  are fine as long as they're reasonably and publicly accessible, which
  Google AI Studio's free tier is.
- **No quantization-quality variance for a medical-accuracy-sensitive task.**
  Locally-run open-weight models are typically served quantized (4-bit/8-bit)
  to fit consumer hardware, which can measurably affect translation fidelity
  and tool-call JSON reliability. Since MedVoice's entire value proposition
  is *not* losing or garbling clinical meaning in translation, we default to
  the un-quantized, provider-hosted model rather than trade accuracy for
  local-inference convenience.
- **It's a default, not a lock-in.** Because of the provider-agnostic
  tool-calling protocol (point 3 above), switching to fully local, offline
  inference is a one-line env var change (`GEMMA_BACKEND=ollama`) for anyone
  who needs on-prem/offline deployment (e.g. a rural clinic with no reliable
  internet) — see Section 9's Vision for how this fits the production
  scaling story.

## 4. How It Works

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/dd9005ff-c07f-4adc-88c4-07557a1c5a18" />


```

## Everything is **streamed**, not batched: speech-to-text starts transcribing
while the person is still talking (not after they finish), Gemma's
translation is generated and spoken incrementally, and the clinical
tool-calling analysis (which is slower — a full reasoning round-trip) runs
**concurrently in the background** instead of gating the doctor hearing the
translated reply.

## 5. Key Features

- 🎙️ **Hands-free live listening** — no "tap to talk" button; the mic is
  always on, VAD segments utterances automatically.
- 🌐 **3 languages, either side** — Tamil / Telugu / English, independently
  selectable for patient and doctor, switchable mid-call.
- ⚡ **~1–2 second round trip** — streaming STT → streaming Gemma
  translation → streaming TTS, all pipelined rather than run as three
  sequential blocking calls.
- 📝 **Live bilingual transcript** — every utterance appears as text in both
  languages, alongside the spoken translation.
- 🩺 **AI symptom extraction** — Gemma 4 tool-calling pulls out symptoms,
  duration, and a severity read from natural conversation, shown live as
  chips on the doctor's side.
- 🚨 **AI emergency detection** — a second Gemma 4 tool call screens for
  chest pain, breathing difficulty, severe bleeding, stroke symptoms, and
  severe allergic reaction, backed by a deterministic keyword fail-safe;
  triggers a red alert banner with the reasoning shown.
- 🔒 **No raw cross-language audio leak** — only the synthesized, translated
  track is played back; the room never exposes one party's raw mic audio to
  the other untranslated.

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Real-time audio transport | [LiveKit](https://livekit.io) (WebRTC SFU + Agents framework) |
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) via `livekit-plugins-silero` |
| Speech-to-Text | [Sarvam AI](https://www.sarvam.ai) Saaras v3 (streaming WebSocket + batch REST) |
| Text-to-Speech | Sarvam AI Bulbul v3 (streaming WebSocket, 24kHz LINEAR16) |
| Reasoning / translation / function-calling | **Gemma 4** via Google AI Studio (`generateContent` / `streamGenerateContent`), or locally via Ollama |
| Backend | FastAPI (token issuance, appointment booking, term-explainer endpoints) |
| Frontend | Vanilla JS + `livekit-client`, no framework |

## 7. Project Structure

| File | Role |
|---|---|
| `agent.py` | LiveKit Agent worker — the real-time pipeline: VAD → streaming STT → streaming Gemma translation → streaming TTS, plus the background clinical-analysis task and all data-channel signaling (call state, transcripts, symptoms, emergency alerts) |
| `gemma_brain.py` | All Gemma 4 integration: streaming/non-streaming translation, the `extract_symptoms`/`detect_emergency` tool-calling analysis, the full clinical-triage loop, and the medical-terminology explainer |
| `healthcare_tools.py` | The function-calling tools Gemma 4 invokes: `extract_symptoms`, `detect_emergency`, `route_doctor`, `book_appointment` — plus their JSON schemas |
| `sarvam_client.py` | Thin async client for Sarvam's STT/TTS, both batch REST and streaming WebSocket variants |
| `server.py` | FastAPI backend: LiveKit token issuance, `/book-appointment`, `/explain-terms`, `/health` |
| `config.py` | All environment-driven configuration: API endpoints, supported languages, emergency keyword list |
| `frontend/patient.html` | The main consultation UI — single shared-device kiosk, both patient and doctor cards, live transcript, symptom chips, emergency banner |
| `frontend/doctor.html` | Redirects to `patient.html` (the app moved from a two-device model to one shared screen) |
| `frontend/chat-patient.html`, `frontend/chat-doctor.html` | An alternate typed-chat mode (WhatsApp-style), sharing the same backend translation pipeline |
| `frontend/shared.css` | Shared design tokens |

## 8. Setup & Run

### Prerequisites

- Python 3.10+
- A [LiveKit Cloud](https://cloud.livekit.io) project (free tier works)
- A [Sarvam AI](https://dashboard.sarvam.ai) API key
- A [Google AI Studio](https://aistudio.google.com) API key (for Gemma 4)

### 1. Install dependencies

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS/Linux, or Windows cmd.exe (use venv\Scripts\activate.bat instead)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> PowerShell needs the `.ps1` script with an explicit `.\` prefix — the
> bash-style `venv\Scripts\activate` (no extension, no `.\`) will fail with
> `CouldNotAutoLoadModule` in PowerShell, since it tries to interpret that as
> a module name rather than a script in the current directory.

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

SARVAM_API_KEY=...

GEMMA_BACKEND=google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemma-4-26b-a4b-it

# Optional: run Gemma locally instead of via Google AI Studio
# GEMMA_BACKEND=ollama
# OLLAMA_MODEL=gemma4        # `ollama pull gemma4` first
```

### 3. Run the three processes

Each of these is a long-running, blocking process — run them in **three
separate terminal windows/tabs**, all with the venv activated, all starting
from the **project root** (not `frontend/`, which only matters for step 3).
Pasting all three into one terminal one after another won't work: the first
one blocks that terminal until you stop it.

```powershell
# Terminal 1 — the voice-processing agent (Gemma 4 + Sarvam pipeline)
python agent.py dev
```

```powershell
# Terminal 2 — the token/booking backend (must run from the project root,
# since it imports server.py — running it from frontend/ fails with
# "Could not import module 'server'")
uvicorn server:app --reload --port 8080
```

```powershell
# Terminal 3 — serve the frontend
cd frontend
python -m http.server 5500
```

### 4. Open the app

```
http://localhost:5500/patient.html?room=consult-1
```

Use a new, unused `room=` value each time you start a fresh call — LiveKit
only dispatches the agent when a room is newly created.

## 9. Vision: Scaling to 10+ Languages in Production

This hackathon build supports Tamil, Telugu, and English end-to-end. Getting
to 10+ languages in a real deployment is deliberately **not** a rewrite —
the architecture was built so it's a configuration and data problem, not a
code problem:

**Architecture**

- Every language-specific decision in the pipeline already flows through one
  place: `config.SUPPORTED_PATIENT_LANGUAGES`, a plain `{name: BCP-47 code}`
  dict. `agent.py`'s `direction`/`get_turn_state` abstraction treats "source
  language" and "target language" as parameters passed through to Sarvam and
  Gemma, not hardcoded per-language branches — so adding a language is
  adding one dict entry and one `<option>` in the frontend's language
  picker, not N new code paths. Sarvam's Bulbul v3/Saaras v3 already cover
  10+ Indian languages (Hindi, Bengali, Kannada, Malayalam, Marathi,
  Gujarati, Punjabi, Odia, and more) beyond the three wired up here, so the
  STT/TTS leg is ready today.
- The `extract_symptoms`/`detect_emergency` tool-calling contract is
  language-agnostic by design (it operates on the already-translated English
  text), so the clinical-reasoning layer doesn't need to be duplicated per
  language at all — only the translation layer does.

**Data**

- Ship each new language with a small clinician-reviewed evaluation set
  (real or synthetic doctor-patient exchanges with known-correct
  translations and symptom labels) to benchmark Gemma's medical-translation
  accuracy before enabling it live, not after.
- Add a "flag this translation" affordance for doctors, feeding a growing,
  consented feedback dataset per language — used to build few-shot exemplars
  in the prompt for weaker languages, and eventually to fine-tune a
  Gemma checkpoint if a language's zero-shot quality plateaus below target.
- `config.EMERGENCY_KEYWORDS_EN`'s safety-net list must be independently
  built per language with clinical translators, not machine-translated from
  English — a mistranslated emergency keyword is exactly the kind of
  silent failure this fail-safe exists to prevent.

**Deployment**

- LiveKit Agent workers are stateless per-call processes that register with
  a worker pool — scaling to more concurrent consultations (across more
  languages or more clinics) means running more worker replicas (e.g. a
  Kubernetes `HorizontalPodAutoscaler` on active-job count), with no change
  to `agent.py` itself.
- Deploy the LiveKit SFU and agent workers in-region, close to both the
  clinic and Sarvam's infrastructure (this build already runs in LiveKit's
  India South region) to keep the STT/TTS leg fast; at scale, pin Gemma 4
  calls to a regional Vertex AI endpoint rather than a single global Google
  AI Studio endpoint for tighter latency and data-residency control.
- Streaming STT/TTS connections are held open per **utterance**, not per
  call, so infrastructure cost scales with actual speech duration rather
  than total call duration — a more predictable basis for budgeting at
  clinic-network scale.
- For low-connectivity rural clinics, `GEMMA_BACKEND=ollama` is already a
  drop-in local-inference path in this codebase — a fully offline variant
  (local Gemma 4 + a locally-hosted STT/TTS model) could run the same
  `gemma_brain.py`/`agent.py` code paths with no internet dependency for the
  AI reasoning leg, syncing to the cloud only when connectivity is
  available.
- Instrument per-stage latency and per-language accuracy from day one — the
  `call-status` data-channel events `agent.py` already emits
  (listening/processing/speaking/idle/error) are natural tracing points to
  extend, so a regression in one new language surfaces before it reaches a
  real patient.

## 10. Kaggle Competition Compliance

- **License**: This repository is licensed under the [MIT License](LICENSE),
  an OSI-approved open-source license, satisfying the competition's Winner
  License requirement (non-commercial deployment and study of the code is
  unrestricted).
- **External APIs used**: LiveKit Cloud, Sarvam AI, and Google AI Studio
  (Gemma 4) are all publicly accessible developer APIs with free/low-cost
  tiers, in line with the competition's Reasonableness Standard for
  auxiliary tooling. No proprietary or restricted datasets are used.
- **Gemma 4 integration is directly reviewable**: see Section 3 above and
  `gemma_brain.py` for the exact prompts, tool schemas, and streaming call
  sites.
- **Repository & demo links**: _add your public GitHub repository URL and
  live demo link here before submitting your Kaggle Writeup._

## 11. Known Limitations / Roadmap

- No persistent patient identity or end-of-call clinical summary yet (in
  progress) — currently each utterance is processed independently rather
  than aggregated into a single case record for the doctor.
- `healthcare_tools.py`'s doctor directory and appointment booking are
  in-memory stubs, not a real scheduling system.
- No authentication/consent flow — not yet suitable for real patient data
  without adding that layer.
- Only Tamil, Telugu, and English are wired up end-to-end today; Sarvam
  supports additional Indian languages that could be added via
  `config.SUPPORTED_PATIENT_LANGUAGES`.
