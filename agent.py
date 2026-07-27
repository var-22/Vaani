"""LiveKit worker for MedVoice patient<->doctor voice translation."""

import asyncio
import json
import logging
from collections import deque

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, vad
from livekit.plugins import silero

from config import DOCTOR_LANGUAGE_CODE, SUPPORTED_PATIENT_LANGUAGES
from gemma_brain import GemmaBrain
from sarvam_client import SarvamClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medvoice-agent")

brain = GemmaBrain()
sarvam = SarvamClient()

# Fire-and-forget tasks (call-status pings, background clinical analysis,
# streamed TTS playback) aren't awaited by their caller. asyncio only holds
# tasks alive via the loop's own scheduling machinery -- if nothing else
# references a task, it's possible (and a well-known asyncio pitfall,
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)
# for it to be garbage-collected mid-flight, especially longer-running ones
# like _run_clinical_analysis's Gemma round-trip. This set just holds a
# strong reference until each task finishes.
_background_tasks: set = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

# Sample rate for the published TTS track. The streaming TTS path requests
# this exact rate from Sarvam (speech_sample_rate=TTS_SAMPLE_RATE), so audio
# arrives already matching the track -- no resampling needed.
TTS_SAMPLE_RATE = 24000
TTS_NUM_CHANNELS = 1

# Sample rate fed to both the local VAD and Sarvam's streaming STT. LiveKit's
# rtc.AudioStream resamples incoming mic audio to this rate for us, and
# Sarvam's streaming STT only accepts 8000 or 16000 Hz (unlike its batch
# endpoint), so this has to be one of those two.
STT_SAMPLE_RATE = 16000

# How long the VAD waits after speech stops before firing END_OF_SPEECH.
# Tried lowering this below the livekit-plugins-silero default (0.55s) to
# shave latency, but a real end-to-end integration test showed even 0.4s
# fragments multi-clause sentences -- "I have fever... headache... chest
# pain" got split into separate utterances at clause boundaries, silently
# losing everything after the first fragment (including, in that test, the
# clause that should have triggered the emergency alert). The actual
# multi-second delay this app had was the batch STT/TTS round-trips (now
# fixed via streaming), not this -- so it's left at the proven default
# rather than trading sentence-splitting risk for a few hundred ms.
VAD_MIN_SILENCE_DURATION = 0.55

# How many pre-speech frames to keep buffered so streaming STT still gets a
# clean lead-in even though it only starts receiving audio once VAD fires
# START_OF_SPEECH (which needs a short run of confirmed speech first, so the
# very first ~100-200ms would otherwise be clipped).
STT_PREROLL_FRAMES = 15

# VAD occasionally fires a spurious START/END_OF_SPEECH pair for a breath,
# click, or brief noise -- too short to be real speech. Skip streaming STT
# entirely for anything shorter than this instead of sending near-empty
# audio over the network.
MIN_UTTERANCE_SECONDS = 0.3

DEFAULT_PATIENT_LANGUAGE_CODE = SUPPORTED_PATIENT_LANGUAGES["tamil"]


async def _publish_call_status(
    ctx: JobContext, state: str, direction: str | None = None, message: str | None = None
):
    """Broadcasts the call's turn-taking state (idle/listening/processing/
    speaking/error) to the room so the call-interface UI can show real
    pipeline stages instead of guessing from audio levels alone — there's no
    audio to detect during "processing", so this is the only way the
    frontend can know that stage is happening at all."""
    payload = {"type": "call_status", "state": state}
    if direction:
        payload["direction"] = direction
    if message:
        payload["message"] = message
    try:
        await ctx.room.local_participant.publish_data(
            payload=json.dumps(payload).encode("utf-8"),
            reliable=True,
            topic="call-status",
        )
    except Exception:
        logger.exception("Failed to publish call-status %r", payload)


def _resolve_patient_language(participant: rtc.RemoteParticipant) -> str:
    """Patients pick their language in the UI (patient.html); server.py embeds
    it as `{"language": "<code>"}` in the participant's LiveKit token metadata
    so the agent can honor it per-session instead of assuming one language
    for every patient."""
    try:
        metadata = json.loads(participant.metadata or "{}")
    except json.JSONDecodeError:
        metadata = {}

    language_code = metadata.get("language")
    if language_code in SUPPORTED_PATIENT_LANGUAGES.values():
        return language_code

    logger.warning(
        "No valid language in participant %r metadata (%r); defaulting to %s",
        participant.identity, participant.metadata, DEFAULT_PATIENT_LANGUAGE_CODE,
    )
    return DEFAULT_PATIENT_LANGUAGE_CODE


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Agent joined room: %s", ctx.room.name)

    # Create one reusable audio source/track for spoken (TTS) output in
    # *either* direction (patient -> doctor English, or doctor -> patient
    # reply). Publishing it here means it exists as soon as the agent joins.
    # NOTE: both browsers subscribe to and auto-play any track published in
    # the room (see patient.html / doctor.html), so with a single shared
    # track both parties hear both directions' synthesized speech. That's an
    # accepted tradeoff for this one-patient/one-doctor demo rather than
    # building per-listener track subscription filtering.
    tts_source = rtc.AudioSource(
        sample_rate=TTS_SAMPLE_RATE, num_channels=TTS_NUM_CHANNELS
    )
    tts_track = rtc.LocalAudioTrack.create_audio_track("medvoice-tts", tts_source)
    await ctx.room.local_participant.publish_track(
        tts_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("Published medvoice-tts audio track")

    # Both directions share this ONE track (see the note above), and
    # speak_into_track() is always fire-and-forget (asyncio.create_task), so
    # without this lock two utterances close together — very possible now
    # that the kiosk lets patient/doctor alternate rapidly on one device —
    # would both call tts_source.capture_frame() concurrently and interleave
    # their audio into a garbled mix. The lock serializes actual playback
    # (not synthesis) so a queued reply waits its turn instead of overlapping.
    tts_lock = asyncio.Lock()

    # The doctor picks their spoken-reply target language from a dropdown
    # that can change mid-call, so it travels over the data channel rather
    # than being fixed once at token-issue time (unlike the two-device
    # patient's language, which doesn't change during a call).
    reply_language_state = {"code": DEFAULT_PATIENT_LANGUAGE_CODE}

    # Kiosk-mode (single shared device, patient.html) state: there's only
    # one LiveKit participant alternating roles, so — unlike the two-device
    # doctor-*/patient-* identities — we can't tell direction from identity
    # at all. The page tells us explicitly which card was tapped, and both
    # languages are independently switchable over the data channel since a
    # shared device has no per-role token to embed a language into.
    active_role_state = {"role": "patient"}
    patient_language_state = {"code": DEFAULT_PATIENT_LANGUAGE_CODE}

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        if data_packet.topic == "doctor-reply-language":
            try:
                payload = json.loads(data_packet.data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Malformed doctor-reply-language payload: %r", data_packet.data)
                return
            language_code = payload.get("language")
            if language_code in SUPPORTED_PATIENT_LANGUAGES.values():
                reply_language_state["code"] = language_code
                logger.info("Doctor language set to %s", language_code)
            else:
                logger.warning("Ignoring unsupported doctor-reply-language: %r", language_code)
            return

        if data_packet.topic == "patient-language":
            try:
                payload = json.loads(data_packet.data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Malformed patient-language payload: %r", data_packet.data)
                return
            language_code = payload.get("language")
            if language_code in SUPPORTED_PATIENT_LANGUAGES.values():
                patient_language_state["code"] = language_code
                logger.info("Patient language set to %s", language_code)
            else:
                logger.warning("Ignoring unsupported patient-language: %r", language_code)
            return

        if data_packet.topic == "active-role":
            try:
                payload = json.loads(data_packet.data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Malformed active-role payload: %r", data_packet.data)
                return
            role = payload.get("role")
            if role in ("patient", "doctor"):
                active_role_state["role"] = role
                logger.info("Kiosk active role set to %s", role)
            else:
                logger.warning("Ignoring unsupported active-role: %r", role)
            return

        if data_packet.topic == "chat-message":
            # Typed chat messages (chat-patient.html / chat-doctor.html) skip
            # STT and go straight into the same translation pipeline as
            # voice, routed by the sender's identity prefix exactly like
            # on_track_subscribed does below.
            participant = data_packet.participant
            if participant is None:
                return
            try:
                payload = json.loads(data_packet.data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Malformed chat-message payload: %r", data_packet.data)
                return
            text = (payload.get("text") or "").strip()
            if not text:
                return
            message_id = payload.get("message_id")

            if participant.identity.startswith("doctor"):
                _spawn(
                    _translate_and_publish(
                        text, ctx, direction="doctor_to_patient",
                        stt_language_code=DOCTOR_LANGUAGE_CODE,
                        get_reply_language_code=lambda: reply_language_state["code"],
                        tts_source=tts_source, tts_lock=tts_lock,
                        message_id=message_id, speak=False,
                    )
                )
            else:
                patient_language_code = _resolve_patient_language(participant)
                _spawn(
                    _translate_and_publish(
                        text, ctx, direction="patient_to_doctor",
                        stt_language_code=patient_language_code,
                        get_reply_language_code=lambda: DOCTOR_LANGUAGE_CODE,
                        tts_source=tts_source, tts_lock=tts_lock,
                        message_id=message_id, speak=False,
                    )
                )

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info(
            "Track subscribed from participant identity=%r kind=%s",
            participant.identity,
            track.kind,
        )
        # server.py prefixes identities by role (e.g. "doctor-d123",
        # "patient-p456", "kiosk-k789"). In `console` mode there's only ever
        # one other participant and its identity isn't guaranteed to carry a
        # prefix, so anything not explicitly identified as the doctor or the
        # kiosk is treated as patient audio, same as before.
        if participant.identity.startswith("doctor"):
            _spawn(
                handle_utterance_stream(
                    track,
                    ctx,
                    get_turn_state=lambda: (
                        "doctor_to_patient",
                        DOCTOR_LANGUAGE_CODE,
                        reply_language_state["code"],
                    ),
                    tts_source=tts_source,
                    tts_lock=tts_lock,
                )
            )
        elif participant.identity.startswith("kiosk"):
            # Single shared-device mode (patient.html): there's only one
            # participant/track here, subscribed exactly ONCE for the whole
            # call, so direction can't be decided at subscription time like
            # the two-device paths above — tapping a card later wouldn't do
            # anything. Instead get_turn_state() is called fresh for EVERY
            # utterance, re-reading active_role_state at that moment, so
            # whichever card was tapped most recently determines direction.
            def _kiosk_turn_state():
                if active_role_state["role"] == "doctor":
                    return (
                        "doctor_to_patient",
                        reply_language_state["code"],
                        patient_language_state["code"],
                    )
                return (
                    "patient_to_doctor",
                    patient_language_state["code"],
                    DOCTOR_LANGUAGE_CODE,
                )

            _spawn(
                handle_utterance_stream(
                    track,
                    ctx,
                    get_turn_state=_kiosk_turn_state,
                    tts_source=tts_source,
                    tts_lock=tts_lock,
                )
            )
        else:
            patient_language_code = _resolve_patient_language(participant)
            _spawn(
                handle_utterance_stream(
                    track,
                    ctx,
                    get_turn_state=lambda: (
                        "patient_to_doctor",
                        patient_language_code,
                        DOCTOR_LANGUAGE_CODE,
                    ),
                    tts_source=tts_source,
                    tts_lock=tts_lock,
                )
            )


async def handle_utterance_stream(
    track: rtc.Track,
    ctx: JobContext,
    get_turn_state,
    tts_source: rtc.AudioSource,
    tts_lock: asyncio.Lock,
):
    """Feed LiveKit audio frames into VAD AND, once speech is detected,
    concurrently into Sarvam's streaming STT — so most of an utterance is
    already transcribed by the time the person stops talking, instead of
    only starting STT after we already have the whole recording. Shared by
    both the patient->doctor and doctor->patient directions.

    `get_turn_state()` returns (direction, stt_language_code,
    reply_language_code) and is called fresh for every utterance rather than
    once when the track is subscribed. The kiosk's single shared mic track is
    only ever subscribed once per call, so if direction were captured just
    once here, tapping the other person's card mid-call would have no effect
    on already-running processing — every utterance would keep being
    translated in whatever direction was active at call start. Two-device
    (doctor.html/patient.html-style) callers just return the same fixed
    tuple every time, so this is a no-op for them."""
    audio_stream = rtc.AudioStream(track, sample_rate=STT_SAMPLE_RATE, num_channels=1)
    vad_stream = silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE_DURATION).stream()

    # Rolling lead-in buffer, always kept topped up regardless of speech
    # state. START_OF_SPEECH only fires after VAD has already confirmed a
    # short run of real speech, so without this the first ~100-200ms of
    # every utterance would be missing from what we send to streaming STT.
    preroll: deque = deque(maxlen=STT_PREROLL_FRAMES)
    stt_queue: asyncio.Queue | None = None

    async def forward_audio():
        nonlocal stt_queue
        async for frame_event in audio_stream:
            raw = bytes(frame_event.frame.data)
            vad_stream.push_frame(frame_event.frame)
            if stt_queue is not None:
                stt_queue.put_nowait(raw)
            else:
                preroll.append(raw)
        vad_stream.end_input()

    async def _drain_queue(queue: asyncio.Queue):
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    forward_task = asyncio.create_task(forward_audio())
    direction = None
    stt_task: asyncio.Task | None = None
    stt_language_code = None
    try:
        async for event in vad_stream:
            if event.type == vad.VADEventType.START_OF_SPEECH:
                direction, stt_language_code, _ = get_turn_state()
                _spawn(_publish_call_status(ctx, "listening", direction))
                stt_queue = asyncio.Queue()
                for chunk in preroll:
                    stt_queue.put_nowait(chunk)
                # Open the Sarvam STT connection and start draining/sending
                # from the queue RIGHT NOW, concurrently with the still-
                # ongoing speech capture -- this is what actually gets the
                # "already transcribed by the time speech ends" latency
                # benefit. Waiting until END_OF_SPEECH to do this (the
                # earlier version of this code) meant the whole utterance
                # sat buffered and got burst-sent all at once, which is no
                # faster than the old batch endpoint and was also observed
                # to trip Sarvam into returning a truncated transcript.
                stt_task = asyncio.create_task(
                    sarvam.stream_speech_to_text(
                        _drain_queue(stt_queue), language_code=stt_language_code,
                        sample_rate=STT_SAMPLE_RATE,
                    )
                )
            elif event.type == vad.VADEventType.END_OF_SPEECH and event.frames:
                direction, _, reply_language_code = get_turn_state()
                logger.info(
                    "Detected %.1f seconds of speech (direction=%s)",
                    event.speech_duration, direction,
                )
                queue, stt_queue = stt_queue, None
                task, stt_task = stt_task, None
                if queue is None or task is None:
                    logger.warning("END_OF_SPEECH with no active STT stream (direction=%s)", direction)
                    continue
                queue.put_nowait(None)
                if event.speech_duration < MIN_UTTERANCE_SECONDS:
                    # A spurious VAD blip (breath, click, background noise)
                    # rather than real speech -- skip it entirely instead of
                    # waiting on streaming STT for it. Observed empirically:
                    # near-empty streams could take far longer than a real
                    # utterance to resolve (network idle timeouts), which
                    # delayed the person's *next* real utterance since this
                    # loop handles one utterance at a time.
                    task.cancel()
                    _spawn(_publish_call_status(ctx, "idle", direction))
                    continue
                _spawn(_publish_call_status(ctx, "processing", direction))
                try:
                    transcript = await task
                except Exception:
                    logger.exception("Streaming STT failed (direction=%s)", direction)
                    await _publish_call_status(
                        ctx, "error", direction, message="Something went wrong — please try again."
                    )
                    continue
                await process_utterance_streaming(
                    transcript,
                    ctx,
                    direction,
                    stt_language_code,
                    lambda code=reply_language_code: code,
                    tts_source,
                    tts_lock,
                )
    except Exception:
        logger.exception("Audio processing failed (direction=%s)", direction)
    finally:
        if not forward_task.done():
            forward_task.cancel()
        await asyncio.gather(forward_task, return_exceptions=True)
        await vad_stream.aclose()


async def process_utterance_streaming(
    transcript: str,
    ctx: JobContext,
    direction: str,
    stt_language_code: str,
    get_reply_language_code,
    tts_source: rtc.AudioSource,
    tts_lock: asyncio.Lock,
):
    """Hands off an already-transcribed utterance (STT now happens
    concurrently with speech capture -- see handle_utterance_stream above)
    to _translate_and_publish for the shared translate/publish/speak
    logic."""
    try:
        if not transcript.strip():
            logger.info("Sarvam returned an empty transcript (direction=%s)", direction)
            await _publish_call_status(
                ctx, "error", direction, message="Didn't catch that — please repeat."
            )
            return
        await _translate_and_publish(
            transcript, ctx, direction, stt_language_code,
            get_reply_language_code, tts_source, tts_lock, message_id=None, speak=True,
        )
    except Exception:
        logger.exception("Failed to process utterance (direction=%s)", direction)
        await _publish_call_status(
            ctx, "error", direction, message="Something went wrong — please try again."
        )


async def _run_clinical_analysis(ctx: JobContext, transcript: str, message_id: str | None):
    """Runs the (slower) symptom-extraction + emergency-detection +
    patient-detail-extraction tools in the background so it never gates the
    doctor hearing the spoken translation — publishes its own dashboard-
    facing data-channel messages whenever it resolves. See
    GemmaBrain.analyze_clinical()."""
    try:
        analysis = await brain.analyze_clinical(transcript)
        symptoms = analysis["symptoms"]
        emergency = analysis["emergency"]
        patient_details = analysis["patient_details"]

        await ctx.room.local_participant.publish_data(
            payload=json.dumps({
                "type": "symptom_extraction",
                "symptoms": symptoms["symptoms"],
                "duration": symptoms["duration"],
                "severity": symptoms["severity"],
                "original_transcript": transcript,
                "message_id": message_id,
            }).encode("utf-8"),
            reliable=True,
            topic="symptom-extraction",
        )

        # Most utterances won't mention a name/age/gender at all -- only
        # publish when the patient actually said something, so the frontend
        # isn't flooded with empty updates it would just ignore anyway.
        if any(patient_details.get(k) for k in ("name", "age", "gender")):
            await ctx.room.local_participant.publish_data(
                payload=json.dumps({
                    "type": "patient_details",
                    "name": patient_details.get("name", ""),
                    "age": patient_details.get("age", ""),
                    "gender": patient_details.get("gender", ""),
                    "message_id": message_id,
                }).encode("utf-8"),
                reliable=True,
                topic="patient-details",
            )

        if emergency["emergency"]:
            await ctx.room.local_participant.publish_data(
                payload=json.dumps({
                    "type": "emergency_alert",
                    "emergency": True,
                    "level": emergency["level"],
                    "reason": emergency["reason"],
                    "action": emergency["action"],
                    "original_transcript": transcript,
                    "message_id": message_id,
                }).encode("utf-8"),
                reliable=True,
                topic="emergency-alert",
            )
    except Exception:
        # Deliberately no call-status "error" publish here — this is a
        # background dashboard enrichment step, not part of the spoken
        # pipeline the call-status state machine represents, so a failure
        # here shouldn't flash an error over an otherwise-successful
        # translation the doctor already heard.
        logger.exception("Background clinical analysis failed (message_id=%s)", message_id)


async def _translate_and_publish(
    transcript: str,
    ctx: JobContext,
    direction: str,
    stt_language_code: str,
    get_reply_language_code,
    tts_source: rtc.AudioSource,
    tts_lock: asyncio.Lock,
    message_id: str | None,
    speak: bool,
):
    """Streams the translation (patient -> doctor, or doctor -> patient
    reply), publishes it to the relevant data channel, and optionally speaks
    it out loud as it's generated. Shared by both the voice pipeline (after
    STT) and typed chat messages (which skip STT entirely, so `speak`
    defaults to False there — see on_data_received's "chat-message" branch
    for why).

    For patient -> doctor, the (slower) clinical tool analysis —
    extract_symptoms / detect_emergency — runs as a separate background task
    (see _run_clinical_analysis) instead of blocking this translation, since
    the doctor hearing the translated speech is the latency-critical part
    and shouldn't wait on dashboard enrichment."""
    try:
        if direction == "patient_to_doctor":
            logger.info("Patient said (%s): %s", stt_language_code, transcript)
            reply_language_code = DOCTOR_LANGUAGE_CODE
            _spawn(_run_clinical_analysis(ctx, transcript, message_id))
            text_stream = brain.translate_simple_stream(
                transcript, source_language=stt_language_code, target_language=DOCTOR_LANGUAGE_CODE,
            )

            async def publish_transcript_entry(english_translation: str):
                if not english_translation.strip():
                    return
                await ctx.room.local_participant.publish_data(
                    payload=json.dumps({
                        "type": "medical_translation",
                        "patient_language": stt_language_code,
                        "original_transcript": transcript,
                        "english_translation": english_translation,
                        "message_id": message_id,
                    }).encode("utf-8"),
                    reliable=True,
                    topic="doctor-transcript",
                )
                logger.info("Published translation to doctor console")
        else:  # doctor_to_patient
            target_language_code = get_reply_language_code()
            logger.info("Doctor said (%s): %s", stt_language_code, transcript)
            reply_language_code = target_language_code
            text_stream = brain.translate_simple_stream(
                transcript, source_language=stt_language_code, target_language=target_language_code,
            )

            async def publish_transcript_entry(translated_text: str):
                if not translated_text.strip():
                    return
                await ctx.room.local_participant.publish_data(
                    payload=json.dumps({
                        "type": "doctor_reply",
                        "original_transcript": transcript,
                        "translated_text": translated_text,
                        "target_language": target_language_code,
                        "message_id": message_id,
                    }).encode("utf-8"),
                    reliable=True,
                    topic="patient-transcript",
                )
                logger.info("Published doctor reply to patient console")

        if speak:
            # Streams straight into TTS as text is generated — playback can
            # start before the whole sentence has even been translated.
            # publish_transcript_entry fires as soon as text generation is
            # done (independent of how long audio takes to finish playing),
            # so the transcript panel updates promptly either way.
            _spawn(
                speak_into_track_streaming(
                    text_stream, reply_language_code, tts_source, tts_lock, ctx, direction,
                    on_text_complete=lambda text: _spawn(publish_transcript_entry(text)),
                )
            )
        else:
            full_text = "".join([chunk async for chunk in text_stream])
            await publish_transcript_entry(full_text)
    except Exception:
        logger.exception("Failed to process utterance (direction=%s)", direction)
        # Without this, a failure here (e.g. Gemma/Sarvam raising) would be
        # swallowed silently and the call-interface UI would be stuck
        # showing "Translating..." forever, since nothing would ever move
        # it past the "processing" state.
        await _publish_call_status(
            ctx, "error", direction, message="Something went wrong — please try again."
        )


async def speak_into_track_streaming(
    text_chunks,
    language_code: str,
    tts_source: rtc.AudioSource,
    tts_lock: asyncio.Lock,
    ctx: JobContext,
    direction: str,
    on_text_complete=None,
):
    """Streams `text_chunks` (as Gemma generates them) straight into
    Sarvam's streaming TTS and plays audio back as chunks arrive, instead of
    waiting for the full sentence to be both translated AND fully
    synthesized before any sound plays. Requests audio at exactly
    tts_source.sample_rate (see SarvamClient.stream_text_to_speech), so no
    resampling is needed here, unlike the old batch TTS path.

    `on_text_complete(full_text)` fires as soon as text generation finishes
    — typically well before all of its audio has finished streaming/playing
    — so callers can publish a transcript promptly without waiting on
    playback."""
    full_text_parts: list[str] = []

    async def _tap():
        async for chunk in text_chunks:
            if chunk:
                full_text_parts.append(chunk)
                yield chunk
        if on_text_complete:
            on_text_complete("".join(full_text_parts))

    try:
        frame_ms = 20
        samples_per_frame = int(tts_source.sample_rate * frame_ms / 1000)
        bytes_per_frame = samples_per_frame * tts_source.num_channels * 2  # 16-bit PCM
        pending = bytearray()

        # Only one utterance may stream into the shared medvoice-tts track at
        # a time — this blocks here (not "speaking" yet) if another reply is
        # still playing, then proceeds once it's clear.
        async with tts_lock:
            announced = False
            async for pcm_chunk in sarvam.stream_text_to_speech(
                _tap(), target_language_code=language_code, sample_rate=tts_source.sample_rate,
            ):
                if not announced:
                    # Announce "speaking" only once real audio has actually
                    # arrived — this is the moment output audio begins,
                    # which the frontend uses to drive its output waveform.
                    await _publish_call_status(ctx, "speaking", direction)
                    announced = True
                pending.extend(pcm_chunk)
                while len(pending) >= bytes_per_frame:
                    frame_bytes = bytes(pending[:bytes_per_frame])
                    del pending[:bytes_per_frame]
                    await tts_source.capture_frame(
                        rtc.AudioFrame(
                            data=frame_bytes,
                            sample_rate=tts_source.sample_rate,
                            num_channels=tts_source.num_channels,
                            samples_per_channel=samples_per_frame,
                        )
                    )

            if announced:
                logger.info("Finished streaming spoken reply into medvoice-tts track")
                await _publish_call_status(ctx, "idle", direction)
            else:
                logger.warning("Streaming TTS produced no audio for direction=%s", direction)
    except Exception:
        logger.exception("Failed to synthesize/stream TTS audio")
        await _publish_call_status(
            ctx, "error", direction, message="Couldn't play the translated voice."
        )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
