"""
sarvam_client.py
Thin async wrapper around Sarvam AI's Speech-to-Text (Saaras v3) and
Text-to-Speech (Bulbul v3) REST APIs.

Docs: https://docs.sarvam.ai
Auth header: api-subscription-key
"""

import asyncio
import base64
import io
import json
import logging
from typing import AsyncIterator, Optional

import aiohttp
import websockets

from config import (
    SARVAM_API_KEY,
    SARVAM_STT_ENDPOINT,
    SARVAM_STT_MODEL,
    SARVAM_STT_WS_ENDPOINT,
    SARVAM_TTS_ENDPOINT,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_WS_ENDPOINT,
)

logger = logging.getLogger("sarvam_client")


class SarvamSTTResult:
    def __init__(self, transcript: str, detected_language: Optional[str], raw: dict):
        self.transcript = transcript
        self.detected_language = detected_language
        self.raw = raw


class SarvamClient:
    """
    Wraps Sarvam's REST endpoints. Designed to be called per-utterance
    (LiveKit hands us a finalized audio chunk once VAD detects end-of-speech).
    """

    def __init__(self, api_key: str = SARVAM_API_KEY):
        if not api_key:
            logger.warning("SARVAM_API_KEY is not set — requests will fail auth.")
        self.api_key = api_key
        self._headers = {"api-subscription-key": self.api_key}

    async def _connect_ws(self, url: str, timeout: float = 6.0, attempts: int = 3):
        """Opens a websocket with a timeout, retrying on a stall.
        Connection handshakes have been observed to occasionally take longer
        than a single short utterance should reasonably wait for (cold
        DNS/TLS setup, a momentary network blip, or -- observed directly via
        livekit.plugins.silero's "inference is slower than realtime"
        warnings firing with multi-second delays right alongside these
        timeouts -- severe CPU contention on the host stalling the whole
        event loop, not just this connection). Retrying gives a sustained
        stall a real chance to pass before giving up on the utterance."""
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(
                    websockets.connect(url, additional_headers=self._headers), timeout=timeout
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt == attempts:
                    raise
                logger.warning(
                    "Websocket connect to %s timed out (attempt %d/%d), retrying...",
                    url, attempt, attempts,
                )
        raise last_error  # pragma: no cover -- unreachable, satisfies type checkers

    # ------------------------------------------------------------------
    # Speech -> Text  (mode="translate" gives us English directly,
    # mode="transcribe" gives us text in the original language — we use
    # transcribe here because the medical-understanding step needs the
    # patient's own words, and Gemma handles translation with context).
    # ------------------------------------------------------------------
    async def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "utterance.wav",
        language_code: Optional[str] = None,
        mode: str = "transcribe",
    ) -> SarvamSTTResult:
        form = aiohttp.FormData()
        form.add_field(
            "file",
            io.BytesIO(audio_bytes),
            filename=filename,
            content_type="audio/wav",
        )
        form.add_field("model", SARVAM_STT_MODEL)
        form.add_field("mode", mode)
        if language_code:
            form.add_field("language_code", language_code)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                SARVAM_STT_ENDPOINT, data=form, headers=self._headers
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Sarvam STT failed [{resp.status}]: {text}")
                data = await resp.json()

        transcript = data.get("transcript", "")
        detected_lang = data.get("language_code")
        return SarvamSTTResult(transcript=transcript, detected_language=detected_lang, raw=data)

    # ------------------------------------------------------------------
    # Text -> Speech
    # Sarvam returns a JSON body with an `audios` array of base64 WAV
    # strings (not raw binary), so we decode before handing back to
    # the LiveKit audio track publisher.
    # ------------------------------------------------------------------
    async def text_to_speech(
        self,
        text: str,
        target_language_code: str = "en-IN",
        speaker: str = "priya",
        pace: float = 1.0,
    ) -> bytes:
        payload = {
            "text": text,
            "target_language_code": target_language_code,
            "model": SARVAM_TTS_MODEL,
            "speaker": speaker,
            "pace": pace,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                SARVAM_TTS_ENDPOINT,
                json=payload,
                headers={**self._headers, "Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    text_err = await resp.text()
                    raise RuntimeError(f"Sarvam TTS failed [{resp.status}]: {text_err}")
                data = await resp.json()

        audios = data.get("audios", [])
        if not audios:
            raise RuntimeError("Sarvam TTS returned no audio")

        # First chunk only — for long text, iterate and concatenate/stream
        # each entry in `audios` in order instead.
        return base64.b64decode(audios[0])

    # ------------------------------------------------------------------
    # Streaming Speech -> Text
    # Feeds audio to Sarvam's STT websocket as it arrives instead of only
    # starting transcription once the whole utterance is already recorded —
    # most of the audio is transcribed *while the person is still talking*,
    # so the transcript is ready almost immediately once `audio_chunks` ends
    # (matching a `flush`), instead of only starting the network round-trip
    # at that point like the batch speech_to_text() above.
    # ------------------------------------------------------------------
    async def stream_speech_to_text(
        self,
        audio_chunks: AsyncIterator[bytes],
        language_code: str,
        sample_rate: int = 16000,
    ) -> str:
        """`audio_chunks` yields raw 16-bit PCM mono chunks. Returns the
        final transcript once the chunk iterator is exhausted and Sarvam has
        flushed its buffer. Note: Sarvam's streaming STT only supports
        8000/16000 Hz (unlike the batch endpoint) — callers must resample
        first if their source audio is a different rate."""
        url = (
            f"{SARVAM_STT_WS_ENDPOINT}?language-code={language_code}&model={SARVAM_STT_MODEL}"
            f"&mode=transcribe&sample_rate={sample_rate}&input_audio_codec=pcm_s16le"
        )
        transcript = ""

        ws = await self._connect_ws(url)
        try:
            async def sender():
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    await ws.send(json.dumps({
                        "audio": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "sample_rate": str(sample_rate),
                            "encoding": "audio/wav",
                        }
                    }))
                await ws.send(json.dumps({"type": "flush"}))

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.ensure_future(ws.recv())
            # Sarvam doesn't reliably send an explicit "end of stream" signal
            # after flush (observed empirically), so a GRACE_PERIOD of idle
            # time after sending finishes is treated as "no more results
            # coming". This is a real wall-clock deadline computed fresh the
            # moment sending completes (and refreshed on every new message),
            # not a fixed polling cadence -- an earlier version picked a
            # timeout at fixed intervals, which could let send_task finish
            # just before a check and cut the grace period down to
            # milliseconds instead of a full window (observed: an utterance
            # whose real transcript arrived a few hundred ms after flush got
            # discarded because the check landed right as sending finished).
            GRACE_PERIOD = 2.0
            grace_deadline: float | None = None
            try:
                while True:
                    wait_set = {recv_task}
                    if not send_task.done():
                        wait_set.add(send_task)
                        timeout = None
                    else:
                        if grace_deadline is None:
                            grace_deadline = asyncio.get_event_loop().time() + GRACE_PERIOD
                        timeout = max(0.0, grace_deadline - asyncio.get_event_loop().time())

                    done, _ = await asyncio.wait(
                        wait_set, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                    )

                    if recv_task in done:
                        msg = recv_task.result()
                        data = json.loads(msg)
                        if data.get("type") == "data":
                            new_transcript = data.get("data", {}).get("transcript")
                            if new_transcript:
                                transcript = new_transcript
                            if send_task.done():
                                # Every observed run of Sarvam's transcribe
                                # mode sends exactly one "data" message once
                                # sending is complete -- no reason to wait
                                # out a further grace period once we have
                                # it, that's pure tail latency for nothing.
                                break
                            # Still sending -- more audio (and possibly an
                            # interim result) may still be coming.
                            grace_deadline = None
                        elif data.get("type") == "error":
                            raise RuntimeError(f"Sarvam streaming STT error: {data}")
                        recv_task = asyncio.ensure_future(ws.recv())
                        continue

                    # recv_task didn't complete: either send_task just
                    # finished (loop again to start a fresh grace window) or
                    # the grace period genuinely ran out with nothing new.
                    if send_task.done() and (
                        timeout is not None
                        and asyncio.get_event_loop().time() >= grace_deadline
                    ):
                        break
            finally:
                if not recv_task.done():
                    recv_task.cancel()
                    await asyncio.gather(recv_task, return_exceptions=True)
                if not send_task.done():
                    send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
        finally:
            await ws.close()

        return transcript

    # ------------------------------------------------------------------
    # Streaming Text -> Speech
    # Feeds `text_chunks` to Sarvam's TTS websocket as they're produced (by
    # Gemma's own streaming generation) and yields raw PCM audio chunks as
    # they're synthesized, so playback can begin before the whole reply has
    # even finished generating — unlike the batch text_to_speech() above,
    # which waits for one complete WAV response.
    # ------------------------------------------------------------------
    async def stream_text_to_speech(
        self,
        text_chunks: AsyncIterator[str],
        target_language_code: str,
        speaker: str = "priya",
        sample_rate: int = 24000,
        pace: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Yields raw 16-bit PCM mono chunks at `sample_rate` (requesting
        `output_audio_codec: "linear16"` so no WAV/MP3 decoding is needed —
        callers can feed the bytes straight into a fixed-rate audio track)."""
        url = f"{SARVAM_TTS_WS_ENDPOINT}?model={SARVAM_TTS_MODEL}"

        ws = await self._connect_ws(url)
        try:
            await ws.send(json.dumps({
                "type": "config",
                "data": {
                    "speaker": speaker,
                    "target_language_code": target_language_code,
                    "pace": pace,
                    "output_audio_codec": "linear16",
                    "speech_sample_rate": sample_rate,
                },
            }))

            async def sender():
                # Gemma's SSE stream occasionally isolates a trailing
                # fragment (e.g. a lone ".") into its own chunk. Sending that
                # straight to Sarvam as a standalone "text" message can 422
                # ("Text must contain at least one character from the
                # allowed languages") since punctuation/whitespace alone
                # isn't valid speech text -- so small fragments are merged
                # here before being forwarded, instead of relayed 1:1.
                MIN_CHUNK_CHARS = 20
                pending = ""
                async for chunk in text_chunks:
                    if not chunk:
                        continue
                    pending += chunk
                    if len(pending) >= MIN_CHUNK_CHARS:
                        await ws.send(json.dumps({"type": "text", "data": {"text": pending}}))
                        pending = ""
                if pending and any(c.isalnum() for c in pending):
                    await ws.send(json.dumps({"type": "text", "data": {"text": pending}}))
                await ws.send(json.dumps({"type": "flush"}))

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.ensure_future(ws.recv())
            # See stream_speech_to_text() for why this is a real wall-clock
            # grace deadline (started fresh once sending finishes, and
            # refreshed on every new audio chunk) rather than a fixed
            # polling cadence -- a fixed-interval check can land right as
            # send_task finishes and cut the grace period down to
            # milliseconds. Unlike STT (one final message), TTS legitimately
            # keeps sending many audio chunks after flush, so — unlike
            # stream_speech_to_text() — this does NOT stop on the first
            # chunk once sending is done, only once the grace period
            # elapses with no further chunk.
            GRACE_PERIOD = 1.5
            grace_deadline: float | None = None
            try:
                while True:
                    wait_set = {recv_task}
                    if not send_task.done():
                        wait_set.add(send_task)
                        timeout = None
                    else:
                        if grace_deadline is None:
                            grace_deadline = asyncio.get_event_loop().time() + GRACE_PERIOD
                        timeout = max(0.0, grace_deadline - asyncio.get_event_loop().time())

                    done, _ = await asyncio.wait(
                        wait_set, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                    )

                    if recv_task in done:
                        msg = recv_task.result()
                        data = json.loads(msg)
                        mtype = data.get("type")
                        if mtype == "audio":
                            audio_b64 = data.get("data", {}).get("audio")
                            if audio_b64:
                                yield base64.b64decode(audio_b64)
                            grace_deadline = None
                        elif mtype == "error":
                            raise RuntimeError(f"Sarvam streaming TTS error: {data}")
                        recv_task = asyncio.ensure_future(ws.recv())
                        continue

                    if send_task.done() and (
                        timeout is not None
                        and asyncio.get_event_loop().time() >= grace_deadline
                    ):
                        break
            finally:
                if not recv_task.done():
                    recv_task.cancel()
                    await asyncio.gather(recv_task, return_exceptions=True)
                if not send_task.done():
                    send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
        finally:
            await ws.close()
