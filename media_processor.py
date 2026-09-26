"""Media validation/compression for SAARTHI_SERVER.

Validates, rate-limits, and compresses every image, video, and document
coming from the robot (WebSocket), web, and app channels *before* any of
it reaches the local Qwen3-VL model. This module owns none of the model
loading or inference logic (see ai_services.py) - it is a standalone,
thread-safe utility module.
"""

import base64
import io
import logging
import os
import tempfile
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import List, Union

import cv2
from PIL import Image
from pypdf import PdfReader

from config import (
    DOC_ALLOWED_EXTENSIONS,
    DOC_MAX_EXTRACTED_CHARS,
    DOC_MAX_PAGES,
    DOC_MAX_UPLOAD_MB,
    IMAGE_COMPRESSION_QUALITY,
    IMAGE_MAX_COMPRESSED_KB,
    IMAGE_MAX_UPLOAD_MB,
    IMAGE_TARGET_RESOLUTION,
    VIDEO_FRAME_RESOLUTION,
    VIDEO_MAX_DURATION_SECONDS,
    VIDEO_MAX_FRAMES,
    VIDEO_MAX_PER_SECOND,
    VIDEO_MAX_UPLOAD_MB,
    VIDEO_RATE_LIMIT_SECONDS,
    VIDEO_TARGET_FPS,
)

logger = logging.getLogger(__name__)

# Pillow renamed Image.LANCZOS to Image.Resampling.LANCZOS; support both.
_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

_MIN_JPEG_QUALITY = 30
_QUALITY_STEP = 10

_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class MediaValidationError(ValueError):
    """Raised when an uploaded image, video, or document fails validation
    (bad size, bad/corrupt format, disallowed extension, duration limit,
    etc.). Callers should turn this into a user-facing error response,
    never a 500."""


class VideoRateLimitError(ValueError):
    """Raised when a session sends more videos than VIDEO_MAX_PER_SECOND
    allows within VIDEO_RATE_LIMIT_SECONDS."""


# ---------------------------------------------------------------------------
# Video rate limiting (1 video / second by default)
# ---------------------------------------------------------------------------
# Per-session_id timestamp tracking behind a single lock, mirroring the
# sweep-old-entries pattern used elsewhere in this project (server.py) so
# long-running deployments don't leak memory across thousands of old
# session_ids.

_video_timestamps: "defaultdict[str, deque]" = defaultdict(deque)
_video_lock = threading.Lock()
_last_sweep_monotonic = 0.0
_SWEEP_INTERVAL_SECONDS = 60.0


def _sweep_old_sessions_locked(now: float) -> None:
    """Drop timestamps older than the rate-limit window and forget any
    session_id whose deque becomes empty. Caller must hold _video_lock."""
    global _last_sweep_monotonic
    if now - _last_sweep_monotonic < _SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep_monotonic = now
    cutoff = now - VIDEO_RATE_LIMIT_SECONDS
    empty_sessions = []
    for session_id, timestamps in _video_timestamps.items():
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if not timestamps:
            empty_sessions.append(session_id)
    for session_id in empty_sessions:
        del _video_timestamps[session_id]


def check_video_rate_limit(session_id: str) -> None:
    """Raise VideoRateLimitError if `session_id` has already sent
    VIDEO_MAX_PER_SECOND videos within the last VIDEO_RATE_LIMIT_SECONDS
    seconds; otherwise records this attempt and returns None. Thread-safe -
    safe to call concurrently from multiple requests/sessions."""
    now = time.monotonic()
    with _video_lock:
        _sweep_old_sessions_locked(now)
        timestamps = _video_timestamps[session_id]
        cutoff = now - VIDEO_RATE_LIMIT_SECONDS
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if len(timestamps) >= VIDEO_MAX_PER_SECOND:
            raise VideoRateLimitError(
                f"Only {VIDEO_MAX_PER_SECOND} video allowed per second. Please wait."
            )
        timestamps.append(now)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _decode_image_bytes(image_input: Union[bytes, str, Image.Image]) -> bytes:
    """Normalize raw bytes / a base64 string (with or without a
    data:image/...;base64, prefix) / an already-decoded PIL.Image into raw
    bytes, for size validation before decoding."""
    if isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    if isinstance(image_input, str):
        data = image_input
        if data.startswith("data:"):
            _, _, data = data.partition(",")
        try:
            return base64.b64decode(data, validate=False)
        except Exception as e:
            raise MediaValidationError(f"Invalid base64 image data: {e}") from e
    raise MediaValidationError(f"Unsupported image input type: {type(image_input)!r}")


def process_image(image_input: Union[bytes, str, Image.Image]) -> Image.Image:
    """Validate, resize, and JPEG-compress an image before it reaches the
    model.

    Accepts raw bytes, a base64 string (optionally with a
    data:image/...;base64, prefix), or a PIL.Image. Rejects anything over
    IMAGE_MAX_UPLOAD_MB. Converts to RGB, resizes to fit within
    IMAGE_TARGET_RESOLUTION x IMAGE_TARGET_RESOLUTION (aspect ratio
    preserved), and compresses to JPEG at IMAGE_COMPRESSION_QUALITY,
    stepping quality down (to a floor of 30) if needed to stay within
    IMAGE_MAX_COMPRESSED_KB. Returns a compressed, RGB, in-memory
    PIL.Image (fully loaded, safe to use after this function returns).
    """
    raw_bytes = _decode_image_bytes(image_input)

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > IMAGE_MAX_UPLOAD_MB:
        raise MediaValidationError(
            f"Image exceeds maximum allowed upload size of {IMAGE_MAX_UPLOAD_MB} MB."
        )

    try:
        image = image_input if isinstance(image_input, Image.Image) else Image.open(io.BytesIO(raw_bytes))
        image = image.convert("RGB")
    except Exception as e:
        raise MediaValidationError(f"Could not decode image data: {e}") from e

    image.thumbnail((IMAGE_TARGET_RESOLUTION, IMAGE_TARGET_RESOLUTION), _RESAMPLE_LANCZOS)

    quality = IMAGE_COMPRESSION_QUALITY
    buffer = io.BytesIO()
    while True:
        buffer.seek(0)
        buffer.truncate(0)
        image.save(buffer, format="JPEG", quality=quality)
        size_kb = buffer.tell() / 1024
        if size_kb <= IMAGE_MAX_COMPRESSED_KB or quality <= _MIN_JPEG_QUALITY:
            break
        quality = max(_MIN_JPEG_QUALITY, quality - _QUALITY_STEP)

    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed.load()  # force-read pixel data now; buffer can go out of scope safely
    return compressed.convert("RGB")


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def _decode_video_bytes(video_input: Union[bytes, str]) -> bytes:
    if isinstance(video_input, (bytes, bytearray)):
        return bytes(video_input)
    if isinstance(video_input, str):
        data = video_input
        if data.startswith("data:"):
            _, _, data = data.partition(",")
        try:
            return base64.b64decode(data, validate=False)
        except Exception as e:
            raise MediaValidationError(f"Invalid base64 video data: {e}") from e
    raise MediaValidationError(f"Unsupported video input type: {type(video_input)!r}")


def process_video(video_input: Union[bytes, str], session_id: str) -> List[Image.Image]:
    """Rate-limit, validate, and extract a handful of low-res frames from
    a short video clip.

    Always calls check_video_rate_limit(session_id) FIRST, so this
    function can never be used to bypass the per-second video limit.
    Rejects clips over VIDEO_MAX_UPLOAD_MB or longer than
    VIDEO_MAX_DURATION_SECONDS. Extracts up to VIDEO_MAX_FRAMES frames at
    roughly VIDEO_TARGET_FPS, each resized to VIDEO_FRAME_RESOLUTION x
    VIDEO_FRAME_RESOLUTION and returned as RGB PIL.Image objects.

    Note on the temp file: the spec called for
    NamedTemporaryFile(delete=True), but on this platform cv2.VideoCapture
    needs to reopen the file by path while it's still open, which doesn't
    reliably survive delete=True. This uses delete=False plus an explicit
    os.remove() in the `finally` block instead, which gets the same
    guarantee (the temp file is always removed, even on error) in a way
    that's safe cross-platform.
    """
    check_video_rate_limit(session_id)

    video_bytes = _decode_video_bytes(video_input)

    size_mb = len(video_bytes) / (1024 * 1024)
    if size_mb > VIDEO_MAX_UPLOAD_MB:
        raise MediaValidationError(
            f"Video exceeds maximum allowed upload size of {VIDEO_MAX_UPLOAD_MB} MB."
        )

    tmp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name

        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise MediaValidationError("Could not open video file for reading.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0:
            fps = 30.0  # sane fallback so a duration check is still possible

        duration_seconds = (frame_count / fps) if fps else 0.0
        if duration_seconds > VIDEO_MAX_DURATION_SECONDS:
            raise MediaValidationError(
                f"Video exceeds maximum allowed duration of {VIDEO_MAX_DURATION_SECONDS} seconds."
            )

        frame_interval = max(1, round(fps / VIDEO_TARGET_FPS)) if VIDEO_TARGET_FPS > 0 else max(1, round(fps))

        frames: List[Image.Image] = []
        frame_index = 0
        while len(frames) < VIDEO_MAX_FRAMES:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frame.thumbnail((VIDEO_FRAME_RESOLUTION, VIDEO_FRAME_RESOLUTION), _RESAMPLE_LANCZOS)

                out_buffer = io.BytesIO()
                pil_frame.save(out_buffer, format="JPEG", quality=IMAGE_COMPRESSION_QUALITY)
                out_buffer.seek(0)
                compressed = Image.open(out_buffer)
                compressed.load()
                frames.append(compressed.convert("RGB"))
            frame_index += 1

        return frames
    finally:
        if capture is not None:
            capture.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract plain text from a .docx file's word/document.xml using only
    the standard library (zipfile + ElementTree) - no extra dependency."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            with archive.open("word/document.xml") as doc_xml:
                tree = ET.parse(doc_xml)
    except (KeyError, zipfile.BadZipFile, ET.ParseError) as e:
        raise MediaValidationError(f"Could not read .docx document: {e}") from e

    root = tree.getroot()
    paragraphs = []
    for para in root.iter(f"{_DOCX_NS}p"):
        run_texts = [t.text for t in para.iter(f"{_DOCX_NS}t") if t.text]
        if run_texts:
            paragraphs.append("".join(run_texts))
    return "\n".join(paragraphs)


def process_document(file_bytes: bytes, filename: str) -> str:
    """Validate a document upload (extension + size), extract its text,
    collapse whitespace, and truncate to DOC_MAX_EXTRACTED_CHARS before any
    of it can enter a prompt or RAG context.

    Supports .pdf (via pypdf, capped at DOC_MAX_PAGES pages), .txt/.md
    (UTF-8 decode), and .docx (stdlib zipfile + ElementTree, no extra
    dependency).
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in DOC_ALLOWED_EXTENSIONS:
        raise MediaValidationError(
            f"Unsupported document extension '{ext}'. Allowed: {DOC_ALLOWED_EXTENSIONS}."
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > DOC_MAX_UPLOAD_MB:
        raise MediaValidationError(
            f"Document exceeds maximum allowed upload size of {DOC_MAX_UPLOAD_MB} MB."
        )

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise MediaValidationError(f"Could not read PDF document: {e}") from e
        pages = reader.pages[:DOC_MAX_PAGES]
        text = "\n".join((page.extract_text() or "") for page in pages)
    elif ext in (".txt", ".md"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise MediaValidationError(f"Could not decode text document as UTF-8: {e}") from e
    elif ext == ".docx":
        text = _extract_docx_text(file_bytes)
    else:
        # Unreachable given the DOC_ALLOWED_EXTENSIONS check above; kept as
        # a defensive fallback rather than assuming the check above can
        # never be bypassed.
        raise MediaValidationError(f"Unsupported document extension '{ext}'.")

    text = " ".join(text.split())
    return text[:DOC_MAX_EXTRACTED_CHARS]
