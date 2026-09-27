import asyncio
import logging
import os
import threading
from typing import Any, AsyncIterator, Dict, List, Optional

import torch
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer

try:
    # Preferred modern class for vision-language / image-text-to-text
    # models (available in recent `transformers` releases). NOTE: verify
    # this against whatever `transformers` version you deploy on Kaggle -
    # if it's older, or Qwen3-VL-4B-Instruct's model card documents a
    # model-specific class instead (the way Qwen2-VL used
    # Qwen2VLForConditionalGeneration), swap this one import for that
    # class. This is the single place in this file that is sensitive to
    # your exact transformers version / Qwen3-VL support.
    from transformers import AutoModelForImageTextToText as _SaarthiModelClass
except ImportError:  # pragma: no cover - depends on installed transformers version
    from transformers import AutoModelForCausalLM as _SaarthiModelClass

from config import *
from media_processor import MediaValidationError, process_image

logger = logging.getLogger(__name__)

hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None


# ---------------------------------------------------------------------------
# Response-policy support (additive, backward compatible)
# ---------------------------------------------------------------------------
#
# This module remains responsible ONLY for talking to the AI model
# (previously Groq, now the local Saarthi model) and Hugging Face
# embeddings. It does NOT own user profiles, database storage, WebSocket
# management, game logic, intent detection, emotion detection, or
# personality calculation - those live in their own modules
# (user_profile.py, response_policy.py, personalization.py,
# intent_router.py, database.py, server.py).
#
# The only responsibility here is: if a caller already produced a
# structured response-policy (e.g. via response_policy.build_response_policy),
# fold a short, bounded instruction snippet into the system-level
# guidance sent to the AI provider. If no policy is supplied, behavior is
# 100% identical to before.
#
# UNCHANGED FROM THE ORIGINAL FILE - not a single character in this
# section was modified.

# Bound how much influence a policy snippet can have on the prompt so it
# can never balloon the payload or smuggle in arbitrary instructions.
_MAX_POLICY_NOTES = 6
_MAX_PERSONALITY_NOTES = 4
_MAX_POLICY_NOTE_LENGTH = 200
_MAX_POLICY_PREAMBLE_LENGTH = 1500

# Whitelisted top-level fields. Anything else in the incoming dict is
# silently ignored - we never blindly concatenate arbitrary dictionaries
# into the prompt.
_ALLOWED_POLICY_KEYS = {
    "intent",
    "response_length",
    "explanation_depth",
    "tone",
    "educational_emphasis",
    "emotion_guidance",
    "personality_guidance",
    "example_usage",
    "follow_up_behavior",
    "requires_current_information",
    "recommendation_behavior",
    "notes",
}

# Emotion guidance is only ever accepted as a known category label.
# We never expose confidence scores, lexical signals, or raw detector
# output to the model - only this fixed, hand-written instruction text.
_EMOTION_INSTRUCTIONS = {
    "ACHIEVEMENT": "Use warm congratulatory language.",
    "FAILURE": "Do not sound disappointed. Be supportive and focus on recovery.",
    "CONFUSION": "Explain patiently and clearly.",
    "FRUSTRATION": "Stay calm and validating.",
    "EXCITEMENT": "Match positive energy without becoming excessive.",
    "NEUTRAL": None,
}


def _sanitize_policy_value(value: Any) -> Optional[str]:
    """Coerce a single policy field into a short, safe string, or None
    if it isn't something sensible to include in a prompt."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text[:_MAX_POLICY_NOTE_LENGTH] if text else None
    return None


def _resolve_emotion_instruction(emotion_guidance: Any) -> Optional[str]:
    """
    Convert emotion guidance into a single, fixed, concise instruction.

    Accepts either a plain category string ("FAILURE") or a dict that
    contains a "category" key (e.g. {"category": "FAILURE", "confidence": 0.87,
    "signals": [...]}). Only the category is ever used - confidence scores
    and lexical signals are deliberately dropped here so they never reach
    the model or leak internal detection details.
    """
    category = None
    if isinstance(emotion_guidance, str):
        category = emotion_guidance
    elif isinstance(emotion_guidance, dict):
        category = emotion_guidance.get("category")

    if not isinstance(category, str):
        return None

    return _EMOTION_INSTRUCTIONS.get(category.strip().upper())


def _resolve_personality_notes(personality_guidance: Any) -> List[str]:
    """
    Accept personality guidance only as a bounded list of short style
    instructions (e.g. "Use a warm tone.", "Keep the response concise.").
    Never lets this data become anything other than plain style text -
    no dicts, no nested structures, no free-form override instructions.
    """
    if isinstance(personality_guidance, str):
        candidates = [personality_guidance]
    elif isinstance(personality_guidance, list):
        candidates = personality_guidance
    else:
        return []

    notes: List[str] = []
    for item in candidates[:_MAX_PERSONALITY_NOTES]:
        sanitized = _sanitize_policy_value(item)
        if sanitized:
            notes.append(sanitized)
    return notes


def _build_policy_preamble(response_policy: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Build a short, bounded natural-language instruction block from a
    structured response-policy dict (as produced by response_policy.py's
    ResponsePolicy.to_dict()).

    This function NEVER trusts arbitrary/unbounded input: unknown keys
    are ignored (only _ALLOWED_POLICY_KEYS are read), strings are
    length-capped, list fields are count-capped, and the whole preamble
    is length-capped. Returns None if no usable policy is supplied, so
    existing callers that never pass a policy see no behavior change.

    Personality/emotion guidance is folded in as style-only notes and is
    explicitly framed as non-overriding with respect to intent, facts,
    safety, and current-information requirements.
    """
    if not response_policy or not isinstance(response_policy, dict):
        return None

    # Only look at whitelisted keys - never blindly iterate/concatenate
    # the caller's dict.
    policy = {k: v for k, v in response_policy.items() if k in _ALLOWED_POLICY_KEYS}
    if not policy:
        return None

    lines: List[str] = []

    length = _sanitize_policy_value(policy.get("response_length"))
    if length:
        lines.append(f"Preferred response length: {length}.")

    depth = _sanitize_policy_value(policy.get("explanation_depth"))
    if depth:
        lines.append(f"Explanation depth: {depth}.")

    tone = _sanitize_policy_value(policy.get("tone"))
    if tone:
        lines.append(f"Tone: {tone}.")

    if policy.get("educational_emphasis"):
        lines.append("Emphasize clear, educational explanation where relevant.")

    emotion_instruction = _resolve_emotion_instruction(policy.get("emotion_guidance"))
    if emotion_instruction:
        lines.append(emotion_instruction)

    personality_notes = _resolve_personality_notes(policy.get("personality_guidance"))
    lines.extend(personality_notes)

    examples = _sanitize_policy_value(policy.get("example_usage"))
    if examples and examples.upper() != "NONE":
        lines.append(f"Use of examples: {examples}.")

    follow_up = _sanitize_policy_value(policy.get("follow_up_behavior"))
    if follow_up and follow_up.upper() != "NONE":
        lines.append(f"Follow-up behavior: {follow_up}.")

    if policy.get("requires_current_information"):
        lines.append(
            "This topic may require current information; do not fabricate "
            "recent facts you are not confident about."
        )

    recommendation_behavior = _sanitize_policy_value(policy.get("recommendation_behavior"))
    if recommendation_behavior and recommendation_behavior.upper() not in ("NONE",):
        lines.append(f"Recommendation approach: {recommendation_behavior}.")

    notes = policy.get("notes")
    if isinstance(notes, list):
        for note in notes[:_MAX_POLICY_NOTES]:
            sanitized = _sanitize_policy_value(note)
            if sanitized:
                lines.append(sanitized)

    if not lines:
        return None

    preamble = (
        "Response guidance (style only - this may shape tone, length, and "
        "presentation, but must never override user intent, facts, safety, "
        "or current-information requirements): " + " ".join(lines)
    )
    return preamble[:_MAX_POLICY_PREAMBLE_LENGTH]


def _apply_policy_to_messages(
    messages: List[dict],
    response_policy: Optional[Dict[str, Any]],
) -> List[dict]:
    """
    Return a new messages list with an optional policy-guidance system
    message prepended/merged. Never mutates the caller's original list.
    If response_policy is None/empty, returns messages unchanged
    (same object reference is fine since nothing is modified).
    """
    if not response_policy:
        return messages

    preamble = _build_policy_preamble(response_policy)
    if not preamble:
        return messages

    new_messages = list(messages)  # shallow copy; never mutate caller's list

    if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
        merged = dict(new_messages[0])
        existing_content = merged.get("content", "") or ""
        merged["content"] = f"{existing_content}\n\n{preamble}".strip()
        new_messages[0] = merged
    else:
        new_messages.insert(0, {"role": "system", "content": preamble})

    return new_messages


def generate_embedding(text: str):
    # Kept synchronous on purpose: database.py calls this directly (not awaited),
    # so changing its signature would break that call site. It's a fast local
    # feature-extraction call, not a network-bound chat completion.
    #
    # UNCHANGED FROM THE ORIGINAL FILE. Already has the local fallback the
    # spec asked for: if HF_TOKEN isn't set, hf_client is None and this
    # returns None immediately below; if the call fails for any reason
    # (network down, HF API error, etc.), the except clause below also
    # returns None instead of raising - so database.py never crashes for
    # lack of embeddings.
    if not hf_client or not text:
        return None
    try:
        vector = hf_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            return vector[0]
        return vector
    except Exception as e:
        logger.error(f"Hugging Face API Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Local Saarthi model: singleton lazy loader
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_saarthi_model = None
_saarthi_processor = None


def load_saarthi_model():
    """Thread-safe singleton loader for the local Qwen3-VL base model plus
    the fine-tuned Saarthi LoRA adapter.

    Loads at most once per process - subsequent calls return the
    already-loaded (model, processor) pair immediately. If
    SAARTHI_ADAPTER_PATH doesn't exist on disk, logs a warning and falls
    back to running the bare base model instead of raising, so a bad/
    missing adapter path can never crash the server.
    """
    global _saarthi_model, _saarthi_processor

    if _saarthi_model is not None and _saarthi_processor is not None:
        return _saarthi_model, _saarthi_processor

    with _model_lock:
        if _saarthi_model is not None and _saarthi_processor is not None:
            return _saarthi_model, _saarthi_processor

        logger.info(f"Loading Saarthi base model: {BASE_MODEL_ID}")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

        load_kwargs: Dict[str, Any] = {}
        use_cuda = torch.cuda.is_available()
        if LOAD_IN_4BIT and use_cuda:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        elif use_cuda:
            load_kwargs["device_map"] = "auto"

        base_model = _SaarthiModelClass.from_pretrained(BASE_MODEL_ID, **load_kwargs)

        if SAARTHI_ADAPTER_PATH and os.path.isdir(SAARTHI_ADAPTER_PATH):
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(base_model, SAARTHI_ADAPTER_PATH)
                logger.info(f"Loaded Saarthi LoRA adapter from '{SAARTHI_ADAPTER_PATH}'")
            except Exception as e:
                logger.warning(
                    f"Could not load Saarthi LoRA adapter from '{SAARTHI_ADAPTER_PATH}': {e}. "
                    "Falling back to the base model."
                )
                model = base_model
        else:
            logger.warning(
                f"Saarthi LoRA adapter folder not found at '{SAARTHI_ADAPTER_PATH}'. "
                "Falling back to the base model."
            )
            model = base_model

        model.config.use_cache = True
        model.eval()

        _saarthi_model = model
        _saarthi_processor = processor
        return _saarthi_model, _saarthi_processor


# GPU concurrency guard: at most GPU_MAX_CONCURRENT_INFERENCES generate()
# calls run at once, however many callers (robot/web/app) hit this module
# concurrently - extra requests queue instead of racing for VRAM and
# crashing with a CUDA out-of-memory error.
_gpu_semaphore = asyncio.Semaphore(max(1, GPU_MAX_CONCURRENT_INFERENCES))


def _extract_content_images(messages: List[dict]) -> List[Any]:
    """Pull out every {"type": "image", "image": pil_img} entry already
    present in a multimodal messages list, in order, for the processor
    call."""
    images: List[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                images.append(item["image"])
    return images


def _attach_media_to_messages(
    messages: List[dict],
    images: Optional[List[Any]] = None,
    video_frames: Optional[List[Any]] = None,
) -> List[dict]:
    """Return a new messages list (never mutates the caller's list) with
    any images/video frames folded into the LAST user message's content,
    in the multimodal content-list format Qwen3-VL's chat template
    expects: [{"type": "image", "image": pil_img}, ..., {"type": "text",
    "text": "..."}].

    Every raw image/frame is run through media_processor.process_image
    first, so only validated, compressed, RGB PIL.Image objects ever
    reach the model - items that fail validation are logged and skipped
    rather than raising, so one bad attachment can't take down an entire
    multi-item request.
    """
    all_media = list(images or []) + list(video_frames or [])
    if not all_media:
        return messages

    processed_media = []
    for item in all_media:
        try:
            processed_media.append(process_image(item))
        except MediaValidationError as e:
            logger.warning(f"Skipping media item that failed validation: {e}")

    if not processed_media:
        return messages

    new_messages = [dict(m) for m in messages]  # shallow copy; don't mutate caller's list/dicts

    last_user_index = None
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            last_user_index = i
            break

    content_items = [{"type": "image", "image": img} for img in processed_media]

    if last_user_index is not None:
        target = dict(new_messages[last_user_index])
        existing_content = target.get("content", "")
        if isinstance(existing_content, str):
            text_items = [{"type": "text", "text": existing_content}] if existing_content else []
            target["content"] = content_items + text_items
        elif isinstance(existing_content, list):
            target["content"] = content_items + existing_content
        else:
            target["content"] = content_items
        new_messages[last_user_index] = target
    else:
        new_messages.append({"role": "user", "content": content_items})

    return new_messages


def _build_model_inputs(messages: List[dict], processor):
    """Shared helper: turn a (policy + media applied) messages list into
    tokenized model inputs on the right device, for both call_groq and
    stream_saarthi_model."""
    normalized_msgs = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            normalized_msgs.append({"role": m.get("role", "user"), "content": [{"type": "text", "text": c}]})
        else:
            normalized_msgs.append(m)
    prompt_text = processor.apply_chat_template(
        normalized_msgs, tokenize=False, add_generation_prompt=True
    )
    media_images = _extract_content_images(messages)
    return processor(text=[prompt_text], images=media_images or None, return_tensors="pt")


async def call_groq(
    messages: list,
    response_policy: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
    video_frames: Optional[List[Any]] = None,
    mode: str = "conversational",
    max_new_tokens: int = 350,
    temperature: float = 0.2,
    **kwargs,
) -> str:
    """Local Qwen3-VL + Saarthi LoRA chat completion.

    Kept under the name `call_groq` (also exported below as
    `call_saarthi_model`, an identical alias) so server.py and logic.py
    keep working with zero changes - nothing in this function talks to
    the Groq cloud API anymore; GROQ_API_KEY / GROQ_API_URL are no longer
    read here at all.

    Backward compatible: existing callers using call_groq(messages) or
    call_groq(messages, response_policy) work unchanged. `images` and
    `video_frames` are optional lists of raw bytes / base64 strings /
    PIL.Image objects, validated and compressed via
    media_processor.process_image before being folded into the prompt.
    `mode` is accepted (e.g. "robot" vs "conversational") for callers
    that want to pick between ROBOT_CONTROL_SYSTEM_PROMPT and
    CONVERSATIONAL_SYSTEM_PROMPT themselves; this function does not
    inject a system prompt on its own, matching the original call_groq's
    behavior of only ever touching what the caller already put in
    `messages`.
    """
    model, processor = load_saarthi_model()

    outgoing_messages = _apply_policy_to_messages(messages, response_policy)
    outgoing_messages = _attach_media_to_messages(outgoing_messages, images, video_frames)

    def _run_generate() -> str:
        inputs = _build_model_inputs(outgoing_messages, processor).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        decoded = processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0].strip() if decoded else ""

    async with _gpu_semaphore:
        return await asyncio.to_thread(_run_generate)


# Clean new name for new call sites; identical behavior to call_groq. Both
# names point at the exact same function object.
call_saarthi_model = call_groq


async def stream_saarthi_model(
    messages: list,
    response_policy: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
    video_frames: Optional[List[Any]] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.4,
) -> AsyncIterator[str]:
    """Token-by-token streaming generation for the web/app channels
    (web_channel/stream_ai.py).

    Runs model.generate in a background thread with
    transformers.TextIteratorStreamer so tokens can be yielded to the
    caller as soon as they're produced - no Groq call involved anywhere
    in this path. Shares the same GPU_MAX_CONCURRENT_INFERENCES semaphore
    as call_groq/call_saarthi_model, so robot, web, and app requests are
    still queued one at a time on the GPU rather than racing for VRAM.
    """
    model, processor = load_saarthi_model()

    outgoing_messages = _apply_policy_to_messages(messages, response_policy)
    outgoing_messages = _attach_media_to_messages(outgoing_messages, images, video_frames)

    inputs = _build_model_inputs(outgoing_messages, processor).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        streamer=streamer,
    )

    async with _gpu_semaphore:
        thread = threading.Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()
        try:
            for token_text in streamer:
                yield token_text
                await asyncio.sleep(0)  # let other coroutines run between tokens
        finally:
            thread.join()
