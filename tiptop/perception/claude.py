import base64
import io
import json
import logging
import os
from functools import cache
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from PIL import Image

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "anthropic/claude-opus-5.5"
_MAX_TOKENS = 16000


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text(encoding="utf-8").strip()


def _detect_and_translate_prompt_name() -> str:
    if tiptop_cfg().experimental.pick_place_next_to:
        return "detect_and_translate_pick_place_next_to"
    return "detect_and_translate"


def _api_key() -> str:
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY environment variable must be set to call Claude via OpenRouter")
    return api_key


@cache
def claude_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=_api_key())


@cache
def async_claude_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=_api_key())


def _image_block(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _request_kwargs(image: Image.Image, prompt: str, model_id: str, effort: str) -> dict:
    return {
        "model": model_id,
        "max_tokens": _MAX_TOKENS,
        "messages": [{"role": "user", "content": [_image_block(image), {"type": "text", "text": prompt}]}],
        "extra_body": {"reasoning": {"effort": effort}},
    }


def _response_text(response: ChatCompletion) -> str:
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(f"Claude response was truncated at {_MAX_TOKENS} tokens")
    if not choice.message.content:
        raise RuntimeError(f"Claude returned no text (finish_reason={choice.finish_reason})")
    return choice.message.content


def query_image(
    image: Image.Image,
    prompt: str,
    client: OpenAI | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str = "medium",
) -> str:
    """Send an image and prompt to Claude via OpenRouter and return the text response."""
    client = client or claude_client()
    response = client.chat.completions.create(**_request_kwargs(image, prompt, model_id, effort))
    return _response_text(response)


async def query_image_async(
    image: Image.Image,
    prompt: str,
    client: AsyncOpenAI | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str = "medium",
) -> str:
    """Asynchronously send an image and prompt to Claude via OpenRouter and return the text response."""
    client = client or async_claude_client()
    response = await client.chat.completions.create(**_request_kwargs(image, prompt, model_id, effort))
    return _response_text(response)


def load_json(response_text: str) -> list | dict:
    """Parse the JSON object or array in the response, ignoring any surrounding text or code fencing."""
    starts = [i for i in (response_text.find("{"), response_text.find("[")) if i != -1]
    end = max(response_text.rfind("}"), response_text.rfind("]"))
    if not starts or end == -1:
        _log.error(f"No JSON found in response: {response_text}")
        raise ValueError(f"No JSON found in response: {response_text}")
    cleaned_text = response_text[min(starts) : end + 1]

    try:
        results = json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise
    return results


def _parse_response(response_text: str) -> tuple[list, list]:
    """Parse Claude response text into bboxes and grounded atoms."""
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(f"Claude returned a non-JSON response; check for a discrepancy in your image: {response_text}")
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: OpenAI | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str = "medium",
) -> tuple[list[dict], list[dict]]:
    """Detect objects and translate task in a single Claude API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: OpenRouter client. If None, a new client will be created.
        model_id: OpenRouter model ID to use.
        effort: Reasoning effort level (low, medium, high, xhigh, max).

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes
        - grounded_atoms: List of predicate specifications
    """
    prompt = load_prompt(_detect_and_translate_prompt_name()).format(task_instruction=task_instruction)
    return _parse_response(query_image(image, prompt, client, model_id, effort))


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: AsyncOpenAI | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str = "medium",
) -> tuple[list[dict], list[dict]]:
    """Asynchronously detect objects and translate task in a single Claude API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: OpenRouter client. If None, a new client will be created.
        model_id: OpenRouter model ID to use.
        effort: Reasoning effort level (low, medium, high, xhigh, max).

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes.
        - grounded_atoms: List of predicate specifications.
    """
    prompt = load_prompt(_detect_and_translate_prompt_name()).format(task_instruction=task_instruction)
    return _parse_response(await query_image_async(image, prompt, client, model_id, effort))
