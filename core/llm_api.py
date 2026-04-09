import base64
import os

from openai import OpenAI


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"


if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Framework-EOG")
OPENROUTER_VLM_MODEL = os.getenv("OPENROUTER_VLM_MODEL", "qwen/qwen2.5-vl-32b-instruct")
OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", OPENROUTER_VLM_MODEL)
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180"))


def _build_openrouter_client():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=OPENROUTER_TIMEOUT_SECONDS,
        default_headers={
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
        },
    )


def _encode_image_as_data_url(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    mime_type = "image/jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "image/png"
    return f"data:{mime_type};base64,{encoded_image}"


def VLM(paths, prompt):
    client = _build_openrouter_client()

    image_paths = paths if isinstance(paths, list) else [paths]
    content = []

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_as_data_url(image_path)},
            }
        )

    content.append({"type": "text", "text": str(prompt)})

    response = client.chat.completions.create(
        model=OPENROUTER_VLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        top_p=0.9,
        max_tokens=OPENROUTER_MAX_TOKENS,
    )
    return response.choices[0].message.content, 0, 0, 0


def LLM(prompt):
    client = _build_openrouter_client()
    response = client.chat.completions.create(
        model=OPENROUTER_LLM_MODEL,
        messages=[{"role": "user", "content": str(prompt)}],
        temperature=0.0,
        top_p=0.9,
        max_tokens=OPENROUTER_MAX_TOKENS,
    )
    return response.choices[0].message.content
