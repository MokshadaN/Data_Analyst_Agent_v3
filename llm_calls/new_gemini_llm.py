import os
import re
import json
import time
import hashlib
import threading
from typing import Optional, Tuple

from google import genai
from google.genai.types import GenerateContentConfig

# ----------------------------
# Config
# ----------------------------
# Primary & fallback models (override via env if you like)
MODEL_CHAIN = [
    os.getenv("GEMINI_MODEL_PRIMARY", "gemini-2.5-pro"),
    os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash"),
]

# Default for ad-hoc calls/tests (not critical once MODEL_CHAIN is set)
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL_DEFAULT", "gemini-2.5-flash")

# Prefer environment variable, fall back to hardcoded (you can delete the hardcoded key)
api_key = "AIzaSyD8d74JXRIoAMctsgaJL3SfxSaDXKcf5ys"
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# ----------------------------
# Simple disk cache for code generations
# ----------------------------
_CACHE_DIR = "cache"
_CACHE_FILE = os.path.join(_CACHE_DIR, "code_cache.json")
_lock = threading.Lock()


def _ensure_cache():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load_cache() -> dict:
    _ensure_cache()
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    with _lock:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


def _prompt_key(system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8", "ignore"))
    h.update(b"\n---\n")
    h.update(user_prompt.encode("utf-8", "ignore"))
    return h.hexdigest()


# ----------------------------
# Code extraction helpers
# ----------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)
_CODE_LIKE_LINE = re.compile(r'^\s*(from\s+\w+|import\s+\w+|def\s+\w+\(|class\s+\w+\(|#|"""|\'\'\')')


def _extract_code(text: str) -> str:
    """
    Prefer the last fenced block; otherwise if it looks like code,
    strip preamble lines until we hit a code-like line.
    """
    text = (text or "").strip()
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()

    # Heuristic fallback: drop non-code preamble
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if _CODE_LIKE_LINE.search(ln):
            return "\n".join(lines[i:]).strip()
    return text  # last resort


def _should_retry(status_code: Optional[int], message: str) -> bool:
    """
    Decide whether to retry based on status/message.
    Handles common transient classes: 429, 5xx, gateway/timeouts, or generic 'internal' errors.
    """
    if status_code in (429, 500, 502, 503, 504):
        return True
    if status_code in (0, None):  # transport-layer or unknown
        return True
    if "internal" in (message or "").lower():
        return True
    if "timeout" in (message or "").lower():
        return True
    return False


# ----------------------------
# Main API
# ----------------------------
def gemini_call_for_code(
    system_prompt: str,
    user_prompt: str,
    content=None,
    total_attempts: int = 6,
    per_model_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 8.0,
    prefer_cached: bool = True,
) -> str:
    """
    Robust Gemini caller that:
      - retries with exponential backoff + jitter on transient failures,
      - falls back across models in MODEL_CHAIN,
      - caches successful generations by (system_prompt, user_prompt) hash,
      - returns cleaned Python code only.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized (missing API key?).")

    key = _prompt_key(system_prompt, user_prompt)

    # Try cache first
    if prefer_cached:
        cache = _load_cache()
        cached = cache.get(key)
        if cached:
            return cached

    attempts_left = total_attempts
    last_err = None
    code_out: Optional[str] = None

    for model in MODEL_CHAIN:
        model_attempts = min(per_model_attempts, attempts_left)
        if model_attempts <= 0:
            break

        backoff = initial_backoff
        for i in range(1, model_attempts + 1):
            try:
                contents = user_prompt
                if content is not None:
                    # Attach structured content in a safe way
                    content_str = json.dumps(content, ensure_ascii=False, indent=2) \
                        if isinstance(content, (dict, list)) else str(content)
                    contents = f"{user_prompt}\n\n--- Additional Context ---\n{content_str}"

                print(f"[Gemini] Attempt {total_attempts - attempts_left + 1}/{total_attempts} using model={model}")

                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,  # deterministic codegen
                    ),
                )

                # Pull text from the most reliable spots
                text = getattr(resp, "text", "") or ""
                if not text and getattr(resp, "candidates", None):
                    try:
                        text = resp.candidates[0].content.parts[0].text or ""
                    except Exception:
                        text = ""

                code_out = _extract_code(text)
                if not code_out.strip():
                    raise RuntimeError("Empty code block returned.")

                # Persist success in cache
                if prefer_cached:
                    cache = _load_cache()
                    cache[key] = code_out
                    _save_cache(cache)

                # Optionally write the raw text for debugging (comment out if not needed)
                try:
                    with open("temp.py", "w", encoding="utf-8", newline="\n") as f:
                        f.write(text)
                except Exception:
                    pass

                return code_out

            except Exception as e:
                # Extract best-effort status/message
                status = None
                msg = str(e)
                try:
                    data = getattr(e, "args", [None])[0]
                    if isinstance(data, dict):
                        status = data.get("error", {}).get("code")
                        msg = data.get("error", {}).get("message", msg)
                except Exception:
                    pass

                last_err = f"(model={model}) attempt {i}/{model_attempts}: {status} {msg}"
                print(f"[Retry] {last_err}")

                # Retry?
                if i < model_attempts and _should_retry(status, msg):
                    sleep_for = min(backoff, max_backoff)
                    # jitter: +/- 20%
                    jitter = sleep_for * 0.2
                    time.sleep(sleep_for + (jitter * (0.5 - (time.time() % 1))))
                    backoff *= 2
                    continue
                else:
                    break  # break this model and try next

        attempts_left -= model_attempts
        if attempts_left <= 0:
            break

    # As a last resort, return cached code if present even if prefer_cached=False
    cache = _load_cache()
    if key in cache:
        return cache[key]

    raise RuntimeError(f"Gemini API call failed: {last_err or 'no further details'}")


# ----------------------------
# Quick test harness
# ----------------------------
def test_gemini_call_for_code():
    # Minimal, code-only instruction
    system_prompt = "You are a Python code generator. Respond ONLY with executable Python code."

    # Simple task
    user_prompt = (
        "Write Python code that creates a pandas DataFrame with two columns A and B, "
        "each containing numbers 1 to 5, and prints the sum of column A."
    )

    try:
        code = gemini_call_for_code(system_prompt, user_prompt)
        print("\n--- Cleaned Code from Gemini ---\n")
        print(code)

        # Optional: Execute to verify
        exec_globals = {}
        exec(code, exec_globals)
        print("\n--- Execution Completed ---\n")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_gemini_call_for_code()
