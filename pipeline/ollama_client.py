"""VeriGraph AI — Unified Shared Ollama Client.

Features:
- Single shared requests.Session with connection pooling
- Standardized timeout (default 90s in review mode)
- think=False, temperature=0, format="json"
- Robust multi-stage JSON repair with fallback
"""
from __future__ import annotations
import json
import re
import requests
from typing import Any, Dict, Optional
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_DISABLE_THINK

_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Return shared requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=1)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def repair_and_extract_json(raw_text: str) -> Dict[str, Any]:
    """Attempt robust extraction and multi-stage repair of JSON from LLM output."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty text received from LLM")

    # 1. Strip markdown fences if present
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    # 2. Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    except Exception:
        pass

    # 3. Find outermost brackets { ... }
    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace > start_brace:
        candidate = text[start_brace : end_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 4. Find array [ ... ]
    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if start_bracket != -1 and end_bracket > start_bracket:
        candidate = text[start_bracket : end_bracket + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return {"items": parsed}
        except Exception:
            pass

    # 5. Regex heuristic search through all nested { } blocks
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        for end in reversed([m.start() for m in re.finditer(r"\}", text)]):
            if end <= start:
                continue
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    # 6. Syntax repair: replace trailing commas or unescaped newlines
    cleaned = re.sub(r",\s*([\}\]])", r"\1", text)
    if start_brace != -1 and end_brace > start_brace:
        candidate = cleaned[start_brace : end_brace + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError(f"Failed to parse valid JSON from LLM output: {raw_text[:200]}...")


def query_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    num_predict: int = 400,
    temperature: float = 0.0,
    timeout: Optional[int] = None,
    format_type: str = "json",
) -> str:
    """Execute chat completion request to local Ollama server using shared session."""
    session = get_session()
    model_name = model or OLLAMA_MODEL
    actual_timeout = timeout if timeout is not None else OLLAMA_TIMEOUT

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    if format_type:
        payload["format"] = format_type
    if OLLAMA_DISABLE_THINK:
        payload["think"] = False

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    resp = session.post(url, json=payload, timeout=actual_timeout)
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not (content or "").strip():
        raise RuntimeError("Ollama returned empty response content.")
    return content


def query_ollama_json(
    system_prompt: str,
    user_prompt: str,
    fallback_dict: Optional[Dict[str, Any]] = None,
    num_predict: int = 500,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute LLM call and parse output as JSON with repair and deterministic fallback."""
    try:
        raw = query_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_predict=num_predict,
            temperature=0.0,
            timeout=timeout,
            format_type="json",
        )
        return repair_and_extract_json(raw)
    except Exception as exc:
        print(f"[ollama_client] JSON query failed: {exc}")
        if fallback_dict is not None:
            res = dict(fallback_dict)
            res["_error"] = str(exc)
            return res
        raise
