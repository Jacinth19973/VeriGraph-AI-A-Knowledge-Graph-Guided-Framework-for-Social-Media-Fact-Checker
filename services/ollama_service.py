"""VeriGraph AI — Ollama LLM service."""
import json
import re
import time
import os
from typing import Any, Dict
import requests

from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_DISABLE_THINK, OLLAMA_TIMEOUT, CONFIDENCE_DECIMALS
)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        for end in reversed([m.start() for m in re.finditer(r"\}", text)]):
            if end <= start:
                continue
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired
    raise ValueError("LLM response did not contain valid JSON.")


def _repair_truncated_json(text: str):
    """Close a reply that was cut off mid-object (token limit) so its complete fields are usable."""
    start = text.find("{")
    if start < 0:
        return None
    stack, in_str, esc, chars = [], False, False, []
    for ch in text[start:]:
        chars.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                break
    body = "".join(chars) + ('"' if in_str else "")
    for _ in range(4):
        body = re.sub(r"[,:\s]+$", "", body)
        try:
            obj = json.loads(body + "".join(reversed(stack)))
            return obj if isinstance(obj, dict) else None
        except Exception:
            cut = max(body.rfind(","), body.rfind("{"), body.rfind("["))
            if cut < 0:
                return None
            body = body[:cut + 1] if body[cut] in "{[" else body[:cut]
    return None


def _with_current_date(system_prompt: str) -> str:
    """Append the runtime date so no agent assumes its training-cutoff year is 'now'."""
    try:
        from services.time_service import utc_now, india_now
        utc, ist = utc_now(), india_now()
        stamp = (f"CURRENT DATE (authoritative system clock): {utc.date().isoformat()} UTC; "
                 f"{ist.strftime('%B %d, %Y')} IST.\n"
                 "Use this as 'today' for ALL temporal reasoning. Do not use your training cutoff or any "
                 "other date as the present. A date is in the future only if it is after the current date "
                 "above; an event dated before it is in the past and may already have happened.")
    except Exception:
        return system_prompt
    return f"{system_prompt}\n\n{stamp}"


def _measure_prompt(system_prompt: str, user_prompt: str) -> Dict[str, int]:
    """Measure prompt character count and approximate token count."""
    combined = f"{system_prompt}\n{user_prompt}"
    char_count = len(combined)
    # Approximate tokens: ~4 characters per token for English
    token_count = max(1, int(char_count / 4))
    return {"char_count": char_count, "token_count": token_count}


def _budget_strategy(
    evidence_items: list,
    prompt_measurement: Dict[str, int],
    max_tokens: int = 32000,
    min_evidence: int = 1,
    max_evidence: int = 10,
) -> list:
    """Budget-aware evidence selection strategy.
    
    Strategy: rank -> retain diverse evidence -> limit snippet length -> preserve source diversity
    
    Args:
        evidence_items: List of evidence dicts with 'final_score', 'provider', 'snippet' keys
        prompt_measurement: {'char_count': int, 'token_count': int} from _measure_prompt
        max_tokens: Maximum total tokens allowed for evidence snippets in context
        min_evidence: Minimum number of evidence items to always retain
        max_evidence: Maximum number of evidence items to consider
    
    Returns:
        Selected evidence items within token budget
    """
    # Cap the number of evidence items to consider
    items = evidence_items[:max_evidence]
    
    # Sort by final_score descending (rank)
    items.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    # Retain diverse evidence - track source diversity
    selected = []
    seen_sources = set()
    total_tokens = prompt_measurement["token_count"]
    
    # Always include at least min_evidence items
    for item in items:
        if len(selected) >= min_evidence:
            # Check source diversity
            provider = item.get("provider", "")
            if provider in seen_sources and len(selected) >= min_evidence + 2:
                # If we already have this source and have enough items, skip
                snippet_tokens = len(item.get("snippet", "")) // 4
                if total_tokens + snippet_tokens > max_tokens:
                    break
            else:
                snippet_tokens = len(item.get("snippet", "")) // 4 if item.get("snippet") else 0
                if total_tokens + snippet_tokens > max_tokens:
                    break
        else:
            snippet_tokens = len(item.get("snippet", "")) // 4 if item.get("snippet") else 0
        
        # Check token budget
        if total_tokens + snippet_tokens > max_tokens and len(selected) >= min_evidence:
            break
        
        snippet = item.get("snippet", "") or ""
        snippet_tokens = len(snippet) // 4 if snippet else 0
        
        # Check token budget
        if total_tokens + snippet_tokens > max_tokens and len(selected) >= min_evidence:
            break
        
        selected.append(item)
        seen_sources.add(provider)
        total_tokens += snippet_tokens
    
    # If we have fewer than min_evidence, pad with lowest-scoring items
    if len(selected) < min_evidence and items:
        remaining = [i for i in items if i not in selected]
        remaining.sort(key=lambda x: x.get("final_score", 0))
        for item in remaining[::-1]:  # Take lowest-scored to fill minimum
            if len(selected) >= min_evidence:
                break
            snippet = item.get("snippet", "") or ""
            snippet_tokens = len(snippet) // 4 if snippet else 0
            if total_tokens + snippet_tokens <= max_tokens:
                selected.append(item)
                total_tokens += snippet_tokens
    
    return selected[:max_evidence]


def call_ollama(system_prompt: str, user_prompt: str, model: str = None,
                num_predict: int = 400, temperature: float = 0.0,
                timeout: int = None) -> Dict[str, Any]:
    """Call Ollama with performance measurement and timeout handling.
    
    Returns dict with:
    - "content": the LLM response text
    - "prompt_chars": number of prompt characters
    - "prompt_tokens": approximate prompt token count
    - "generation_duration": time in seconds
    - "timeout": whether timeout occurred
    - "evidence_count": number of evidence items considered
    
    Key behaviors:
    - Timeout does NOT erase successful retrieval - partial results are preserved
    - Budget strategy ranks evidence and respects token limits
    - All error modes handled gracefully
    """
    model = model or OLLAMA_MODEL
    actual_timeout = timeout if timeout is not None else OLLAMA_TIMEOUT
    
    system_prompt = _with_current_date(system_prompt)

    # Measure prompt performance
    prompt_measurement = _measure_prompt(system_prompt, user_prompt)
    
    start_time = time.time()
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if OLLAMA_DISABLE_THINK:
        payload["think"] = False
    import os as _os
    payload["keep_alive"] = _os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    _ctx = _os.getenv("OLLAMA_NUM_CTX", "").strip()
    if _ctx.isdigit():
        payload["options"]["num_ctx"] = int(_ctx)
    
    try:
        r = requests.post(
            OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
            json=payload, timeout=actual_timeout
        )
        r.raise_for_status()
        
        content = r.json().get("message", {}).get("content", "")
        generation_duration = time.time() - start_time
        
        if not (content or "").strip():
            # Empty response - return structured empty result
            return {
                "content": "",
                "prompt_chars": prompt_measurement["char_count"],
                "prompt_tokens": prompt_measurement["token_count"],
                "generation_duration": generation_duration,
                "timeout": False,
                "evidence_count": 0,
                "error": "empty_response",
            }
        
        # Success - return structured result
        return {
            "content": content,
            "prompt_chars": prompt_measurement["char_count"],
            "prompt_tokens": prompt_measurement["token_count"],
            "generation_duration": round(generation_duration, 3),
            "timeout": False,
            "evidence_count": len(payload.get("messages", [])),
        }
        
    except requests.exceptions.Timeout:
        generation_duration = time.time() - start_time
        # Timeout does NOT erase - return attempt metrics
        return {
            "content": "",
            "prompt_chars": prompt_measurement["char_count"],
            "prompt_tokens": prompt_measurement["token_count"],
            "generation_duration": round(generation_duration, 3),
            "timeout": True,
            "evidence_count": 0,
            "error": "timeout",
        }
        
    except Exception as e:
        generation_duration = time.time() - start_time
        return {
            "content": "",
            "prompt_chars": prompt_measurement["char_count"],
            "prompt_tokens": prompt_measurement["token_count"],
            "generation_duration": round(generation_duration, 3),
            "timeout": False,
            "evidence_count": 0,
            "error": type(e).__name__,
        }


def call_llm(system_prompt: str, user_prompt: str, role: str = "general",
             num_predict: int = 400, timeout: int = None) -> Dict[str, Any]:
    """Call Ollama LLM with performance measurement.
    
    Returns structured result with metrics.
    """
    result = call_ollama(system_prompt, user_prompt, num_predict=num_predict,
                         timeout=timeout if timeout is not None else OLLAMA_TIMEOUT)
    # Also extract content for backward compatibility
    try:
        json_result = extract_json_object(result["content"]) if result["content"] else {}
        content = json_result.get("text", result["content"]) if isinstance(json_result, dict) else result["content"]
    except Exception:
        content = result["content"] if result["content"] else ""
    
    return {
        "content": content,
        "prompt_chars": result["prompt_chars"],
        "prompt_tokens": result["prompt_tokens"],
        "generation_duration": result["generation_duration"],
        "timeout": result["timeout"],
        "error": result.get("error"),
    }


def check_ollama_health() -> bool:
    try:
        r = requests.get(OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False