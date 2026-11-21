"""VeriGraph AI — Live search service (NewsAPI, GDELT, SerpApi)."""
from __future__ import annotations
import time
import requests
from typing import Any, Dict, List
from config.settings import SERPAPI_KEY, NEWSAPI_KEY, GDELT_ENABLED, REQUEST_TIMEOUT

#: Default timeout per external provider request (seconds)
PROVIDER_TIMEOUT_CONNECT = 10
PROVIDER_TIMEOUT_READ = 30


def _get(url: str, params: dict = None, timeout: tuple = (PROVIDER_TIMEOUT_CONNECT, PROVIDER_TIMEOUT_READ)) -> dict:
    """Make an HTTP GET request with explicit failure classification.

    Returns a structured dict on every path. Failures are never silently
    converted into an empty result set: each distinct failure mode
    (timeout, TLS, HTTP status, malformed JSON) keeps its own status so
    diagnostics can tell them apart.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return {"status": "success", "data": r.json()}
    except requests.exceptions.SSLError as e:
        # TLS verification failed - do not globally disable verification
        return {"status": "ssl_error", "error": f"SSL verification failed: {e}", "error_type": "SSLError"}
    except requests.exceptions.Timeout as e:
        return {"status": "timeout", "error": f"Request timed out: {e}", "error_type": "Timeout"}
    except requests.exceptions.HTTPError as e:
        response = e.response
        return {
            "status": "http_error",
            "http_status": getattr(response, "status_code", None),
            "error": f"HTTP {getattr(response, 'status_code', 'error')}",
            "error_type": "HTTPError",
        }
    except ValueError as e:
        return {"status": "invalid_json", "error": f"Invalid JSON response: {e}", "error_type": "ValueError"}
    except requests.exceptions.RequestException as e:
        return {"status": "exception", "error": f"Request failed: {e}", "error_type": type(e).__name__}
    except Exception as e:
        return {"status": "exception", "error": f"Request failed: {e}", "error_type": type(e).__name__}


def _report_provider_status(name: str, result: dict, hit_count: int = 0) -> str:
    """Generate a structured provider status string.

    The status vocabulary distinguishes an empty result set from a genuine
    provider failure. "0 results" and "timeout" are different outcomes and must
    never be collapsed into one another.

    Returns one of:
    - "success (N results)"  - provider returned N results
    - "empty"                - provider responded successfully with zero results
    - "unconfigured"         - API key missing, provider never attempted
    - "timeout"              - request exceeded the timeout
    - "http_error (STATUS)"  - provider returned a non-2xx HTTP status
    - "ssl_error"            - TLS verification failed
    - "invalid_json"         - response body was not valid JSON
    - "exception (TYPE)"     - unexpected client-side error
    """
    status = result.get("status", "exception")

    if status == "success":
        return f"success ({hit_count} results)" if hit_count > 0 else "empty"

    if status == "unconfigured":
        return "unconfigured"

    if status == "timeout":
        return "timeout"

    if status == "ssl_error":
        return "ssl_error"

    if status == "invalid_json":
        return "invalid_json"

    if status == "http_error":
        http_status = result.get("http_status")
        return f"http_error ({http_status})" if http_status else "http_error"

    # status == "exception"
    error = str(result.get("error", ""))
    if "SSL" in error:
        return "ssl_error"
    if "timed out" in error.lower():
        return "timeout"
    error_type = result.get("error_type", "")
    return f"exception ({error_type})" if error_type else "exception"


def search_newsapi(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search NewsAPI with structured diagnostics.

    Returns: {"results": [...], "provider_status": "status_string"}
    """
    if not NEWSAPI_KEY:
        return {"results": [], "provider_status": "unconfigured"}

    result = _get("https://newsapi.org/v2/everything", {
        "apiKey": NEWSAPI_KEY, "q": query,
        "pageSize": min(limit, 100), "language": "en", "sortBy": "relevancy",
    })

    if result["status"] == "success":
        data = result["data"]
        articles = data.get("articles", [])
        hits = [{
            "provider": "newsapi", "title": (a.get("title") or "")[:200],
            "url": a.get("url", ""), "snippet": (a.get("description") or "")[:300],
            "published_at": a.get("publishedAt", ""),
            "source_name": (a.get("source") or {}).get("name", ""),
        } for a in articles[:limit]]
        status = _report_provider_status("newsapi", result, len(hits))
    else:
        hits = []
        status = _report_provider_status("newsapi", result)

    return {"results": hits, "provider_status": status, "query": query}


def search_gdelt(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search GDELT with structured diagnostics.

    Returns: {"results": [...], "provider_status": "status_string"}
    """
    if not GDELT_ENABLED:
        # A disabled provider is not a failure and not an empty result.
        return {"results": [], "provider_status": "disabled", "query": query}

    result = {"status": "exception", "error": "gdelt not attempted", "error_type": "NotAttempted"}
    data = None
    for _ in range(3):
        result = _get("https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": query, "mode": "artlist", "format": "json",
            "maxrecords": min(limit, 50), "sort": "HybridRel",
        })
        if result["status"] == "success" and result["data"]:
            data = result["data"]
            break
        # Short retry only - GDELT is rate-limited

    if data:
        articles = data.get("articles", [])
        hits = [{
            "provider": "gdelt", "title": (a.get("title") or "")[:200],
            "url": a.get("url", ""), "snippet": (a.get("seendate") or "")[:300],
            "published_at": a.get("seendate", ""),
        } for a in articles[:limit]]
        status = _report_provider_status("gdelt", result, len(hits))
    else:
        hits = []
        status = _report_provider_status("gdelt", result)

    return {"results": hits, "provider_status": status, "query": query}


def search_serpapi(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search SerpApi with structured diagnostics.

    gl="in" is used because this project is India-centric.

    Returns: {"results": [...], "provider_status": "status_string"}
    """
    if not SERPAPI_KEY:
        return {"results": [], "provider_status": "unconfigured", "query": query}

    result = _get("https://serpapi.com/search.json", {
        "engine": "google", "api_key": SERPAPI_KEY,
        "q": query, "num": min(limit, 20), "hl": "en", "gl": "in",
    })

    if result["status"] == "success":
        data = result["data"]
        organic = data.get("organic_results", [])
        hits = [{
            "provider": "serpapi", "title": (r.get("title") or "")[:200],
            "url": r.get("link", ""), "snippet": (r.get("snippet") or "")[:300],
            "published_at": r.get("date", ""),
        } for r in organic[:limit]]
        status = _report_provider_status("serpapi", result, len(hits))
    else:
        hits = []
        status = _report_provider_status("serpapi", result)

    return {"results": hits, "provider_status": status, "query": query}


def run_live_retrieval(queries: List[str], max_per_provider: int = 10) -> Dict[str, Any]:
    """Run live retrieval across all providers with per-query diagnostics.

    Provider status is recorded once per (provider, query) pair. A flat
    ``provider_status[name] = ...`` assignment would keep only the last query's
    outcome and silently hide which query produced which evidence.

    Returns:
        {
            "results": [...],
            "provider_status": {provider: aggregate_status},
            "provider_diagnostics": {
                provider: {"queries": [{query, status, raw_count, accepted_count}]}
            },
        }

    Aggregate status vocabulary: success, empty, disabled, unconfigured,
    timeout, http_error, ssl_error, invalid_json, exception.
    """
    results: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}

    providers = [("newsapi", search_newsapi), ("serpapi", search_serpapi)]

    # GDELT is reported as disabled when GDELT_ENABLED=false and is never called.
    if not GDELT_ENABLED:
        diagnostics["gdelt"] = {
            "status": "disabled",
            "queries": [{"query": q, "status": "disabled", "raw_count": 0, "accepted_count": 0}
                        for q in queries],
        }
    else:
        providers.append(("gdelt", search_gdelt))

    for name, fn in providers:
        diagnostics[name] = {"status": None, "queries": []}

    for q in queries:
        for name, fn in providers:
            entry: Dict[str, Any] = {"query": q, "status": "exception", "raw_count": 0, "accepted_count": 0}
            try:
                output = fn(q, limit=max_per_provider)
                hits = output.get("results", []) or []
                entry["status"] = output.get("provider_status", "exception")
                entry["raw_count"] = len(hits)
                entry["accepted_count"] = len(hits)
                # Preserve query provenance on every evidence item.
                for hit in hits:
                    hit["query"] = q
                    hit["retrieval_provider"] = name
                results.extend(hits)
            except Exception as e:
                # Provider-level failure is recorded, never swallowed.
                entry["status"] = "exception"
                entry["error"] = f"{type(e).__name__}: {e}"
            diagnostics[name]["queries"].append(entry)

    # Aggregate per-provider status, preferring the most informative outcome.
    for name, diag in diagnostics.items():
        query_statuses = [q["status"] for q in diag["queries"]]
        if not query_statuses:
            diag["status"] = "not_attempted"
        elif any(s.startswith("success") for s in query_statuses):
            diag["status"] = "success"
        elif any(s == "timeout" for s in query_statuses):
            diag["status"] = "timeout"
        elif any(s.startswith("http_error") for s in query_statuses):
            diag["status"] = "http_error"
        elif any(s == "ssl_error" for s in query_statuses):
            diag["status"] = "ssl_error"
        elif any(s == "invalid_json" for s in query_statuses):
            diag["status"] = "invalid_json"
        elif all(s == "empty" for s in query_statuses):
            diag["status"] = "empty"
        elif all(s == "unconfigured" for s in query_statuses):
            diag["status"] = "unconfigured"
        elif all(s == "disabled" for s in query_statuses):
            diag["status"] = "disabled"
        else:
            diag["status"] = "exception"

    provider_status = {name: diag["status"] for name, diag in diagnostics.items()}

    # Deduplicate results by normalized URL, preserving first occurrence.
    seen_urls = set()
    deduped = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)

    return {
        "results": deduped[:max_per_provider * 3],
        "provider_status": provider_status,
        "provider_diagnostics": diagnostics,
        "raw_result_count": len(results),
        "deduplicated_count": len(results) - len(deduped),
    }
