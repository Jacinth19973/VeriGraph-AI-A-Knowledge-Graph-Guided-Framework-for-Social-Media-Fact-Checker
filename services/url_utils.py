"""VeriGraph AI — URL validation and provenance utilities."""
from __future__ import annotations
import re
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, Optional, Literal

ProvenanceType = Literal[
    "REAL_EXTERNAL_URL",
    "DATASET_EVIDENCE",
    "SOCIAL_MEDIA_SOURCE",
    "KNOWLEDGE_GRAPH",
    "UNRESOLVED"
]

URL_STATUS_RE = re.compile(
    r"^https?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Normalize a URL by ensuring it has a scheme and proper formatting."""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        # Check if it looks like a domain
        if re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", url):
            url = "https://" + url
        else:
            return ""
    # Parse and reconstruct to normalize
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return ""
        # Rebuild without fragment
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ""
        ))
        return normalized
    except Exception:
        return ""


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL using urlparse."""
    if not url:
        return None
    try:
        parsed = urlparse(normalize_url(url))
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


def validate_url(url: str) -> bool:
    """Validate if a string is a properly formatted URL."""
    if not url:
        return False
    normalized = normalize_url(url)
    if not normalized:
        return False
    return bool(URL_STATUS_RE.match(normalized))


def resolve_source_provenance(
    url: str,
    source: str = "",
    provider: str = "",
    dataset: str = "",
    retrieval_channel: str = ""
) -> Dict[str, Any]:
    """
    Resolve the provenance type and URL status for an evidence item.
    
    Returns a dict with provenance information.
    """
    provenance: Dict[str, Any] = {
        "url": url or "",
        "canonical_url": "",
        "domain": None,
        "url_status": "not_available",
        "url_verified": False,
        "provenance_type": "UNRESOLVED",
    }
    
    normalized = normalize_url(url)
    provenance["canonical_url"] = normalized
    provenance["domain"] = extract_domain(normalized) if normalized else None
    
    if not normalized:
        provenance["url_status"] = "not_available"
        # Determine provenance from other fields
        if dataset:
            provenance["provenance_type"] = "DATASET_EVIDENCE"
        elif retrieval_channel == "live_search" or provider in ("newsapi", "gdelt", "serpapi"):
            provenance["provenance_type"] = "REAL_EXTERNAL_URL"
        elif retrieval_channel == "social_media":
            provenance["provenance_type"] = "SOCIAL_MEDIA_SOURCE"
        elif retrieval_channel == "kg" or provider == "knowledge_graph":
            provenance["provenance_type"] = "KNOWLEDGE_GRAPH"
        return provenance
    
    # URL exists, validate it
    is_valid = validate_url(normalized)
    provenance["url_verified"] = is_valid
    
    if is_valid:
        provenance["url_status"] = "reachable"  # We assume reachable if valid format
        provenance["provenance_type"] = "REAL_EXTERNAL_URL"
    else:
        provenance["url_status"] = "invalid"
        provenance["provenance_type"] = "UNRESOLVED"
    
    # Override provenance type based on known sources
    if dataset and not normalized:
        provenance["provenance_type"] = "DATASET_EVIDENCE"
    elif retrieval_channel == "social_media":
        provenance["provenance_type"] = "SOCIAL_MEDIA_SOURCE"
    elif retrieval_channel == "kg" or provider == "knowledge_graph":
        provenance["provenance_type"] = "KNOWLEDGE_GRAPH"
    
    return provenance


def format_url_for_display(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format URL for display in UI based on provenance.
    Returns dict with display_url, is_clickable, display_text.
    """
    url = provenance.get("canonical_url") or provenance.get("url", "")
    url_status = provenance.get("url_status", "not_available")
    prov_type = provenance.get("provenance_type", "UNRESOLVED")
    domain = provenance.get("domain")
    
    if prov_type == "DATASET_EVIDENCE":
        return {
            "display_url": "",
            "is_clickable": False,
            "display_text": "Dataset evidence — no public URL available",
            "domain": domain,
            "dataset_note": True,
        }
    
    if not url or url_status == "not_available":
        return {
            "display_url": "",
            "is_clickable": False,
            "display_text": "Source unavailable",
            "domain": domain,
        }
    
    if url_status == "invalid":
        return {
            "display_url": url,
            "is_clickable": False,
            "display_text": f"Invalid URL format: {url}",
            "domain": domain,
        }
    
    # Valid URL
    display_domain = domain or extract_domain(url) or "unknown"
    return {
        "display_url": url,
        "is_clickable": True,
        "display_text": url,
        "domain": display_domain,
    }


def clean_snippet(text: str, max_len: int = 350) -> str:
    """Clean and truncate snippet text."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text.strip())
    # Truncate
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text