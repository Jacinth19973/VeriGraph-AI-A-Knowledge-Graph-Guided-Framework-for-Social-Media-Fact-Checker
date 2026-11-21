"""VeriGraph AI - domain-based source credibility tiers.

Edit the sets below to change the policy; nothing else needs to change.

Tier 1  government, intergovernmental, academic, wire services, certified fact-checkers
Tier 2  mainstream news and reference sites (and verified official social accounts)
Tier 3  any other publisher that is not listed (default for unknown domains)
Tier 4  social media, blogs, forums and other user-generated platforms
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

TIER_LABELS = {
    1: "Tier 1: High Authority (Gov/Academic/Certified Fact-Checker)",
    2: "Tier 2: Mainstream Media / Reference",
    3: "Tier 3: Secondary / Regional Publisher",
    4: "Tier 4: Unverified / Personal / Social Media",
}
TIER_SCORES = {1: 0.90, 2: 0.75, 3: 0.55, 4: 0.35}
UNKNOWN_SCORE = 0.50  # dataset evidence with no source domain

# Matched against the END of the host name (so "mea.gov.in" matches ".gov.in",
# but "fakegov.com" does not match ".gov").
TIER1_SUFFIXES = (
    ".gov", ".gov.in", ".nic.in", ".gov.uk", ".gov.au", ".gov.cn", ".gov.br",
    ".gov.za", ".gov.sg", ".edu", ".edu.in", ".ac.in", ".ac.uk", ".edu.au", ".mil", ".int",
)
# Matched as the host itself or any sub-domain of it.
TIER1_DOMAINS = {
    "un.org", "who.int", "worldbank.org", "imf.org", "wto.org", "oecd.org", "europa.eu",
    "icrc.org", "unesco.org", "unicef.org", "brics.utoronto.ca",
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk", "ptinews.com",
    "politifact.com", "factcheck.org", "snopes.com", "fullfact.org", "altnews.in",
    "boomlive.in", "factly.in", "newschecker.in", "vishvasnews.com",
    "nature.com", "science.org", "thelancet.com",
}
TIER2_DOMAINS = {
    "wikipedia.org", "britannica.com",
    "thehindu.com", "indianexpress.com", "hindustantimes.com", "indiatimes.com", "ndtv.com",
    "livemint.com", "business-standard.com", "theprint.in", "scroll.in", "thewire.in",
    "news18.com", "indiatoday.in", "deccanherald.com", "tribuneindia.com", "thenewsminute.com",
    "aninews.in", "wionews.com", "firstpost.com", "outlookindia.com", "theweek.in",
    "aljazeera.com", "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "ft.com", "bloomberg.com", "cnn.com", "cnbc.com", "npr.org", "dw.com", "france24.com",
    "euronews.com", "scmp.com", "straitstimes.com", "economist.com", "time.com",
    "abcnews.go.com", "nbcnews.com", "cbsnews.com", "japantimes.co.jp", "orfonline.org",
}
TIER4_DOMAINS = {
    "facebook.com", "fb.com", "fb.watch", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "quora.com", "pinterest.com", "linkedin.com",
    "t.me", "telegram.org", "whatsapp.com", "medium.com", "substack.com", "blogspot.com",
    "wordpress.com", "tumblr.com", "wixsite.com", "weebly.com", "sharechat.com",
}
# Verified official accounts on social platforms. A post whose URL path starts with one of
# these handles is lifted from Tier 4 to Tier 2.
OFFICIAL_SOCIAL_HANDLES = {
    "meaindia", "pib_india", "pibindia", "mygovindia", "pmoindia", "narendramodi",
    "drsjaishankar", "brics2026", "who", "un", "unitednations",
}


def _host(domain: str, url: str) -> str:
    candidate = (domain or "").strip().lower()
    if candidate in ("", "unavailable", "unknown", "none", "nan"):
        candidate = ""
        if url:
            try:
                candidate = (urlparse(url).hostname or "").lower()
            except Exception:
                candidate = ""
    elif "://" in candidate:
        candidate = (urlparse(candidate).hostname or "").lower()
    return candidate[4:] if candidate.startswith("www.") else candidate


def _in(host: str, domains: set) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _official_social(url: str) -> bool:
    try:
        path = (urlparse(url or "").path or "").lower()
    except Exception:
        return False
    return any(re.match(rf"^/{re.escape(h)}(/|$)", path) for h in OFFICIAL_SOCIAL_HANDLES)


def classify_source(domain: Optional[str], url: str = "") -> Dict[str, Any]:
    """Return {"tier": 1-4 or None, "label", "score", "reason"} for a source."""
    host = _host(domain or "", url or "")
    if not host:
        return {"tier": None, "label": "Unclassified: dataset evidence (no source)",
                "score": UNKNOWN_SCORE, "reason": "source_unavailable"}

    if _in(host, TIER4_DOMAINS):
        if _official_social(url):
            tier, reason = 2, "verified_official_account"
        else:
            tier, reason = 4, "social_or_user_generated"
    elif any(host.endswith(s) for s in TIER1_SUFFIXES) or _in(host, TIER1_DOMAINS):
        tier, reason = 1, "trusted_domain"
    elif _in(host, TIER2_DOMAINS):
        tier, reason = 2, "mainstream_or_reference"
    else:
        tier, reason = 3, "unlisted_publisher"

    return {"tier": tier, "label": TIER_LABELS[tier], "score": TIER_SCORES[tier], "reason": reason}
