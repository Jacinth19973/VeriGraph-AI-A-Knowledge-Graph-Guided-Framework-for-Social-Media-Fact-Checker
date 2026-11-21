"""VeriGraph AI - detect when evidence about an event contradicts the claim's date or place.

Relevance is not stance: an article about "the 18th BRICS Summit" is highly relevant to
"held in Mumbai in March 2026" and still contradicts it, because it says New Delhi and September.
These checks are deterministic and every conflict carries a readable explanation, so the
Judge, the specialists and the UI can all cite the same facts.

Call them only with evidence that is already about the same event (the narrative stage keeps
only evidence that mentions the claim's specific entity, e.g. BRICS).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
                8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
_M = "|".join(sorted(_MONTHS, key=len, reverse=True))
_Y = r"(?:19|20)\d{2}"
_DATE_PATTERNS = [
    re.compile(rf"\b(?P<m>{_M})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*[-\u2013]\s*\d{{1,2}})?,?\s*(?P<y>{_Y})?\b", re.I),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?(?:\s*(?:[-\u2013]|to)\s*\d{{1,2}}(?:st|nd|rd|th)?)?\s+(?:of\s+)?(?P<m>{_M})\.?,?\s*(?P<y>{_Y})?\b", re.I),
    re.compile(rf"\b(?P<m>{_M})\.?,?\s+(?P<y>{_Y})\b", re.I),
]

# Cities / regions (compared with each other) and countries (never a conflict by themselves).
_COUNTRIES = {
    "india", "bharat", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan", "myanmar", "china", "japan", "russia",
    "brazil", "mexico", "egypt", "nigeria", "kenya", "ethiopia", "iran", "indonesia", "thailand", "vietnam", "malaysia",
    "australia", "canada", "france", "germany", "italy", "spain", "portugal", "sweden", "norway", "denmark", "finland",
    "poland", "ukraine", "turkey", "greece", "argentina", "chile", "peru", "colombia", "cuba", "uae", "qatar", "israel",
    "united states", "usa", "united kingdom", "uk", "south africa", "saudi arabia", "south korea", "north korea",
    "costa rica", "new zealand", "switzerland", "austria", "belgium", "netherlands", "ireland", "afghanistan", "iraq",
    "syria", "philippines", "united arab emirates",
}
_PLACES = [
    # India: cities, states, union territories
    "New Delhi", "Delhi", "Mumbai", "Bombay", "Chennai", "Madras", "Bangalore", "Bengaluru", "Hyderabad", "Kolkata",
    "Calcutta", "Ahmedabad", "Pune", "Surat", "Indore", "Bhubaneswar", "Agra", "Jaipur", "Lucknow", "Bhopal", "Nagpur",
    "Goa", "Panaji", "Kochi", "Cochin", "Thiruvananthapuram", "Trivandrum", "Kozhikode", "Coimbatore", "Madurai",
    "Visakhapatnam", "Vijayawada", "Amaravati", "Patna", "Ranchi", "Raipur", "Guwahati", "Shillong", "Imphal",
    "Kohima", "Aizawl", "Agartala", "Itanagar", "Gangtok", "Dehradun", "Shimla", "Srinagar", "Jammu", "Leh", "Chandigarh",
    "Gurugram", "Gurgaon", "Noida", "Ghaziabad", "Faridabad", "Kanpur", "Varanasi", "Prayagraj", "Allahabad", "Meerut",
    "Amritsar", "Ludhiana", "Jodhpur", "Udaipur", "Kota", "Vadodara", "Rajkot", "Gandhinagar", "Nashik", "Aurangabad",
    "Mysuru", "Mysore", "Mangaluru", "Hubballi", "Puducherry", "Pondicherry", "Kevadia", "Dharamshala",
    "Maharashtra", "Tamil Nadu", "Karnataka", "Kerala", "Gujarat", "Rajasthan", "Punjab", "Haryana", "Bihar", "Odisha",
    "Assam", "Telangana", "West Bengal", "Uttar Pradesh", "Madhya Pradesh", "Andhra Pradesh", "Uttarakhand", "Jharkhand",
    "Chhattisgarh", "Himachal Pradesh", "Jammu and Kashmir", "Ladakh", "Manipur", "Meghalaya", "Sikkim", "Tripura",
    # world cities
    "London", "Paris", "Berlin", "Madrid", "Barcelona", "Rome", "Milan", "Moscow", "St. Petersburg", "Washington",
    "New York", "Los Angeles", "San Francisco", "Chicago", "Houston", "Boston", "Seattle", "Miami", "Las Vegas",
    "Toronto", "Ottawa", "Vancouver", "Montreal", "Beijing", "Shanghai", "Shenzhen", "Hong Kong", "Guangzhou", "Tokyo",
    "Osaka", "Kyoto", "Seoul", "Pyongyang", "Taipei", "Canberra", "Sydney", "Melbourne", "Brasilia", "Sao Paulo",
    "Rio de Janeiro", "Buenos Aires", "Santiago", "Lima", "Bogota", "Mexico City", "Havana", "Jakarta", "Bali",
    "Bangkok", "Hanoi", "Ho Chi Minh City", "Manila", "Kuala Lumpur", "Singapore", "Dubai", "Abu Dhabi", "Doha",
    "Riyadh", "Jeddah", "Tehran", "Baghdad", "Istanbul", "Ankara", "Jerusalem", "Tel Aviv", "Cairo", "Johannesburg",
    "Cape Town", "Durban", "Pretoria", "Nairobi", "Lagos", "Abuja", "Addis Ababa", "Kigali", "Accra", "Casablanca",
    "Geneva", "Zurich", "Brussels", "Vienna", "Amsterdam", "The Hague", "Stockholm", "Oslo", "Copenhagen", "Helsinki",
    "Warsaw", "Prague", "Budapest", "Kyiv", "Kiev", "Athens", "Lisbon", "Dublin", "Edinburgh", "Islamabad", "Lahore",
    "Karachi", "Dhaka", "Kathmandu", "Colombo", "Thimphu", "Kabul", "Kazan", "Sochi", "Xiamen", "Fortaleza", "Davos",
    "Tashkent", "Astana", "Nur-Sultan", "Baku", "Tbilisi", "Yerevan", "Minsk",
]
_ALIASES = {
    "new delhi": "delhi", "bengaluru": "bangalore", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurugram": "gurgaon", "cochin": "kochi", "trivandrum": "thiruvananthapuram", "allahabad": "prayagraj",
    "mysuru": "mysore", "pondicherry": "puducherry", "kiev": "kyiv", "nur-sultan": "astana",
    "st. petersburg": "saint petersburg", "banaras": "varanasi", "vizag": "visakhapatnam",
}
_PLACE_RE = [(p, re.compile(rf"\b{re.escape(p)}\b", re.I)) for p in sorted(_PLACES, key=len, reverse=True)]

# Words that make a capitalised phrase a venue/building rather than a city.
_VENUE_WORDS = {
    "mandapam", "centre", "center", "hall", "hotel", "stadium", "arena", "convention", "university", "college",
    "institute", "palace", "fort", "park", "airport", "station", "temple", "mosque", "church", "hospital", "complex",
    "bhavan", "bhawan", "auditorium", "garden", "gardens", "square", "tower", "mall", "hub", "campus", "house",
    "parliament", "ministry", "embassy", "court", "bank", "club",
}
_NOT_PLACES = {"the", "this", "that", "a", "an", "it", "its", "his", "her", "their", "our", "my", "your", "facebook",
               "twitter", "instagram", "youtube", "whatsapp", "telegram", "x", "news", "media", "public", "person"}
_EVENT_VERB = (r"(?:held|hosted|took\s+place|taking\s+place|happened|occurred|conducted|organi[sz]ed|inaugurated|"
               r"launched|opened|convened|signed|began|begins|started|ended|concluded|staged|located|born|died)")
_CLAIM_PLACE_RE = re.compile(
    rf"\b{_EVENT_VERB}\s+(?:\w+\s+){{0,2}}?(?:in|at|near)\s+"
    r"((?:[A-Z][\w'\u2019.-]*)(?:\s+(?:[A-Z][\w'\u2019.-]*)){0,2})")


def _norm_place(name: str) -> str:
    n = re.sub(r"[^\w\s.-]", "", name.lower()).strip()
    return _ALIASES.get(n, n)


_DAYS = (r"(?P<d1>\d{1,2})(?!\d)(?:st|nd|rd|th)?"
         r"(?:\s*(?:[-\u2013\u2014]|to|and)\s*(?P<d2>\d{1,2})(?!\d)(?:st|nd|rd|th)?)?")
_P_DM = re.compile(rf"\b{_DAYS}\s+(?:of\s+)?(?P<m>{_M})\.?(?:,?\s*(?P<y>{_Y}))?\b", re.I)
_P_MD = re.compile(rf"\b(?P<m>{_M})\.?\s+{_DAYS}(?:,?\s*(?P<y>{_Y}))?\b", re.I)
_P_MY = re.compile(rf"\b(?P<m>{_M})\.?,?\s+(?P<y>{_Y})\b", re.I)


def date_mentions(text: str) -> List[Dict[str, Any]]:
    """Every dated mention: month, year (or None) and day range (or None)."""
    text = text or ""
    spans: List[Tuple[int, int]] = []
    out: List[Dict[str, Any]] = []
    for pat in (_P_DM, _P_MD, _P_MY):
        for m in pat.finditer(text):
            lo, hi = m.span()
            if any(not (hi <= x or lo >= y) for x, y in spans):
                continue
            month = _MONTHS.get(m.group("m").lower())
            gd = m.groupdict()
            d1 = int(gd["d1"]) if gd.get("d1") else None
            d2 = int(gd["d2"]) if gd.get("d2") else d1
            if not month or (d1 is not None and not 1 <= d1 <= 31):
                continue
            spans.append((lo, hi))
            out.append({"month": month, "year": int(gd["y"]) if gd.get("y") else None, "d1": d1, "d2": d2})
    return out


def month_mentions(text: str) -> Set[Tuple[int, Optional[int]]]:
    """(month number, year or None) for every dated mention in the text."""
    return {(d["month"], d["year"]) for d in date_mentions(text)}


def _fmt(d: Dict[str, Any]) -> str:
    day = "" if d["d1"] is None else (f"{d['d1']} " if d["d2"] in (None, d["d1"]) else f"{d['d1']}\u2013{d['d2']} ")
    year = f" {d['year']}" if d["year"] else ""
    return f"{day}{_MONTH_NAMES[d['month']]}{year}"


def _label(month: int, year: Optional[int]) -> str:
    return f"{_MONTH_NAMES[month]} {year}" if year else _MONTH_NAMES[month]


def _date_conflict(claim_dates, ev_dates):
    """(kind, evidence dates) when the evidence contradicts the claim's date; None when consistent or unknown."""
    found = None
    for c in claim_dates:
        same = [e for e in ev_dates if e["month"] == c["month"] and (c["year"] is None or e["year"] is None or e["year"] == c["year"])]
        if same:
            with_days = [e for e in same if e["d1"] is not None]
            if c["d1"] is not None and with_days and not any(e["d1"] <= c["d2"] and c["d1"] <= e["d2"] for e in with_days):
                found = found or ("day", with_days)
                continue
            return None                                   # a matching date exists -> consistent
        other_year = [e for e in ev_dates if e["month"] == c["month"] and c["year"] and e["year"] and e["year"] != c["year"]]
        if other_year and not any(e["year"] == c["year"] for e in ev_dates):
            found = found or ("year", other_year)
            continue
        in_year = [e for e in ev_dates if c["year"] is None or e["year"] is None or e["year"] == c["year"]]
        if in_year:
            found = found or ("month", in_year)
    return found


def places_in(text: str) -> List[str]:
    """Cities/regions named in the text (longest match first, so 'New Delhi' wins over 'Delhi')."""
    remaining = text or ""
    out: List[str] = []
    for name, rx in _PLACE_RE:
        if rx.search(remaining):
            if _norm_place(name) not in [_norm_place(o) for o in out]:
                out.append(name)
            remaining = rx.sub(" ", remaining)
    return out


def claim_places(claim: str) -> List[str]:
    """Places a claim says an event happened: known cities plus a capitalised name after an event verb
    ('held in Bengaluru'), which covers places the gazetteer does not list. Venues and countries are skipped."""
    out = [p for p in places_in(claim) if p.lower() not in _COUNTRIES]
    for m in _CLAIM_PLACE_RE.finditer(claim or ""):
        name = re.sub(r"[\u2019']s?$", "", m.group(1)).strip(" .,-")
        words = [w.lower() for w in name.split()]
        if (not name or any(w in _MONTHS or w in _NOT_PLACES or w in _VENUE_WORDS or w.isdigit() for w in words)
                or name.lower() in _COUNTRIES):
            continue
        if _norm_place(name) not in [_norm_place(o) for o in out] and not any(_norm_place(o) in _norm_place(name) for o in out):
            out.append(name)
    return out


def _present(place: str, evidence_text: str, ev_norm: Set[str]) -> bool:
    return _norm_place(place) in ev_norm or re.search(rf"\b{re.escape(place)}\b", evidence_text or "", re.I) is not None


def find_conflicts(claim: str, evidence_text: str) -> List[Dict[str, Any]]:
    """Date and place conflicts between the claim and one evidence text."""
    conflicts: List[Dict[str, Any]] = []

    claim_dates, ev_dates = date_mentions(claim), date_mentions(evidence_text)
    if claim_dates and ev_dates:
        hit = _date_conflict(claim_dates, ev_dates)
        if hit:
            kind, evs = hit
            claim_label = ", ".join(sorted({_fmt(d) for d in claim_dates}))
            ev_label = ", ".join(sorted({_fmt(d) for d in evs}))
            conflicts.append({"type": "date", "kind": kind, "claim": claim_label, "evidence": ev_label,
                              "detail": f"Evidence gives {ev_label} but the claim says {claim_label}"})

    wanted = claim_places(claim)
    if wanted:
        ev_places = [p for p in places_in(evidence_text) if p.lower() not in _COUNTRIES]
        ev_norm = {_norm_place(p) for p in ev_places}
        if ev_places and not any(_present(p, evidence_text, ev_norm) for p in wanted):
            conflicts.append({"type": "location", "claim": ", ".join(wanted), "evidence": ", ".join(ev_places),
                              "detail": f"Evidence places it in {', '.join(ev_places)} but the claim says {', '.join(wanted)}"})
    return conflicts


def unmatched_details(claim: str, evidence_text: str) -> List[str]:
    """Specific claim details (place, month) the evidence does not mention at all."""
    missing: List[str] = []
    ev_norm = {_norm_place(p) for p in places_in(evidence_text)}
    for p in claim_places(claim):
        if not _present(p, evidence_text, ev_norm):
            missing.append(p)
    ev_months = {d["month"] for d in date_mentions(evidence_text)}
    for c in date_mentions(claim):
        if c["month"] not in ev_months:
            missing.append(_label(c["month"], c["year"]))
    return missing


def evidence_conflicts(claim: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Conflicts for one evidence item. None when the item is not about the claimed kind of event
    (e.g. a BRICS trade bazaar in Mumbai says nothing about where the BRICS Summit was held)."""
    try:
        if float(item.get("event_match_score", 1) or 0) <= 0:
            return []
    except (TypeError, ValueError):
        pass
    return find_conflicts(claim, f"{item.get('title', '')} {item.get('snippet', '')}")
