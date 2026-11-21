"""VeriGraph AI — Canonical runtime clock and temporal analysis utility.

Provides a single source of truth for all timestamps in the system.
All backend components MUST use this utility instead of calling
datetime.now(), datetime.utcnow(), datetime.today(), or date.today()
directly.

Time zones used:
- UTC: for internal persistence and API responses
- Asia/Kolkata (IST): for frontend display and Judge temporal context

Fixed time references for deterministic testing:
- Override via environment variables or set_utc_override()/set_ist_override()
- Default: uses real system time
"""

import datetime
import re

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30), "IST")

#: Tolerance in days for treating an event date as "present" rather than past/future.
PRESENT_TOLERANCE_DAYS = 3

#: Relations used across the pipeline. Kept as plain strings so the values
#: serialize directly into JSON and the LLM prompt.
RELATION_PAST = "PAST"
RELATION_PRESENT = "PRESENT"
RELATION_FUTURE = "FUTURE"
RELATION_RANGE = "RANGE"
RELATION_UNKNOWN = "UNKNOWN"

#: Fixed time references for deterministic testing
#: These can be overridden for testing
UTC_NOW_OVERRIDE = None
IST_NOW_OVERRIDE = None


def _utc_now():
    """Get current UTC datetime, with optional override for testing."""
    global UTC_NOW_OVERRIDE
    if UTC_NOW_OVERRIDE is not None:
        return UTC_NOW_OVERRIDE
    return datetime.datetime.now(datetime.timezone.utc)


def _ist_now():
    """Get current Asia/Kolkata (IST) datetime, with optional override for testing."""
    global IST_NOW_OVERRIDE
    if IST_NOW_OVERRIDE is not None:
        return IST_NOW_OVERRIDE
    return datetime.datetime.now(IST)


def utc_now():
    """Get current UTC datetime."""
    return _utc_now()


def india_now():
    """Get current India (IST) datetime - THIS IS THE CANONICAL SOURCE.

    MUST be used for all Judge temporal reasoning, frontend display,
    and any application timestamp that should show Indian Standard Time.
    """
    return _ist_now()


def utc_iso_now():
    """Get current UTC datetime as ISO format string."""
    return utc_now().isoformat()


def india_iso_now():
    """Get current India time as ISO format string."""
    return india_now().isoformat()


def utc_date():
    """Get current UTC date."""
    return utc_now().date()


def india_date():
    """Get current India date."""
    return india_now().date()


def set_utc_override(dt):
    """Set UTC datetime override for testing (must be timezone-aware)."""
    global UTC_NOW_OVERRIDE
    UTC_NOW_OVERRIDE = dt


def set_ist_override(dt):
    """Set India datetime override for testing (must be timezone-aware)."""
    global IST_NOW_OVERRIDE
    IST_NOW_OVERRIDE = dt


def clear_overrides():
    """Clear overrides and use real system time."""
    global UTC_NOW_OVERRIDE, IST_NOW_OVERRIDE
    UTC_NOW_OVERRIDE = None
    IST_NOW_OVERRIDE = None


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

#: Month names used when parsing human-readable dates such as "September 12, 2026".
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def parse_date_string(value):
    """Best-effort parse of a date from mixed input.

    Handles ISO strings ("2026-09-13", "2026-09-13T00:00:00+00:00"),
    human-readable strings ("September 12, 2026", "12 September 2026"),
    and bare years ("2026").

    Returns a datetime.date, or None when nothing can be parsed.
    """
    if value is None:
        return None

    if isinstance(value, datetime.datetime):
        return value.date()

    if isinstance(value, datetime.date):
        return value

    text = str(value).strip()
    if not text:
        return None

    # ISO-like prefix: 2026-09-13
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # "September 12, 2026" / "Sep 12 2026"
    m = re.search(
        r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            try:
                return datetime.date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                return None

    # "12 September 2026"
    m = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            try:
                return datetime.date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None

    # Bare year
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    if m:
        return datetime.date(int(m.group(1)), 1, 1)

    return None


def extract_date_range(text):
    """Extract a start/end date range from text such as "September 12-13, 2026".

    Returns (start_date, end_date) as datetime.date, or (None, None).
    """
    if not text:
        return (None, None)

    source = str(text)

    # "12-13, 2026" or "12 to 13 September 2026" -> range sharing one year/month
    m = re.search(
        r"([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})",
        source, re.IGNORECASE)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            try:
                year = int(m.group(4))
                return (datetime.date(year, month, int(m.group(2))),
                        datetime.date(year, month, int(m.group(3))))
            except ValueError:
                pass

    # "12-13 September 2026"
    m = re.search(
        r"(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})",
        source, re.IGNORECASE)
    if m:
        month = _MONTHS.get(m.group(3).lower())
        if month:
            try:
                year = int(m.group(4))
                return (datetime.date(year, month, int(m.group(1))),
                        datetime.date(year, month, int(m.group(2))))
            except ValueError:
                pass

    # Two distinct ISO dates in the same text
    found = re.findall(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b", source)
    if len(found) >= 2:
        try:
            a = datetime.date(int(found[0][0]), int(found[0][1]), int(found[0][2]))
            b = datetime.date(int(found[1][0]), int(found[1][1]), int(found[1][2]))
            return (min(a, b), max(a, b))
        except ValueError:
            pass

    return (None, None)


def analyze_temporal_relation(event_date=None, event_end_date=None, reference_date=None):
    """Classify an event date relative to the current date.

    Never hard-codes the current year: the reference date defaults to the real
    runtime clock (or the test override).

    Returns a dict:
        {
            "event_date": "2026-09-13" | None,
            "event_end_date": "2026-09-14" | None,
            "current_date": "2026-09-26",
            "relation": "PAST" | "PRESENT" | "FUTURE" | "RANGE" | "UNKNOWN",
            "days_difference": int | None,
            "is_known": bool,
            "note": str,
        }

    A date range spanning the reference date yields RANGE with days_difference
    measured to the nearer endpoint.
    """
    ref = reference_date or utc_now().date()
    if isinstance(ref, datetime.datetime):
        ref = ref.date()

    start = parse_date_string(event_date) if not isinstance(event_date, datetime.date) else event_date
    end = parse_date_string(event_end_date) if event_end_date and not isinstance(event_end_date, datetime.date) else event_end_date

    result = {
        "event_date": start.isoformat() if start else None,
        "event_end_date": end.isoformat() if end else None,
        "current_date": ref.isoformat(),
        "relation": RELATION_UNKNOWN,
        "days_difference": None,
        "is_known": False,
        "note": "",
    }

    if start is None:
        result["note"] = "No parseable event date found."
        return result

    # Explicit range: classify using the whole span.
    if end is not None:
        result["relation"] = RELATION_RANGE
        result["is_known"] = True
        if start <= ref <= end:
            result["days_difference"] = 0
            result["note"] = "Reference date falls inside the event range."
        elif ref < start:
            result["days_difference"] = (start - ref).days
            result["note"] = "Entire event range is in the future."
        else:
            result["days_difference"] = (ref - end).days
            result["note"] = "Entire event range is in the past."
        return result

    delta = (ref - start).days
    result["days_difference"] = delta
    result["is_known"] = True

    if -PRESENT_TOLERANCE_DAYS <= delta <= PRESENT_TOLERANCE_DAYS:
        result["relation"] = RELATION_PRESENT
        result["note"] = f"Event date is within {PRESENT_TOLERANCE_DAYS} days of the current date."
    elif delta > 0:
        result["relation"] = RELATION_PAST
        result["note"] = f"Event occurred {delta} day(s) before the current date."
    else:
        result["relation"] = RELATION_FUTURE
        result["note"] = f"Event is scheduled {abs(delta)} day(s) in the future."

    return result


def analyze_claim_temporality(claim: str, reference_date=None):
    """Analyze the temporal framing of a claim text.

    Distinguishes event date from publication date by preferring an explicit
    range, then an explicit date, and finally falling back to UNKNOWN.
    """
    start, end = extract_date_range(claim)
    if start is not None:
        return analyze_temporal_relation(start, end, reference_date)

    # Month + year with no day ("in September 2026") covers the whole month.
    has_day = (re.search(r"[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}", claim or "")
               or re.search(r"\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[A-Za-z]{3,9}\.?,?\s+\d{4}", claim or "")
               or re.search(r"\d{4}-\d{1,2}-\d{1,2}", claim or ""))
    if not has_day:
        mm = re.search(r"\b([A-Za-z]{3,9})\.?,?\s+(\d{4})\b", claim or "")
        if mm and _MONTHS.get(mm.group(1).lower()):
            import calendar
            month, year = _MONTHS[mm.group(1).lower()], int(mm.group(2))
            first = datetime.date(year, month, 1)
            last = datetime.date(year, month, calendar.monthrange(year, month)[1])
            res = analyze_temporal_relation(first, last, reference_date)
            res["precision"] = "month"
            return res

    single = parse_date_string(claim)
    if single is not None:
        return analyze_temporal_relation(single, None, reference_date)

    return analyze_temporal_relation(None, None, reference_date)


def temporal_summary_text(analysis):
    """Render a temporal analysis dict as a Judge-readable sentence."""
    if not isinstance(analysis, dict):
        return ""

    relation = analysis.get("relation", RELATION_UNKNOWN)
    event_date = analysis.get("event_date")
    current_date = analysis.get("current_date")
    days = analysis.get("days_difference")

    if relation == RELATION_UNKNOWN or not event_date:
        return f"Event date is UNKNOWN. Current date is {current_date}."

    if analysis.get("precision") == "month":
        return (f"The claim gives only a month ({str(event_date)[:7]}); the exact day is not stated, "
                f"so do NOT read it as an event lasting the whole month. "
                f"Current date is {current_date}. {analysis.get('note', '')}")

    if relation == RELATION_RANGE:
        end_date = analysis.get("event_end_date")
        return (f"Event spans {event_date} to {end_date}. "
                f"Relation to current date ({current_date}): {relation}. "
                f"Days difference: {days}.")

    return (f"Event date is {event_date}. Current date is {current_date}. "
            f"Relation: {relation}. Days difference: {days}.")