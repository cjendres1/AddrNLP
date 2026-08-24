import re

from config.locations import STATE_ABBREVIATIONS


PHONE_PATTERN = re.compile(
    r"""
    (?:
        \(\d{3}\)\s*
        |
        \d{3}[-.\s]
    )?
    \d{3}[-.\s]\d{4}
    """,
    re.VERBOSE,
)

ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")

STATE_PATTERN = re.compile(
    r"\b(?:" + "|".join(STATE_ABBREVIATIONS.values()) + r")\b",
    re.IGNORECASE,
)

ADDRESS_PATTERN = re.compile(
    r"""
    \b
    \d{1,6}
    \s+
    [A-Za-z0-9.'#&-]+
    (?:\s+[A-Za-z0-9.'#&-]+){0,8}
    \s+
    (?:ST|STREET|AVE|AVENUE|RD|ROAD|DR|DRIVE|BLVD|BOULEVARD|
       LN|LANE|CT|COURT|PL|PLACE|PKWY|PARKWAY|HWY|HIGHWAY|
       WAY|TER|TERRACE|CIR|CIRCLE|PIKE|TRL|TRAIL)
    (?:\s+(?:STE|SUITE|APT|UNIT)\s*[A-Za-z0-9-]+)?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

USPS_REPLACEMENTS = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR", r"\bLANE\b": "LN",
    r"\bCOURT\b": "CT", r"\bPLACE\b": "PL", r"\bPARKWAY\b": "PKWY",
    r"\bHIGHWAY\b": "HWY", r"\bTERRACE\b": "TER", r"\bCIRCLE\b": "CIR",
    r"\bTRAIL\b": "TRL", r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT",
    r"\bUNIT\b": "UNIT",
}


def standardize_address(address: str) -> str:
    if not address or address == "N/A":
        return "N/A"

    result = address.upper().strip()
    result = re.sub(r"[.,]+", " ", result)
    result = re.sub(r"\s+", " ", result)

    for pattern, replacement in USPS_REPLACEMENTS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    directional_map = {
        r"\bNORTH\b": "N", r"\bSOUTH\b": "S", r"\bEAST\b": "E",
        r"\bWEST\b": "W", r"\bNORTHEAST\b": "NE", r"\bNORTHWEST\b": "NW",
        r"\bSOUTHEAST\b": "SE", r"\bSOUTHWEST\b": "SW",
    }

    for pattern, replacement in directional_map.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result.strip()
