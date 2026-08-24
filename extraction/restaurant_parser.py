import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List

import pandas as pd

from config.locations import STATE_ABBREVIATIONS
from nlp.pipeline import load_spacy_pipeline
from nlp.candidates import (
    choose_best_candidate,
    classify_cuisine,
    normalize_name,
)


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
    r"\bAVENUE\b": "AVE",
    r"\bSTREET\b": "ST",
    r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD",
    r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN",
    r"\bCOURT\b": "CT",
    r"\bPLACE\b": "PL",
    r"\bPARKWAY\b": "PKWY",
    r"\bHIGHWAY\b": "HWY",
    r"\bTERRACE\b": "TER",
    r"\bCIRCLE\b": "CIR",
    r"\bTRAIL\b": "TRL",
    r"\bSUITE\b": "STE",
    r"\bAPARTMENT\b": "APT",
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
        r"\bNORTH\b": "N",
        r"\bSOUTH\b": "S",
        r"\bEAST\b": "E",
        r"\bWEST\b": "W",
        r"\bNORTHEAST\b": "NE",
        r"\bNORTHWEST\b": "NW",
        r"\bSOUTHEAST\b": "SE",
        r"\bSOUTHWEST\b": "SW",
    }

    for pattern, replacement in directional_map.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result.strip()

# =============================================================================
# WEB PAGE ENRICHMENT
# =============================================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_page_text(url: str) -> str:
    """
    Fetch visible text from a restaurant webpage.

    This is intentionally lightweight:
    - Only used for promising restaurant candidates.
    - Cached for one hour.
    - Returns empty text if the page cannot be retrieved.
    """

    if not url:
        return ""

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                )
            },
        )

        response.raise_for_status()

        # Don't process unexpectedly large pages.
        html = response.text[:2_000_000]

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Remove content that is unlikely to contain useful business data.
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        return re.sub(
            r"\s+",
            " ",
            text,
        )

    except Exception:
        return ""

def extract_contact_fields(
    text: str,
) -> Dict[str, str]:
    """
    Extract address, ZIP, and phone from arbitrary webpage text
    using the application's existing regex patterns.
    """

    if not text:
        return {
            "Address": "N/A",
            "ZIP": "N/A",
            "Phone": "N/A",
        }

    phone_match = PHONE_PATTERN.search(text)
    zip_match = ZIP_PATTERN.search(text)
    address_match = ADDRESS_PATTERN.search(text)

    phone = (
        phone_match.group(0).strip()
        if phone_match
        else "N/A"
    )

    zip_code = (
        zip_match.group(0).strip()
        if zip_match
        else "N/A"
    )

    raw_address = (
        address_match.group(0).strip()
        if address_match
        else "N/A"
    )

    return {
        "Address": raw_address,
        "ZIP": zip_code,
        "Phone": phone,
    }

def enrich_record_from_webpage(
    record: Dict[str, str],
) -> Dict[str, str]:
    """
    Try to fill missing address/contact fields from the restaurant's
    source webpage.

    Existing values are never overwritten.
    """

    url = record.get(
        "Website / Source",
        "",
    )

    if not url:
        return record

    # Don't fetch a page if we already have everything we need.
    needs_enrichment = any(
        record.get(field, "N/A") == "N/A"
        for field in [
            "Address",
            "Phone",
            "ZIP",
        ]
    )

    if not needs_enrichment:
        return record

    page_text = fetch_page_text(url)

    if not page_text:
        return record

    extracted = extract_contact_fields(
        page_text
    )

    # Only fill missing values.
    for field in [
        "Address",
        "Phone",
        "ZIP",
    ]:

        if (
            record.get(field, "N/A") == "N/A"
            and extracted[field] != "N/A"
        ):
            record[field] = extracted[field]

    # Standardize an address if we found one.
    if (
        record.get("Address", "N/A") != "N/A"
    ):
        record["Standardized Address"] = (
            standardize_address(
                record["Address"]
            )
        )

    return record

def parse_restaurant_record(
    result: Dict[str, str],
    target_city: str,
    target_state: str,
    doc=None,
) -> Dict[str, str]:
    title = result.get("title", "")
    body = result.get("body", "")
    url = result.get("url", "")

    raw_text = f"{title}. {body}. {target_city}, {target_state}."

    business_name, candidate_score, candidate_reason = choose_best_candidate(
        result=result,
        target_city=target_city,
        target_state=target_state,
        doc=doc,
    )

    organizations = [
        ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"
    ] if doc is not None else []

    locations = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"GPE", "LOC"}
    ] if doc is not None else []

    restaurant_types = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "RESTAURANT_TYPE"
    ] if doc is not None else []

    all_entities = [
        f"{ent.text} ({ent.label_})" for ent in doc.ents
    ] if doc is not None else []

    phone_match = PHONE_PATTERN.search(raw_text)
    zip_match = ZIP_PATTERN.search(raw_text)
    address_match = ADDRESS_PATTERN.search(raw_text)

    phone = phone_match.group(0).strip() if phone_match else "N/A"
    zip_code = zip_match.group(0).strip() if zip_match else "N/A"
    raw_address = address_match.group(0).strip() if address_match else "N/A"

    standardized_address = standardize_address(raw_address)

    # User-selected location is authoritative.
    state_abbreviation = STATE_ABBREVIATIONS[target_state]

    cuisine = classify_cuisine(raw_text)

    if candidate_score >= 8:
        confidence = "High"
    elif candidate_score >= 4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "Restaurant Name": business_name,
        "Confidence": confidence,
        "Candidate Score": candidate_score,
        "Candidate Evidence": candidate_reason,
        "Cuisine": cuisine,
        "Restaurant Type": (
            ", ".join(dict.fromkeys(restaurant_types))
            if restaurant_types else "Restaurant"
        ),
        "Address": raw_address,
        "Standardized Address": standardized_address,
        "City": target_city,
        "State": state_abbreviation,
        "ZIP": zip_code,
        "Phone": phone,
        "Website / Source": url,
        "spaCy Organizations": ", ".join(organizations) or "N/A",
        "spaCy Locations": ", ".join(locations) or "N/A",
        "spaCy Entities": " | ".join(all_entities) or "N/A",
        "Search Query": result.get("query", ""),
        "Raw Search Text": raw_text,
    }


def deduplicate_restaurants(records: List[Dict]) -> List[Dict]:
    best_records = {}

    for record in records:
        name = record["Restaurant Name"]

        if name == "Unknown Restaurant" or not name:
            continue

        key = normalize_name(name)
        if not key:
            continue

        existing = best_records.get(key)
        if existing is None:
            best_records[key] = record
            continue

        def quality(item):
            return (
                item["Candidate Score"]
                + (3 if item["Address"] != "N/A" else 0)
                + (2 if item["Phone"] != "N/A" else 0)
                + (1 if item["ZIP"] != "N/A" else 0)
            )

        if quality(record) > quality(existing):
            best_records[key] = record

    return list(best_records.values())


def process_restaurant_results(
    search_results: List[Dict[str, str]],
    target_city: str,
    target_state: str,
    requested_count: int,
) -> pd.DataFrame:

    if not search_results:
        return pd.DataFrame()

    records = []

    # -------------------------------------------------------------------------
    # Phase 1:
    # Search-result NLP / regex extraction
    # -------------------------------------------------------------------------

    for result in search_results:

        record = parse_restaurant_record(
            result,
            target_city,
            target_state,
        )

        records.append(record)

    # -------------------------------------------------------------------------
    # Phase 2:
    # Deduplicate restaurants found across multiple searches.
    # -------------------------------------------------------------------------

    records = deduplicate_restaurants(
        records
    )

    if not records:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Phase 3:
    # Web enrichment
    #
    # Only enrich promising candidates rather than every search result.
    # -------------------------------------------------------------------------

    records.sort(
        key=lambda x: (
            x["Candidate Score"]
            + (
                3
                if x["Address"] != "N/A"
                else 0
            )
            + (
                2
                if x["Phone"] != "N/A"
                else 0
            )
            + (
                1
                if x["ZIP"] != "N/A"
                else 0
            )
        ),
        reverse=True,
    )

    enrichment_limit = min(
        len(records),
        max(
            requested_count * 2,
            10,
        ),
    )

    for i in range(enrichment_limit):

        records[i] = enrich_record_from_webpage(
            records[i]
        )

    # -------------------------------------------------------------------------
    # Phase 4:
    # Deduplicate again because enrichment may give previously incomplete
    # records better contact information.
    # -------------------------------------------------------------------------

    records = deduplicate_restaurants(
        records
    )

    if not records:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Phase 5:
    # Final sort and requested result count.
    # -------------------------------------------------------------------------

    records.sort(
        key=lambda x: (
            x["Candidate Score"]
            + (
                3
                if x["Address"] != "N/A"
                else 0
            )
            + (
                2
                if x["Phone"] != "N/A"
                else 0
            )
            + (
                1
                if x["ZIP"] != "N/A"
                else 0
            )
        ),
        reverse=True,
    )

    return pd.DataFrame(
        records[:requested_count]
    ).reset_index(
        drop=True
    )
