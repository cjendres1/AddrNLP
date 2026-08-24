import re
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

    # Load the cached spaCy pipeline once, then process all snippets as a batch.
    nlp = load_spacy_pipeline()

    texts = [
        f"{result.get('title', '')}. "
        f"{result.get('body', '')}. "
        f"{target_city}, {target_state}."
        for result in search_results
    ]

    # nlp.pipe() avoids repeatedly entering the spaCy pipeline for each result.
    docs = nlp.pipe(texts, batch_size=32)

    records = []

    for result, doc in zip(search_results, docs):
        try:
            records.append(
                parse_restaurant_record(
                    result=result,
                    target_city=target_city,
                    target_state=target_state,
                    doc=doc,
                )
            )
        except Exception:
            # One malformed result should not terminate the whole search.
            continue

    records = deduplicate_restaurants(records)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(by="Candidate Score", ascending=False)

    return df.head(requested_count).reset_index(drop=True)
