from typing import Dict, List

import pandas as pd

from config.locations import STATE_ABBREVIATIONS
from nlp.pipeline import load_spacy_pipeline
from nlp.candidates import (
    choose_best_candidate,
    classify_cuisine,
    normalize_name,
)
from extraction.addresses import (
    PHONE_PATTERN,
    ZIP_PATTERN,
    ADDRESS_PATTERN,
    standardize_address,
)


def parse_restaurant_record(
    result: Dict[str, str],
    target_city: str,
    target_state: str,
) -> Dict[str, str]:
    title = result.get("title", "")
    body = result.get("body", "")
    url = result.get("url", "")
    search_query = result.get("query", "")

    # The target location is authoritative.
    # Search-result text can contain references to other cities/states.
    target_abbreviation = STATE_ABBREVIATIONS[target_state]

    raw_text = f"{title}. {body}. {target_city}, {target_abbreviation}."

    # Run spaCy exactly once for this result.
    nlp = load_spacy_pipeline()
    doc = nlp(raw_text)

    business_name, candidate_score, candidate_reason = choose_best_candidate(
        result,
        target_city,
        target_state,
        doc=doc,
    )

    organizations = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
    ]

    locations = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"GPE", "LOC"}
    ]

    restaurant_types = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "RESTAURANT_TYPE"
    ]

    all_entities = [
        f"{ent.text} ({ent.label_})"
        for ent in doc.ents
    ]

    phone_match = PHONE_PATTERN.search(raw_text)
    zip_match = ZIP_PATTERN.search(raw_text)
    address_match = ADDRESS_PATTERN.search(raw_text)

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

    standardized_address = standardize_address(raw_address)

    # IMPORTANT:
    # Never infer the restaurant's state from arbitrary search-result text.
    # The state selected by the user is authoritative.
    state_abbreviation = target_abbreviation

    cuisine = classify_cuisine(raw_text)

    restaurant_type = (
        ", ".join(dict.fromkeys(restaurant_types))
        if restaurant_types
        else "Restaurant"
    )

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
        "Restaurant Type": restaurant_type,
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
        "Search Query": search_query,
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

        def quality(record):
            return (
                record["Candidate Score"]
                + (3 if record["Address"] != "N/A" else 0)
                + (2 if record["Phone"] != "N/A" else 0)
                + (1 if record["ZIP"] != "N/A" else 0)
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

    for result in search_results:
        try:
            record = parse_restaurant_record(
                result=result,
                target_city=target_city,
                target_state=target_state,
            )

            records.append(record)

        except Exception:
            # One bad search result should not kill the entire search.
            continue

    records = deduplicate_restaurants(records)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Highest-quality candidates first.
    df = df.sort_values(
        by="Candidate Score",
        ascending=False,
    )

    return df.head(requested_count).reset_index(drop=True)
