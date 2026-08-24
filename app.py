import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import spacy
import streamlit as st
from ddgs import DDGS
from spacy import displacy


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AddrNLP | Restaurant NER Demo",
    page_icon="📍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📍 AddrNLP Restaurant Entity Extraction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "Live web search → spaCy NER → regex extraction → address normalization"
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# 2. STATE / CITY LOOKUP
# =============================================================================

STATE_CITIES: Dict[str, List[str]] = {
    "Alabama": ["Birmingham", "Mobile", "Montgomery"],
    "Alaska": ["Anchorage", "Fairbanks", "Juneau"],
    "Arizona": ["Phoenix", "Tucson", "Scottsdale"],
    "Arkansas": ["Little Rock", "Fayetteville", "Fort Smith"],
    "California": ["Los Angeles", "San Diego", "San Francisco"],
    "Colorado": ["Denver", "Colorado Springs", "Boulder"],
    "Connecticut": ["Hartford", "New Haven", "Stamford"],
    "Delaware": ["Wilmington", "Dover", "Newark"],
    "Florida": ["Miami", "Orlando", "Tampa"],
    "Georgia": ["Atlanta", "Savannah", "Augusta"],
    "Hawaii": ["Honolulu", "Hilo", "Kailua"],
    "Idaho": ["Boise", "Idaho Falls", "Nampa"],
    "Illinois": ["Chicago", "Springfield", "Rockford"],
    "Indiana": ["Indianapolis", "Fort Wayne", "Bloomington"],
    "Iowa": ["Des Moines", "Cedar Rapids", "Davenport"],
    "Kansas": ["Wichita", "Topeka", "Overland Park"],
    "Kentucky": ["Louisville", "Lexington", "Bowling Green"],
    "Louisiana": ["New Orleans", "Baton Rouge", "Shreveport"],
    "Maine": ["Portland", "Bangor", "Augusta"],
    "Maryland": ["Baltimore", "Annapolis", "Frederick"],
    "Massachusetts": ["Boston", "Worcester", "Springfield"],
    "Michigan": ["Detroit", "Grand Rapids", "Ann Arbor"],
    "Minnesota": ["Minneapolis", "Saint Paul", "Duluth"],
    "Mississippi": ["Jackson", "Gulfport", "Hattiesburg"],
    "Missouri": ["St. Louis", "Kansas City", "Springfield"],
    "Montana": ["Billings", "Missoula", "Bozeman"],
    "Nebraska": ["Omaha", "Lincoln", "Bellevue"],
    "Nevada": ["Las Vegas", "Reno", "Henderson"],
    "New Hampshire": ["Manchester", "Nashua", "Concord"],
    "New Jersey": ["Newark", "Jersey City", "Trenton"],
    "New Mexico": ["Albuquerque", "Santa Fe", "Las Cruces"],
    "New York": ["New York City", "Buffalo", "Rochester"],
    "North Carolina": ["Charlotte", "Raleigh", "Durham"],
    "North Dakota": ["Fargo", "Bismarck", "Grand Forks"],
    "Ohio": ["Columbus", "Cleveland", "Cincinnati"],
    "Oklahoma": ["Oklahoma City", "Tulsa", "Norman"],
    "Oregon": ["Portland", "Eugene", "Salem"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Harrisburg"],
    "Rhode Island": ["Providence", "Newport", "Warwick"],
    "South Carolina": ["Charleston", "Columbia", "Greenville"],
    "South Dakota": ["Sioux Falls", "Rapid City", "Pierre"],
    "Tennessee": ["Nashville", "Memphis", "Knoxville"],
    "Texas": ["Austin", "Houston", "Dallas"],
    "Utah": ["Salt Lake City", "Provo", "Ogden"],
    "Vermont": ["Burlington", "Montpelier", "Rutland"],
    "Virginia": ["Richmond", "Virginia Beach", "Arlington"],
    "Washington": ["Seattle", "Spokane", "Tacoma"],
    "West Virginia": ["Charleston", "Morgantown", "Huntington"],
    "Wisconsin": ["Milwaukee", "Madison", "Green Bay"],
    "Wyoming": ["Cheyenne", "Casper", "Laramie"],
}

STATE_ABBREVIATIONS: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


# =============================================================================
# 3. NLP CONFIGURATION
# =============================================================================

@st.cache_resource
def load_spacy_pipeline():
    """
    Load spaCy once and add deterministic restaurant-type patterns.
    """

    nlp = spacy.load("en_core_web_sm")

    if "restaurant_ruler" not in nlp.pipe_names:

        ruler = nlp.add_pipe(
            "entity_ruler",
            name="restaurant_ruler",
            before="ner",
        )

        patterns = [
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "restaurant",
                                "cafe",
                                "café",
                                "bistro",
                                "bar",
                                "grill",
                                "steakhouse",
                                "diner",
                                "bakery",
                                "pizzeria",
                                "taqueria",
                                "brewery",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [
                    {"LOWER": "coffee"},
                    {"LOWER": {"IN": ["shop", "house"]}},
                ],
            },
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [
                    {"LOWER": "ice"},
                    {"LOWER": "cream"},
                    {"LOWER": {"IN": ["shop", "parlor"]}},
                ],
            },
        ]

        ruler.add_patterns(patterns)

    return nlp


nlp = load_spacy_pipeline()


# =============================================================================
# 4. REGEX PATTERNS
# =============================================================================

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

ZIP_PATTERN = re.compile(
    r"\b\d{5}(?:-\d{4})?\b"
)

STATE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(STATE_ABBREVIATIONS.values())
    + r")\b",
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
    (?:ST|STREET|
       AVE|AVENUE|
       RD|ROAD|
       DR|DRIVE|
       BLVD|BOULEVARD|
       LN|LANE|
       CT|COURT|
       PL|PLACE|
       PKWY|PARKWAY|
       HWY|HIGHWAY|
       WAY|TER|TERRACE|
       CIR|CIRCLE|
       PIKE|TRL|TRAIL)
    (?:\s+(?:STE|SUITE|APT|UNIT)\s*[A-Za-z0-9-]+)?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =============================================================================
# 5. ADDRESS NORMALIZATION
# =============================================================================

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
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

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
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

    return result.strip()


# =============================================================================
# 6. CUISINE TAXONOMY
# =============================================================================

CUISINE_TAXONOMY = {
    "Japanese": [
        "japanese",
        "sushi",
        "ramen",
        "izakaya",
        "teriyaki",
        "hibachi",
        "tempura",
    ],
    "Thai": [
        "thai",
        "pad thai",
        "bangkok",
        "siam",
    ],
    "Mexican": [
        "mexican",
        "taqueria",
        "taco",
        "tacos",
        "cantina",
        "burrito",
        "enchilada",
    ],
    "Italian": [
        "italian",
        "trattoria",
        "pasta",
        "pizzeria",
        "pizza",
    ],
    "Indian": [
        "indian",
        "curry",
        "tandoori",
        "masala",
        "naan",
        "biryani",
    ],
    "Chinese": [
        "chinese",
        "dim sum",
        "szechuan",
        "sichuan",
        "wok",
        "mandarin",
    ],
    "Korean": [
        "korean",
        "bibimbap",
        "bulgogi",
        "kimchi",
        "korean bbq",
    ],
    "Vietnamese": [
        "vietnamese",
        "pho",
        "banh mi",
    ],
    "French": [
        "french",
        "bistro",
        "brasserie",
        "creperie",
        "crepe",
    ],
    "Mediterranean": [
        "mediterranean",
        "greek",
        "gyro",
        "falafel",
        "hummus",
        "lebanese",
    ],
    "Seafood": [
        "seafood",
        "oyster",
        "lobster",
        "crab",
        "fish",
    ],
    "Steakhouse": [
        "steakhouse",
        "steak",
        "prime rib",
    ],
    "Breakfast & Brunch": [
        "breakfast",
        "brunch",
        "omelet",
        "pancake",
        "waffle",
    ],
    "Bakery / Cafe": [
        "bakery",
        "cafe",
        "café",
        "coffee",
        "espresso",
    ],
}


def classify_cuisine(text: str) -> str:

    lowered = text.lower()

    matches = []

    for cuisine, keywords in CUISINE_TAXONOMY.items():

        if any(keyword in lowered for keyword in keywords):
            matches.append(cuisine)

    return ", ".join(matches) if matches else "General Dining"


# =============================================================================
# 7. SEARCH-SITE FILTERING
# =============================================================================

NON_RESTAURANT_SITES = {
    "tripadvisor",
    "yelp",
    "opentable",
    "resy",
    "facebook",
    "instagram",
    "ubereats",
    "doordash",
    "grubhub",
    "restaurantguru",
    "yellowpages",
    "mapquest",
    "foursquare",
    "google",
    "bing",
    "zomato",
}

GENERIC_RESULT_NAMES = {
    "restaurants",
    "restaurant",
    "restaurants near me",
    "best restaurants",
    "best restaurants near me",
    "restaurant guide",
    "restaurant guides",
    "dining",
    "dining guide",
    "food",
    "food guide",
    "local restaurants",
    "best places to eat",
}


def domain_is_directory(url: str) -> bool:

    if not url:
        return False

    domain = urlparse(url).netloc.lower()

    return any(
        site in domain
        for site in NON_RESTAURANT_SITES
    )


def normalize_name(name: str) -> str:

    name = name.lower().strip()

    name = re.sub(
        r"[^a-z0-9\s]",
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name


def is_bad_candidate(name: str) -> bool:

    normalized = normalize_name(name)

    if not normalized:
        return True

    if normalized in GENERIC_RESULT_NAMES:
        return True

    if len(normalized) < 3:
        return True

    # Directory/publisher names.
    for site in NON_RESTAURANT_SITES:

        if site in normalized:
            return True

    # Obvious non-business result titles.
    bad_phrases = [
        "best restaurants",
        "top restaurants",
        "restaurants in",
        "restaurants near",
        "restaurant guide",
        "where to eat",
        "places to eat",
        "things to do",
    ]

    return any(
        phrase in normalized
        for phrase in bad_phrases
    )


# =============================================================================
# 8. SEARCH STRATEGY
# =============================================================================

def build_search_queries(
    city: str,
    state: str,
) -> List[str]:

    return [
        f"best restaurants in {city} {state}",
        f"restaurants in {city} {state} address phone",
        f"independent restaurants in {city} {state}",
        f"Japanese restaurants in {city} {state}",
        f"Italian restaurants in {city} {state}",
        f"Mexican restaurants in {city} {state}",
        f"seafood restaurants in {city} {state}",
        f"Thai restaurants in {city} {state}",
    ]


def fetch_live_restaurant_search(
    city: str,
    state: str,
    requested_count: int,
) -> List[Dict[str, str]]:

    queries = build_search_queries(
        city,
        state,
    )

    raw_results = []

    # We intentionally retrieve more search results than the user requested.
    # Candidate filtering and deduplication will reduce the final set.
    results_per_query = max(
        requested_count,
        8,
    )

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    search_results = ddgs.text(
                        query,
                        max_results=results_per_query,
                    )

                    for result in search_results:

                        title = (
                            result.get("title", "")
                            .strip()
                        )

                        body = (
                            result.get("body", "")
                            .strip()
                        )

                        url = (
                            result.get("href", "")
                            .strip()
                        )

                        if not title and not body:
                            continue

                        raw_results.append(
                            {
                                "title": title,
                                "body": body,
                                "url": url,
                                "query": query,
                            }
                        )

                except Exception:
                    # One failed query should not terminate the entire search.
                    continue

    except Exception as exc:

        st.error(
            f"Live restaurant search failed: {exc}"
        )

        return []

    # Deduplicate search results by URL/title.
    unique_results = []
    seen = set()

    for result in raw_results:

        key = (
            result["url"].lower().strip()
            if result["url"]
            else result["title"].lower().strip()
        )

        if key in seen:
            continue

        seen.add(key)
        unique_results.append(result)

    return unique_results


# =============================================================================
# 9. RESTAURANT CANDIDATE EXTRACTION
# =============================================================================

def clean_candidate_name(name: str) -> str:

    name = name.strip()

    # Remove common search-result separators.
    name = re.split(
        r"\s+[|•]\s+",
        name,
        maxsplit=1,
    )[0]

    # Remove common trailing descriptors.
    name = re.sub(
        r"\s*[-–—]\s*"
        r"(restaurant|menu|official site|"
        r"official website|"
        r"reviews|hours|"
        r"baltimore|austin|houston|"
        r"new york).*$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name.strip(" -–—|•")


def score_candidate(
    candidate: str,
    result: Dict[str, str],
    source_text: str,
) -> Tuple[int, List[str]]:

    score = 0
    reasons = []

    normalized = normalize_name(candidate)

    if is_bad_candidate(candidate):
        return -100, ["Rejected as directory/generic result"]

    # -------------------------------------------------------------------------
    # Positive evidence
    # -------------------------------------------------------------------------

    if len(candidate.split()) >= 2:
        score += 2
        reasons.append("multi-word business name")

    restaurant_terms = [
        "restaurant",
        "cafe",
        "café",
        "bistro",
        "grill",
        "kitchen",
        "diner",
        "steakhouse",
        "bakery",
        "pizzeria",
        "taqueria",
        "bar",
        "brewery",
    ]

    if any(
        term in source_text.lower()
        for term in restaurant_terms
    ):
        score += 2
        reasons.append("restaurant context")

    if PHONE_PATTERN.search(source_text):
        score += 2
        reasons.append("phone number found")

    if ADDRESS_PATTERN.search(source_text):
        score += 3
        reasons.append("street address found")

    if ZIP_PATTERN.search(source_text):
        score += 1
        reasons.append("ZIP code found")

    cuisine = classify_cuisine(source_text)

    if cuisine != "General Dining":
        score += 2
        reasons.append("cuisine context")

    # -------------------------------------------------------------------------
    # Search-result source
    # -------------------------------------------------------------------------

    if domain_is_directory(
        result.get("url", "")
    ):
        score -= 1
        reasons.append("directory/listing source")

    else:
        score += 2
        reasons.append("non-directory source")

    # -------------------------------------------------------------------------
    # Penalize obviously generic candidates
    # -------------------------------------------------------------------------

    if normalized in {
        "tripadvisor",
        "yelp",
        "opentable",
        "resy",
        "facebook",
        "instagram",
        "restaurant guru",
    }:
        return -100, ["Known directory/social platform"]

    return score, reasons


def extract_restaurant_candidates(
    result: Dict[str, str],
    target_city: str,
) -> List[Dict]:

    title = result.get("title", "")
    body = result.get("body", "")

    source_text = (
        f"{title}. "
        f"{body}"
    )

    doc = nlp(source_text)

    candidates = []

    # -------------------------------------------------------------------------
    # Candidate source 1: spaCy ORG
    # -------------------------------------------------------------------------

    for ent in doc.ents:

        if ent.label_ != "ORG":
            continue

        candidate = clean_candidate_name(
            ent.text
        )

        score, reasons = score_candidate(
            candidate,
            result,
            source_text,
        )

        candidates.append(
            {
                "candidate": candidate,
                "score": score,
                "reasons": reasons,
                "source": "spaCy ORG",
            }
        )

    # -------------------------------------------------------------------------
    # Candidate source 2: Search-result title
    #
    # A title such as:
    #
    #     Clavel Baltimore | Mexican Restaurant
    #
    # can be a useful candidate even if spaCy does not recognize "Clavel"
    # as ORG.
    # -------------------------------------------------------------------------

    if title:

        title_candidate = clean_candidate_name(
            title
        )

        score, reasons = score_candidate(
            title_candidate,
            result,
            source_text,
        )

        score -= 1
        reasons.append(
            "title-derived candidate"
        )

        candidates.append(
            {
                "candidate": title_candidate,
                "score": score,
                "reasons": reasons,
                "source": "search title",
            }
        )

    # -------------------------------------------------------------------------
    # Remove duplicates within this result.
    # -------------------------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = normalize_name(
            candidate["candidate"]
        )

        if not key:
            continue

        existing = unique.get(key)

        if (
            existing is None
            or candidate["score"] > existing["score"]
        ):
            unique[key] = candidate

    return list(unique.values())


# =============================================================================
# 10. CHOOSE BEST RESTAURANT CANDIDATE
# =============================================================================

def choose_best_candidate(
    result: Dict[str, str],
    target_city: str,
) -> Tuple[str, int, str]:

    candidates = extract_restaurant_candidates(
        result,
        target_city,
    )

    if not candidates:
        return (
            "Unknown Restaurant",
            0,
            "No valid restaurant candidate",
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    if best["score"] < 3:
        return (
            "Unknown Restaurant",
            best["score"],
            "Candidate confidence too low",
        )

    return (
        best["candidate"],
        best["score"],
        "; ".join(best["reasons"]),
    )


# =============================================================================
# 11. PARSE A RESTAURANT RECORD
# =============================================================================

def parse_restaurant_record(
    result: Dict[str, str],
    target_city: str,
    target_state: str,
) -> Dict[str, str]:

    title = result.get("title", "")
    body = result.get("body", "")
    url = result.get("url", "")

    raw_text = (
        f"{title}. "
        f"{body}. "
        f"{target_city}, {target_state}."
    )

    doc = nlp(raw_text)

    # -------------------------------------------------------------------------
    # Restaurant name
    # -------------------------------------------------------------------------

    (
        business_name,
        candidate_score,
        candidate_reason,
    ) = choose_best_candidate(
        result,
        target_city,
    )

    # -------------------------------------------------------------------------
    # spaCy entities
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Regex extraction
    # -------------------------------------------------------------------------

    phone_match = PHONE_PATTERN.search(
        raw_text
    )

    zip_match = ZIP_PATTERN.search(
        raw_text
    )

    address_match = ADDRESS_PATTERN.search(
        raw_text
    )

    state_match = STATE_PATTERN.search(
        raw_text
    )

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

    standardized_address = standardize_address(
        raw_address
    )

    state_abbreviation = (
        state_match.group(0).upper()
        if state_match
        else STATE_ABBREVIATIONS[
            target_state
        ]
    )

    cuisine = classify_cuisine(
        raw_text
    )

    restaurant_type = (
        ", ".join(
            dict.fromkeys(
                restaurant_types
            )
        )
        if restaurant_types
        else "Restaurant"
    )

    # -------------------------------------------------------------------------
    # Confidence classification
    # -------------------------------------------------------------------------

    if candidate_score >= 8:
        confidence = "High"
    elif candidate_score >= 5:
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
        "spaCy Organizations": ", ".join(
            organizations
        ) or "N/A",
        "spaCy Locations": ", ".join(
            locations
        ) or "N/A",
        "spaCy Entities": " | ".join(
            all_entities
        ) or "N/A",
        "Search Query": result.get(
            "query",
            "",
        ),
        "Raw Search Text": raw_text,
    }


# =============================================================================
# 12. DEDUPLICATION
# =============================================================================

def deduplicate_restaurants(
    records: List[Dict],
) -> List[Dict]:

    best_records = {}

    for record in records:

        name = record[
            "Restaurant Name"
        ]

        if name == "Unknown Restaurant":
            continue

        key = normalize_name(name)

        if not key:
            continue

        existing = best_records.get(key)

        if existing is None:
            best_records[key] = record
            continue

        # Keep the record with the strongest extraction evidence.
        existing_score = (
            existing["Candidate Score"]
            + (
                2
                if existing["Address"] != "N/A"
                else 0
            )
            + (
                2
                if existing["Phone"] != "N/A"
                else 0
            )
        )

        new_score = (
            record["Candidate Score"]
            + (
                2
                if record["Address"] != "N/A"
                else 0
            )
            + (
                2
                if record["Phone"] != "N/A"
                else 0
            )
        )

        if new_score > existing_score:
            best_records[key] = record

    return list(
        best_records.values()
    )


# =============================================================================
# 13. PROCESS SEARCH RESULTS
# =============================================================================

def process_restaurant_results(
    search_results: List[Dict[str, str]],
    target_city: str,
    target_state: str,
    requested_count: int,
) -> pd.DataFrame:

    if not search_results:
        return pd.DataFrame()

    records = []

    # Parse all records.
    for result in search_results:

        record = parse_restaurant_record(
            result,
            target_city,
            target_state,
        )

        records.append(record)

    # Deduplicate across search queries.
    records = deduplicate_restaurants(
        records
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Sort strongest records first.
    df = df.sort_values(
        by=[
            "Candidate Score",
            "Confidence",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # Return only the number requested.
    return df.head(
        requested_count
    ).reset_index(drop=True)


# =============================================================================
# 14. SESSION STATE
# =============================================================================

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "df_results" not in st.session_state:
    st.session_state.df_results = pd.DataFrame()

if "last_search" not in st.session_state:
    st.session_state.last_search = None


# =============================================================================
# 15. SIDEBAR CONTROLS
# =============================================================================

st.sidebar.header("🔍 Search Controls")

selected_state = st.sidebar.selectbox(
    "State",
    options=list(
        STATE_CITIES.keys()
    ),
)

selected_city = st.sidebar.selectbox(
    "City",
    options=STATE_CITIES[
        selected_state
    ],
)

record_limit = st.sidebar.slider(
    "Number of restaurants",
    min_value=5,
    max_value=20,
    value=10,
)

search_button = st.sidebar.button(
    "🔍 Search Restaurants",
    type="primary",
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.caption(
    "The application searches multiple restaurant-oriented "
    "queries, extracts spaCy ORG candidates, filters "
    "directory sites, scores candidates, and then applies "
    "regex-based address/phone extraction."
)


# =============================================================================
# 16. SEARCH EXECUTION
# =============================================================================

if search_button:

    state_name = selected_state

    state_abbreviation = (
        STATE_ABBREVIATIONS[
            selected_state
        ]
    )

    search_key = (
        state_name,
        selected_city,
        record_limit,
    )

    with st.spinner(
        f"Searching for restaurants in "
        f"{selected_city}, {state_name}..."
    ):

        # ---------------------------------------------------------------------
        # Retrieve more raw results than requested because the pipeline will
        # filter directory pages and deduplicate restaurants.
        # ---------------------------------------------------------------------

        results = fetch_live_restaurant_search(
            city=selected_city,
            state=state_abbreviation,
            requested_count=record_limit,
        )

        st.session_state.search_results = (
            results
        )

        # ---------------------------------------------------------------------
        # Parse → score → deduplicate → return top N
        # ---------------------------------------------------------------------

        st.session_state.df_results = (
            process_restaurant_results(
                search_results=results,
                target_city=selected_city,
                target_state=state_name,
                requested_count=record_limit,
            )
        )

        st.session_state.last_search = (
            search_key
        )


# =============================================================================
# 17. DISPLAY RESULTS
# =============================================================================

df_results = (
    st.session_state.df_results
)


if not df_results.empty:

    st.success(
        f"Found {len(df_results)} restaurant "
        f"candidates for "
        f"{selected_city}, "
        f"{STATE_ABBREVIATIONS[selected_state]}."
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Restaurants",
        len(df_results),
    )

    col2.metric(
        "High Confidence",
        (
            df_results["Confidence"]
            == "High"
        ).sum(),
    )

    col3.metric(
        "Addresses",
        (
            df_results["Address"]
            != "N/A"
        ).sum(),
    )

    col4.metric(
        "Phone Numbers",
        (
            df_results["Phone"]
            != "N/A"
        ).sum(),
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Main restaurant table
    # -------------------------------------------------------------------------

    st.subheader(
        f"📊 Parsed Restaurants — "
        f"{selected_city}, "
        f"{STATE_ABBREVIATIONS[selected_state]}"
    )

    display_columns = [
        "Restaurant Name",
        "Confidence",
        "Cuisine",
        "Restaurant Type",
        "Address",
        "Standardized Address",
        "City",
        "State",
        "ZIP",
        "Phone",
        "Website / Source",
    ]

    st.dataframe(
        df_results[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Website / Source": st.column_config.LinkColumn(
                "Source",
                display_text="Open Source",
            ),
            "Confidence": st.column_config.TextColumn(
                "NER Confidence"
            ),
        },
    )

    # -------------------------------------------------------------------------
    # NLP inspection
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "🧠 NLP + Regex Inspection"
    )

    selected_restaurant_index = (
        st.selectbox(
            "Select a restaurant to inspect",
            options=list(
                range(
                    len(df_results)
                )
            ),
            format_func=lambda x: (
                f"{x + 1}. "
                f"{df_results.iloc[x]['Restaurant Name']}"
            ),
        )
    )

    selected_record = (
        df_results.iloc[
            selected_restaurant_index
        ]
    )

    detail_col1, detail_col2 = (
        st.columns(2)
    )

    with detail_col1:

        st.markdown(
            "#### spaCy NER"
        )

        st.write(
            "**Restaurant Candidate:**",
            selected_record[
                "Restaurant Name"
            ],
        )

        st.write(
            "**Confidence:**",
            selected_record[
                "Confidence"
            ],
        )

        st.write(
            "**Candidate Evidence:**",
            selected_record[
                "Candidate Evidence"
            ],
        )

        st.write(
            "**ORG entities:**",
            selected_record[
                "spaCy Organizations"
            ],
        )

        st.write(
            "**GPE / LOC entities:**",
            selected_record[
                "spaCy Locations"
            ],
        )

        st.write(
            "**All entities:**",
            selected_record[
                "spaCy Entities"
            ],
        )

    with detail_col2:

        st.markdown(
            "#### Regex Extraction"
        )

        st.write(
            "**Raw Address:**",
            selected_record[
                "Address"
            ],
        )

        st.write(
            "**Standardized Address:**",
            selected_record[
                "Standardized Address"
            ],
        )

        st.write(
            "**Phone:**",
            selected_record[
                "Phone"
            ],
        )

        st.write(
            "**ZIP:**",
            selected_record[
                "ZIP"
            ],
        )

        st.write(
            "**Cuisine:**",
            selected_record[
                "Cuisine"
            ],
        )

    # -------------------------------------------------------------------------
    # displaCy
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "🏷️ spaCy displaCy Visualization"
    )

    selected_raw_text = (
        selected_record[
            "Raw Search Text"
        ]
    )

    selected_doc = nlp(
        selected_raw_text
    )

    html_visualization = (
        displacy.render(
            selected_doc,
            style="ent",
            page=False,
        )
    )

    st.components.v1.html(
        html_visualization,
        height=250,
        scrolling=True,
    )

    # -------------------------------------------------------------------------
    # Raw source text
    # -------------------------------------------------------------------------

    with st.expander(
        "View raw search text"
    ):

        st.text(
            selected_record[
                "Raw Search Text"
            ]
        )

    # -------------------------------------------------------------------------
    # Search provenance
    # -------------------------------------------------------------------------

    with st.expander(
        "View search provenance"
    ):

        st.write(
            "**Search query:**",
            selected_record[
                "Search Query"
            ],
        )

        st.write(
            "**Source:**",
            selected_record[
                "Website / Source"
            ],
        )

    # -------------------------------------------------------------------------
    # Complete record
    # -------------------------------------------------------------------------

    with st.expander(
        "View complete parsed record"
    ):

        st.json(
            selected_record.to_dict()
        )

else:

    st.info(
        "Select a state, city, and number of "
        "restaurants, then click "
        "**Search Restaurants**."
    )

    st.markdown(
        """
        ### NLP pipeline

        This demonstration intentionally separates the responsibilities of
        the NLP and regex components:

        **1. Web search**
        - Retrieve unstructured restaurant-related text
        - Use multiple search queries to increase candidate coverage

        **2. spaCy NER**
        - Identify `ORG` entities
        - Identify `GPE` / `LOC` entities
        - Apply custom `EntityRuler` patterns
        - Generate restaurant-name candidates

        **3. Candidate scoring**
        - Reject directory sites such as Yelp and TripAdvisor
        - Look for restaurant-related context
        - Look for address, phone, ZIP, and cuisine evidence
        - Rank competing ORG candidates

        **4. Regex**
        - Extract street addresses
        - Extract phone numbers
        - Extract ZIP codes
        - Normalize common street suffixes and directions

        **5. Structured output**
        - Deduplicate restaurant candidates
        - Preserve source information
        - Display confidence and extraction evidence

        **Note:** Address normalization is demonstration-oriented and is not
        USPS-certified address validation.
        """
    )
