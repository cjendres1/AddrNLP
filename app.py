import re
from typing import Dict, List

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

    .small-note {
        font-size: 0.85rem;
        color: #6B7280;
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
    "Live restaurant search + spaCy NER + regex address/phone parsing"
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# 2. STATE / CITY LOOKUP
# =============================================================================
#
# Three representative cities per state keeps the demonstration manageable
# while still showing that the UI supports all 50 states.
#
# The dictionary can easily be replaced with a database table or API later.
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
# 3. SPACY PIPELINE
# =============================================================================

@st.cache_resource
def load_spacy_pipeline():
    """
    Load spaCy once per Streamlit session.

    The EntityRuler supplements the statistical NER model with deterministic
    patterns for common restaurant/business language.
    """

    nlp = spacy.load("en_core_web_sm")

    # Avoid adding the ruler multiple times if the cached object is reused.
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
                    {"LOWER": {"IN": [
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
                    ]}}
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
# 4. REGEX DEFINITIONS
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

# Street number + street name + optional suffix/unit.
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
# 5. ADDRESS STANDARDIZATION
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
    """
    Demonstration-oriented USPS-style address normalization.

    This is intentionally regex-based to illustrate text normalization.
    It is not USPS-certified address validation.
    """

    if not address:
        return "N/A"

    result = address.upper().strip()

    # Normalize punctuation.
    result = re.sub(r"[.,]+", " ", result)

    # Collapse whitespace.
    result = re.sub(r"\s+", " ", result)

    # Standardize common street suffixes.
    for pattern, replacement in USPS_REPLACEMENTS.items():
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

    # Standardize common directional prefixes/suffixes.
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
    """
    Rule-based cuisine classification.

    Returns all matching categories rather than forcing a single label.
    """

    lowered = text.lower()

    matches = []

    for cuisine, keywords in CUISINE_TAXONOMY.items():
        if any(keyword in lowered for keyword in keywords):
            matches.append(cuisine)

    return ", ".join(matches) if matches else "General Dining"


# =============================================================================
# 7. SEARCH
# =============================================================================

def fetch_live_restaurant_search(
    city: str,
    state: str,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Retrieve live search results.

    This is deliberately simple for an interview demonstration.
    The search layer could later be replaced with a structured business API.
    """

    query = (
        f"restaurants in {city}, {state} "
        f"address phone cuisine"
    )

    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=limit,
            )

            for item in search_results:
                title = item.get("title", "").strip()
                body = item.get("body", "").strip()
                url = item.get("href", "").strip()

                if not title and not body:
                    continue

                results.append(
                    {
                        "title": title,
                        "body": body,
                        "url": url,
                    }
                )

    except Exception as exc:
        st.error(f"Live search failed: {exc}")
        return []

    return results


# =============================================================================
# 8. RESTAURANT NAME EXTRACTION
# =============================================================================

def extract_restaurant_name(
    doc,
    title: str,
    body: str,
) -> str:
    """
    Use spaCy NER first, then deterministic fallbacks.

    ORG is the primary NER label used for business names.
    """

    # First look for ORG entities.
    organizations = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
        and len(ent.text.strip()) > 2
    ]

    if organizations:
        return organizations[0]

    # EntityRuler restaurant type can help identify a likely title.
    # Remove common separator text.
    candidate = re.split(
        r"\s+(?:located|at|in|on)\s+",
        title,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if candidate:
        return candidate

    return title if title else "Unknown Restaurant"


# =============================================================================
# 9. RECORD PARSER
# =============================================================================

def parse_restaurant_record(
    search_result: Dict[str, str],
    target_city: str,
    target_state: str,
) -> Dict[str, str]:

    title = search_result.get("title", "")
    body = search_result.get("body", "")
    url = search_result.get("url", "")

    raw_text = (
        f"{title}. "
        f"{body}. "
        f"{target_city}, {target_state}."
    )

    # spaCy NER
    doc = nlp(raw_text)

    # -------------------------------------------------------------------------
    # spaCy entities
    # -------------------------------------------------------------------------

    entity_text = [
        f"{ent.text} ({ent.label_})"
        for ent in doc.ents
    ]

    organizations = [
        ent.text
        for ent in doc.ents
        if ent.label_ == "ORG"
    ]

    locations = [
        ent.text
        for ent in doc.ents
        if ent.label_ in {"GPE", "LOC"}
    ]

    restaurant_types = [
        ent.text
        for ent in doc.ents
        if ent.label_ == "RESTAURANT_TYPE"
    ]

    business_name = extract_restaurant_name(
        doc,
        title,
        body,
    )

    # -------------------------------------------------------------------------
    # Regex extraction
    # -------------------------------------------------------------------------

    phone_match = PHONE_PATTERN.search(raw_text)
    zip_match = ZIP_PATTERN.search(raw_text)
    address_match = ADDRESS_PATTERN.search(raw_text)
    state_match = STATE_PATTERN.search(raw_text)

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
        else STATE_ABBREVIATIONS[target_state]
    )

    cuisine = classify_cuisine(raw_text)

    restaurant_type = (
        ", ".join(dict.fromkeys(restaurant_types))
        if restaurant_types
        else "Restaurant"
    )

    return {
        "Restaurant Name": business_name,
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
        "spaCy Entities": " | ".join(entity_text) or "N/A",
        "Raw Search Text": raw_text,
    }


# =============================================================================
# 10. BATCH NLP PROCESSING
# =============================================================================

def process_restaurant_results(
    search_results: List[Dict[str, str]],
    target_city: str,
    target_state: str,
) -> pd.DataFrame:

    if not search_results:
        return pd.DataFrame()

    records = []

    for result in search_results:
        records.append(
            parse_restaurant_record(
                result,
                target_city,
                target_state,
            )
        )

    return pd.DataFrame(records)


# =============================================================================
# 11. SESSION STATE
# =============================================================================

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "df_results" not in st.session_state:
    st.session_state.df_results = pd.DataFrame()

if "last_search" not in st.session_state:
    st.session_state.last_search = None


# =============================================================================
# 12. SIDEBAR CONTROLS
# =============================================================================

st.sidebar.header("🔍 Search Controls")

selected_state = st.sidebar.selectbox(
    "State",
    options=list(STATE_CITIES.keys()),
)

selected_city = st.sidebar.selectbox(
    "City",
    options=STATE_CITIES[selected_state],
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


# =============================================================================
# 13. SEARCH EXECUTION
# =============================================================================

if search_button:

    # Keep both representations:
    #   state_name = human-readable name, e.g. "Texas"
    #   state_abbreviation = standardized code, e.g. "TX"
    state_name = selected_state
    state_abbreviation = STATE_ABBREVIATIONS[selected_state]

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
        # Step 1: Retrieve live unstructured web results
        # ---------------------------------------------------------------------
        results = fetch_live_restaurant_search(
            city=selected_city,
            state=state_abbreviation,
            limit=record_limit,
        )

        # ---------------------------------------------------------------------
        # Step 2: Store raw search results
        # ---------------------------------------------------------------------
        st.session_state.search_results = results

        # ---------------------------------------------------------------------
        # Step 3: Parse results with spaCy + regex
        #
        # Pass the full state name to the parser because it is the
        # user-facing value we want associated with the record.
        # The parser separately extracts/standardizes the abbreviation.
        # ---------------------------------------------------------------------
        st.session_state.df_results = (
            process_restaurant_results(
                results,
                target_city=selected_city,
                target_state=state_name,
            )
        )

        # ---------------------------------------------------------------------
        # Step 4: Remember the parameters used for this search
        # ---------------------------------------------------------------------
        st.session_state.last_search = search_key


# =============================================================================
# 14. DISPLAY RESULTS
# =============================================================================

df_results = st.session_state.df_results


if not df_results.empty:

    st.success(
        f"Processed {len(df_results)} live search results "
        f"for {selected_city}, {selected_state}."
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Search Results",
        len(df_results),
    )

    col2.metric(
        "Restaurants Identified",
        (
            df_results["Restaurant Name"]
            != "Unknown Restaurant"
        ).sum(),
    )

    col3.metric(
        "Phone Numbers",
        (
            df_results["Phone"]
            != "N/A"
        ).sum(),
    )

    col4.metric(
        "Addresses",
        (
            df_results["Address"]
            != "N/A"
        ).sum(),
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Main data table
    # -------------------------------------------------------------------------

    st.subheader(
        f"📊 Parsed Restaurants — "
        f"{selected_city}, {STATE_ABBREVIATIONS[selected_state]}"
    )

    display_columns = [
        "Restaurant Name",
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
        df_results[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Website / Source": st.column_config.LinkColumn(
                "Source",
                display_text="Open Source",
            )
        },
    )

    # -------------------------------------------------------------------------
    # Raw NLP / Regex details
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader("🧠 NLP + Regex Extraction Details")

    selected_restaurant = st.selectbox(
        "Select a result to inspect",
        options=list(range(len(df_results))),
        format_func=lambda x: (
            f"{x + 1}. "
            f"{df_results.iloc[x]['Restaurant Name']}"
        ),
    )

    selected_record = df_results.iloc[selected_restaurant]

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.markdown("#### spaCy NER")

        st.write(
            f"**Organizations (ORG):** "
            f"{selected_record['spaCy Organizations']}"
        )

        st.write(
            f"**Locations (GPE/LOC):** "
            f"{selected_record['spaCy Locations']}"
        )

        st.write(
            f"**Restaurant Types:** "
            f"{selected_record['Restaurant Type']}"
        )

        st.write(
            f"**All detected entities:** "
            f"{selected_record['spaCy Entities']}"
        )

    with detail_col2:

        st.markdown("#### Regex Extraction")

        st.write(
            f"**Address:** "
            f"{selected_record['Address']}"
        )

        st.write(
            f"**Standardized:** "
            f"{selected_record['Standardized Address']}"
        )

        st.write(
            f"**Phone:** "
            f"{selected_record['Phone']}"
        )

        st.write(
            f"**ZIP:** "
            f"{selected_record['ZIP']}"
        )

    # -------------------------------------------------------------------------
    # displaCy visualization
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader("🏷️ spaCy displaCy Entity Visualization")

    selected_raw_text = selected_record["Raw Search Text"]

    selected_doc = nlp(selected_raw_text)

    html_visualization = displacy.render(
        selected_doc,
        style="ent",
        page=False,
    )

    st.components.v1.html(
        html_visualization,
        height=220,
        scrolling=True,
    )

    # -------------------------------------------------------------------------
    # Raw source text
    # -------------------------------------------------------------------------

    with st.expander("View raw search text"):

        st.text(
            selected_record["Raw Search Text"]
        )

    # -------------------------------------------------------------------------
    # Full parsed record
    # -------------------------------------------------------------------------

    with st.expander("View complete parsed record"):

        st.json(
            selected_record.to_dict()
        )

else:

    st.info(
        "Select a state, city, and number of restaurants, "
        "then click **Search Restaurants**."
    )

    st.markdown(
        """
        ### What this demonstration illustrates

        **spaCy**
        - Statistical Named Entity Recognition
        - `ORG` extraction for business names
        - `GPE` / `LOC` extraction
        - Custom `EntityRuler` patterns
        - displaCy visualization

        **Regex**
        - Telephone number extraction
        - ZIP / ZIP+4 extraction
        - Street address extraction
        - Address normalization
        - USPS-style street suffix abbreviations

        **Streamlit**
        - Dynamic state/city controls
        - Session state
        - Cached NLP model
        - Interactive dataframe
        - Interactive NLP inspection
        """
    )
