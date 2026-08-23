import re
import requests
import pandas as pd
import streamlit as st
import spacy
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIG & BRANDING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | Dynamic Address Extractor",
    page_icon="📍",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #4B5563; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📍 AddrNLP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live API Fetcher + spaCy NER + USPS Address Standardization</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD SPACY MODEL (OPTIMIZED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_spacy_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])

nlp = load_spacy_nlp()

# -----------------------------------------------------------------------------
# 3. LIVE OPENSTREETMAP (OVERPASS) API INTEGRATION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache API responses for 1 hour
def fetch_live_restaurants(city: str, state: str, limit: int = 10) -> List[Dict[str, str]]:
    """Fetches real restaurant addresses and contact info using OpenStreetMap's Overpass API."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Overpass QL Query searching for restaurants inside the specified city/state
    query = f"""
    [out:json][timeout:25];
    area["name"="{city}"]->.searchArea;
    (
      node["amenity"="restaurant"](area.searchArea);
      way["amenity"="restaurant"](area.searchArea);
    );
    out tags center {limit};
    """
    
    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=10)
        data = response.json()
        
        results = []
        for elem in data.get('elements', []):
            tags = elem.get('tags', {})
            name = tags.get('name', 'Unknown Restaurant')
            
            # Reconstruct raw address string from tags
            house = tags.get('addr:housenumber', '')
            street = tags.get('addr:street', '')
            postcode = tags.get('addr:postcode', '')
            phone = tags.get('phone', tags.get('contact:phone', ''))
            website = tags.get('website', tags.get('contact:website', ''))
            
            # Compose a realistic messy unstructured raw record
            raw_text = f"Restaurant: {name}, Located at {house} {street}, {city}, {state} {postcode}. Contact: {phone} {website}"
            
            results.append({
                "raw_text": raw_text,
                "api_name": name,
                "api_phone": phone if phone else "N/A"
            })
            
        return results
    except Exception as e:
        st.error(f"Error fetching live API data: {e}")
        return []

# -----------------------------------------------------------------------------
# 4. EXTRACTION & STANDARDIZATION ENGINE
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bAVE\.\b": "AVE",
    r"\bSTREET\b": "ST", r"\bST\.\b": "ST",
    r"\bROAD\b": "RD", r"\bRD\.\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bBLVD\.\b": "BLVD",
    r"\bDRIVE\b": "DR", r"\bDR\.\b": "DR",
    r"\bLANE\b": "LN", r"\bLN\.\b": "LN",
    r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT",
    r"\bNORTH\b": "N", r"\bSOUTH\b": "S", r"\bEAST\b": "E", r"\bWEST\b": "W"
}

REGEX_PATTERNS = {
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
    "state": r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b",
    "phone": r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "url": r"https?://[^\s]+"
}

def process_live_batch(records: List[Dict[str, str]]) -> pd.DataFrame:
    """Runs spaCy NER and Regex cleaning on live API records."""
    texts = [r["raw_text"] for r in records]
    docs = list(nlp.pipe(texts, batch_size=20))
    
    processed = []
    for idx, doc in enumerate(docs):
        text = texts[idx]
        
        # spaCy NER
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        
        # Regex Parsing
        zip_match = re.search(REGEX_PATTERNS["zip_code"], text)
        state_match = re.search(REGEX_PATTERNS["state"], text, re.IGNORECASE)
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        url_match = re.search(REGEX_PATTERNS["url"], text)
        
        # Address Standardization
        std_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            std_text = re.sub(pattern, replacement, std_text, flags=re.IGNORECASE)
            
        clean_addr = re.sub(REGEX_PATTERNS["url"], "", std_text)
        clean_addr = re.sub(r"\s+", " ", clean_addr).strip()
        
        processed.append({
            "ID": idx + 1,
            "Raw API Input": text,
            "spaCy ORG": ", ".join(orgs) if orgs else "N/A",
            "spaCy GPE": ", ".join(gpes) if gpes else "N/A",
            "Extracted Phone": phone_match.group(0) if phone_match else records[idx]["api_phone"],
            "Extracted Website": url_match.group(0) if url_match else "N/A",
            "Regex State": state_match.group(0).upper() if state_match else "N/A",
            "Regex ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized Address": clean_addr
        })
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. DYNAMIC UI & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 Dynamic Location Controls")

# Location Selectors
selected_state = st.sidebar.selectbox("Select State:", ["TX", "CA", "NY", "IL", "FL"])

city_map = {
    "TX": ["Austin", "Houston", "Dallas"],
    "CA": ["Los Angeles", "San Francisco", "San Diego"],
    "NY": ["New York", "Buffalo", "Rochester"],
    "IL": ["Chicago", "Springfield", "Peoria"],
    "FL": ["Miami", "Orlando", "Tampa"]
}

selected_city = st.sidebar.selectbox("Select City:", city_map[selected_state])
record_limit = st.sidebar.slider("Number of Live Records:", min_value=5, max_value=20, value=10)

# Fetch Live Data
with st.spinner(f"Fetching live OpenStreetMap listings for {selected_city}, {selected_state}..."):
    raw_api_data = fetch_live_restaurants(selected_city, selected_state, limit=record_limit)

if raw_api_data:
    df_results = process_live_batch(raw_api_data)
    
    # Metric KPI Header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Records Fetched", len(df_results))
    col2.metric("Orgs Identified (spaCy)", (df_results["spaCy ORG"] != "N/A").sum())
    col3.metric("Phones Extracted", (df_results["Extracted Phone"] != "N/A").sum())
    col4.metric("Valid ZIPs Parsed", (df_results["Regex ZIP"] != "N/A").sum())

    st.divider()

    st.markdown(f"### 📡 Live API Results: {selected_city}, {selected_state}")
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("No live records found for this location. Try selecting another city.")
