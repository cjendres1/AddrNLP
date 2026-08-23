import re
import spacy
import pandas as pd
import streamlit as st
from spacy import displacy
from typing import List
from duckduckgo_search import DDGS

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SPACY PIPELINE CACHING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | Live Search & NER Standardization",
    page_icon="📍",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #4B5563; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📍 AddrNLP Live</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Web Search + spaCy Named Entity Recognition & USPS Address Standardization</div>', unsafe_allow_html=True)

# Cache spaCy model in memory so app re-runs remain instant
@st.cache_resource
def load_spacy_pipeline():
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "ORG", "pattern": [{"OP": "+"}, {"LOWER": "located"}, {"LOWER": "at"}]},
    ]
    ruler.add_patterns(patterns)
    return nlp

nlp = load_spacy_pipeline()

# -----------------------------------------------------------------------------
# 2. FAST LIVE SEARCH ENGINE (DuckDuckGo API)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching live web results...")
def fetch_live_restaurant_search(city: str, state: str, limit: int) -> List[str]:
    """Queries live web search for restaurant listings and builds unstructured text streams."""
    query = f"top restaurants in {city} {state} address phone number"
    raw_texts = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=limit * 2))
            
            for item in results:
                title = item.get("title", "").split("-")[0].split("|")[0].strip()
                snippet = item.get("body", "")
                
                # Format into raw unstructured input for our NLP pipeline
                unstructured = f"{title} located in {city}, {state}. Details: {snippet}"
                raw_texts.append(unstructured)
                
                if len(raw_texts) >= limit:
                    break
    except Exception as e:
        st.warning(f"Live search fallback triggered due to rate limit/network: {e}")
        # Graceful fallback if web access fails
        raw_texts = [
            f"Titaya's Thai Cuisine located at 2700 W Anderson Ln, Ste 201, {city}, {state} 78757. Call (512) 555-0101.",
            f"Maru Japanese Restaurant located at 4636 Burnet Rd, {city}, {state} 78756. Call (512) 555-0102."
        ]
        
    return raw_texts

# -----------------------------------------------------------------------------
# 3. CUISINE TAXONOMY CLASSIFIER
# -----------------------------------------------------------------------------
CUISINE_TAXONOMY = {
    "Japanese": ["japanese", "sushi", "ramen", "izakaya", "teriyaki", "bento", "hibachi"],
    "Thai": ["thai", "pad thai", "curry", "bangkok", "siam"],
    "Mexican": ["mexican", "taqueria", "tacos", "cantina", "burrito"],
    "Breakfast & Brunch": ["omelet", "pancake", "waffle", "brunch", "diner", "bakery", "cafe"],
    "Italian": ["italian", "trattoria", "pasta", "pizza", "bistro"],
    "Steakhouse & Seafood": ["steak", "steakhouse", "seafood", "oyster", "grill", "prime"]
}

def classify_cuisine(text: str) -> str:
    lowered = text.lower()
    matches = [cat for cat, kws in CUISINE_TAXONOMY.items() if any(kw in lowered for kw in kws)]
    return ", ".join(matches) if matches else "General Dining"

# -----------------------------------------------------------------------------
# 4. PROCESSING ENGINE (NER + REGEX + STANDARDIZATION)
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR", r"\bLANE\b": "LN",
    r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT", r"\bHIGHWAY\b": "HWY"
}

REGEX_PATTERNS = {
    "phone": r"(?:\(\d{3}\)[-. ]?|\d{3}[-. ]?)?\d{3}[-. ]?\d{4}\b",
    "zip": r"\b\d{5}(?:-\d{4})?\b",
    "state": r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"
}

@st.cache_data
def process_records_with_spacy(raw_texts: List[str], target_city: str, target_state: str) -> pd.DataFrame:
    processed = []
    
    for text in raw_texts:
        doc = nlp(text)
        
        # 1. ORG Entity Extraction via spaCy
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG" and not re.search(r"\b\d{5}\b", ent.text)]
        business_name = orgs[0] if orgs else text.split(" located")[0].strip()
        
        # 2. Contextual Pattern Extraction
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        zip_match = re.search(REGEX_PATTERNS["zip"], text)
        state_match = re.search(REGEX_PATTERNS["state"], text)
        
        # 3. USPS Standardization
        std_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            std_text = re.sub(pattern, replacement, std_text, flags=re.IGNORECASE)
            
        processed.append({
            "Business Name (ORG)": business_name,
            "NLP Cuisine": classify_cuisine(text),
            "State": state_match.group(0) if state_match else target_state,
            "City (GPE)": target_city,
            "Extracted Phone": phone_match.group(0) if phone_match else "N/A",
            "Extracted ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized USPS Text": std_text,
            "Raw Live Input": text
        })
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. CONTROLS & SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Live Search Controls")

selected_state = st.sidebar.selectbox("Select State:", ["TX", "CA", "NY", "FL", "IL", "WA", "FL", "GA"])
selected_city = st.sidebar.text_input("City Name:", value="Austin")

record_limit = st.sidebar.slider("Number of Live Records:", min_value=5, max_value=20, value=10)

# Fetch Live Search & Process
raw_records = fetch_live_restaurant_search(selected_city, selected_state, record_limit)
df_results = process_records_with_spacy(raw_records, selected_city, selected_state)

# Performance Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Live Results Fetched", len(df_results))
col2.metric("Business Names Extracted", (df_results["Business Name (ORG)"] != "N/A").sum())
col3.metric("Valid Phone #s", (df_results["Extracted Phone"] != "N/A").sum())
col4.metric("Valid ZIP Codes", (df_results["Extracted ZIP"] != "N/A").sum())

st.divider()

# -----------------------------------------------------------------------------
# 6. DISPLAY RESULTS & VISUALIZER
# -----------------------------------------------------------------------------
st.markdown(f"### 📊 Live Processed Dataset for **{selected_city}, {selected_state}**")
st.dataframe(df_results, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### 🏷️ Interactive spaCy Visualizer (displaCy)")
st.caption("Real-time token sequence tagging on live web payload:")

if raw_records:
    sample_text = raw_records[0]
    sample_doc = nlp(sample_text)
    html_vis = displacy.render(sample_doc, style="ent", page=False)
    st.components.v1.html(html_vis, height=180, scrolling=True)
