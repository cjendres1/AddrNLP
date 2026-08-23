import re
import spacy
import pandas as pd
import streamlit as st
from spacy import displacy
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | NER & Address Standardization",
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
st.markdown('<div class="sub-title">Demonstrating spaCy Named Entity Recognition (NER) & USPS Address Standardization</div>', unsafe_allow_html=True)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -----------------------------------------------------------------------------
# 2. CUISINE TAXONOMY CLASSIFIER
# -----------------------------------------------------------------------------
CUISINE_TAXONOMY = {
    "Japanese": ["japanese", "sushi", "ramen", "izakaya", "teriyaki", "maru", "bento", "hibachi", "sakura"],
    "Thai": ["thai", "pad thai", "curry", "bangkok", "siam", "titaya"],
    "Mexican": ["mexican", "palapa", "taqueria", "tacos", "cantina", "burrito", "hugo"],
    "Breakfast & Brunch": ["omelet", "omelettry", "pancake", "waffle", "brunch", "diner", "bakery", "cafe", "snooze"],
    "Italian": ["italian", "trattoria", "pasta", "pizza", "bistro", "red ash", "coltivare"],
    "Steakhouse & Seafood": ["steak", "steakhouse", "seafood", "oyster", "grill", "prime", "vic & anthony"]
}

def classify_cuisine(text: str) -> str:
    lowered = text.lower()
    matches = [cat for cat, kws in CUISINE_TAXONOMY.items() if any(kw in lowered for kw in kws)]
    return ", ".join(matches) if matches else "General Dining"

# -----------------------------------------------------------------------------
# 3. SAMPLE DATA
# -----------------------------------------------------------------------------
CITY_LOCAL_RESTAURANTS = {
    "Austin": [
        ("Titaya's Thai Cuisine", "2700 W Anderson Ln", "Ste 201", "78757", "555-0101", "https://titayasthai.com"),
        ("Maru Japanese Restaurant", "4636 Burnet Rd", "Ste 100", "78756", "555-0102", "https://marujapanese.com"),
        ("The Omelettry", "105 E 53rd St", "Bldg A", "78751", "555-0103", "https://theomelettry.com"),
        ("La Palapa Mexican Restaurant", "6640 E Hwy 290", "Unit 12", "78723", "555-0104", "https://lapalapaatx.com"),
        ("Red Ash Craft Italian", "303 Colorado St", "Ste 200", "78701", "555-0144", "https://redashgrill.com")
    ],
    "Houston": [
        ("Lost & Found Lounge", "160 W Gray St", "Suite 100", "77019", "832-649-3050", "https://lostandfoundmidtown.com"),
        ("Hugo's Authentic Mexican Cuisine", "1600 Westheimer Rd", "Ste 1", "77006", "713-524-7744", "https://hugosrestaurant.net"),
        ("Kata Robata Japanese & Sushi", "2170 Kirby Dr", "Ste 100", "77019", "713-526-8858", "https://katarobata.com")
    ]
}

def get_restaurant_data(city: str, state: str) -> List[str]:
    places = CITY_LOCAL_RESTAURANTS.get(city, CITY_LOCAL_RESTAURANTS["Austin"])
    return [
        f"{name} located at {street}, {unit}, {city}, {state} {zip_code}. Call {phone} or visit {web}."
        for name, street, unit, zip_code, phone, web in places
    ]

# -----------------------------------------------------------------------------
# 4. SPACY NER & USPS STANDARDIZATION ENGINE
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR", r"\bLANE\b": "LN",
    r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT", r"\bHIGHWAY\b": "HWY"
}

@st.cache_data
def process_records_with_spacy(raw_texts: List[str]) -> pd.DataFrame:
    processed = []
    
    for text in raw_texts:
        doc = nlp(text)
        
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
        facs = [ent.text for ent in doc.ents if ent.label_ == "FAC"]
        
        zip_match = re.search(r"\b\d{5}(?:-\d{4})?\b", text)
        phone_match = re.search(r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", text)
        
        std_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            std_text = re.sub(pattern, replacement, std_text, flags=re.IGNORECASE)
            
        # Reordered columns: Business first, cleaned data middle, raw input last
        processed.append({
            "Business Name (ORG)": orgs[0] if orgs else "N/A",
            "NLP Cuisine": classify_cuisine(text),
            "Location (GPE/LOC)": ", ".join(gpes) if gpes else "N/A",
            "Facility (FAC)": ", ".join(facs) if facs else "N/A",
            "Extracted Phone": phone_match.group(0) if phone_match else "N/A",
            "Extracted ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized USPS Text": std_text,
            "Raw Unstructured Input": text
        })
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. CONTROLS & SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 Location Controls")
selected_state = st.sidebar.selectbox("Select State:", ["TX"])
selected_city = st.sidebar.selectbox("Select City:", list(CITY_LOCAL_RESTAURANTS.keys()))

raw_records = get_restaurant_data(selected_city, selected_state)
df_results = process_records_with_spacy(raw_records)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df_results))
col2.metric("Entities Extracted (ORG)", (df_results["Business Name (ORG)"] != "N/A").sum())
col3.metric("Locations Identified (GPE)", (df_results["Location (GPE/LOC)"] != "N/A").sum())
col4.metric("Standardized Addresses", len(df_results))

st.divider()

# -----------------------------------------------------------------------------
# 6. RESULTS & SPACY VISUALIZER
# -----------------------------------------------------------------------------
st.markdown(f"### 📊 Processed Dataset with spaCy NER ({selected_city}, {selected_state})")

# Hide dataframe index for a cleaner table view
st.dataframe(df_results, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### 🏷️ Interactive spaCy Visualizer (displaCy)")
st.caption("Sequence token highlights generated in real time using spaCy's statistical NER model:")

sample_text = raw_records[0]
sample_doc = nlp(sample_text)
html_vis = displacy.render(sample_doc, style="ent", page=False)
st.components.v1.html(html_vis, height=180, scrolling=True)
