import re
import spacy
import pandas as pd
import streamlit as st
from spacy import displacy
from spacy.pipeline import EntityRuler
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SPACY PIPELINE WITH CUSTOM ENTITY RULER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | Custom NER & Standardization",
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
st.markdown('<div class="sub-title">Custom spaCy EntityRuler Pipeline + USPS Address Standardization</div>', unsafe_allow_html=True)

@st.cache_resource
def load_custom_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    
    # Add custom EntityRuler BEFORE the statistical 'ner' component
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Rules to help spaCy correctly capture restaurant names & phone numbers
    patterns = [
        # Match restaurant name at start of sentence before "located at"
        {"label": "ORG", "pattern": [{"OP": "+"}, {"LOWER": "located"}, {"LOWER": "at"}]},
    ]
    ruler.add_patterns(patterns)
    return nlp

nlp = load_custom_spacy_model()

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
# 3. EXPANDED SEED DATASETS
# -----------------------------------------------------------------------------
CITY_LOCAL_RESTAURANTS = {
    "Austin": [
        ("Titaya's Thai Cuisine", "2700 W Anderson Ln", "Ste 201", "78757", "555-0101", "https://titayasthai.com"),
        ("Maru Japanese Restaurant", "4636 Burnet Rd", "Ste 100", "78756", "555-0102", "https://marujapanese.com"),
        ("The Omelettry", "105 E 53rd St", "Bldg A", "78751", "555-0103", "https://theomelettry.com"),
        ("La Palapa Mexican Restaurant", "6640 E Hwy 290", "Unit 12", "78723", "555-0104", "https://lapalapaatx.com"),
        ("Red Ash Craft Italian", "303 Colorado St", "Ste 200", "78701", "555-0144", "https://redashgrill.com"),
        ("Mandola's Italian Kitchen", "4700 W Guadalupe St", "Ste 12", "78751", "555-0191", "https://mandolas.com"),
        ("Siam Fine Thai Dining", "1100 S Congress Ave", "Ste 10", "78704", "555-0105", "https://siamatx.com"),
        ("Taqueria El General", "120 Grand Ave", "Apt 2B", "78704", "555-0123", "https://taqueriaelgeneral.com"),
        ("Apex Burger Diner", "888 E 42nd St", "Fl 5", "78751", "555-0155", "https://apexburger.com"),
        ("Ocean Prime Steakhouse", "400 Wilshire Blvd", "# 40", "78701", "555-0167", "https://oceanprime.com")
    ],
    "Houston": [
        ("Lost & Found Lounge", "160 W Gray St", "Suite 100", "77019", "832-649-3050", "https://lostandfoundmidtown.com"),
        ("Hugo's Authentic Mexican Cuisine", "1600 Westheimer Rd", "Ste 1", "77006", "713-524-7744", "https://hugosrestaurant.net"),
        ("Kata Robata Japanese & Sushi", "2170 Kirby Dr", "Ste 100", "77019", "713-526-8858", "https://katarobata.com"),
        ("Brenner's on the Bayou", "1 Birdsall St", "Main Fl", "77007", "713-868-4444", "https://brennerssteakhouse.com"),
        ("Aquarium Seafood Restaurant", "410 Bagby St", "Ste 200", "77002", "713-223-3474", "https://downtownaquariumhouston.com")
    ]
}

def get_restaurant_data(city: str, state: str, limit: int) -> List[str]:
    places = CITY_LOCAL_RESTAURANTS.get(city, CITY_LOCAL_RESTAURANTS["Austin"])
    results = []
    for idx in range(min(limit, len(places))):
        name, street, unit, zip_code, phone, web = places[idx]
        text = f"{name} located at {street}, {unit}, {city}, {state} {zip_code}. Call {phone} or visit {web}."
        results.append(text)
    return results

# -----------------------------------------------------------------------------
# 4. PROCESSING ENGINE (NER + REGEX + STANDARDIZATION)
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR", r"\bLANE\b": "LN",
    r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT", r"\bHIGHWAY\b": "HWY"
}

REGEX_PATTERNS = {
    "phone": r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "zip": r"\b\d{5}(?:-\d{4})?\b",
    "state": r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"
}

@st.cache_data
def process_records_with_spacy(raw_texts: List[str], target_city: str, target_state: str) -> pd.DataFrame:
    processed = []
    
    for text in raw_texts:
        # 1. Custom Rules Extraction for Business Name
        business_name = "N/A"
        match = re.match(r"^(.*?)\s+located at", text, re.IGNORECASE)
        if match:
            business_name = match.group(1).strip()
            
        # 2. Run spaCy NLP Doc
        doc = nlp(text)
        
        # Fallback to spaCy ORG if regex rule fails
        if business_name == "N/A":
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG" and not re.search(r"\b\d{5}\b", ent.text)]
            business_name = orgs[0] if orgs else "N/A"

        # 3. Contextual Extraction via Regex
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        zip_match = re.search(REGEX_PATTERNS["zip"], text)
        state_match = re.search(REGEX_PATTERNS["state"], text)
        
        # 4. USPS Address Standardization
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
            "Raw Unstructured Input": text
        })
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. SIDEBAR & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 Location Controls")
selected_state = st.sidebar.selectbox("Select State:", ["TX"])
selected_city = st.sidebar.selectbox("Select City:", list(CITY_LOCAL_RESTAURANTS.keys()))
record_limit = st.sidebar.slider("Number of Restaurants:", min_value=5, max_value=20, value=10)

raw_records = get_restaurant_data(selected_city, selected_state, limit=record_limit)
df_results = process_records_with_spacy(raw_records, selected_city, selected_state)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df_results))
col2.metric("Business Names Extracted", (df_results["Business Name (ORG)"] != "N/A").sum())
col3.metric("Valid Phone #s", (df_results["Extracted Phone"] != "N/A").sum())
col4.metric("Valid ZIP Codes", (df_results["Extracted ZIP"] != "N/A").sum())

st.divider()

# -----------------------------------------------------------------------------
# 6. DISPLAY RESULTS & VISUALIZER
# -----------------------------------------------------------------------------
st.markdown(f"### 📊 Processed Dataset ({selected_city}, {selected_state})")
st.dataframe(df_results, use_container_width=True, hide_index=True)

st.divider()

st.markdown("### 🏷️ Interactive spaCy Visualizer (displaCy)")
st.caption("Live entity sequence tagging powered by spaCy:")

sample_text = raw_records[0]
sample_doc = nlp(sample_text)
html_vis = displacy.render(sample_doc, style="ent", page=False)
st.components.v1.html(html_vis, height=180, scrolling=True)
