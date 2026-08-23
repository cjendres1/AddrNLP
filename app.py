import re
import requests
import pandas as pd
import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & BRANDING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | Address & Category Extractor",
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
st.markdown('<div class="sub-title">Rule-Based NLP Category Classifier + Contextual Regex Address Parsing</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD SPACY & INITIALIZE PHRASE MATCHER FOR CUISINES
# -----------------------------------------------------------------------------
CUISINE_CATEGORIES = {
    "Italian": ["Italian", "Trattoria", "Pasta", "Pizza", "Pizzeria", "Bistro"],
    "Mexican": ["Mexican", "Tacos", "Taqueria", "Cantina", "Burrito", "Grill"],
    "Steakhouse / Seafood": ["Steakhouse", "Seafood", "Oyster", "Grill", "Prime", "Fish"],
    "Asian / Japanese": ["Sushi", "Ramen", "Thai", "Chinese", "Asian", "Dim Sum", "Noodle"],
    "Cafe & Bakery": ["Cafe", "Coffee", "Bakery", "Espresso", "Roasters", "Roast"],
    "Fast Food / Casual": ["Burgers", "Wings", "Diner", "Deli", "Sandwich", "Fast Food"]
}

@st.cache_resource
def load_spacy_pipeline():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for category, terms in CUISINE_CATEGORIES.items():
        patterns = [nlp.make_doc(text) for text in terms]
        matcher.add(category, patterns)
        
    return nlp, matcher

nlp, matcher = load_spacy_pipeline()

def classify_cuisine(text: str) -> str:
    """Assigns restaurant categories using spaCy PhraseMatcher."""
    doc = nlp(text)
    matches = matcher(doc)
    
    found_categories = set()
    for match_id, start, end in matches:
        category_name = nlp.vocab.strings[match_id]
        found_categories.add(category_name)
        
    return ", ".join(found_categories) if found_categories else "General Dining"

# -----------------------------------------------------------------------------
# 3. FALLBACK DATASET GENERATOR (EXPANDED TO 20 RECORDS)
# -----------------------------------------------------------------------------
def generate_fallback_data(city: str, state: str, limit: int) -> List[Dict[str, str]]:
    sample_places = [
        ("Mandola's Italian Kitchen", "4700 W Guadalupe St", "Ste 12", "78751", "555-0191", "https://mandolas.com", "Italian"),
        ("Red Ash Craft Italian", "303 Colorado St", "Ste 200", "78701", "555-0144", "https://redashgrill.com", "Italian"),
        ("Intero Farm-to-Table", "2612 E Cesar Chavez St", "Ste 105", "78702", "555-0177", "https://interorestaurant.com", "Italian"),
        ("Bistro De Paris", "333 W Park Ave", "Unit 12", "78701", "555-0188", "https://bistroparis.org", "Italian"),
        ("Apex Burger Diner", "888 E 42nd St", "Fl 5", "78751", "555-0155", "https://apexburger.com", "Fast Food / Casual"),
        ("Taqueria El General", "120 Grand Ave", "Apt 2B", "78704", "555-0123", "https://taqueriaelgeneral.com", "Mexican"),
        ("Ocean Prime Steakhouse", "400 Wilshire Blvd", "# 40", "78701", "555-0167", "https://oceanprime.com", "Steakhouse / Seafood"),
        ("Ramen Tatsu-Ya", "123 S 1st St", "Ste 10", "78704", "555-0111", "https://ramen-tatsuya.com", "Asian / Japanese"),
        ("The Daily Roast Cafe", "555 MLK Jr Blvd", "Apt 4A", "78701", "555-0133", "https://dailyroast.com", "Cafe & Bakery"),
        ("Blue Harbor Seafood", "101 Ocean Dr", "Unit 301", "78701", "555-0167", "https://blueharbor.org", "Steakhouse / Seafood"),
        ("Luigi's Trattoria & Pizzeria", "150 Main St", "Suite B", "78701", "555-0210", "https://luigistrattoria.com", "Italian"),
        ("El Sol Mexican Cantina", "820 S Congress Ave", "Ste 101", "78704", "555-0222", "https://elsolcantina.com", "Mexican"),
        ("Prime Cut Steakhouse", "500 Downtown Blvd", "Floor 2", "78701", "555-0233", "https://primecutsteak.com", "Steakhouse / Seafood"),
        ("Sakura Sushi & Noodle Bar", "901 W 6th St", "Unit 4", "78703", "555-0244", "https://sakurasushi.io", "Asian / Japanese"),
        ("Sweet Wheat Bakery & Espresso", "1200 E 11th St", "Ste A", "78702", "555-0255", "https://sweetwheatbakery.com", "Cafe & Bakery"),
        ("Smokey BBQ Pit & Grill", "3400 E MLK Jr Blvd", "Apt 1C", "78702", "555-0266", "https://smokeybbqpit.com", "Fast Food / Casual"),
        ("Caffè Milano Espresso Bar", "220 Lamar Blvd", "Ste 300", "78704", "555-0277", "https://caffemilano.org", "Cafe & Bakery"),
        ("Taco Town Express", "4500 N IH 35", "Bldg 3", "78751", "555-0288", "https://tacotownexpress.com", "Mexican"),
        ("Golden Dragon Chinese & Dim Sum", "6700 N Lamar Blvd", "Ste 108", "78752", "555-0299", "https://goldendragonatx.com", "Asian / Japanese"),
        ("Coastline Fish & Oyster Bar", "1100 Barton Springs Rd", "Ste 5", "78704", "555-0311", "https://coastlineoyster.com", "Steakhouse / Seafood")
    ]
    
    results = []
    for idx in range(min(limit, len(sample_places))):
        name, street, unit, zip_code, phone, web, cuisine = sample_places[idx]
        raw_text = f"Restaurant Name: {name} ({cuisine}), Address: {street}, {unit}, {city}, {state} {zip_code}. Contact: {phone} Website: {web}"
        results.append({
            "raw_text": raw_text,
            "api_name": name,
            "api_phone": phone
        })
    return results

# -----------------------------------------------------------------------------
# 4. LIVE OPENSTREETMAP API INTEGRATION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_live_restaurants(city: str, state: str, limit: int = 10) -> List[Dict[str, str]]:
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:10];
    area["name"="{city}"]->.searchArea;
    ( node["amenity"="restaurant"](area.searchArea); );
    out tags {limit};
    """
    headers = {"User-Agent": "AddrNLP_Streamlit_App/1.0"}
    
    try:
        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=6)
        if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
            data = response.json()
            elements = data.get('elements', [])
            if elements:
                results = []
                for elem in elements:
                    tags = elem.get('tags', {})
                    name = tags.get('name', 'Unknown Restaurant')
                    cuisine = tags.get('cuisine', 'General Dining').title()
                    house = tags.get('addr:housenumber', '100')
                    street = tags.get('addr:street', 'Main St')
                    postcode = tags.get('addr:postcode', '78701')
                    phone = tags.get('phone', tags.get('contact:phone', 'N/A'))
                    website = tags.get('website', tags.get('contact:website', ''))
                    
                    raw_text = f"Restaurant: {name} ({cuisine} Cuisine), Address: {house} {street}, {city}, {state} {postcode}. Phone: {phone} Web: {website}"
                    results.append({"raw_text": raw_text, "api_name": name, "api_phone": phone})
                return results
    except Exception:
        pass
        
    return generate_fallback_data(city, state, limit)

# -----------------------------------------------------------------------------
# 5. CONTEXT-AWARE EXTRACTION & STANDARDIZATION ENGINE
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bAVE\.\b": "AVE",
    r"\bSTREET\b": "ST", r"\bST\.\b": "ST",
    r"\bROAD\b": "RD", r"\bRD\.\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bBLVD\.\b": "BLVD",
    r"\bDRIVE\b": "DR", r"\bDR\.\b": "DR",
    r"\bLANE\b": "LN", r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT"
}

REGEX_PATTERNS = {
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
    "phone": r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "url": r"https?://[^\s]+"
}

ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY"
]

def extract_state_contextual(text: str, target_city: str) -> str:
    """Extracts 2-letter state code using contextual rules."""
    states_pattern = "|".join(ALL_STATES)
    
    city_state_match = re.search(fr"{re.escape(target_city)},?\s+\b({states_pattern})\b", text, re.IGNORECASE)
    if city_state_match:
        return city_state_match.group(1).upper()
        
    state_zip_match = re.search(fr"\b({states_pattern})\b\s+\d{{5}}", text)
    if state_zip_match:
        return state_zip_match.group(1).upper()
        
    strict_match = re.search(fr"\b({states_pattern})\b", text)
    if strict_match:
        return strict_match.group(1)
        
    return "N/A"

def process_live_batch(records: List[Dict[str, str]], target_city: str) -> pd.DataFrame:
    texts = [r["raw_text"] for r in records]
    docs = list(nlp.pipe(texts, batch_size=20))
    
    processed = []
    for idx, doc in enumerate(docs):
        text = texts[idx]
        
        cuisine_type = classify_cuisine(text)
        extracted_state = extract_state_contextual(text, target_city)
        
        zip_match = re.search(REGEX_PATTERNS["zip_code"], text)
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        url_match = re.search(REGEX_PATTERNS["url"], text)
        
        std_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            std_text = re.sub(pattern, replacement, std_text, flags=re.IGNORECASE)
            
        clean_addr = re.sub(REGEX_PATTERNS["url"], "", std_text)
        clean_addr = re.sub(r"\s+", " ", clean_addr).strip()
        
        processed.append({
            "ID": idx + 1,
            "Raw Unstructured Input": text,
            "NLP Cuisine Category": cuisine_type,
            "Extracted Phone": phone_match.group(0) if phone_match else records[idx]["api_phone"],
            "Extracted Website": url_match.group(0) if url_match else "N/A",
            "Contextual State": extracted_state,
            "Regex ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized USPS Address": clean_addr
        })
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 6. DYNAMIC UI & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🌍 Dynamic Location Controls")

selected_state = st.sidebar.selectbox("Select State:", ["TX", "CA", "NY", "IL", "FL"])
city_map = {
    "TX": ["Austin", "Houston", "Dallas"],
    "CA": ["Los Angeles", "San Francisco", "San Diego"],
    "NY": ["New York", "Buffalo", "Rochester"],
    "IL": ["Chicago", "Springfield", "Peoria"],
    "FL": ["Miami", "Orlando", "Tampa"]
}

selected_city = st.sidebar.selectbox("Select City:", city_map[selected_state])
record_limit = st.sidebar.slider("Number of Records:", min_value=5, max_value=20, value=10)

with st.spinner(f"Classifying listings for {selected_city}, {selected_state}..."):
    raw_api_data = fetch_live_restaurants(selected_city, selected_state, limit=record_limit)

df_results = process_live_batch(raw_api_data, selected_city)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records Loaded", len(df_results))
col2.metric("Categories Identified", (df_results["NLP Cuisine Category"] != "General Dining").sum())
col3.metric("Phones Extracted", (df_results["Extracted Phone"] != "N/A").sum())
col4.metric("Valid ZIPs Parsed", (df_results["Regex ZIP"] != "N/A").sum())

st.divider()

st.markdown(f"### 📊 Categorized Location Dataset: {selected_city}, {selected_state}")
st.dataframe(df_results, use_container_width=True)
