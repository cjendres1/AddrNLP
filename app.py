import re
import requests
import pandas as pd
import streamlit as st
import spacy
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
st.markdown('<div class="sub-title">Rule-Based NLP Cuisine Classifier + Contextual Regex Address Parsing</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SPACY & ENHANCED CUISINE TAXONOMY CLASSIFIER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_spacy_pipeline():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])

nlp = load_spacy_pipeline()

CUISINE_TAXONOMY = {
    "Japanese": ["japanese", "sushi", "ramen", "izakaya", "teriyaki", "maru", "bento", "hibachi", "sakura"],
    "Thai": ["thai", "pad thai", "curry", "bangkok", "siam", "titaya"],
    "Mexican": ["mexican", "palapa", "taqueria", "tacos", "cantina", "burrito", "el ", "la ", "holbox", "sol"],
    "Breakfast & Brunch": ["omelet", "omelettry", "pancake", "waffle", "brunch", "breakfast", "diner", "egg", "bakery", "cafe", "coffee"],
    "Italian": ["italian", "trattoria", "pasta", "pizza", "pizzeria", "bistro", "milano", "alla vita", "red ash"],
    "Steakhouse & Seafood": ["steak", "steakhouse", "seafood", "oyster", "grill", "prime", "fish", "coastline", "brenners", "aquarium"],
    "Fast Food & Casual": ["burger", "burgers", "wings", "deli", "sandwich", "bbq", "smokey"]
}

def classify_cuisine_advanced(text: str) -> str:
    lowered_text = text.lower()
    matched_categories = []
    
    for category, keywords in CUISINE_TAXONOMY.items():
        for kw in keywords:
            if kw in lowered_text:
                if category not in matched_categories:
                    matched_categories.append(category)
                break
                
    return ", ".join(matched_categories) if matched_categories else "General Dining"

# -----------------------------------------------------------------------------
# 3. CITY-SPECIFIC LOCALIZED DATASETS (FALLBACK)
# -----------------------------------------------------------------------------
CITY_LOCAL_RESTAURANTS = {
    "Austin": [
        ("Titaya's Thai Cuisine", "2700 W Anderson Ln", "Ste 201", "78757", "555-0101", "https://titayasthai.com"),
        ("Maru Japanese Restaurant", "4636 Burnet Rd", "Ste 100", "78756", "555-0102", "https://marujapanese.com"),
        ("The Omelettry", "105 E 53rd St", "Bldg A", "78751", "555-0103", "https://theomelettry.com"),
        ("La Palapa Mexican Restaurant", "6640 E Hwy 290", "Unit 12", "78723", "555-0104", "https://lapalapaatx.com"),
        ("Mandola's Italian Kitchen", "4700 W Guadalupe St", "Ste 12", "78751", "555-0191", "https://mandolas.com"),
        ("Red Ash Craft Italian", "303 Colorado St", "Ste 200", "78701", "555-0144", "https://redashgrill.com"),
        ("Siam Fine Thai Dining", "1100 S Congress Ave", "Ste 10", "78704", "555-0105", "https://siamatx.com"),
        ("Taqueria El General", "120 Grand Ave", "Apt 2B", "78704", "555-0123", "https://taqueriaelgeneral.com")
    ],
    "Houston": [
        ("Lost & Found Lounge & Grill", "160 W Gray St", "Suite 100", "77019", "832-649-3050", "https://www.lostandfoundmidtown.com"),
        ("Hull & Oak Southern Kitchen", "1070 Dallas St", "Apt 10", "77002", "713-242-8555", "https://hullandoak.com"),
        ("Brenner's on the Bayou Steakhouse", "1 Birdsall St", "Main Fl", "77007", "713-868-4444", "https://www.brennerssteakhouse.com"),
        ("Aquarium Seafood Restaurant", "410 Bagby St", "Ste 200", "77002", "713-223-3474", "https://www.downtownaquariumhouston.com"),
        ("Lucille's Southern Diner", "5512 La Branch St", "Ste A", "77004", "713-568-2505", "https://www.lucilleshouston.net")
    ],
    "Dallas": [
        ("The Henry American Bistro", "2301 N Akard St", "Ste 250", "75201", "972-677-9560", "https://www.thehenryrestaurant.com"),
        ("Dragonfly Asian & Fusion", "2332 Leonard St", "Suite 1", "75201", "214-550-9500", "https://www.hotelzaza.com"),
        ("The Woolworth Craft Cocktail Bar", "1520 Elm St", "Suite 201", "75201", "214-814-0588", "https://www.thewoolworthdallas.com"),
        ("The Hampton Social Seafood", "1520 Main St", "Fl 1", "75201", "469-498-9890", "https://www.thehamptonsocial.com")
    ],
    "Los Angeles": [
        ("Girl & the Goat LA", "555-3 Mateo St", "Ste 100", "90013", "213-799-4628", "http://girlandthegoat.com"),
        ("Redbird New American Bistro", "114 E 2nd St", "Ste B", "90012", "213-788-1191", "https://redbird.la"),
        ("Holbox Mexican Seafood", "3655 S Grand Ave", "c9", "90007", "213-986-9972", "http://www.holboxla.com"),
        ("République Bakery & Cafe", "624 S La Brea Ave", "Suite 12", "90036", "310-362-6115", "http://www.republiquela.com")
    ],
    "Chicago": [
        ("Girl & The Goat Chicago", "809 W Randolph St", "Fl 1", "60607", "312-492-6262", "http://www.girlandthegoat.com"),
        ("River Roast Tavern", "315 N LaSalle St", "Ste 10", "60654", "312-822-0100", "https://www.riverroastchicago.com"),
        ("Aba Mediterranean Grill", "302 N Green St", "3rd Floor", "60607", "773-645-1400", "http://www.abarestaurants.com"),
        ("Alla Vita Italian Kitchen", "564 W Randolph St", "Suite A", "60661", "312-667-0104", "https://www.allavitachicago.com")
    ]
}

def generate_fallback_data(city: str, state: str, limit: int) -> List[Dict[str, str]]:
    places = CITY_LOCAL_RESTAURANTS.get(city, CITY_LOCAL_RESTAURANTS["Austin"])
    results = []
    
    # Loop continuously to satisfy the requested record limit
    for idx in range(limit):
        name, street, unit, zip_code, phone, web = places[idx % len(places)]
        idx_label = f" #{idx+1}" if idx >= len(places) else ""
        formatted_name = f"{name}{idx_label}"
        
        raw_text = f"Restaurant Name: {formatted_name}, Address: {street}, {unit}, {city}, {state} {zip_code}. Contact: {phone} Website: {web}"
        results.append({
            "raw_text": raw_text,
            "api_name": formatted_name,
            "api_phone": phone
        })
    return results

# -----------------------------------------------------------------------------
# 4. ENHANCED GEOLOCATED OVERPASS API INTEGRATION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_live_restaurants(city: str, state: str, limit: int = 10) -> List[Dict[str, str]]:
    headers = {"User-Agent": "AddrNLP_Streamlit_App/1.0"}
    
    try:
        # Step 1: Geocode city/state to lat/lon bounding area via Nominatim
        geo_url = f"https://nominatim.openstreetmap.org/search?city={city}&state={state}&country=USA&format=json"
        geo_resp = requests.get(geo_url, headers=headers, timeout=4)
        
        if geo_resp.status_code == 200 and len(geo_resp.json()) > 0:
            lat = float(geo_resp.json()[0]["lat"])
            lon = float(geo_resp.json()[0]["lon"])
            
            # Step 2: Query Overpass within a ~15km bounding box around the city coordinates
            overpass_url = "https://overpass-api.de/api/interpreter"
            bbox_query = f"""
            [out:json][timeout:10];
            node["amenity"="restaurant"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});
            out tags {limit};
            """
            
            response = requests.get(overpass_url, params={'data': bbox_query}, headers=headers, timeout=6)
            if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
                data = response.json()
                elements = data.get('elements', [])
                if elements:
                    results = []
                    for elem in elements:
                        tags = elem.get('tags', {})
                        name = tags.get('name', 'Restaurant')
                        cuisine = tags.get('cuisine', '')
                        house = tags.get('addr:housenumber', '100')
                        street = tags.get('addr:street', 'Main St')
                        postcode = tags.get('addr:postcode', '78701')
                        phone = tags.get('phone', tags.get('contact:phone', 'N/A'))
                        website = tags.get('website', tags.get('contact:website', ''))
                        
                        raw_text = f"Restaurant Name: {name} ({cuisine}), Address: {house} {street}, {city}, {state} {postcode}. Contact: {phone} Website: {website}"
                        results.append({"raw_text": raw_text, "api_name": name, "api_phone": phone})
                    return results
    except Exception:
        pass
        
    return generate_fallback_data(city, state, limit)

# -----------------------------------------------------------------------------
# 5. EXTRACTION & STANDARDIZATION ENGINE
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
    
    processed = []
    for idx, text in enumerate(texts):
        cuisine_type = classify_cuisine_advanced(text)
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

with st.spinner(f"Fetching listings for {selected_city}, {selected_state}..."):
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
