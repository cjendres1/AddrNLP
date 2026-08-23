import re
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG
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
# 2. CUISINE TAXONOMY CLASSIFIER
# -----------------------------------------------------------------------------
CUISINE_TAXONOMY = {
    "Japanese": ["japanese", "sushi", "ramen", "izakaya", "teriyaki", "maru", "bento", "hibachi", "sakura", "kata robata"],
    "Thai": ["thai", "pad thai", "curry", "bangkok", "siam", "titaya"],
    "Mexican": ["mexican", "palapa", "taqueria", "tacos", "cantina", "burrito", "el ", "la ", "holbox", "sol", "hugo"],
    "Breakfast & Brunch": ["omelet", "omelettry", "pancake", "waffle", "brunch", "breakfast", "diner", "egg", "bakery", "cafe", "coffee", "snooze"],
    "Italian": ["italian", "trattoria", "pasta", "pizza", "pizzeria", "bistro", "milano", "alla vita", "red ash", "coltivare"],
    "Steakhouse & Seafood": ["steak", "steakhouse", "seafood", "oyster", "grill", "prime", "fish", "coastline", "brenners", "aquarium", "vic & anthony"],
    "Fast Food & Casual": ["burger", "burgers", "wings", "deli", "sandwich", "bbq", "smokey", "killen"]
}

def classify_cuisine_fast(text: str) -> str:
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
# 3. INSTANT LOCAL DATASETS (Zero Network Dependency)
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
        ("Taqueria El General", "120 Grand Ave", "Apt 2B", "78704", "555-0123", "https://taqueriaelgeneral.com"),
        ("Apex Burger Diner", "888 E 42nd St", "Fl 5", "78751", "555-0155", "https://apexburger.com"),
        ("Ocean Prime Steakhouse", "400 Wilshire Blvd", "# 40", "78701", "555-0167", "https://oceanprime.com")
    ],
    "Houston": [
        ("Lost & Found Lounge & Grill", "160 W Gray St", "Suite 100", "77019", "832-649-3050", "https://www.lostandfoundmidtown.com"),
        ("Hull & Oak Southern Kitchen", "1070 Dallas St", "Apt 10", "77002", "713-242-8555", "https://hullandoak.com"),
        ("Brenner's on the Bayou Steakhouse", "1 Birdsall St", "Main Fl", "77007", "713-868-4444", "https://www.brennerssteakhouse.com"),
        ("Aquarium Seafood Restaurant", "410 Bagby St", "Ste 200", "77002", "713-223-3474", "https://www.downtownaquariumhouston.com"),
        ("Lucille's Southern Diner", "5512 La Branch St", "Ste A", "77004", "713-568-2505", "https://www.lucilleshouston.net"),
        ("Hugo's Authentic Mexican Cuisine", "1600 Westheimer Rd", "Ste 1", "77006", "713-524-7744", "https://www.hugosrestaurant.net"),
        ("Kata Robata Japanese & Sushi", "2170 Kirby Dr", "Ste 100", "77019", "713-526-8858", "https://www.katarobata.com"),
        ("Coltivare Italian Rustic Bistro", "3320 White Oak Dr", "Unit B", "77007", "713-637-4095", "https://www.agricolehospitality.com"),
        ("Vic & Anthony's Steakhouse", "1510 Texas Ave", "Fl 1", "77002", "713-228-1111", "https://www.vicandanthonys.com"),
        ("Snooze A.M. Eatery Breakfast", "3217 Montrose Blvd", "Ste 100", "77006", "713-574-6710", "https://snoozeeatery.com")
    ],
    "Dallas": [
        ("Pecan Lodge BBQ", "2703 Main St", "Ste 100", "75226", "214-748-8900", "https://pecanlodge.com"),
        ("Uchi Dallas Sushi", "2817 Maple Ave", "Ste 100", "75201", "214-855-5454", "https://uchidallas.com"),
        ("Campisi's Italian Restaurant", "5610 E Mockingbird Ln", "Suite A", "75206", "214-827-0355", "https://campisis.us")
    ]
}

@st.cache_data
def get_restaurant_data(city: str, state: str, limit: int) -> List[Dict[str, str]]:
    places = CITY_LOCAL_RESTAURANTS.get(city, CITY_LOCAL_RESTAURANTS["Austin"])
    results = []
    
    # Loop over seed records instantaneously
    for idx in range(min(limit, len(places))):
        name, street, unit, zip_code, phone, web = places[idx]
        raw_text = f"Restaurant Name: {name}, Address: {street}, {unit}, {city}, {state} {zip_code}. Contact: {phone} Website: {web}"
        results.append({
            "raw_text": raw_text,
            "api_name": name,
            "api_phone": phone
        })
        
    return results

# -----------------------------------------------------------------------------
# 4. EXTRACTION ENGINE
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
        
    return "N/A"

@st.cache_data
def process_live_batch(records: List[Dict[str, str]], target_city: str) -> pd.DataFrame:
    processed = []
    seen_phones = set()
    record_id = 1
    
    for idx, r in enumerate(records):
        text = r["raw_text"]
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        raw_phone = phone_match.group(0) if phone_match else r["api_phone"]
        
        clean_digits = re.sub(r"\D", "", raw_phone)
        if len(clean_digits) >= 10:
            if clean_digits in seen_phones:
                continue
            seen_phones.add(clean_digits)
            
        cuisine_type = classify_cuisine_fast(text)
        extracted_state = extract_state_contextual(text, target_city)
        zip_match = re.search(REGEX_PATTERNS["zip_code"], text)
        url_match = re.search(REGEX_PATTERNS["url"], text)
        
        std_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            std_text = re.sub(pattern, replacement, std_text, flags=re.IGNORECASE)
            
        clean_addr = re.sub(REGEX_PATTERNS["url"], "", std_text)
        clean_addr = re.sub(r"\s+", " ", clean_addr).strip()
        
        processed.append({
            "ID": record_id,
            "Raw Unstructured Input": text,
            "NLP Cuisine Category": cuisine_type,
            "Extracted Phone": raw_phone,
            "Extracted Website": url_match.group(0) if url_match else "N/A",
            "Contextual State": extracted_state,
            "Regex ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized USPS Address": clean_addr
        })
        record_id += 1
        
    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. CONTROLS & STATE LOGIC
# -----------------------------------------------------------------------------
city_map = {
    "TX": ["Austin", "Houston", "Dallas"],
    "CA": ["Los Angeles", "San Francisco", "San Diego"],
    "NY": ["New York", "Buffalo", "Rochester"],
    "IL": ["Chicago", "Springfield", "Peoria"],
    "FL": ["Miami", "Orlando", "Tampa"]
}

if "selected_state" not in st.session_state:
    st.session_state.selected_state = "TX"

if "selected_city" not in st.session_state or st.session_state.selected_city not in city_map[st.session_state.selected_state]:
    st.session_state.selected_city = city_map[st.session_state.selected_state][0]

def on_state_change():
    st.session_state.selected_city = city_map[st.session_state.selected_state][0]

st.sidebar.header("🌍 Dynamic Location Controls")

st.sidebar.selectbox(
    "Select State:",
    options=list(city_map.keys()),
    key="selected_state",
    on_change=on_state_change
)

st.sidebar.selectbox(
    "Select City:",
    options=city_map[st.session_state.selected_state],
    key="selected_city"
)

record_limit = st.sidebar.slider("Number of Records:", min_value=5, max_value=20, value=10)

# -----------------------------------------------------------------------------
# 6. MAIN RENDER
# -----------------------------------------------------------------------------
cur_city = st.session_state.selected_city
cur_state = st.session_state.selected_state

raw_api_data = get_restaurant_data(cur_city, cur_state, limit=record_limit)
df_results = process_live_batch(raw_api_data, cur_city)

classified_count = (df_results["NLP Cuisine Category"] != "General Dining").sum()
unique_categories = df_results[df_results["NLP Cuisine Category"] != "General Dining"]["NLP Cuisine Category"].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", len(df_results))
col2.metric("Classified", classified_count)
col3.metric("Unique Cuisines", unique_categories)
col4.metric("Phones", (df_results["Extracted Phone"] != "N/A").sum())
col5.metric("Valid ZIPs", (df_results["Regex ZIP"] != "N/A").sum())

st.divider()

st.markdown(f"### 📊 Categorized Location Dataset: {cur_city}, {cur_state}")
st.dataframe(df_results, use_container_width=True)
