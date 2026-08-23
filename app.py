import re
import pandas as pd
import streamlit as st
import spacy
from typing import Dict, Any

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & BRANDING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AddrNLP | Address Standardization Engine",
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
st.markdown('<div class="sub-title">Hybrid Pipeline: Fast Regex + Contextual spaCy NER</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED SPACY MODEL (DISABLE UNNEEDED PIPELINE COMPONENTS)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_spacy_nlp():
    """Loads spaCy with only NER enabled for maximum inference speed."""
    try:
        # Disable parser and lemmatizer to speed up inference dramatically
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])

nlp = load_spacy_nlp()

# -----------------------------------------------------------------------------
# 3. SYNTHETIC DATASET GENERATOR
# -----------------------------------------------------------------------------
@st.cache_data
def generate_messy_addresses() -> pd.DataFrame:
    raw_samples = [
        "Contact JOHN DOE at 123 North Main St, Apt 4B, New York, NY 10001, phone 555-0192 or john@example.com",
        "Ship to ACME CORP, 456 WEST 5TH AVENUE, SUITE 100, AUSTIN TX 78701 (RECV: Jane Smith)",
        "deliver to: Dr. Robert Bruce, 789 S. BROADWAY BLVD #12, LOS ANGELES, CA 90014, email: rbruce@med.org",
        "101 Ocean Dr. Unit 301, Miami FL 33139 - Attn: Maria Garcia Ph: 3055550143",
        "555 MARTIN LUTHER KING JR BLVD, SUITE A, ATLANTA GA 30303 (Contact: Sales Dept)",
        "Invoice to Apex Solutions: 888 E. 42nd St, Fl 5, Chicago, IL 60605, USA",
        "Send mail to 1244 ROUTE 9 NORTH, BUILDING B, EDISON NJ 08817. RE: Alice Cooper",
        "333 W Grand Ave, Apt 2C, Chicago, IL 60654. Emergency Contact: 312-555-0188",
        "1234 S 1st Street, Apt 101, Phoenix AZ 85001 - Call Bob @ 6025550199",
        "999 Industrial Pkwy Ste 400, Seattle WA 98101 / Operations Dept",
    ]
    
    names = ["Alice Smith", "Bob Jones", "Charlie Brown", "David Miller", "Emma Wilson", "Frank Wright", "Grace Lee", "Henry Ford", "Isabella Martinez", "Jack Taylor"]
    streets = ["Main St", "Broadway Ave", "Market St", "Washington Rd", "Park Ave", "Oak St", "Pine Rd", "Cedar Ln", "Elm St", "Maple Ave"]
    cities = [("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"), ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"), ("Phoenix", "AZ", "85001"), 
              ("Philadelphia", "PA", "19101"), ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"), ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101")]
    units = ["Apt 101", "Suite 200", "Bldg B", "Unit 4", "Floor 3", "Ste 50", "# 12B", "", "", ""]

    records = []
    for i, s in enumerate(raw_samples):
        records.append({"id": i + 1, "raw_text": s})

    idx = 11
    for name in names:
        for street_idx, street in enumerate(streets):
            city, state, zip_code = cities[street_idx]
            unit = units[(street_idx + len(name)) % len(units)]
            unit_str = f", {unit}" if unit else ""
            
            if idx % 3 == 0:
                raw = f"Delivery for {name}: {idx * 10} {street}{unit_str}, {city}, {state} {zip_code}. Direct line: ({idx*10%800+100})-555-01{idx:02d}"
            elif idx % 3 == 1:
                raw = f"ATTN: {name.upper()} @ {idx * 10} N. {street.upper()} {unit_str.upper()} - {city.upper()} {state} {zip_code} ({name.lower().replace(' ', '.')}@business.org)"
            else:
                raw = f"{idx * 10} S {street.lower()}{unit_str.lower()}, {city}, {state} {zip_code} - RE: {name}"
            
            records.append({"id": idx, "raw_text": raw})
            idx += 1
            if idx > 100:
                break
        if idx > 100:
            break

    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 4. EXTRACTION ENGINE (FAST REGEX + BATCH SPACY)
# -----------------------------------------------------------------------------
USPS_STREET_ABBR = {
    r"\bAVENUE\b": "AVE", r"\bAVE\.\b": "AVE",
    r"\bSTREET\b": "ST", r"\bST\.\b": "ST",
    r"\bROAD\b": "RD", r"\bRD\.\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bBLVD\.\b": "BLVD",
    r"\bDRIVE\b": "DR", r"\bDR\.\b": "DR",
    r"\bLANE\b": "LN", r"\bLN\.\b": "LN",
    r"\bPARKWAY\b": "PKWY", r"\bPKWY\.\b": "PKWY",
    r"\bSUITE\b": "STE", r"\bSTE\.\b": "STE",
    r"\bAPARTMENT\b": "APT", r"\bAPT\.\b": "APT",
    r"\bBUILDING\b": "BLDG", r"\bBLDG\.\b": "BLDG",
    r"\bFLOOR\b": "FL", r"\bFL\.\b": "FL",
    r"\bNORTH\b": "N", r"\bSOUTH\b": "S", r"\bEAST\b": "E", r"\bWEST\b": "W"
}

REGEX_PATTERNS = {
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
    "state": r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b",
    "phone": r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
}

# CACHE THE FULL BATCH PROCESSING RESULT
@st.cache_data
def process_entire_dataset() -> pd.DataFrame:
    """Executes spaCy nlp.pipe() and Regex standardization in a cached single pass."""
    df = generate_messy_addresses()
    texts = df["raw_text"].tolist()
    
    # 1. High-speed batch processing with spaCy nlp.pipe()
    docs = list(nlp.pipe(texts, batch_size=50))
    
    processed = []
    for idx, doc in enumerate(docs):
        text = texts[idx]
        
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

        # 2. Regex parsing
        zip_match = re.search(REGEX_PATTERNS["zip_code"], text)
        state_match = re.search(REGEX_PATTERNS["state"], text, re.IGNORECASE)
        phone_match = re.search(REGEX_PATTERNS["phone"], text)
        email_match = re.search(REGEX_PATTERNS["email"], text)

        # 3. Standardization
        standardized_text = text.upper()
        for pattern, replacement in USPS_STREET_ABBR.items():
            standardized_text = re.sub(pattern, replacement, standardized_text, flags=re.IGNORECASE)
        
        clean_address = re.sub(REGEX_PATTERNS["email"], "", standardized_text)
        clean_address = re.sub(REGEX_PATTERNS["phone"], "", clean_address)
        clean_address = re.sub(r"\s+", " ", clean_address).strip()

        processed.append({
            "id": df.iloc[idx]["id"],
            "raw_text": text,
            "spaCy PERSON": ", ".join(persons) if persons else "N/A",
            "spaCy ORG": ", ".join(orgs) if orgs else "N/A",
            "spaCy GPE": ", ".join(gpes) if gpes else "N/A",
            "Regex Phone": phone_match.group(0) if phone_match else "N/A",
            "Regex Email": email_match.group(0) if email_match else "N/A",
            "Regex State": state_match.group(0).upper() if state_match else "N/A",
            "Regex ZIP": zip_match.group(0) if zip_match else "N/A",
            "Standardized Address": clean_address
        })

    return pd.DataFrame(processed)

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI (INSTANT LOADING)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
search_term = st.sidebar.text_input("Filter Raw Text Records:", "")

# Fetch cached dataset instantly
df_processed = process_entire_dataset()

if search_term:
    df_processed = df_processed[df_processed["raw_text"].str.contains(search_term, case=False)]

# Metric Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Batch Records", len(df_processed))
col2.metric("Persons Detected", (df_processed["spaCy PERSON"] != "N/A").sum())
col3.metric("Orgs Detected", (df_processed["spaCy ORG"] != "N/A").sum())
col4.metric("Valid ZIPs", (df_processed["Regex ZIP"] != "N/A").sum())

st.divider()

st.markdown("### 📊 Extracted & Standardized Dataset")
st.dataframe(df_processed, use_container_width=True)
