import re
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

# Set page configuration
st.set_page_config(
    page_title="AddrNLP | Entity Extraction & Address Standardization",
    page_icon="📍",
    layout="wide"
)

# Header Section
st.title("📍 AddrNLP")
st.caption("Intelligent Unstructured Text Parsing & USPS Address Standardization Engine")

# -----------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATOR (100 Messy Records)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_messy_addresses() -> pd.DataFrame:
    """Generates 100 realistic, messy raw address strings for extraction and cleaning."""
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
    
    # Variations to expand to 100 items
    names = ["Alice Smith", "Bob Jones", "Charlie Brown", "David Miller", "Emma Wilson", "Frank Wright", "Grace Lee", "Henry Ford", "Isabella Martinez", "Jack Taylor"]
    streets = ["Main St", "Broadway Ave", "Market St", "Washington Rd", "Park Ave", "Oak St", "Pine Rd", "Cedar Ln", "Elm St", "Maple Ave"]
    cities = [("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"), ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"), ("Phoenix", "AZ", "85001"), 
              ("Philadelphia", "PA", "19101"), ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"), ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101")]
    units = ["Apt 101", "Suite 200", "Bldg B", "Unit 4", "Floor 3", "Ste 50", "# 12B", "", "", ""]

    records = []
    # Seed with base raw samples
    for i, s in enumerate(raw_samples):
        records.append({"id": i + 1, "raw_text": s})

    # Generate remaining up to 100
    idx = 11
    for name in names:
        for street_idx, street in enumerate(streets):
            city, state, zip_code = cities[street_idx]
            unit = units[(street_idx + len(name)) % len(units)]
            unit_str = f", {unit}" if unit else ""
            
            # Create variations in casing, punctuation, and phone number formats
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
# 2. NLP, REGEX & ADDRESS STANDARDIZATION ENGINE
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
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "street_num": r"\b\d{1,5}\b",
    "attn_person": r"(?:ATTN:|Attn:|Contact:|RE:|for|to:)\s*([A-Za-z\.\s]+?)(?=[,\.\(\-]|$)"
}

def extract_and_standardize(text: str) -> Dict[str, Any]:
    """Applies Regex and rule-based NLP extraction to parse and standardize raw text."""
    # Pattern matching / NER-like extraction via Regex
    zip_match = re.search(REGEX_PATTERNS["zip_code"], text)
    state_match = re.search(REGEX_PATTERNS["state"], text, re.IGNORECASE)
    phone_match = re.search(REGEX_PATTERNS["phone"], text)
    email_match = re.search(REGEX_PATTERNS["email"], text)
    person_match = re.search(REGEX_PATTERNS["attn_person"], text)

    zip_code = zip_match.group(0) if zip_match else "N/A"
    state = state_match.group(0).upper() if state_match else "N/A"
    phone = phone_match.group(0) if phone_match else "N/A"
    email = email_match.group(0) if email_match else "N/A"
    person = person_match.group(1).strip() if person_match else "N/A"

    # Address Cleansing and USPS Standardization
    standardized_text = text.upper()
    for pattern, replacement in USPS_STREET_ABBR.items():
        standardized_text = re.sub(pattern, replacement, standardized_text, flags=re.IGNORECASE)
    
    # Strip non-address metadata (phones, emails)
    clean_address = re.sub(REGEX_PATTERNS["email"], "", standardized_text)
    clean_address = re.sub(REGEX_PATTERNS["phone"], "", clean_address)
    clean_address = re.sub(r"\s+", " ", clean_address).strip()

    return {
        "Extracted Entity (Person/Contact)": person,
        "Extracted Phone": phone,
        "Extracted Email": email,
        "State": state,
        "ZIP Code": zip_code,
        "Standardized Clean Address": clean_address
    }

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI LAYOUT
# -----------------------------------------------------------------------------
st.title("📍 NLP Information Extraction & Address Standardization")
st.markdown("""
This dashboard demonstrates practical NLP techniques: **Named Entity & Contact Extraction**, **Regex Pattern Matching**, 
and **Address Cleansing / USPS Standardization** across unstructured address text.
""")

df_raw = generate_messy_addresses()

# Sidebar Controls
st.sidebar.header("Pipeline Settings")
search_term = st.sidebar.text_input("Filter Raw Records:", "")
show_raw_only = st.sidebar.checkbox("Show Only Unprocessed Data", False)

# Data Processing Pipeline
processed_records = []
for _, row in df_raw.iterrows():
    extracted = extract_and_standardize(row["raw_text"])
    processed_records.append({**row, **extracted})

df_processed = pd.DataFrame(processed_records)

if search_term:
    df_processed = df_processed[df_processed["raw_text"].str.contains(search_term, case=False)]

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records Processed", len(df_processed))
col2.metric("Emails Extracted", (df_processed["Extracted Email"] != "N/A").sum())
col3.metric("Phones Extracted", (df_processed["Extracted Phone"] != "N/A").sum())
col4.metric("Valid ZIPs Found", (df_processed["ZIP Code"] != "N/A").sum())

st.divider()

# Main Interactive View
st.subheader("📊 Extracted Dataset & Standardization Results")

if show_raw_only:
    st.dataframe(df_processed[["id", "raw_text"]], use_container_width=True)
else:
    st.dataframe(
        df_processed[[
            "id", "raw_text", "Extracted Entity (Person/Contact)", 
            "Extracted Phone", "Extracted Email", "State", "ZIP Code", "Standardized Clean Address"
        ]],
        use_container_width=True
    )

# Interactive Playground Section
st.divider()
st.subheader("🧪 Interactive Regex & NLP Playground")
st.write("Test the extraction logic live on custom unstructured text:")

user_input = st.text_area("Input Raw Address Text:", 
                          value="Contact SARAH CONNOR at 999 SKYNET DRIVE, SUITE 500, LOS ANGELES CA 90210 - Tel: (310) 555-0199 / email: sconnor@resistance.io")

if user_input:
    res = extract_and_standardize(user_input)
    res_df = pd.DataFrame([res]).T.reset_index()
    res_df.columns = ["Extraction Field", "Parsed Result"]
    st.table(res_df)
