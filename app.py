import streamlit as st

from config.locations import STATE_ABBREVIATIONS
from search.ddgs_search import fetch_live_restaurant_search
from extraction.restaurant_parser import process_restaurant_results
from ui.sidebar import render_sidebar
from ui.results import render_results
from ui.inspection import render_inspection


def initialize_session_state():
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    if "df_results" not in st.session_state:
        import pandas as pd
        st.session_state.df_results = pd.DataFrame()

    if "last_search" not in st.session_state:
        st.session_state.last_search = None


def render_header():
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
        "Live web search → spaCy NER → regex extraction → address normalization"
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="AddrNLP | Restaurant NER Demo",
        page_icon="📍",
        layout="wide",
    )

    # Do not load spaCy here. It is lazy-loaded only when needed.
    initialize_session_state()
    render_header()

    selected_state, selected_city, record_limit, search_button = render_sidebar()

    if search_button:
        state_abbreviation = STATE_ABBREVIATIONS[selected_state]
        search_key = (selected_state, selected_city, record_limit)

        with st.spinner(
            f"Searching for restaurants in {selected_city}, {selected_state}..."
        ):
            results = fetch_live_restaurant_search(
                city=selected_city,
                state=state_abbreviation,
                requested_count=record_limit,
            )

            st.session_state.search_results = results
            st.session_state.df_results = process_restaurant_results(
                search_results=results,
                target_city=selected_city,
                target_state=selected_state,
                requested_count=record_limit,
            )
            st.session_state.last_search = search_key

    df_results = st.session_state.df_results

    if not df_results.empty:
        render_results(df_results)
        render_inspection(df_results)
    elif st.session_state.search_results:
        st.warning(
            "Search results were returned, but no restaurant candidates "
            "survived the extraction process."
        )
        with st.expander("🔎 Debug: View raw search results"):
            st.dataframe(
                st.session_state.search_results,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Select a state, city, and number of restaurants, then click "
            "**Search Restaurants**."
        )
        st.markdown(
            """
            ### NLP pipeline

            **1. Web search**
            - Retrieve unstructured restaurant-related text
            - Use multiple search queries to increase candidate coverage

            **2. spaCy NER**
            - Identify `ORG` entities
            - Identify `GPE` / `LOC` entities
            - Apply custom `EntityRuler` patterns
            - Generate restaurant-name candidates

            **3. Candidate scoring**
            - Reject directory sites such as Yelp and TripAdvisor
            - Look for restaurant-related context
            - Look for address, phone, ZIP, and cuisine evidence
            - Rank competing ORG candidates

            **4. Regex**
            - Extract street addresses
            - Extract phone numbers
            - Extract ZIP codes
            - Normalize common street suffixes and directions

            **5. Structured output**
            - Deduplicate restaurant candidates
            - Preserve source information
            - Display confidence and extraction evidence

            **Note:** Address normalization is demonstration-oriented and is not
            USPS-certified address validation.
            """
        )


if __name__ == "__main__":
    main()
