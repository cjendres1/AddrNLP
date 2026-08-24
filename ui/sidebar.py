import streamlit as st
from config.locations import STATE_CITIES


def render_sidebar():
    st.sidebar.header("🔍 Search Controls")

    selected_state = st.sidebar.selectbox(
        "State", options=list(STATE_CITIES.keys())
    )

    selected_city = st.sidebar.selectbox(
        "City", options=STATE_CITIES[selected_state]
    )

    record_limit = st.sidebar.slider(
        "Number of restaurants",
        min_value=5, max_value=20, value=10,
    )

    search_button = st.sidebar.button(
        "🔍 Search Restaurants",
        type="primary",
        use_container_width=True,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "The application searches multiple restaurant-oriented queries, "
        "extracts spaCy ORG candidates, filters directory sites, scores "
        "candidates, and then applies regex-based address/phone extraction."
    )

    return selected_state, selected_city, record_limit, search_button
