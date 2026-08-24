import streamlit as st
from config.locations import STATE_ABBREVIATIONS

def render_results(
    df_results,
    selected_city: str,
    selected_state: str,
):
    if df_results.empty:
        return


    state_abbreviation = STATE_ABBREVIATIONS[selected_state]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Restaurants",
        len(df_results),
    )

    col2.metric(
        "High Confidence",
        (
            df_results["Confidence"] == "High"
        ).sum(),
    )

    col3.metric(
        "Addresses",
        (
            df_results["Address"] != "N/A"
        ).sum(),
    )

    col4.metric(
        "Phone Numbers",
        (
            df_results["Phone"] != "N/A"
        ).sum(),
    )

    st.divider()

    # IMPORTANT:
    # Use the user's selected location rather than allowing a search
    # result to determine the displayed state.
    st.subheader(
        f"📊 Parsed Restaurants — "
        f"{selected_city}, {state_abbreviation}"
    )

    display_columns = [
        "Restaurant Name",
        "Confidence",
        "Cuisine",
#        "Restaurant Type",
        "Address",
        "Phone",
        "Standardized Address",
        "City",
        "State",
        "ZIP",
        "Website / Source",
    ]

    st.dataframe(
        df_results[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Website / Source": st.column_config.LinkColumn(
                "Source",
                display_text="Open Source",
            ),
            "Confidence": st.column_config.TextColumn(
                "NER Confidence",
            ),
        },
    )
