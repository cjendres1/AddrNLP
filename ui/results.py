import streamlit as st
from config.locations import STATE_ABBREVIATIONS


def render_results(df_results):
    if df_results.empty:
        return

    selected_state = df_results.iloc[0]["City"]  # retained only for API shape
    st.success(f"Found {len(df_results)} restaurant candidates.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurants", len(df_results))
    col2.metric("High Confidence", (df_results["Confidence"] == "High").sum())
    col3.metric("Addresses", (df_results["Address"] != "N/A").sum())
    col4.metric("Phone Numbers", (df_results["Phone"] != "N/A").sum())

    st.divider()
    st.subheader(
        f"📊 Parsed Restaurants — {df_results.iloc[0]['City']}, "
        f"{df_results.iloc[0]['State']}"
    )

    display_columns = [
        "Restaurant Name", "Confidence", "Cuisine", "Restaurant Type",
        "Address", "Standardized Address", "City", "State", "ZIP",
        "Phone", "Website / Source",
    ]

    st.dataframe(
        df_results[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Website / Source": st.column_config.LinkColumn(
                "Source", display_text="Open Source"
            ),
            "Confidence": st.column_config.TextColumn("NER Confidence"),
        },
    )
