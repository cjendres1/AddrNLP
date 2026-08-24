import streamlit as st
from spacy import displacy
from nlp.pipeline import load_spacy_pipeline


def render_inspection(df_results):
    if df_results.empty:
        return

    st.divider()
    st.subheader("🧠 NLP + Regex Inspection")

    selected_index = st.selectbox(
        "Select a restaurant to inspect",
        options=list(range(len(df_results))),
        format_func=lambda x: (
            f"{x + 1}. {df_results.iloc[x]['Restaurant Name']}"
        ),
    )

    selected_record = df_results.iloc[selected_index]

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("#### spaCy NER")
        st.write("**Restaurant Candidate:**", selected_record["Restaurant Name"])
        st.write("**Confidence:**", selected_record["Confidence"])
        st.write("**Candidate Evidence:**", selected_record["Candidate Evidence"])
        st.write("**ORG entities:**", selected_record["spaCy Organizations"])
        st.write("**GPE / LOC entities:**", selected_record["spaCy Locations"])
        st.write("**All entities:**", selected_record["spaCy Entities"])

    with detail_col2:
        st.markdown("#### Regex Extraction")
        st.write("**Raw Address:**", selected_record["Address"])
        st.write("**Standardized Address:**", selected_record["Standardized Address"])
        st.write("**Phone:**", selected_record["Phone"])
        st.write("**ZIP:**", selected_record["ZIP"])
        st.write("**Cuisine:**", selected_record["Cuisine"])

    st.divider()
    st.subheader("🏷️ spaCy displaCy Visualization")

    selected_raw_text = selected_record["Raw Search Text"]
    doc = load_spacy_pipeline()(selected_raw_text)

    html_visualization = displacy.render(
        doc, style="ent", page=False
    )
    st.components.v1.html(
        html_visualization, height=250, scrolling=True
    )

    with st.expander("View raw search text"):
        st.text(selected_raw_text)

    with st.expander("View search provenance"):
        st.write("**Search query:**", selected_record["Search Query"])
        st.write("**Source:**", selected_record["Website / Source"])

    with st.expander("View complete parsed record"):
        st.json(selected_record.to_dict())
