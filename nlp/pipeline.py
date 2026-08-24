import streamlit as st
import spacy


@st.cache_resource
def load_spacy_pipeline():
    """Load spaCy once and add deterministic restaurant-type patterns."""
    nlp = spacy.load("en_core_web_sm")

    if "restaurant_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe(
            "entity_ruler",
            name="restaurant_ruler",
            before="ner",
        )

        patterns = [
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [{"LOWER": {"IN": [
                    "restaurant", "cafe", "café", "bistro", "bar", "grill",
                    "steakhouse", "diner", "bakery", "pizzeria", "taqueria",
                    "brewery",
                ]}}],
            },
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [{"LOWER": "coffee"}, {"LOWER": {"IN": ["shop", "house"]}}],
            },
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [
                    {"LOWER": "ice"},
                    {"LOWER": "cream"},
                    {"LOWER": {"IN": ["shop", "parlor"]}},
                ],
            },
        ]
        ruler.add_patterns(patterns)

    return nlp
