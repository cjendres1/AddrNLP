import streamlit as st
import en_core_web_sm
import spacy


@st.cache_resource
def load_spacy_pipeline():
    nlp = en_core_web_sm.load()

    if "restaurant_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe(
            "entity_ruler",
            name="restaurant_ruler",
            before="ner",
        )

        patterns = [
            {
                "label": "RESTAURANT_TYPE",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "restaurant",
                                "cafe",
                                "café",
                                "bistro",
                                "bar",
                                "grill",
                                "steakhouse",
                                "diner",
                                "bakery",
                                "pizzeria",
                                "taqueria",
                                "brewery",
                            ]
                        }
                    }
                ],
            }
        ]

        ruler.add_patterns(patterns)

    return nlp
