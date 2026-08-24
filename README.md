# AddrNLP Restaurant NER

Refactored from the supplied `app.py` into responsibility-based modules.

## Structure

- `app.py` — Streamlit orchestration and page layout
- `config/locations.py` — state/city and abbreviation configuration
- `config/taxonomy.py` — cuisine taxonomy
- `nlp/pipeline.py` — cached, lazy spaCy pipeline
- `nlp/candidates.py` — restaurant candidate generation/scoring
- `search/ddgs_search.py` — DDGS live search and caching
- `extraction/addresses.py` — regex address/phone/ZIP extraction
- `extraction/restaurant_parser.py` — result parsing and deduplication
- `ui/sidebar.py` — sidebar controls
- `ui/results.py` — result table and metrics
- `ui/inspection.py` — NLP/regex inspection and displaCy
- `requirements.txt` — Python dependencies

## Run

Install dependencies, install the spaCy model, then run:

```bash
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Performance changes

1. spaCy is no longer initialized during page startup; it is loaded lazily on the first search.
2. The spaCy model remains cached with `st.cache_resource`.
3. DDGS search results are cached for one hour.
4. Session state is initialized before it is accessed.
5. Each search result is processed through spaCy once rather than twice.
