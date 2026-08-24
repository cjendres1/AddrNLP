from typing import Dict, List
from urllib.parse import urlparse
import streamlit as st
from ddgs import DDGS


NON_RESTAURANT_SITES = {
    "tripadvisor", "yelp", "opentable", "resy", "facebook", "instagram",
    "ubereats", "doordash", "grubhub", "restaurantguru", "yellowpages",
    "mapquest", "foursquare", "google", "bing", "zomato",
}

GENERIC_RESULT_NAMES = {
    "restaurants", "restaurant", "restaurants near me", "best restaurants",
    "best restaurants near me", "restaurant guide", "restaurant guides",
    "dining", "dining guide", "food", "food guide", "local restaurants",
    "best places to eat",
}


def domain_is_directory(url: str) -> bool:
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    return any(site in domain for site in NON_RESTAURANT_SITES)


def build_search_queries(city: str, state: str) -> List[str]:
    return [
        f"best restaurants in {city} {state}",
        f"restaurants in {city} {state} address phone",
        f"independent restaurants in {city} {state}",
        f"Japanese restaurants in {city} {state}",
        f"Italian restaurants in {city} {state}",
        f"Mexican restaurants in {city} {state}",
        f"seafood restaurants in {city} {state}",
        f"Thai restaurants in {city} {state}",
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_restaurant_search(
    city: str, state: str, requested_count: int
) -> List[Dict[str, str]]:
    queries = build_search_queries(city, state)
    raw_results = []
    results_per_query = max(requested_count, 8)

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    search_results = ddgs.text(query, max_results=results_per_query)
                    for result in search_results:
                        title = result.get("title", "").strip()
                        body = result.get("body", "").strip()
                        url = result.get("href", "").strip()
                        if not title and not body:
                            continue
                        raw_results.append({
                            "title": title,
                            "body": body,
                            "url": url,
                            "query": query,
                        })
                except Exception:
                    continue
    except Exception as exc:
        st.error(f"Live restaurant search failed: {exc}")
        return []

    unique_results = []
    seen = set()

    for result in raw_results:
        key = (
            result["url"].lower().strip()
            if result["url"]
            else result["title"].lower().strip()
        )
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(result)

    return unique_results
