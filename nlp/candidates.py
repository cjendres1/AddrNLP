import re
from typing import Dict, List, Tuple

from nlp.pipeline import load_spacy_pipeline
from config.taxonomy import CUISINE_TAXONOMY
from search.ddgs_search import NON_RESTAURANT_SITES, GENERIC_RESULT_NAMES


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def is_bad_candidate(name: str) -> bool:
    normalized = normalize_name(name)
    if not normalized or normalized in GENERIC_RESULT_NAMES or len(normalized) < 3:
        return True

    if any(site in normalized for site in NON_RESTAURANT_SITES):
        return True

    bad_phrases = [
        "best restaurants", "top restaurants", "restaurants in",
        "restaurants near", "restaurant guide", "where to eat",
        "places to eat", "things to do",
    ]
    return any(phrase in normalized for phrase in bad_phrases)


def clean_candidate_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip(" |•–—-")).strip()


def classify_cuisine(text: str) -> str:
    lowered = text.lower()
    matches = []
    for cuisine, keywords in CUISINE_TAXONOMY.items():
        if any(keyword in lowered for keyword in keywords):
            matches.append(cuisine)
    return ", ".join(matches) if matches else "General Dining"


def score_candidate(
    candidate: str, result: Dict[str, str], source_text: str
) -> Tuple[int, List[str]]:
    score = 0
    reasons = []
    lowered = source_text.lower()
    candidate_lower = candidate.lower()

    if any(term in lowered for term in [
        "restaurant", "cafe", "café", "bistro", "bar", "grill",
        "steakhouse", "diner", "bakery", "pizzeria", "taqueria",
    ]):
        score += 2
        reasons.append("restaurant context")

    if re.search(r"\b\d{1,6}\s+[A-Za-z0-9.'#&-]+", source_text):
        score += 2
        reasons.append("address evidence")

    if re.search(r"(?:\(\d{3}\)|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}", source_text):
        score += 1
        reasons.append("phone evidence")

    if re.search(r"\b\d{5}(?:-\d{4})?\b", source_text):
        score += 1
        reasons.append("ZIP evidence")

    if candidate_lower in lowered and candidate_lower not in {
        "restaurant", "restaurants", "food", "dining"
    }:
        score += 1
        reasons.append("named business evidence")

    return score, reasons


def extract_restaurant_candidates(
    result: Dict[str, str],
    target_city: str,
    doc=None,
) -> List[Dict]:
    title = result.get("title", "")
    body = result.get("body", "")
    source_text = f"{title}. {body}"

    if doc is None:
        doc = load_spacy_pipeline()(source_text)

    candidates = []

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue

        candidate = clean_candidate_name(ent.text)
        if is_bad_candidate(candidate):
            continue

        score, reasons = score_candidate(candidate, result, source_text)
        candidates.append({
            "candidate": candidate,
            "score": score,
            "reasons": reasons,
            "source": "spaCy ORG",
        })

    if title:
        title_candidate = clean_candidate_name(title)
        if not is_bad_candidate(title_candidate):
            score, reasons = score_candidate(title_candidate, result, source_text)
            reasons.append("search-result title")
            candidates.append({
                "candidate": title_candidate,
                "score": score,
                "reasons": reasons,
                "source": "search title",
            })

        title_parts = re.split(r"\s+[|•–—-]\s+", title)
        for part in title_parts:
            candidate = clean_candidate_name(part)
            if not candidate or is_bad_candidate(candidate):
                continue
            if len(candidate.split()) > 8:
                continue
            score, reasons = score_candidate(candidate, result, source_text)
            score += 1
            reasons.append("title component")
            candidates.append({
                "candidate": candidate,
                "score": score,
                "reasons": reasons,
                "source": "title component",
            })

    unique = {}
    for candidate in candidates:
        key = normalize_name(candidate["candidate"])
        if not key:
            continue
        existing = unique.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            unique[key] = candidate

    return list(unique.values())


def choose_best_candidate(
    result: Dict[str, str],
    target_city: str,
    doc=None,
) -> Tuple[str, int, str]:
    candidates = extract_restaurant_candidates(result, target_city, doc=doc)
    if not candidates:
        return "Unknown Restaurant", 0, "No candidate generated"

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    return best["candidate"], best["score"], "; ".join(best["reasons"])
