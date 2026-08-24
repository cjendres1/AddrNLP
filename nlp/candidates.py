import re
from typing import Dict, List, Tuple

from nlp.pipeline import load_spacy_pipeline
from config.taxonomy import CUISINE_TAXONOMY
from search.ddgs_search import (
    NON_RESTAURANT_SITES,
    GENERIC_RESULT_NAMES,
)


# ---------------------------------------------------------------------------
# Generic / non-restaurant result filtering
# ---------------------------------------------------------------------------

GENERIC_WEB_RESULT_PATTERNS = [
    r"^business profiles?$",
    r"^business listings?$",
    r"^order food",
    r"^food delivery",
    r"^delivery near me",
    r"^restaurants?$",
    r"^restaurants? near",
    r"^restaurants? in",
    r"^best restaurants?",
    r"^top restaurants?",
    r"^restaurant guide",
    r"^restaurant directory",
    r"^restaurant reviews?$",
    r"^restaurant listings?$",
    r"^dining guide",
    r"^places to eat",
    r"^where to eat",
    r"^things to do",
    r"^updated ",
    r"^reviews?$",
    r"^menu$",
    r"^menus$",
    r"^locations?$",
    r"^near me$",
    r"^find restaurants?",
    r"^find a restaurant",
    r"^food near me",
    r"^restaurants and dining",
    r"^dining and restaurants",
]

GENERIC_LIST_TITLE_PATTERNS = [
    r"^\(\d{4}\s+guide\)$",
    r"^best\s+.+\s+restaurants?$",
    r"^top\s+\d+\s+best\s+.+",
    r"^top\s+\d+\s+.+\s+in\s+",
    r"^best\s+.+\s+in\s+",
    r"^best\s+.+\s+near\s+",
]

def is_generic_list_title(name: str) -> bool:
    normalized = re.sub(
        r"\s+",
        " ",
        name.strip().lower(),
    )

    return any(
        re.search(pattern, normalized)
        for pattern in GENERIC_LIST_TITLE_PATTERNS
    )

def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def is_generic_web_result(name: str) -> bool:
    normalized = re.sub(
        r"\s+",
        " ",
        name.strip().lower(),
    )

    if not normalized:
        return True

    return any(
        re.search(pattern, normalized)
        for pattern in GENERIC_WEB_RESULT_PATTERNS
    )


def is_bad_candidate(name: str) -> bool:
    normalized = normalize_name(name)

    if not normalized:
        return True

    if len(normalized) < 3:
        return True

    if normalized in GENERIC_RESULT_NAMES:
        return True

    if is_generic_web_result(name):
        return True

    if is_generic_list_title(name):
        return True

    # Search engines frequently truncate page titles.
    if name.endswith("...") or name.endswith("…"):
        return True

    # Avoid titles that are clearly years / guide labels.
    if re.fullmatch(
        r"\(?\d{4}\s+(guide|edition|list)\)?",
        normalized,
    ):
        return True

# Incomplete search-result titles are not useful restaurant names.
    if re.search(r"\b(the real)\b", normalized) and (
        name.endswith("...")
        or name.endswith("…")
    ):
        return True

    if any(
        site in normalized
        for site in NON_RESTAURANT_SITES
    ):
        return True

    # Generic search/list-page phrases.
    bad_phrases = [
        "best restaurants",
        "top restaurants",
        "restaurants in",
        "restaurants near",
        "restaurant guide",
        "restaurant directory",
        "restaurant reviews",
        "where to eat",
        "places to eat",
        "things to do",
        "order food",
        "food delivery",
        "business profiles",
        "business listings",
        "updated july",
        "updated june",
        "updated may",
    ]

    return any(
        phrase in normalized
        for phrase in bad_phrases
    )


def clean_candidate_name(name: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        name.strip(" |•–—-"),
    ).strip()


def classify_cuisine(text: str) -> str:
    lowered = text.lower()

    matches = []

    for cuisine, keywords in CUISINE_TAXONOMY.items():
        if any(
            keyword in lowered
            for keyword in keywords
        ):
            matches.append(cuisine)

    return (
        ", ".join(matches)
        if matches
        else "General Dining"
    )


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

RESTAURANT_TERMS = [
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
    "eatery",
    "tavern",
    "pub",
    "kitchen",
    "chophouse",
]


def score_candidate(
    candidate: str,
    result: Dict[str, str],
    source_text: str,
    target_city: str,
    target_state: str,
) -> Tuple[int, List[str]]:

    score = 0
    reasons = []

    lowered = source_text.lower()
    candidate_lower = candidate.lower()

    query = result.get("query", "").lower()

    # ---------------------------------------------------------------
    # Restaurant-context evidence
    # ---------------------------------------------------------------

    if any(
        term in lowered
        for term in RESTAURANT_TERMS
    ):
        score += 2
        reasons.append("restaurant context")

    # ---------------------------------------------------------------
    # Address evidence
    # ---------------------------------------------------------------

    if re.search(
        r"\b\d{1,6}\s+[A-Za-z0-9.'#&-]+",
        source_text,
    ):
        score += 2
        reasons.append("address evidence")

    # ---------------------------------------------------------------
    # Phone evidence
    # ---------------------------------------------------------------

    if re.search(
        r"(?:\(\d{3}\)|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}",
        source_text,
    ):
        score += 1
        reasons.append("phone evidence")

    # ---------------------------------------------------------------
    # ZIP evidence
    # ---------------------------------------------------------------

    if re.search(
        r"\b\d{5}(?:-\d{4})?\b",
        source_text,
    ):
        score += 1
        reasons.append("ZIP evidence")

    # ---------------------------------------------------------------
    # Target-location evidence
    # ---------------------------------------------------------------

    city_lower = target_city.lower()
    state_lower = target_state.lower()

    if city_lower in lowered:
        score += 2
        reasons.append("target city evidence")

    # Search query itself is authoritative evidence of intended location.
    if (
        city_lower in query
        and state_lower in query
    ):
        score += 2
        reasons.append("target location in search query")

    # ---------------------------------------------------------------
    # Named-business evidence
    # ---------------------------------------------------------------

    if (
        candidate_lower in lowered
        and candidate_lower not in {
            "restaurant",
            "restaurants",
            "food",
            "dining",
            "business profiles",
            "business listings",
        }
    ):
        score += 1
        reasons.append("named business evidence")

    # ---------------------------------------------------------------
    # Search-result query weighting
    # ---------------------------------------------------------------

    if "address phone" in query:
        score += 2
        reasons.append("address/phone search")

    elif "independent restaurants" in query:
        score += 1
        reasons.append("independent restaurant search")

    elif any(
        cuisine in query
        for cuisine in [
            "japanese",
            "italian",
            "mexican",
            "seafood",
            "thai",
        ]
    ):
        score += 1
        reasons.append("cuisine-specific search")

    return score, reasons


def extract_restaurant_candidates(
    result: Dict[str, str],
    target_city: str,
    target_state: str,
    doc=None,
) -> List[Dict]:

    title = result.get("title", "")
    body = result.get("body", "")

    source_text = f"{title}. {body}"

    if doc is None:
        doc = load_spacy_pipeline()(source_text)

    candidates = []

    # ------------------------------------------------------------------
    # spaCy ORG entities
    # ------------------------------------------------------------------

    for ent in doc.ents:

        if ent.label_ != "ORG":
            continue

        candidate = clean_candidate_name(ent.text)

        if is_bad_candidate(candidate):
            continue

        score, reasons = score_candidate(
            candidate=candidate,
            result=result,
            source_text=source_text,
            target_city=target_city,
            target_state=target_state,
        )

        candidates.append({
            "candidate": candidate,
            "score": score,
            "reasons": reasons,
            "source": "spaCy ORG",
        })

    # ------------------------------------------------------------------
    # Search-result title
    #
    # We still consider titles, but only if they look like actual
    # business names. Generic page titles are explicitly rejected.
    # ------------------------------------------------------------------

    if title:

        title_candidate = clean_candidate_name(title)

        if (
            not is_bad_candidate(title_candidate)
            and len(title_candidate.split()) <= 8
        ):
            score, reasons = score_candidate(
                candidate=title_candidate,
                result=result,
                source_text=source_text,
                target_city=target_city,
                target_state=target_state,
            )

            score += 1
            reasons.append("search-result title")

            candidates.append({
                "candidate": title_candidate,
                "score": score,
                "reasons": reasons,
                "source": "search title",
            })

        # Some titles use:
        #
        #   Restaurant Name | Restaurant Type
        #
        # or:
        #
        #   Restaurant Name - Baltimore
        #
        title_parts = re.split(
            r"\s+[|•–—-]\s+",
            title,
        )

        for part in title_parts:

            candidate = clean_candidate_name(part)

            if not candidate:
                continue

            if is_bad_candidate(candidate):
                continue

            if len(candidate.split()) > 8:
                continue

            score, reasons = score_candidate(
                candidate=candidate,
                result=result,
                source_text=source_text,
                target_city=target_city,
                target_state=target_state,
            )

            score += 1
            reasons.append("title component")

            candidates.append({
                "candidate": candidate,
                "score": score,
                "reasons": reasons,
                "source": "title component",
            })

    # ------------------------------------------------------------------
    # Deduplicate candidates within this search result.
    # ------------------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = normalize_name(
            candidate["candidate"]
        )

        if not key:
            continue

        existing = unique.get(key)

        if (
            existing is None
            or candidate["score"] > existing["score"]
        ):
            unique[key] = candidate

    return list(unique.values())


def choose_best_candidate(
    result: Dict[str, str],
    target_city: str,
    target_state: str,
    doc=None,
) -> Tuple[str, int, str]:

    candidates = extract_restaurant_candidates(
        result=result,
        target_city=target_city,
        target_state=target_state,
        doc=doc,
    )

    if not candidates:
        return (
            "Unknown Restaurant",
            0,
            "No candidate generated",
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    # Require meaningful evidence.
    # This prevents things like "Business Profiles" from surviving
    # merely because spaCy labeled them as ORG.
    if best["score"] < 4:
        return (
            "Unknown Restaurant",
            best["score"],
            "; ".join(best["reasons"]),
        )

    return (
        best["candidate"],
        best["score"],
        "; ".join(best["reasons"]),
    )
