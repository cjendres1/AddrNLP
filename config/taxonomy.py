CUISINE_TAXONOMY = {
    "Japanese": [
        "japanese",
        "sushi",
        "ramen",
        "izakaya",
        "teriyaki",
        "hibachi",
        "tempura",
    ],
    "Thai": [
        "thai",
        "pad thai",
        "bangkok",
        "siam",
    ],
    "Mexican": [
        "mexican",
        "taqueria",
        "taco",
        "tacos",
        "cantina",
        "burrito",
        "enchilada",
    ],
    "Italian": [
        "italian",
        "trattoria",
        "pasta",
        "pizzeria",
        "pizza",
    ],
    "Indian": [
        "indian",
        "curry",
        "tandoori",
        "masala",
        "naan",
        "biryani",
    ],
    "Chinese": [
        "chinese",
        "dim sum",
        "szechuan",
        "sichuan",
        "wok",
        "mandarin",
    ],
    "Korean": [
        "korean",
        "bibimbap",
        "bulgogi",
        "kimchi",
        "korean bbq",
    ],
    "Vietnamese": [
        "vietnamese",
        "pho",
        "banh mi",
    ],
    "French": [
        "french",
        "bistro",
        "brasserie",
        "creperie",
        "crepe",
    ],
    "Mediterranean": [
        "mediterranean",
        "greek",
        "gyro",
        "falafel",
        "hummus",
        "lebanese",
    ],
    "Seafood": [
        "seafood",
        "oyster",
        "lobster",
        "crab",
        "fish",
    ],
    "Steakhouse": [
        "steakhouse",
        "steak",
        "prime rib",
    ],
    "Breakfast & Brunch": [
        "breakfast",
        "brunch",
        "omelet",
        "pancake",
        "waffle",
    ],
    "Bakery / Cafe": [
        "bakery",
        "cafe",
        "café",
        "coffee",
        "espresso",
    ],
}


def classify_cuisine(text: str) -> str:

    lowered = text.lower()

    matches = []

    for cuisine, keywords in CUISINE_TAXONOMY.items():

        if any(keyword in lowered for keyword in keywords):
            matches.append(cuisine)

    return ", ".join(matches) if matches else "General Dining"


# =============================================================================
