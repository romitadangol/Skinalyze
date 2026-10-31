import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "products_template.json"

CONCERNS = [
    "acne",
    "blackheads",
    "dark spots",
    "pores",
    "wrinkles",
]

SKIN_TYPES = {
    "Oily": ["Oily"],
    "Dry": ["Dry"],
    "Normal / Combination": ["Normal", "Combination"],
    "Sensitive": ["Sensitive"],
}

CATEGORIES = [
    "Cleanser",
    "Toner",
    "Serum",
    "Moisturizer",
    "Sunscreen",
]


if OUTPUT_PATH.exists():
    raise SystemExit(
        "products_template.json already exists. "
        "Rename or remove it before generating another template."
    )

products = []

for concern in CONCERNS:
    for skin_type_label, tags in SKIN_TYPES.items():
        for category in CATEGORIES:
            products.append({
                "category": category,
                "name": "",
                "description": "",
                "url": "",
                "image": "",
                "skin_types": tags,
                "concerns": [concern],
                "editing_note": (
                    f"{concern} | {skin_type_label} | {category}"
                ),
            })

OUTPUT_PATH.write_text(
    json.dumps(products, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Created {len(products)} product slots.")
print(f"Template: {OUTPUT_PATH}")
print("Your existing products.json has not been changed.")