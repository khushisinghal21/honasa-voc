"""Generate SYNTHETIC reviews for an offline smoke test of the pipeline.

These are template sentences, not real consumer opinions. Every row has source="synthetic" and the
digest prints a warning banner. Never present results from this data as findings about real products.

Usage: python scripts/make_synthetic.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = datetime(2026, 9, 28)

# (aspect, sentiment, text). Severity isn't needed for gold labels.
T = {
    "shampoo": [
        ("efficacy", "negative", "No difference in hair fall even after finishing the full bottle."),
        ("efficacy", "negative", "Koi fark nahi pada hair fall mein, paisa waste ho gaya."),
        ("efficacy", "negative", "Hair fall actually increased after three weeks of use."),
        ("efficacy", "positive", "My hair fall reduced a lot within a month."),
        ("efficacy", "positive", "Hair fall kam ho gaya, bahut accha result hai."),
        ("fragrance", "positive", "The rosemary smell is so fresh and pleasant."),
        ("fragrance", "negative", "The smell is too strong and gives me a headache."),
        ("texture", "negative", "Lather is very low so I end up using too much."),
        ("packaging", "negative", "The pump stopped working halfway through the bottle."),
        ("packaging", "negative", "Pump jammed after a week and I had to open the cap to use it."),
        ("packaging", "negative", "Cap was broken and the bottle leaked inside the box."),
        ("price_value", "negative", "Too expensive for such a small bottle."),
        ("skin_reaction", "negative", "Got itching and a rash on my scalp after using it."),
        ("authenticity", "negative", "I think I received a fake product, the seal was tampered."),
    ],
    "face_wash": [
        ("efficacy", "positive", "It controls oil really well for the whole day."),
        ("efficacy", "positive", "My pimples have reduced since I started using it."),
        ("efficacy", "negative", "Did nothing for my acne, no change at all."),
        ("skin_reaction", "negative", "It made my skin very dry and tight after every wash."),
        ("skin_reaction", "negative", "Leaves my face dry and stretchy, need moisturiser immediately."),
        ("skin_reaction", "negative", "Broke me out badly, small pimples all over my forehead."),
        ("skin_reaction", "negative", "Chehre pe dane aa gaye isse, bilkul suit nahi hua."),
        ("skin_reaction", "negative", "Jalan hoti hai lagane ke baad, real burning sensation."),
        ("fragrance", "positive", "Smells lovely and fresh."),
        ("fragrance", "negative", "The fragrance is too artificial and strong."),
        ("texture", "positive", "The gel texture foams nicely."),
        ("packaging", "negative", "The tube cap broke within a week."),
        ("price_value", "positive", "Affordable and lasts long, value for money."),
        ("delivery", "negative", "Ordered on Blinkit but the box was damaged."),
    ],
    "sunscreen": [
        ("texture", "positive", "No white cast at all and very lightweight."),
        ("texture", "positive", "Bahut accha hai, no white cast bilkul."),
        ("texture", "negative", "Leaves a strong white cast on my skin."),
        ("texture", "negative", "Too greasy and sticky in Delhi summer."),
        ("texture", "negative", "Feels greasy and my face looks oily within an hour."),
        ("efficacy", "negative", "I still got tanned even after reapplying."),
        ("efficacy", "positive", "Great protection, I didn't get tanned on my beach trip."),
        ("skin_reaction", "negative", "It stings around my eyes."),
        ("skin_reaction", "negative", "Got an allergic rash on my cheeks after two days."),
        ("fragrance", "negative", "Strong perfume smell, not good for sensitive skin."),
        ("price_value", "negative", "Overpriced for 50 ml."),
        ("packaging", "negative", "The nozzle keeps leaking and the bottle gets messy."),
        ("packaging", "negative", "Pump leaks in my bag every single time."),
        ("authenticity", "negative", "The bottle was expired when it arrived."),
    ],
}

# Which templates each product leans on, so each product has its own story in the demo.
PRODUCTS = {
    "me_rosemary_shampoo": ("shampoo", {"packaging": 3, "efficacy": 3, "fragrance": 2}),
    "me_teatree_facewash": ("face_wash", {"skin_reaction": 4, "efficacy": 2}),
    "me_ubtan_facewash": ("face_wash", {"fragrance": 3, "skin_reaction": 2, "price_value": 2}),
    "tdc_hyaluronic_sunscreen": ("sunscreen", {"texture": 4, "efficacy": 1}),
    "aq_dewy_sunscreen": ("sunscreen", {"packaging": 4, "texture": 2}),
    "comp_minimalist_sunscreen": ("sunscreen", {"texture": 3, "skin_reaction": 2}),
}
OPENERS = ["", "", "Honest review: ", "Been using this for a month. ", "Update after 3 weeks: ", "Bought it last month. ",
           "My second purchase. ", "Tried it on my sister's suggestion. "]
CLOSERS = ["", "", "", " Using it every day.", " Rating 3/5.", " Will see how it goes.", " Bought the bigger size.",
           " Using it since June.", " Not sure about repurchasing."]
NOISE = [
    "First comment! Love your videos didi",
    "Which camera do you use for these videos?",
    "Please review the vitamin C serum next",
    "Can you do a video on night routine?",
    "Where did you buy your top from?",
]


def pick(rng: random.Random, templates: list, weights: dict) -> tuple:
    w = [weights.get(a, 1) for a, _, _ in templates]
    return rng.choices(templates, weights=w, k=1)[0]


def main(n_per_product: int = 45, seed: int = 42) -> None:
    rng = random.Random(seed)
    reviews, gold = [], []
    for pid, (cat, weights) in PRODUCTS.items():
        for i in range(n_per_product):
            parts = [pick(rng, T[cat], weights)]
            if rng.random() < 0.4:
                second = pick(rng, T[cat], weights)
                if second[0] != parts[0][0]:
                    parts.append(second)
            text = rng.choice(OPENERS) + " ".join(p[2] for p in parts) + rng.choice(CLOSERS)
            # Packaging complaints skew recent for two products so the trend metric has something to find.
            recent_bias = parts[0][0] == "packaging" and pid in ("me_rosemary_shampoo", "aq_dewy_sunscreen")
            days_ago = rng.randint(0, 27) if recent_bias and rng.random() < 0.8 else rng.randint(0, 83)
            rid = f"synthetic:{pid}:{i}"
            reviews.append({
                "review_id": rid, "product_id": pid, "text": text,
                "created_at": (REF - timedelta(days=days_ago)).isoformat(),
                "likes": rng.randint(0, 40), "source_url": None,
            })
            gold.append({"review_id": rid, "product_id": pid, "text": text, "is_relevant": True,
                         "aspects": [{"aspect": a, "sentiment": s} for a, s, _ in parts]})
        for j in range(4):
            rid = f"synthetic:{pid}:noise{j}"
            text = rng.choice(NOISE)
            reviews.append({"review_id": rid, "product_id": pid, "text": text,
                            "created_at": (REF - timedelta(days=rng.randint(0, 83))).isoformat(), "likes": 0})
            gold.append({"review_id": rid, "product_id": pid, "text": text, "is_relevant": False, "aspects": []})
        dup = dict(reviews[-6])  # an exact duplicate, to exercise the dedupe step
        dup["review_id"] = f"synthetic:{pid}:dup"
        reviews.append(dup)

    out = ROOT / "data/sample/synthetic_reviews.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reviews) + "\n", encoding="utf-8")
    gold_out = ROOT / "evals/demo_gold_synthetic.jsonl"
    gold_out.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n", encoding="utf-8")
    print(f"wrote {len(reviews)} synthetic reviews -> {out}\nwrote {len(gold)} synthetic gold labels -> {gold_out}")


if __name__ == "__main__":
    main()
