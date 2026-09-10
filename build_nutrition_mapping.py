"""
Builds a class-name -> nutrition-data mapping by querying USDA FoodData Central once per
class, offline. The result (`nutrition_mapping.json`) is committed to the repo and used at
inference time as a plain dictionary lookup -- no API key, no network, no rate limits, and
deterministic results in the deployed app.

WHY OFFLINE: the class set is fixed and known (315 classes), so there's no reason to query
live per request. Querying once here also means the demo can't break because an external API
is rate-limited or down.

WHY FNDDS FIRST: FoodData Central is five databases. Foundation/SR Legacy are mostly raw
ingredients ("beef, ribeye, raw"), while Survey (FNDDS) catalogs *prepared, mixed dishes as
consumed* ("beef steak, grilled") -- which is what our classes actually are. We search FNDDS
first and fall back to the others, which materially improves match quality for dish names.

USDA's international coverage is weak, so expect poor or missing matches for many of the
UEC-256 (Japanese/Asian) classes. Those are recorded with a low match score and/or
`"needs_review": true` rather than silently accepted -- see the summary printed at the end.

Re-running is NON-DESTRUCTIVE: entries already present in the output file are preserved
untouched (including any you've hand-corrected), and only classes missing from it are queried.

Setup:
    Get a free API key from https://api.data.gov/signup (takes a minute), then:
        export USDA_API_KEY=your_key_here

Usage:
    # SPIKE FIRST -- try a representative handful before committing to all 315:
    python build_nutrition_mapping.py --classes-file ./checkpoints/classes.txt \\
        --output ./nutrition_mapping.json \\
        --only apple_pie filet_mignon khao_soi ramen sushi tempura miso_soup pho gyoza churros

    # Then the full run:
    python build_nutrition_mapping.py --classes-file ./checkpoints/classes.txt \\
        --output ./nutrition_mapping.json

Data licensing: USDA FoodData Central data are public domain (CC0 1.0). No permission is
needed to use them; FDC asks only to be cited as the source. The generated file records this.
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

FDC_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
FDC_DETAIL_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

# USDA nutrient IDs for the fields we care about (calories + standard macros).
NUTRIENT_IDS = {
    "calories_kcal": 1008,
    "protein_g": 1003,
    "fat_g": 1004,
    "carbs_g": 1005,
}

# Searched in order; the first data type that yields a good match wins. FNDDS ("Survey")
# holds prepared dishes as consumed, which matches our class names far better than the
# raw-ingredient-oriented Foundation / SR Legacy sets.
DATA_TYPE_PRIORITY = ["Survey (FNDDS)", "SR Legacy", "Foundation"]

# Below this blended score, an entry is flagged for review rather than trusted. Permissive on
# purpose -- it's a review prompt, not a rejection.
MATCH_SCORE_THRESHOLD = 0.45

# Separately, an entry is flagged if fewer than this fraction of the dish name's words were
# found in the USDA description. Catches plausible-but-partial matches that a blended score
# alone lets through (see token_coverage). 0.99 rather than 1.0 because a fuzzy per-token
# match scores 0.9, and a fully-fuzzy-matched name should still count as covered.
COVERAGE_THRESHOLD = 0.89


def humanize(class_name: str) -> str:
    """`filet_mignon` -> `filet mignon`, for use as a search query."""
    return class_name.replace("_", " ").strip()


def tokenize(text: str) -> list:
    """Lowercased alphanumeric tokens, dropping punctuation and USDA's comma-separated structure."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def token_coverage(query: str, description: str) -> float:
    """
    Fraction of the query's words that appear in the description (fuzzy per token, to absorb
    plurals and small spelling variations like "donut"/"doughnut").

    This is tracked separately from the blended score because *incomplete* coverage is a
    meaningful signal on its own: "beef tartare" against "Beef, ground, raw" covers only half
    the query -- USDA found beef, but not beef tartare. That's exactly the kind of
    plausible-but-wrong match that needs human review, and it isn't reliably caught by
    thresholding a single blended number.
    """
    query_tokens = tokenize(query)
    description_tokens = tokenize(description)
    if not query_tokens or not description_tokens:
        return 0.0

    matched = 0.0
    for q_token in query_tokens:
        if q_token in description_tokens:
            matched += 1.0
        elif difflib.get_close_matches(q_token, description_tokens, n=1, cutoff=0.8):
            matched += 0.9
    return matched / len(query_tokens)


def similarity(query: str, description: str) -> float:
    """
    0-1 score for how well a USDA food description matches our class name, used to RANK
    candidates against each other.

    Deliberately NOT a plain SequenceMatcher on the full strings: USDA writes descriptions in
    inverted form with trailing qualifiers ("Pie, apple, commercially prepared"), so raw
    sequence similarity heavily penalises genuinely correct matches -- "apple pie" against
    that description scores only ~0.38, below any sane threshold.
    """
    coverage = token_coverage(query, description)
    if coverage == 0.0:
        return 0.0
    query_tokens = tokenize(query)
    description_tokens = tokenize(description)
    # Mild preference for concise descriptions over ones padded with extra qualifiers,
    # used only to break ties between candidates with equal coverage.
    conciseness = len(query_tokens) / max(len(description_tokens), len(query_tokens))
    return round(min(1.0, 0.85 * coverage + 0.15 * conciseness), 4)


def fdc_request(url: str, params: dict, timeout: int = 30) -> dict:
    """
    Plain stdlib GET -- avoids adding `requests` as a dependency just for this.

    On a non-2xx response, raises with USDA's actual response BODY included, not just
    urllib's generic reason phrase ("HTTP Error 400: Bad Request" on its own says nothing
    about *why* -- USDA's body is typically JSON with a real explanation).
    """
    full_url = f"{url}?{urlencode(params)}"
    try:
        with urlopen(full_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason} -- response body: {body}") from e


def search_food(query: str, api_key: str, data_type: str, page_size: int = 10) -> list:
    params = {
        "api_key": api_key,
        "query": query,
        "dataType": data_type,
        "pageSize": page_size,
    }
    try:
        return fdc_request(FDC_SEARCH_URL, params).get("foods", [])
    except Exception as e:
        print(f"    search failed for {query!r} in {data_type}: {e}")
        return []


def fetch_detail(fdc_id: int, api_key: str) -> dict:
    """
    The detail endpoint is needed for `foodPortions` (gram weights of typical servings),
    which the search endpoint doesn't reliably return. That's what lets us report
    per-serving values alongside per-100g.
    """
    try:
        return fdc_request(FDC_DETAIL_URL.format(fdc_id=fdc_id), {"api_key": api_key})
    except Exception as e:
        print(f"    detail fetch failed for fdcId={fdc_id}: {e}")
        return {}


def extract_nutrients(food: dict) -> dict:
    """
    Pulls calories + macros out of a food record, per 100g.

    The search and detail endpoints nest the nutrient ID differently
    (`nutrientId` vs `nutrient.id`), so both shapes are handled.
    """
    found = {}
    for entry in food.get("foodNutrients", []):
        nutrient_id = entry.get("nutrientId") or entry.get("nutrient", {}).get("id")
        value = entry.get("value") if "value" in entry else entry.get("amount")
        if nutrient_id is None or value is None:
            continue
        for field, target_id in NUTRIENT_IDS.items():
            if nutrient_id == target_id:
                found[field] = round(float(value), 2)
    return found


def extract_serving(food: dict) -> dict | None:
    """
    Finds a representative serving size in grams.

    Prefers an explicit `servingSize` (branded foods), then falls back to the largest
    `foodPortions` entry, which for FNDDS dishes is typically a sensible "1 cup"/"1 piece"
    style portion rather than an arbitrary sub-portion.
    """
    if food.get("servingSize") and food.get("servingSizeUnit", "").lower() in ("g", "gram", "grams"):
        return {"grams": round(float(food["servingSize"]), 1), "description": "serving"}

    portions = food.get("foodPortions") or []
    best = None
    for portion in portions:
        grams = portion.get("gramWeight")
        if not grams:
            continue
        description = (
            portion.get("portionDescription")
            or portion.get("modifier")
            or (portion.get("measureUnit") or {}).get("name")
            or "serving"
        )
        if best is None or grams > best["grams"]:
            best = {"grams": round(float(grams), 1), "description": str(description)}
    return best


def best_match(class_name: str, api_key: str, request_delay: float) -> dict | None:
    """
    Searches each data type in priority order and returns the best-scoring candidate that
    actually has calorie data. Stops early on a strong match to save API calls.
    """
    query = humanize(class_name)
    best = None

    for data_type in DATA_TYPE_PRIORITY:
        foods = search_food(query, api_key, data_type)
        time.sleep(request_delay)
        # Always logged, success or not -- this is what makes "we tried this data type and
        # it genuinely had nothing" distinguishable from "this data type never got queried",
        # which the output couldn't previously tell apart.
        print(f"    {data_type}: {len(foods)} candidate(s)")
        for food in foods:
            description = food.get("description", "")
            score = similarity(query, description)
            if best is None or score > best["score"]:
                best = {"score": score, "food": food, "data_type": data_type}
        # A strong match in a higher-priority data type means we don't need to keep looking.
        if best and best["score"] >= 0.8:
            break

    if best is None:
        print("    (no candidates found in any data type)")
        return None

    food = best["food"]
    fdc_id = food.get("fdcId")
    print(f"    best candidate: {food.get('description')!r} (fdcId={fdc_id}, "
          f"data_type={best['data_type']}, score={best['score']:.3f})")

    # Always fetch detail: search results carry partial nutrient data and no foodPortions.
    detail = fetch_detail(fdc_id, api_key) if fdc_id else {}
    time.sleep(request_delay)
    if fdc_id and not detail:
        print(f"    detail fetch for fdcId={fdc_id} returned nothing -- falling back to search-result nutrients")

    nutrients_from_detail = extract_nutrients(detail)
    nutrients = nutrients_from_detail or extract_nutrients(food)
    if not nutrients_from_detail and nutrients:
        print("    calories/macros came from the search result, not the detail endpoint")
    if "calories_kcal" not in nutrients:
        print(f"    found a candidate but could not extract calorie data from it "
              f"(detail keys present: {sorted(detail.keys()) if detail else 'none'})")
        return None  # a match with no calorie data is useless for our purposes

    description = food.get("description", "")
    coverage = token_coverage(query, description)
    # Two independent review triggers: a low overall score, OR incomplete token coverage
    # (some word in the dish name went unmatched, so the match may be of a related but
    # different food -- e.g. plain beef standing in for beef tartare).
    needs_review = best["score"] < MATCH_SCORE_THRESHOLD or coverage < COVERAGE_THRESHOLD

    return {
        "matched_description": description,
        "fdc_id": fdc_id,
        "data_type": best["data_type"],
        "match_score": round(best["score"], 3),
        "token_coverage": round(coverage, 3),
        "needs_review": needs_review,
        "per_100g": nutrients,
        "serving": extract_serving(detail) or extract_serving(food),
    }


def main():
    parser = argparse.ArgumentParser(description="Build class -> USDA nutrition mapping")
    parser.add_argument("--classes-file", type=str, required=True,
                         help="Path to classes.txt (one class name per line)")
    parser.add_argument("--output", type=str, default="./nutrition_mapping.json")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                         help="Only process these specific class names -- use for a small spike run first")
    parser.add_argument("--api-key", type=str, default=None,
                         help="USDA API key. Defaults to the USDA_API_KEY environment variable.")
    parser.add_argument("--request-delay", type=float, default=0.4,
                         help="Seconds to sleep between API calls. USDA allows 1000 req/hour; this "
                              "script makes ~2 calls per class, so 315 classes is ~630 calls. The "
                              "default keeps a comfortable margin under the limit.")
    parser.add_argument("--retry", type=str, nargs="*", default=None,
                         help="Class names to force re-query even though they're already present in "
                              "--output (e.g. classes that got a bad match or 'no match' last run). "
                              "Non-destructive re-runs otherwise skip anything already recorded -- "
                              "including a failed/null result -- since that's indistinguishable from "
                              "a hand-reviewed entry without this flag. For more than a handful of "
                              "names, prefer --retry-from-file instead -- passing dozens/hundreds of "
                              "names through shell command substitution is easy to get subtly wrong.")
    parser.add_argument("--retry-from-file", type=str, default=None,
                         help="Path to a text file with one class name per line to force re-query. "
                              "Same effect as --retry, but reads the list directly instead of relying "
                              "on the shell to pass it as arguments -- the robust way to combine with "
                              "summarize_nutrition_mapping.py --list-only for a large retry batch.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("USDA_API_KEY")
    if not api_key:
        sys.exit("No API key. Set USDA_API_KEY or pass --api-key. Get one free at https://api.data.gov/signup")

    class_names = [c.strip() for c in Path(args.classes_file).read_text(encoding="utf-8").splitlines() if c.strip()]
    if args.only:
        requested = set(args.only)
        unknown = requested - set(class_names)
        if unknown:
            print(f"Warning: these --only names aren't in {args.classes_file}: {sorted(unknown)}")
        class_names = [c for c in class_names if c in requested]
    print(f"Processing {len(class_names)} classes")

    output_path = Path(args.output)
    mapping = {}
    if output_path.exists():
        mapping = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Found existing {args.output} with {len(mapping)} entries -- preserving all of them")

    retry_names = list(args.retry or [])
    if args.retry_from_file:
        file_names = [c.strip() for c in Path(args.retry_from_file).read_text(encoding="utf-8").splitlines() if c.strip()]
        print(f"Read {len(file_names)} names from {args.retry_from_file}")
        retry_names.extend(file_names)

    if retry_names:
        cleared = [c for c in retry_names if mapping.pop(c, None) is not None]
        not_found = set(retry_names) - set(cleared)
        print(f"Cleared {len(cleared)} entries for forced retry: {sorted(cleared)}")
        if not_found:
            print(f"  (not previously present, will be queried normally anyway: {sorted(not_found)})")

    todo = [c for c in class_names if c not in mapping]
    print(f"{len(class_names) - len(todo)} already present, {len(todo)} to query\n")

    n_matched, n_flagged, n_missing = 0, 0, 0
    for i, class_name in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {class_name}")
        try:
            result = best_match(class_name, api_key, args.request_delay)
        except KeyboardInterrupt:
            print("\nInterrupted -- saving progress so far.")
            break

        if result is None:
            mapping[class_name] = {"matched_description": None, "needs_review": True, "per_100g": None,
                                    "serving": None, "note": "no usable USDA match found"}
            n_missing += 1
            print("    -> NO MATCH")
        else:
            mapping[class_name] = result
            n_matched += 1
            if result["needs_review"]:
                n_flagged += 1
            flag = "  [NEEDS REVIEW]" if result["needs_review"] else ""
            print(f"    -> {result['matched_description']!r} "
                  f"({result['data_type']}, score={result['match_score']}){flag}")

        # Save incrementally so an interruption (or a rate-limit block) doesn't lose work.
        output_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(mapping)} total entries to {args.output}")
    print(f"  {n_matched} matched this run ({n_flagged} flagged as low-confidence)")
    print(f"  {n_missing} had no usable match")
    print(
        "\nReview entries with \"needs_review\": true -- USDA's coverage of non-US dishes is weak, "
        "so expect the Japanese/Asian classes to need the most attention. Correcting an entry by "
        "hand is safe: re-running this script preserves anything already in the file."
    )
    print("\nData source: USDA FoodData Central (public domain, CC0 1.0). Cite as: "
          "U.S. Department of Agriculture, Agricultural Research Service. FoodData Central, fdc.nal.usda.gov")


if __name__ == "__main__":
    main()
