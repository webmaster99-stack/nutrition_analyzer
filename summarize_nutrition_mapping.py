"""
Summarizes a nutrition_mapping.json so 313 entries are actually reviewable, instead of reading
through the raw file by hand.

Usage:
    python summarize_nutrition_mapping.py --mapping ./nutrition_mapping.json

    # Just the list of classes needing review, one per line (e.g. to pipe into an editor)
    python summarize_nutrition_mapping.py --mapping ./nutrition_mapping.json --list-only
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=str, default="./nutrition_mapping.json")
    parser.add_argument("--list-only", action="store_true", help="Print only the class names needing review, one per line")
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    total = len(mapping)

    no_match = []
    flagged = []
    clean = []
    data_type_counts = Counter()

    for class_name, entry in mapping.items():
        if not entry.get("per_100g"):
            no_match.append(class_name)

        elif entry.get("needs_review"):
            flagged.append((class_name, entry))

        else:
            clean.append(class_name)
            data_type_counts[entry.get("data_type", "unknown")] += 1

    if args.list_only:
        for class_name in no_match:
            print(class_name)

        for class_name, _ in flagged:
            print(class_name)

        return 

    print(f"Total classes: {total}")
    print(f"  Clean matches:        {len(clean):4d} ({100 * len(clean) / total:.1f}%)")
    print(f"  Flagged for review:   {len(flagged):4d} ({100 * len(flagged) / total:.1f}%)")
    print(f"  No match at all:      {len(no_match):4d} ({100 * len(no_match) / total:.1f}%")

    print(f"\nClean matches by USDA data type (higher priority = better dish-level data):")

    for data_type, count in data_type_counts.most_common():
        print(f"  {data_type:20s} {count:4d}")

    if flagged:
        # Worst matches first - these are the ones most likely to be actually wrong,
        # not just imperfectly worded.
        flagged.sort(key=lambda item: item[1].get("match_score", 0))
        print(f"\nFlagged for review, worst first (showing up to 40 of {len(flagged)}):")

        for class_name, entry in flagged[:40]:
            desc = entry.get("matched_description", "(no match)")
            score = entry.get("match_score", 0.0)
            coverage = entry.get("token_coverage", 0.0)
            print(f"  {class_name:35s} score={score:.2f} coverage={coverage:.2f}  -> {desc}")

    if no_match:
        print(f"\nNo match at all ({len(no_match)}):")
        for class_name in sorted(no_match):
            print(f"  {class_name}")

    print(
        f"\n{len(flagged) + len(no_match)} of {total} classes need attention "
        f"({100 * (len(flagged) + len(no_match)) / total:.1f}%). "
        f"Correct entries directly in {args.mapping} -- re-running build_nutrition_mapping.py "
        f"preserves any hand edits and only fills in classes still missing."
    )


if __name__ == "__main__":
    main()