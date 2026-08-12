"""
Proposes a mapping from UEC-Food256 categories onto existing Food-101 class
names (for dishes that are essentially the same food) -- everything else
becomes a new class in the merged taxonomy.

THIS IS A STARTING POINT, NOT A FINAL ANSWER. Fuzzy string matching on
English glosses will get real matches wrong in both directions:
  - False positive: two categories share a word but are different dishes
    (e.g. UEC's "beef_noodle" fuzzy-matching Food-101's "pho" would be WRONG
    -- pho is a specific Vietnamese dish, not just any beef noodle soup).
  - False negative: a genuine match worded completely differently won't be
    caught by string similarity at all.

Review class_mapping.json by hand and edit "final_class_name" for every
entry before running merge_datasets.py. When in doubt, leave two categories
separate (a new class) rather than merging them -- an incorrect merge
silently poisons a class with wrong-labeled images, which is worse for
training than just having two closely-related classes.

Usage:
    python build_class_mapping.py --uec-root ./data/uecfood256/UECFOOD256 --food101-root ./data
"""
import argparse
import difflib
import json
import re
from pathlib import Path

from torchvision import datasets


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_category_names(uec_root: Path) -> dict:
    names = {}
    with open(uec_root / "category.txt", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) < 2:
                continue
            names[int(parts[0])] = parts[1].strip()
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uec-root", type=str, required=True)
    parser.add_argument("--food101-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./class_mapping.json")
    parser.add_argument("--cutoff", type=float, default=0.6,
                         help="difflib similarity cutoff (0-1) for proposing a Food-101 match. "
                              "Lower = more (noisier) proposed matches; higher = fewer, safer proposals.")
    args = parser.parse_args()

    food101_classes = datasets.Food101(root=args.food101_root, split="train", download=True).classes
    print(f"Food-101: {len(food101_classes)} classes")

    uec_names = load_category_names(Path(args.uec_root))
    print(f"UEC-256: {len(uec_names)} categories")

    # If an output file already exists (e.g. from a previous run you've since
    # hand-reviewed), preserve every entry already in it untouched -- only
    # add entries for category IDs that aren't in it yet. Re-running this
    # script should never silently clobber manual review work.
    output_path = Path(args.output)
    existing_mapping = {}
    if output_path.exists():
        existing_mapping = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Found existing {args.output} with {len(existing_mapping)} entries -- preserving all of them")

    mapping = dict(existing_mapping)  # start from what's already been reviewed
    n_proposed, n_new, n_preserved = 0, 0, len(existing_mapping)
    for cat_id, name in sorted(uec_names.items()):
        if str(cat_id) in mapping:
            continue  # already reviewed in a previous run -- don't touch it
        slug = slugify(name)
        matches = difflib.get_close_matches(slug, food101_classes, n=1, cutoff=args.cutoff)
        proposed = matches[0] if matches else None
        if proposed:
            n_proposed += 1
        n_new += 1
        mapping[str(cat_id)] = {
            "uec_name": name,
            "proposed_food101_match": proposed,
            # <-- EDIT THIS per entry: either an existing Food-101 class name
            # (to merge into it) or any new class name (defaults to the UEC
            # slug, i.e. "keep as its own new class").
            "final_class_name": proposed if proposed else slug,
        }

    output_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(mapping)} total entries to {args.output}")
    print(f"  {n_preserved} preserved unchanged from your previous review")
    print(f"  {n_new} new entries added ({n_proposed} got a proposed Food-101 match, needs review)")
    print(
        "\nIMPORTANT: open class_mapping.json and review 'final_class_name' for any NEW entries.\n"
        "Fuzzy matching WILL propose some wrong merges and miss some real ones. When unsure,\n"
        "leave categories separate -- a bad merge silently mislabels images."
    )


if __name__ == "__main__":
    main()