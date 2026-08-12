"""
Crops UEC-Food256 images to their bounding boxes and reorganizes them into
a clean per-category folder structure: <output_dir>/<category_slug>/*.jpg

UEC-256's raw layout is annotation-centric, not classification-ready: a
single photo can appear in more than one category folder (if it contains
several different foods), and a category's bb_info.txt can reference the
same photo more than once (if it contains several instances of that food).
Cropping to each bounding box turns every annotated food instance into its
own clean, correctly-labeled training image, instead of guessing a single
label for a whole (possibly multi-food) photo.

Usage:
    python prepare_uec256.py --uec-root ./data/uecfood256/UECFOOD256 --output-dir ./data/uec256_cropped

Note: the exact column layout of bb_info.txt is parsed based on the
documented "img x1 y1 x2 y2" convention used across the published papers
and third-party tooling for this dataset. After running this once, sanity
check a handful of crops visually (see the printed sample paths at the end)
before trusting the full output -- if boxes look off, the format may differ
slightly from what's assumed here and the parser will need adjusting.
"""
import argparse
import random
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def slugify(name: str) -> str:
    """
    Normalizes a UEC category name into a filesystem- and mapping-safe slug:
    lowercase, non-alphanumeric runs collapsed to a single underscore.
    Used identically in merge_datasets.py -- keep these in sync.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_category_names(uec_root: Path) -> dict:
    """Parses category.txt: '<id><whitespace><name>' per line, first line is a header."""
    names = {}
    with open(uec_root / "category.txt", encoding="utf-8") as f:
        next(f)  # header line
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) < 2:
                continue
            cat_id, name = parts[0], parts[1]
            names[int(cat_id)] = name.strip()
    return names


def parse_bb_info(path: Path):
    """
    Parses a bb_info.txt file: one header line, then rows of
        img_id x1 y1 x2 y2
    where img_id matches <img_id>.jpg in the same category folder.
    Returns a list of (img_id: str, x1, y1, x2, y2) tuples.
    """
    boxes = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            img_id = parts[0]
            x1, y1, x2, y2 = (int(float(p)) for p in parts[1:])
        except ValueError:
            continue  # header row -- doesn't parse as "id + 4 numbers"
        boxes.append((img_id, x1, y1, x2, y2))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uec-root", type=str, required=True,
                         help="Path to the extracted UECFOOD256 folder (contains category.txt, 1/, 2/, ...)")
    parser.add_argument("--output-dir", type=str, default="./data/uec256_cropped")
    parser.add_argument("--min-size", type=int, default=32,
                         help="Skip crops smaller than this many pixels on either side (likely bad/degenerate boxes)")
    args = parser.parse_args()

    uec_root = Path(args.uec_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_names = load_category_names(uec_root)
    print(f"Loaded {len(category_names)} categories from category.txt")

    total_crops, skipped, sample_paths = 0, 0, []
    for cat_id, name in tqdm(sorted(category_names.items()), desc="categories"):
        slug = slugify(name)
        cat_dir = uec_root / str(cat_id)
        bb_path = cat_dir / "bb_info.txt"
        if not bb_path.exists():
            print(f"  warning: no bb_info.txt for category {cat_id} ({slug}), skipping")
            continue

        out_cat_dir = output_dir / slug
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        for img_id, x1, y1, x2, y2 in parse_bb_info(bb_path):
            src_path = cat_dir / f"{img_id}.jpg"
            if not src_path.exists():
                continue
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(img.width, x2), min(img.height, y2)
                    if x2c - x1c < args.min_size or y2c - y1c < args.min_size:
                        skipped += 1
                        continue
                    crop = img.crop((x1c, y1c, x2c, y2c))
                    out_path = out_cat_dir / f"{img_id}_{x1}_{y1}.jpg"
                    crop.save(out_path, quality=95)
                    total_crops += 1
                    if len(sample_paths) < 8 and random.random() < 0.02:
                        sample_paths.append(str(out_path))
            except Exception as e:
                print(f"  failed on {src_path}: {e}")
                skipped += 1

    print(f"\nDone. {total_crops} cropped images written to {output_dir} ({skipped} skipped as too small/invalid).")
    if sample_paths:
        print("\nSanity-check a few of these visually before trusting the full run:")
        for p in sample_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
