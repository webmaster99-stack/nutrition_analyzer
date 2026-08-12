"""
Builds a single unified dataset combining Food-101's train/test images with
UEC-Food256's cropped images, using a reviewed class_mapping.json (see
build_class_mapping.py) to decide which UEC categories merge into an
existing Food-101 class vs become a brand new class.

Output layout (standard torchvision.datasets.ImageFolder structure):
    <output_dir>/train/<class_name>/*.jpg
    <output_dir>/test/<class_name>/*.jpg

Food-101 images keep their original official train/test split, so accuracy
numbers on the original 101 classes stay comparable to your earlier runs.
UEC images have no official split, so they get a random per-category 85/15
train/test split instead.

Files are symlinked by default (fast, no extra disk usage for the ~5GB of
Food-101 images) and fall back to copying if symlinks aren't supported.

NOTE for Colab + Google Drive: symlinks can behave unreliably through
Drive's FUSE-mounted filesystem. If --output-dir points into
/content/drive/..., pass --copy, or (better) write the merged dataset to
local Colab disk (e.g. /content/merged_food_dataset) and just re-run this
script each session -- it's fast to regenerate from the already-downloaded
source data, unlike checkpoints/logs which are worth persisting to Drive.

Usage:
    python merge_datasets.py \\
        --food101-root ./data \\
        --uec-cropped-dir ./data/uec256_cropped \\
        --class-mapping ./class_mapping.json \\
        --output-dir ./data/merged_food_dataset
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

from torchvision import datasets

random.seed(42)


def slugify(name: str) -> str:
    """Must match the slugify used in prepare_uec256.py / build_class_mapping.py."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass  # fall back to copying, e.g. on filesystems without symlink support
    shutil.copy2(src, dst)


def merge_food101(output_dir: Path, food101_root: str, use_symlink: bool):
    for split in ("train", "test"):
        ds = datasets.Food101(root=food101_root, split=split, download=True)
        for img_path, label_idx in zip(ds._image_files, ds._labels):
            class_name = ds.classes[label_idx]
            dst = output_dir / split / class_name / Path(img_path).name
            link_or_copy(Path(img_path), dst, use_symlink)
        print(f"  Food-101 {split}: {len(ds._image_files)} images")
    print("Food-101 merged.")


def merge_uec(output_dir: Path, uec_cropped_dir: Path, mapping: dict, use_symlink: bool, test_fraction: float):
    n_train, n_test, n_missing = 0, 0, 0
    for cat_id, info in mapping.items():
        final_class = info["final_class_name"]
        uec_slug = slugify(info["uec_name"])
        src_dir = uec_cropped_dir / uec_slug
        if not src_dir.exists():
            print(f"  warning: no cropped images found for UEC category {cat_id} ({uec_slug}), skipping")
            n_missing += 1
            continue

        images = sorted(src_dir.glob("*.jpg"))
        random.shuffle(images)
        n_test_i = max(1, int(len(images) * test_fraction))
        test_images, train_images = images[:n_test_i], images[n_test_i:]

        for img in train_images:
            link_or_copy(img, output_dir / "train" / final_class / img.name, use_symlink)
        for img in test_images:
            link_or_copy(img, output_dir / "test" / final_class / img.name, use_symlink)

        n_train += len(train_images)
        n_test += len(test_images)

    print(f"  UEC-256: {n_train} train images, {n_test} test images ({n_missing} categories missing cropped data)")
    print("UEC-256 merged.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--food101-root", type=str, default="./data")
    parser.add_argument("--uec-cropped-dir", type=str, default="./data/uec256_cropped")
    parser.add_argument("--class-mapping", type=str, default="./class_mapping.json")
    parser.add_argument("--output-dir", type=str, default="./data/merged_food_dataset")
    parser.add_argument("--uec-test-fraction", type=float, default=0.15)
    parser.add_argument("--copy", action="store_true",
                         help="Copy files instead of symlinking -- needed on filesystems without symlink "
                              "support, e.g. some Google Drive FUSE mounts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    mapping = json.loads(Path(args.class_mapping).read_text(encoding="utf-8"))
    use_symlink = not args.copy

    print("Merging Food-101...")
    merge_food101(output_dir, args.food101_root, use_symlink)

    print("\nMerging UEC-256...")
    merge_uec(output_dir, Path(args.uec_cropped_dir), mapping, use_symlink, args.uec_test_fraction)

    train_classes = sorted(p.name for p in (output_dir / "train").iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in (output_dir / "test").iterdir() if p.is_dir())
    print(f"\nMerged dataset written to {output_dir}")
    print(f"Total classes: train={len(train_classes)}, test={len(test_classes)}")
    if set(train_classes) != set(test_classes):
        only_train = set(train_classes) - set(test_classes)
        only_test = set(test_classes) - set(train_classes)
        print(f"  warning: class mismatch between splits -- only in train: {only_train or 'none'}, "
              f"only in test: {only_test or 'none'}")


if __name__ == "__main__":
    main()
