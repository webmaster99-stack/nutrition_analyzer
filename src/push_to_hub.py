"""
Pushes a trained checkpoint + class list + class_mapping.json to Hugging Face Hub as a
model repo, with an auto-generated model card.

Usage:
    python push_to_hub.py \\
        --repo-id your-username/food-classifier-resnet50 \\
        --checkpoint ./checkpoints/resnet50_food_best.pt \\
        --classes ./checkpoints/classes.txt \\
        --class-mapping ./class_mapping.json \\
        --num-classes 315 \\
        --image-size 320 \\
        --finetune-unfreeze-from layer3 \\
        --best-val-accuracy 0.798 \\
        --test-accuracy 0.844

Authentication: run `huggingface-cli login` beforehand (recommended -- caches a token so you
don't need to pass one on the command line every time), or pass --token explicitly, or set the
HF_TOKEN environment variable. Needs a token with "write" permission from
https://huggingface.co/settings/tokens.

Only the checkpoint, classes.txt, and class_mapping.json (all text/weights, no images) are
uploaded -- this never uploads UEC-256's raw images, which its license doesn't clearly permit
redistributing. See the generated model card for the reproduction instructions given instead.
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def build_usage_code(num_classes: int, image_size: int) -> str:
    """
    Built as a plain (non-f) string and patched with .replace() rather than embedded directly
    in an f-string -- avoids having to escape every curly brace in this nested code sample.
    """
    usage_code = """import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2048, __NUM_CLASSES__))
model.load_state_dict(torch.load("resnet50_food_best.pt", map_location="cpu"))
model.eval()

classes = open("classes.txt").read().splitlines()

transform = transforms.Compose([
    transforms.Resize(int(__IMAGE_SIZE__ * 1.14)),
    transforms.CenterCrop(__IMAGE_SIZE__),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("your_photo.jpg").convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1).squeeze(0)
top5 = probs.topk(5)
for prob, idx in zip(*top5):
    print(f"{classes[idx]}: {prob.item()*100:.1f}%")
"""
    return usage_code.replace("__NUM_CLASSES__", str(num_classes)).replace("__IMAGE_SIZE__", str(image_size))


def build_model_card(args: argparse.Namespace) -> str:
    usage_code = build_usage_code(args.num_classes, args.image_size)
    return f"""---
license: cc-by-nc-4.0
tags:
- image-classification
- pytorch
- resnet50
- food
datasets:
- food101
library_name: pytorch
---

# Food Classifier (ResNet50, {args.num_classes} classes)

ResNet50 fine-tuned via transfer learning to classify food dishes from a photo, built for a
nutrition-estimation pipeline. Trained on a merged dataset combining **Food-101** and
**UEC-Food256**.

## \u26a0\ufe0f License note

This model was trained in part on UEC-Food256, which is licensed for **non-commercial research
use only**. Treat this model under the same restriction unless you have separately obtained
permission from UEC-Food256's authors for other uses. UEC-256's raw images are NOT included in
this repo -- only `class_mapping.json` (text, category names and merge decisions) is, which
documents how to reproduce the training set from your own UEC-256 download.

## Training details

- Backbone: `torchvision.models.resnet50` (ImageNet-pretrained), two-phase transfer learning:
  frozen head, then partial fine-tuning from `{args.finetune_unfreeze_from}` onward
- Classes: {args.num_classes} (see `classes.txt`)
- Image size: {args.image_size}x{args.image_size}
- Regularization: dropout, label smoothing, MixUp/CutMix, gradient clipping, LR warmup
- Best validation accuracy: {args.best_val_accuracy:.4f}
- Final held-out test accuracy: {args.test_accuracy:.4f}

## Usage

```python
{usage_code}```

## Reproducing the training set

This repo does not redistribute UEC-256's images. To reproduce the exact merged training set:
1. Download UEC-Food256 yourself from the official source, under its own license terms.
2. Use `class_mapping.json` from this repo with the `prepare_uec256.py` / `merge_datasets.py`
   scripts (see the project's GitHub repo) to rebuild the identical merged dataset.
"""


def main():
    parser = argparse.ArgumentParser(description="Push a trained checkpoint to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                         help="e.g. your-username/food-classifier-resnet50")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best .pt checkpoint")
    parser.add_argument("--classes", type=str, required=True, help="Path to classes.txt")
    parser.add_argument("--class-mapping", type=str, required=True, help="Path to class_mapping.json")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--finetune-unfreeze-from", type=str, default="layer3")
    parser.add_argument("--best-val-accuracy", type=float, required=True)
    parser.add_argument("--test-accuracy", type=float, required=True)
    parser.add_argument("--private", action="store_true", help="Create/keep the repo private")
    parser.add_argument("--token", type=str, default=None,
                         help="HF access token with write permission. If omitted, uses a cached "
                              "login (`huggingface-cli login`) or the HF_TOKEN environment variable.")
    parser.add_argument("--staging-dir", type=str, default="./hf_upload",
                         help="Local folder to assemble the upload contents in before pushing")
    args = parser.parse_args()

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
    print(f"Repo ready: https://huggingface.co/{args.repo_id}")

    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "README.md").write_text(build_model_card(args), encoding="utf-8")
    shutil.copy2(args.checkpoint, staging_dir / "resnet50_food_best.pt")
    shutil.copy2(args.classes, staging_dir / "classes.txt")
    shutil.copy2(args.class_mapping, staging_dir / "class_mapping.json")

    api = HfApi(token=args.token)
    api.upload_folder(folder_path=str(staging_dir), repo_id=args.repo_id, repo_type="model")
    print(f"\nUploaded! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
