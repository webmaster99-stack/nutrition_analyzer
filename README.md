# Food Classifier

A ResNet50-based food photo classifier, built as the first stage of a nutrition-estimation app
(photo → dish → nutrition facts lookup). Trained via transfer learning on a merged
**Food-101 + UEC-Food256** dataset (313 classes), with a **Food-101-only baseline** (101
classes) also published for comparison.

## Results

| Model | Classes | Image size | Val accuracy | Test accuracy |
|---|---|---|---|---|
| **Merged (main)** | 313 | 320×320 | 78.9% | 84.0% |
| Food-101-only baseline | 101 | 224×224 | 81% | 85% |

Both checkpoints are published on Hugging Face Hub:
**https://huggingface.co/ilian-hadzhidimitrov/food-classifier-resnet50** 

## ⚠️ License note

**UEC-Food256 is licensed for non-commercial research use only.** The merged (313-class) model
inherits this restriction. The Food-101-only baseline does not, since Food-101 itself is
permissively licensed. See the model card on Hugging Face Hub for details, and see
[Licensing](#licensing) below before using either model commercially.

## Repo structure

```
nutrition_analyzer/
├── .dvc/ # DVC repo files
│   ├── .gitignore
│   └── config
├── .dvcignore
├── .gitignore
├── data/ 
│   ├── .gitignore
│   ├── food-101/
│   │   ├── .gitignore
│   │   └── food-101.dvc
│   ├── uec256_cropped.dvc
│   └── uecfood256/
│       └── .gitignore
├── food_classifier/ # tensor board runs
│   └── runs/
├── LICENSE
├── notebooks/
│   ├── eda_and_problem_statement.ipynb 
│   ├── model_training_food101.ipynb # Food-101 baseline model
│   └── model_training_food101+uec256.ipynb # Food-101 + UEC-Food 256 model 
├── README.md
├── requirements.txt # Project dependencies
└── src/
    ├── build_class_mapping.py
    ├── merge_datasets.py
    ├── prepare_uec256.py
    └── push_to_hub.py
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for training (the notebook is built around a free Colab T4). CPU
is fine for inference only.

## Reproducing the full project

### 1. Get Food-101

Nothing to do manually - `torchvision.datasets.Food101(download=True)` fetches it automatically
the first time the notebook/scripts run.

### 2. Get UEC-Food256

Download it yourself, under its own license terms:

```bash
mkdir -p ./data/uecfood256 && cd ./data/uecfood256
wget http://foodcam.mobi/dataset256.zip && unzip -q dataset256.zip
cd ../..
```

### 3. Crop UEC-256 to bounding boxes

```bash
python prepare_uec256.py --uec-root ./data/uecfood256/UECFOOD256 --output-dir ./data/uec256_cropped
```
Sanity-check a few of the printed sample crop paths visually before trusting the full run.

### 4. Build (or reuse) the class mapping

```bash
python build_class_mapping.py --uec-root ./data/uecfood256/UECFOOD256 --food101-root ./data
```
This proposes a fuzzy-matched mapping to `class_mapping.json`. **Review every entry by hand** --
merge genuine duplicates (e.g. UEC's "ramen noodle" into Food-101's existing "ramen"), and leave
genuinely distinct-but-similar dishes as separate classes. Re-running this script is
non-destructive: it preserves any entries you've already reviewed and only proposes new ones for
categories it hasn't seen before.

If you already have a reviewed `class_mapping.json` you can skip
straight to the next step.

### 5. Train

Open `model_training_food101` or `model_training_food101+uec256.ipynb` in Google Colab (a T4 GPU runtime is enough).
Run the cells top to bottom

If your Colab runtime disconnects mid-training, re-run from the top and set `CFG.resume` to the
path of the last `..._latest.pt` checkpoint - it resumes from the correct phase/epoch
automatically.

### 6. Publish to Hugging Face Hub

```bash
python push_to_hub.py \
    --repo-id your-username/food-classifier-resnet50 \
    --checkpoint ./checkpoints/resnet50_food_best.pt \
    --classes ./checkpoints/classes.txt \
    --class-mapping ./class_mapping.json \
    --num-classes 313 --image-size 320 --finetune-unfreeze-from layer3 \
    --best-val-accuracy 0.789 --test-accuracy 0.840 \
    --food101-checkpoint ./checkpoints_old/resnet50_food101_best.pt \
    --food101-classes ./checkpoints_old/classes.txt \
    --food101-image-size 224 \
    --food101-best-val-accuracy 0.81 --food101-test-accuracy 0.85
```

Requires `huggingface-cli login` beforehand (or `--token` / an `HF_TOKEN` env var). The
`--food101-*` arguments are optional -- omit them to publish only the merged model. This never
uploads UEC-256's raw images (only weights, class lists, and the text-only `class_mapping.json`)
-- see the [Licensing](#licensing) note.

## Using a trained model without retraining

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="your-username/food-classifier-resnet50")
```

Then follow the `## Usage` code sample on the model card (also printed by `push_to_hub.py`) to
load the checkpoint and run inference on a new photo.

## Licensing

- **Food-101**: permissively licensed, redistribution and commercial use are fine.
- **UEC-Food256**: **non-commercial research use only**. Its raw images are never redistributed
  by anything in this repo -- `class_mapping.json` is text only (category names and merge
  decisions), letting anyone reproduce the merged dataset from their own UEC-256 download rather
  than from a redistributed copy.
- **The merged (313-class) model** inherits UEC-256's non-commercial restriction, since it was
  trained in part on that data. **The Food-101-only baseline does not.**

## Possible next steps

- Move on to the nutrition-lookup stage (predicted class → USDA FoodData Central) -- this
  repo only covers dish classification, step one of the original goal.
- Portion-size / calorie estimation from the photo itself (see the Nutrition5k dataset) --
  classification alone only gives "typical serving" nutrition, not what's actually in a specific
  photo.
- A lightweight demo app (FastAPI + the published model + nutrition lookup), which would also
  start a real feedback loop from user-submitted photos and corrections.
- Consider top-k or nutrition-equivalent-class accuracy alongside top-1 -- several of the
  remaining confused class pairs (different cuts of meat, similar desserts) matter less for a
  nutrition app than for a pure classification benchmark.