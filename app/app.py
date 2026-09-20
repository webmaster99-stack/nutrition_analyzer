import os

import torch
import gradio as gr

from model_loader import load_model_and_classes
from nutrition import NutritionLookup

REPO_ID = "ilian-hadzhidimitrov/food-classifier-resnet50"
CHECKPOINT_FILENAME = "resnet50_food_best.pt"        # the merged (313-class) model
CLASSES_FILENAME = "classes.txt"
NUTRITION_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nutrition_mapping.json")
IMAGE_SIZE = 320                                      # must match what the merged model was trained at
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visual de-emphasis by rank: rank 1 full strength, 2 and 3 progressively faded. Rank-based
# rather than a raw-confidence cutoff - for a top-k that's already sorted by confidence, rank
# order and confidence order are identical, so this is a simpler equivalent implementation of
# "de-emphasize the lower options."
RANK_OPACITY = [1.0, 0.75, 0.55]

FOOTER_NOTE = f"""
---
**Data sources & licensing:** Nutrition data from [USDA FoodData Central](https://fdc.nal.usda.gov/)
(public domain, CC0 1.0). This model was trained in part on UEC-Food256, which is licensed for
**non-commercial research use only** - this demo and its predictions should be treated under the
same restriction. See the [model card](https://huggingface.co/{REPO_ID}) for details.
Not all dishes have nutrition data yet; the app will say so rather than guess.
"""


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def radio_label(class_name: str, confidence: float) -> str:
    return f"{display_name(class_name)} ({confidence * 100:.1f}%)"


def render_prediction_cards(predictions: list) -> str:
    """De-emphasized HTML summary of the top-k predictions. Purely visual - actual selection
    happens through the separate Radio component, not by clicking these."""
    if not predictions:
        return ""
    rows = []
    for rank, (class_name, confidence) in enumerate(predictions):
        opacity = RANK_OPACITY[min(rank, len(RANK_OPACITY) - 1)]
        rows.append(
            f'<div style="opacity:{opacity}; padding:4px 0; font-size:{18 - rank}px;">'
            f'<strong>{display_name(class_name)}</strong> &mdash; {confidence * 100:.1f}%'
            f"</div>"
        )
    return "\n".join(rows)


def render_nutrition(selected_label: str, predictions: list, grams) -> str:
    if not predictions or not selected_label:
        return "Upload a photo to get started."

    labels = [radio_label(name, conf) for name, conf in predictions]
    idx = labels.index(selected_label) if selected_label in labels else 0
    class_name, confidence = predictions[idx]

    result = NUTRITION_LOOKUP.get(class_name, confidence)

    if not result.found:
        return (
            f"### {result.display_name}\n\n"
            f"Nutrition data not available for this dish yet. "
            f"({result.note or 'No matching entry in the nutrition database.'})"
        )

    try:
        grams_value = max(0.0, grams)

    except (TypeError, ValueError):
        grams_value = 100.0

    scaled = result.per_100g.scaled(grams_value / 100.0)

    lines = [f"### {result.display_name}", f"**Amount: {grams_value:g}g**", ""]
    if scaled.calories_kcal is not None:
        lines.append(f"- Calories: {scaled.calories_kcal:g} kcal")
    if scaled.protein_g is not None:
        lines.append(f"- Protein: {scaled.protein_g:g} g")
    if scaled.carbs_g is not None:
        lines.append(f"- Carbs: {scaled.carbs_g:g} g")
    if scaled.fat_g is not None:
        lines.append(f"- Fat: {scaled.fat_g:g} g")

    if result.serving_grams and result.serving_description:
        lines.append(f"\n_Typical serving: {result.serving_description}, {result.serving_grams:g}g_")
    if result.low_confidence_match:
        lines.append("\n_Note: this dish was matched loosely to a USDA entry -- treat values as approximate._")
    if result.matched_description:
        lines.append(f'\n_Source: USDA "{result.matched_description}"_')

    return "\n".join(lines)


@torch.no_grad()
def predict_top_k(image):
    if image is None:
        return None

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(MODEL(tensor), dim=1).squeeze(0)
    top_probs, top_idxs = probs.topk(TOP_K)
    return [(CLASS_NAMES[idx], float(prob)) for prob, idx in zip(top_probs, top_idxs)]


def on_image_change(image):
    predictions = predict_top_k(image)

    if not predictions:
        return gr.update(choices=[], value=None), "", "Upload a photo to get started.", [], 100

    labels = [radio_label(name, conf) for name, conf in predictions]
    cards_html = render_prediction_cards(predictions)

    top_class, top_confidence = predictions[0]
    top_result = NUTRITION_LOOKUP.get(top_class, top_confidence)
    default_grams = top_result.serving_grams or 100.0

    nutrition_md = render_nutrition(labels[0], predictions, default_grams)
    return gr.update(choices=labels, value=labels[0]), cards_html, nutrition_md, predictions, default_grams


def on_selection_or_portion_change(selected_label, predictions, grams):
    return render_nutrition(selected_label, predictions, grams)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Food Classifier & Nutrition Lookup") as demo:
        gr.Markdown("# Food Classifier & Nutrition Lookup")
        gr.Markdown(
            "Upload a photo of a dish. The model's top 3 guesses appear below - pick the "
            "correct one if it isn't the top pick, then adjust the amount eaten."
        )

        predictions_state = gr.State([])

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Food photo")
                cards_output = gr.HTML(label="Top matches")
                selection_radio = gr.Radio(choices=[], label="Which one is it?", interactive=True)
            with gr.Column():
                grams_input = gr.Number(value=100, label="Amount eaten (grams)", precision=0)
                nutrition_output = gr.Markdown("Upload a photo to get started.")

        gr.Markdown(FOOTER_NOTE)

        image_input.change(
            fn=on_image_change,
            inputs=image_input,
            outputs=[selection_radio, cards_output, nutrition_output, predictions_state, grams_input],
        )
        selection_radio.change(
            fn=on_selection_or_portion_change,
            inputs=[selection_radio, predictions_state, grams_input],
            outputs=nutrition_output,
        )
        grams_input.change(
            fn=on_selection_or_portion_change,
            inputs=[selection_radio, predictions_state, grams_input],
            outputs=nutrition_output,
        )

    return demo


print("Loading model...")

MODEL, CLASS_NAMES, TRANSFORM = load_model_and_classes(
    REPO_ID, 
    CHECKPOINT_FILENAME, 
    CLASSES_FILENAME, 
    IMAGE_SIZE, 
    DEVICE
)

NUTRITION_LOOKUP = NutritionLookup(NUTRITION_MAPPING_PATH)
print(f"Loaded {len(CLASS_NAMES)} classes, {len(NUTRITION_LOOKUP)} nutrition entries, device={DEVICE}")

demo = build_demo()


if __name__ == "__main__":
    demo.launch()
