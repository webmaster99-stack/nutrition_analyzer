"""
Runtime nutrition lookup: maps a predicted class name to calories + macros.

Reads the offline-built `nutrition_mapping.json` (see build_nutrition_mapping.py) as a plain
dictionary - no network, no API key, no rate limits, fully deterministic.

Designed to be imported by a UI layer:

    from nutrition import NutritionLookup

    lookup = NutritionLookup("nutrition_mapping.json")

    # Single class
    result = lookup.get("apple_pie")

    # Top-k from the model, for "let the user pick" UIs
    options = lookup.get_top_k([("apple_pie", 0.62), ("bread_pudding", 0.21), ("cheesecake", 0.08)])

Values are reported BOTH per 100g and per typical serving, clearly labelled - a photo alone
doesn't determine how much food is present, so neither number is a claim about the specific
photo. Per-serving is only populated when the source data had a real serving weight; it's
None otherwise rather than being guessed at.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Macros:
    calories_kcal: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None

    def scaled(self, factor: float) -> "Macros":
        """Scales all present values by `factor`, leaving missing ones as None."""
        def scale(value):
            return round(value * factor, 1) if value is not None else None

        return Macros(
            calories_kcal=scale(self.calories_kcal),
            protein_g=scale(self.protein_g),
            carbs_g=scale(self.carbs_g),
            fat_g=scale(self.fat_g)
        )


@dataclass
class NutritionResult:
    class_name: str 
    display_name: str
    confidence: float | None # the model's softmax probability, if provided
    found: bool # False when we have no nutrition data for this class
    per_100g: Macros | None = None
    per_serving: Macros | None = None
    serving_grams: float | None = None
    serving_description: str | None = None
    matched_description: str | None = None
    low_confidence_match: bool = False
    note: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["per_100g"] = asdict(self.per_100g) if self.per_100g else None
        data["per_serving"] = asdict(self.per_serving) if self.per_serving else None
        return data


class NutritionLookup:
    def __init__(self, mapping_path: str | Path):
        self.mapping_path = Path(mapping_path)

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"No nutrition mapping at {self.mapping_path}. "
                f"Build one first with build_nutrition_mapping.py"
            )

        self._mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._mapping)

    @staticmethod
    def _display_name(class_name: str) -> str:
        return class_name.replace("_", " ").title()

    def get(self, class_name: str, confidence: float | None) -> NutritionResult:
        """
        Looks up one class. Always returns a NutritionResult - check `.found` rather than
        expecting None, so UI code has a display name and confidence to show even for classes
        with no nutrition data.
        """
        entry = self._mapping.get(class_name)
        display_name = self._display_name(class_name)

        if entry is None:
            return NutritionResult(
                class_name=class_name, display_name=display_name, confidence=confidence,
                found=False, note="Class not present in the nutrition mapping.",
            )

        per_100g_raw = entry.get("per_100g")

        if not per_100g_raw:
            return NutritionResult(
                class_name=class_name,
                display_name=display_name,
                confidence=confidence,
                found=False,
                note=entry.get("note") or "No nutrition data available for this dish."
            )

        per_100g = Macros(
            calories_kcal=per_100g_raw.get("calories_kcal"),
            protein_g=per_100g_raw.get("protein_g"),
            carbs_g=per_100g_raw.get("carbs_g"),
            fat_g=per_100g_raw.get("fat_g")
        )

        serving = entry.get("serving") or {}
        serving_grams = serving.get("grams")
        per_serving = per_100g.scaled(serving_grams / 100.0) if serving_grams else None

        return NutritionResult(
            class_name=class_name,
            display_name=display_name,
            confidence=confidence,
            found=True,
            per_100g=per_100g,
            per_serving=per_serving,
            serving_grams=serving_grams,
            serving_description=serving.get("description"),
            matched_description=entry.get("matched_description"),
            low_confidence_match=bool(entry.get("needs_review")),
        )

    def get_top_k(self, predictions: list[tuple[str, float]]) -> list[NutritionResult]:
        """
        Looks up several candidate classes at once, for a "model suggests, user picks" UI.

        `predictions` is a list of (class_name, confidence) in the model's ranked order -
        exactly the shape of a softmax topk. Order is preserved.
        """
        return [self.get(class_name, confidence) for class_name, confidence in predictions]


def format_result(result: NutritionResult) -> str:
    """Human-readable one-block summary. Useful for CLI checks and as a UI reference."""
    lines = [f"{result.display_name}"]
    if result.confidence is not None:
        lines[0] += f"  ({result.confidence * 100:.1f}% confidence)"

    if not result.found:
        lines.append(f"  {result.note}")
        return "\n".join(lines)

    def macro_line(macros: Macros, label: str) -> str:
        parts = []
        if macros.calories_kcal is not None:
            parts.append(f"{macros.calories_kcal:g} kcal")
        if macros.protein_g is not None:
            parts.append(f"protein {macros.protein_g:g}g")
        if macros.carbs_g is not None:
            parts.append(f"carbs {macros.carbs_g:g}g")
        if macros.fat_g is not None:
            parts.append(f"fat {macros.fat_g:g}g")
        return f"  {label}: " + ", ".join(parts)

    lines.append(macro_line(result.per_100g, "Per 100g"))
    if result.per_serving and result.serving_grams:
        label = f"Per serving ({result.serving_description or 'serving'}, {result.serving_grams:g}g)"
        lines.append(macro_line(result.per_serving, label))
    else:
        lines.append("  Per serving: no serving size available for this dish")

    if result.matched_description:
        lines.append(f"  Source: USDA \"{result.matched_description}\"")
    if result.low_confidence_match:
        lines.append("  Note: approximate -- this dish was matched loosely to a USDA entry.")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Look up nutrition for a class name")
    parser.add_argument("class_names", nargs="+", help="One or more class names, e.g. apple_pie")
    parser.add_argument("--mapping", type=str, default="./nutrition_mapping.json")
    args = parser.parse_args()

    lookup = NutritionLookup(args.mapping)
    print(f"Loaded {len(lookup)} entries from {args.mapping}\n")
    for name in args.class_names:
        print(format_result(lookup.get(name)))
        print()