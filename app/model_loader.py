"""
Downloads (and caches, via huggingface_hub's normal local cache) the merged food classifier
checkpoint + class list from Hugging Face Hub, and builds the matching preprocessing transform.
"""
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(2048, num_classes))
    return model


def build_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def load_model_and_classes(
        repo_id: str, 
        checkpoint_filename: str = "restnet50_food_best.pt", 
        classes_filename: str = "classes.txt", 
        image_size: int = 320, 
        device: str = "cpu"
    ):
    """
    Downloads the checkpoint and class list, builds the matching architecture, and loads the
    weights.

    Verifies the checkpoint's output layer size matches the number of classes read from
    classes.txt - a mismatch means the wrong pair of files got loaded (e.g. mixing the merged
    checkpoint with the Food-101-only classes.txt, an easy mistake given both live in the same
    repo). Fails loudly here rather than silently producing predictions with scrambled labels.
    """
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_filename)
    classes_path = hf_hub_download(repo_id=repo_id, filename=classes_filename)

    class_names = [c.strip() for c in Path(classes_path).read_text(encoding="utf-8").splitlines() if c.split()]
    num_classes = len(class_names)

    state_dict = torch.load(checkpoint_path, map_location=device)
    checkpoint_num_classes = state_dict["fc.1.weight"].shape[0]

    if checkpoint_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint '{checkpoint_filename}' expects {checkpoint_num_classes} classes but "
            f"'{classes_filename}' has {num_classes} -- these two files don't match each other."
        )

    model = build_model(num_classes=num_classes)
    model.load_state_dict(state_dict=state_dict)
    model.to(device=device)
    model.eval()

    transform = build_transforms(image_size=image_size)
    return model, class_names, transform