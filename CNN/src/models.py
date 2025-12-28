import os
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.unet_small import UNetSmall


def load_classes(classes_dir: str) -> list[str]:
    dataset = ImageFolder(classes_dir)
    return list(dataset.classes)


def load_cnn(model_path: str, classes_dir: str) -> tuple[torch.nn.Module, list[str], torch.device]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"missing CNN model '{model_path}'"
        )
    if not os.path.isdir(classes_dir):
        raise FileNotFoundError(f"missing dataset folder '{classes_dir}' for class mapping.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    classes = load_classes(classes_dir)
    
    return model, classes, device


def load_plate_seg(weights_path: Path, device: torch.device) -> torch.nn.Module:

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(device)
        model.eval()
        return model

    if isinstance(ckpt, dict):
        model = UNetSmall(base=32).to(device)
        model.load_state_dict(ckpt)
        model.eval()
        return model

    raise TypeError(
        f"unsupported checkpoint type for plate seg: {type(ckpt)}. "
        f"expected nn.Module or state_dict dict."
    )
