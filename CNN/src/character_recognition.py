import numpy as np
import torch
from torchvision import transforms

from .config import CNN_IMAGE_SIZE



DIGITS = set("0123456789")
ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")



_char_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.ToTensor(),
])


def predict_char_cnn(model: torch.nn.Module, classes: list[str], device: torch.device, img_roi: np.ndarray, digits_only: bool = False,) -> str:

    if img_roi is None or img_roi.size == 0:
        return "?"

    x = _char_transform(img_roi).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        
        if digits_only:

            masked_logits = logits.clone()
            for i, cls in enumerate(classes):
                if cls.upper() not in DIGITS:
                    masked_logits[0, i] = float('-inf')
            idx = int(torch.argmax(masked_logits, dim=1).item())
        else:
            idx = int(torch.argmax(logits, dim=1).item())
    
    return classes[idx]


def predict_char_with_position(model: torch.nn.Module, classes: list[str], device: torch.device, img_roi: np.ndarray, position: int) -> str:

    if img_roi is None or img_roi.size == 0:
        return "?"
    
    x = _char_transform(img_roi).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        masked_logits = logits.clone()
        
        if position in (0, 1):
            
            for i, cls in enumerate(classes):
                if cls.upper() not in DIGITS:
                    masked_logits[0, i] = float('-inf')
        elif position == 2:

            for i, cls in enumerate(classes):
                if cls.upper() not in ALPHABET:
                    masked_logits[0, i] = float('-inf')
        else:

            for i, cls in enumerate(classes):
                if cls.upper() not in DIGITS:
                    masked_logits[0, i] = float('-inf')
        
        idx = int(torch.argmax(masked_logits, dim=1).item())
    
    return classes[idx]


def is_alphabet(char: str) -> bool:
    return char.upper() in ALPHABET


def is_digit(char: str) -> bool:
    return char.upper() in DIGITS
