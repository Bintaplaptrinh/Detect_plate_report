import cv2
import numpy as np
import torch

from .utils import order_points


def predict_plate_mask(model: torch.nn.Module, bgr: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:

    h0, w0 = bgr.shape[:2]

    bgr_r = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_CUBIC)
    mask = (prob > 0.5).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def find_plate_contour(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    
    return cnt, box


def warp_plate(
    bgr: np.ndarray,
    box: np.ndarray,
    out_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:

    ordered = order_points(box)
    
    dst = np.array([
        [0, 0],
        [out_size[0] - 1, 0],
        [out_size[0] - 1, out_size[1] - 1],
        [0, out_size[1] - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped_bgr = cv2.warpPerspective(bgr, M, out_size)
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    
    return warped_bgr, warped_gray
