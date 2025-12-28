import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class KNNConfig:
    resized_image_width: int = 20
    resized_image_height: int = 30
    k: int = 3

    @property
    def feature_len(self) -> int:
        return self.resized_image_width * self.resized_image_height


DEFAULT_CONFIG = KNNConfig()


def load_training_data(
    classifications_path: str = "classifications.txt",
    flattened_images_path: str = "flattened_images.txt",
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(classifications_path):
        raise FileNotFoundError(f"Missing classifications file: {classifications_path}")
    if not os.path.exists(flattened_images_path):
        raise FileNotFoundError(f"Missing flattened images file: {flattened_images_path}")

    classifications = np.loadtxt(classifications_path, np.float32)
    flattened = np.loadtxt(flattened_images_path, np.float32)

    classifications = classifications.reshape((classifications.size, 1))
    return flattened, classifications


def train_knn(
    flattened_images: np.ndarray,
    classifications: np.ndarray,
) -> cv2.ml_KNearest:
    knn = cv2.ml.KNearest_create()
    knn.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)
    return knn


def save_knn(model: cv2.ml_KNearest, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)


def load_knn(model_path: str) -> cv2.ml_KNearest:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return cv2.ml.KNearest_load(model_path)


def predict_ascii_code(
    model: cv2.ml_KNearest,
    flattened_row: np.ndarray,
    k: int = DEFAULT_CONFIG.k,
) -> int:
    row = np.asarray(flattened_row, dtype=np.float32)
    row = row.reshape((1, -1))
    _ret, results, _neigh_resp, _dists = model.findNearest(row, k=k)
    return int(results[0][0])


def predict_char(
    model: cv2.ml_KNearest,
    flattened_row: np.ndarray,
    k: int = DEFAULT_CONFIG.k,
) -> str:
    return chr(predict_ascii_code(model, flattened_row, k=k))
