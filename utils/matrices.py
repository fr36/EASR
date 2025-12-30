import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


_SIMILARITY_DATA = {
    "rafdb": [
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0],
        [0.3, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0],
    ],
    "fer2013": [
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0],
        [0.3, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0],
    ],
    "affectnet7": [
        [0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.0],
        [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0],
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.2],
        [0.1, 0.0, 0.2, 0.0, 0.2, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.0],
    ],
    "affectnet8": [
        [0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1],
        [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.2],
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.2, 0.1],
        [0.1, 0.0, 0.2, 0.0, 0.2, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.1],
        [0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0],
    ],
    "ferplus8": [
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1],
        [0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
        [0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1],
        [0.1, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1, 0.0],
    ],
}

_CLASS_LABELS = {
    "rafdb": ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
    "fer2013": ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
    "affectnet7": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"],
    "affectnet8": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
    "ferplus8": ["Anger", "Disgust", "Surprise", "Happy", "Sad", "Fear", "Neutral", "Contempt"],
}


def get_similarity_matrix(name: str) -> torch.Tensor:
    key = name.lower()
    if key not in _SIMILARITY_DATA:
        raise ValueError(f"Unknown similarity matrix '{name}'.")
    return torch.tensor(_SIMILARITY_DATA[key], dtype=torch.float32)


def get_emotion_labels(name: str) -> List[str]:
    key = name.lower()
    if key not in _CLASS_LABELS:
        raise ValueError(f"Unknown matrix class mapping '{name}'.")
    return _CLASS_LABELS[key]


def load_similarity_from_json(path: Path, target_classes: Iterable[str]) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    src_classes = [cls.lower() for cls in payload["classes"]]
    matrix = np.array(payload["matrix"], dtype=np.float32)

    target_lower = [cls.lower() for cls in target_classes]

    if "contempt" not in target_lower and "contempt" in src_classes:
        idx = src_classes.index("contempt")
        matrix = np.delete(matrix, idx, axis=0)
        matrix = np.delete(matrix, idx, axis=1)
        src_classes.pop(idx)

    order = []
    for cls in target_lower:
        if cls not in src_classes:
            raise ValueError(f"Class '{cls}' missing in similarity json.")
        order.append(src_classes.index(cls))

    reordered = matrix[np.ix_(order, order)]
    np.fill_diagonal(reordered, 1.0)
    reordered = np.clip(reordered, 0.0, 1.0)
    return torch.tensor(reordered, dtype=torch.float32)


