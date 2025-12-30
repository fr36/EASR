from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

from src.utils.paths import project_path


DATASETS_ROOT = project_path("datasets")


def _is_valid_file(path: str) -> bool:
    file_path = Path(path)
    if file_path.name.startswith("."):
        return False
    return file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, phase: str, transform: Callable | None = None) -> None:
        self.root = Path(root) / phase
        self.transform = transform
        self.folder = datasets.ImageFolder(str(self.root), is_valid_file=_is_valid_file)
        self.samples = self.folder.samples
        self.targets = self.folder.targets
        self.classes = self.folder.classes
        self.class_to_idx = self.folder.class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label


class TextLabelDataset(Dataset):
    def __init__(self, root: Path, label_file: Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        with Path(label_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_path = self.root / parts[0]
                if image_path.exists():
                    self.samples.append((image_path, int(parts[1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label


class RAFDBDataset(Dataset):
    def __init__(self, phase: str, transform: Callable | None = None) -> None:
        base = DATASETS_ROOT / "RAF-DB"
        self.dataset = ImageFolderDataset(base, phase=phase, transform=transform)
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[index]


class FERPlusDataset(Dataset):
    def __init__(self, phase: str, transform: Callable | None = None) -> None:
        base = DATASETS_ROOT / "FERPlus"
        self.dataset = ImageFolderDataset(base, phase=phase, transform=transform)
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[index]


class FER2013ImageFolderDataset(Dataset):
    _ALIASES = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "publictest": "val",
        "test": "test",
        "private_test": "test",
        "privatetest": "test",
    }

    def __init__(self, phase: str, transform: Callable | None = None) -> None:
        normalized = phase.lower().replace(" ", "")
        if normalized not in self._ALIASES:
            valid = ", ".join(sorted(self._ALIASES))
            raise ValueError(f"Unsupported phase '{phase}'. Valid: {valid}")
        mapped = self._ALIASES[normalized]
        base = DATASETS_ROOT / "FER2013_ImageFolder"
        self.dataset = ImageFolderDataset(base, phase=mapped, transform=transform)
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[index]


class AffectNetDataset(Dataset):
    def __init__(self, phase: str, transform: Callable | None = None, num_classes: int = 7) -> None:
        if num_classes not in (7, 8):
            raise ValueError("AffectNet num_classes must be 7 or 8.")
        folder = "trainnew" if phase == "train" else "validnew"
        self.data_path = DATASETS_ROOT / "AffectNet" / folder
        label_name = f"{num_classes}cls_train.txt" if phase == "train" else f"{num_classes}cls_val.txt"
        self.label_file = DATASETS_ROOT / "AffectNet" / label_name
        self.dataset = TextLabelDataset(self.data_path, self.label_file, transform=transform)
        self.samples = self.dataset.samples
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[index]


def concatenate_datasets(datasets_to_merge: Iterable[Dataset]) -> Dataset:
    return ConcatDataset(list(datasets_to_merge))


