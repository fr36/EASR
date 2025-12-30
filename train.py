from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models, transforms

from data.datasets import (
    AffectNetDataset,
    FER2013ImageFolderDataset,
    FERPlusDataset,
    RAFDBDataset,
    concatenate_datasets,
)
from engine import evaluate, train_one_epoch
from losses.eal import ExpressionAwareLoss
from models.easr import EASRModel
from utils.matrices import (
    get_emotion_labels,
    get_similarity_matrix,
    load_similarity_from_json,
)
from utils.paths import ensure_dir, project_path


DEFAULT_MATRIX = {
    "rafdb": "rafdb",
    "fer2013": "fer2013",
    "ferplus": "ferplus8",
    "affectnet": "affectnet7",
}


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            target_dist = torch.zeros_like(log_probs)
            target_dist.fill_(self.eps / (num_classes - 1))
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
        loss = -(target_dist * log_probs).sum(dim=-1)
        return loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PAE model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rafdb",
        choices=["rafdb"],
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        choices=["rafdb", "affectnet7", "affectnet8", "ferplus8", "fer2013"],
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="eal",
        choices=["ce", "ce_ls", "eal"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pfr-iter", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--s0-json",
        type=str,
        default=None,
    )
    parser.add_argument("--fixed-s0", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, eval_transforms


def build_datasets(args: argparse.Namespace, num_classes: int):
    train_tf, eval_tf = build_transforms()
    if args.dataset == "rafdb":
        train_dataset = RAFDBDataset(phase="train", transform=train_tf)
        val_dataset = RAFDBDataset(phase="test", transform=eval_tf)
    elif args.dataset == "ferplus":
        train_dataset = FERPlusDataset(phase="train", transform=train_tf)
        val_dataset = FERPlusDataset(phase="test", transform=eval_tf)
    elif args.dataset == "affectnet":
        train_dataset = AffectNetDataset(phase="train", transform=train_tf, num_classes=num_classes)
        val_dataset = AffectNetDataset(phase="test", transform=eval_tf, num_classes=num_classes)
    elif args.dataset == "fer2013":
        train_ds = FER2013ImageFolderDataset(phase="train", transform=train_tf)
        val_ds = FER2013ImageFolderDataset(phase="val", transform=train_tf)
        train_dataset = concatenate_datasets([train_ds, val_ds])
        val_dataset = FER2013ImageFolderDataset(phase="test", transform=eval_tf)
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'.")
    return train_dataset, val_dataset


def build_dataloaders(
    args: argparse.Namespace,
    train_dataset,
    val_dataset,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def load_backbone(weights_path: str | None, device: torch.device) -> nn.Module:
    backbone = models.resnet18(weights=None)
    if weights_path:
        weights_file = Path(weights_path).expanduser()
        if not weights_file.exists():
            raise FileNotFoundError(f"Backbone weights not found at '{weights_path}'.")
        checkpoint = torch.load(weights_file, map_location=device)
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
        backbone.load_state_dict(state_dict, strict=False)
    return backbone


def build_optimizer(model: nn.Module, opt_name: str, lr: float, weight_decay: float):
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    params = [
        {"params": nodecay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": weight_decay},
    ]
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        return SGD(params, lr=lr, momentum=0.9)
    if opt_name == "adam":
        return Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if opt_name == "adamw":
        return AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    raise ValueError(f"Unsupported optimizer '{opt_name}'.")


def main() -> None:
    args = parse_args()
    if args.matrix is None:
        args.matrix = DEFAULT_MATRIX[args.dataset]

    emotion_classes = get_emotion_labels(args.matrix)
    num_classes = len(emotion_classes)

    if args.dataset in {"rafdb", "fer2013"} and num_classes != 7:
        raise ValueError("Selected dataset requires a 7-class similarity matrix.")
    if args.dataset == "ferplus" and num_classes != 8:
        raise ValueError("FERPlus requires an 8-class similarity matrix.")

    if args.output_dir:
        output_path = Path(args.output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = project_path(output_path)
    else:
        output_path = project_path("outputs")
    output_dir = ensure_dir(output_path)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_dataset, val_dataset = build_datasets(args, num_classes)
    train_loader, val_loader = build_dataloaders(args, train_dataset, val_dataset, device)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    similarity_matrix = get_similarity_matrix(args.matrix)
    if args.loss == "eal" and args.s0_json:
        try:
            similarity_matrix = load_similarity_from_json(Path(args.s0_json), emotion_classes)
            print(f"Loaded similarity matrix from {args.s0_json}")
        except Exception as exc:
            print(f"Failed to load similarity JSON: {exc}. Falling back to predefined matrix.")

    model = EASRModel(
        backbone=load_backbone(args.backbone_weights, device),
        num_classes=num_classes,
        dropout=args.dropout,
        pfr_iterations=args.pfr_iter,
    ).to(device)

    lr = args.lr if args.lr is not None else 1e-3 * (args.batch_size / 128)

    criterion: nn.Module
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "ce_ls":
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = ExpressionAwareLoss(num_classes=num_classes, similarity_matrix=similarity_matrix)
    criterion = criterion.to(device)

    optimizer = build_optimizer(model, args.opt, lr, args.weight_decay)
    if isinstance(criterion, ExpressionAwareLoss):
        if args.fixed_s0:
            for param in criterion.parameters():
                param.requires_grad_(False)
        else:
            optimizer.add_param_group({"params": criterion.parameters(), "weight_decay": 0.0})

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    start_epoch = 0
    best_acc = 0.0
    best_state = None
    log_path = output_dir / f"training_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_acc = float(checkpoint.get("best_acc", 0.0))
        print(f"Resumed from checkpoint at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        with log_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_metrics['loss']:.4f}",
                    f"{train_metrics['accuracy']:.2f}",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['accuracy']:.2f}",
                    f"{current_lr:.6f}",
                ]
            )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"- Train: {train_metrics['loss']:.4f} | {train_metrics['accuracy']:.2f}% "
            f"- Val: {val_metrics['loss']:.4f} | {val_metrics['accuracy']:.2f}% "
            f"- LR: {current_lr:.6f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "config": vars(args),
            }
            torch.save(best_state, output_dir / "best_model.pth")
            print(f"New best validation accuracy: {best_acc:.2f}%")

    final_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": args.epochs,
        "best_acc": best_acc,
        "config": vars(args),
    }
    torch.save(final_state, output_dir / "last_model.pth")
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()


