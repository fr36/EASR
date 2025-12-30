from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        running_correct += int((preds == targets).sum().item())
        total += inputs.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = 100.0 * running_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += float(loss.item()) * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        running_correct += int((preds == targets).sum().item())
        total += inputs.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = 100.0 * running_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


