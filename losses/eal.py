from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSimilarityMatrix(nn.Module):
    def __init__(
        self,
        num_classes: int,
        init_matrix: torch.Tensor | None = None,
        delta_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if init_matrix is None:
            prior = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        else:
            if not isinstance(init_matrix, torch.Tensor):
                init_matrix = torch.tensor(init_matrix, dtype=torch.float32)
            if init_matrix.size() != (num_classes, num_classes):
                raise ValueError("Initial similarity matrix has invalid shape.")
            prior = init_matrix.float()
        self.register_buffer("prior_matrix", prior)
        mask = 1 - torch.eye(num_classes)
        self.register_buffer("non_diag_mask", mask)
        self.delta_scale = delta_scale
        self.similarity_params = nn.Parameter(torch.zeros(int(mask.sum().item())))

    def forward(self) -> torch.Tensor:
        n = self.prior_matrix.size(0)
        delta = torch.zeros(n, n, device=self.prior_matrix.device)
        delta[self.non_diag_mask.bool()] = torch.tanh(self.similarity_params)
        delta.fill_diagonal_(0.0)
        row_mean = delta.sum(dim=1, keepdim=True) / (n - 1)
        col_mean = delta.sum(dim=0, keepdim=True) / (n - 1)
        global_mean = delta.sum() / (n * (n - 1))
        delta = delta - row_mean - col_mean + global_mean
        delta.fill_diagonal_(0.0)
        matrix = self.prior_matrix + self.delta_scale * delta
        matrix = matrix.clamp_(0.0, 1.0)
        matrix.fill_diagonal_(0.0)
        return matrix

    def regularization(self, strength: float) -> torch.Tensor:
        deltas = self.forward() - self.prior_matrix
        return strength * torch.mean((deltas[self.non_diag_mask.bool()]) ** 2)


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss


class ExpressionAwareLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        similarity_matrix: torch.Tensor | None = None,
        alpha: float = 1.0,
        beta: float = 0.8,
        gamma: float = 2.0,
        hard_sample_threshold: float = 0.5,
        similarity_reg_strength: float = 0.01,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.hard_sample_threshold = hard_sample_threshold
        self.similarity_reg_strength = similarity_reg_strength
        self.similarity = LearnableSimilarityMatrix(
            num_classes=num_classes, init_matrix=similarity_matrix
        )
        self.focal = AdaptiveFocalLoss(alpha=alpha, gamma=gamma)
        self.register_buffer("hard_ratio", torch.tensor(0.3))
        self.momentum = 0.9

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        focal_mean = focal_loss.mean()
        normalized = focal_loss / (focal_loss.max() + 1e-8)
        hard_mask = normalized > self.hard_sample_threshold
        current_ratio = hard_mask.float().mean()
        self.hard_ratio = self.momentum * self.hard_ratio + (1 - self.momentum) * current_ratio

        if hard_mask.any():
            sim_matrix = self.similarity()
            if sim_matrix.size(0) > 1:
                sim_matrix = sim_matrix[:-1, :-1]
            probs = torch.softmax(logits[hard_mask], dim=1)
            target_rows = sim_matrix[targets[hard_mask]]
            penalties = 1.0 - target_rows
            penalties.scatter_(1, targets[hard_mask].unsqueeze(1), 0.0)
            similarity_loss = torch.sum(probs * penalties, dim=1).mean()
        else:
            similarity_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        reg_loss = self.similarity.regularization(self.similarity_reg_strength)
        return focal_mean + self.beta * similarity_loss + reg_loss

    def get_similarity(self) -> torch.Tensor:
        return self.similarity()


