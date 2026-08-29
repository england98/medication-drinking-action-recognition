"""Small, explicit Stage A training/evaluation orchestration helpers."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from src.stage_a import StageAMobileNetV3, evaluate_predictions


def is_better_checkpoint(value: float, loss: float, best_value: float, best_loss: float) -> bool:
    """Higher monitor wins; exact ties use lower loss and then keep the earlier epoch."""
    return value > best_value or (value == best_value and loss < best_loss)


def make_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [value for value in model.parameters() if value.requires_grad]
    if name == "adam": return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw": return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd": return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(model: StageAMobileNetV3, loader: Iterable[dict[str, Any]], device: torch.device,
              optimizer: torch.optim.Optimizer | None = None, max_batches: int | None = None) -> dict[str, Any]:
    training = optimizer is not None; model.train(training); criterion = nn.CrossEntropyLoss()
    total_loss, targets, logits, video_ids, batches = 0.0, [], [], [], 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches: break
        images, target = batch["image"].to(device), batch["target"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(images); loss = criterion(output, target)
            if training: loss.backward(); optimizer.step()
        total_loss += float(loss.detach()); targets.extend(target.cpu().tolist()); logits.append(output.detach().cpu())
        video_ids.extend(list(batch["video_id"])); batches += 1
    if not batches: raise ValueError("DataLoader produced no batches")
    result = {"loss": total_loss / batches, "batches": batches}
    # Partial smoke batches need not contain all three frames per video.
    if max_batches is None: result["metrics"] = evaluate_predictions(targets, torch.cat(logits), video_ids)
    return result
