"""Common Phase 7 Stage B training and validation helpers."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from src.etri_stage_b import stage_b_metrics


def run_stage_b_epoch(model: nn.Module, loader: Iterable[dict[str, Any]], device: torch.device,
                      optimizer: torch.optim.Optimizer | None = None,
                      max_batches: int | None = None) -> dict[str, Any]:
    training = optimizer is not None; model.train(training); criterion = nn.CrossEntropyLoss()
    total_loss, targets, logits, metadata, batches = 0.0, [], [], [], 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches: break
        embedding, target = batch["embedding"].to(device), batch["target"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(embedding); loss = criterion(output, target)
            if training: loss.backward(); optimizer.step()
        total_loss += float(loss.detach()); targets.extend(target.cpu().tolist()); logits.append(output.detach().cpu())
        for index in range(len(target)):
            metadata.append({"clip_key": batch["clip_key"][index], "participant": batch["participant"][index],
                "fold": int(batch["fold"][index]), "roi_status": batch["roi_status"][index]})
        batches += 1
    if not batches: raise ValueError("DataLoader produced no batches")
    result = {"loss": total_loss / batches, "batches": batches, "targets": targets,
              "logits": torch.cat(logits), "metadata": metadata}
    result["metrics"] = stage_b_metrics(targets, result["logits"])
    return result
