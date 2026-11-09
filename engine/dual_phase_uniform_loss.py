"""Measurement constraints on a uniform temporal grid."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DualPhaseUniformConsistencyLoss(nn.Module):
    def __init__(self, lambda_bracket=0.005, lambda_direction=0.01,
                 lambda_curvature=0.001, bracket_tolerance=0.05,
                 direction_slack=0.01):
        super().__init__()
        self.lambda_bracket = float(lambda_bracket)
        self.lambda_direction = float(lambda_direction)
        self.lambda_curvature = float(lambda_curvature)
        self.bracket_tolerance = float(bracket_tolerance)
        self.direction_slack = float(direction_slack)

    def forward(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        areas = torch.sigmoid(logits).mean(dim=(2, 3, 4))
        device = logits.device
        source = batch["source_indices"].to(device).long()
        ed = batch["ed_index"].to(device).long()
        es = batch["es_index"].to(device).long()
        lo, hi = torch.minimum(ed, es), torch.maximum(ed, es)
        inside = ((source >= lo[:, None]) & (source <= hi[:, None])).float()

        ed_area = batch["ed_mask"].to(device).float().mean(dim=(1, 2))
        es_area = batch["es_mask"].to(device).float().mean(dim=(1, 2))
        upper, lower = torch.maximum(ed_area, es_area), torch.minimum(ed_area, es_area)
        width = (upper - lower).clamp_min(1e-3)
        slack = self.bracket_tolerance * width
        violation = (torch.relu(areas - (upper + slack)[:, None])
                     + torch.relu((lower - slack)[:, None] - areas)) / width[:, None]
        bracket = (violation * inside).sum() / inside.sum().clamp_min(1)

        delta = (areas[:, 1:] - areas[:, :-1]) / width[:, None]
        step_inside = inside[:, :-1] * inside[:, 1:]
        direction = torch.where(ed < es, -1.0, 1.0).to(device).view(-1, 1)
        wrong = torch.relu(-(direction * delta) - self.direction_slack)
        direction_loss = (wrong * step_inside).sum() / step_inside.sum().clamp_min(1)

        curve = (areas[:, 2:] - 2 * areas[:, 1:-1] + areas[:, :-2]) / width[:, None]
        curvature = curve.square().mean()
        total = (self.lambda_bracket * bracket
                 + self.lambda_direction * direction_loss
                 + self.lambda_curvature * curvature)
        return total, {
            "mac_total": float(total.detach()),
            "mac_bracket": float(bracket.detach()),
            "mac_direction": float(direction_loss.detach()),
            "mac_curvature": float(curvature.detach()),
        }

