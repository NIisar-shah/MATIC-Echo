"""Differentiable multiscale contour supervision for LV segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_boundary(values: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = 2 * radius + 1
    maximum = F.max_pool2d(values, kernel, stride=1, padding=radius)
    minimum = -F.max_pool2d(-values, kernel, stride=1, padding=radius)
    return (maximum - minimum).clamp(0, 1)


class MultiscaleBoundaryDiceLoss(nn.Module):
    """Match probability and target contours at several spatial tolerances."""

    def __init__(self, radii=(1, 2, 4), eps=1e-6):
        super().__init__()
        self.radii = tuple(int(radius) for radius in radii)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        targets = targets.float()
        losses = []
        for radius in self.radii:
            predicted_boundary = _soft_boundary(probabilities, radius).flatten(1)
            target_boundary = _soft_boundary(targets, radius).flatten(1)
            intersection = (predicted_boundary * target_boundary).sum(1)
            denominator = predicted_boundary.sum(1) + target_boundary.sum(1)
            losses.append(1 - ((2 * intersection + self.eps) /
                               (denominator + self.eps)).mean())
        return torch.stack(losses).mean()
