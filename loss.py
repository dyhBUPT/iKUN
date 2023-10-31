import torch
from torch import nn, log
import torch.nn.functional as F

from utils import ID_TO_POS_BBOX_NUMS


__all__ = [
    'SimilarityLoss',
]


class SimilarityLoss(nn.Module):
    """
    Ref:
    - https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
    - https://github.com/tztztztztz/eqlv2/blob/master/mmdet/models/losses/eqlv2.py
    TODO: reset
    """
    def __init__(
            self,
            rho: float = None,
            gamma: float = 2.,
            reduction: str = 'sum',
    ):
        super().__init__()
        self.rho = rho  # pos/neg samples
        self.gamma = gamma  # easy/hard samples
        self.reduction = reduction

    def forward(self, scores, labels):
        loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
        weights = 1

        if self.gamma is not None:
            logits = scores.sigmoid()
            p_t = logits * labels + (1 - logits) * (1 - labels)
            weights *= ((1 - p_t) ** self.gamma)

        if self.rho is not None:
            weights *= self.rho * labels + (1 - labels)

        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
