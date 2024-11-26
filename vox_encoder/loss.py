import torch
import torch.nn.functional as F
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive=10.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, pred, target):
        # Weight loss more heavily for occupied voxels (1s)
        weight = torch.ones_like(target)
        weight[target == 1] = self.weight_positive
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction="mean")
        return bce


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output + 1e-8)) + weights[0] * (
            (1 - target) * torch.log(1 - output + 1e-8)
        )
    else:
        loss = target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8)
    return torch.neg(torch.mean(loss))


# Usage
# criterion = WeightedBCELoss(weight_positive=10.0)
