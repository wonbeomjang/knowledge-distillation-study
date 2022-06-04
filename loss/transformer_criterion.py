import torch
import torch.nn as nn
import torch.nn.functional as F


def make_patch(x: torch.Tensor, embedding_size: int, device: torch.device):
    x = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    x = torch.cat([x, torch.zeros(x.size(0), embedding_size - x.size(1), device=device)], dim=1)
    return x


class TMSE(nn.Module):
    def forward(self, preds, trgs):
        loss = 0

        for pred, trg in zip(preds, trgs):
            loss += F.mse_loss(pred, trg)

        return loss

