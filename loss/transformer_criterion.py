import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distillation_transformer import make_attention


def make_patch(x: torch.Tensor, embedding_size: int, device: torch.device):
    x = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    x = torch.cat([x, torch.zeros(x.size(0), embedding_size - x.size(1), device=device)], dim=1)
    return x


class TMSE(nn.Module):
    def forward(self, preds: list[torch.Tensor], trgs: list[torch.Tensor]):
        loss = 0

        for pred, trg in zip(preds, trgs):
            trg = make_attention(trg)
            loss += F.mse_loss(pred, trg)

        return loss


if __name__ == "__main__":
    from model.distillation_transformer import DistillationTransformer

    batch_size = 8
    image_size = 224
    embedding_size = 56 * 56

    attention_1 = torch.rand((batch_size, 2 ** 6, image_size // (2 ** 2), image_size // (2 ** 2)))
    attention_2 = torch.rand((batch_size, 2 ** 7, image_size // (2 ** 3), image_size // (2 ** 3)))
    attention_3 = torch.rand((batch_size, 2 ** 8, image_size // (2 ** 4), image_size // (2 ** 4)))
    attention_4 = torch.rand((batch_size, 2 ** 9, image_size // (2 ** 5), image_size // (2 ** 5)))

    src = [attention_1, attention_2, attention_3, attention_4]
    trg = [attention_1, attention_2, attention_3, attention_4]

    transformer = DistillationTransformer(embedding_size, device=torch.device("cpu"))
    tmse = TMSE(embedding_size, trg, torch.device("cpu"))

    preds = transformer(src, trg)
    loss = tmse(preds, trg)
