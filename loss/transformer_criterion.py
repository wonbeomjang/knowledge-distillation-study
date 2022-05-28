import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distillation_transformer import DistillationTransformer

def make_patch(x: torch.Tensor, embedding_size: int):
    x = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    x = torch.cat([x, torch.zeros(x.size(0), embedding_size - x.size(1))], dim=1)
    return x


class TMSE(nn.Module):
    def __init__(self, embedding_size: int, feature_example: list[torch.Tensor], device: torch.device):
        super(TMSE, self).__init__()
        self.embedding_size = embedding_size

        mask = []
        for i, x in enumerate(feature_example):
            x = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
            mask += [torch.cat([torch.ones(x.shape, device=device, requires_grad=False),
                                torch.zeros([x.size(0), self.embedding_size - x.size(1)], device=device, requires_grad=False)], dim=1)]
        self.mask = torch.stack(mask, dim=1)

    def forward(self, preds: torch.Tensor, student_features: list[torch.Tensor]):
        student_features = torch.stack([make_patch(t, self.embedding_size) for t in student_features], dim=1)
        preds *= self.mask

        return F.mse_loss(preds, student_features)



if __name__ == "__main__":
    batch_size = 8
    image_size = 224

    attention_1 = torch.rand((batch_size, 2 ** 6, image_size // (2 ** 2), 224 // 4))
    attention_2 = torch.rand((batch_size, 2 ** 7, image_size // (2 ** 3), 224 // 4))
    attention_3 = torch.rand((batch_size, 2 ** 8, image_size // (2 ** 4), 224 // 4))
    attention_4 = torch.rand((batch_size, 2 ** 9, image_size // (2 ** 5), 224 // 4))

    transformer = DistillationTransformer(56 * 56, device=torch.device("cpu"))
    tmse = TMSE(56 * 56, [attention_1, attention_2, attention_3, attention_4], torch.device("cpu"))

    preds = transformer([attention_1, attention_2, attention_3, attention_4], [attention_1, attention_2, attention_3, attention_4])
    loss = tmse(preds, [attention_1, attention_2, attention_3, attention_4])
    print(loss)
