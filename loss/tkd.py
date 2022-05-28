import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distillation_transformer import AttentionEncoder, AttentionDecoder


class TKD(nn.Module):
    def forward(self, image, target, student, teacher):
        pred = student(image)
        return F.cross_entropy(pred, target), pred


if __name__ == "__main__":
    batch_size = 8
    image_size = 224

    attention_1 = torch.rand((batch_size, 2 ** 6, image_size // (2 ** 2), 224 // 4))
    attention_2 = torch.rand((batch_size, 2 ** 7, image_size // (2 ** 3), 224 // 4))
    attention_3 = torch.rand((batch_size, 2 ** 8, image_size // (2 ** 4), 224 // 4))
    attention_4 = torch.rand((batch_size, 2 ** 9, image_size // (2 ** 5), 224 // 4))

    attention_encoder = AttentionEncoder(56 * 56, torch.device("cpu"), num_heads=8)
    attention_decoder = AttentionDecoder(56 * 56, torch.device("cpu"), num_heads=8)

    enc_src = attention_encoder([attention_1, attention_2, attention_3, attention_4])
    attention = attention_decoder([attention_1, attention_2, attention_3, attention_4], enc_src)
