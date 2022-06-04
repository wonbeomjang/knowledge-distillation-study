import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import loss_at
from model.distillation_transformer import Transform
from utils import get_attention


class TKD(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy):
        super(TKD, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_criterion = base_criterion
        self.t_ratio = 50
        self.at_ratio = 50
        self.transformer = Transform(4, 4, device=device, pretrained=True)

        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, image, target, student, teacher):
        b1, b2, b3, b4, _, s_pred = student(image, True)

        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, _, t_pred = teacher(image, True)

        teacher_features = [get_attention(x).detach() for x in [t_b1, t_b2, t_b3, t_b4]]
        student_features = [get_attention(x) for x in [b1, b2, b3, b4]]

        at_loss = loss_at(b2, t_b2) + loss_at(b3, t_b3) + loss_at(b4, t_b4)

        with torch.no_grad():
            expected_feature = self.transformer(src=teacher_features, trg=student_features)
        t_loss = sum((loss_at(feature, expect) for expect, feature in zip(expected_feature, student_features)))

        at_loss *= self.at_ratio
        t_loss *= self.t_ratio

        return self.base_criterion(s_pred, target) + at_loss + t_loss, s_pred
