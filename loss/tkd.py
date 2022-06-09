import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_attention


def loss_at(s_attention, t_attention):
    return (s_attention - t_attention).pow(2).mean()


class TKD(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy):
        super(TKD, self).__init__()

    def forward(self, image, target, student, teacher):
        b1, b2, b3, b4, _, s_pred = student(image, True)

        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, _, t_pred = teacher(image, True)

        teacher_features = [get_attention(x).detach() for x in [t_b1, t_b2, t_b3, t_b4]]
        student_features = [get_attention(x) for x in [b1, b2, b3, b4]]

        student_features = student.adj(teacher_features, student_features)

        b1, b2, b3, b4 = student_features
        t_b1, t_b2, t_b3, t_b4 = teacher_features

        at_loss = loss_at(b2, t_b2) + loss_at(b3, t_b3) + loss_at(b4, t_b4) + loss_at(b4, t_b4)

        return self.base_criterion(s_pred, target) + at_loss, s_pred
