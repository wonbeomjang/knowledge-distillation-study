import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_at(student, teacher):
    s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

    with torch.no_grad():
        t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))
    return (s_attention - t_attention).pow(2).mean()


class AT(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy):
        super(AT, self).__init__()
        self.base_criterion = base_criterion
        self.lam = 50

    def forward(self, image, target, student, teacher):
        b1, b2, b3, b4, _, s_pred = student(image, True)

        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, t_pool, t_pred = teacher(image, True)

        at_loss = loss_at(b2, t_b2) + loss_at(b3, t_b3) + loss_at(b4, t_b4)
        return self.base_criterion(s_pred, target) + at_loss * self.lam, s_pred