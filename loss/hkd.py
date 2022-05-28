import torch
import torch.nn as nn
import torch.nn.functional as F


class HKD(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy):
        super(HKD, self).__init__()
        self.T = 4
        self.lam = 16
        self.base_criterion = base_criterion

    def loss_kd(self, preds, teacher_preds):
        return F.kl_div(F.log_softmax(preds / self.T, dim=1), F.softmax(teacher_preds / self.T, dim=1),
                        reduction='batchmean') * self.lam

    def forward(self, image, target, student, teacher):
        pred = student(image)

        with torch.no_grad():
            teacher_pred = teacher(image)

        return self.base_criterion(pred, target) + self.loss_kd(pred, teacher_pred.detach()), pred