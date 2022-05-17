import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return


def loss_at(student, teacher):
    s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

    with torch.no_grad():
        t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

    return (s_attention - t_attention).pow(2).mean()


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

        return self.base_criterion(pred, target) + self.loss_kd(pred, teacher_pred.detach())


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
        return self.base_criterion(s_pred, target) + at_loss * self.lam


def rkd_angle(student, teacher):
    # N x C
    # N x N x C

    with torch.no_grad():
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
    return loss


def rkd_distacne(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d>0].mean()
        t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d>0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
    return loss


class RKD(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy):
        super(RKD, self).__init__()
        self.distance_ratio = 25
        self.angle_ratio = 50
        self.at_ratio = 50

    def forward(self, image, target, student, teacher):
        b1, b2, b3, b4, _, s_pred = student(image, True)

        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, _, t_pred = teacher(image, True)

        distance_loss = rkd_distacne(s_pred, t_pred) * self.distance_ratio
        angle_loss = rkd_angle(s_pred, t_pred) * self.angle_ratio
        at_loss = loss_at(b2, t_b2) + loss_at(b3, t_b3) + loss_at(b4, t_b4)
        at_loss *= self.at_ratio

        return self.base_criterion(s_pred, target) + distance_loss + angle_loss + at_loss
