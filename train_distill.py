import os

import psutil
import argparse
from typing import Optional, Union, List

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
import loss
from utils import *
from data import get_loader

parser = argparse.ArgumentParser()

parser.add_argument("--backbone",
                    choices=dict(vgg11=model.VGG11,
                                 resnet18=model.ResNet18,
                                 resnet50=model.ResNet50),
                    default=model.VGG11,
                    action=LookupChoices)
parser.add_argument("--teacher",
                    choices=dict(vgg11=model.VGG11,
                                 resnet18=model.ResNet18,
                                 resnet50=model.ResNet50),
                    default=model.ResNet50,
                    action=LookupChoices)
parser.add_argument("--loss",
                    choices=dict(cle=loss.CrossEntropyLoss,
                                 hkd=loss.HKD,
                                 attention=loss.AT,
                                 rkd=loss.RKD,
                                 tkd=loss.TKD),
                    default=loss.AT,
                    action=LookupChoices)
parser.add_argument("--dataset",
                    choices=dict(cifar100=dataset.CIFAR100),
                    default=dataset.CIFAR100,
                    action=LookupChoices)

parser.add_argument("--image_size", type=int, default=224, help="size of train image")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=200, help="the number of epochs")
parser.add_argument('--lr_decay_epochs', type=int, default=[100, 150, 170], nargs='+', help="decay epoch")
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help="decay ratio")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--checkpoint_dir", default="checkpoint", help="check point directory")
parser.add_argument("--num_classes", type=int, default=100, help="the number of classes")
parser.add_argument("--resume", nargs='?', const=True, default=False, help="resume most recent training")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="whether use gpu or net")
parser.add_argument("--data_dir", default="dataset", help="data directory")
parser.add_argument("--num_workers", type=int, default=psutil.cpu_count())
parser.add_argument("--best", type=int, default=0)
parser.add_argument("--test", nargs='?', const=True, default=False, help="resume most recent training")

config = parser.parse_args()
config.device = torch.device(config.device)


def test(student: nn.Module, teacher: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = wandb.config.device

    student = student.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    for image, target in pbar:
        image: torch.Tensor = image.to(device)
        target: torch.Tensor = target.to(device)

        out: Union[torch.Tensor, torch.Tensor] = criterion(image, target, student, teacher)
        loss, predict = out

        predict = predict.argmax(dim=1)

        loss_meter.update(loss.mean().item())
        acc_meter.update((predict == target).sum().item() / image.shape[0])
        pbar.set_description(f"Validate... Loss: {loss_meter.avg: .4f}, Acc: {acc_meter.avg: .4f}")

    result = {"val/acc": acc_meter.avg, "val/loss": loss_meter.avg}

    student = student.train()
    return result


def train(student: nn.Module, teacher: nn.Module, data_loader: DataLoader, criterion: nn.Module,
          optimizer: optim.Optimizer, lr_scheduler, wandb, run_id, val_loader: Optional[DataLoader]):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = wandb.config.device
    criterion = criterion.to(device)

    student = student.to(device).train()

    best = wandb.config.best

    for epoch in range(wandb.config.start_epoch, wandb.config.epoch):
        pbar = tqdm(data_loader, total=len(data_loader))
        loss_meter.reset()
        acc_meter.reset()
        for image, target in pbar:
            image: torch.Tensor = image.to(device)
            target: torch.Tensor = target.to(device)
            out: Union[torch.Tensor, torch.Tensor] = criterion(image, target, student, teacher)
            loss, predict = out

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            predict = predict.argmax(dim=1)

            loss_meter.update(loss.mean().item())
            acc_meter.update((predict == target).sum().item() / image.shape[0])

            pbar.set_description(f"[{epoch + 1}/{wandb.config.epoch}] Loss: {loss_meter.avg: .4f}, "
                                 f"Acc: {acc_meter.avg: .4f}")

        result = {"train/acc": acc_meter.avg, "train/loss": loss_meter.avg}
        acc = acc_meter.avg
        if val_loader:
            with torch.no_grad():
                val_result = test(student, teacher, criterion, val_loader)
                result.update(val_result)
            acc = result["val/acc"]

        save_info = {"run_id": run_id, "state_dict": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "best": best, "net": student}
        if acc > best:
            best = acc
            save_info["best"] = best
            torch.save(save_info, os.path.join(wandb.config.run_dir, "best.pth"))
        torch.save(save_info, os.path.join(wandb.config.run_dir, "last.pth"))

        wandb.log(result)


if __name__ == "__main__":
    run_dir = os.path.join(os.getcwd(), config.checkpoint_dir, "run")
    attempt_make_dir(run_dir)

    teacher: nn.Module = config.teacher(pretrained=True, num_classes=config.num_classes, teacher=True).to(config.device)
    student: nn.Module = config.backbone(pretrained=True, num_classes=config.num_classes).to(config.device)

    print(f"Teacher model is {teacher}")
    print(f"Student model is {student}")

    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.eval()

    optimizer = optim.Adam(student.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs,
                                                  gamma=config.lr_decay_gamma)
    criterion = config.loss().to(config.device)

    if not config.resume:
        run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir)) + 1}")
        attempt_make_dir(run_dir)
        config.run_dir = run_dir
        run = wandb.init(project='knowledge_distillation', dir=run_dir, config=vars(config))
    else:
        if isinstance(config.resume, bool):
            run_dir = os.path.join(config.checkpoint_dir, "run", "exp" + f"{len(os.listdir(run_dir))}")
            save_info = torch.load(os.path.join(run_dir, "last.pth"), map_location=config.device)
        else:
            run_dir = os.path.split(config.resume)
            save_info = torch.load(config.resume, map_location=config.device)

        student.load_state_dict(save_info["state_dict"])
        optimizer.load_state_dict(save_info["optimizer"])
        lr_scheduler.load_state_dict(save_info["lr_scheduler"])
        config.start_epoch = save_info["epoch"] + 1
        config.best = save_info["best"]
        config.run_dir = run_dir
        run = wandb.init(id=save_info["run_id"], project='knowledge_distillation', resume="allow", dir=run_dir,
                         config=vars(config))

    train_loader, test_loader = get_loader(config)

    print("Start test teacher")
    result = test(teacher, teacher, loss.CrossEntropyLoss().to(wandb.config.device), test_loader)
    print(f"Teacher matrix: Loss{result['val/loss']:.4f}, Accuracy{result['val/acc']:.4f}")

    if not config.test:
        train(student, teacher, train_loader, criterion, optimizer, lr_scheduler, wandb, run.id, test_loader)

    student.load_state_dict(torch.load(os.path.join(wandb.config.run_dir, "last.pth"), map_location=wandb.config.device)["state_dict"])
    result = test(student, teacher, criterion, test_loader)

    wandb.save(os.path.join(wandb.config.run_dir, "last.pth"))
    wandb.save(os.path.join(wandb.config.run_dir, "best.pth"))

    print(f"Best result... Loss: {result['val/loss']}, Acc: {result['val/acc']}")
    result['test/loss'], result['test/acc'] = result['val/loss'], result['val/acc']
    wandb.log(result)

    wandb.finish()
