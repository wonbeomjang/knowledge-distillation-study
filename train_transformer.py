import os
import argparse
import psutil
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
from tqdm import tqdm
import wandb

import loss
import model
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
parser.add_argument("--dataset",
                    choices=dict(cifar100=dataset.CIFAR100),
                    default=dataset.CIFAR100,
                    action=LookupChoices)
parser.add_argument("--num_heads", type=int, default=4, help="number of multi head attention")
parser.add_argument("--image_size", type=int, default=224, help="size of train image")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_classes", type=int, default=100, help="number of classes")
parser.add_argument("--epoch", default=100, type=int, help="the number of epoch")
parser.add_argument('--lr_decay_epochs', type=int, default=[100, 150, 170], nargs='+', help="decay epoch")
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help="decay ratio")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--checkpoint_dir", default="checkpoint", help="check point directory")
parser.add_argument("--resume", nargs='?', const=True, default=False, help="resume most recent training")
parser.add_argument("--embedding_size", type=int, default=14 * 14)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="whether use gpu or net")
parser.add_argument("--data_dir", default="dataset", help="data directory")
parser.add_argument("--num_workers", type=int, default=psutil.cpu_count())
parser.add_argument("--best", type=int, default=float('inf'))

config = parser.parse_args()
config.device = torch.device(config.device)


def test(transformer: nn.Module, teacher: nn.Module, student: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss_meter = AverageMeter()
    device = wandb.config.device

    student = student.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    for image, target in pbar:
        image: torch.Tensor = image.to(device)

        with torch.no_grad():
            student_features = student(image, True)[:-2]
            teacher_features = teacher(image, True)[:-2]

        out = transformer(teacher_features, student_features)
        loss = criterion(out, student_features)

        loss_meter.update(loss.mean().item())
        pbar.set_description(f"Validate... Loss: {loss_meter.avg: .4f}")

    result = {"val/loss": loss_meter.avg}

    student = student.train()
    return result


def train(transformer: model.DistillationTransformer, teacher: nn.Module, student: nn.Module, data_loader: DataLoader,
          criterion: nn.Module, optimizer: optim.Optimizer, lr_scheduler, wandb, run_id, val_loader: Optional[DataLoader]):
    loss_meter = AverageMeter()
    device = wandb.config.device
    criterion = criterion.to(device)

    best = wandb.config.best

    for epoch in range(wandb.config.start_epoch, wandb.config.epoch):
        pbar = tqdm(data_loader, total=len(data_loader))
        loss_meter.reset()
        for image, target in pbar:
            image: torch.Tensor = image.to(device)

            with torch.no_grad():
                teacher_features = teacher(image, True)[:-2]
                student_features = student(image, True)[:-2]

            out = transformer(src=teacher_features, trg=student_features)
            loss = criterion(out, student_features)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_meter.update(loss.mean().item())
            pbar.set_description(f"[{epoch + 1}/{wandb.config.epoch}] Loss: {loss_meter.avg: .4f}")

        result = {"train/loss": loss_meter.avg}
        loss = loss_meter.avg
        if val_loader:
            with torch.no_grad():
                val_result = test(transformer, teacher, student, criterion, val_loader)
                result.update(val_result)
            loss = result["val/loss"]

        save_info = {"run_id": run_id, "state_dict": transformer.state_dict(), "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "best": best, "net": transformer}
        if loss < best:
            best = loss
            save_info["best"] = best
            torch.save(save_info, os.path.join(wandb.config.run_dir, "best.pth"))
        torch.save(save_info, os.path.join(wandb.config.run_dir, "last.pth"))

        wandb.log(result)


if __name__ == "__main__":
    run_dir = os.path.join(os.getcwd(), config.checkpoint_dir, "run")
    attempt_make_dir(run_dir)

    train_loader, test_loader = get_loader(config)

    transformer: model.DistillationTransformer = model.DistillationTransformer(config.embedding_size, config.device,
                                                                               num_heads=config.num_heads).to(config.device)
    teacher: nn.Module = config.teacher(pretrained=True, num_classes=config.num_classes, teacher=True).to(config.device)
    student: nn.Module = config.backbone(pretrained=True, num_classes=config.num_classes, teacher=True).to(config.device)

    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.eval()

    for param in student.parameters():
        param.requires_grad = False
    student = student.eval()

    optimizer = optim.Adam(transformer.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs,
                                                  gamma=config.lr_decay_gamma)

    features = get_sample_features(train_loader, student, config.device)
    criterion = loss.TMSE().to(config.device)

    if not config.resume:
        run_dir = os.path.join(config.checkpoint_dir, "loss", "exp" + f"{len(os.listdir(run_dir)) + 1}")
        attempt_make_dir(run_dir)
        config.run_dir = run_dir
        run = wandb.init(project='attention_transformer', dir=run_dir, config=vars(config))
    else:
        if isinstance(config.resume, bool):
            run_dir = os.path.join(config.checkpoint_dir, "loss", "exp" + f"{len(os.listdir(run_dir))}")
            save_info = torch.load(os.path.join(run_dir, "last.pth"), map_location=config.device)
        else:
            run_dir = os.path.split(config.resume)
            save_info = torch.load(config.resume, map_location=config.device)

        transformer.load_state_dict(save_info["state_dict"])
        optimizer.load_state_dict(save_info["optimizer"])
        lr_scheduler.load_state_dict(save_info["lr_scheduler"])
        config.start_epoch = save_info["epoch"] + 1
        config.best = save_info["best"]
        config.run_dir = run_dir
        run = wandb.init(id=save_info["run_id"], project='attention_transformer', resume="allow", dir=run_dir,
                         config=vars(config))

    train(transformer, teacher, student, train_loader, criterion, optimizer, lr_scheduler, wandb, run.id, test_loader)

    wandb.save(os.path.join(wandb.config.run_dir, "last.pth"))
    wandb.save(os.path.join(wandb.config.run_dir, "best.pth"))
    wandb.finish()
