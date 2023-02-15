import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from networks.centernet import CenterNet
from utils.dataset import CenterNetDataset
from utils.transform import DEFAULT_TRANSFORMS, VAL_TRANSFORMS
from utils.loss import compute_loss
from val import evaluate
from utils.utils import get_lr

os.makedirs('./log', exist_ok=True)
writer = SummaryWriter("./log")

def train_one_epoch(model, epoch, train_loader, optimizer, device):
    model.to(device)
    model.train()

    total_loss = 0.0
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(imgs)

        loss = compute_loss(pred, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f"Train epoch[{epoch}]: [{i}/{len(train_loader)}], total loss: {loss.item()} lr:{get_lr(optimizer):.5}")
    torch.save(model.state_dict(), f"./run/centernet_{str(epoch)}.pth")
    writer.add_scalar("train loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("lr", get_lr(optimizer), epoch)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    warmup: 经过 warmup_iters 轮训练之后： learning rate 从 warmup_factor * lr 上升到 lr
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

if __name__ == '__main__':
    model = CenterNet(backbone_weight="resnet50.pth")
    device = torch.device("cuda:0")

    train_data = CenterNetDataset('./my_yolo_dataset', isTrain=True, transform=DEFAULT_TRANSFORMS)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)

    eval_data = CenterNetDataset('./my_yolo_dataset', isTrain=False, transform=VAL_TRANSFORMS)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=eval_data.collate_fn)

    init_lr = 0.02
    warm_up_epochs = 5
    freeze_epoch = 30
    total_epoch = 120

    model.freeze_backbone()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = warmup_lr_scheduler(optimizer, warm_up_epochs, 0.01)

    for epoch in range(freeze_epoch):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        lr_scheduler.step()
        ap = evaluate(model, eval_dataloader, device, plot=False)
        writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)

    model.unfreeze_backbone()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=init_lr * 0.1, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch - freeze_epoch)

    for epoch in range(freeze_epoch, total_epoch, 1):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        lr_scheduler.step()
        ap = evaluate(model, eval_dataloader, device, plot=False)
        writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)