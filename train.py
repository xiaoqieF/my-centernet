import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os

from networks.centernetplus import CenterNetPlus
from networks.centernet import CenterNet
from utils.dataset import CenterNetDataset
from utils.loss import compute_loss
from val import evaluate
from utils.utils import get_lr, model_info

os.makedirs('./log', exist_ok=True)
writer = SummaryWriter("./log")

def train_one_epoch(model, epoch, train_loader, optimizer, device):
    model.to(device)
    model.train()

    total_loss = [0.0, 0.0, 0.0]
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device).float() / 255
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(imgs)

        c_loss, wh_loss, off_loss = compute_loss(pred, targets)
        total_loss[0] += c_loss.item()
        total_loss[1] += wh_loss.item()
        total_loss[2] += off_loss.item()

        loss = c_loss + wh_loss + off_loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f"Train epoch[{epoch}]: [{i}/{len(train_loader)}], total loss: {loss.item()} lr:{get_lr(optimizer):.5}")
    torch.save(model.state_dict(), f"./run/20centernet_r50_{str(epoch)}.pth")
    writer.add_scalar("c_loss", total_loss[0]/ len(train_loader), epoch)
    writer.add_scalar("wh_loss", total_loss[1] / len(train_loader), epoch)
    writer.add_scalar("off_loss", total_loss[2] / len(train_loader), epoch)
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
    model = CenterNet(num_classes=20, backbone="r50", pretrained=True)
    device = torch.device("cuda:0")

    test_input = torch.randn((1, 3, 512, 512), device=device)
    model.to(device)

    writer.add_graph(model, test_input)
    writer.flush()

    model_info(model, verbose=True)

    train_data = CenterNetDataset('./my_yolo_dataset', isTrain=True, augment=True)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, collate_fn=train_data.collate_fn)

    eval_data = CenterNetDataset('./my_yolo_dataset', isTrain=False, augment=False)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=8, collate_fn=eval_data.collate_fn)

    init_lr = 0.01
    warm_up_epochs = 5
    total_epoch = 200

    # model.freeze_backbone()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = warmup_lr_scheduler(optimizer, warm_up_epochs, 0.01)

    for epoch in range(warm_up_epochs):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        lr_scheduler.step()

        if epoch % 3 == 0:
            ap = evaluate(model, eval_dataloader, device, plot=False)
            writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)

    model.unfreeze_backbone()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch - warm_up_epochs)

    for epoch in range(warm_up_epochs, total_epoch, 1):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        lr_scheduler.step()

        if epoch % 3 == 0:
            ap = evaluate(model, eval_dataloader, device, plot=False)
            writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)