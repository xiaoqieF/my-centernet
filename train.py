import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os
import wandb
import argparse

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
    writer.add_scalar("c_loss", total_loss[0]/ len(train_loader), epoch)
    writer.add_scalar("wh_loss", total_loss[1] / len(train_loader), epoch)
    writer.add_scalar("off_loss", total_loss[2] / len(train_loader), epoch)
    writer.add_scalar("lr", get_lr(optimizer), epoch)
    wandb.log({
        "c_loss": total_loss[0]/ len(train_loader),
        "wh_loss": total_loss[1] / len(train_loader),
        "off_loss": total_loss[2] / len(train_loader),
        "total_loss": sum(total_loss) / len(train_loader),
        "lr": get_lr(optimizer)
    }, commit=False)

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

def main():
    wandb.init(project="my-centernet")

    parser = argparse.ArgumentParser(description='my-centernet')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for trainning, default:64')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='num of epochs to train, default:200')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='num of classes')
    parser.add_argument('--dataset', type=str, default='./DroneVsBirds',
                        help='path of dataset')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='num of workers to load data')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default: 0.01')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for sgd, default: 0.9')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='num epochs for warmup, default: 5')
    parser.add_argument('--model', type=str, default='centernetplus',
                        help='choose model, [centernet] or [centernetplus]')
    parser.add_argument('--backbone', type=str, default='r18',
                        help='backbone of model')
    args = parser.parse_args()
    wandb.config.update(args)

    # model
    if args.model == "centernet":
        model = CenterNet(num_classes=args.num_classes, backbone=args.backbone, pretrained=True)
    elif args.model == "centernetplus":
        model = CenterNetPlus(num_classes=args.num_classes, backbone=args.backbone, pretrained=True)
    else:
        print("bad args for --model")
        exit(-1)
    device = torch.device("cuda:0")

    test_input = torch.randn((1, 3, 512, 512), device=device)
    model.to(device)

    writer.add_graph(model, test_input)
    writer.flush()

    # model params and GFLOPs
    model_info(model, verbose=True)

    train_data = CenterNetDataset(args.dataset, isTrain=True, augment=True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_data.collate_fn)

    eval_data = CenterNetDataset(args.dataset, isTrain=False, augment=False)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=eval_data.collate_fn)

    init_lr = args.lr
    warm_up_epochs = args.warmup_epochs
    total_epoch = args.epochs

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=args.momentum, weight_decay=0.0005)
    lr_scheduler = warmup_lr_scheduler(optimizer, warm_up_epochs, 0.01)

    for epoch in range(warm_up_epochs):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        lr_scheduler.step()

        ap = evaluate(model, eval_dataloader, device, label_path= os.path.join(args.dataset, 'my_data_label.names'), plot=False)
        writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)
        wandb.log({
            'mAP50': ap[:, 0].mean(),
        })
    
    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=args.momentum, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch - warm_up_epochs)

    for epoch in range(warm_up_epochs, total_epoch, 1):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device)
        if epoch > total_epoch - 50:
            torch.save(model.state_dict(), f"./run/{args.dataset.split('/')[-1]}_{args.model}_{args.backbone}_{str(epoch)}.pth")
        lr_scheduler.step()

        ap = evaluate(model, eval_dataloader, device, label_path= os.path.join(args.dataset, 'my_data_label.names'), plot=False)
        writer.add_scalar('mAP50', ap[:, 0].mean(), epoch)
        wandb.log({
            'mAP50': ap[:, 0].mean(),
        })
if __name__ == '__main__':
    main()