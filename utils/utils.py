import torch

def load_class_names(path):
    class_names = None
    with open(path, "r") as fp:
        class_names = fp.read().splitlines()
    class_names = {k: v for k, v in enumerate(class_names)}
    return class_names

def get_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def box_iou(box1, box2):
    """用于计算混淆矩阵
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    :params box1: (Tensor[N, 4])  [N, x1y1x2y2]
    :params box2: (Tensor[M, 4])  [M, x1y1x2y2]
    :return box1和box2的iou [N, M]
    """
    def box_area(box):
        # 求出box的面积
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)   # box1面积
    area2 = box_area(box2.T)   # box2面积

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # 等价于(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 + 1e-16 - inter)  # iou = inter / (area1 + area2 - inter)