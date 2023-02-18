import math
import torch
import torch.nn.functional as F
import numpy as np

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        torch.maximum(masked_heatmap, torch.from_numpy(masked_gaussian * k).to(masked_heatmap.device), out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def compute_loss(pred, targets):
    hm, wh, offset = pred
    bs, class_num, feat_h, feat_w = hm.shape
    device = targets.device

    targets_hm = torch.zeros((bs, feat_h, feat_w, class_num), device=device)
    targets_wh = torch.zeros((bs, feat_h, feat_w, 2), device=device)
    targets_reg = torch.zeros((bs, feat_h, feat_w, 2), device=device)
    targets_reg_mask = torch.zeros((bs, feat_h, feat_w), device=device)

    for i in range(bs):
        target = targets[targets[:, 0] == i]
        boxes = torch.zeros((target.shape[0], 4), device=device)
        boxes[:, [0, 2]] = target[:, [2, 4]] * feat_h
        boxes[:, [1, 3]] = target[:, [3, 5]] * feat_w

        for j in range(len(target)):
            cls_id = int(target[j, 1])
            h, w = boxes[j][3] - boxes[j][1], boxes[j][2] - boxes[j][0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = torch.tensor([(boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2], dtype=torch.float32, device=device)
                ct_int = ct.long()

                targets_hm[i, :, :, cls_id] = draw_gaussian(targets_hm[i, :, :, cls_id], ct_int, radius)
                targets_wh[i, ct_int[1], ct_int[0]] = torch.tensor([1. * w, 1. * h], dtype=torch.float32, device=device)
                targets_reg[i, ct_int[1], ct_int[0]] = ct - ct_int
                targets_reg_mask[i, ct_int[1], ct_int[0]] = 1
    
    c_loss = focal_loss(hm, targets_hm)  # 将 pred_hm 映射至 0 - 1, 对输出解码时也需先这样操作
    wh_loss = 0.1 * reg_l1_loss(wh, targets_wh, targets_reg_mask)
    off_loss = reg_l1_loss(offset, targets_reg, targets_reg_mask)

    loss = c_loss + wh_loss + off_loss

    return loss


def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
