import torch
import torch.nn as nn
import torchvision
import numpy as np

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

class BBoxDecoder:
    hm_height, hm_width = 128, 128
    yv, xv = torch.meshgrid(torch.arange(0, hm_height), torch.arange(0, hm_width), indexing='ij')
    xv, yv = xv.flatten().float().cuda(), yv.flatten().float().cuda()
    
    def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence):
        pred_hms = pool_nms(pred_hms)
        b, c, output_h, output_w = pred_hms.shape
        detects = []
        #-------------------------------------------------------------------------#
        #   只传入一张图片，循环只进行一次
        #-------------------------------------------------------------------------#
        for batch in range(b):
            #-------------------------------------------------------------------------#
            #   heat_map        128*128, num_classes    热力图
            #   pred_wh         128*128, 2              特征点的预测宽高
            #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
            #   pred_offset     128*128, 2              特征点的xy轴偏移情况
            #-------------------------------------------------------------------------#
            heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
            pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
            pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

            #-------------------------------------------------------------------------#
            #   class_conf      128*128,    特征点的种类置信度
            #   class_pred      128*128,    特征点的种类
            #-------------------------------------------------------------------------#
            class_conf, class_pred  = torch.max(heat_map, dim = -1)
            mask                    = class_conf > confidence

            #-----------------------------------------#
            #   取出得分筛选后对应的结果
            #-----------------------------------------#
            pred_wh_mask        = pred_wh[mask]
            pred_offset_mask    = pred_offset[mask]
            if len(pred_wh_mask) == 0:
                detects.append(torch.zeros((0, 6), device=pred_hms.device))
                continue     

            #----------------------------------------#
            #   计算调整后预测框的中心
            #----------------------------------------#
            xv_mask = torch.unsqueeze(BBoxDecoder.xv[mask] + pred_offset_mask[..., 0], -1)
            yv_mask = torch.unsqueeze(BBoxDecoder.yv[mask] + pred_offset_mask[..., 1], -1)
            #----------------------------------------#
            #   计算预测框的宽高
            #----------------------------------------#
            half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
            #----------------------------------------#
            #   获得预测框的左上角和右下角
            #----------------------------------------#
            bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
            bboxes[:, [0, 2]] *= 4
            bboxes[:, [1, 3]] *= 4
            detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
            detects.append(detect)

        return detects

def postprocess(predictions, nms_thres=0.4, classes=None):
    output = [torch.zeros((0, 6))] * len(predictions)

    for i, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue

        unique_labels = prediction[:, -1].unique()

        # 过滤某些类
        if classes != None:
            unique_labels = [k for k in unique_labels if k in classes]

        for c in unique_labels:
            prediction_per_class = prediction[prediction[:, -1] == c]
            keep = torchvision.ops.nms(prediction_per_class[:, :4], prediction_per_class[:, 4], nms_thres)

            detection = prediction_per_class[keep]

            output[i] = torch.cat((output[i], detection.cpu()))
        
    return output

def correct_boxes(boxes, input_shape, image_shape):
    # relative box [x1, y1, x2, y2]
    boxes[:, [0, 2]] /= input_shape[0]    # / w
    boxes[:, [1, 3]] /= input_shape[1]    # / h

    boxes_xy, boxes_wh = (boxes[:, 0:2] + boxes[:, 2:4]) / 2, (boxes[:, 2:4] - boxes[:, 0:2])
    boxes_yx, boxes_hw = boxes_xy[:, ::-1], boxes_wh[:, ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape[::-1])  # change to [h, w]

    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    boxes_yx = (boxes_yx - offset) * scale
    boxes_hw *= scale

    boxes_min = boxes_yx - (boxes_hw / 2.)
    boxes_max = boxes_yx + (boxes_hw / 2.)
    boxes = np.concatenate([boxes_min[..., 1:2], boxes_min[..., 0:1], boxes_max[..., 1:2], boxes_max[..., 0:1]], axis=-1)
    boxes[:, [0, 2]] *= image_shape[1]
    boxes[:, [1, 3]] *= image_shape[0]
    return boxes