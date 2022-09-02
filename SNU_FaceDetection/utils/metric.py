import numpy as np
import json
import torch
from torchvision.ops import box_iou

def get_TP_FP(pred_bbox, pred_score, gt_bbox, threshold):
    TP_list = torch.zeros(pred_bbox.shape[0], threshold.shape[0]).cuda()

    ious = box_iou(pred_bbox, gt_bbox)  # (pred, gt)
    best_iou_of_gt, best_gt_idx = ious.max(1) # (pred)
    _ , best_pred_idx = ious.max(0)  # (gt)
    over_threshold_idx = (best_iou_of_gt.cuda() >= 0.4)   #(thres_num, pred)
    valid_num = torch.sum(over_threshold_idx)
    over_threshold_idx = over_threshold_idx.unsqueeze(1).expand(pred_bbox.shape[0],  threshold.shape[0])

    # best_pred_idx = best matching_pred idx
    # 실제 gt기준 best iou를 갖는 pred만 TP -> 1
    for i in range(best_pred_idx.shape[0]):
        TP_list[best_pred_idx[i]] = 1
    is_valid = pred_score.cuda() >= threshold
    is_valid = torch.transpose(is_valid, 1, 0)
    TP_list = torch.mul(TP_list, is_valid)
    TP_list = torch.mul(TP_list, over_threshold_idx)

    return TP_list, valid_num


def draw_pr_curve(thresh_num, pred_bboxes, pred_scores, gt_bboxes):
    pr_info = torch.zeros((thresh_num, 2))
    threshold = torch.zeros((thresh_num, 1)).cuda()
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        threshold[t] = thresh
    TP_list, valid_pred_num = get_TP_FP(pred_bboxes, pred_scores, gt_bboxes, threshold)
    TP = torch.sum(TP_list, dim =0)

    pr_info[:, 0] = TP / valid_pred_num
    pr_info[:, 1] = TP / gt_bboxes.shape[0]

    return pr_info


def ElevenPointInterpolatedAP(prec, rec, thresh_num):
    rec_ = np.asarray(rec)
    prec_ = np.asarray(prec)

    mrec = [e for e in rec_]
    mpre = [e for e in prec_]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, thresh_num + 1)
    recallValues = list(recallValues[::-1])

    rhoInterp, recallValid = [], []

    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / (thresh_num+1)

    return ap


def get_metric(pred_bboxes, pred_scores, targets, images_shape, official=False):
    """
    :param pred_bboxes: bbox coordinates [x1, y1, x2, y2]  of size [nms, 4]  *(not normalized)
    :param pred_scores: confidence score of bbox of size [nms]
    :param targets: contains gt boxes [x1, y1, x2, y2]  of size [gt, 4]      *(normalized)
    :param image_shape: image shape of original image
    :return:
    """
    ###################
    # if temp = True for test official code,
    # pred_bboxes and gt_boxes are not normalized
    ###################

    # 0. prepare gts, preds, confidences
    thresh_num = 100
    if not official:
        _, _, H, W = images_shape

        # normalize bbox_pred coordinates   # check pred_norm
        pred_bboxes[:, 0] = pred_bboxes[:, 0] / W
        pred_bboxes[:, 1] = pred_bboxes[:, 1] / H
        pred_bboxes[:, 2] = pred_bboxes[:, 2] / W
        pred_bboxes[:, 3] = pred_bboxes[:, 3] / H

        # clamp bbox_pred coordinates for using box_iou
        pred_bboxes = torch.clamp(pred_bboxes, min=0, max =1)


    difficulty = targets[0][:, -1].data  # 0,1,2 for easy, medium, hard
    difficulty = difficulty[0]
    gt_bboxes = targets[0][:, :4].data

    if pred_bboxes.nelement()==0:
        ap = 0
    else:
        pr_curve = draw_pr_curve(thresh_num, pred_bboxes, pred_scores, gt_bboxes)

        precision = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = ElevenPointInterpolatedAP(recall, precision, thresh_num)

    return ap, difficulty


if __name__ == '__main__':
    import os
    import glob
    import json
    import numpy as np

    ## prepare gts
    gt_file = open('data/widerface/val/label.txt','r')
    gt_lines = gt_file.readlines()
    image_name = None
    gts_dict = dict()

    for gtline in gt_lines:
        if '.jpg' in gtline:
            image_name = (gtline.split('/')[-1]).split('.')[0]
            gts_dict[image_name] = []
        else:
            gtline = gtline.split(' ')
            box = gtline[:4]
            difficulty = gtline[-2]

            box[0] = int(box[0])
            box[1] = int(box[1])
            box[2] = int(box[2]) + int(box[0])
            box[3] = int(box[3]) + int(box[1])
            item = np.array([box[0], box[1], box[2], box[3], int(difficulty)])
            gts_dict[image_name].append(item)

    folders = glob.glob("./widerface_txt/*")
    easy_count = 0
    medium_count = 0
    hard_count = 0
    image_num_count = 0

    easy_average = 0
    medium_average = 0
    hard_average = 0

    for folder in folders:
        txts = glob.glob(folder + '/*')
        for txt in txts:
            image_key = (txt.split('/')[-1]).split('.')[0]
            f = open(txt, 'r')
            lines = f.readlines()
            pred_bboxes = []
            pred_scores = []
            for idx, line in enumerate(lines):
                if idx > 1:
                    bbox = line.split(' ')[:4]
                    bbox[0] = int(bbox[0])
                    bbox[1] = int(bbox[1])
                    bbox[2] = int(bbox[2]) + int(bbox[0])
                    bbox[3] = int(bbox[3]) + int(bbox[1])
                    pred_bboxes.append(bbox)
                    score = float(line.split(' ')[-2])
                    pred_scores.append(score)
            gt_target = torch.tensor(gts_dict[image_key])

            pred_scores = torch.tensor(pred_scores)
            pred_bboxes = torch.tensor(pred_bboxes)

            ap, diff = get_metric(pred_bboxes, pred_scores, [gt_target], images_shape = None, official=True)
            if diff == 0:
                easy_average += ap
                easy_count += 1
            elif diff == 1:
                medium_average += ap
                medium_count += 1
            elif diff == 2:
                hard_average += ap
                hard_count += 1
            else:
                if (easy_ap + medium_ap + hard_ap) != 0 : print("Error 2")

            image_num_count += 1

    print("Total number of test image : ", image_num_count)
    print("Average Easy AP : ", easy_average/easy_count,
          "Average Medium AP : ", medium_average/medium_count,
          "Average Hard AP : ", hard_average/ hard_count)