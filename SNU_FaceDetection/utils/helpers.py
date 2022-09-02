import torch
import os
import numpy as np

from .box_utils import decode, decode_landm
from .nms.py_cpu_nms import py_cpu_nms
from ..layers.prior_box import PriorBox


def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device('cpu'):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)

    return x

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(network):
    return sum(p.numel() for p in network.parameters())

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def decode_output(img, loc, conf, landms, device):

    # Get scale
    _, _, H, W = img.shape
    scale_box = torch.Tensor([W, H, W, H])
    scale_box = scale_box.to(device)
    scale_landm = torch.Tensor([W, H, W, H, 
                                W, H, W, H,
                                W, H])
    scale_landm = scale_landm.to(device)                            

    # Priorbox
    priorbox = PriorBox(image_size=(H, W))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    # Decode
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    boxes = decode(loc.data.squeeze(0), prior_data, variances=[0.1,0.2])
    boxes = boxes * scale_box
    boxes = boxes.cpu().numpy()
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, variances=[0.1,0.2])
    landms = landms * scale_landm
    landms = landms.cpu().numpy()

    return scores, boxes, landms

def decode_output_inf(loc, conf, landms, prior_data, scale_box, scale_landm):

    # Decode
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    boxes = decode(loc.data.squeeze(0), prior_data, variances=[0.1,0.2])
    boxes = boxes * scale_box
    boxes = boxes.cpu().numpy()
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, variances=[0.1,0.2])
    landms = landms * scale_landm
    landms = landms.cpu().numpy()

    return scores, boxes, landms

def do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD):

    inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)

    return dets

def pred2txt(img_name, dets, SAVE_FOLDER):

    img_name = img_name[0]
    save_name = SAVE_FOLDER + img_name[:-4] + ".txt"
    dirname = os.path.dirname(save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        bboxs = dets
        file_name = os.path.basename(save_name)[:-4] + "\n"
        bboxs_num = str(len(bboxs)) + "\n"
        fd.write(file_name)
        fd.write(bboxs_num)
        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
            fd.write(line)