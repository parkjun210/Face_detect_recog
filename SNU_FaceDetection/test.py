from utils.metric import get_metric
from utils.helpers import decode_output
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import os
import numpy as np
from time import time

from config import parse_args
from models.retinaface import RetinaFace
from utils.average_meter import AverageMeter
from utils.data_loaders import Dataset, detection_collate
from utils.data_transformer import preproc
from utils.helpers import *

torch.set_grad_enabled(False)

args = parse_args()

# Design Parameters
NETWORK = 'resnet50'

# Session Parameters
GPU_NUM = args.gpu_num

# Directory Parameters
DATASET = args.dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, "ckpt/")
LOG_DIR = os.path.join(EXP_DIR, "log/")
IMG_DIR = os.path.join(EXP_DIR, 'img/')
WEIGHTS = "ckpt-best.pth"

# Inference Parameters
CONFIDENCE_THRESHOLD = 0.02
NMS_THRESHOLD = 0.3
EXP_NAME = args.experiment_name
RES_DIR = 'experiments/' + EXP_NAME + '/results/'

NUM_ANCHOR = args.num_anchor
if 'tina' in EXP_NAME:
    NUM_ANCHOR = 3
else:
    NUM_ANCHOR = 2

# Set up dataset
test_dataset = Dataset(DATASET, 'val', preproc(img_dim=None, rgb_means=(104, 117, 123)))
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             num_workers=4,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=detection_collate)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate results.txt
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

f = open(RES_DIR + 'results.txt', 'w')
f.write("[Test for experiments '" + EXP_NAME + "' has started]\n\n")
num_images = len(test_dataset)
f.write("Number of test images: " + str(num_images) + "\n\n")

# Set up Network
net = RetinaFace(phase='test', version=args.model_version, anchor_num=NUM_ANCHOR)
output_path = CKPT_DIR + NETWORK + '_' + WEIGHTS
checkpoint = torch.load(output_path)
f.write("Start loading weights...\n")
net.load_state_dict(checkpoint['network'])
net.eval()
net = net.to(device)
cudnn.benchmark = True

# Evaluation Start
easy_metric = AverageMeter()
medium_metric = AverageMeter()
hard_metric = AverageMeter()
normal_metric = AverageMeter()
FP_metric = AverageMeter()

Start = time()
for i, data in enumerate(test_dataloader):

    # Obtain data / to GPU
    img, targets, img_name = data
    img = img.to(device)
    targets = [anno.cuda() for anno in targets]

    # Forward
    _, out = net(img)

    loc, conf, landms, ious = out

    # Decode
    scores, boxes, landms = decode_output(img, loc, conf, landms, device, NUM_ANCHOR)

    # NMS
    dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Calculate AP
    pred_bboxes = torch.from_numpy(dets[:, :4]).to(device)
    pred_scores = torch.from_numpy(dets[:, 4]).to(device)

    ap, diff = get_metric(pred_bboxes, pred_scores, targets, img.shape, official=False)

    # Update Holders
    if diff == 0:
        easy_metric.update(ap)
    elif diff == 1:
        medium_metric.update(ap)
    elif diff == 2:
        hard_metric.update(ap)
    else:
        print("Invalid difficulty")

    if i % 10 == 0:
        print('im_detect: {:d}/{:d}'.format(i + 1, num_images))
End = time()
f.write("Calculating process has done !\n")
avg_metric = (easy_metric.sum() + medium_metric.sum() + hard_metric.sum()) / len(test_dataset)
print('Avg AP : %.4f   Easy AP : %.4f   Medium AP : %.4f   Hard AP : %.4f' % (avg_metric, easy_metric.avg(), medium_metric.avg(), hard_metric.avg()))

# Print Test Measures
Test_time = End - Start
Average_time = round(Test_time / float(num_images), 2)

f.write("Total test time is " + str(round(Test_time, 2)) + " secs\n")
f.write("Average test time per image is " + str(Average_time) + " secs, which is" + str(round(1 / Average_time, 2)) + "fps.\n")

f.write("The Evaluation results are as below.\n")
f.write('Avg AP : %.4f   Easy AP : %.4f   Medium AP : %.4f   Hard AP : %.4f \n' % (avg_metric, easy_metric.avg(), medium_metric.avg(), hard_metric.avg()))
f.write("[Test for experiments '" + EXP_NAME + "' has succesfully Ended]\n")
f.close()

