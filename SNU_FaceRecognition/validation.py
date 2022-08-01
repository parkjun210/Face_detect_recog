from multiprocessing.pool import TERMINATE
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from model.backbone.model_irse import IR_SE_50
from model.head import metrics as hm
from model.loss.focal import FocalLoss
from util.utils import get_val_data, perform_val, perform_val_sr
from SR.model_RCAN import RCAN
from config import parse_args

import argparse
import numpy as np
import random

from tqdm import tqdm

if __name__ == '__main__':
    args = parse_args()

    #Session parameters
    SEED = args.seed
    GPU_ID = args.gpu_num
    MULTI_GPU = len(GPU_ID) != 1
    PIN_MEMORY = True
    NUM_WORKERS = 8

    #Directory parameters
    DATA_ROOT = args.data_dir # the parent root where your train/val/test data are stored
    CHECKPOINT_ROOT = args.checkpoint_dir # the root to buffer your checkpoints
    LOG_ROOT = args.log_dir # the root to log your train/val status
    CHECKPOINT_BEST_ROOT = args.best_checkpoint_dir
    RESUME_ROOT = args.resume_dir
    BACKBONE_RESUME_ROOT = args.backbone_dir # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = args.head_dir  # the root to resume training from a saved checkpoint

    #Train parameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCH = args.epochs
    #further configurable
    INPUT_SIZE = [112, 112] # support: [112, 112] and [224, 224]
    RGB_MEAN = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    RGB_STD = [0.5, 0.5, 0.5]
    EMBEDDING_SIZE = 512 # feature dimension
    DROP_LAST = True # whether drop the last batch to ensure consistent batch_norm statistics
    LR = 0.025 # initial LR
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    STAGES = [15, 23, 30] # epoch stages to decay learning rate

    LR_TRAIN = args.LR_train
    LR_SCALE = args.LR_scale
    LR_EVAL = args.LR_eval
    SR_EVAL = args.SR_eval
    #LR_ONE_SIDE = args.LR_oneside
    

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    lfw, cfp_ff, cfp_fp, agedb, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame = get_val_data(DATA_ROOT)

    #======= model & loss & optimizer =======#
    BACKBONE = IR_SE_50(INPUT_SIZE)
    print("=" * 60)

    #Load checkpoint
    print("=" * 60)
    subdirlist = []
    if RESUME_ROOT == "/data/parkjun210/Jae_Detect_Recog/Code_face_recog/checkpoints_best/":
        for subdir, dirs, files in os.walk(RESUME_ROOT):
            subdirlist.append(subdir)
        subdirlist = subdirlist[1:]
        subdirlist.sort()
        RESUME_ROOT = subdirlist[-1]
    for subdir, dirs, files in os.walk(RESUME_ROOT):
        for file in files:
            if file.startswith("Backbone"):
                BACKBONE_RESUME_ROOT = os.path.join(RESUME_ROOT, file)
                print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    

    #======= train & validation & save checkpoint =======#

    # perform validation & save checkpoints per epoch
    # validation statistics per epoch (buffer for visualization)
    print("=" * 60)

    if SR_EVAL:
        parser = argparse.ArgumentParser()
        parser.add_argument('--arch', type=str, default='RCAN')
        parser.add_argument('--weights_path', type=str, default='/data/parkjun210/ArcFace/RCAN-pytorch/output/RCAN_epoch_19.pth')
        parser.add_argument('--image_path', type=str, default='/data/parkjun210/ArcFace/FSRCNN-PyTorch/eval_data')
        parser.add_argument('--outputs_dir', type=str, default='/data/parkjun210/ArcFace/RCAN-pytorch/eval_output')
        parser.add_argument('--scale', type=int, default=8)
        parser.add_argument('--num_features', type=int, default=64)
        parser.add_argument('--num_rg', type=int, default=10)
        parser.add_argument('--num_rcab', type=int, default=20)
        parser.add_argument('--reduction', type=int, default=16)
        opt = parser.parse_args()

        sr_model = RCAN(opt).to(DEVICE)
        checkpoint = torch.load('/data/parkjun210/ArcFace/RCAN-pytorch/output/scale_x8_pretrain_x2/RCAN_epoch_20_best.pth')
        sr_model.load_state_dict(checkpoint)

        print("Perform SR Evaluation on LFW, CFP_FF, CFP_FP and AgeDB")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val_sr(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, LR_EVAL, sr_model)
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val_sr(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame, LR_EVAL, sr_model)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val_sr(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame, LR_EVAL, sr_model)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val_sr(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame, LR_EVAL, sr_model)
    else:
        print("Perform Evaluation on LFW, CFP_FF, CFP_FP and AgeDB")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, LR_EVAL)
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame, LR_EVAL)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame, LR_EVAL)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame, LR_EVAL)


    avg = (accuracy_lfw + accuracy_cfp_ff + accuracy_cfp_fp + accuracy_agedb) / 4

    if SR_EVAL : 
        print("SR Evaluation: LFW Acc: {:.4f}, CFP_FF Acc: {:.4f}, CFP_FP Acc: {:.4f}, AgeDB Acc: {:.4f}, Avg Acc: {:.4f}\n".format(accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, avg ))
        print("=" * 60)
    else:
        print("Evaluation: LFW Acc: {:.4f}, CFP_FF Acc: {:.4f}, CFP_FP Acc: {:.4f}, AgeDB Acc: {:.4f}, Avg Acc: {:.4f}\n".format(accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, avg ))
        print("=" * 60)
