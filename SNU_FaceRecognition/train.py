import os
from time import time
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import parse_args
from model.backbone.model_irse import IR_SE_50, Backbone
import model.head.metrics as hm
from model.loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
import numpy as np
import random

from tensorboardX import SummaryWriter
from tqdm import tqdm

import re

if __name__ == '__main__':
    args = parse_args()

    #Session parameters
    SEED = args.seed
    GPU_ID = args.gpu_num
    PIN_MEMORY = True
    NUM_WORKERS = 8

    #Directory parameters
    DATA_ROOT = args.data_dir # the parent root where your train/val/test data are stored
    CHECKPOINT_ROOT = args.checkpoint_dir # the root to buffer your checkpoints
    LOG_ROOT = args.log_dir # the root to log your train/val status
    CHECKPOINT_BEST_ROOT = args.best_checkpoint_dir
    RESUME_ROOT = args.resume_dir

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 60)

    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results
    if LR_TRAIN:
        train_transform = transforms.Compose([
            transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), 
            transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
            transforms.RandomApply([transforms.Resize([int(112/LR_SCALE), int(112/LR_SCALE)], interpolation=transforms.InterpolationMode.BICUBIC)], p=0.5),
            transforms.Resize([112,112], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = RGB_MEAN,
                                std = RGB_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), 
            transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = RGB_MEAN,
                                std = RGB_STD),
        ])
    
    # train dataset root setting
    dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'imgs'), train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    # load test_dataset
    lfw, cfp_ff, cfp_fp, agedb, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame = get_val_data(DATA_ROOT)


    #======= model & loss & optimizer =======#
    BACKBONE = IR_SE_50(INPUT_SIZE)
    print("=" * 60)
    print("IR_SE_50 Backbone Generated")

    HEAD = hm.ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)
    print("ArcFace Head Generated")

    LOSS = FocalLoss()
    print("Focal Loss Generated")

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
 
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    print("=" * 60)
    subdirlist = []
    if RESUME_ROOT != "":
        for filename in os.listdir(RESUME_ROOT):
            if filename.startswith("Backbone"):
                BACKBONE_RESUME_ROOT = os.path.join(RESUME_ROOT, filename)
                print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            if filename.startswith("Head"):
                HEAD_RESUME_ROOT = os.path.join(RESUME_ROOT, filename)
                print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
                HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))

    BACKBONE = BACKBONE.to(DEVICE)


    #======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 10 # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = 5  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    
    batch = 0
    best_acc = 0
    best_epoch = 0


    if LR_TRAIN:
        checkpoint_name = "LRTRAIN_LRx{}".format(LR_SCALE)
    else:
        checkpoint_name = "HRTRAIN"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, checkpoint_name)
    best_checkpoint_dir = os.path.join(CHECKPOINT_BEST_ROOT, checkpoint_name)
    checkpoint_dir = checkpoint_dir + "_" + get_time()
    best_checkpoint_dir = best_checkpoint_dir + "_" + get_time()
    if not os.path.exists(checkpoint_dir):        
        # Create a new directory because it does not exist 
        os.makedirs(checkpoint_dir)
        os.makedirs(best_checkpoint_dir)
        print("New checkpoint directory is created!")

    print("=" * 60)

    for epoch in range(NUM_EPOCH): # start training process
        
        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels in tqdm(iter(train_loader)):

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        print("Perform Evaluation on LFW, CFP_FF, CFP_FP and AgeDB")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame, LR_EVAL)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame, LR_EVAL)
        buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame, LR_EVAL)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame, LR_EVAL)
        buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
        
        avg = (accuracy_lfw + accuracy_cfp_ff + accuracy_cfp_fp + accuracy_agedb) / 4

        # save the best model
        if avg > best_acc:
            best_acc = avg
            best_epoch = epoch + 1

            torch.save(BACKBONE.state_dict(), os.path.join(best_checkpoint_dir, "Backbone_Best.pth"))
            torch.save(OPTIMIZER.state_dict(), os.path.join(best_checkpoint_dir, "Optimizer_Best.pth"))
            torch.save(HEAD.state_dict(), os.path.join(best_checkpoint_dir, "Head_Best.pth"))

        
        print("Epoch {}/{}, Evaluation: LFW Acc: {:.4f}, CFP_FF Acc: {:.4f}, CFP_FP Acc: {:.4f}, AgeDB Acc: {:.4f}, Avg Acc: {:.4f}\n".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, avg ))
        print("Best Epoch : {}, Avg : {:.4f}".format(best_epoch, best_acc))
        print("=" * 60)
        
        # save evaluation results by text file
        fname = os.path.join(checkpoint_dir, "Evaluation.txt")
        try:
            fobj = open(fname, 'a')
        except IOError:
            print('open error')
        else:
            fobj.write("Epoch {}/{}, Evaluation: LFW : {:.4f}, CFP_FF : {:.4f}, CFP_FP : {:.4f}, AgeDB : {:.4f}, Avg : {:.4f}\n".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, avg ))
            fobj.close()
        

        # save checkpoints per epoch
        torch.save(BACKBONE.state_dict(), os.path.join(checkpoint_dir, "Backbone_Epoch_{}_Batch_{}.pth".format(epoch + 1, batch, LR_TRAIN, LR_SCALE)))
        torch.save(OPTIMIZER.state_dict(), os.path.join(checkpoint_dir, "Optimizer_Epoch{}_Batch{}.pth".format(epoch + 1, batch, LR_TRAIN, LR_SCALE)))
        torch.save(HEAD.state_dict(), os.path.join(checkpoint_dir, "Head_Epoch_{}_Batch_{}.pth".format(epoch + 1, batch, LR_TRAIN, LR_SCALE)))
        
