import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from util.verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
import cv2
from config import parse_args


import time

# Support: ['get_time', 'l2_norm', 'make_weights_for_balanced_classes', 'get_val_pair', 'get_val_data', 'separate_irse_bn_paras', 'separate_resnet_bn_paras', 'warm_up_lr', 'schedule_lr', 'de_preprocess', 'hflip_batch', 'ccrop_batch', 'gen_plot', 'perform_val', 'buffer_val', 'AverageMeter', 'accuracy']
args = parse_args()


def get_time(): ##
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1): ##
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses): ##
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name): ##
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path): ##
    lfw, lfw_issame = get_val_pair(os.path.join(data_path, 'lfw_align_112'), 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(os.path.join(data_path, 'cfp_align_112'), 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(os.path.join(data_path, 'cfp_align_112'), 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(os.path.join(data_path, 'AgeDB'), 'agedb_30')

    return lfw, cfp_ff, cfp_fp, agedb_30, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame


def separate_irse_bn_paras(modules): ##
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer): ##
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def schedule_lr(optimizer): ##
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor): ##
    return tensor * 0.5 + 0.5


ccrop_LR_sr = transforms.Compose([ ##
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.Resize([int(112/args.LR_scale), int(112/args.LR_scale)], transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
to_normal = transforms.Compose([ ##
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

hflip = transforms.Compose([ ##
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor): ##
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs



ccrop_LR = transforms.Compose([ ##
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),

            # transforms.Resize([int(112/args.LR_scale), int(112/args.LR_scale)], transforms.InterpolationMode.BICUBIC),
            transforms.Resize([int(112), int(112)], transforms.InterpolationMode.BICUBIC),

            transforms.Resize([112,112], transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


ccrop = transforms.Compose([ ##
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def ccrop_batch(imgs_tensor, LR_EVAL): ##
    ccropped_imgs = torch.empty_like(imgs_tensor)
    
    for i, img_ten in enumerate(imgs_tensor):
        
        if LR_EVAL:
            if args.LR_oneside:
                if i%2 == 0:
                    ccropped_imgs[i] = ccrop_LR(img_ten)
                else:
                    ccropped_imgs[i] = ccrop(img_ten)
            else:
                ccropped_imgs[i] = ccrop_LR(img_ten)
        else:
            ccropped_imgs[i] = ccrop(img_ten)
     

    return ccropped_imgs

def gen_plot(fpr, tpr):  ##
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(device, embedding_size, batch_size, backbone, carray, issame, LR_EVAL, nrof_folds = 10): ##

    # nrof_fold : ---


    backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    start = time.time()

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :]) # 64, 3, 112, 112
            # show image
            # batch_ = torch.permute(batch, (0,2,3,1))
            # for i in range(10):
            #     cv2.imwrite('/data2/jaep0805/face.evoLVe.PyTorch/eval_LR{}.png'.format(i),cv2.cvtColor(batch_[i].numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255) 
            ccropped = ccrop_batch(batch, LR_EVAL)
            #ccropped_ = torch.permute(ccropped, (0,2,3,1))
            #for i in range(10):
            #    cv2.imwrite('/data2/jaep0805/face.evoLVe.PyTorch/ccroped_LR{}.png'.format(i),cv2.cvtColor(ccropped_[i].numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255)
            fliped = hflip_batch(ccropped)
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            ccropped = ccrop_batch(batch, LR_EVAL)
            fliped = hflip_batch(ccropped)
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
            embeddings[idx:] = l2_norm(emb_batch)


    mid = time.time()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    end = time.time()
    #print("Time : ", mid - start, "   ", end - mid, "   ", end - start)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

# SR 추가
def perform_val_sr(device, embedding_size, batch_size, backbone, carray, issame, LR_EVAL, sr_module, nrof_folds = 10, tta = True): ##
    backbone = backbone.to(device)
    sr_module = sr_module.to(device)
    backbone.eval() # switch to evaluation mode
    sr_module.eval()

    start = time.time()
    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            # for i in range(batch.shape[0]):
            #     batch_ = torch.permute(batch, (0,2,3,1))
            #     cv2.imwrite('/data2/jaep0805/face.evoLVe.PyTorch/eval_LR{}.png'.format(i),cv2.cvtColor(batch_[i].numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255)
            
            if tta:                
                ccropped = ccrop_batch_sr(batch, LR_EVAL, sr_module, device)
                #ccropped_ = torch.permute(ccropped, (0,2,3,1))
                
                #for i in range(10):
                #    cv2.imwrite('/data2/jaep0805/face.evoLVe.PyTorch/ccroped_LR{}.png'.format(i),cv2.cvtColor(ccropped_[i].numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch_sr(batch, LR_EVAL, sr_module, device)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = ccrop_batch_sr(batch, LR_EVAL, sr_module, device)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch_sr(batch, LR_EVAL, sr_module, device)
                embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()


    mid = time.time()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    end = time.time()
    print("Time : ", mid - start, "   ", end - mid, "   ", end - start)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def ccrop_batch_sr(imgs_tensor, LR_EVAL, sr_module, device): ##
    ccropped_imgs = torch.empty_like(imgs_tensor)
    
    for i, img_ten in enumerate(imgs_tensor):
        
        if LR_EVAL:
            if args.LR_oneside:
                if i%2 == 0:
                    ccropped_imgs[i] = sr_module(ccrop_LR_sr(img_ten).to(device))
                    # ccropped_imgs[i] = sr_module(ccropped_imgs[i])
                    ccropped_imgs[i] = to_normal(ccropped_imgs[i])

                else:
                    ccropped_imgs[i] = ccrop(img_ten)
            else:
                ccropped_imgs[i] = ccrop_LR_sr(img_ten)
        else:
            ccropped_imgs[i] = ccrop(img_ten)
        # if DISTORT_EVAL:
        #     ccropped_imgs[i] = ccrop_DISTORT(img_ten)            

    return ccropped_imgs

def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch): ##
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object): ##
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)): ##
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
