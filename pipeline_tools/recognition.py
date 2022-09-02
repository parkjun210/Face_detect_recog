import numpy as np
import torch
import torchvision.transforms as transforms


def calculate_recog(target_embed, source_embed, thresholds):

    target_embed = np.expand_dims(target_embed, axis=1)
    target_embed = np.repeat(target_embed, source_embed.shape[0], axis=1)

    diff = np.subtract(target_embed, source_embed)
    dist = np.sum(np.square(diff), 2)
    pred_dist = np.min(dist, axis = 1)
    pred_label = np.argmin(dist, axis=1) + 1
    thres_flag = np.less(pred_dist, thresholds)

    pred_label = np.multiply(pred_label, thres_flag)

    # score = (thresholds - pred_dist)/thresholds
    score = 1 - (pred_dist / (2 * thresholds))



    return pred_label, score


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

normalize = transforms.Compose([
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(), 
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs

def perform_recog(device, embedding_size, backbone, target, source, thres = 0.1):

    # target = [N, 112, 112, 3]

    backbone = backbone.to(device)
    backbone.eval()

    target = target / 255
    source = source / 255


    num_people = target.shape[0]
    num_source = source.shape[0]

    target_embed = np.zeros([num_people, embedding_size])
    source_embed = np.zeros([num_source, embedding_size])
    
    target = torch.tensor(target[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2).float()
    source = torch.tensor(source[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2).float()
       
    target = normalize(target)
    ccropped = ccrop_batch(target)
    fliped = hflip_batch(ccropped)
    emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
    target_embed[0:] = l2_norm(emb_batch)

    source = normalize(source)
    source_ccropped = ccrop_batch(source)
    soruce_fliped = hflip_batch(source_ccropped)
    source_emb_batch = backbone(source_ccropped.to(device)).cpu() + backbone(soruce_fliped.to(device)).cpu()
    source_embed[0:] = l2_norm(source_emb_batch)

    pred_label, confidence = calculate_recog(target_embed, source_embed, thresholds = thres)

    return pred_label, confidence

