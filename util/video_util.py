import os
import torch
import torchvision.transforms as transforms

import numpy as np
import cv2
from sklearn.preprocessing import normalize as sklearn_normalize
from SNU_FaceDetection.utils.helpers import *
from SNU_FaceRecognition.applications.align.align_trans import warp_and_crop_face


from SNU_FaceDetection.layers.prior_box import PriorBox


from torch.utils.data import Dataset

class Video_Dataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.ToTensor()
        self.rgb_mean = (104, 117, 123)

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """Return a single frame from the video."""
        frame = self.frames[idx]

        frame_tensor = frame.astype(np.float32)
        frame_tensor -= self.rgb_mean
        frame_tensor = self.transform(frame_tensor)

        return (frame_tensor, frame)

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

def do_recog(target, source, thres = 0.05):

    pred_label, confidence = calculate_recog(target, source, thresholds = thres)

    return pred_label, confidence

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
            # transforms.Resize([128, 128]),  # smaller side resized
            # transforms.CenterCrop([112, 112]),
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


def img2feature(device, net_recog, imgs, embedding_size = 512):

    feature_size = embedding_size

    imgs = imgs / 255
    num_imgs = imgs.shape[0]
    imgs_embed = np.zeros([num_imgs, feature_size])
    imgs = torch.tensor(imgs[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2).float()

    imgs = normalize(imgs)
    imgs_ccropped = ccrop_batch(imgs)
    imgs_fliped = hflip_batch(imgs_ccropped)
    imgs_emb_batch = net_recog(imgs_ccropped.to(device))[0].cpu() + net_recog(imgs_fliped.to(device))[0].cpu()
    imgs_embed[0:] = l2_norm(imgs_emb_batch)
    feature = imgs_embed

    return feature


def set_library(device, net_recog, lib_dir):

    # color library
    colors = {
        0 : (0, 0, 255),
        1 : (147, 20, 255),
        2 : (0, 205, 0),
        3 : (155, 215, 77),
        4 : (0, 205, 205),
        5 : (205, 0, 205),
        6 : (255, 65, 29),
        7 : (0, 102, 252),
        8 : (39, 157, 229),
        9 : (212, 100, 249),
        10 : (223, 205, 248),
    }
    # colors = {
    #     0: (246, 251, 163),
    #     1: (130, 2, 67),
    #     2: (180, 99, 63),
    #     3: (24, 127, 208),
    #     4: (75, 81, 20),
    #     5: (161, 109, 23),
    #     6: (255, 75, 46),
    #     7: (74, 153, 167),
    #     8: (115, 158, 147),
    #     9: (19, 86, 122),
    #     10: (181, 29, 247),
    #     11: (88, 164, 221),
    #     12: (79, 66, 31),
    #     13: (169, 111, 160),
    #     14: (112, 252, 114),
    #     15: (23, 179, 19),
    #     16: (122, 255, 239),
    #     17: (143, 55, 114),
    #     18: (150, 155, 238),
    #     19: (44, 179, 15),
    #     20: (111, 135, 65),
    #     21: (154, 241, 183),
    #     22: (220, 69, 33),
    #     23: (63, 214, 207),
    #     24: (185, 85, 182),
    #     25: (15, 49, 75),
    #     26: (204, 135, 221),
    #     27: (46, 131, 42),
    #     28: (196, 117, 165),
    #     29: (180, 229, 244),
    #     30: (136, 62, 110),
    #     31: (33, 90, 248),
    #     32: (88, 171, 129),
    #     33: (14, 253, 54),
    #     34: (165, 179, 94),
    #     35: (216, 119, 169),
    #     36: (99, 20, 79),
    #     37: (81, 107, 52),
    #     38: (176, 100, 186),
    #     39: (190, 101, 198),
    #     40: (89, 221, 73),
    #     41: (80, 208, 181),
    #     42: (129, 157, 154),
    #     43: (31, 64, 224),
    #     44: (117, 177, 126),
    #     45: (175, 124, 55),
    #     46: (46, 121, 19),
    #     47: (111, 214, 49),
    #     48: (70, 26, 211),
    #     49: (124, 75, 180)
    # }


    dir_faces = []

    # Set library
    library_face = []
    library_dict = {}
    color_dict = {}
    library_dict[0] = 'Unknown'
    color_dict[0] = colors[0]
    k = 0

    library_folder = lib_dir
    library_count = 0

    library_feature = []

    for file in os.listdir(library_folder):
        path = os.path.join(library_folder, file)
        if os.path.isdir(path):
            for img_name in os.listdir(path):
                libimg = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR)
                library_face.append(libimg)
                library_dict[k + 1] = os.path.basename(path)
                color_dict[k + 1] = colors[library_count + 1]
                k += 1
            library_count += 1
        if file.endswith('.jpg'):
            libimg = cv2.imread(path, cv2.IMREAD_COLOR)
            library_face.append(libimg)
            library_dict[k + 1] = os.path.splitext(file)[0]
            color_dict[k + 1] = colors[library_count + 1]
            library_count += 1
            k += 1


    library_face = np.array(library_face)

    library_feature = img2feature(device, net_recog, library_face)

    return (color_dict, library_dict, library_feature)


def detect(device, net_detect, img, same_flag, **kwargs):

    # Inference Parameters
    CONFIDENCE_THRESHOLD = 0.99
    NMS_THRESHOLD = 0.3

    # Forward
    _, out = net_detect(img)

    loc, conf, landms, ious = out

    # Decode
    # if same_flag:
    scores, boxes, landms = decode_output_inf(loc, conf, landms, prior_data = kwargs['p_data'], scale_box = kwargs['s_box'], scale_landm = kwargs['s_landm'])
    # else:
    # scores, boxes, landms = decode_output(img, loc, conf, landms, device, 3)

    # NMS
    # dets : (box, 4(loc) + 1(conf) + 10(landm))
    dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # 검출된 face가 없을 경우 return None
    if len(dets) == 0:
        return [], []

    result_bboxes = dets[:, :4].astype(int)

    return result_bboxes, dets

def recognize(device, net_recog, img_raw, index, bbox, dets, library, output_dir, save_flag = False, show_bbox = True, show_identity = True):

    # For save
    SAVE_FOLDER = output_dir
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    recognition_save_img = save_flag

    prefix = f'{index:04}'
    suffix = 'jpg'
    

    if len(bbox) == 0:
        # if recognition_save_img:
        #     cv2.imwrite(os.path.join(SAVE_FOLDER, prefix +  "_recognized_." + suffix), img_raw)
        return [], [], img_raw

    # Set face reference (112 x 112 based)
    reference = np.array([[38.29459953, 51.69630051],
       [73.53179932, 51.50139999],
       [56.02519989, 71.73660278],
       [41.54930115, 92.3655014 ],
       [70.72990036, 92.20410156]])
    crop_size = 112

    OUTPUT_IMAGE_PATH = output_dir


    OUTPUT_IMAGE_LIBRARY_PATH= os.path.join(OUTPUT_IMAGE_PATH, "library")

    # if detect_save_library is True:
    #     if not os.path.exists(OUTPUT_IMAGE_LIBRARY_PATH):
    #         os.makedirs(OUTPUT_IMAGE_LIBRARY_PATH)

    # Detected face images to numpy
    warped_face = np.zeros((len(dets), crop_size, crop_size, 3))
    for j, b in enumerate(dets):
        facial5points = b[5:].reshape((5, 2)) - b[0:2]
        b = list(map(int, b))

        h = b[3] - b[1]
        w = b[2] - b[0]
        shift = ( ( min(b[0], int(w/5)), min(b[1],int(h/5)) ) )

        # recognition을 위해서 face 영역 주변부까지 image 영역을 받아와 alignment & resize (112 x 112)
        x1 = max(b[0] - int(w/5), 0)
        y1 = max(b[1] - int(h/5), 0)
        x2 = min(b[2] + int(w/5), img_raw.shape[1])
        y2 = min(b[3] + int(h/5), img_raw.shape[0])

        warped_face[j] = warp_and_crop_face(img_raw[y1:y2, x1:x2], facial5points + shift, reference, crop_size=(crop_size, crop_size))

        # if detect_save_library == True:
        # save_path = os.path.join(OUTPUT_IMAGE_LIBRARY_PATH, "wapred_{}.jpg".format(j))
        # cv2.imwrite(save_path, warped_face[j])
        
        # ori_save_path = os.path.join(OUTPUT_IMAGE_LIBRARY_PATH, "ori_{}.jpg".format(j))
        # cv2.imwrite(ori_save_path, img_raw[y1:y2, x1:x2])


    # color & identity dictionary and library feature vector
    color_dict, library_dict, library_face = library[0], library[1], library[2]

    # detect 함수로 검출된 face 영역에 대해서 recognition을 위한 feature vector로 변환
    target = img2feature(device, net_recog, warped_face)
    # print(target.shape)
    pred_label, confidence = do_recog(target, library_face, thres = 2.5) #1.5


    # print(pred_label)
    names = list(map(lambda i: library_dict[i], pred_label))


    if recognition_save_img:
        for idx, b in enumerate(bbox):

            text_name = "{}".format(library_dict[pred_label[idx]])
            text_conf = "{:.2f}".format(confidence[idx])

            # Face rectangle
            if show_bbox:
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color_dict[pred_label[idx]], 2)

            # Text
            if show_identity:
                cx = b[0]
                cy = b[1]
                cv2.putText(img_raw, text_name, (cx, cy - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])
                # if confidence[idx] > 0.5:
                #     cv2.putText(img_raw, text_conf, (cx, cy + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])


    return names, confidence, img_raw


def make_video(image_folder, video_name, fps=30):
    # Get all image file names sorted by name or modification time
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure images are processed in order

    # Read the first image to get its size (height, width)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec (you can use 'XVID', 'MJPG', etc.)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Iterate over all images and write them to the video file
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer
    video.release()



def calculate_preset(device, img_path):
    # rand_img = cv2.imread("/data/deidentification/Projects/DATASET/friends_imgs/0.png", cv2.IMREAD_COLOR)
    rand_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H,W,C = rand_img.shape

    # Get scale / priorbox
    scale_box = torch.Tensor([W, H, W, H])
    scale_box = scale_box.to(device)
    scale_landm = torch.Tensor([W, H, W, H,
                                W, H, W, H,
                                W, H])
    scale_landm = scale_landm.to(device)

    # Get Priorbox
    priorbox = PriorBox(image_size=(H, W))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    return prior_data, scale_box, scale_landm

def calculate_preset_tensor(device, img):
    # rand_img = cv2.imread("/data/deidentification/Projects/DATASET/friends_imgs/0.png", cv2.IMREAD_COLOR)
    H,W,C = img.shape

    # Get scale / priorbox
    scale_box = torch.Tensor([W, H, W, H])
    scale_box = scale_box.to(device)
    scale_landm = torch.Tensor([W, H, W, H,
                                W, H, W, H,
                                W, H])
    scale_landm = scale_landm.to(device)

    # Get Priorbox
    priorbox = PriorBox(image_size=(H, W), num_anc=3)
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    return prior_data, scale_box, scale_landm