# from dbm.ndbm import library
from turtle import clear
import torch
import torch.backends.cudnn as cudnn

import os
import shutil
import numpy as np
import cv2
import configparser
import argparse

from SNU_FaceDetection.models.retinaface import RetinaFace
from SNU_FaceDetection.utils.helpers import *

from tqdm import tqdm

# For Detection
from SNU_FaceRecognition.applications.align.align_trans import get_reference_facial_points, warp_and_crop_face

# For Recog
from SNU_FaceRecognition.model.backbone.model_irse import IR_SE_50
from pipeline_tools.recognition import *



# deIdentification function 
# process_type = 2 : blurring image
def deIdentify(roi_img, process_type):
    if process_type == 0:
        return roi_img
    elif process_type == 1:
        return roi_img
    elif process_type == 2:
        # GaussianBlur
        blur = cv2.GaussianBlur(roi_img, (99, 99), 20)

        return blur
    else:
        return roi_img


# init() function
# input : config file path
# output : args, device, detection network, recognition network
def init(cfg_dir, useGPU = True):

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img', type=str, default='',
                        help='image file for face dectection ')

    parser.add_argument('--output_dir', type=str, default='',
                        help='deidentified image save folder')

    args = parser.parse_args()
    
    if os.path.exists(cfg_dir):
        # Config 파일로 부터 변수 내용을 가져온다.
        config = configparser.RawConfigParser()
        config.read("detect.cfg")

        basic_config = config["basic_config"]

        args.detection_weight_file = basic_config["detection_weight_file"]
        if not os.path.exists(args.detection_weight_file):
            print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
            #sys.exit(2)

        args.recognition_weight_file = basic_config["recognition_weight_file"]
        if not os.path.exists(args.recognition_weight_file):
            print(">>> NOT Exist RECOGNITION WEIGHT File {0}".format(args.recognition_weight_file))
            #sys.exit(2)

        if args.output_dir == "":
            args.output_dir = basic_config["output_dir"]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.gpu_num = basic_config["gpu_num"]
        if args.gpu_num == "" :
            print(">>> NOT Assign GPU Number")
            #sys.exit(2)

        args.infer_imsize_same = config.getboolean('basic_config', 'infer_imsize_same')
        if args.infer_imsize_same == "" :
            args.infer_imsize_same = False

        args.result_savefile = config.getboolean('basic_config', 'result_savefile')
        if args.result_savefile == "" :
            args.result_savefile = False

        args.deidentified_type = basic_config["deidentified_type"]
        if args.deidentified_type == "" :
            args.deidentified_type = '2'

        args.detect_save_library = config.getboolean('basic_config', 'detect_save_library')
        if args.detect_save_library == "" :
            args.detect_save_library = False

        args.recognition_library_path = basic_config["recognition_library_path"]
        if not os.path.exists(args.recognition_library_path):
            print(">>> NOT Exist RECOGNITION LIBRARY {0}".format(args.recognition_library_path))

        args.recognition_inference_save_folder = basic_config["recognition_inference_save_folder"]
        args.recognition_save_img = config.getboolean('basic_config', 'recognition_save_img')
        if args.recognition_save_img == "" :
            args.recognition_save_img = False

    torch.set_grad_enabled(False)

    # Session Parameters
    GPU_NUM = args.gpu_num

    # weight 파일들
    DETECTION_WEIGHT_FILE = args.detection_weight_file
    RECOGNITION_WEIGHT_FILE = args.recognition_weight_file


    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    if useGPU:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("CUDA is not available, device set to CPU")
    else:
        device = torch.device("cpu")

    # Set up Network
    net_detect = RetinaFace(phase='test')

    checkpoint = torch.load(DETECTION_WEIGHT_FILE, map_location=device)

    net_detect.load_state_dict(checkpoint['network'])
    net_detect = net_detect.to(device)
    net_detect.eval()
    cudnn.benchmark = True

    # For Recog
    # #######################################################################
    net_recog = IR_SE_50([112, 112])

    if os.path.isfile(RECOGNITION_WEIGHT_FILE):
        net_recog.load_state_dict(torch.load(RECOGNITION_WEIGHT_FILE))
    net_recog = net_recog.to(device)
    net_recog.eval()
    # #######################################################################

    print(" 학습한 가중치 파일은 {0}".format(DETECTION_WEIGHT_FILE))


    return (args, device, net_detect, net_recog)


# inference할 dataset 처리하는 함수
# input : inference할 image directory
# output
    # imglist       = detection network에 넣을 수 있게 processing한 image tensor
    # rawimglist    = numpy image
    # imgpathlist   = img path list
def make_dataset(device, img_dir):
    rawimglist = []
    imglist = []
    imgpathlist = []
    rgb_mean = (104, 117, 123)
    imgsuffix = [".png", ".jpeg", '.jpg', ".PNG", ".JPEG", ".JPG"]

    # img_dir가 폴더가 아닌 파일일 때
    if any(img_dir.endswith(s) for s in imgsuffix):
        imgpathlist.append(img_dir)
        img_raw = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        rawimglist.append(img_raw)
        # 이미지 연산하기 위하 형태로 변환
        img = img_raw.astype(np.float32)
        img -= rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        img = img.to(device)
        img = torch.unsqueeze(img, dim=0)
        imglist.append(img)

        return (imglist, rawimglist, imgpathlist)
        
    # img_dir가 폴더일 때   
    for image in os.listdir(img_dir):
        if any(image.endswith(s) for s in imgsuffix):
            imgpathlist.append(os.path.join(img_dir, image))
            img_raw = cv2.imread(os.path.join(img_dir, image), cv2.IMREAD_COLOR)
            rawimglist.append(img_raw)
            # 이미지 연산하기 위하 형태로 변환
            img = img_raw.astype(np.float32)
            img -= rgb_mean
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)

            img = img.to(device)
            img = torch.unsqueeze(img, dim=0)
            imglist.append(img)

    return (imglist, rawimglist, imgpathlist)



# detection funtion : Input image에 대해 face detection 수행
# input : 
    # args, device, net_detect
    # img       = tensor image (by make_dataset func)
    # img_raw   = numpy image (by make_dataset func)
    # imgpath   = image path (by make_dataset func)
# output :
    # result_bboxes = image에서 검출된 face의 bounding box 좌표
    # warped_face   = 검출된 face image에 대해 face alignment, deidentified된 face image patch(numpy)
def detect(args, device, net_detect, img, img_raw, imgpath):

    # Set face reference (112 x 112 based)
    reference = np.array([[38.29459953, 51.69630051],
       [73.53179932, 51.50139999],
       [56.02519989, 71.73660278],
       [41.54930115, 92.3655014 ],
       [70.72990036, 92.20410156]])
    crop_size = 112

    # Inference Parameters
    CONFIDENCE_THRESHOLD = 0.99
    NMS_THRESHOLD = 0.3


    # 입력파일 파일명, 확장자 추출,
    # 결과 폴더에 출력할 파일명 생성 
    INPUT_IMAGE_FILE = imgpath
    OUTPUT_IMAGE_PATH = args.output_dir

    INPUT_IMAGE_FILE_ONLYNM = os.path.basename(INPUT_IMAGE_FILE)
    file_ext = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[-1]
    file_nm = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[0]
    OUTPUT_IMAGE_FILE = os.path.join(OUTPUT_IMAGE_PATH, file_nm + "_deidentified" + file_ext)
    OUTPUT_IMAGE_LIBRARY_PATH= os.path.join(OUTPUT_IMAGE_PATH, "library")

    if args.detect_save_library is True:
        if not os.path.exists(OUTPUT_IMAGE_LIBRARY_PATH):
            os.makedirs(OUTPUT_IMAGE_LIBRARY_PATH)
    
    # input image directory의 image resolution이 같을 경우 detection을 위한 image prior 계산
    if args.infer_imsize_same:
        rand_img = cv2.imread(INPUT_IMAGE_FILE, cv2.IMREAD_COLOR)
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


    # Forward
    _, out = net_detect(img)

    loc, conf, landms = out

    # Decode
    if args.infer_imsize_same:
        scores, boxes, landms = decode_output_inf(loc, conf, landms, prior_data, scale_box, scale_landm)
    else:
        scores, boxes, landms = decode_output(img, loc, conf, landms, device)


    # NMS
    # dets : (box, 4(loc) + 1(conf) + 10(landm))
    dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # 검출된 face가 없을 경우 return None
    if len(dets) == 0:
        print("{} : face does not exist.".format(file_nm))
        return [], []

    
    result_bboxes = dets[:, :4].astype(int)

    if len(dets) != 0:

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


            # Detect영역 떼와서 비식별화처리하고 다시 원본에 적용하고하는 부분
            roi = img_raw[b[1]:b[3], b[0]:b[2]]

            if args.detect_save_library == True:
                save_path = os.path.join(OUTPUT_IMAGE_LIBRARY_PATH, "{}_{}.jpg".format(file_nm, j))
                cv2.imwrite(save_path, warped_face[j])

            # 원본 image에 대해서 비식별화 & recognition function에 넘길 부분까지 비식별화
            processed_img = deIdentify(roi, int(args.deidentified_type))
            warped_face[j] = deIdentify(warped_face[j], int(args.deidentified_type))
            
            if args.result_savefile is True:
                # img_raw[y1:y1 + h, x1:x1 + w] = processed_img
                img_raw[b[1]:b[3], b[0]:b[2]] = processed_img
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    if args.result_savefile is True:
        cv2.imwrite(OUTPUT_IMAGE_FILE, img_raw)

    return result_bboxes, warped_face


# image를 받아와서 recognition network를 통해 feature vector로 변환하는 함수
def img2feature(device, net_recog, imgs, embedding_size = 512):

    feature_size = embedding_size

    imgs = imgs / 255
    num_imgs = imgs.shape[0]
    imgs_embed = np.zeros([num_imgs, feature_size])
    imgs = torch.tensor(imgs[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2).float()

    imgs = normalize(imgs)
    imgs_ccropped = ccrop_batch(imgs)
    imgs_fliped = hflip_batch(imgs_ccropped)
    imgs_emb_batch = net_recog(imgs_ccropped.to(device)).cpu() + net_recog(imgs_fliped.to(device)).cpu()
    imgs_embed[0:] = l2_norm(imgs_emb_batch)
    feature = imgs_embed

    return feature


# Face Library로부터 color & identity dictionary, library feature vector 생성하는 함수
# output :
    # color_dict        = identity에 따라 bounding box 색 dictionary
    # library_dict      = identity dictionary
    # library_feature   = library folder의 image들의 recognition용 feature vector
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

    # Set library
    library_face = []
    library_dict = {}
    color_dict = {}
    library_dict[0] = 'Unknown'
    color_dict[0] = colors[0]
    k = 0

    library_folder = lib_dir
    library_count = 0

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

    # return (dict, dict, numpy)
    return (color_dict, library_dict, library_feature)



def add_img_to_library(args, img_dir, identity):
    #adds image to corresponding identity library, and creates one if the identity does not exists

    if not os.path.exists(img_dir):
        print("Image does not exist!")
        return

    filename = img_dir.split('/')[-1]
    library_folder = args.recognition_library_path
    identities = [folder for folder in os.listdir(library_folder)]
    if not identity in identities:
        print("Identity does not exist, creating '" + identity + "' library")
        os.makedirs(os.path.join(library_folder, identity))
    savefolder = os.path.join((os.path.join(library_folder, identity)), filename)
    shutil.copyfile(img_dir, savefolder)
    print("Image added to corresponding identity library")
    
def clear_library(args):
    #clears all identity folders
    library_folder = args.recognition_library_path
    shutil.rmtree(library_folder)
    print("Library successfully cleared")

def clear_identity(args, identity):
    #clears identity folder
    library_folder = args.recognition_library_path
    identities = [folder for folder in os.listdir(library_folder)]
    if not identity in identities:
        print("Identity does not exist")
    else:
        shutil.rmtree(os.path.join(library_folder, identity))
        print("Identity '" + identity + "' library cleared!")



# recognize for single image
# input : recognition network / detect function으로 추출한 bbox, warped_face / library
# output :
    # names     = Single image에 들어있는 face image identity 
    # img_raw   = recognition ouput image
def recognize(args, device, net_recog, img_raw, imgpath, bbox, warped_face, library, show_bbox = True, show_identity = True, return_img = False):

    if len(bbox) == 0:
        return []

    # color & identity dictionary and library feature vector
    color_dict, library_dict, library_face = library[0], library[1], library[2]

    # detect 함수로 검출된 face 영역에 대해서 recognition을 위한 feature vector로 변환
    target = img2feature(device, net_recog, warped_face)

    pred_label, confidence = do_recog(target, library_face, thres = .05) #1.5

    names = []
    if args.recognition_save_img:
        for idx, b in enumerate(bbox):

            text_name = "{}".format(library_dict[pred_label[idx]])
            names.append(library_dict[pred_label[idx]])
            text_conf = "{:.2f}".format(confidence[idx])

            b = list(map(int, b))

            # Face rectangle
            if show_bbox:
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color_dict[pred_label[idx]], 2)

            # Text
            if show_identity:
                cx = b[0]
                cy = b[1]
                cv2.putText(img_raw, text_name, (cx, cy - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])
                if confidence[idx] > 0.5:
                    cv2.putText(img_raw, text_conf, (cx, cy + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])

    
    SAVE_FOLDER = os.path.join(args.output_dir, args.recognition_inference_save_folder)
    os.makedirs(SAVE_FOLDER, exist_ok=True)
        
    val = imgpath.split("/")
    name = val[-1]
    suffix = name.split(".")[-1]
    prefix = name.split(".")[0]
    cv2.imwrite(os.path.join(SAVE_FOLDER, prefix +  "_recognized." + suffix), img_raw)
    if return_img:
        return names, img_raw
    return names




if __name__=="__main__":

    args, device, net_detect, net_recog = init(cfg_dir = "detect.cfg")
    imgs_tensor, imgs_raw, imgs_path= make_dataset(device, img_dir = "data/sample_imgs_ispl")
    library = set_library(device, net_recog, lib_dir = args.recognition_library_path)

    for img_tensor, img_raw, img_path in zip(imgs_tensor, imgs_raw, imgs_path):
        bbox, warped = detect(args, device, net_detect, img_tensor, img_raw, img_path)
        names = recognize(args, device, net_recog, img_raw, img_path, bbox, warped, library)
        print(bbox)
        print(names)
