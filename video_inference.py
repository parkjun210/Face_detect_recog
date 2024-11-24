import torch
import warnings
warnings.filterwarnings('ignore')
import os
import configparser
import argparse
import numpy as np
import cv2
from time import time
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from SNU_FaceDetection.models.retinaface import RetinaFace
from SNU_FaceDetection.utils.helpers import *
from util.video_util import *

from SNU_FaceRecognition.model.backbone.model import IR_50


def init(cfg_dir, useGPU = True):

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='',
                        help='deidentified image save folder')

    args = parser.parse_args()

    
    if os.path.exists(cfg_dir):
        # Config 파일로 부터 변수 내용을 가져온다.
        config = configparser.RawConfigParser()
        config.read("video_detect.cfg")

        basic_config = config["basic_config"]

        args.detection_weight_file = basic_config["detection_weight_file"]
        if not os.path.exists(args.detection_weight_file):
            print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
            #sys.exit(2)

        args.recognition_weight_file = basic_config["recognition_weight_file"]
        if not os.path.exists(args.recognition_weight_file):
            print(">>> NOT Exist RECOGNITION WEIGHT File {0}".format(args.recognition_weight_file))
            #sys.exit(2)

        args.input_video_path = basic_config["input_video_path"]
        if args.input_video_path == "" :
            print(">>> NOT Exist Inference files")

        if args.output_dir == "":
            args.output_dir = basic_config["output_dir"]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.recognition_library_path = basic_config["recognition_library_path"]
        if not os.path.exists(args.recognition_library_path):
            print(">>> NOT Exist RECOGNITION LIBRARY {0}".format(args.recognition_library_path))

        args.gpu_num = basic_config["gpu_num"]
        if args.gpu_num == "" :
            print(">>> NOT Assign GPU Number")
            #sys.exit(2)

    torch.set_grad_enabled(False)

    # Session Parameters
    GPU_NUM = args.gpu_num

    # weight 파일들
    DETECTION_WEIGHT_FILE = args.detection_weight_file
    RECOGNITION_WEIGHT_FILE = args.recognition_weight_file
    
    # set parameter for detection model    
    if "resnet_anc2_casT_fpn3" in args.detection_weight_file:
        DETECT_MODEL_VERSION = "retina"
        NUM_ANCHOR = 2
    elif "tina_iou_anc3_casT_fpn3" in args.detection_weight_file:
        DETECT_MODEL_VERSION = "tina"
        NUM_ANCHOR = 3        
    else:
        print(">>> CHECK FOR DETECTION WEIGHT FILE PATH {0}".format(args.detection_weight_file))  


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
    net_detect = RetinaFace(phase='test', version=DETECT_MODEL_VERSION, anchor_num=NUM_ANCHOR)

    checkpoint = torch.load(DETECTION_WEIGHT_FILE, map_location=device)

    net_detect.load_state_dict(checkpoint['network'])
    net_detect = net_detect.to(device)
    net_detect.eval()
    cudnn.benchmark = True


    net_recog = IR_50([112, 112])
    net_recog.load_state_dict(torch.load(RECOGNITION_WEIGHT_FILE))
    net_recog = net_recog.to(device)
    net_recog.eval()


    return (args, device, net_detect, net_recog)




def inference(args, device, net_detect, net_recog):
    torch.set_grad_enabled(False)

    library = set_library(device, net_recog, lib_dir=args.recognition_library_path)

    frames = []
    cap = cv2.VideoCapture(args.input_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 데이터셋 로드
    test_dataset = Video_Dataset(frames)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    prior_data, scale_box, scale_landm = calculate_preset_tensor(device, test_dataset[0][1])

    out_frames = []

    for i, data in enumerate(test_dataloader):
        img_t, img_np = data
        img_t = img_t.to(device)
        img_np = img_np[0].numpy()

        # 얼굴 감지
        bbox, dets = detect(device, net_detect, img_t, same_flag=True, p_data=prior_data, s_box=scale_box, s_landm=scale_landm)

        # 얼굴 인식
        names, confidence, out_frame = recognize(device, net_recog, img_np, i, bbox, dets, library, output_dir=args.output_dir, save_flag=True)

        out_frames.append(out_frame)

    height, width, _ = out_frames[0].shape
    result_video_path = os.path.join(args.output_dir, 'result_video.mp4')
    output_video = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in out_frames:
        output_video.write(frame)
    output_video.release()




if __name__=="__main__":

    args, device, net_detect, net_recog = init(cfg_dir = "video_detect.cfg")

    inference(args, device, net_detect, net_recog)


