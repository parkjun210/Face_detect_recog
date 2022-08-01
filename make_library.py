import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np
import cv2

from config import parse_args
from SNU_FaceDetection.models.retinaface import RetinaFace
from SNU_FaceDetection.utils.helpers import *

from tqdm import tqdm

from SNU_FaceRecognition.applications.align.align_trans import get_reference_facial_points, warp_and_crop_face


def make_library(args):

    torch.set_grad_enabled(False)

    # Design Parameters
    NETWORK = 'resnet50'

    # Session Parameters
    GPU_NUM = args.gpu_num

    # Directory Parameters
    INFERENCE_DIR = args.inference_dir
    EXP_NAME = args.experiment_name
    EXP_DIR = 'SNU_FaceDetection/experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, "ckpt/")
    WEIGHTS = "ckpt-best.pth"

    # Inference Parameters
    CONFIDENCE_THRESHOLD = 0.99
    NMS_THRESHOLD = 0.3
    SAVE_FOLDER = os.path.join("results", args.inference_save_folder)

    create_path(SAVE_FOLDER)

    rgb_mean = (104, 117, 123)
    result_bboxes = []


    img_path = []
    DATA_DIR = os.path.join("data", INFERENCE_DIR)
    for file in os.listdir(DATA_DIR):
        if file.endswith('.png'):
            img_path.append(os.path.join(DATA_DIR, file))
        elif file.endswith('.jpg'):
            img_path.append(os.path.join(DATA_DIR, file))


    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set up Network
    net = RetinaFace(phase='test')
    output_path = CKPT_DIR + NETWORK + '_' + WEIGHTS
    checkpoint = torch.load(output_path)
    net.load_state_dict(checkpoint['network'])
    net.eval()
    net = net.to(device)
    cudnn.benchmark = True

    
    # Get image size
    if args.infer_imsize_same:
        rand_img = cv2.imread(img_path[0], cv2.IMREAD_COLOR)
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


    # Set face reference
    reference = get_reference_facial_points(default_square = True)
    crop_size = 112

    if len(img_path) == 0:
        print("There is no image")
        return 0

    # Evaluation Start
    pbar = tqdm(img_path)
    for i, img_name in enumerate(pbar):
        pbar.set_description("Processing %s" % img_name.split('/')[-1])
        
        # Obtain data / to GPU
        img_raw = cv2.imread(img_path[i], cv2.IMREAD_COLOR)
        
        img = img_raw.astype(np.float32)
        img -= rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        img = img.to(device)
        img = torch.unsqueeze(img, dim=0)


        # _, _, H, W = img.shape

        # Forward
        _, out = net(img)

        loc, conf, landms = out

        # Decode
        if args.infer_imsize_same:
            scores, boxes, landms = decode_output_inf(loc, conf, landms, prior_data, scale_box, scale_landm)
        else:
            scores, boxes, landms = decode_output(img, loc, conf, landms, device)


        # NMS
        # dets : (box, 4(loc) + 1(conf) + 10(landm))
        dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


        if len(dets) != 0:
            result_bboxes.append(dets[:, :4].astype(int))

            # Detected face images to numpy
            warped_face = np.zeros((len(dets), crop_size, crop_size, 3))
            for j, b in enumerate(dets):
                facial5points = b[5:].reshape((5, 2)) - b[0:2]
                b = list(map(int, b))

                h = b[3] - b[1]
                w = b[2] - b[0]
                shift = ( ( min(b[0], int(w/5)), min(b[1],int(h/5)) ) )

                x1 = max(b[0] - int(w/5), 0)
                y1 = max(b[1] - int(h/5), 0)
                x2 = min(b[2] + int(w/5), img_raw.shape[1])
                y2 = min(b[3] + int(h/5), img_raw.shape[0])

                warped_face[j] = warp_and_crop_face(img_raw[y1:y2, x1:x2], facial5points + shift, reference, crop_size=(crop_size, crop_size))
                
                # Write alinged face images
                cv2.imwrite(os.path.join(SAVE_FOLDER, "{}_{}.jpg".format(os.path.basename(img_name).split('.')[0], j)), warped_face[j])


    return result_bboxes



if __name__=="__main__":
    args = parse_args()
    result = make_library(args)
    print("Done!")
