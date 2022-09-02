from utils.helpers import decode_output
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import os
import numpy as np
import cv2
from time import time

from config import parse_args
from models.retinaface import RetinaFace
from utils.helpers import *

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def inference(args):

    torch.set_grad_enabled(False)

    # Design Parameters
    NETWORK = 'resnet50'

    # Session Parameters
    GPU_NUM = args.gpu_num

    # Directory Parameters
    INFERENCE_DIR = args.inference_dir
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, "ckpt/")
    WEIGHTS = "ckpt-best.pth"

    # Inference Parameters
    MS_INFERENCE = 1
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.3
    SAVE_FOLDER = args.inference_save_folder
    SAVE_IMG = args.save_img

    MASK = args.mask
    SAVE_MASK = args.save_mask

    create_path(SAVE_FOLDER)

    rgb_mean = (104, 117, 123)

    img_path = []

    DATA_DIR = os.path.join("data/", INFERENCE_DIR)

    for file in os.listdir(DATA_DIR):
        if file.endswith('.png'):
            img_path.append(DATA_DIR + file)
        elif file.endswith('.jpg'):
            img_path.append(DATA_DIR + file)


    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate results.txt
    # f = open(SAVE_FOLDER +'/'+ EXP_NAME+'_' +'inference_results.txt', 'w')
    # f.write("[Inference for experiments '" + EXP_NAME + "' has started]\n\n")
    num_images = len(img_path)
    # f.write("Number of inference images: " + str(num_images) + "\n\n")

    # Set up Network
    net = RetinaFace(phase='test')
    output_path = CKPT_DIR + NETWORK + '_' + WEIGHTS
    checkpoint = torch.load(output_path)
    # f.write("Start loading weights...\n")
    net.load_state_dict(checkpoint['network'])
    net.eval()
    net = net.to(device)
    cudnn.benchmark = True

    # Evaluation Start
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

    avg_ftime = 0
    avg_dtime = 0
    avg_ntime = 0

    result_bboxes = []
    f1 = open(os.path.join(SAVE_FOLDER,'inference_results.txt'), 'w')

    result_face_masks, result_face_bboxes, result_head_masks, result_head_bboxes = [], [], [], []

    for i, img_name in enumerate(img_path):

        # Obtain data / to GPU
        img_raw = cv2.imread(img_path[i], cv2.IMREAD_COLOR)
        mask_img = img_raw.copy()

        img = img_raw.astype(np.float32)
        img -= rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        img = img.to(device)
        img = torch.unsqueeze(img, dim=0)

        f_ftime = 0
        f_dtime = 0

        # _, _, H, W = img.shape

        # Forward
        fs_time = time()
        _, out = net(img)

        loc, conf, landms = out
        fe_time = time()

        # Decode
        ds_time = time()
        if args.infer_imsize_same:
            scores, boxes, landms = decode_output_inf(loc, conf, landms, prior_data, scale_box, scale_landm)
        else:
            scores, boxes, landms = decode_output(img, loc, conf, landms, device)
        de_time = time()

        f_ftime += (fe_time - fs_time)
        f_dtime += (de_time - ds_time)

        # NMS
        # dets : (box, 4(loc) + 1(conf) + 10(landm))
        ns_time = time()
        dets = do_nms(scores, boxes, landms, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        ne_time = time()

        f_ntime = (ne_time - ns_time)


        if len(dets) != 0:
            result_bboxes.append(dets[:, :4].astype(int))

        if SAVE_IMG:
            create_path(SAVE_FOLDER + 'result_images/')
            for b in dets:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))

                # Face
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                # Text
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # Landmark
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            img_name = img_name
            img_name = img_name.split('/')
            img_name = img_name[-1]

            cv2.imwrite(SAVE_FOLDER + 'result_images/' + img_name, img_raw)

        avg_ftime += f_ftime
        avg_dtime += f_dtime
        avg_ntime += f_ntime

        if MASK:
            # BG_COLOR = (192, 192, 192)  # gray
            BG_COLOR = (0, 0, 0)  # black
            MASK_COLOR = (255, 255, 255)  # white
            create_path(os.path.join(SAVE_FOLDER, 'masks'))
            create_path(os.path.join(SAVE_FOLDER, 'faces'))
            create_path(os.path.join(SAVE_FOLDER, 'head_masks'))

            if not SAVE_IMG:
                img_name = img_name
                img_name = img_name.split('/')
                img_name = img_name[-1]

            with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
                image_height, image_width, _ = mask_img.shape
                face_masks, face_bboxes, head_masks, head_bboxes = [], [], [], []
                for idx, b in enumerate(dets):
                    b = list(map(int, b))

                    cx, cy, w, h = (b[0]+b[2])//2, (b[1]+b[3])//2, (b[2] - b[0]), (b[3] - b[1])
                    new_x1, new_y1, new_x2, new_y2 = int(cx - 2*w), int(cy - 2*h), int(cx + 2*w), int(cy + 2*h)
                    new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
                    new_x2, new_y2 = min(image_width, new_x2), min(image_height, new_y2)

                    head_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                    head = mask_img[new_y1:new_y2, new_x1:new_x2, :]

                    # Convert the BGR image to RGB before processing.
                    results = selfie_segmentation.process(cv2.cvtColor(head, cv2.COLOR_BGR2RGB))

                    # Draw selfie segmentation on the background image.
                    # To improve segmentation around boundaries, consider applying a joint
                    # bilateral filter to "results.segmentation_mask" with "image".
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    # Generate solid color images for showing the output selfie segmentation mask.
                    fg_image = np.zeros(head.shape, dtype=np.uint8)
                    fg_image[:] = MASK_COLOR
                    bg_image = np.zeros(head.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    mask = np.where(condition, fg_image, bg_image)

                    head_masks.append(mask)
                    CY, CX = mask.shape[0]//2, mask.shape[1]//2
                    H, W = h//2, w//2
                    face_masks.append(mask[CY-H:CY+H, CX-W:CX+W, :])
                    face_bboxes.append([CX-W, CY-H, CX+W, CY+H])

                    if SAVE_MASK:
                        original_img_name, original_img_format = img_name.split(".")
                        cv2.imwrite(os.path.join(SAVE_FOLDER, 'head_masks', original_img_name + '_' + str(idx) + '.' + original_img_format), mask)
                        cv2.imwrite(os.path.join(SAVE_FOLDER, 'faces', original_img_name + '_' + str(idx) + '.' + original_img_format), head)
                        cv2.imwrite(os.path.join(SAVE_FOLDER, 'masks', original_img_name + '_' + str(idx) + '.' + original_img_format), mask[CY-H:CY+H, CX-W:CX+W, :])


                result_face_masks.append(face_masks)
                result_face_bboxes.append(face_bboxes)
                result_head_masks.append(head_masks)
                result_head_bboxes.append(head_bboxes)


        if i % 10 == 0:
            print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s decode: {:.4f}s nms: {:.4f}s'.format(i + 1, num_images, f_ftime, f_dtime, f_ntime))

        f1.write(img_name + '\n')
        for b in dets:
            b_ = list(map(int, b[:4]))
            f1.write('%d %d %d %d %f\n' % (b_[0], b_[1], b_[2], b_[3], b[4]))

    f1.close()


    # f.write("Calculating process has done !\n\n")
    avg_ftime /= len(img_path)
    avg_dtime /= len(img_path)
    avg_ntime /= len(img_path)

    print('Average inference Time : {:.4f}s = {:.4f}s + {:.4f}s + {:.4f}s (forward / decode / nms)'.format(avg_ftime + avg_dtime + avg_ntime, avg_ftime, avg_dtime, avg_ntime))

    # f.write("Total inference time is " + str(round(avg_ftime + avg_dtime + avg_ntime, 2)) + " secs\n")
    # f.write('Average inference Time per image : {:.4f}s = {:.4f}s + {:.4f}s + {:.4f}s (forward / decode / nms)\n\n'.format(avg_ftime + avg_dtime + avg_ntime, avg_ftime, avg_dtime, avg_ntime))

    # f.write("[Infernce for experiments '" + EXP_NAME + "' has succesfully Ended]\n")
    # f.close()
    return result_bboxes, result_face_bboxes, result_face_masks, result_head_bboxes, result_head_masks



if __name__=="__main__":
    args = parse_args()
    result = inference(args)
    result_bboxes, face_bboxes, face_masks, head_bboxes, head_masks = result
    print("Done!")