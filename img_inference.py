import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np
import cv2

from config import parse_args
from SNU_FaceDetection.models.retinaface import RetinaFace
from SNU_FaceDetection.utils.helpers import *

from tqdm import tqdm
from time import time

from SNU_FaceRecognition.applications.align.align_trans import get_reference_facial_points, warp_and_crop_face
from SNU_FaceRecognition.model.backbone.model_irse import IR_SE_50

from pipeline_tools.recognition import perform_recog


def inference(args):

    torch.set_grad_enabled(False)

    # Design Parameters
    NETWORK = 'resnet50'

    # Session Parameters
    GPU_NUM = args.gpu_num

    # Directory Parameters
    INFERENCE_DIR = args.inference_dir
    EXP_NAME = args.experiment_name
    EXP_DIR = os.path.join('SNU_FaceDetection/experiments/', EXP_NAME)
    CKPT_DIR = os.path.join(EXP_DIR, "ckpt/")
    WEIGHTS = "ckpt-best.pth"

    RECOG_EXP_NAME = args.recog_experiment_name
    BACKBONE_DIR = os.path.join('SNU_FaceRecognition/checkpoints_best', RECOG_EXP_NAME)

    # Inference Parameters
    CONFIDENCE_THRESHOLD = 0.99
    NMS_THRESHOLD = 0.3
    SAVE_FOLDER = os.path.join("results/", args.inference_save_folder)
    SAVE_IMG = args.save_img
    LIBRARY_DIR = args.library_dir

    create_path(SAVE_FOLDER)

    rgb_mean = (104, 117, 123)
    result_bboxes = []


    img_path = []
    DATA_DIR = os.path.join("data/", INFERENCE_DIR)
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

    # Define the Backbone (ArcFace)
    BACKBONE = IR_SE_50([112, 112])
    BACKBONE_RESUME_ROOT = os.path.join(BACKBONE_DIR, 'Backbone_Best.pth')
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    BACKBONE = BACKBONE.to(device)

    avg_ftime = 0
    avg_dtime = 0
    avg_ntime = 0

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

    f1 = open(SAVE_FOLDER + '/' + INFERENCE_DIR + '_' + 'inference_results.txt', 'w')


    reference = get_reference_facial_points(default_square = True)
    embedding_size = 512
    crop_size = 112
    recog_flag = True

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
    library_folder = os.path.join("library", LIBRARY_DIR)
    library_count = 0

    for file in os.listdir(library_folder):
        path = os.path.join(library_folder, file)
        if os.path.isdir(path):
            for img_name in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR)
                library_face.append(img)
                library_dict[k + 1] = os.path.basename(path)
                color_dict[k + 1] = colors[library_count + 1]
                k += 1
            library_count += 1
        if file.endswith('.jpg'):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            library_face.append(img)
            library_dict[k + 1] = os.path.splitext(file)[0]
            color_dict[k + 1] = colors[library_count + 1]
            library_count += 1
            k += 1

    library_face = np.array(library_face)



    if len(img_path) == 0:
        print("There is no image")
        return 0

    # Image inference
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


            # Detected face images to numpy
            if recog_flag:
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
                    
                
                # Recognition using detected faces & library faces                
                pred_label, confidence = perform_recog(device, embedding_size, BACKBONE, warped_face, library_face, thres = 0.1)

        if SAVE_IMG:
            create_path(SAVE_FOLDER + '/result_images/')
            for idx, b in enumerate(dets):
                text_name = "{}".format(library_dict[pred_label[idx]])
                text_conf = "{:.2f}".format(confidence[idx])

                b = list(map(int, b))

                # Face rectangle
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color_dict[pred_label[idx]], 2)

                # Text
                cx = b[0]
                cy = b[1]
                cv2.putText(img_raw, text_name, (cx, cy - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])
                if confidence[idx] > 0.5:
                    cv2.putText(img_raw, text_conf, (cx, cy + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[pred_label[idx]])

                # Landmark
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            img_name = img_name.split('/')
            img_name = img_name[-1]

            cv2.imwrite(SAVE_FOLDER + '/result_images/' + img_name, img_raw)

        avg_ftime += f_ftime
        avg_dtime += f_dtime
        avg_ntime += f_ntime

        f1.write(img_name + '\n')
        for b in dets:
            b_ = list(map(int, b[:4]))
            f1.write('%d %d %d %d %f\n' % (b_[0], b_[1], b_[2], b_[3], b[4]))

    f1.close()

    avg_ftime /= len(img_path)
    avg_dtime /= len(img_path)
    avg_ntime /= len(img_path)

    print('Average inference Time : {:.4f}s = {:.4f}s + {:.4f}s + {:.4f}s (forward / decode / nms)'.format(avg_ftime + avg_dtime + avg_ntime, avg_ftime, avg_dtime, avg_ntime))


    return result_bboxes



if __name__=="__main__":
    args = parse_args()
    result = inference(args)
    print("Done!")
