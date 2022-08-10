import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import torchvision.transforms as transforms
from model_RCAN import RCAN
from tqdm import tqdm
import cv2
# from utils import preprocess, calc_psnr

from torch.utils.data.dataloader import DataLoader


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

up = transforms.Compose([
            transforms.Resize([int(112), int(112)], interpolation=transforms.InterpolationMode.BICUBIC)
        ])

hr_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

lr_transforms = transforms.Compose([
            transforms.Resize([int(112 / 4), int(112 / 4)], interpolation=transforms.InterpolationMode.BICUBIC),
        ])

root_dir = '/data/parkjun210/ArcFace/Code_face_evolve/data/imgs_casia_sr'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, default='/data/parkjun210/ArcFace/RCAN-pytorch/output/scale_x4/RCAN_epoch_20_best.pth')
    parser.add_argument('--image_path', type=str, default='/data/parkjun210/ArcFace/Code_face_evolve/data/imgs_casia_sr')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    opt = parser.parse_args()


    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RCAN(opt).to(device)

    model.load_state_dict(torch.load('/data/parkjun210/ArcFace/RCAN-pytorch/output/scale_x4/RCAN_epoch_20_best.pth'))

    model.eval()
    count = 0
    with torch.no_grad():
        for (root, dirs, files) in tqdm(os.walk(root_dir)):
            if len(files) > 0:
                for file_name in files:
                    src_file = os.path.join(root, file_name)
                    hr = hr_transforms(pil_image.open(src_file).convert("RGB"))
                    lr = lr_transforms(hr)
                    lr = lr.to(device)


                    # lr_img = torch.permute(lr, (1, 2, 0))
                    # cv2.imwrite( os.path.join(root,'lr_{}.jpg'.format(file_name.replace(".jpg", ""))), cv2.cvtColor(lr_img.detach().cpu().numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255, )

                    # bi_img = torch.permute(up(lr), (1,2,0))
                    # cv2.imwrite( os.path.join(root,'bi_{}.jpg'.format(file_name.replace(".jpg", ""))), cv2.cvtColor(bi_img.detach().cpu().numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255, )

                    preds = model(lr)
                    preds = torch.permute(preds, (1, 2, 0))
                    cv2.imwrite( os.path.join(root,'sr_{}.jpg'.format(file_name.replace(".jpg", ""))), cv2.cvtColor(preds.detach().cpu().numpy(), cv2.COLOR_RGB2BGR).astype(float) * 255, )
                    count = count + 1
    
    print(count)
