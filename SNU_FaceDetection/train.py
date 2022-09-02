from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import logging
from time import time

from config import parse_args
from models.retinaface import RetinaFace
from layers.prior_box import PriorBox
from layers.multibox_loss import MultiBoxLoss
from utils.average_meter import AverageMeter
from utils.helpers import *
from utils.data_loaders import Dataset, detection_collate
from utils.data_transformer import preproc
from utils.metric import get_metric

#### Configuration ####
args = parse_args()

# Design Parameters
CROP_SIZE = 640
NETWORK = 'resnet50'
LAMBDA_LOC = 2.0

# Session Parameters
GPU_NUM = args.gpu_num
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

PRINT_EVERY = 1
SAVE_EVERY = 10
EVALUATE_EVERY = 10
# Directory Parameters
DATASET = "widerface/"
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, "ckpt/")
LOG_DIR = os.path.join(EXP_DIR, "log/")
IMG_DIR = os.path.join(EXP_DIR, 'img/')
WEIGHTS = "ckpt-best.pth"

# Inference Parameters
CONFIDENCE_THRESHOLD = 0.02
NMS_THRESHOLD = 0.3

# Create directory if it does not exist
create_path('experiments/')
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(IMG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'val'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

# Set up dataset
rgb_mean = (104, 117, 123) # bgr order
dataset = Dataset('train', preproc(CROP_SIZE, rgb_mean))
dataloader = DataLoader(dataset=dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=4,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=detection_collate)

test_dataset = Dataset('val', preproc(None, rgb_mean))
test_dataloader = DataLoader(dataset=test_dataset, 
                            batch_size=1, 
                            num_workers=4,
                             shuffle=False, 
                             drop_last=False, 
                             collate_fn=detection_collate)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up network
net = RetinaFace( phase='train')

# Move the network to gpu if possible
if torch.cuda.is_available():
    net.to(device)

cudnn.benchmark = True

# Load the pretrained model if exists
init_epoch = 0
best_easy = 0
best_medium = 0
best_hard = 0
best_metrics = 0
info_path = CKPT_DIR + NETWORK + '_info'

if os.path.exists(info_path):
    checkpoint = torch.load(info_path)
    init_epoch = checkpoint['epoch_idx']
    epoch_model_path = CKPT_DIR + NETWORK + '_' + 'ckpt-' + str(init_epoch).zfill(3) + '.pth'
    logging.info('Recovering from %s ...' % epoch_model_path)
    checkpoint = torch.load(epoch_model_path)
    net.load_state_dict(checkpoint['network'])
    logging.info('Recover completed. Current epoch = #%d' % (init_epoch))

# Set up loss function
criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)

# Set up Optimizer
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Prior Box
priorbox = PriorBox(image_size=(CROP_SIZE, CROP_SIZE))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

for epoch_idx in range(init_epoch+1, N_EPOCHS):

    net.train()

    # Metric holders
    cls_loss = AverageMeter()
    bbox_loss = AverageMeter()
    landm_loss = AverageMeter()
    total_loss = AverageMeter()

    start_time = time()

    for batch_idx, data in enumerate(dataloader):

        # Obtain data / to GPU
        images, targets, _ = data
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # Forward
        out1, out2 = net(images)

        # Calculate Loss
        loss_l, loss_c, loss_landm = criterion(out1, priors, targets)
        loss_l_2, loss_c_2, loss_landm_2 = criterion(out2, priors, targets)

        loss_l += loss_l_2
        loss_c += loss_c_2
        loss_landm += loss_landm_2

        loss_l *= LAMBDA_LOC
        loss = loss_l + loss_c + loss_landm

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update Holders
        cls_loss.update(loss_c.item())
        bbox_loss.update(loss_l.item())
        landm_loss.update(loss_landm.item())
        total_loss.update(loss.item())

    end_time = time()
    epoch_time = end_time - start_time

    # Print Measures
    if epoch_idx % PRINT_EVERY == 0:
        logging.info('[Epoch %d/%d] Loss = %.4f    Cls = %.4f    Bbox = %.4f    Landmark = %.4f    Epoch Time = %.3f' %
        (epoch_idx, N_EPOCHS, total_loss.avg(), cls_loss.avg(), bbox_loss.avg(), landm_loss.avg(), epoch_time))

    if epoch_idx % SAVE_EVERY == 0:
        info_path = CKPT_DIR + NETWORK + '_info'
        epoch_model_path = CKPT_DIR + NETWORK + '_' + 'ckpt-' + str(epoch_idx).zfill(3) + '.pth'

        torch.save({
            'epoch_idx': epoch_idx
        }, info_path)

        torch.save({
            'epoch_idx': epoch_idx,
            'network': net.state_dict()
        }, epoch_model_path)

        logging.info('Model Saved')
