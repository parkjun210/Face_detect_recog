import argparse

def parse_training_args(parser):
    """Add args used for training only.
    Args:
        parser: An argparse object.
    """
    # Session parameters
    parser.add_argument('--seed', type=int, default=1337,
                        help='seed number')

    parser.add_argument('--gpu_num', type=list, action = 'store', default=[0],
                        help='GPU number to use')

    # parser.add_argument('--multi_gpu', type=str2bool, default=False,
    #                     help='Use Multi-GPU')

    #Directory parameters
    parser.add_argument('--data_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/data",
                        help='data directory')

    parser.add_argument('--checkpoint_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/checkpoints",
                        help='checkpoint directory for saving')
    
    parser.add_argument('--best_checkpoint_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/checkpoints_best",
                        help='best checkpoint directory for saving best models')

    parser.add_argument('--log_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/log",
                        help='')

    parser.add_argument('--backbone_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/checkpoints_best/Backbone_IR_SE_50_LRTRAIN_False_LRx4_checkpoint.pth",
                        help='')

    parser.add_argument('--head_dir', type=str, default="./",
                        help='')
    
    parser.add_argument('--resume_dir', type=str, default="/data/parkjun210/Jae_Detect_Recog/Code_face_recog/checkpoints_best/",
                        help='resume checkpoint directory for loading')

    # Train parameters

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Minibatch size')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')

    parser.add_argument('--LR_train', type=str2bool, default=False,
                        help='Set True if train with LR version of training data')

    parser.add_argument('--LR_scale', type=int, default=4,
                        help='Training data LR scale configuration, only effective if LR_train is True')

    # Evaluation parameters

    parser.add_argument('--LR_eval', type=str2bool, default=False,
                        help='Set True if evaluate with LR validation data')

    parser.add_argument('--SR_eval', type=str2bool, default=False,
                        help='Set True if evaluate with SR version of LR validation data')

    # parser.add_argument('--LR_oneside', type=str2bool, default=False,
    #                     help='Set True if ')

    

def parse_args():
    """Initializes a parser and reads the command line parameters.
    Raises:save_folder
        ValueError: If the parameters are incorrect.
    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser()
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """
    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))