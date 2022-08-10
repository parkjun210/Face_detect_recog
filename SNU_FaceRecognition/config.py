import argparse

def parse_training_args(parser):
    """Add args used for training only.
    Args:
        parser: An argparse object.
    """
    # Session parameters
    parser.add_argument('--seed', type=int, default=1337,
                        help='seed number')

    parser.add_argument('--gpu_num', type=int, default=0,
                        help='gpu number')
                        
    # parser.add_argument('--multi_gpu', type=str2bool, default=False,
    #                     help='Use Multi-GPU')

    #Directory parameters
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='data directory')

    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints",
                        help='checkpoint directory for saving')
    
    parser.add_argument('--best_checkpoint_dir', type=str, default="./checkpoints_best",
                        help='best checkpoint directory for saving best models')

    parser.add_argument('--sr_checkpoint_dir', type=str, default="",
                        help='checkpoint for sr module, must be specified if SR_EVAL is true')    

    parser.add_argument('--log_dir', type=str, default="./log",
                        help='')
    
    parser.add_argument('--load_dir', type=str, default="./checkpoints_best/",
                        help='load checkpoint directory for validation') #for validation

    parser.add_argument('--resume_dir', type=str, default="",
                        help='resume checkpoint directory for training') #for resume training

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