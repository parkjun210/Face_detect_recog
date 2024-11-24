import argparse

def parse_training_args(parser):
    """Add args used for training only.
    Args:
        parser: An argparse object.
    """
    # Design paramters

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU number to use')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Minibatch size')

    parser.add_argument('--epochs', type=int, default=8000,
                        help='Number of epochs to train')

    parser.add_argument('--inference_dir', type=str, default="sample_widerface/images/",
                        help='widerface, etc')

    parser.add_argument('--experiment_name', type=str, default='tina_iou_anc3_casT_fpn3',
                        help='Experiment Name directory')
    
    parser.add_argument('--model_version', type=str, default='tina',
                        help='Model version whether retina or tina')
    
    parser.add_argument('--num_anchor', type=int, default=3,
                        help='Number of anchor - should be 2 or 3')
    
    parser.add_argument('--landmark_loss', type=str2bool, default=False,
                        help='whether to use landmark loss or not')

    parser.add_argument('--dataset', type=str, default='widerface/',
                        help='Dir to dataset')

    # Inference parameters
    parser.add_argument('--save_img', type=str2bool, default=False,
                        help='Save validation image or not')

    parser.add_argument('--inference_save_folder', type=str, default='inference_results/',
                        help='Dir to save txt results')

    parser.add_argument('--infer_imsize_same', type=str2bool, default=False,
                        help='Whether inference image size different or not')

    parser.add_argument('--mask', type=str2bool, default=False,
                        help='Detecting Mask or Not')

    parser.add_argument('--save_mask', type=str2bool, default=False,
                        help='Detecting Mask or Not')


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