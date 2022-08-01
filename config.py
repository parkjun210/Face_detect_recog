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

    parser.add_argument('--inference_dir', type=str, default="sample_imgs",
                        help='inferece directory name')

    parser.add_argument('--experiment_name', type=str, default='resnet_anc2_casT_fpn3',
                        help='Experiment Name directory')

    parser.add_argument('--recog_experiment_name', type=str, default='casia_HR_LR4',
                        help='Recognition Experiment Name directory')

    # Inference parameters
    parser.add_argument('--save_img', type=str2bool, default=True,
                        help='Save validation image or not')

    parser.add_argument('--inference_save_folder', type=str, default='inference_results',
                        help='Dir to save txt results')

    parser.add_argument('--infer_imsize_same', type=str2bool, default=False,
                        help='Whether inference image size different or not')

    parser.add_argument('--library_dir', type=str, default='class',
                        help='recognition library folder directory')


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