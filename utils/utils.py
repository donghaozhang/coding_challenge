import os
import random

import numpy as np
import torch

cifar10 = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Seed fixed to {}.".format(seed))


def mkdir_if_missing(directory):
    '''
    The function mkdir_if_missing checks whether a directory exists or not.
    If the directory does not exist, the function creates the directory using the os.makedirs method.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_project_dirs(save_dir):
    '''
    Create the required directories
    '''
    mkdir_if_missing(save_dir)
    mkdir_if_missing(os.path.join(save_dir, "checkpoints"))
    mkdir_if_missing(os.path.join(save_dir, "false_positives"))
    mkdir_if_missing(os.path.join(save_dir, "false_positives_feature"))
    for item in cifar10:
        mkdir_if_missing(os.path.join(save_dir, "false_positives", item))
        mkdir_if_missing(os.path.join(save_dir, "false_positives_feature", item))
