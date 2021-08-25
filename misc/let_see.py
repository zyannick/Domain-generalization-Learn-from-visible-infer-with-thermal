import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-edge_type', type=str)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-num_classes', type=int)
parser.add_argument('-num_frames', type=int)
parser.add_argument('-blur_kernel', type=int)
parser.add_argument('-operator_kernel', type=int)
parser.add_argument('-dataset_name', type=str)
parser.add_argument('-train_split', type=str)
parser.add_argument('-data_aug', type=int)



args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
#from MulticoreTSNE import MulticoreTSNE as TSNE

import random
import glob

import numpy as np

from pytorch_i3d import InceptionI3d

from cme_dataset import CME as Dataset_cme
from cmd_data_helpers import Source_Datasets as Dataset_cme_sep
from ucf_101_dataset import UCF as Dataset_ucf



print('here')