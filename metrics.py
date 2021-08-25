import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import random
import glob

import numpy as np



def cross_batch_accuracy(per_frame_logits, labels):
    y_predicted = torch.max(per_frame_logits, dim=2)[0]
    y_predicted = y_predicted.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis=1)

    y_ground_truth = torch.max(labels, dim=2)[0]
    y_ground_truth = y_ground_truth.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    y_ground_truth = np.argmax(y_ground_truth, axis=1)

    # print(y_predicted)
    # print(y_ground_truth)
    # print('\n')

    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    # print(multilabel_confusion_matrix(y_ground_truth, y_predicted))
    # print('Cross batch accuracy %4.4f'%(accuracy_score(y_ground_truth, y_predicted)))
    return accuracy_score(y_ground_truth, y_predicted), f1_score(y_ground_truth, y_predicted, average='weighted')


def per_video_accuracy(per_frame_logits, labels):
    y_predicted = per_frame_logits.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()

    y_ground_truth = labels.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()

    nb_batches = y_predicted.shape[0]

    i = random.randint(0, nb_batches - 1)

    r_shape = y_predicted.shape

    y_predicted = y_predicted[i]
    y_ground_truth = y_ground_truth[i]

    y_predicted = np.reshape(y_predicted, (r_shape[1], r_shape[2]))
    y_ground_truth = np.reshape(y_ground_truth, (r_shape[1], r_shape[2]))

    y_predicted = np.argmax(y_predicted, axis=0)
    y_ground_truth = np.argmax(y_ground_truth, axis=0)

    from sklearn.metrics import accuracy_score
    print('Per video batch accuracy %4.4f' % (accuracy_score(y_ground_truth, y_predicted)))