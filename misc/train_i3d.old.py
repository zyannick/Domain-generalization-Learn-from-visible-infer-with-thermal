import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-sobel_feat', type=str)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-num_classes', type=int)

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


import numpy as np

from pytorch_i3d import InceptionI3d

#from charades_dataset import Charades as Dataset
from cme_dataset import CME as Dataset



def my_softmax(x):     
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def my_softmax_multi_dim(x):     
    res = np.zeros(x.shape)
    """Compute softmax values for each sets of scores in x.""" 
    for i in range(x.shape[0]):
        res[i] = my_softmax(x[i])
    return res

def cross_batch_accuracy(per_frame_logits, labels):
    
    y_predicted = torch.max(per_frame_logits, dim=2)[0]
    y_predicted = y_predicted.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis = 1)
    
    y_ground_truth = torch.max(labels, dim=2)[0]
    y_ground_truth = y_ground_truth.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    y_ground_truth = np.argmax(y_ground_truth, axis = 1)
    
    print(y_predicted)
    print(y_ground_truth)
    
    
    #from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import accuracy_score
    #print(multilabel_confusion_matrix(y_ground_truth, y_predicted)) 
    print('Cross batch accuracy %4.4f'%(accuracy_score(y_ground_truth, y_predicted)))
    
def per_video_accuracy(per_frame_logits, labels):
    
    y_predicted = per_frame_logits.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    
    y_ground_truth = labels.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    
    nb_batches = y_predicted.shape[0]
    
    i = random.randint(0,nb_batches - 1)
    
    r_shape = y_predicted.shape
    
    y_predicted = y_predicted[i]
    y_ground_truth = y_ground_truth[i]
    
    y_predicted = np.reshape(y_predicted, (r_shape[1], r_shape[2]))
    y_ground_truth = np.reshape(y_ground_truth, (r_shape[1], r_shape[2]))
    
    y_predicted = np.argmax(y_predicted, axis = 0)
    y_ground_truth = np.argmax(y_ground_truth, axis = 0)
        
    from sklearn.metrics import accuracy_score
    print('Per video batch accuracy %4.4f'%(accuracy_score(y_ground_truth, y_predicted)))


def run(init_lr=0.1, max_steps=64e3, mode='rgb', 
        root='../charades/Charades_v1_rgb', 
        train_split='./cme.json', 
        sobel_feat = True,
        batch_size=8*5, num_classes = 15,
        save_model=''):
    #print("run")
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    
    print(root)
    print(train_split)
    print("go dataset")
    dataset = Dataset(train_split, 'training', root, mode, train_transforms, sobel_feat, num_classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms, sobel_feat, num_classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
                print("training")
            else:
                print("validating")
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                                

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]
                
                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], 
                                                              torch.max(labels, dim=2)[0])
                   
                
                tot_cls_loss += cls_loss.data[0]

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()
                
                print('voyons voir %d'  %(num_iter))
                
                
                if phase == 'val':
                    #cross_batch_accuracy(per_frame_logits, labels)
                    #per_video_accuracy(per_frame_logits, labels)
                    print('Validate here')

                if num_iter == num_steps_per_update and phase == 'train':
                    print('voyons voir %d'  %(steps))
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        line_train = '{} Loc Loss:_{:.4f}_Cls Loss:_{:.4f}_Tot Loss:_{:.4f}\n'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10)
                        print (line_train)
                        with open('./logs_train.txt', 'a') as f :
                            f.write(line_train)
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                line_val = '{} Loc Loss:_{:.4f}_Cls Loss:_{:.4f}_Tot Loss:_{:.4f}\n'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter)
                print (line_val)
                with open('./logs_val.txt', 'a') as f :
                    f.write(line_val)
    


if __name__ == '__main__':
    # need to add argparse
    type_ = args.sobel_feat == 'yes'
    run(mode=args.mode, root=args.root, 
        save_model=args.save_model, 
        sobel_feat = type_, batch_size = args.batch_size, num_classes = args.num_classes)
