import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import glob
import pandas as pd

import cv2

def choose_24_of_75(label):
    new_label = np.zeros([24])
    j = 0
    for i in label.shape[0]:
        if i % 3 == 0:
            new_label[j] = label[i]
    return

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def laplace_dev(img, blur_kernel, operator_kernel):
    

    ddepth = cv2.CV_16S
    
    if blur_kernel > 0:
        src = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    else:
        src = img
     
    
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    dst = cv2.Laplacian(gray, ddepth, ksize=operator_kernel)
    # [laplacian]
    # [convert]
    # converting back to uint8
    grad = cv2.convertScaleAbs(dst)
    
    grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    
    return grad


def transform_img_sobel(img, blur_kernel, operator_kernel):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    if blur_kernel > 0:
        src = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    else:
        src = img
    
    #print(src.shape)


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=operator_kernel, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=operator_kernel, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    return grad
   
def count_number(tab, val):
    nb = 0
    for v in tab:
        if v == val:
            nb = nb + 1
    return nb     

def count_number_of_frame(file_name):
    count_frame = 0
    
    vid_visible = cv2.VideoCapture(file_name)
    while True:
        ret, frame = vid_visible.read()
        if ret:
            count_frame = count_frame + 1
        else:
            break
    
    return count_frame

def load_rgb_frames(root, video_file, start, nb_frames_per_shot, edge_type, number_of_frames, action_name, blur_kernel, operator_kernel):
        
    r_f = random.randint(1,10)
    
    frames = []
    file_name = root + '/' + action_name + '/' + video_file
    

    vid_visible = cv2.VideoCapture(file_name)

    
    cp = 0
    
    c_operator = random.randint(1,2)
    c_blur = random.randint(1,2)
        
    rw = random.randint(5,100)
    rh = random.randint(5,100)
    
    ecart_rectangle = random.randint(5,100)
    
    red = random.randint(100,255)
    green = random.randint(100,255)
    blue = random.randint(100,255)
    
    #print(edge_type)
    
    if edge_type == 'normal':
        c_blur = 0
        c_operator = 0
        
    #print('magix nb load_rgb_frames from %s ---> [%d; %d; %d]' %(file_name, start , nb_frames_per_shot, number_of_frames))

    #while cp < start and vid_visible.isOpened():
    #    _, img = vid_visible.read()
        
    
    ts = 0
    
    tt = 0
    
    while ts < nb_frames_per_shot and vid_visible.isOpened():
        ret, img = vid_visible.read()
        tt = tt + 1
        
        if not ret :
            #print('why none ? ret value %s ---- ts  %d ---- cp  %d --- nf  %d --- tt %d' %(str(ret), ts, cp, number_of_frames, tt))
            break
        
        #print('cp value %d, start   %d,  nb_fr %d' %(cp, ))
        
        if cp >= start and cp < start + nb_frames_per_shot: 

            if img is None :
                #print('why none ? ret value %s ---- ts  %d ---- cp  %d --- nf  %d' %(str(ret), ts, cp, number_of_frames))
                cp = cp + 1
                continue
            
            ts = ts + 1
            
            #print('load_rgb_frames from %s ---> [%d; %d; %d] ,  shape %s' %(video_file, start , nb_frames_per_shot, number_of_frames , str(img.shape)))
            
            h, w, _ = img.shape
            
            
            if r_f > 5:
                img = cv2.flip(img, 1)
                
            
            dx = random.randint(-3,3)
            dy = random.randint(-3,3)
            
            #cv2.rectangle(img, (rh + dx, rw + dy), (rh + dx + ecart_rectangle, rw + dy + ecart_rectangle), (red,green,blue), -1)
            
            if c_blur == 1:
                blur_kernel = 0
                operator_kernel = 3  
            elif c_blur == 2:
                blur_kernel = 3
                operator_kernel = 5   
            
            if c_operator == 1 :
                img = transform_img_sobel(img, blur_kernel, operator_kernel)
            elif c_operator == 2:
                img = laplace_dev(img, blur_kernel, operator_kernel)
            else:
                img = img
                
            img = img[:, :, [2, 1, 0]]
                
            
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            
            frames.append(img)
        
        cp = cp + 1
    #print('we will see here %d and again %d' %(ts, len(frames)))
    return np.asarray(frames, dtype=np.float32)

'''
def load_rgb_frames_old(video_file, start, nb_frames, sobel_feat):
    
        
    frames = []

    vid_visible = cv2.VideoCapture(video_file)
    nf = int(vid_visible.get(cv2.CAP_PROP_FRAME_COUNT))
    

    cp = 0
    
    while cp < start and vid_visible.isOpened():
        _, img = vid_visible.read()
        cp = cp + 1
    
    ts = 0
    
    while ts < nb_frames and vid_visible.isOpened():
        _, img = vid_visible.read()

        
        if img is None :
            continue
        
        ts = ts + 1
        img = img[:, :, [2, 1, 0]]
        
        #print(img.shape)
        #if sobel_feat :
            #print("sobel here")
            #img = transform_img_sobel(img)
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        
        frames.append(img)
    #print('we will see here %d' %(ts))
    return np.asarray(frames, dtype=np.float32)'''



def load_flow_frames(image_dir, vid, start, nb_frames):
  frames = []
  for i in range(start, start+nb_frames):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def get_labels_dict():
    dict_label = {}
    f = open('../ucf101/classInd.txt', 'r')
    line = f.readline()
    while line:
        line = (line.split('\n'))[0]
        val = line.split(' ')
        label_index = int(val[0])
        label_name = str(val[1])
        dict_label[label_name] = label_index
        line = f.readline()
    
    return dict_label

'''def make_dataset_old( split , mode, num_frame_min, dict_label,num_classes=157):
    
    
    
    if split == 'training':
        f = open('../ucf101/trainlist01.txt', 'r')
        #list_files = pd.read_csv('../ucf101/trainlist01.txt', sep=" ", header=None)
    else:
        f = open('../ucf101/testlist01.txt', 'r')
        #list_files = pd.read_csv('../ucf101/testlist01.txt', sep=" ", header=None)
        
        
    line = f.readline()
        
    dataset = []
    i = 0
    list_actions = []
    while line:
        
        temp_split  = line.split(' ')
        file_name = temp_split[0]
        file_name = '../ucf101/' + (file_name.split('\n'))[0]
        
        print(file_name)
        print(os.path.getsize(file_name) / (1024.0))
        
        temp_label = file_name.split('/')
        
        #temp_label = (temp_split[0]).split('_')
        action_id = dict_label[ str(temp_label[2]) ]
        
        if action_id not in list_actions:
            list_actions.append(action_id)
        
        video = cv2.VideoCapture(file_name)
        nf = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            nf = nf//2
            
        if nf < num_frame_min + 2:
            print('probleme continue %d' %(nf))
            line = f.readline()
            continue

        label = np.zeros((num_classes,nf), np.float32)
        
        dur = nf/25
        
        print('action_id %d    duration %4.4f    number of frame %d' %(action_id, dur, num_frame_min))

        for fr in range(0,nf,1):
            label[action_id-1, fr] = 1 # binary classification
        dataset.append((file_name, label, dur, nf))
        i += 1
        
        line = f.readline()
    
    
    print(list_actions)
    return dataset'''


def make_dataset(split_file, split, root, mode, num_classes, nb_frames_per_shot):
    
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue
        
        
        action_name = data[vid]['action_name']
        
        file_name = root + '/' + action_name + '/' + vid 
        
        if not os.path.exists(file_name):
            continue
        
        #nb_frames_per_shot
        
        vid_visible = cv2.VideoCapture(file_name)
        nf = data[vid]['nb_frames']
        
        #print(nf)
        num_frames = nf
        
        
        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < nb_frames_per_shot + 2:
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0]-1, fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames, action_name))
        i += 1
    
    return dataset


class UCF(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, edge_type = 'sobel', num_classes = 15, nb_frames_per_shot = 24, blur_kernel = 0, operator_kernel = 3):
        
        print("prepare dataset")
        self.label_dict = get_labels_dict()
        self.data = make_dataset(split_file, split, root, mode,num_classes, nb_frames_per_shot)
        self.nb_frames_per_shot = nb_frames_per_shot
        self.blur_kernel = blur_kernel
        self.operator_kernel = operator_kernel
        #self.data = make_dataset_sep(root, mode, num_classes)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.edge_type = edge_type
        print("dataset prepared")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        #print("here")
        vid, label, dur, number_of_frames, action_name = self.data[index]
        
        start_f = random.randint(1,number_of_frames - (self.nb_frames_per_shot + 1))
        #print("on verra ici %d et %d" %(number_of_frames, start_f))

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, self.nb_frames_per_shot, 
                                   self.edge_type, number_of_frames, action_name, self.blur_kernel, self.operator_kernel)
            #print('Let see here %s  %s\n' %(str(imgs.shape), vid))
        else:
            imgs = load_flow_frames(self.root, vid, start_f, self.nb_frames_per_shot)
        label = label[:, start_f:start_f+self.nb_frames_per_shot]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
