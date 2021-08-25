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



def load_rgb_frames(root, video_file, start, nb_frames_per_shot, edge_type, label, blur_kernel, operator_kernel):
        
    r_f = random.randint(1,10)
    
    frames = []
    file_name = root + '/' + video_file + '.avi'

    vid_visible = cv2.VideoCapture(file_name)
    nf = int(vid_visible.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    at_label = label[:, start:start+nb_frames_per_shot]
    at_label = np.argmax(at_label, axis = 0)
    d = 0

    while np.count_nonzero(at_label) < nb_frames_per_shot*0.9 :
        d = d + 1
        start = random.randint(1,nf-nb_frames_per_shot+1)
        at_label = label[:, start:start+nb_frames_per_shot]
        at_label = np.argmax(at_label, axis = 0)
        if d > 50:
            break
        
    cp = 0
    
    c_operator = random.randint(1,2)
    c_blur = random.randint(1,2)
        
    rw = random.randint(5,100)
    rh = random.randint(5,100)
    
    ecart_rectangle = random.randint(5,100)
    
    red = random.randint(100,255)
    green = random.randint(100,255)
    blue = random.randint(100,255)
    
    if edge_type == 'normal':
        c_blur = 0
        c_operator = 0
        
    #print('load_rgb_frames from %s ---> [%d; %d; %d]' %(file_name, start , nb_frames_per_shot, nf))

    while cp < start and vid_visible.isOpened():
        _, img = vid_visible.read()
        cp = cp + 1
    
    ts = 0
    
    while ts < nb_frames_per_shot and vid_visible.isOpened():
        _, img = vid_visible.read()
        
        ts = ts + 1
        
        #print('load_rgb_frames from %s ---> [%d; %d; %d] ,  shape %s' %(video_file, start , nb_frames_per_shot, nf, str(img.shape)))
        
        if img is None :
            continue
        
        h, w, _ = img.shape
        
        
        if r_f > 5:
            img = cv2.flip(img, 1)
            
        
        dx = random.randint(-3,3)
        dy = random.randint(-3,3)
        
        cv2.rectangle(img, (rh + dx, rw + dy), (rh + dx + ecart_rectangle, rw + dy + ecart_rectangle), (red,green,blue), -1)
        
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
    #print('we will see here %d' %(ts))
    return np.asarray(frames, dtype=np.float32), start



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


def correct_labels(label):
    if(label == 21):
        return 0
    return label
    
def correct_labels_1(label):
    if(label == 21):
        return 0
    elif(label == 1 or label == 2):
        return 1
    elif(label == 3):
        return 2
    elif(label == 4):
        return 3
    elif(label == 5 or label == 6):
        return 4
    elif(label == 7):
        return 5
    elif(label == 8 or label == 9 or label == 10 or label ==11):
        return 6
    elif(label == 12):
        return 7
    elif(label == 13 or label == 17):
        return 8
    elif(label == 14):
        return 9
    elif(label == 15 or label == 16):
        return 10
    elif(label == 18):
        return 11
    elif(label == 19 or label == 20):
        return 12
    
def correct_labels_2(label):
    label = int(label)
    if label == 21 :
        return 0
    elif label in [1, 2, 3, 4 , 7, 14] :
        return 1
    elif label in [5, 6]:
        return 2
    elif label in [8, 9, 10, 11]:
        return 3
    elif label in [12]:
        return 4
    elif label in [13, 17]:
        return 5
    elif label in [15, 16]:
        return 6
    elif label in [18]:
        return 7
    elif label in [19, 20]:
        return 8
    return label
    
def correct_labels_3(label):
    label = int(label)
    #print(label)
    if label in [21, 18] :
        return 0
    elif label in [1, 2, 3, 4, 5, 6, 7, 13, 14, 17] :
        return 1
    elif label in [8, 9, 10, 11, 12, 15, 16, 19, 20]:
        return 2
    


def make_dataset(split_file, split, root, mode, num_classes, nb_frames_per_shot):
    
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue
        
        #print('make_dataset  %s  %s '%(root, vid))
        
        if 'K7' in vid:
            continue
        
        #if 'K1' not in vid:
        #    continue
        
        file_name = root + '/' + vid + '.avi'
        
    

        if not os.path.exists(file_name):
            continue
        
        vid_visible = cv2.VideoCapture(file_name)
        nf = int(vid_visible.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
                    label[correct_labels_2(ann[0]), fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
    
    return dataset


class CME(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, edge_type = 'sobel', num_classes = 15, nb_frames = 24, blur_kernel = 0, operator_kernel = 3):
        
        print("prepare dataset")
        self.data = make_dataset(split_file, split, root, mode,num_classes, nb_frames)
        self.nb_frames = nb_frames
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
        vid, label, dur, nf = self.data[index]
        start_f = random.randint(1,nf - self.nb_frames + 1)

        if self.mode == 'rgb':
            imgs, start_f = load_rgb_frames(self.root, vid, start_f, self.nb_frames, self.edge_type, label, self.blur_kernel, self.operator_kernel)
            #imgs = load_rgb_frames_sep(vid, start_f, nb_to, self.edge_type)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, self.nb_frames)
        label = label[:, start_f:start_f+self.nb_frames]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
