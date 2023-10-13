import os
import cv2
import timm
import random
import sklearn
import warnings
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import hiddenlayer as hl
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as f

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(): 
    
    valid_transform = A.Compose([
        A.Resize(width=128, height=128, p=1),
        A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    data_dir='unlabel'
    paths=[os.path.join(data_dir, i ) for i in os.listdir(data_dir)]
    
    print('loading model')
    net = torch.load('Model/Resnet18/fold_3_model.pkl')  
    net.eval()
    net.to(device)


    print('start iteration')
    with torch.no_grad():
        for imag_path in paths:
            img=cv2.cvtColor(cv2.imread(imag_path), cv2.COLOR_BGR2RGB) 
            data= valid_transform(image=img)['image'].reshape([1,3,128,128])
            print(data.shape)
            inputs= data
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)      
            probability=f.softmax(outputs[0])
            print(probability)
            if max(probability)>0.8:
                print('This image is {}'.format(predicted))
                

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if device == 'cuda':
        device_name = torch.cuda.get_device_name()

        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # hyper-parameters
    seed = 1
    BATCH_SIZE = 32
    data_dir = "unlabel"
    indexes=np.arange(len([os.path.join(data_dir, i ) for i in os.listdir(data_dir)]))
    set_seed(seed)

    main()
    