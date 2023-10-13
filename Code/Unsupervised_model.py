from cgi import test
from datetime import datetime
from functools import partial
from time import sleep
from PIL import Image
from matplotlib.transforms import Transform
from numpy import imag
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser(description='Unsupervised learning model')
parser.add_argument('--knn-k', default=20, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
args = parser.parse_args('')  # running in ipynb

def split_name(address):
    n = address.rfind("\\")
    dot = address[:n].rfind("\\")
    return address[dot+1:n]

class MyDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, isLabel=True):
        """
        My Dataset for CAD with BUSI dataset
            param data_dir: str, path for the dataset
            param train: whether this is defined for training or testing
            param transform: torch.transform, data pre-processing pipeline
        """
        ### Begin your code ###
        super(MyDataset,self).__init__()
        self.train = train
        self.transform = transform
        self.isLabel = isLabel
        
        if isLabel==False:
            self.root = data_dir
        else:
            # self.root = os.path.join(data_dir,"pathological-findings")
            self.root = data_dir
            
            # filepath_csv = os.path.join(data_dir,"image-labels.csv")
            self.class_map = {"polyps":0, "ulcerative-colitis-grade-1":1, "ulcerative-colitis-grade-2":2, "ulcerative-colitis-grade-3":3}
            self.classes = list(self.class_map.keys())

            # with open(filepath_csv) as f:
            #     reader = csv.reader(f)
            #     header_row = next(reader)
                
            #     for row in reader:
            #         name = row[0]
            #         class_ = row[2]
            #         if type(class_map.get(class_))==int:
            #             self.label_map[name] = class_map[class_]
        
        self.filepaths = []
        self.data = []
        self.targets = []
        
        self.get_filename(self.root)
        self.length = len(self.filepaths)
        self.get_data()
        self.filepaths = []
        
        ### End your code ###
        
    
    def __getitem__(self, index):
        if self.isLabel == False:
            img = self.data[index]

            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)
            return im_1, im_2
        else: 
            img = self.data[index]
            img = self.transform(img)
            return img, self.targets[index]


    def __len__(self): 
        '''return the size of the dataset'''
        ### Begin your code ###
        return len(self.data)
        ### End your code ###
    
    def get_filename(self, rootDir):
        for root,dirs,files in os.walk(rootDir):
            for file in files:
                file_path = os.path.join(root, file)
                self.filepaths.append(file_path)
        
    def get_data(self):
        '''
        Load the dataset and store it in your own data structure(s)
        '''
        ### Begin your code ###
        length = len(self.filepaths)
        
        index_list = torch.arange(0,length)
        index_list = torch.randperm(index_list.size(0))
        
        for i in range(length):
            index = index_list[i].item()
            filepath = self.filepaths[index]

            if self.isLabel == False:
                image = Image.open(filepath).resize((128,128))
                self.data.append(image.copy())
                image.close()

            else:
                label_name = split_name(filepath)
                label = self.class_map[label_name]
                image = Image.open(filepath).resize((128,128))
                self.data.append(image.copy())
                image.close()
                self.targets.append(label)
        
        total_length = len(self.data)
        if self.isLabel == False:
            begin = 0
            end = total_length
        else: 
            threshold = int(total_length * 0.8)
            if self.train == True:
                begin = 0
                end = threshold
            else:
                begin = threshold
                end = total_length

        self.length = end - begin        
        self.data = self.data[begin:end]
        self.targets = self.targets[begin:end]


class Unsupervised_model():
    @torch.no_grad()
    def __init__(self, modelpath, testdir):
        self.testdir = testdir
        self.encoder = torch.load(modelpath).encoder_q

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        
        # print("unsupervised model: data load begin")
        self.memory_data = MyDataset(self.testdir, train=True, transform=self.transform,isLabel=True)
        # print("unsupervised model: data load half done")
        self.memory_loader = DataLoader(self.memory_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        # print("unsupervised model: data load done")
        
        self.classes = len(self.memory_loader.dataset.classes)
        self.class_map = self.memory_loader.dataset.classes
        self.feature_bank = []
        for data, target in self.memory_loader:
            feature = self.encoder(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            self.feature_bank.append(feature)
        
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=self.feature_bank.device)

    @torch.no_grad()
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)

        return pred_labels

    # def predict(self, image):
    #     self.encoder.eval()
    #     # image = Image.open(filepath).resize((128,128))
    #     image = transforms.ToPILImage()(image)
    #     image = self.transform(image)
    #     image = image.unsqueeze(0) 
    #     image = image.cuda(non_blocking=True)
    #     feature = self.encoder(image)
    #     pred_labels = self.knn_predict(feature, self.feature_bank, self.feature_labels, self.classes, args.knn_k, args.knn_t)

    #     return self.class_map[int(pred_labels[0][0])]

    def predict(self, filepath):
        self.encoder.eval()
        image = Image.open(filepath).resize((128,128))
        # image = transforms.ToPILImage()(image)
        image = self.transform(image)
        image = image.unsqueeze(0) 
        image = image.cuda(non_blocking=True)
        feature = self.encoder(image)
        pred_labels = self.knn_predict(feature, self.feature_bank, self.feature_labels, self.classes, args.knn_k, args.knn_t)

        return int(pred_labels[0][0])

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        #print(logits.shape, labels.shape)
        #loss = my_cross_entropy(logits, labels)
        loss, tmp1, tmp2 = my_loss_3(l_pos, l_neg, logits, self.T, labels)
        return loss, q, k, tmp1, tmp2

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k, tmp1, tmp2 = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss, tmp1, tmp2

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase(nn.Module):
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x


def my_loss_3(l_pos, l_neg, logits, T, labels, reduction="mean"):
    tmp1 = torch.exp(l_pos/T)
    exp_neg = torch.exp(l_neg/T)
    tmp2 = exp_neg.sum(1)
    softmax = tmp1 / (tmp1 + tmp2)
    
    log = -torch.log(softmax)

#    b = torch.exp(l_neg/T)
#    log = -torch.log(3./(3. + b.sum(1)))
    if reduction == "mean": return log.mean(), tmp1.mean(), tmp2.mean()
    else: return log.sum()

# modelpath3 = "model_last.pth"
# datapath3 = r"G:\Test\data\labeled-images\unsupervised"
# filepath = r"G:\Test\data\labeled-images\unsupervised\ulcerative-colitis-grade-1\0dbc215d-90f1-427e-8786-244a1d1e8956.jpg"
# # image = cv2.imread(filepath)
# model = Unsupervised_model(modelpath3,datapath3)
# # pred = model.predict(image)
# pred = model.predict(filepath)
# print(pred)