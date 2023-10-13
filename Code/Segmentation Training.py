import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from Segmentation_models.UNet_3Plus import UNet_3Plus_DeepSup_CGM

class MyDataset(Dataset):
    def __init__(self, data_dir, train, transform=None):
        self.data_dir = data_dir
        self.data_path = np.array(os.listdir(data_dir + '/images'))
        #self.label_path = np.array(os.listdir(data_dir + '/masks'))
        #self.data_dir = data_dir
        #self.data_path = data_path
        self.transform = transform
        if train:
            idx, _ = train_test_split(list(range(len(self.data_path))), test_size=0.2, random_state=1)
        else:
            _, idx = train_test_split(list(range(len(self.data_path))), test_size=0.2, random_state=1)
        self.data_path = self.data_path[idx]
    
    def __getitem__(self, index): 
        ### Begin your code ###
        file_name = self.data_path[index]
        path_image = self.data_dir + "/images/" + file_name
        path_label = self.data_dir + "/masks/" + file_name
        image = Image.open(path_image)
        label = Image.open(path_label)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        
        # Data Augmentation
        augType = random.choice([0, 1, 2])
        if augType != 0:
            image = self.augment(image, augType)
            label = self.augment(label, augType)
        return image, label
        ### End your code ###

    def __len__(self): 
        '''return the size of the dataset'''
        ### Begin your code ###
        return(len(self.data_path))
        ### End your code ###
    
    def augment(self, image, augType):
        if augType == 1:
            aug_transform = transforms.Compose([transforms.RandomRotation(degrees=45)])
        if augType == 2:
            aug_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        data = aug_transform(image)
        return data

def BCE_loss(pred,label):
    bce_loss = nn.BCELoss(size_average=True)
    bce_out = bce_loss(pred, label)
    #print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    nume = 2 * torch.sum(y_true * y_pred)
    deno = torch.sum(torch.square(y_pred) + torch.square(y_true))
    return 1-torch.mean(nume/(deno + epsilon))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # hyper-parameters
    seed = 1
    MAX_EPOCH = 10
    LR = 0.001
    weight_decay = 1e-3
    data_dir = 'segmented-images'
    
    set_seed(seed)
    print('random seed:', seed)
    
    ###### Data Loader ######
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_data = MyDataset(data_dir=data_dir, train=True, transform=transform)
    valid_data = MyDataset(data_dir=data_dir, train=False, transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1)

    net = UNet_3Plus_DeepSup_CGM()
    #net = UNet()
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=0, last_epoch=-1)
    #criterion = nn.BCELoss()
    #criterion = 
    print('\nTraining start!\n')
    start = time.time()

    for epoch in range(1, MAX_EPOCH + 1):
        # Training
        loss_trn = 0.; dice_trn = 0.; iter_num = 0
        pred_trn = []; label_trn = []
        net.train()
        for image, label in tqdm(train_loader):
            iter_num += 1
            
            image = image.to(device)
            label = label[:,0,:,:].reshape([1] + list(label[:,0,:,:].shape))
            label[label<0.5] = 0; label[label>=0.5] = 1
            label = label.to(device)
            pred = net(image)
            
            if epoch == MAX_EPOCH:
                pred_trn.append(pred); label_trn.append(label)

            optimizer.zero_grad()
            #loss_iou = IOU_loss(pred, label)
            #loss = loss_iou #+ msssim(pred, label) + BCE_loss(pred, label)
            loss = BCE_loss(pred, label)
            loss.backward()
            optimizer.step()
            loss_trn += loss.item()

            pred[pred<0.5] = 0; pred[pred>=0.5] = 1
            dice_trn += torch.sum(pred == label) / (256*256)
        
        # print log
        print("Training: Epoch[{:0>3}/{:0>3}], Loss: {:.4f} Dice:{:.2%}".format(
                epoch, MAX_EPOCH, loss_trn / iter_num, dice_trn / iter_num))

        # Validating
        loss_val = 0.; dice_val = 0.; iter_num = 0
        pred_val = []; label_val = []
        net.eval()
        with torch.no_grad():
            for image, label in tqdm(valid_loader):
                iter_num += 1
                
                image = image.to(device)
                label = label[:,0,:,:].reshape([1] + list(label[:,0,:,:].shape))
                label[label<0.5] = 0; label[label>=0.5] = 1
                label = label.to(device)
                pred = net(image)
                
                if epoch == MAX_EPOCH:
                    pred_val.append(pred); label_val.append(label)
                
                #loss_iou = IOU_loss(pred, label)
                #loss = loss_iou #+ msssim(pred, label) + BCE_loss(pred + 1e-6, label)
                #loss_iou = IOU_loss(pred, label)
                #loss = loss_iou #+ msssim(pred, label) + BCE_loss(pred, label)
                loss = BCE_loss(pred, label)
                loss_val += loss.item()

                pred[pred<0.5] = 0; pred[pred>=0.5] = 1
                dice_val += torch.sum(pred == label) / (256*256)
        
            print("Valid: Epoch[{:0>3}/{:0>3}], Loss: {:.4f} Dice:{:.2%}\n".format(
                epoch, MAX_EPOCH, loss_val / iter_num, dice_val / iter_num))

    torch.save(net, 'unet3plus_v1.pth')
    #torch.save(net, 'unet_v1.pth')

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))

