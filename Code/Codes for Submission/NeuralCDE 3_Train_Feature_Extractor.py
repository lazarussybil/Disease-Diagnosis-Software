import os
import torch
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
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from NeuralCDE_utils import *

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=UserWarning)
sklearn.set_config(print_changed_only=True)
sns.set_style("white")

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def main(arg):
    set_seed(arg.seed)

    train_transforms = A.Compose([
        A.RandomResizedCrop(width=arg.input_size, height=arg.input_size, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit = 3),

        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=0.05 * 120, alpha_affine=0.03 * 120),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8),
        A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    img_df = {'path': [], 'label': []}
    for i_class, label in enumerate(arg.labels):
        file_dir = os.path.join(arg.raw_img_dir, label)
        files = os.listdir(file_dir)
        img_df['path'] += [os.path.join(file_dir, file_name) for file_name in files]
        img_df['label'] += [i_class] * len(files)
    img_df = sklearn.utils.shuffle(pd.DataFrame(img_df)).reset_index(drop=True)

    print('Loading Training Set')
    train_datasets = KvasirDataset(img_df, train_transforms)
    train_loader = DataLoader(train_datasets, batch_size=arg.batch_size, shuffle=True, num_workers=arg.workers)

    print('Loading Model')
    if arg.model == 'resnet18':
        model = get_resnet18(arg.output_size).to(arg.device)
    elif arg.model == 'swin':
        model = get_swintransformer(arg.output_size).to(arg.device)
    else:
        model = get_resnet50(arg.output_size).to(arg.device)

    if arg.loss == 'FocalLoss':
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=arg.w.to(arg.device))

    optimizer = optim.Adam(model.parameters(), lr = arg.lr, weight_decay = arg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arg.epochs, eta_min = 0, last_epoch = -1)

    h = hl.History()
    c = hl.Canvas()

    train_loss_all = []
    train_acc_all = []
    train_auc_all = []

    for epoch in range(arg.epochs):
        print('Epoch {}/{}'.format(epoch + 1, arg.epochs))
        ys, pre_scores, pre_labs = [], [], []
        epoch_loss = 0

        for (b_x, b_y) in tqdm(train_loader):
            ys.append(b_y)
            b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)

            output = model(b_x)
            loss = criterion(output, b_y)

            epoch_loss += loss.item() * b_x.size(0)
            pre_labs.append(torch.argmax(output, 1).cpu().detach())
            pre_scores.append(nn.Softmax(dim=1)(output).cpu().detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        ys = torch.cat([torch.stack(ys[:-1]).view(-1, ), ys[-1]])
        pre_scores = torch.cat([torch.stack(pre_scores[:-1]).view(-1, arg.output_size), pre_scores[-1]]).numpy()
        if arg.output_size == 2:
            pre_scores = np.max(pre_scores, axis=1)
        pre_labs = torch.cat([torch.stack(pre_labs[:-1]).view(-1, ), pre_labs[-1]])

        train_loss_all.append(float(epoch_loss) / len(ys))
        train_acc_all.append(int(torch.sum(pre_labs == ys)) / len(ys))
        train_auc_all.append(roc_auc_score(ys.numpy(), pre_scores, multi_class='ovo'))
        print('{} Train Loss: {:.4f} Train Acc: {:.4f} Auc Score: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1], train_auc_all[-1]))

        h.log(
            (epoch),
            train_loss = train_loss_all[-1],
            train_acc = train_acc_all[-1],
            train_auc = train_auc_all[-1],
        )
        with c:
            c.draw_plot(h['train_loss'])
            c.draw_plot(h['train_acc'])
            c.draw_plot(h['train_auc'])

    os.makedirs(arg.figure_save_dir, exist_ok=True)
    os.makedirs(arg.model_dir, exist_ok=True)

    h.save(os.path.join(arg.model_dir, 'log_file.pkl'))
    torch.save(model, os.path.join(arg.model_dir, 'model.pkl'))

    plt.savefig(os.path.join(arg.figure_save_dir, f'Training Process.png'))
    plt.savefig(os.path.join(arg.figure_save_dir, f'Training Process.eps'), format='eps')

if __name__ == '__main__':
    data_type = 'Co_Po'
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--raw_img_dir', type = str, default = f'Data\\Raw\\Images_{data_type}')
    parser.add_argument('--model_dir', type = str, default = f'Model\\Feature_Extractor_{data_type}')
    parser.add_argument('--figure_save_dir', type = str, default = f'Figures\\Feature_Extractor_{data_type}')
    parser.add_argument('--model', type = str, default = 'resnet18')
    parser.add_argument('--input_size', type = int, default = 128)
    parser.add_argument('--output_size', type = int, default = 6 if data_type == 'Co_Po' else 2)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--workers', type = int, default = 0)
    parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--loss', default = 'FocalLoss' if data_type == 'Co_Po' else 'CE')
    parser.add_argument('--w', type = Tensor, default = torch.tensor([1, 2], dtype=torch.float))
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--weight_decay', type = float, default = 0.01)
    parser.add_argument('--labels', type = list, default = os.listdir(f'Data\\Raw\\Images_{data_type}'))

    args = parser.parse_args([])
    main(arg=args)