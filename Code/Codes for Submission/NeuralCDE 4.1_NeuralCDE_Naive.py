import os
import torch
import random
import sklearn
import warnings
import argparse

import numpy as np
import seaborn as sns
import hiddenlayer as hl
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

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

    print('Loading Training Set')
    train_datasets = KvasirVideoDataset(os.path.join(arg.video_img_dir, 'Train'), arg.img_input_size, True)
    train_loader = DataLoader(train_datasets, batch_size=arg.batch_size, shuffle=True, num_workers=arg.workers)
    print('Loading Testing Set')
    test_datasets = KvasirVideoDataset(os.path.join(arg.video_img_dir, 'Test'), arg.img_input_size, False)
    test_loader = DataLoader(test_datasets, batch_size=arg.batch_size, shuffle=False, num_workers=arg.workers)

    print('Loading Model')
    model = NeuralCDE(args).to(device=arg.device)

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
    test_loss_all = []
    test_acc_all = []
    test_auc_all = []

    for epoch in range(arg.epochs):
        print('Epoch {}/{}'.format(epoch + 1, arg.epochs))

        model.train()
        ys, pre_scores, pre_labs = [], [], []
        epoch_loss = 0

        for (b_x, b_y) in tqdm(train_loader):
            ys.append(int(b_y))
            b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)
            output = model(b_x)
            loss = criterion(output, b_y)

            output = output.squeeze()
            epoch_loss += loss.item()
            pre_labs.append(int(torch.argmax(output)))
            pre_scores.append(np.max(nn.Softmax(dim=0)(output).cpu().detach().numpy()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        ys = np.array(ys)
        pre_labs = np.array(pre_labs)
        pre_scores = np.array(pre_scores)

        train_loss_all.append(float(epoch_loss) / len(ys))
        train_acc_all.append(int(np.sum(pre_labs == ys)) / len(ys))
        train_auc_all.append(roc_auc_score(ys, pre_scores, multi_class='ovo'))
        print('{} Train Loss: {:.4f} Train Acc: {:.4f} Auc Score: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1], train_auc_all[-1]))

        model.eval()
        ys, pre_scores, pre_labs = [], [], []
        epoch_loss = 0
        with torch.no_grad():
            for (b_x, b_y) in tqdm(test_loader):
                ys.append(int(b_y))
                b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)
                output = model(b_x)
                loss = criterion(output, b_y)

                output = output.squeeze()
                epoch_loss += loss.item()
                pre_labs.append(int(torch.argmax(output)))
                pre_scores.append(np.max(nn.Softmax(dim=0)(output).cpu().detach().numpy()))

            ys = np.array(ys)
            pre_labs = np.array(pre_labs)
            pre_scores = np.array(pre_scores)

            test_loss_all.append(float(epoch_loss) / len(ys))
            test_acc_all.append(int(np.sum(pre_labs == ys)) / len(ys))
            test_auc_all.append(roc_auc_score(ys, pre_scores, multi_class='ovo'))
            print('{} Test Loss: {:.4f} Test Acc: {:.4f} Auc Score: {:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1], test_auc_all[-1]))

        h.log(
            (epoch),
            train_loss = train_loss_all[-1],
            train_acc = train_acc_all[-1],
            train_auc = train_auc_all[-1],
            test_loss = test_loss_all[-1],
            test_acc = test_acc_all[-1],
            test_auc = test_auc_all[-1],
        )
        with c:
            c.draw_plot([h['train_loss'], h['test_loss']])
            c.draw_plot([h['train_acc'], h['test_acc']])
            c.draw_plot([h['train_auc'], h['test_auc']])

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
    parser.add_argument('--video_img_dir', type = str, default = f'Data\\Video_Images_{data_type}')
    parser.add_argument('--model_dir', type = str, default = f'Model\\NeuralCDE_Naive_{data_type}')
    parser.add_argument('--figure_save_dir', type = str, default = f'Figures\\NeuralCDE_Naive_{data_type}')
    
    parser.add_argument('--backbone_path', type = str, default = f'Model\Feature_Extractor_{data_type}\model.pkl')
    parser.add_argument('--adjoint', type = bool, default = True)
    parser.add_argument('--img_input_size', type = int, default = 128)
    parser.add_argument('--img_output_size', type = int, default = 32)
    parser.add_argument('--hidden_size', type = int, default = 16)
    parser.add_argument('--output_size', type = int, default = 2)

    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--workers', type = int, default = 0)
    parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--loss', default = 'CE')
    parser.add_argument('--w', type = Tensor, default = torch.tensor([1, 5], dtype=torch.float) if data_type == 'Co_Po' else torch.tensor([1, 1], dtype=torch.float))
    parser.add_argument('--lr', type = float, default = 0.00003)
    parser.add_argument('--weight_decay', type = float, default = 0.01)


    args = parser.parse_args([])
    main(arg=args)