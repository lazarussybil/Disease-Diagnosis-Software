import os
import cv2
import timm
import warnings
import numpy as np
import pandas as pd
import albumentations as A

import torch
import torchdiffeq
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from operator import itemgetter

from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from typing import Tuple

__all__ = [
    'KvasirDataset',
    'KvasirVideoDataset',
    'FocalLoss',
    'WeightedFocalLoss',
    'get_resnet18',
    'get_resnet50',
    'get_swintransformer',
    'get_convnext',
    'VectorField',
    'NaturalCubicSpline',
    'CDEFunc',
    'CDEMatrixFunc',
    'NeuralCDE',
    'NeuralCDEVisual',
    'AttentionCDE',
    'HiddenStatePred',
    'tridiagonal_solve',
    'cubic_spline_with_T'
]

class KvasirDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose = None) -> None:
        super(KvasirDataset, self).__init__()
        print('Decoding images')
        self.imgs = [cv2.imread(path, 1)[:, :, ::-1] for path in tqdm(df['path'].values)]
        self.labels = df['label'].values
        self.transform = transform

    def __getitem__(self, index): 
        if self.transform is not None:
            img = self.transform(image=self.imgs[index])['image']
        return img, self.labels[index]

    def __len__(self): 
        return len(self.labels)

class KvasirVideoDataset(Dataset):
    def __init__(self, data_path: str, input_size: int = 128, strong_transform: bool = True, visualize: bool = False) -> None:
        super(KvasirVideoDataset, self).__init__()
        self.samples = []
        self.labels = []
        self.names = []
        self.visualize = visualize
        total_img = 0
        print('Decoding images')
        for ith_class, label in enumerate(os.listdir(data_path)):
            one_video = []
            file_names = [(name.split('_')[1:], name) for name in os.listdir(os.path.join(data_path, label))]
            file_names = [(int(name[0][0]), int(name[0][1][:-4]), name[1]) for name in file_names]
            file_names = sorted(file_names, key=itemgetter(0, 1))
            ith_img = file_names[0][0]
            total_img += len(file_names)

            for filename in tqdm(file_names):
                if not filename[-1].startswith(f'{label}_{ith_img}'):
                    self.samples.append(one_video)
                    if visualize:
                        self.names.append(f'{label}_{ith_img}')
                    self.labels.append(ith_class)
                    one_video = []
                    ith_img += 1
                one_video.append(cv2.imread(os.path.join(data_path, label, filename[-1]), 1)[:, :, ::-1])
            self.samples.append(one_video)
            if visualize:
                self.names.append(f'{label}_{ith_img}')
            self.labels.append(ith_class)
        
        if sum([len(video) for video in self.samples]) != total_img:
            warnings.resetwarnings()
            warnings.warn(f'Failed to load some images')
            warnings.filterwarnings('ignore')

        self.input_size = input_size
        self.strong_transform = strong_transform

    def __getitem__(self, index): 
        if self.strong_transform:
            transforms = A.Compose([
                A.RandomResizedCrop(width=self.input_size, height=self.input_size, p=1),
                A.HorizontalFlip(p=np.random.randint(0, 2)),
                A.VerticalFlip(p=np.random.randint(0, 2)),
                A.RandomRotate90(p=np.random.randint(0, 2)),
                A.Transpose(p=np.random.randint(0, 2)),
                A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            # transforms = A.Compose([
            #     A.RandomResizedCrop(width=self.input_size, height=self.input_size, p=1),
            #     A.HorizontalFlip(p=np.random.randint(0, 2)),
            #     A.VerticalFlip(p=np.random.randint(0, 2)),
            #     A.RandomRotate90(p=np.random.randint(0, 2)),
            #     A.Transpose(p=np.random.randint(0, 2)),
            #     A.RandomBrightnessContrast(p=np.random.randint(0, 2)),
            #     A.ElasticTransform(p=np.random.randint(0, 2), alpha=120, sigma=0.05 * 120, alpha_affine=0.03 * 120),
            #     A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ToTensorV2(),
            # ])
        else:
            transforms = A.Compose([
                A.Resize(width=self.input_size, height=self.input_size, p=1),
                A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        video_list = [transforms(image=img)['image'] for img in self.samples[index]]
        if self.visualize:
            return torch.stack(video_list), self.labels[index], self.names[index]
        return torch.stack(video_list), self.labels[index]

    def __len__(self): 
        return len(self.labels)

class FocalLoss(nn.Module):
    def __init__(self, focusing_param = 2):
        super(FocalLoss, self).__init__()
        self.focusing_param = focusing_param

    def forward(self, output: Tensor, target: Tensor):
        logpt = - F.cross_entropy(output, target)
        return -((1 - torch.exp(logpt)) ** self.focusing_param) * logpt

class WeightedFocalLoss(nn.Module):
    def __init__(self, w = np.array([1, 1]), focusing_param = 2, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(WeightedFocalLoss, self).__init__()
        self.w = torch.from_numpy(w).float().to(device)
        self.focusing_param = focusing_param

    def forward(self, output: Tensor, target: Tensor):
        logpt = - F.cross_entropy(output, target, weight=self.w)
        return -((1 - torch.exp(logpt)) ** self.focusing_param) * logpt

def get_resnet18(num_class: int = 2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet50(num_class: int = 2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet18_relu(num_class: int = 2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet50_relu(num_class: int = 2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_class)
    )
    return model

def get_swintransformer(num_class: int = 2):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_class)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_convnext(num_class: int = 2):
    return timm.create_model('convnext_tiny', pretrained=True, num_classes=num_class)

class VectorField(nn.Module):
    def __init__(self, dX_dt, func):
        super(VectorField, self).__init__()
        if not isinstance(func, nn.Module):
            raise ValueError("func must be a nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t: Tensor, z: Tensor) -> Tensor:
        control_gradient = self.dX_dt(t)
        vector_field = self.func(z)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

class NaturalCubicSpline:
    def __init__(self, times: Tensor, coeffs: Tuple[Tensor]):
        super(NaturalCubicSpline, self).__init__()
        (a, b, two_c, three_d) = coeffs
        self._times = times
        self._a = a
        self._b = b
        self._two_c = two_c
        self._three_d = three_d

    def _interpret_t(self, t: Tensor):
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum() - 1
        index = index.clamp(0, maxlen)
        fractional_part = t - self._times[index]
        return fractional_part, index

    def evaluate(self, t: Tensor) -> Tensor:
        fractional_part, index = self._interpret_t(t)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t: Tensor) -> Tensor:
        fractional_part, index = self._interpret_t(t)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv

class CDEFunc(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CDEFunc, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, input_size * hidden_size)

    def forward(self, z: Tensor):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        return z.view(*z.shape[:-1], self.hidden_size, self.input_size)

class CDEMatrixFunc(nn.Module):
    def __init__(self, x_input_size, h_hidden_size, n_blocks, device):
        super(CDEMatrixFunc, self).__init__()
        self.x_input_size = x_input_size
        self.h_hidden_size = h_hidden_size
        self.n_blocks = n_blocks
        self.device = device

        self.functions = [CDEFunc(x_input_size, h_hidden_size).to(device=device)] * n_blocks

    def forward(self, Z):
        Z = Z.permute(1, 0)
        Z_ = torch.rand((self.n_blocks, self.h_hidden_size, self.x_input_size)).to(device=self.device)
        for i in range(self.n_blocks):
            Z_[i] = self.functions[i](Z[i])
        return Z_

class NeuralCDE(nn.Module):
    def __init__(self, arg):
        super(NeuralCDE, self).__init__()
        self.device = arg.device
        self.adjoint = arg.adjoint
        self.hidden_size = arg.hidden_size

        self.feature_extra = torch.load(arg.backbone_path).to(arg.device)
        self.feature_extra.fc = nn.Sequential(
            nn.Linear(self.feature_extra.fc[0].in_features, 128),
            nn.ReLU(),
            nn.Linear(128, arg.img_output_size)
        )

        self.z_init = get_resnet18_relu(self.hidden_size)
        self.z_decode = nn.Linear(self.hidden_size, arg.output_size)
        
        self.func = CDEFunc(arg.img_output_size, self.hidden_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, video: Tensor):
        t = torch.range(0, len(video[0]) - 1).to(self.device)
        x_a, x_b, x_c, x_d = cubic_spline_with_T(t, self.feature_extra(video[0]))
        x_spline = NaturalCubicSpline(t, (x_a, x_b, x_c, x_d))
        z0 = self.z_init(video[0][0].unsqueeze(dim=0)).squeeze()

        odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
        vector_field = VectorField(dX_dt=x_spline.derivative, func=self.func)
        z_T = odeint(func=vector_field, y0=z0, t=t[[0, -1]], atol=1e-2, rtol=1e-2)
        return self.z_decode(z_T[1]).unsqueeze(dim=0)

class NeuralCDEVisual(nn.Module):
    def __init__(self, arg):
        super(NeuralCDEVisual, self).__init__()
        self.device = arg.device
        self.adjoint = arg.adjoint
        self.hidden_size = arg.hidden_size

        self.feature_extra = torch.load(arg.backbone_path).to(arg.device)
        self.feature_extra.fc = nn.Sequential(
            nn.Linear(self.feature_extra.fc[0].in_features, 128),
            nn.ReLU(),
            nn.Linear(128, arg.img_output_size)
        )

        self.z_init = get_resnet18_relu(self.hidden_size)
        self.z_decode = nn.Linear(self.hidden_size, arg.output_size)
        
        self.func = CDEFunc(arg.img_output_size, self.hidden_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, video: Tensor):
        with torch.no_grad():
            t = torch.range(0, len(video[0]) - 1).to(self.device)
            x_a, x_b, x_c, x_d = cubic_spline_with_T(t, self.feature_extra(video[0]))
            x_spline = NaturalCubicSpline(t, (x_a, x_b, x_c, x_d))
            z0 = self.z_init(video[0][0].unsqueeze(dim=0)).squeeze()

            odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
            vector_field = VectorField(dX_dt=x_spline.derivative, func=self.func)
            z_T = odeint(func=vector_field, y0=z0, t=torch.range(0, len(video[0]) - 1, 0.2).to(self.device), atol=1e-2, rtol=1e-2)
            return z_T, torch.argmax(self.z_decode(z_T[-1]).unsqueeze(dim=0))

class AttentionCDE(nn.Module):
    def __init__(self, arg):
        super(AttentionCDE, self).__init__()
        self.device = arg.device

        self.feature_extra = torch.load(arg.backbone_path).to(arg.device)
        self.feature_extra.fc = nn.Sequential(
            nn.Linear(self.feature_extra.fc[0].in_features, 128),
            nn.ReLU(),
            nn.Linear(128, arg.img_output_size)
        )

        self.hidden_size = arg.hidden_size
        self.n_blocks = arg.n_blocks
        
        self.g_h_encoder = nn.Parameter(0.01 * torch.randn(self.n_blocks, self.hidden_size, arg.img_output_size))
        self.h_decoder = nn.Linear(self.n_blocks * self.hidden_size, arg.output_size)

        self.hidden_func = CDEMatrixFunc(arg.img_output_size, self.hidden_size, self.n_blocks, self.device).to(self.device)
        self.z_to_att = nn.Parameter(0.01 * torch.randn(self.n_blocks, self.n_blocks, self.hidden_size))
        self.att_func = nn.Sequential(
            nn.Linear(self.n_blocks, 2 * self.n_blocks),
            nn.ReLU(),
            nn.Linear(2 * self.n_blocks, self.n_blocks),
            nn.Tanh()
        )
        self.x_to_datt = nn.Linear(arg.img_output_size, self.n_blocks)

        self.key = nn.Linear(arg.img_output_size, self.n_blocks * self.n_blocks)
        self.value = nn.Linear(arg.img_output_size, self.n_blocks * self.n_blocks)
        self.query = nn.Linear(arg.img_output_size, self.n_blocks * self.n_blocks)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, video: Tensor):
        with torch.no_grad():
            t = torch.range(0, len(video[0]) - 1).to(self.device)
            x = self.feature_extra(video[0])

        k = self.transpose_for_scores(self.key(x).view(1, x.shape[0], self.n_blocks * self.n_blocks))
        v = self.transpose_for_scores(self.value(x).view(1, x.shape[0], self.n_blocks * self.n_blocks))
        q = self.transpose_for_scores(self.query(x).view(1, x.shape[0], self.n_blocks * self.n_blocks))
        attention = torch.matmul(q, k.transpose(-1, -2)) / self.n_blocks
        attention = nn.Softmax(dim = -1)(attention)
        attention = torch.matmul(attention, v).view(x.shape[0], self.n_blocks * self.n_blocks)

        x_a, x_b, x_c, x_d = cubic_spline_with_T(t, x)

        x_spline = NaturalCubicSpline(t, (x_a, x_b, x_c, x_d))

        a0 = attention[0].view(self.n_blocks, self.n_blocks)
        z0 = torch.matmul(self.g_h_encoder, x[0]).T
        
        insert_t = torch.linspace(int(t[0]), int(t[-1]), 3 * int(t[-1] - t[0]) + 1).to(device=self.device)

        for i in insert_t:
            dz = torch.bmm(
                self.hidden_func(torch.mm(z0, a0)),
                x_spline.evaluate(i).repeat(self.n_blocks).view(self.n_blocks, -1, 1)
            ).squeeze(dim=-1).T
            z0 = z0 + dz

            da =  torch.bmm(
                self.z_to_att,
                z0.T.unsqueeze(dim=-1)
            ).squeeze()

            a0 = a0 + self.att_func(da) * self.x_to_datt(x_spline.evaluate(i))
        return self.h_decoder(z0.reshape(1, -1))

class HiddenStatePred(nn.Module):
    def __init__(self, arg):
        super(HiddenStatePred, self).__init__()
        self.device = arg.device
        self.adjoint = arg.adjoint
        self.hidden_size = arg.hidden_size

        self.z_decode = nn.Linear(self.hidden_size, arg.output_size)

    def forward(self, z: Tensor, p: bool = False):
        with torch.no_grad():
            return nn.Softmax(dim=1)(self.z_decode(z))[:, 0] if p else torch.argmax(self.z_decode(z), dim=1)

def tridiagonal_solve(b, A_upper: Tensor, A_diagonal: Tensor, A_lower: Tensor) -> Tensor:
    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)

def cubic_spline_with_T(times: Tensor, x: Tensor):
    path = x.transpose(-1, -2)
    length = path.size(-1)

    if length < 2:
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = path[..., :1]
        b = (path[..., 1:] - path[..., :1]) / (times[..., 1:] - times[..., :1])
        two_c = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
        three_d = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
    else:
        time_diffs = times[1:] - times[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        system_diagonal = torch.empty(length, dtype=path.dtype, device=path.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(path)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal)

        a = path[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a.transpose(-1, -2), b.transpose(-1, -2), two_c.transpose(-1, -2), three_d.transpose(-1, -2)

    