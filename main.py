import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models
import wandb

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m','pems-bay-2022flow','pems-bay-2022speed'])
    # parser.add_argument('--dataset', type=str, default='pems-bay', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--n_components', type=int, default=3, help='number of components for covariance matrix')
    parser.add_argument('--rho', type=float, default=1, help='rho for loss function')
    # parser.add_argument('--mix_mean', type=str, default="False", help='mix mean for loss function')
    parser.add_argument('--mix_mean', type=str, default="True", help='mix mean for loss function')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    if args.n_components == 0:
        args.rho = 0
        args.n_components = 1
    
    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()
        self.min = data[data>0].min()
        self.max = data.max()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_rate = 0.1
    test_rate = 0.2

    len_val = int(math.floor(data_col * val_rate))
    len_test = int(math.floor(data_col * test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    # zscore = preprocessing.StandardScaler()
    # zscore = StandardScaler(train.mean(), train.std())
    zscore = StandardScaler(train)
    train = zscore.transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


import torch.optim as optim
from model import *
import math

import torch.nn as nn
import torch.distributions as Dist

# from tensorboardX import SummaryWriter
import wandb
import properscoring as ps
import numpy as np

def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


class FixedResCov(nn.Module):
    def __init__(self, n_components, n_vars, num_pred, rho=0.5, diag=False, trainL=True, device='cpu'):
        super(FixedResCov, self).__init__()
        self.dim_L_1 = (n_components, n_vars, n_vars)
        self.dim_L_2 = (n_components, num_pred, num_pred)

        init_L1 = torch.diag_embed(torch.rand(*self.dim_L_1[:2])) * 0.01
        init_L2 = torch.diag_embed(torch.rand(*self.dim_L_2[:2])) * 0.01

        self._L1 = nn.Parameter(init_L1.detach(), requires_grad=trainL)
        self._L2 = nn.Parameter(init_L2.detach(), requires_grad=trainL)
        self.diag = diag
        self.rho = rho

    @property
    def L1(self):
        # Ltemp = torch.tanh(self._L) * self.rho
        Ltemp = self._L1
        # Ltemp = Ltemp / Ltemp[:, 0, 0].unsqueeze(-1).unsqueeze(-1)

        if self.diag:
            return torch.diag_embed(torch.diagonal(Ltemp, dim1=1, dim2=2))
        else:
            return torch.tril(Ltemp)

    @property
    def L2(self):
        Ltemp = self._L2
        # Ltemp = Ltemp / Ltemp[:, 0, 0].unsqueeze(-1).unsqueeze(-1)

        if self.diag:
            return torch.diag_embed(torch.diagonal(Ltemp, dim1=1, dim2=2))
        else:
            return torch.tril(Ltemp)


class CholeskyResHead(nn.Module):
    def __init__(self, n_vars, n_components, pred_len=12, rho=0.1, loss="mse"):
        super(CholeskyResHead, self).__init__()
        self.n_vars = n_vars
        self.n_rank = n_components
        self.pred_len = pred_len

        self.rho = rho
        self.training = True

        self.loss = loss

    def forward(self, features):
        target = features['target']
        unscaled_target = features["unscaled_target"]

        mask = (unscaled_target != 0).float()

        # features["mu"] = features["mu"] 

        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        # initialize nll_loss and mse_loss as torch tensor
        nll_loss = torch.tensor(10000)
        mse_loss = torch.tensor(100)

        if not self.rho == 0:
            nll_loss = self.get_nll(features, target, mask).mean()

        # if not self.rho == 1:
        predict = (features["mu"] * features['logw'].exp().unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True)

        target = features["target"]
        unscaled_target = features["unscaled_target"]

        mask = (unscaled_target != 0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        if self.loss == "mae":
            mse_loss = torch.abs(predict-target)
        elif self.loss == "mse":
            mse_loss = (predict-target) ** 2
        else:
            raise NotImplementedError

        mse_loss = mse_loss * mask
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
        mse_loss = torch.mean(mse_loss)

        loss = self.rho * nll_loss + (1-self.rho) * mse_loss
        # dev = (predict - unscaled_target)
        return loss, nll_loss.item(), mse_loss.item()

    def get_nll(self, features, target, mask):
        mu, L_s, L_t = self.get_parameters(features)
        # w = features["w"]
        logw = features["logw"].squeeze(-1)

        b, n, t, r = mu.shape
        n = self.n_vars
        t = self.pred_len

        R_ext = (mu - target) * mask
        # R_flatten = R_ext.permute(0, 3, 1, 2)
        R_flatten = R_ext

        L_t = L_t.unsqueeze(0).repeat(b, 1, 1, 1)  # L_t L_tT = prc_T
        L_s = L_s.unsqueeze(0).repeat(b, 1, 1, 1)  # L_s L_sT = prc_S

        Ulogdet = L_s.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        Vlogdet = L_t.diagonal(dim1=-1, dim2=-2).log().sum(-1)

        Q_t = torch.einsum("brij,brjk,brkl->bril", L_t.transpose(-1, -2), R_flatten, L_s)
        mahabolis = -0.5 * torch.pow(Q_t, 2).sum((-1, -2))

        nll = -n*t/2 * math.log(2*math.pi) + mahabolis + n * Vlogdet + t * Ulogdet + logw

        nll = - torch.logsumexp(nll, dim=1)
        # print(f"maxnll {nll.max():.5f} minnll {nll.min():.5f}")
        return nll

    def sample(self, features, nsample=None):
        mu, L_s, L_t = self.get_parameters(features)
        b, n, t, _ = mu.shape
        r = L_s.shape[0]
        logw = features["w"].squeeze(-1)

        U = torch.inverse(L_s).unsqueeze(0).repeat(b, 1, 1, 1)
        V = torch.inverse(L_t).unsqueeze(0).repeat(b, 1, 1, 1)

        mixture_dist = Dist.Categorical(logits=logw)
        mixture_sample = mixture_dist.sample((nsample,))
        mixture_sample_r = mixture_sample.reshape(b,nsample,1,1,1).repeat(1,1,r,n,t)

        eps = torch.randn((b,nsample,r,n,t), device=mu.device)
        com_samples = mu.reshape(b,r,n,t).unsqueeze(1) + torch.einsum("brln,bsrnt,brtk->bsrlk", U , eps , V.transpose(-1,-2))

        samples = torch.gather(com_samples, 2, mixture_sample_r)

        return samples[:,:,0,:,:]
        # torch.einsum("ln,nt,tk->lk", U[0],samples[0,0],V[0])

    def get_parameters(self, features):
        return features['mu'], features["L_spatial"], features["L_temporal"]


def prepare_model(args, blocks, n_vertex):
    # loss = nn.MSELoss()

    loss = CholeskyResHead(n_vertex, args.n_components, pred_len=args.n_pred, rho=args.rho)
    covariance = FixedResCov(args.n_components, n_vertex, args.n_pred, rho=args.rho)
    
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    parameters = list(model.parameters()) + list(covariance.parameters())
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler, covariance

def train(l:CholeskyResHead, args, optimizer, scheduler, es, model, covariance:FixedResCov, zscore:StandardScaler, device,train_iter, val_iter, test_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        nll_loss_sum = 0.0
        mse_loss_sum = 0.0
        mae_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
        rmse_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
        mape_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
        mae_n = torch.FloatTensor([0.0] * args.n_pred).to(device)
        rmse_n = torch.FloatTensor([0.0] * args.n_pred).to(device)
        mape_n = torch.FloatTensor([0.0] * args.n_pred).to(device)

        best_val_loss = 10000

        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred, logw = model(x)

            L_s, L_t = covariance.L1.to(device), covariance.L2.to(device)
            L_s[:,torch.arange(L_s.shape[1]),torch.arange(L_s.shape[1])] = torch.exp(L_s[:,torch.arange(L_s.shape[1]),torch.arange(L_s.shape[1])])
            L_t[:,torch.arange(L_t.shape[1]),torch.arange(L_t.shape[1])] = torch.exp(L_t[:,torch.arange(L_t.shape[1]),torch.arange(L_t.shape[1])])

            y_shape = y.shape
            unscaled_y = zscore.inverse_transform(y)

            if not args.mix_mean:
                y_pred = y_pred[:,0:1,:,:].repeat(1,args.n_components,1,1)

            features = {
                "mu": y_pred,
                "logw": logw,
                "target": y,
                "unscaled_target": unscaled_y,
                "L_spatial": L_s,
                "L_temporal": L_t
            }
            mask = (unscaled_y >= zscore.min*0.9 ).float()
            if mask.sum() == 0:
                continue

            # l = loss(y_pred, y)
            l, nll_loss, mse_loss = loss.forward(features)

            predict = (y_pred * logw.exp().unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True)
            unscaled_predict = zscore.inverse_transform(predict)
            dev = (unscaled_predict - unscaled_y)* mask

            mae = (torch.abs(dev) ).sum((0,1,3)) / mask.sum((0,1,3))
            rmse = torch.sqrt(((dev ** 2) ).sum((0,1,3)) / mask.sum((0,1,3)))
            
            mape = (torch.abs(dev) / torch.abs(unscaled_y+1e-6)) * mask
            mape = mape.sum((0,1,3)) / mask.sum((0,1,3)) *100

            mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
            rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
            mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # scheduler.step()
            l_sum += l.item() * y.shape[0]
            nll_loss_sum += nll_loss * y.shape[0]
            mse_loss_sum += mse_loss * y.shape[0]

            mae_sum += mae * mask.sum((0,1,3))
            rmse_sum += rmse * mask.sum((0,1,3))
            mape_sum += mape * mask.sum((0,1,3))
            mae_n  += mask.sum((0,1,3))
            rmse_n += mask.sum((0,1,3))
            mape_n += mask.sum((0,1,3))

            n += y.shape[0]

        train_loss =  l_sum / n
        train_nll_loss = nll_loss_sum / n
        train_mse_loss = mse_loss_sum / n

        train_mae = mae_sum / mae_n
        train_rmse = rmse_sum / rmse_n
        train_mape = mape_sum / mape_n

        val_loss, val_nll_loss, val_mse_loss, val_mae, val_rmse, val_mape = val(model, val_iter, args, zscore)
        test_loss, test_nll_loss, test_mse_loss, test_mae, test_rmse, test_mape = val(model, test_iter, args, zscore)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], train_loss, val_loss, gpu_mem_alloc))

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{wandb.run.dir}/best_model.pt")
            best_val_loss = val_loss

        wandb.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
            "loss/train_loss": train_loss,
            "loss/train_nll_loss": train_nll_loss,
            "loss/train_mse_loss": train_mse_loss,
            "error/train_mae": train_mae.mean().cpu().detach().item(),
            "error/train_rmse": train_rmse.mean().cpu().detach().item(),
            "error/train_mape": train_mape.mean().cpu().detach().item(),
            "loss/val_loss": val_loss,
            "loss/val_nll_loss": val_nll_loss,
            "loss/val_mse_loss": val_mse_loss,
            "error/val_mae": val_mae.mean().cpu().detach().item(),
            "error/val_rmse": val_rmse.mean().cpu().detach().item(),
            "error/val_mape": val_mape.mean().cpu().detach().item(),
            "loss/test_loss": test_loss,
            "loss/test_nll_loss": test_nll_loss,
            "loss/test_mse_loss": test_mse_loss,
            "test/test_mae": test_mae.mean().cpu().detach().item(),
            "test/test_rmse": test_rmse.mean().cpu().detach().item(),
            "test/test_mape": test_mape.mean().cpu().detach().item(),
        }, step=epoch)

        # for i in [2,5,8,11]:
        #     wandb.log({
        #         # f"error_spec/train_mae_{str(i).zfill(2)}": train_mae[i].cpu().detach().item(),
        #         # f"error_spec/train_rmse_{str(i).zfill(2)}": train_rmse[i].cpu().detach().item(),
        #         # f"error_spec/train_mape_{str(i).zfill(2)}": train_mape[i].cpu().detach().item(),
        #         f"error_spec/val_mae_{str(i).zfill(2)}": val_mae[i].cpu().detach().item(),
        #         f"error_spec/val_rmse_{str(i).zfill(2)}": val_rmse[i].cpu().detach().item(),
        #         f"error_spec/val_mape_{str(i).zfill(2)}": val_mape[i].cpu().detach().item(),
        #         f"test/test_mae_{str(i).zfill(2)}": test_mae[i].cpu().detach().item(),
        #         f"test/test_rmse_{str(i).zfill(2)}": test_rmse[i].cpu().detach().item(),
        #         f"test/test_mape_{str(i).zfill(2)}": test_mape[i].cpu().detach().item(),
        #     }, step=epoch)
            

        if es.step(torch.tensor(val_loss)):
            print('Early stopping.')
            break

@torch.no_grad()
def val(model, val_iter, args, zscore):
    model.eval()
    l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
    nll_loss_sum = 0.0
    mse_loss_sum = 0.0
    mae_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
    rmse_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
    mape_sum = torch.FloatTensor([0.0] * args.n_pred).to(device)
    mae_n = torch.FloatTensor([0.0] * args.n_pred).to(device)
    rmse_n = torch.FloatTensor([0.0] * args.n_pred).to(device)
    mape_n = torch.FloatTensor([0.0] * args.n_pred).to(device)

    for x, y in tqdm.tqdm(val_iter):
        y_pred, logw = model(x)

        L_s, L_t = covariance.L1.to(device), covariance.L2.to(device)
        L_s[:,torch.arange(L_s.shape[1]),torch.arange(L_s.shape[1])] = torch.exp(L_s[:,torch.arange(L_s.shape[1]),torch.arange(L_s.shape[1])])
        L_t[:,torch.arange(L_t.shape[1]),torch.arange(L_t.shape[1])] = torch.exp(L_t[:,torch.arange(L_t.shape[1]),torch.arange(L_t.shape[1])])

        y_shape = y.shape
        unscaled_y = zscore.inverse_transform(y)

        features = {
            "mu": y_pred,
            "logw": logw,
            "target": y,
            "unscaled_target": unscaled_y,
            "L_spatial": L_s,
            "L_temporal": L_t
        }
        mask = (unscaled_y >= zscore.min*0.9 ).float()
        if mask.sum() == 0:
            continue

        # l = loss(y_pred, y)
        l, nll_loss, mse_loss = loss.forward(features)

        predict = (y_pred * logw.exp().unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True)
        unscaled_predict = zscore.inverse_transform(predict)
        dev = (unscaled_predict - unscaled_y)* mask

        mae = (torch.abs(dev) ).sum((0,1,3)) / mask.sum((0,1,3))
        rmse = torch.sqrt(((dev ** 2) ).sum((0,1,3)) / mask.sum((0,1,3)))
        
        mape = (torch.abs(dev) / torch.abs(unscaled_y+1e-6)) * mask
        mape = mape.sum((0,1,3)) / mask.sum((0,1,3)) *100

        mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
        rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
        mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)

        l_sum += l.item() * y.shape[0]
        nll_loss_sum += nll_loss * y.shape[0]
        mse_loss_sum += mse_loss * y.shape[0]

        mae_sum += mae * mask.sum((0,1,3))
        rmse_sum += rmse * mask.sum((0,1,3))
        mape_sum += mape * mask.sum((0,1,3))
        mae_n  += mask.sum((0,1,3))
        rmse_n += mask.sum((0,1,3))
        mape_n += mask.sum((0,1,3))

        n += y.shape[0]

    val_loss =  l_sum / n
    val_nll_loss = nll_loss_sum / n
    val_mse_loss = mse_loss_sum / n

    val_mae = mae_sum / mae_n
    val_rmse = rmse_sum / rmse_n
    val_mape = mape_sum / mape_n

    return val_loss, val_nll_loss, val_mse_loss, val_mae, val_rmse, val_mape

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()

    wandb.init(project="STGCN_1114", config=args)
    args.mix_mean = args.mix_mean == "True"

    print(args)
    print(wandb.config)
    
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler, covariance = prepare_model(args, blocks, n_vertex)
    train(loss, args, optimizer, scheduler, es, model, covariance,zscore, device, train_iter, val_iter, test_iter)
    # test(zscore, loss, model, test_iter, args)

    model.load_state_dict(torch.load(f"{wandb.run.dir}/best_model.pt"))
    test_loss, test_nll_loss, test_mse_loss, test_mae, test_rmse, test_mape = val(model, test_iter, args, zscore)

    for i in [2,5,8,11]:
        wandb.log({
            f"test/test_mae_{str(i).zfill(2)}": test_mae[i].cpu().detach().item(),
            f"test/test_rmse_{str(i).zfill(2)}": test_rmse[i].cpu().detach().item(),
            f"test/test_mape_{str(i).zfill(2)}": test_mape[i].cpu().detach().item(),
        })
        