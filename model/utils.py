import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import anndata as ad


def predict(dataloader, type_list, model_use, if_pure=False):
    all_rate = []
    all_y = []
    with torch.no_grad():
        for batch in dataloader:
            x_sim = batch['x_sim']
            y = batch['y']
            if model_use.gpu_available:
                x_sim = x_sim.to(model_use.gpu)
                y = y.to(model_use.gpu)
            if if_pure:
                all_res, pred_rate = model_use.pure_forward(x_sim)
            else:
                extract_cell, noise, pred_rate = model_use.forward(x_sim)
            pred_rate = pred_rate.view(-1, len(type_list))
            all_rate.append(pred_rate)
            all_y.append(y)
    all_rate = torch.cat(all_rate, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_rate_cpu = all_rate.cpu()
    all_y_cpu = all_y.cpu()

    # 将 PyTorch 张量转换为 NumPy 数组
    all_rate_np = all_rate_cpu.numpy()
    all_y_np = all_y_cpu.numpy()

    # 创建 Pandas DataFrame
    df_rate = pd.DataFrame(all_rate_np, columns=type_list)
    df_y = pd.DataFrame(all_y_np, columns=type_list)
    CCC, RMSE, Corr = compute_metrics(df_rate, df_y)
    return CCC, RMSE, Corr, df_rate, df_y

def data2h5ad (trainortest_data, y, type_list):
    df_list = [series.to_frame().T for series in trainortest_data]
    df = pd.concat(df_list, ignore_index=True)
    adata = ad.AnnData(df.values)
    y = np.array(y)
    for i, cell_type in enumerate(type_list):
        adata.obs[cell_type] = y[:, i].reshape(-1)
    adata.uns['cell_types'] = type_list
    print(adata)
    return adata

def ccc(preds, gt):
    numerator = 2 * np.corrcoef(gt, preds)[0][1] * np.std(gt) * np.std(preds)
    denominator = np.var(gt) + np.var(preds) + (np.mean(gt) - np.mean(preds)) ** 2
    ccc_value = numerator / denominator
    return ccc_value

def compute_metrics(preds, gt):
    gt = gt[preds.columns] # Align pred order and gt order  
    x = pd.melt(preds)['value']
    y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr

class TrainCustomDataset(Dataset):
    def __init__(self, x_sim, x_sim_noise1, x_sim_noise2, y):
        self.x_sim = x_sim
        self.x_sim_noise1 = x_sim_noise1
        self.x_sim_noise2 = x_sim_noise2
        self.y = y

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     sample = {
    #         'x_sim': torch.Tensor(self.x_sim[idx]),
    #         'x_sim_noise1': torch.Tensor(self.x_sim_noise1[idx]),
    #         'x_sim_noise2': torch.Tensor(self.x_sim_noise2[idx]),
    #         'y': torch.Tensor(self.y[idx])
    #     }
    #     return sample
    def __getitem__(self, idx):  
        sample = {  
            'x_sim': torch.Tensor(np.array(self.x_sim[idx])),  # 优化这里  
            'x_sim_noise1': torch.Tensor(np.array(self.x_sim_noise1[idx])),  
            'x_sim_noise2': torch.Tensor(np.array(self.x_sim_noise2[idx])),  
            'y': torch.Tensor(np.array(self.y[idx]))  
        }  
        return sample 
    
class TestCustomDataset(Dataset):
    def __init__(self, x_sim, y):
        self.x_sim = x_sim
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {
            'x_sim': torch.Tensor(np.array(self.x_sim[idx])),
            'y': torch.Tensor(np.array(self.y[idx]))  
        }
        return sample
    
class TestNoiseCustomDataset(Dataset):
    def __init__(self, x_sim):
        self.x_sim = x_sim

    def __len__(self):
        return len(self.x_sim)

    def __getitem__(self, idx):
        sample = {
            'x_sim': torch.Tensor(self.x_sim[idx])
        }
        return sample


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def loss2rate(pred_rate, real_rate):
    return F.cross_entropy(pred_rate, real_rate)


def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss


def calculate_mse(real_data, predicted_data):
    mse_loss = torch.nn.MSELoss().cuda()
    mse = mse_loss(predicted_data, real_data)
    return mse.item()


class PatchNCELoss(nn.Module):
    def __init__(self, batchsize, temperature):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.batchsize = batchsize
        self.temperature = temperature

    def forward(self, feat_q, feat_p, feat_n):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_p = feat_p.detach()
        feat_n = feat_n.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_p.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        batch_dim_for_bmm = self.batchsize

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_n = feat_n.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_n.transpose(2, 1))

        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

