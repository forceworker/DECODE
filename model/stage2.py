import os
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
from tqdm import tqdm
import time
import numpy as np
from colorama import Fore, Style, init
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .utils import *

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DANN(object):
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def DANN_model(self, celltype_num):
        feature_num = len(self.used_features)
        
        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, 512, 0), 
                                        EncoderBlock(512, 256, 0.3))

        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2), 
                                          nn.Linear(128, celltype_num), 
                                          nn.Softmax(dim=1))
        
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2), 
                                              nn.Linear(128, 1), 
                                              nn.Sigmoid())

        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        model_da.append(self.discriminator_da)
        return model_da

    def prepare_dataloader(self, source_data, target_data, valid_data, batch_size):
        ### Prepare data loader for training ###
        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        target_ratios = [target_data.obs[ctype] for ctype in self.labels]
        self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)
        
        # valid dataset
        self.valid_data_x = valid_data.X.astype(np.float32)
        valid_ratios = [valid_data.obs[ctype] for ctype in self.labels]
        self.valid_data_y = np.array(valid_ratios, dtype=np.float32).transpose()
        va_data = torch.FloatTensor(self.valid_data_x)
        va_labels = torch.FloatTensor(self.valid_data_y)
        valid_dataset = Data.TensorDataset(va_data, va_labels)
        self.valid_target_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        
    def train(self, source_data, target_data, valid_data, patience):
        ### prepare model structure ###
        self.prepare_dataloader(source_data, target_data, valid_data, self.batch_size)
        self.model_da = self.DANN_model(self.celltype_num).cuda()

        # 设置优化器
        optimizer_da1 = torch.optim.Adam([
            {'params': self.encoder_da.parameters()},
            {'params': self.predictor_da.parameters()},
            {'params': self.discriminator_da.parameters()}
        ], lr=self.learning_rate)

        optimizer_da2 = torch.optim.Adam([
            {'params': self.encoder_da.parameters()},
            {'params': self.discriminator_da.parameters()}
        ], lr=self.learning_rate)

        # 定义损失函数和标签
        criterion_da = nn.BCELoss().cuda()
        source_label = torch.ones(self.batch_size).unsqueeze(1).cuda()
        target_label = torch.zeros(self.batch_size).unsqueeze(1).cuda()

        # 初始化变量
        counter = 0
        best_model_weights = None
        best_rmse = float('inf')
        pred_loss_list = []
        disc_loss_list = []
        disc_da_loss_list = []
        valid_rmse_list = []

        # 颜色定义
        HEADER = Fore.CYAN
        METRIC = Fore.GREEN
        WARNING = Fore.YELLOW
        BEST = Fore.MAGENTA
        RESET = Style.RESET_ALL

        print(f"\n{HEADER}===== Starting Training (Total Epochs: {self.num_epochs}) =====")
        print(f"Patience for early stopping: {patience} epochs")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}{RESET}\n")

        for epoch in range(self.num_epochs):
            self.model_da.train()
            total_iterations = len(self.train_source_loader)

            # 创建进度条
            pbar = tqdm(total=total_iterations, 
                        desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                        dynamic_ncols=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches')

            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.

            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # 获取目标域数据
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                # 数据处理
                src_x = source_x.cuda()
                tgt_x = target_x.cuda()
                src_y = source_y.cuda()

                # 第一次前向传播和优化
                embedding_source = self.encoder_da(src_x)
                embedding_target = self.encoder_da(tgt_x)
                frac_pred = self.predictor_da(embedding_source)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # 计算第一次损失
                pred_loss = L1_loss(frac_pred, src_y)
                disc_loss = (criterion_da(domain_pred_source, source_label[:domain_pred_source.shape[0]]) +
                            criterion_da(domain_pred_target, target_label[:domain_pred_target.shape[0]]))

                loss = pred_loss + disc_loss

                # 优化步骤1
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()

                # 第二次前向传播和优化
                embedding_source = self.encoder_da(src_x)
                embedding_target = self.encoder_da(tgt_x)
                domain_pred_source = self.discriminator_da(embedding_source)
                domain_pred_target = self.discriminator_da(embedding_target)

                # 计算第二次损失（域对抗损失）
                disc_loss_DA = (criterion_da(domain_pred_target, source_label[:domain_pred_target.shape[0]]) +
                               criterion_da(domain_pred_source, target_label[:domain_pred_source.shape[0]]))

                # 优化步骤2
                optimizer_da2.zero_grad()
                disc_loss_DA.backward(retain_graph=True)
                optimizer_da2.step()

                # 收集损失
                pred_loss_epoch += pred_loss.item()
                disc_loss_epoch += disc_loss.item()
                disc_loss_DA_epoch += disc_loss_DA.item()

                # 更新进度条
                pbar.update(1)
                if batch_idx % 10 == 0:  # 每10个batch更新一次
                    avg_pred = pred_loss_epoch / (batch_idx + 1)
                    avg_disc = disc_loss_epoch / (batch_idx + 1)
                    avg_disc_da = disc_loss_DA_epoch / (batch_idx + 1)

                    pbar.set_postfix({
                        'pred': f'{avg_pred:.4f}',
                        'disc': f'{avg_disc:.4f}',
                        'disc_DA': f'{avg_disc_da:.4f}'
                    })

            # 关闭进度条
            pbar.close()

            # 计算本轮平均损失
            pred_avg = pred_loss_epoch / total_iterations
            disc_avg = disc_loss_epoch / total_iterations
            disc_da_avg = disc_loss_DA_epoch / total_iterations

            # 保存本轮损失
            pred_loss_list.append(pred_avg)
            disc_loss_list.append(disc_avg)
            disc_da_loss_list.append(disc_da_avg)

            # 验证模型性能
            valid_rmse = self.evaluate(self.valid_target_loader)
            valid_rmse_list.append(valid_rmse)

            # 输出本轮结果
            print(f"{HEADER}[Ep {epoch+1}] | "
                 f"Pred: {METRIC}{pred_avg:.4f}{RESET} | "
                 f"Disc: {disc_avg:.4f} | "
                 f"Disc_DA: {disc_da_avg:.4f} | "
                 f"Valid RMSE: {METRIC}{valid_rmse:.4f}{RESET}")

            # 检查是否需要保存模型
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                counter = 0
                best_model_weights = {
                    'encoder': copy.deepcopy(self.encoder_da.state_dict()),
                    'predictor': copy.deepcopy(self.predictor_da.state_dict()),
                    'discriminator': copy.deepcopy(self.discriminator_da.state_dict())
                }
                print(f"  {BEST}★ New best RMSE! Model saved.{RESET}")
            else:
                counter += 1
                print(f"  {WARNING}↯ No improvement ({counter}/{patience}){RESET}")

            # 检查是否需要早停
            if counter >= patience:
                print(f"{HEADER}\nEarly stopping triggered at epoch {epoch+1}!")
                print(f"Best RMSE achieved: {best_rmse:.4f}{RESET}\n")
                break

        # 最终训练报告
        print(f"\n{HEADER}===== Training Complete! =====")
        print(f"Total epochs: {len(pred_loss_list)}/{self.num_epochs}")
        print(f"Best RMSE: {BEST}{best_rmse:.4f}{RESET}")
        print(f"Final losses: Pred={pred_loss_list[-1]:.4f}, Disc={disc_loss_list[-1]:.4f}, "
              f"Disc_DA={disc_da_loss_list[-1]:.4f}")
        print("===============================")
        print(f"{RESET}")

        return pred_loss_list, disc_loss_list, disc_da_loss_list, best_model_weights

    
    def prediction(self, data_test):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(data_test):
            logits = self.predictor_da(self.encoder_da(x.cuda())).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth
    
    def evaluate(self, valid_data):
        final_preds_target, ground_truth_target = self.prediction(valid_data)
        _ = []
        for label in self.labels:  
            rmse = np.sqrt(np.mean((final_preds_target[label] - ground_truth_target[label]) ** 2))  
            _.append(rmse)  

            # 计算平均RMSE  
        avg_rmse = np.mean(_)  
        return avg_rmse