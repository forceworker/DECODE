import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm, trange
import time
import numpy as np
from colorama import Fore, Style
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


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Layer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LayerNorm(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out = self.layer(x)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, out_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, out_size)

    def forward(self, x):
        values, keys, query = x, x, x
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split the embedding into self.heads different pieces
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        values = values.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)
        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # Run through final linear layer
        out = self.fc_out(out)
        return out


class MaskingNet(nn.Module):
    def __init__(self, feat_map_w, feat_map_h, num_cell_type, heads=4):
        super(MaskingNet, self).__init__()
        self.attn = MultiHeadAttention(feat_map_w, feat_map_w * 2, heads)
        self.feat_map_w = feat_map_w
        self.feat_map_h = feat_map_h
        self.num_cell_type = num_cell_type

    def forward(self, x):
        mask_res = []
        attn_x = self.attn(x)
        attn_x = attn_x.reshape(-1, self.feat_map_h, self.feat_map_w, 2)
        for n in range(2):
            mask = nn.functional.relu(attn_x[:, :, :, n])
            # multiply with mask and magnitude
            mid_x = (mask * x)
            mask_res.append(mid_x)
        return mask_res


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    def __init__(self, nc=256):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc
        self.mlp = nn.Sequential(*[nn.Linear(1, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])

    def forward(self, feats, patch_ids=None):
        return_feats = []
        for feat_id, feat in enumerate(feats):
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            patch_id = patch_ids[feat_id]
            patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            x_sample = self.mlp(x_sample)
            x_sample = self.l2norm(x_sample)
            return_feats.append(x_sample)
        return return_feats


class MBdeconv(nn.Module):
    def __init__(self, num_MB, feat_map_w, feat_map_h, num_cell_type, epoches, Alpha, Beta, train_data, test_data, domain_ep=25):
        super().__init__()
        self.num_MB = num_MB
        self.feat_map_w = feat_map_w
        self.feat_map_h = feat_map_h
        self.num_cell_type = num_cell_type
        self.encoder = nn.Sequential(EncoderBlock(num_MB, feat_map_w * 2, 0),
                                     EncoderBlock(feat_map_w * 2, feat_map_w, 0.3))
        self.enc_linear = nn.Linear(feat_map_w, feat_map_w * feat_map_h)
        self.masknet = MaskingNet(feat_map_w, feat_map_h, num_cell_type)
        self.linear_attn = nn.Linear(feat_map_h, 1)
        self.decoder = nn.Sequential(Layer(feat_map_w, 2 * feat_map_w),
                                     Layer(2 * feat_map_w, feat_map_w))
        self.rate_linear = nn.Sequential(
            nn.Linear(feat_map_w, num_cell_type),
            nn.Softmax(dim=-1)
        )
        self.samplenet = PatchSampleF()
        self.epoches = epoches
        self.Alpha = Alpha
        self.Beta = Beta
        self.domain_ep = domain_ep
        self.gpu_available = torch.cuda.is_available()
        self.train_data = train_data
        self.test_data = test_data
        if self.gpu_available:
            self.gpu = torch.device("cuda:0")
        self.best_rmse = float('inf')
        self.save_path = f'save_models/{num_MB}'

    def forward(self, mix):
        enc_mix = self.encoder(mix)
        enc_mix = self.enc_linear(enc_mix)
        # Each cell vector is processed through an encoder.Each cell vector is processed through an encoder.
        enc_mix = enc_mix.reshape(-1, self.feat_map_h, self.feat_map_w)
        mask_res = self.masknet(enc_mix)
        noise = mask_res[0]
        extract_cell = mask_res[1]
        res = self.linear_attn(extract_cell.transpose(-2, -1)).transpose(-2, -1)
        res = self.decoder(res.squeeze(1))
        pred_rate = self.rate_linear(res)
        return extract_cell, noise, pred_rate

    def pure_forward(self, mix):
        enc_mix = self.encoder(mix)
        enc_mix = self.enc_linear(enc_mix)
        enc_mix = enc_mix.reshape(-1, self.feat_map_h, self.feat_map_w)
        res = self.linear_attn(enc_mix.transpose(-2, -1)).transpose(-2, -1)
        res = self.decoder(res.squeeze(1))
        pred_rate = self.rate_linear(res)
        return enc_mix, pred_rate

    def train_model(self, model_save_name, if_pure, patience):
        batchsize = self.train_data.batch_size
        self.nceloss = PatchNCELoss(batchsize, temperature=0.07)
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        start_time = time.time()
        loss1_list = []
        loss2_list = []
        nce_loss_list = []
        total_loss_list = []
        counter = 0

        # 颜色定义
        HEADER = Fore.CYAN
        METRIC = Fore.GREEN
        WARNING = Fore.YELLOW
        BEST = Fore.MAGENTA
        RESET = Style.RESET_ALL

        print(f"\n{HEADER}===== Starting Training (Total Epochs: {self.epoches}) =====")
        print(f"Patience for early stopping: {patience} epochs{RESET}\n")

        for ep in range(self.epoches):
            self.train()

            # 冻结编码器参数
            for name, param in self.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False

            epoch_loss = []
            loss1_train = []
            loss2_train = []
            nce_loss_train = []

            # 创建进度条
            pbar = tqdm(enumerate(self.train_data), total=len(self.train_data), 
                        desc=f"Epoch {ep+1}/{self.epoches}", 
                        unit='batch',
                        dynamic_ncols=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches')

            for i, _ in pbar:
                # 数据处理
                x_sim = _['x_sim']
                x_noise1 = _['x_sim_noise1']
                x_noise2 = _['x_sim_noise2']
                y = _['y']
                if self.gpu_available:
                    x_sim = x_sim.to(self.gpu)
                    x_noise1 = x_noise1.to(self.gpu)
                    x_noise2 = x_noise2.to(self.gpu)
                    y = y.to(self.gpu)

                # 前向传播
                extract_cell1, noise1, pred_rate1 = self.forward(x_noise1)
                extract_cell2, noise2, pred_rate2 = self.forward(x_noise2)
                enc_mix, pred_rate = self.pure_forward(x_sim)

                # 准备特征
                extract_cell = [extract_cell1.unsqueeze(1), extract_cell2.unsqueeze(1)]
                noise = [noise1.unsqueeze(1), noise2.unsqueeze(1)]
                pure_cell = [enc_mix.unsqueeze(1), enc_mix.unsqueeze(1)]

                # 生成采样ID
                patch_id = np.random.permutation(self.feat_map_h * self.feat_map_w)
                patch_id = [patch_id[:batchsize]] * 2

                # 采样处理
                sample_extract_cell = self.samplenet(extract_cell, patch_id)
                sample_noise = self.samplenet(noise, patch_id)
                sample_pure_cell = self.samplenet(pure_cell, patch_id)

                # 计算NCE损失
                nce_loss = 0.0
                for feat_q, feat_p, feat_n in zip(sample_pure_cell, sample_extract_cell, sample_noise):
                    nce_temp = self.nceloss(feat_q, feat_p, feat_n)
                    nce_loss += nce_temp.mean()

                # 计算各种损失
                loss1 = L1_loss(pred_rate1.squeeze(), y)
                loss2 = L1_loss(pred_rate2.squeeze(), y)
                loss3 = L1_loss(pred_rate.squeeze(), y)
                loss = (self.Alpha * (loss1 + loss2 + loss3) + self.Beta * (nce_loss / 2))

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 收集损失值
                current_loss = loss.item()
                epoch_loss.append(current_loss)
                loss1_train.append(loss1.item())
                loss2_train.append(loss2.item())
                nce_loss_train.append(nce_loss.item())

                # 每10个batch更新进度条描述
                if i % 10 == 0:
                    avg_loss = np.mean(epoch_loss) if epoch_loss else 0.0
                    pbar.set_postfix({
                        'avg_loss': f"{avg_loss:.4f}",
                        'current_loss': f"{current_loss:.4f}"
                    })

            # 计算本轮epoch的平均损失
            avg_total_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            avg_loss1 = np.mean(loss1_train) if loss1_train else 0.0
            avg_loss2 = np.mean(loss2_train) if loss2_train else 0.0
            avg_nce_loss = np.mean(nce_loss_train) if nce_loss_train else 0.0

            # 保存本轮损失
            total_loss_list.append(avg_total_loss)
            loss1_list.append(avg_loss1)
            loss2_list.append(avg_loss2)
            nce_loss_list.append(avg_nce_loss)

            # 关闭进度条
            pbar.close()

            # 评估模型性能
            test_rmse, test_mae = self.evaluate(if_pure=if_pure)
            elapsed_time = time.time() - start_time

            # 打印本轮结果
            print(f"{HEADER}[Ep {ep+1}] {elapsed_time:.1f}s | "
                 f"Loss: {METRIC}{avg_total_loss:.4f}{RESET} "
                 f"(L1: {avg_loss1:.4f}, L2: {avg_loss2:.4f}, NCE: {avg_nce_loss:.4f}) | "
                 f"Test: RMSE={METRIC}{test_rmse:.4f}{RESET}, MAE={METRIC}{test_mae:.4f}{RESET}")

            # 检查是否需要保存模型
            if test_rmse < self.best_rmse:
                self.best_rmse = test_rmse
                counter = 0
                self.save_model(model_save_name)
                print(f"  {BEST}★ New best RMSE! Model saved.{RESET}")
            else:
                counter += 1
                print(f"  {WARNING}↯ No improvement ({counter}/{patience}){RESET}")

            # 检查是否需要早停
            if counter >= patience:
                print(f"{HEADER}\nEarly stopping triggered at epoch {ep+1}!")
                print(f"Best RMSE achieved: {self.best_rmse:.4f}{RESET}\n")
                break

        # 最终训练报告
        total_time = time.time() - start_time
        print(f"\n{HEADER}===== Training Complete! =====")
        print(f"Total training time: {total_time:.1f} seconds")
        print(f"Final losses: Total={total_loss_list[-1]:.4f}, L1={loss1_list[-1]:.4f}, "
              f"L2={loss2_list[-1]:.4f}, NCE={nce_loss_list[-1]:.4f}")
        print("==============================")
        print(f"{RESET}")

        return loss1_list, loss2_list, nce_loss_list

    def save_model(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path+ "/" + name + ".pt")
        
    def evaluate(self, if_pure=False):
        self.eval()
        all_rmse = []
        all_mae = []
        with torch.no_grad():
            for batch in self.test_data:
                x_sim = batch['x_sim']
                y = batch['y']
                if self.gpu_available:
                    x_sim = x_sim.to(self.gpu)
                    y = y.to(self.gpu)
                if if_pure:
                    enc_mix, pred_rate = self.pure_forward(x_sim)
                else:
                    extract_cell, noise, pred_rate = self.forward(x_sim)

                # 计算RMSE
                rmse = ((pred_rate.squeeze() - y) ** 2).mean().pow(1 / 2)

                # 计算MAE
                mae = ((pred_rate.squeeze() - y).abs()).mean()

                # 将每个batch的RMSE和MAE值添加到对应的列表中
                all_rmse.append(rmse.item())
                all_mae.append(mae.item())

        # 计算所有batch的RMSE和MAE值的平均值
        avg_rmse = sum(all_rmse) / len(all_rmse)
        avg_mae = sum(all_mae) / len(all_mae)
        return avg_rmse, avg_mae
    


