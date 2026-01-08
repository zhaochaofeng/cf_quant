import copy
import math
import os
from typing import Union, Text

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from qlib.data.dataset import DatasetH
from qlib.data.dataset import TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        x = self.x2y(src)
        x = self.pe(x)
        x = self.tatten(x)
        x = self.satten(x)
        x = self.temporalatten(x)
        output = self.decoder(x).squeeze(-1)

        return output


def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]


def zscore(x):
    return (x - x.mean()).div(x.std())


def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


def get_device(GPU=0, return_str=False):
    USE_CUDA = torch.cuda.is_available() and GPU >= 0
    USE_MPS = torch.backends.mps.is_available()
    if USE_CUDA:
        device_str = 'cuda:{}'.format(GPU)
    elif USE_MPS:
        device_str = 'mps'
    else:
        device_str = 'cpu'
    print('\n{}\ndevice: {}'.format('-' * 50, device_str))
    if return_str:
        return device_str
    else:
        return torch.device(device_str)


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # Get the MultiIndex and create a DataFrame with position indices
        index = self.data_source.get_index()
        index_df = pd.DataFrame({'idx': np.arange(len(index))}, index=index)
        # Group by datetime and collect actual indices for each day (sorted by datetime)
        grouped = index_df.groupby(level='datetime', sort=True)
        self.daily_indices = [group['idx'].values for _, group in grouped]

    def __iter__(self):
        if self.shuffle:
            day_order = np.arange(len(self.daily_indices))
            np.random.shuffle(day_order)
            for i in day_order:
                yield self.daily_indices[i]
        else:
            for daily_idx in self.daily_indices:
                yield daily_idx

    def __len__(self):
        return len(self.data_source)


class MASTERModel(Model):
    def __init__(self, d_feat: int = 158, d_model: int = 256, t_nhead: int = 4, s_nhead: int = 2,
                 gate_input_start_index=158, gate_input_end_index=221,
                 T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5,
                 n_epochs=40, lr=8e-6, GPU=0, seed=0,
                 save_path='model/',
                 save_prefix='',
                 market='csi300',
                 only_backtest=False,
                 train_stop_loss_thred=0.95,
                 early_stop_num=3,
                 ):

        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.n_epochs = n_epochs
        self.lr = lr
        self.device = get_device(GPU)
        self.seed = seed
        self.market = market
        self.infer_exp_name = f"{self.market}_MASTER_seed{self.seed}_backtest"
        self.train_stop_loss_thred = train_stop_loss_thred

        self.fitted = False

        if self.seed is not None:
            # np.random.seed(self.seed)
            # torch.manual_seed(self.seed)
            np.random.seed(self.seed)  # 设置 NumPy 库的随机种子
            torch.manual_seed(self.seed)  # 设置 PyTorch 库的随机种子
            torch.cuda.manual_seed_all(self.seed)  # 设置所有 CUDA 库的随机种子
            torch.backends.cudnn.deterministic = True
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            gate_input_start_index=self.gate_input_start_index,
                            gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

        self.save_path = save_path
        self.save_prefix = save_prefix
        self.only_backtest = only_backtest
        os.makedirs(save_path, exist_ok=True)

    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            # float() 函数用于将float64 转化为float32，防止 mps 设备错误
            feature = data[:, :, 0:-1].float().to(self.device)
            label = data[:, -1, -1].float().to(self.device)
            assert not torch.any(torch.isnan(label))

            ####
            mask, label = drop_extreme(label)  # 删除异常值
            feature = feature[mask, :, :]
            label = zscore(label)  # CSZscoreNorm。label 取了最后一个时间点
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].float().to(self.device)
            label = data[:, -1, -1].float().to(self.device)
            mask, label = drop_na(label)
            label = zscore(label)
            with torch.no_grad():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_train.config(**{'fillna_type': 'ffill+bfill'})
        dl_valid.config(**{'fillna_type': 'ffill+bfill'})
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        # best_param = self.model.state_dict()    # 最优保存模型参数
        # best_val_loss = 1e3

        # count = 0
        for step in range(self.n_epochs):
            # count += 1
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            if dl_valid:
                metrics = self.predict2(dl_valid)
                print("Epoch %d, train_loss %.6f, valid_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." %
                      (step, train_loss, val_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))

            # if best_val_loss > val_loss:
            #     count = 0
            #     best_param = copy.deepcopy(self.model.state_dict())
            #     best_val_loss = val_loss
            # if count >= self.early_stop_num:
            #     break
            if train_loss < self.train_stop_loss_thred:
                break

        # self.model.load_state_dict(best_param)
        # torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        # if use_pretrained:
        #     self.load_param(f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        dl_test.config(**{'fillna_type': 'ffill+bfill'})

        # 创建 sampler 并获取索引顺序
        sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)
        test_loader = DataLoader(dl_test, sampler=sampler, drop_last=False)

        # 获取 sampler 实际迭代的索引顺序
        index_order = np.concatenate(sampler.daily_indices)

        pred_all = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            # feature = data[:, :, 0:-1].float().to(self.device)
            feature = data.float().to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            pred_all.append(pred.ravel())

        # 用 sampler 的实际索引顺序构建正确的索引
        original_index = dl_test.get_index()
        sorted_index = original_index[index_order]
        pred_all = pd.Series(np.concatenate(pred_all), index=sorted_index)

        return pred_all

    def predict2(self, dl_valid: TSDataSampler):
        # torch.utils.data.DataLoader
        test_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            # torch.Size([1, 299, 8, 222]) -> torch.Size([299, 8, 222])
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]  # [N, ]

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            # ravel：多维数组转化为1维
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic) / np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric) / np.std(ric)
        }

        return metrics