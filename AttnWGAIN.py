import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from iTransformer import iTransformer
import warnings

# 忽略所有 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
     
class Attention_WGAIN(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.stage = stage
        if self.stage == "G":
            actual_feature_num = feature_num * 3
        else :
            actual_feature_num = feature_num * 2

        self.slf_attn2 = iTransformer(
            num_variates=feature_num,  # 特征数
            lookback_len=seq_len,  # 序列长度
            depth=6,  # 深度，可根据需求调整
            dim=d_model,  # 模型维度
            num_tokens_per_variate=1,  # 可根据需求调整
            pred_length=seq_len,  # 预测长度
            dim_head=d_k,  # 注意力头的维度
            heads=n_head,  # 多头注意力头数
            attn_dropout=dropout,
            ff_mult=4,  # 前馈网络放大倍数
            ff_dropout=dropout,
            num_mem_tokens=2,  # memory tokens 数量，可调整
            use_reversible_instance_norm=False,
            reversible_instance_norm_affine=False,
            flash_attn=True,  # 使用 flash attention
            use_mamba=True
        )
        self.embedding_1 = nn.Linear(actual_feature_num, d_model)
        self.reduce_dim = nn.Linear(d_model, feature_num)
        

    def forward(self, X,masks,delta=None):
        if self.stage == "G":
            x = torch.cat([X, delta], dim=2)
            input_X1 = torch.cat([x, masks], dim=2)
        else :
            input_X1 = torch.cat([X, masks], dim=2)
        enc_output = self.slf_attn2(X)
        result_1 = list(enc_output.values())[0]
        result_temp = masks * X + (1 - masks) * result_1
        return result_temp,result_1


class Generator(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device):
        super(Generator, self).__init__()
        self.encoder=Attention_WGAIN(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage="G")
        self.encoder2=Attention_WGAIN(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage="D")

    def forward(self, x, m, delta = None):
        if delta is None:
            output = self.encoder2(x, m)
        else:
            output = self.encoder(x, m, delta)
        return output



class G_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, M, G_sample,X_holdout, indicating_mask, alpha):
        Construction_MSE_loss = torch.sum((M * X - M * G_sample) ** 2) / torch.sum(M)
        impution_MSE_loss = torch.sum((indicating_mask * X_holdout - indicating_mask * G_sample) ** 2) / torch.sum(indicating_mask)
        return alpha[0] * Construction_MSE_loss + alpha[1] * impution_MSE_loss
       
