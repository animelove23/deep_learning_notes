import torch
import torch.nn.functional as F
import numpy
import  pandas
from torch import nn
import math

def transpose_qkv(X, head_num):
    X.reshape(X.shape[0]*head_num, X.shape[1], -1)


def sequence_mask(X, valid_length,value):
    for i in range(X.shape[0]):
        X[i][valid_length[i]:]=value

def mask_softmax(X,valid_lens, value = -1e6):
    if valid_lens == None:
        return F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.shape == 1:
            # valid_lens 要么是一维的，对每个batch做mask， 要么是多维的， 对每个batch中的每个元素做mask
            valid_lens = torch.repeat_interleave(valid_lens, X.shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X,valid_lens,value)
        return F.softmax(X.reshape(shape),dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, hide_num, dropout, bias= False):
        super(AdditiveAttention, self).__init__()
        self.w_k = nn.Linear(key_size, hide_num, bias=bias)
        self.w_q = nn.Linear(query_size, hide_num, bias=bias)
        self.w_v = nn.Linear(hide_num, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, valid_lens):
        key, query = self.w_k(key), self.w_q(query)
        key = key.unsqueeze(2)
        query = query.unsqueeze(1)
        feature = torch.tanh(key + query)
        feature = self.w_v(feature).sequeeze(-1)
        attention_weights = mask_softmax(feature, valid_lens)
        return torch.bmm(self.dropout(attention_weights), value)


class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout= nn.Dropout(dropout)

    def forward(self, key, query, value, valid_lens):
        d = query.shape[-1]
        attention_weight = mask_softmax(torch.bmm(query, key.transpose(1, 2))/ math.sqrt(d), valid_lens)
        return torch.bmm(dropout(attention_weight), value)

def transpose_qkv(X, head_num):
    return X.reshape(X.shape[0]*head_num, X.shape[1], -1)

def transpose_output(X, head_num):
    return X.reshape(X.shape[0]/head_num,X.shape[1], -1)

class MultiheadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, hide_num, heads_num, dropout, bias= False):
        super().__init__()
        self.w_k = nn.Linear(key_size, hide_num, bias=bias)
        self.w_q = nn.Linear(query_size, hide_num, bias=bias)
        self.w_v = nn.Linear(value_size, hide_num, bias=bias)
        self.w_o = nn.Linear(hide_num, hide_num, bias=bias)
        self.heads_num = heads_num
        self.attention = DotProductAttention(dropout)
    def forward(self, key, query, value, valid_lens):
        key = transpose_qkv(w_k(key), self.heads_num)# 提取QKV各自特征，然后将最后一个维度转化为统一的隐藏维度方便裁剪加减和相乘计算
        query = transpose_qkv(w_q(query), self.heads_num)
        value = transpose_qkv(w_v(value), self.heads_num)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.heads_num, dim=0)# 沿着第0维 进行repeat，第0维代表batch，每个batch的valid lens要重复到每个头上， 比如 valid_lens[[1,2,3],[2,3,4]]
            #有两个batch， 两个head
            #就沿着0 重复两次 valid_lens[[1,2,3],[1,2,3]，[2,3,4]，[2,3,4]]， 因为本来valid lens里面就有batchsize个lens， 而qkv的0维就是batchsize 乘以 head num

        output = attention(key, query, value, valid_lens)
        output_concat = transpose_output(output, self.heads_num)
        return w_o(output_concat)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input, ffn_output, ffn_hidden):
        self.dense1 = nn.Linear(ffn_input, ffn_hidden)
        self.dense2 = nn.Linear(ffn_hidden, ffn_output)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))



class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normlize = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.normlize(self.dropout(Y)+X)






