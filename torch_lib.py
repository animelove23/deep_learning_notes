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

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_size, head_nums, norm_shape, ffn_input, ffn_hidden, dropout, bias= False):
        super().__init__()
        self.attention = MultiheadAttention(key_size, query_size, value_size, hidden_size, head_nums, dropout, bias)
        self.norm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, hidden_size, ffn_hidden)
        self.norm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        Y = self.norm1(X, self.attention(X, X, X, valid_lens))
        return self.norm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size,query_size, value_size, hidden_size, head_nums, norm_shape, ffn_input, ffn_hidden, dropout, layer_size, bias = False ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, dropout)
        self.net = nn.Sequential()
        for i in range(layer_size):
            self.net.add_module(f"block {i}", EncoderBlock(key_size, query_size, value_size, hidden_size, head_nums, norm_shape, ffn_input, ffn_hidden, dropout, bias))

    def forward(self, X, valid_lens):
        X = self.position_encoding(embedding(X)/math.sqrt(self.hidden_size))
        for block in self.net:
            X = block(X, valid_lens)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, value_size, key_size, hidden_size, head_nums, norm_shape, ffn_input, ffn_hidden, dropout, i, bias=False):
        super().__init__()
        self.i = i
        self.attention1 = MultiheadAttention(key_size, hidden_size, hidden_size, head_nums, dropout, bias)
        self.norm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiheadAttention(key_size, hidden_size, hidden_size, head_nums, dropout, bias)
        self.norm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, hidden_size, ffn_hidden)
        self.norm3 = AddNorm(norm_shape, dropout)
    def forward(self, X, state):
        enc_output, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:# 有n个decoder block， 第一次会把缓存填充
            key_value = X
        else:
            key_value = torch.cat((key_value, X), dim=1)
        state[2][self.i] = key_value
        if self.training:
            batch_size, num_steps =X.shape[0], X.shape[1]
            dec_valid_lens = torch.arange(1, num_steps+1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X1 = self.attention1(X, key_value, key_value, dec_valid_lens)
        Y = self.norm1(X, X1)
        Y1 = self.attention2(Y, enc_output, enc_output, enc_valid_lens)
        Y2 = self.norm2(Y, Y1)
        Y3 = self.ffn(Y2)
        return self.norm3(Y2, Y3), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    hidden_size, norm_shape, ffn_num_input, ffn_num_hiddens,num_heads, num_layers, dropout, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
            DecoderBlock(key_size, query_size, value_size, hidden_size,norm_shape, ffn_num_input, ffn_num_hiddens,
            num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.hidden_size))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][
            i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][
            i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    def attention_weights(self):
        return self._attention_weights