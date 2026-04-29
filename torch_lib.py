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
        self.attention_weights = None
    def forward(self, key, query, value, valid_lens):
        d = query.shape[-1]
        self.attention_weights = mask_softmax(torch.bmm(query, key.transpose(1, 2))/ math.sqrt(d), valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), value)

def transpose_qkv(X, head_num):
    batch_size, num_steps, num_hiddens = X.shape
    X = X.reshape(batch_size, num_steps, head_num, num_hiddens // head_num)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(batch_size * head_num, num_steps, num_hiddens // head_num)

def transpose_output(X, head_num):
    batch_size_times_heads, num_steps, hidden_per_head = X.shape
    batch_size = batch_size_times_heads // head_num
    X = X.reshape(batch_size, head_num, num_steps, hidden_per_head)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(batch_size, num_steps, head_num * hidden_per_head)

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
        key = transpose_qkv(self.w_k(key), self.heads_num)# 提取QKV各自特征，然后将最后一个维度转化为统一的隐藏维度方便裁剪加减和相乘计算
        query = transpose_qkv(self.w_q(query), self.heads_num)
        value = transpose_qkv(self.w_v(value), self.heads_num)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.heads_num, dim=0)# 沿着第0维 进行repeat，第0维代表batch，每个batch的valid lens要重复到每个头上， 比如 valid_lens[[1,2,3],[2,3,4]]
            #有两个batch， 两个head
            #就沿着0 重复两次 valid_lens[[1,2,3],[1,2,3]，[2,3,4]，[2,3,4]]， 因为本来valid lens里面就有batchsize个lens， 而qkv的0维就是batchsize 乘以 head num

        output = self.attention(key, query, value, valid_lens)
        output_concat = transpose_output(output, self.heads_num)
        return self.w_o(output_concat)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
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
        super().__init__()
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
        X = self.position_encoding(self.embedding(X)/math.sqrt(self.hidden_size))
        for block in self.net:
            X = block(X, valid_lens)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size,value_size, hidden_size, head_nums, norm_shape, ffn_input, ffn_hidden, dropout, i, bias=False):
        super().__init__()
        self.i = i
        self.attention1 = MultiheadAttention(key_size, query_size, value_size, hidden_size,head_nums, dropout, bias)
        self.norm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiheadAttention(key_size, query_size, value_size, hidden_size,head_nums, dropout, bias)
        self.norm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, hidden_size, ffn_hidden)
        self.norm3 = AddNorm(norm_shape, dropout)
    def forward(self, X, state):
        enc_output, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:# 有n个decoder block， 第一次会把缓存填充
            key_value = X
        else:
            key_value = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_value
        if self.training:
            batch_size, num_steps =X.shape[0], X.shape[1]
            dec_valid_lens = torch.arange(1, num_steps+1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X1 = self.attention1(key_value,X,  key_value, dec_valid_lens)
        Y = self.norm1(X, X1)
        Y1 = self.attention2( enc_output,Y, enc_output, enc_valid_lens)
        Y2 = self.norm2(Y, Y1)
        Y3 = self.ffn(Y2)
        return self.norm3(Y2, Y3), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    hidden_size, norm_shape, ffn_num_input, ffn_num_hiddens,num_heads, num_layers, dropout, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
            DecoderBlock(key_size, query_size, value_size, hidden_size,num_heads,norm_shape, ffn_num_input, ffn_num_hiddens,
             dropout, i, False))
        self.dense = nn.Linear(hidden_size, vocab_size)
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

import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch_lib import TransformerEncoder, TransformerDecoder


# =========================
# 1. 特殊 token
# =========================

PAD = 0
BOS = 1
EOS = 2

# 数字 0-9 映射成 token 3-12
DIGIT_OFFSET = 3
VOCAB_SIZE = 13


def number_to_token(x):
    return x + DIGIT_OFFSET


def token_to_number(x):
    return x - DIGIT_OFFSET


# =========================
# 2. 构造一个简单数据集
# =========================

class ReverseDataset(Dataset):
    """
    任务：
    输入:  [1, 2, 3]
    输出:  [3, 2, 1]

    实际 token：
    encoder 输入: [1, 2, 3, <eos>, <pad>, ...]
    decoder 输入: [<bos>, 3, 2, 1, ...]
    label:        [3, 2, 1, <eos>, ...]
    """

    def __init__(self, num_samples=2000, max_len=8):
        self.num_samples = num_samples
        self.max_len = max_len
        self.total_len = max_len + 1  # 多一个位置放 EOS

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = random.randint(3, self.max_len)

        nums = [random.randint(0, 9) for _ in range(length)]
        src_tokens = [number_to_token(x) for x in nums]

        tgt_tokens = list(reversed(src_tokens))

        # encoder 输入：原序列 + EOS
        enc_X = src_tokens + [EOS]

        # decoder 输入：BOS + 目标序列
        dec_X = [BOS] + tgt_tokens

        # decoder 标签：目标序列 + EOS
        Y = tgt_tokens + [EOS]

        # padding
        enc_valid_len = len(enc_X)

        enc_X = enc_X + [PAD] * (self.total_len - len(enc_X))
        dec_X = dec_X + [PAD] * (self.total_len - len(dec_X))
        Y = Y + [PAD] * (self.total_len - len(Y))

        return (
            torch.tensor(enc_X, dtype=torch.long),
            torch.tensor(enc_valid_len, dtype=torch.long),
            torch.tensor(dec_X, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long),
        )


# =========================
# 3. Encoder-Decoder 包装
# =========================

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        dec_outputs, _ = self.decoder(dec_X, dec_state)
        return dec_outputs


# =========================
# 4. mask loss
# =========================

def sequence_mask(X, valid_lens, value=0):
    """
    X: shape = (num_rows, num_steps)
    valid_lens: shape = (num_rows,)
    """

    max_len = X.shape[1]

    mask = torch.arange(
        max_len,
        device=X.device
    )[None, :] < valid_lens[:, None]

    X = X.clone()
    X[~mask] = value

    return X


def mask_softmax(X, valid_lens, value=-1e6):
    """
    X: attention scores

    常见 shape:
    X.shape = (batch_size, num_queries, num_keys)

    在多头注意力中:
    X.shape = (batch_size * num_heads, num_queries, num_keys)
    """

    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape

    # valid_lens 是一维：
    # 例如 [2, 3, 4]
    # 表示每个 batch 的有效长度
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])

    # valid_lens 是二维：
    # 例如 decoder 训练阶段的：
    # [[1, 2, 3, ...],
    #  [1, 2, 3, ...]]
    else:
        valid_lens = valid_lens.reshape(-1)

    # 关键：先把 X 从 3D 展平为 2D
    # 原来: (batch_size * heads, query_len, key_len)
    # 变成: (batch_size * heads * query_len, key_len)
    X = X.reshape(-1, shape[-1])

    X = sequence_mask(X, valid_lens, value)

    # 再变回原来的 3D
    return F.softmax(X.reshape(shape), dim=-1)
def masked_cross_entropy_loss(pred, label, valid_len):
    """
    pred:  shape = (batch_size, num_steps, vocab_size)
    label: shape = (batch_size, num_steps)
    valid_len: shape = (batch_size,)
    """

    weights = torch.ones_like(label, dtype=torch.float32)
    weights = sequence_mask(weights, valid_len, value=0)

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # CrossEntropyLoss 需要 pred 形状是：
    # (batch_size, vocab_size, num_steps)
    unweighted_loss = loss_fn(pred.permute(0, 2, 1), label)

    weighted_loss = unweighted_loss * weights

    return weighted_loss.sum() / weights.sum()


# =========================
# 5. 梯度裁剪
# =========================

def grad_clipping(model, theta):
    params = [p for p in model.parameters() if p.requires_grad]

    norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None)
    )

    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


# =========================
# 6. 构造模型
# =========================

def build_model(device):
    num_hiddens = 32
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    ffn_num_hiddens = 64
    norm_shape = [num_hiddens]

    encoder = TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        hidden_size=num_hiddens,
        head_nums=num_heads,
        norm_shape=norm_shape,
        ffn_input=num_hiddens,
        ffn_hidden=ffn_num_hiddens,
        dropout=dropout,
        layer_size=num_layers,
    )

    decoder = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        hidden_size=num_hiddens,
        norm_shape=norm_shape,
        ffn_num_input=num_hiddens,
        ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = EncoderDecoder(encoder, decoder)
    return model.to(device)


# =========================
# 7. 训练函数
# =========================

def train(model, train_iter, device, num_epochs=20, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_tokens = 0

        for enc_X, enc_valid_lens, dec_X, Y in train_iter:
            enc_X = enc_X.to(device)
            enc_valid_lens = enc_valid_lens.to(device)
            dec_X = dec_X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            Y_hat = model(enc_X, dec_X, enc_valid_lens)

            # label 中非 PAD 的数量就是有效长度
            Y_valid_lens = (Y != PAD).sum(dim=1)

            loss = masked_cross_entropy_loss(Y_hat, Y, Y_valid_lens)

            loss.backward()
            grad_clipping(model, theta=1.0)
            optimizer.step()

            total_loss += loss.item() * Y_valid_lens.sum().item()
            total_tokens += Y_valid_lens.sum().item()

        avg_loss = total_loss / total_tokens

        print(f"epoch {epoch + 1}, loss {avg_loss:.4f}")


# =========================
# 8. 预测函数
# =========================

@torch.no_grad()
def predict(model, nums, device, max_len=8):
    """
    nums: 普通数字列表，比如 [1, 2, 3, 4]
    返回：模型预测的反转结果
    """

    model.eval()

    src_tokens = [number_to_token(x) for x in nums]
    enc_X = src_tokens + [EOS]

    enc_valid_len = torch.tensor([len(enc_X)], device=device)

    total_len = max_len + 1
    enc_X = enc_X + [PAD] * (total_len - len(enc_X))
    enc_X = torch.tensor([enc_X], dtype=torch.long, device=device)

    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)

    # 预测时 decoder 第一个输入是 BOS
    dec_X = torch.tensor([[BOS]], dtype=torch.long, device=device)

    output_tokens = []

    for _ in range(total_len):
        Y_hat, dec_state = model.decoder(dec_X, dec_state)

        # 取最后一个时间步的预测结果
        next_token = int(Y_hat[:, -1, :].argmax(dim=-1).item())

        if next_token == EOS:
            break

        if next_token != PAD:
            output_tokens.append(next_token)

        # 下一轮把当前预测出来的 token 作为输入
        dec_X = torch.tensor([[next_token]], dtype=torch.long, device=device)

    result = [token_to_number(x) for x in output_tokens]

    return result


# =========================
# 9. 主程序
# =========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReverseDataset(num_samples=3000, max_len=8)
    train_iter = DataLoader(dataset, batch_size=64, shuffle=True)

    model = build_model(device)

    train(
        model=model,
        train_iter=train_iter,
        device=device,
        num_epochs=20,
        lr=0.005,
    )

    test_nums = [1, 2, 3, 4, 5]

    pred = predict(
        model=model,
        nums=test_nums,
        device=device,
        max_len=8,
    )

    print("input:     ", test_nums)
    print("prediction:", pred)
    print("target:    ", list(reversed(test_nums)))