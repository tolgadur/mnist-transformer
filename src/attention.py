import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads

        self.qry = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.val = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.attention_weights = None  # Store attention weights

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input from decoder, y is input from encoder
        # dim of x, y: batch_size, seq_len, d_model
        batch_size, seq_len_x, _ = x.shape
        batch_size, seq_len_y, _ = y.shape

        qry = self.qry(x)
        key = self.key(y)
        val = self.val(y)

        # reshape qry, key, val to batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len_x, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len_y, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len_y, self.heads, self.d_k)

        # dim: batch_size, seq_len_, heads, d_k -> batch_size, heads, seq_len_, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        # compute attention. dim: batch_size, heads, seq_len_x, seq_len_y
        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale

        # Store raw attention scores
        self.attention_weights = A.detach().cpu()

        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)

        # dim: batch_size, heads, seq_len_x, d_k -> batch_size, seq_len_x, heads, d_k
        A = A.transpose(1, 2)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, seq_len_x, d_model
        A = A.reshape(batch_size, seq_len_x, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A


class Attention(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4, apply_mask=False):
        super().__init__()

        self.apply_mask = apply_mask
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        # self.Qs = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        # self.Ks = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        # self.Vs = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        self.keys = nn.Linear(d_model, 3 * d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.attention_weights = None  # Store attention weights

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Split into three equal chunks of size d_model each
        qry, key, val = self.keys(x).chunk(3, dim=-1)

        # dim: batch_size, seq_len, d_model -> batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, heads, seq_len, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale

        # Store raw attention scores
        self.attention_weights = A.detach().cpu()

        if self.apply_mask:
            mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
            mask = mask.unsqueeze(0)  # add batch dimension
            A = A.masked_fill(mask == 0, float("-inf"))

        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)  # dim: batch_size, heads, seq_len, d_k

        A = A.transpose(1, 2)  # dim: batch_size, seq_len, heads, d_k
        A = A.reshape(batch_size, seq_len, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A

    # def forward(self, x: torch.Tensor):
    #     # split x into self.head, equal sized chunks across the last dimension.
    #     # dim: batch_size, seq_len, d_model -> list of batch_size, seq_len, d_k
    #     head_chunks = torch.chunk(x, self.heads, dim=-1)

    #     As = []
    #     for i in range(self.heads):
    #         Q = self.Qs[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
    #         K = self.Ks[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
    #         V = self.Vs[i](head_chunks[i])  # dim: batch_size, seq_len, d_v

    #         # dim: batch_size, seq_len, seq_len
    #         A = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

    #         # apply mask if provided
    #         if self.apply_mask:
    #             mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
    #             mask = mask.unsqueeze(0)  # add batch dimension
    #             A = A.masked_fill(mask == 0, float("-inf"))

    #         A = torch.softmax(A, dim=-1)
    #         A = torch.matmul(A, V)  # dim: batch_size, seq_len, d_v

    #         As.append(A)

    #     A = torch.cat(As, dim=-1)
    #     A = self.dropout(A)
    #     A = A + x  # residual connection
    #     A = self.layer_norm(A)

    #     return A
