from math import log2, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def shift(x, amt, dim = -1):
    return F.pad(x, (*((0, 0) * (-dim - 1)), amt, -amt), value = 0.)

def shift_tokens(x, amt, eps = 1e-5):
    n, device = x.shape[1], x.device

    cumsum = x.cumsum(dim = 1)
    *x, x_pass = x.chunk(amt + 1, dim = -1)
    *x_cumsum, _ = cumsum.chunk(amt + 1, dim = -1)

    amts = 2 ** torch.arange(amt)
    amts = amts.tolist()

    shifts = []
    denom = torch.arange(n, device = device)

    for x_chunk, x_cumsum_chunk, amt in zip(x, x_cumsum, amts):
        shifted_chunk = shift(x_cumsum_chunk, amt, dim = -2) - shift(x_cumsum_chunk, 2 * amt, dim = -2)
        shifted_denom = shift(denom, amt, dim = -1) - shift(denom, 2 * amt, dim = -1)
        shifted_denom = rearrange(shifted_denom, 'n -> () n ()')
        normed_shifted_x = shifted_chunk /  (shifted_denom + eps)
        shifts.append(normed_shifted_x)

    return torch.cat((*shifts, x_pass), dim = -1)

def discounted_cumsum(t, gamma):
    try:
        from torch_discounted_cumsum import discounted_cumsum_left
    except ImportError:
        print('unable to import torch_discounted_cumsum - please run `pip install torch-discounted-cumsum`')

    b, n, d = t.shape
    t = rearrange(t, 'b n d -> (b d) n')
    t = discounted_cumsum_left(t, gamma)
    t = rearrange(t, '(b d) n -> b n d', b = b)
    return t

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len,
        num_shifts,
        mult = 4,
        eps = 1e-3,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.project_in = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        self.num_shifts = num_shifts
        hidden_dim = dim * mult // 2

        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, hidden_dim)

        nn.init.constant_(self.to_gate.weight, eps)
        nn.init.constant_(self.to_gate.bias, 1.)

        self.project_out = nn.Linear(hidden_dim, dim)

        # for using discounted cumsum approach

        self.use_discounted_cumsum = use_discounted_cumsum
        self.discount_gamma = discount_gamma

    def forward(self, x):
        x = self.norm(x)

        x = self.project_in(x)

        x, gate = x.chunk(2, dim = -1)

        gate = self.gate_norm(gate)

        if self.use_discounted_cumsum:
            gate = shift(gate, 1, dim = -2)
            gate = discounted_cumsum(gate, self.discount_gamma)
        else:
            gate = shift_tokens(gate, self.num_shifts)

        x = x * self.to_gate(gate)
        return self.project_out(x)

# classes

class TokenShiftGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        ff_mult = 4,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.seq_len = max_seq_len
        num_shifts = ceil(log2(max_seq_len)) - 1

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.net = nn.Sequential(
            *[Residual(FeedForward(dim = dim, num_shifts = num_shifts, mult = ff_mult, max_seq_len = max_seq_len, use_discounted_cumsum = use_discounted_cumsum, discount_gamma = discount_gamma)) for _ in range(depth)],
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    def forward(self, x):
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device = x.device))
        x = x + rearrange(pos_emb, 'n d -> () n d')
        return self.net(x)
