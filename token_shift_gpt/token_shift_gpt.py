from math import log2
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def shift(x, amt):
    return F.pad(x, (0, 0, amt, -amt), value = 0.)

def shift_tokens(x, amt):
    *x, x_pass = x.chunk(amt + 1, dim = -1)
    x = tuple(map(lambda args: shift(*args), zip(x, range(0, amt + 1))))
    return torch.cat((*x, x_pass), dim = -1)

# helper classes

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len,
        num_shifts,
        mult = 4
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.project_in = nn.Linear(dim, dim * mult)
        self.num_shifts = num_shifts
        self.project_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)        
        x = shift_tokens(x, self.num_shifts)
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
        num_shifts = 10
    ):
        super().__init__()
        self.seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([FeedForward(dim = dim, num_shifts = num_shifts, mult = ff_mult, max_seq_len = max_seq_len) for _ in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    def forward(self, x):
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device = x.device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        for ff in self.layers:
            x = ff(x) + x

        return self.to_logits(x)
