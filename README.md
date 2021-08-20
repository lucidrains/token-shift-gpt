## Token Shift GPT

Implementation of Token Shift GPT - An autoregressive model that relies solely on shifting along the sequence dimension and feedforwards.

Update: Inexplicably, it actually works quite well. The feedforward module follows the same design as `gMLP`, except the feature dimension of the gate tensor is divided up into `log2(seq_len)` chunks, and the mean pool of the past consecutive segments (length 1, 2, 4, 8, etc. into the past) are shifted into each chunk before a projection along the feature dimension.

## Install

```bash
$ pip install token-shift-gpt
```

## Usage

```python
import torch
from token_shift_gpt import TokenShiftGPT

model = TokenShiftGPT(
    num_tokens = 256,
    dim = 512,
    max_seq_len = 1024,
    depth = 12,
    ff_mult = 8   # when working with small model dimensions, you may want to increase the intermediate feedforward dimension (here, 8x instead of the usual 4x), so the learning is not bottlenecked by the dimensions of the shifted chunk
)

x = torch.randint(0, 256, (1, 1024))
logits = model(x) # (1, 1024, 256)
```

To use the discounted cumulative sum approach (which only uses one chunk and seems to be just as effective as the above), just set `use_discounted_cumsum = True`

First install an additional library

```bash
$ pip install torch-discounted-cumsum
```

Then

```python
import torch
from token_shift_gpt import TokenShiftGPT

model = TokenShiftGPT(
    num_tokens = 256,
    dim = 512,
    max_seq_len = 1024,
    depth = 12,
    ff_mult = 8,
    use_discounted_cumsum = True,
    discounted_gamma = 0.9              # gamma factor for discount
)

x = torch.randint(0, 256, (1, 1024))
logits = model(x) # (1, 1024, 256)
```

## Citations

```bibtex
@misc{yu2021s2mlp,
    title   = {S$^2$-MLP: Spatial-Shift MLP Architecture for Vision}, 
    author  = {Tan Yu and Xu Li and Yunfeng Cai and Mingming Sun and Ping Li},
    year    = {2021},
    eprint  = {2106.07477},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{liu2021pay,
    title   = {Pay Attention to MLPs}, 
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@software{peng_bo_2021_5196578,
    author       = {PENG Bo},
    title        = {BlinkDL/RWKV-LM: 0.01},
    month        = {aug},
    year         = {2021},
    publisher    = {Zenodo},
    version      = {0.01},
    doi          = {10.5281/zenodo.5196578},
    url          = {https://doi.org/10.5281/zenodo.5196578}
}
```
