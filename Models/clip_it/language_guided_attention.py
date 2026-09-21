import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        mask: boolean inclusion mask

    Returns:
        The processed features and their respective attention.
    """
    d_k = k.size()[-1]

    logits = torch.matmul(q, k.transpose(-2, -1))
    logits = logits / math.sqrt(d_k)

    if mask is not None:
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

    attention = F.softmax(logits, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention


class LanguageGuidedAttention(nn.Module):
    text_dim: int
    num_heads: int
    head_dim: int
    wq: nn.Linear
    wk: nn.Linear
    wv: nn.Linear
    o_proj: nn.Linear

    def __init__(self, frame_dim: int, text_dim: int, num_heads: int) -> None:
        super().__init__()
        if text_dim % num_heads != 0:
            raise ValueError("Text dimension must be 0 modulo number of heads.")

        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads

        self.wq = nn.Linear(frame_dim, text_dim)
        self.wk = nn.Linear(text_dim, text_dim)
        self.wv = nn.Linear(text_dim, text_dim)
        self.o_proj = nn.Linear(text_dim, frame_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.wq.weight)
        self.wq.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.wk.weight)
        self.wk.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.wv.weight)
        self.wv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape [n_batches, n_frames, n_frame_features]
            text_embed: tensor of shape [n_batches, n_sentences, n_text_features]
            mask: boolean mask of the shape [n_batches, n_heads, n_frames, n_sentences]
            return_attention: inclusion or not of attention in output

        Returns:
            Tensor of shape [n_batches, n_frames, n_frame_features] representing the attended features, and the attention tensor if return_attention is True.

        """
        batch_size, n_frames, _ = x.size()
        _, n_sent, _ = text_embed.size()

        q = self.wq(x)
        k = self.wk(text_embed)
        v = self.wv(text_embed)

        q = q.reshape(batch_size, n_frames, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, n_sent, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, n_sent, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, n_frames, self.text_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


if __name__ == "__main__":
    attention = LanguageGuidedAttention(2, 4, 2)

    frame_features = torch.Tensor(((1.0, 1.0), (1.0, 2.0), (1.0, 1.0)))
    frame_features = frame_features.reshape(1, 3, 2)

    text_features = torch.Tensor((1.0, 2.0, 3.0, 4.0))
    text_features = text_features.reshape(1, 1, 4)

    res = attention.forward(frame_features, text_features)
    print(res)
    print(res.size())
