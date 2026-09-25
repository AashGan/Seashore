import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
#Code Sourced from: https://github.com/IDT-ITI/SD-VSum/blob/main/model/layers/attention.py
# Modified to include masking function taken from clip-it 

def combined_mask_construction(x,text_embed,frame_mask=None,text_mask=None):
    if frame_mask is not None and text_mask is not None:
            mask = frame_mask[:, :, None] & text_mask[:, None, :]
            mask = mask.unsqueeze(1)
    elif frame_mask is not None:
        batch_size, n_sentences, _ = text_embed.size()
        text_mask = torch.ones(batch_size, n_sentences)
        mask = frame_mask[:, :, None] & text_mask[:, None, :]
        mask = mask.unsqueeze(1)
    elif text_mask is not None:
        batch_size, n_frames, _ = x.size()
        frame_mask = torch.ones(batch_size, n_frames)
        mask = frame_mask[:, :, None] & text_mask[:, None, :]
        mask = mask.unsqueeze(1)
    else:
        mask = None
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=1024):
        """
        Class implementing sinusoidal absolute positional encoding for sequence data.
        :param int dim: Dimensionality of the embeddings (should be even).
        :param int max_pos: Maximum sequence length to support. Defaults to 512.
        """
        super().__init__()
        pos = torch.arange(max_pos)
        freq = torch.arange(0,dim,2)

        freq = (freq * torch.tensor(10000).log()/dim).exp()
        x = pos[:, None] * freq[None, :]
        pe = torch.cat((x.sin(),x.cos()),dim=-1,)

        # [1, max_len, d_model]
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):
        """
        Generates positional encoding for a sequence of given length.
        :param int n: The number of positions (i.e., sequence length) to generate encodings for.
        :param torch.device device: The device to move the positional encoding tensor to. Defaults to CUDA.
        :return torch.Tensor: A tensor of shape [n, dim] containing the positional encodings.
        """
        N = x.size(-2)
        pe = self.pe[:, :N, :].unsqueeze(1)
        # [1, N, D]
        return x + pe


class CrossAttention(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, heads=8, pos_enc=True):
        """
        Class implementing multi-head (language-guided) cross-attention between video and text features
        :param int input_size: Dimensionality of the input video features.
        :param int text_size: Dimensionality of the input text features.
        :param int output_size: Dimensionality of the hidden/output space for each head. Defaults to 512.
        :param int heads: Number of attention heads. Defaults to 8.
        :param bool pos_enc: Whether to apply sinusoidal positional encoding. Defaults to True.
        """
        super(CrossAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = heads
        self.head_dim = output_size // heads
        self.pos_enc = pos_enc
        self.Wk = nn.Linear(in_features=text_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wv = nn.Linear(in_features=text_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.drop = nn.Dropout(p=0.5) 

        if self.pos_enc:
            self.pe = PositionalEncoding(self.input_size, max_pos=4096)

    def forward(self, video_features, text_features,video_mask=None,text_mask=None):
        """
        Compute multi-head cross-attention between video and text inputs.

        :param torch.Tensor video_features: Input video features with shape [N, input_size], where N is the number of frames.
        :param torch.Tensor text_features: Text feature tensor with shape [Scripts, M, text_size], where M is the number of sentences.
        :return torch.Tensor: Output video features with shape [N, input_size] after attention.
        This has been modified to include batched output for scripts 
        """
        outputs = []
        B = 1 # We always assume a batch size of 1
        N,D = video_features.shape
        S,M,_ = text_features.shape
        q = self.Wq(video_features)
        # [B, N, D]

        # Add script dimension.
        k = self.Wk(text_features)
        
        v = self.Wv(text_features)
       
        q = q.view(
            B, 1, N, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        k = k.view(
            B, S, M, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        v = v.view(
            B, S, M, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q,k.transpose(-2, -1))
      
        scores = scores / math.sqrt(self.head_dim)
        if text_mask is not None:
          mask = text_mask.unsqueeze(2).unsqueeze(2)

          scores = scores.masked_fill(
              ~mask,
              -1e9
          )

        attention = torch.softmax(scores,dim=-1)
        attention = self.drop(attention)
        output = torch.matmul(
            attention,
            v
        )
        output = output.permute(
            0, 1, 3, 2, 4
        )
        output = output.reshape(
            B,
            S,
            N,
            D
        )
        if self.pos_enc:
            y= self.pe(output)
        return y
