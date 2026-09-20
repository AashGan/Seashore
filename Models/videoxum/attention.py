import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,dim, num_heads=8,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.,):
      super().__init__()

      self.num_heads = num_heads
      head_dim = dim // num_heads

      self.scale = qk_scale or head_dim ** -0.5

      # Keep these names identical to the original implementation
      # so existing checkpoints remain compatible.
      self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

      self.attn_drop = nn.Dropout(attn_drop)

      self.proj = nn.Linear(dim, dim)
      self.proj_drop = nn.Dropout(proj_drop)

      # Optional attention visualization support
      self.attn_gradients = None
      self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv.unbind(0)

        attention = (q @ k.transpose(-2, -1)) * self.scale

        

        if attention_mask is not None:
          attention_mask = attention_mask.masked_fill(attention_mask == False, -1e5)
          attention_mask = attention_mask.masked_fill(attention_mask == True, 0)
          attention = attention + attention_mask

        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = attention @ v

        x = (
            x.transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """
    Feed-forward network used inside the temporal transformer.

    Parameter names intentionally match the original ViT implementation:
        fc1.weight
        fc1.bias
        fc2.weight
        fc2.bias
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x
class TransformerBlock(nn.Module):
    """
    Standard temporal transformer block.

    Parameter naming follows the existing implementation:
        norm1
        attn
        norm2
        mlp
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(
            self.norm1(x),
            attention_mask,
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class LocalAttenModule(nn.Module):
    """
    Local attention module used by VTSum_BLIP_TT_CA.

    Structure:

        LayerNorm
            ↓
        LocalAttention
            ↓
        residual connection
    """

    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(
            self.norm1(x),
            attention_mask,
        )

        return x


class TemporalTransformer(nn.Module):
    """
    Temporal transformer operating on precomputed video embeddings.

    Input:
        x: [batch, frames, embedding_dim]

    Output:
        x: [batch, frames, embedding_dim]
    """

    def __init__(
        self,
        embed_dim,
        depth=1,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = embed_dim // 64

        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

    def forward(self, x, attention_mask=None):
        for block in self.blocks:
            x = block(
                x,
                attention_mask=attention_mask,
            )

        x = self.norm(x)

        return x

    def create_tt(dim,
    depth=1,
    **kwargs,):
        """
        Create the temporal transformer.
        """
        model = TemporalTransformer(
            embed_dim=dim,
            depth=depth,
            num_heads=dim // 64,
            **kwargs,
        )

        return model, dim
if __name__ == "__main__":
    x = torch.randn(2, 32, 768)

    # --------------------------------------------------
    # 1. Padding mask
    #    True  = valid token
    #    False = padding
    # --------------------------------------------------

    lengths = torch.tensor([32, 24])

    attention_mask = (
        torch.arange(x.size(1))[None, :]
        < lengths[:, None]
    )
    window_size = 2

    idx = torch.arange(x.size(1))

    local_mask = (
        (idx[:, None] - idx[None, :]).abs()
        <= window_size
    )
    combined_mask = (
    attention_mask[:, None, None, :]
    & local_mask[None, None, :, :]
)
