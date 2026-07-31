import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# SOURCE: https://github.com/pangzss/pytorch-CTVSUM/blob/main/attention.py

from einops import rearrange

#TODO : Check if positional encoding here is a requirement
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
    

class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_inp=1024, n_layers=4, n_head=1, d_k=64, d_v=64,
            d_model=128, d_inner=512, dropout=0., num_patches=300):

        super().__init__()
        self.n_layers = n_layers

        self.proj = nn.Linear(d_inp, d_model) 
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.unq_est = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
        )
    def forward(self, src_seq):
        # -- Forward
        enc_output = self.proj(src_seq)
        enc_output = self.layer_norm(enc_output)
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, _ = enc_layer(enc_output)
        scores = self.score(self.unq_est(enc_output.detach())[0])
        return enc_output, scores.squeeze(-1)
    

class CTVWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio_s =  0
        self.ratio_k1 =  0.1
        self.alpha =  0.5
        self.refinement_module = TransformerEncoder()
        self.bce = nn.BCELoss()
        
    
    def get_values(self,feats,proj,scores):
        assert (len(feats.shape) == 3) and (len(proj.shape) == 3)
        with torch.no_grad():
            norm_raw = F.normalize(feats, p=2, dim=-1)
            xy_raw = torch.einsum('bmc, bnc -> bmn', norm_raw, norm_raw)
        norm_proj = F.normalize(proj, p=2, dim=-1)
        xy = torch.einsum('bmc, bnc -> bmn', norm_proj, norm_proj)
        sort_ids = torch.argsort(xy_raw, -1, descending=True)

        diff_mat = 2 - 2 * xy
        L = feats.shape[1]
        S = int(L * self.ratio_s) 
        K1 = int(L * self.ratio_k1)
    
        pos = torch.gather(diff_mat, -1, sort_ids[:,:,S:S+K1])

        laln = pos.mean(dim=-1)
        
        lunif = diff_mat.mul(-2).exp().mean(dim=-1).log()

        if self.training:
            s = 24
            seg = rearrange(proj, 'b (s l) c -> (b s) l c', s=s)
            seg_feats = F.normalize(seg.mean(dim=1), dim=1) # (b s) c
            fv_xy = torch.einsum("bmc, nc -> bmn", norm_proj, seg_feats) # b m (b s)
            
            mask = torch.ones_like(fv_xy)
            # for i in range(len(mask)):
            #     mask[i,:,i * s : (i+1) * s] = 0
            lunif_fv = (fv_xy.mul(4).exp() * mask).sum(dim=-1) / mask.sum(dim=-1)
            lunif_fv = lunif_fv.log()
            unq_target = (lunif_fv - lunif_fv.min(dim=-1, keepdim=True)[0]) / (lunif_fv.max(dim=-1, keepdim=True)[0] - lunif_fv.min(dim=-1, keepdim=True)[0]).add(1e-9)
            unq_target = unq_target * 0.5 + 0.25
            lunq = self.bce(scores, 1 - unq_target.detach())
            return laln, lunif, lunif_fv, lunq
        else: 
            return laln, lunif
        
    def compute_loss(self,model_input,model_features,scores):

        laln, lunif, lunif_fv, lunq = self.get_values(model_input, model_features, scores)
        loss = laln.mean() + self.alpha * lunif.mean()  + 0.1 * lunif_fv.mean() + 0.1 * lunq
        return loss
    def forward(self,x):
        out, scores = self.refinement_module(x)
        if not self.training:
            return scores
        return out, scores


if __name__ == '__main__':
    model = TransformerEncoder()
    inp = torch.rand(1,300,1024)
    enc_output = model(inp)
    print(enc_output.shape)