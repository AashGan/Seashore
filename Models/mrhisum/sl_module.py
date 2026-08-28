import torch
import torch.nn as nn


from .transformer import Transformer
from .score_net import ScoreFCN
# Source: https://github.com/MRHiSum/MR.HiSum/blob/main/networks/sl_module/sl_module.py
# There seems to be an issue in their original code, they never pass the mask to the actual model, so I don't know if they ever use it
class SL_module(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)
        
    def forward(self, x, mask=None):

        transformed_emb = self.transformer(x,mask)
        score = self.score_model(transformed_emb).squeeze(-1)
        
        score = torch.sigmoid(score)

        return score

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module, self).load_state_dict(state_dict)