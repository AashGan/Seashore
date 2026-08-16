# -*- coding: utf-8 -*-

# CODE SOURCE:  https://github.com/e-apostolidis/CA-SUM/blob/main/model/layers/summarizer.py , https://github.com/e-apostolidis/CA-SUM/blob/main/model/layers/attention.py

# Small modifications to remove certain parameters which are not used 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def getPositionEncoding(seq_len, d, n=100000):
    pe = torch.zeros(seq_len, d)    
    # create position column   
    k = torch.arange(0, seq_len).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d, 2) * -(math.log(n) / d)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)
    pe = pe.unsqueeze(0)
  
    return pe


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """ The basic Attention 'cell' containing the learnable parameters of Q, K and V.

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.Wk = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wv = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size+2, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_entropy(logits):
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T].
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[0])

    def forward(self, x):
        """ Compute the weighted frame features, through the Block diagonal sparse attention matrix and the estimates of
        the frames attentive uniqueness and the diversity.

        :param torch.Tensor x: Frame features with shape [T, input_size].
        :return: A tuple of:
                    y: The computed weighted features, with shape [T, input_size].
                    att_win : The Block diagonal sparse attention matrix, with shape [T, T].
        """
        # Compute the pairwise dissimilarity of each frame, on the initial feature space (GoogleNet features)
        x_unit = F.normalize(x, p=2, dim=1)
        similarity = x_unit @ x_unit.t()
        diversity = 1 - similarity

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        energies = torch.matmul(Q, K.transpose(1, 0))
        att_weights = self.softmax(energies)

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=energies)
        entropy = F.normalize(entropy, p=1, dim=-1)

        # Compute the mask to form the Block diagonal sparse attention matrix
        D = self.block_size
        num_blocks = math.ceil(energies.shape[0] / D)
        keepingMask = torch.ones(num_blocks, D, D, device=att_weights.device)
        keepingMask = torch.block_diag(*keepingMask)[:att_weights.shape[0], :att_weights.shape[0]]
        zeroingMask = (1 - keepingMask)
        att_win = att_weights * keepingMask

        # Pick those frames that are "invisible" to a frame, aka outside the block (mask)
        attn_remainder = att_weights * zeroingMask
        div_remainder = diversity * zeroingMask

        # Compute non-local dependencies based on the diversity of those frames
        dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
        dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
        masked_dep_factor = dep_factor * keepingMask
        att_win += masked_dep_factor

        y = torch.matmul(att_win, V)
        characteristics = (entropy, dep_factor[0, :])
        characteristics = torch.stack(characteristics).detach()
        outputs = torch.cat(tensors=(y, characteristics.t()), dim=-1)

        y = self.out(outputs)
        return y, att_win.clone()
    

#MODIFIED TO ADD POSITIONAL-ENCODING
class CA_SUM(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60,positional_encoding=False ):
        """ Class wrapping the CA-SUM model; its key modules and parameters.
        
        :param int input_size: The expected input feature size.
        :param int output_size: The produced output feature size.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(CA_SUM, self).__init__()
        self.hidden_size = output_size
        self.attention = SelfAttention(input_size=input_size, output_size=output_size, block_size=block_size)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pos_enc = positional_encoding
    def forward(self, frame_features):
        """ Produce frame-level importance scores from the frame features, using the CA-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by 
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        
        seq_len = frame_features.shape[0]
        if self.pos_enc:
            x_pos = getPositionEncoding(seq_len,self.hidden_size)
            frame_features = frame_features + x_pos.to(frame_features.device)
        if len(frame_features.shape)>2:
            frame_features = torch.squeeze(frame_features)
        residual = frame_features

        weighted_value, attn_weights = self.attention(frame_features)
        y = residual + weighted_value
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        #return y, attn_weights
        return y
    


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = CA_SUM(input_size=256, output_size=128, block_size=30).cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """