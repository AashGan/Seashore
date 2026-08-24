import torch.nn as nn
import torch
import math
# Altered to include masking for batch size > 1


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


class DiverseSelfAttention(nn.Module):

    def __init__(self,input_size=1024, output_size=1024):
        super(DiverseSelfAttention, self).__init__()


        self.m = input_size
        self.output_size = output_size
        self.dhidden = 1024
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x,mask = None ):
        """ Added masking for attention.
        """
        n = x.shape[0]  # sequence length
        
        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        diversity_vector = torch.prod(1-att_weights_, 1,keepdim=True)
        diversity_vector = diversity_vector/torch.norm(diversity_vector,p=1)
        #weights = self.drop50(att_weights_)
        #y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = diversity_vector*V
        y = self.output_linear(y)

        return y, att_weights_
    
class SUM_GDA(nn.Module):
    def __init__(self,input_size = 1024,mode = 'Unsupervised',positional_encoding=True):
        super(SUM_GDA,self).__init__()
        self.hidden_size = input_size
        self.attention = DiverseSelfAttention(input_size=input_size)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.embedder = nn.Linear(in_features = self.linear_1.out_features,out_features = self.linear_1.out_features)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.norm_1 = nn.LayerNorm(normalized_shape=input_size)
        self.norm_2 = nn.LayerNorm(normalized_shape=input_size)
        self.mode = mode
        self.pos_enc = positional_encoding
    def forward(self,x):
        if self.pos_enc:
            seq_len = x.shape[1] # Changed as I want all the models to be batch size compatible. 
            x_pos = getPositionEncoding(seq_len,self.hidden_size)
            x = x + x_pos.to(x.device)
        if len(x.shape)>2:
            x = torch.squeeze(x)
        
        
        weighted_x, _ = self.attention(x) 
        # TODO: Check if there is a skip connection needed here
        y = self.norm_1(weighted_x)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_2(y)
        proj = self.embedder(y)
        out = self.sigmoid(self.drop(self.linear_2(y)))

        if self.mode == 'Unsupervised' and self.training:
            return proj,out.view(1,-1)
        else:
            return out.view(1,-1)