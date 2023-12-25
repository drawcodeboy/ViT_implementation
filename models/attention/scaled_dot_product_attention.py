'''
author: Doby
'''

import torch
from torch import nn
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_length, qkv_vec_length):
        '''
        embedding_length : embedding 하나의 길이 -> W_(qkv)의 row 값이 된다.
        qkv_vec_length :  W_(qkv) 행렬의 col 값
        '''
        super().__init__()

        # Query Matrix (Weight)
        self.W_q = nn.Parameter(torch.randn(embedding_length, qkv_vec_length, requires_grad=True))
        
        # Key Matrix (Weight)
        self.W_k = nn.Parameter(torch.randn(embedding_length, qkv_vec_length, requires_grad=True))
        
        # Value Matrix (Weight)
        self.W_v = nn.Parameter(torch.randn(embedding_length, qkv_vec_length, requires_grad=True))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        attn_scores = Q @ torch.transpose(K, -2, -1)

        attn_scores_softmax = self.softmax(attn_scores / math.sqrt(K.shape[-1]))

        weighted_values = attn_scores_softmax @ V
        
        return weighted_values