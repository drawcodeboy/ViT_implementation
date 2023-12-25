'''
author: Doby
'''

import torch
from torch import nn
from models.attention.multi_head_attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, d_dim, qkv_dim, head_num, hidden_dim):
        '''
        embedding_size : 임베딩의 개수
        d_dim : 현재 임베딩의 차원 수
        qkv_dim : attention 후에 임베딩의 차원 수 (현재 같아야 하는 것으로 추측)
        head_num : Multi-Head Self-Attention에서 사용되는 Attention의 개수
        hidden_dim : MLP에 사용되는 Hidden Layer의 Dimension
        n_transformers : transformer layer의 수
        
        Input Shape : (embedding_size, d_dim)
        Multi-Head Self-Attention Shape : (embedding_size, qkv_dim)
        qkv_dim과 d_dim이 같아야 residual connection이 가능하다.
        나의 오류일 수도 있으니 d_dim, qkv_dim 인자는 우선 구분해두자.

        -> Input Shape, Output Shape가 같다.
        '''
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_dim)
        self.mhatt = MultiHeadAttention(d_dim, qkv_dim, head_num)

        self.layer_norm2 = nn.LayerNorm(qkv_dim)
        self.MLP = nn.Sequential(nn.Linear(qkv_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, qkv_dim))

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.mhatt(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.MLP(x)
        x = x + residual

        return x