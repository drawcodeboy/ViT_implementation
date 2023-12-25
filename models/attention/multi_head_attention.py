'''
author: Doby
'''

from models.attention.scaled_dot_product_attention import ScaledDotProductAttention
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_length, qkv_dim, head_num):
        '''
        embedding_length : embedding 하나의 길이 -> W_(qkv)의 row 값이 된다.
        qkv_dim : W_(qkv) 행렬의 col 값
        head_num : ScaledDotProductAttention의 개수

        Output Shape : (Embedding 개수, QKV Matrix의 Col 길이)
        '''
        super().__init__()

        self.att_li = [ScaledDotProductAttention(embedding_length, qkv_dim) for _ in range(head_num)]
        self.W_o = nn.Parameter(torch.randn(qkv_dim * head_num, qkv_dim))

    def forward(self, x):
        att_weighted_values = []
        for att in self.att_li:
            att_weighted_values.append(att(x))

        head_atts = torch.concat(att_weighted_values, dim=-1)

        final_att = head_atts @ self.W_o
        
        return final_att
