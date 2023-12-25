'''
author: Doby
'''

import torch
from torch import nn
from torchinfo import summary
from models.embedding.patch_embedding import PatchEmbedding

class InputEmbedding(nn.Module):
    def __init__(self, channel, img_size, patch_size, d_dim):
        '''
        Input Shape : (channel, img_size, img_size)
        Output Shape : (embedding_size + 1, d_dim)
        embedding_size : patch로 나누었을 때, embedding의 수
        '''
        
        super().__init__()
        self.embedding_size = int((img_size*img_size)/(patch_size*patch_size))

        self.classEmbedding = nn.Parameter(torch.randn(1, d_dim, requires_grad=True))
        self.patchEmbedding = PatchEmbedding(channel, img_size, patch_size, d_dim)
        self.positionalEmbedding = nn.Parameter(torch.randn(self.embedding_size + 1, d_dim), requires_grad=True)

    def forward(self, x):
        x = self.patchEmbedding(x)

        embedding_li = []
        for i in range(x.shape[0]): # Batch의 수에 따라 
            # 각 Batch에 대해서 Class Embedding과 Positional Embedding 더하기
            # 이 때, Embedding은 모두 같다.
            x_ = torch.concat((self.classEmbedding, x[i]), dim=-2)
            x_ = x_ + self.positionalEmbedding
            embedding_li.append(x_.unsqueeze(dim=0))
            
        embeddings = torch.concat(embedding_li, dim=0)
        return embeddings