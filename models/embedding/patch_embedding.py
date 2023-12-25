'''
author: Doby
'''

import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, channel, img_size, patch_size, d_dim):
        super().__init__()
        # img_size: 정사각 이미지인 것으로 정한다
        # patch_size: 원하는 patch size
        # d_dim: Linear Projection을 했을 때, 원하는 차원의 크기

        self.img_size = img_size
        self.channel = channel
            
        self.patch_size = patch_size
        
        self.n_patches = int((img_size * img_size) / (patch_size * patch_size))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
        self.linear_projs = nn.ModuleList([nn.Linear(patch_size*patch_size*channel, d_dim) for _ in range(self.n_patches)])

    def crop_img(self, x):
        patches = []
        row = int(math.sqrt(self.n_patches))
        col = int(math.sqrt(self.n_patches))
        p_len = self.patch_size
        
        
        for i in range(row):
            for j in range(col):
                temp = x[:, :, i*p_len:(i+1)*p_len, j*p_len:(j+1)*p_len]
                patches.append(temp)
                # one patch에 대한 시작점 (y, x)와 끝점 (y, x) 조사
                # print(f'{i*p_len} {(i+1)*p_len} {j*p_len} {(j+1)*p_len}')
                
        return patches
        
    def forward(self, x):
        # Patch 생성
        patches = self.crop_img(x)

        # Flatten
        patches_flatten = [self.flatten(patches[i]).unsqueeze(dim=-2) for i in range(self.n_patches)]
        # print(patches[0].shape)
        # print(patches_flatten[0].shape)

        # Linear Projection
        embedding_li = [self.linear_projs[i](patches_flatten[i]) for i in range(self.n_patches)]
        # print(embedding_li[0].shape)

        # 모든 embedding concatenate
        embeddings = torch.concat(embedding_li, dim=-2)

        return embeddings