'''
author: Doby
'''

import torch
from torch import nn
from torchinfo import summary
from models.embedding.patch_embedding import PatchEmbedding
from models.embedding.input_embedding import InputEmbedding
from models.transformer_encoder.transformer_encoder import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, channel, img_size, patch_size, d_dim, qkv_dim, att_head_num, mlp_hidden_dim, n_transform_layers, mlp_head_hidden_dim, n_classes):
        '''
        channel : 입력 이미지 Channel
        img_size : 입력 이미지 Height or Width
        patch_size : 나눌 Patch의 사이즈
        d_dim : embedding을 Linear Projection 했을 때, Dimension
        qkv_dim : attention 할 때, QKV Matrix의 Col 값 (d_dim과 동일하게 해야하는 것으로 추측)
        att_head_num : 몇 번의 attention을 가지고 Multi-Head를 할 것인가
        mlp_hidden_dim : transformer encoder의 MLP에 사용되는 Hidden Layer의 Dimension
        n_transform_layers : transformer encoder 수
        mlp_head_hidden_dim : MLP Head의 Hidden Layer Dimension
        n_classes : 분류하고자 하는 class의 수
        '''
        super().__init__()
        self.embedding_size = ((img_size*img_size)/(patch_size*patch_size))+1

        self.embedding = InputEmbedding(channel, img_size, patch_size, d_dim)
        self.layers = nn.ModuleList([TransformerEncoder(self.embedding_size, 
                                                        d_dim, 
                                                        qkv_dim, 
                                                        att_head_num, 
                                                        mlp_hidden_dim) for _ in range(n_transform_layers)])
        self.MLP_Head = nn.Sequential(nn.Linear(qkv_dim, mlp_head_hidden_dim),
                                      nn.GELU(),
                                      nn.Linear(mlp_head_hidden_dim, n_classes))
        if n_classes == 2:
            self.acti = nn.Sigmoid()
        else:
            self.acti = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(f'input shape\n{x.shape}\n')
        
        x = self.embedding(x)
        # print(f'after embedding shape\n{x.shape}\n')
        
        for transformer in self.layers:
            x = transformer(x)
        # print(f'after transformer shape\n{x.shape}\n')
        
        x = x[:, 0, :]
        # print(f'z_0^L shape\n{x.shape}\n')
        
        x = self.MLP_Head(x)
        # print(f'last MLP Head for Classification shape\n{x.shape}\n')
        
        return x