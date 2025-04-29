from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.functional import tanh

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init


# Z added this class
class LinearAttention(nn.Module):
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return F.elu(x) + 1  # Simple positive feature map

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads

        q = self.feature_map(self.to_q(x).reshape(B, N, H, self.head_dim))
        k = self.feature_map(self.to_k(x).reshape(B, N, H, self.head_dim))
        v = self.to_v(x).reshape(B, N, H, self.head_dim)

        k_sum = k.sum(dim=1)  # [B, H, D]
        D_inv = 1.0 / (torch.einsum('bnhd,bhd->bnh', q, k_sum) + 1e-6)

        context = torch.einsum('bnhd,bnhv->bhdv', k, v)
        out = torch.einsum('bnhd,bhdv,bnh->bnhv', q, context, D_inv).reshape(B, N, D)

        return self.to_out(out)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) Z comment this out
        self.self_attn = LinearAttention(d_model, nhead) # Z added this

        #self.norm = nn.LayerNorm(d_model)
        self.norm = DyT(d_model, init_alpha=0.5)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     #tgt_mask: Optional[Tensor] = None,
                     #padding_tgt_key_mask: Optional[Tensor] = None, Z removed these 2 lines
                     query_pos: Optional[Tensor] = None):
        #q = k = self.with_pos_embed(tgt, query_pos)
        #tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                      key_padding_mask=tgt_key_padding_mask)[0] Z comment these 3 lines

        x = self.with_pos_embed(tgt, query_pos) # Z added this
        tgt2 = self.self_attn(x) # Z added this

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    #tgt_mask: Optional[Tensor] = None,
                    #tgt_key_padding_mask: Optional[Tensor] = None, Z removed these 2 lines
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        #q = k = self.with_pos_embed(tgt2, query_pos)
        #tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                      key_padding_mask=tgt_key_padding_mask)[0] Z comment these 3 lines

        x = self.with_pos_embed(self.norm(tgt), query_pos) # Z added this
        tgt2 = self.self_attn(x) # Z added this

        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    #def forward(self, tgt,
    #            tgt_mask: Optional[Tensor] = None,
    #            tgt_key_padding_mask: Optional[Tensor] = None,
    #            query_pos: Optional[Tensor] = None): #Z comment this out
    def forward(self, tgt, query_pos: Optional[Tensor] = None): # Z added this
        #if self.normalize_before:
        #    return self.forward_pre(tgt, tgt_mask,
        #                            tgt_key_padding_mask, query_pos)
        #return self.forward_post(tgt, tgt_mask,
        #                         tgt_key_padding_mask, query_pos) Z comment these 5 lines

        if self.normalize_before:
             return self.forward_pre(tgt, query_pos)
        return self.forward_post(tgt, query_pos) # Z added these 3 lines


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # Z comment this out
        self.self_attn = LinearAttention(d_model, nhead) # Z added this

        #self.norm = nn.LayerNorm(d_model)
        self.norm = DyT(d_model, init_alpha=0.5) 
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # def forward_post(self, tgt, memory,
    #                  memory_mask: Optional[Tensor] = None,
    #                  memory_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None,
    #                  query_pos: Optional[Tensor] = None):
    #     tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
    #                                key=self.with_pos_embed(memory, pos),
    #                                value=memory, attn_mask=memory_mask,
    #                                key_padding_mask=memory_key_padding_mask)
    #     tgt = tgt + self.dropout(tgt2)
    #     tgt = self.norm(tgt)
    #     return tgt, avg_attn # Z comment forward_post out

    def forward_post(self, tgt, query_pos: Optional[Tensor] = None):
        x = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(x)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, None  # Z added this new forward_post

    # def forward_pre(self, tgt, memory,
    #                 memory_mask: Optional[Tensor] = None,
    #                 memory_key_padding_mask: Optional[Tensor] = None,
    #                 pos: Optional[Tensor] = None,
    #                 query_pos: Optional[Tensor] = None):
    #     tgt2 = self.norm(tgt)
    #     tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
    #                                key=self.with_pos_embed(memory, pos),
    #                                value=memory, attn_mask=memory_mask,
    #                                key_padding_mask=memory_key_padding_mask)
    #     tgt = tgt + self.dropout(tgt2)

    #     return tgt, avg_attn # Z comment forward_pre out
    
    def forward_pre(self, tgt, query_pos: Optional[Tensor] = None):
        x = self.with_pos_embed(self.norm(tgt), query_pos)
        tgt2 = self.self_attn(x)
        tgt = tgt + self.dropout(tgt2)
        return tgt, None # Z added this new forward_pre

    # def forward(self, tgt, memory,
    #             memory_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None,
    #             query_pos: Optional[Tensor] = None):
    #     if self.normalize_before:
    #         return self.forward_pre(tgt, memory, memory_mask,
    #                                 memory_key_padding_mask, pos, query_pos)
    #     return self.forward_post(tgt, memory, memory_mask,
    #                              memory_key_padding_mask, pos, query_pos) # Z comment this forward out

    def forward(self, tgt, query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, query_pos)
        return self.forward_post(tgt, query_pos) # Z added this new forward

class DyT(nn.Module):
    def __init__(self, c, init_alpha):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.r = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        x = tanh(self.alpha * x)
        return self.r * x + self.beta

# Z added this class
class GatedAttentionUnit(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = DyT(d_model, init_alpha=0.5)
        #self.norm = nn.LayerNorm(d_model)
        self.linear_g = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.linear_g(x_norm))
        value = self.linear_v(x_norm)
        x = x + self.dropout(self.linear_out(gate * value))
        return x

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward) #Z comment this out
        self.linear_v = nn.Linear(d_model, dim_feedforward) # Z added this
        self.linear_g = nn.Linear(d_model, dim_feedforward) # Z added this 
        
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm = nn.LayerNorm(d_model)
        self.norm = DyT(d_model, init_alpha=0.5)  # Replaced nn.LayerNorm with DyT
        
        # self.activation = _get_activation_fn(activation) # Z comment this out, won't use activation
        self.normalize_before = normalize_before

        self._reset_parameters()
   
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt):
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) #Z comment this out 
        tgt2 = self.linear2(self.dropout(F.silu(self.linear_g(tgt)) * self.linear_v(tgt))) # Z added this
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2)))) #Z comment this out 
        tgt2 = self.linear2(self.dropout(F.silu(self.linear_g(tgt2)) * self.linear_v(tgt2))) # Z added this
        tgt = tgt + self.dropout(tgt2)
        return tgt


    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x