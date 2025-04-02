import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tensor


# Define DyT directly here
class DyT(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6) * self.scale + self.bias

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm = DyT(d_model)  # 🔁 Replaced nn.LayerNorm with DyT


        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


        self._reset_parameters()
   
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
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
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
