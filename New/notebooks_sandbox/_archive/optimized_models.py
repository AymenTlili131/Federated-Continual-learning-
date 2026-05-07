"""
Optimized TransformerAE variants for faster training
Includes tiny, small, medium, large models with 500K-25M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple, List


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, device='cuda'):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.device = device
        self.pe = self._generate_positional_encoding(max_seq_len, d_model)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        pe = pe.to(self.device)
        x = x.to(self.device)
        x = x + pe
        return x
    
    def _generate_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        
        return output, scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))  # Use GELU instead of ReLU
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, normalize=True, dropout=0.1, d_ff=2048):
        super().__init__()
        self.normalize = normalize
        if normalize:
            self.norm_1 = Norm(d_model)
            self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        if self.normalize:
            x2 = self.norm_1(x)
        else:
            x2 = x.clone()
        res, sc = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(res)
        if self.normalize:
            x2 = self.norm_2(x)
        else:
            x2 = x.clone()
        x = x + self.dropout_2(self.ff(x2))
        return x, sc


class EmbedderNeuronGroup(nn.Module):
    """Efficient embedding for weight sequences"""
    def __init__(self, d_model, seed=22):
        super().__init__()
        self.neuron_l1 = nn.Linear(16, d_model)
        self.neuron_l2 = nn.Linear(80, d_model)
    
    def forward(self, x):
        return self.multiLinear(x)
    
    def multiLinear(self, v):
        l = []
        for ndx in range(26):
            idx_start = ndx * 80
            idx_end = idx_start + 80
            l.append(self.neuron_l2(v[:, idx_start:idx_end]).clone())
        
        for ndx in range(24):
            idx_start = 26*80 + ndx * 16
            idx_end = idx_start + 16
            l.append(self.neuron_l1(v[:, idx_start:idx_end]).clone())
        
        final = torch.stack(l, dim=1)
        return final


class EncoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.embed = EmbedderNeuronGroup(d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, normalize=True, dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            self.layers[i] = self.layers[i].to(self.device)
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.norm(x), scores


class Seq2Vec(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.linear = nn.Linear(max_seq_len * d_model, 2464)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


class Neck2Seq(nn.Module):
    def __init__(self, d_model, neck, max_seq_length):
        super().__init__()
        self.neurons = nn.ModuleList([nn.Linear(neck, d_model) for _ in range(max_seq_length)])
    
    def forward(self, x):
        l = [neuron(x) for neuron in self.neurons]
        final = torch.stack(l, dim=1)
        return final


class DecoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff, neck):
        super().__init__()
        self.N = N
        self.embed = Neck2Seq(d_model, neck, max_seq_len)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, normalize=True, dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)
        self.lay = Seq2Vec(d_model=d_model, max_seq_len=max_seq_len)
    
    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.lay(self.norm(x)), scores


class OptimizedTransformerAE(nn.Module):
    """
    Optimized TransformerAE with configurable size
    Supports gradient checkpointing and mixed precision training
    """
    def __init__(
        self,
        max_seq_len=50,
        N=2,
        heads=4,
        d_model=128,
        d_ff=512,
        neck=64,
        dropout=0.1,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.heads = heads
        self.dropout = dropout
        self.d_ff = d_ff
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.neck = neck
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.enc1 = EncoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads, 
                                       max_seq_len=self.max_seq_len, dropout=self.dropout, d_ff=self.d_ff)
        self.enc2 = EncoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads,
                                       max_seq_len=self.max_seq_len, dropout=self.dropout, d_ff=self.d_ff)
        self.dec = DecoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads,
                                      max_seq_len=self.max_seq_len, dropout=self.dropout, d_ff=self.d_ff, neck=self.neck)
        
        self.vec2neck = nn.Linear(self.d_ff * 2, self.neck)
        self.tanh = nn.Tanh()
        
        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, inp1, inp2):
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(inp1, inp2)
        else:
            return self._forward_normal(inp1, inp2)
    
    def _forward_normal(self, inp1, inp2):
        out1, scEnc1 = self.enc1(inp1)
        out2, scEnc2 = self.enc2(inp2)
        out3 = torch.cat([out1, out2], dim=2)
        sum_r = torch.sum(out3, dim=1, keepdim=False)
        vec2 = self.vec2neck(sum_r)
        neck_t = self.tanh(vec2)
        out, scDec = self.dec(neck_t)
        return out, neck_t, scEnc1, scEnc2, scDec
    
    def _forward_with_checkpointing(self, inp1, inp2):
        """Use gradient checkpointing to save memory"""
        from torch.utils.checkpoint import checkpoint
        
        out1, scEnc1 = checkpoint(self.enc1, inp1)
        out2, scEnc2 = checkpoint(self.enc2, inp2)
        out3 = torch.cat([out1, out2], dim=2)
        sum_r = torch.sum(out3, dim=1, keepdim=False)
        vec2 = self.vec2neck(sum_r)
        neck_t = self.tanh(vec2)
        out, scDec = checkpoint(self.dec, neck_t)
        return out, neck_t, scEnc1, scEnc2, scDec
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def numParams(self):
        encNumParams = sum(p.numel() for p in self.enc1.parameters() if p.requires_grad)
        neckNumParams = sum(p.numel() for p in self.vec2neck.parameters() if p.requires_grad)
        decNumParams = sum(p.numel() for p in self.dec.parameters() if p.requires_grad)
        modelParams = self.count_parameters()
        
        return (
            f"EncParams: {encNumParams:,}, NeckParams: {neckNumParams:,}, "
            f"DecParams: {decNumParams:,}, || ModelParams: {modelParams:,}"
        )


def create_model_from_config(config):
    """
    Create model from configuration object
    
    Args:
        config: ModelConfig from config.py
    
    Returns:
        OptimizedTransformerAE instance
    """
    return OptimizedTransformerAE(
        max_seq_len=config.max_seq_len,
        N=config.N,
        heads=config.heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        neck=config.neck,
        dropout=config.dropout
    )


if __name__ == "__main__":
    print("Testing Optimized TransformerAE Models")
    print("=" * 60)
    
    from config import MODEL_CONFIGS
    
    # Test each model size
    for name, config in MODEL_CONFIGS.items():
        print(f"\nTesting {name} model:")
        model = create_model_from_config(config)
        print(f"  {model.numParams()}")
        
        # Test forward pass
        batch_size = 2
        inp1 = torch.randn(batch_size, 2464)
        inp2 = torch.randn(batch_size, 2464)
        
        if torch.cuda.is_available():
            inp1 = inp1.cuda()
            inp2 = inp2.cuda()
        
        with torch.no_grad():
            out, neck, scEnc1, scEnc2, scDec = model(inp1, inp2)
        
        print(f"  Output shape: {out.shape}")
        print(f"  Neck shape: {neck.shape}")
        print(f"  ✓ Forward pass successful")
