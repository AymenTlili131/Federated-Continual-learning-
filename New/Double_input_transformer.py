from itertools import product
from itertools import combinations



import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, ConcatDataset

import random
import ast
        

import torch.nn.functional as F
#from torcheval.metrics.functional import multiclass_confusion_matrix
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision import transforms

import optuna

from itertools import product
from itertools import combinations


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation ,FFMpegWriter ,PillowWriter

import dill
import numpy as np

import json
import pandas as pd
from tqdm import tqdm
import copy
import os 
import random

from collections import OrderedDict
def batchify(lst, batch_size, batch_limit=100):
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    return batches[:batch_limit]

class CustomDataset(TensorDataset):
    def __init__(self, m, epoch_key, activ_key, batch_size=20, batch_limit=5, df_path="./data/Merged zoo.csv", data_type="train"):
        self.m = m  # Overlap value
        self.epoch_key = epoch_key
        self.activ_key = activ_key
        self.batch_size = batch_size
        self.batch_limit = batch_limit
        self.data_type = data_type  # 'train', 'val', or 'test'
        self.D_activ = {'0': "gelu", '1': "relu", '2': "silu", '3': "leakyrelu", '4': "sigmoid", '5': "tanh"}
        activ_str = self.D_activ[str(activ_key)]
        results_path = f"./data/Scenario/overlapping_m{m}_epoch{epoch_key}_activ{activ_key}/{data_type}_batches.pt"
        
        # Load pre-saved batch data
        data = torch.load(results_path)
        self.loaded_list = data["loaded_list"]  # List of stacked tensors [Stream1, Stream2, target]
        self.L_ACC_list = data["L_ACC_list"]    # List of accuracy metadata
        self.L_indexes_list = data["L_indexes_list"]  # List of indexes metadata
        
        # Batchify if necessary, but since batches are pre-saved, use existing structure
        self.batches = self.loaded_list  # Assume loaded_list is already a list of batches
        
    def __len__(self):
        return len(self.batches)
        
    def __getitem__(self, idx):
        loaded_batch = self.batches[idx]
        acc_batch = self.L_ACC_list[idx]
        indexes_batch = self.L_indexes_list[idx]
        return loaded_batch, acc_batch, indexes_batch
    
    
    
import torch
import torch.nn as nn
import numpy as np



import itertools
from einops import repeat

import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from collections import OrderedDict
import copy
import numpy as np

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

import random
import ast
        

# # Transformer Shared Layers


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80,device='cuda'): #d"
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.device = device
        # create constant 'pe' matrix with values dependant on
        # pos and i
        self.pe = self._generate_positional_encoding(max_seq_len, d_model)
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        # dynamically adjust positional encoding matrix based on sequence length
        pe = self.pe[:, :seq_len]
        pe=pe.to(self.device)
        x=x.to(self.device)
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
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores

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
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, sc = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, sc

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
        # x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x = x + self.dropout_1(res)
        if self.normalize:
            x2 = self.norm_2(x)
        else:
            x2 = x.clone()
        x = x + self.dropout_2(self.ff(x2))
        return x, sc# return x
    
    
class EmbedderNeuronGroup(nn.Module):
    def __init__(self, d_model, seed=22):
        super().__init__()
        #print("EmbedderNeuroneGroup")
        self.neuron_l1 = nn.Linear(16, d_model) #24
        self.neuron_l2 = nn.Linear(80, d_model) #26

    def forward(self, x):
        return self.multiLinear(x)
    def multiLinear(self, v):
        #print("multi-linear method",v.shape)
        l = []
        for ndx in range(26):
            idx_start = ndx *80 
            idx_end = idx_start + 80
            l.append(self.neuron_l2(v[:,idx_start:idx_end]).clone())
        # l2
        for ndx in range(24):
            idx_start = 26*80 + ndx * 16
            idx_end = idx_start + 16
            l.append(self.neuron_l1(v[:,idx_start:idx_end]).clone())
        #print(len(l))
        #print(len(l[0]))
        final = torch.stack(l, dim=1)
        # print(final.shape)
        return final
class EncoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.embed = EmbedderNeuronGroup(d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        print("encoder droupout init",dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, normalize=True,dropout=dropout, d_ff=d_ff), N)
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

        # Define linear layers
        self.linear = nn.Linear(max_seq_len * d_model, 2464)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the sequence dimension
        x = self.linear(x)
        return x
class Neck2Seq(nn.Module):
    def __init__(self, d_model, neck,max_seq_length):
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
        self.embed = Neck2Seq(d_model, neck,max_seq_len)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        print("decoder droupout init",dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads,normalize=True,dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)
        self.lay = Seq2Vec(d_model=d_model,max_seq_len=max_seq_len)
    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.lay(self.norm(x)), scores
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = (
            self.alpha* (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias)
        return norm
class TransformerAE(nn.Module):
    def __init__(
        self,
        max_seq_len=10,
        N=1,
        heads=1,
        d_model=100,
        d_ff=100,
        neck=20,
        dropout=0.1,
        **kwargs,):
        super().__init__()
        self.N=N
        self.heads=heads
        self.dropout=dropout
        self.d_ff=d_ff
        self.d_model=d_model
        self.max_seq_len=max_seq_len
        self.neck=neck
        
        self.enc1 = EncoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads, max_seq_len=self.max_seq_len, dropout=self.dropout,d_ff=self.d_ff)
        self.enc2 = EncoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads, max_seq_len=self.max_seq_len, dropout=self.dropout,d_ff=self.d_ff)
        self.dec = DecoderNeuronGroup(d_model=self.d_model, N=self.N, heads=self.heads, max_seq_len=self.max_seq_len, dropout=self.dropout,d_ff=self.d_ff,neck=self.neck)
        #print("Addition Approach!")
        self.vec2neck = nn.Linear(self.d_model*2, self.neck)  # Fixed: concatenated encoders output d_model*2, not d_ff*2
        #print("Stack Approach!")
        #self.vec2neck = nn.Linear(2*self.d_ff * self.max_seq_len, self.neck)
        self.tanh = nn.Tanh()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, inp1,inp2):
        out1, scEnc1 = self.enc1(inp1) #z1 and attentions score
        out2, scEnc2 = self.enc2(inp2) #z2 and attention score
        out3=torch.cat([out1,out2], dim=2) #concatenation to get dimentson n_output=n_input*2
        sum_r=torch.sum(out3, dim=1, keepdim=False)
        vec2=self.vec2neck(sum_r) #fusion dense layer
        tanh = nn.Tanh()
        neck_t=tanh(vec2)
        out, scDec = self.dec(neck_t) #predicted weights , decoder attention scores
        return out, neck_t, scEnc1,scEnc2, scDec

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def numParams(self):
        encNumParams = self.count_parameters(self.enc1)
        neckNumParams = self.count_parameters(self.vec2neck)
        decNumParams = self.count_parameters(self.dec)
        modelParams = self.count_parameters(self)

        return (
            "EncParams: {}, NeckParams: {}, DecParams: {}, || ModelParams: {} ".format(
                encNumParams, neckNumParams, decNumParams, modelParams
            )
        )
