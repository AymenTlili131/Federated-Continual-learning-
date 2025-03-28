import torch
import torch.nn as nn
import numpy as np


from torch.utils.tensorboard import SummaryWriter
import itertools
from einops import repeat

from einops import repeat

import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


# # Transformer Shared Layers


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, device="cpu"):
        super().__init__()
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
    '''   
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # get the sequence length of the input tensor
        seq_len = x.size(1)
        # create positional encoding matrix dynamically
        pe = self.pe[:, :seq_len]
        # add positional encoding to the input tensor
        x = x + pe
        return x
  
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x
    '''


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
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


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
        # return x
        return x, sc


class Encoder(nn.Module):
    def __init__(
        self, input_dim, d_model, N, heads, max_seq_len, dropout, d_ff, device
    ):
        super().__init__()
        self.device = device
        self.N = N
        self.embed = Embedder(input_dim, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len, device)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.norm(x), scores


class Embedder(nn.Module):
    def __init__(self, input_dim, embed_dim, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, x):
        y = []
        # use the same embedder to embedd all weights
        for idx in range(self.input_dim):
            # embedd single input / feature dimension
            tmp = self.embed(x[:, idx].unsqueeze(dim=1))
            y.append(tmp)
        # stack along dimension 1
        y = torch.stack(y, dim=1)
        return y


class Debedder(nn.Module):
    def __init__(self, input_dim, d_model, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.weight_debedder = nn.Linear(d_model, 1)

    def forward(self, x):
        y = self.weight_debedder(x)
        y = y.squeeze()
        return y


# # Tranformer Encoder
class EmbedderNeuron(nn.Module):
    # collects all weights connected to one neuron / kernel
    def __init__(self, index_dict, d_model, seed=22):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict
        self.get_kernel_slices()

        for idx, kernel_lst in enumerate(self.slice_lst):
            i_dim = len(kernel_lst[0])
            # check sanity of slices
            for slice in kernel_lst:
                assert (
                    len(slice) == i_dim
                ), f"layer-wise slices are not of the same lenght: {i_dim} vs {len(slice)}"
            # get layers
            self.layer_lst.append(nn.Linear(i_dim, d_model))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                # get current kernel index
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                # get next_layers connected weights
                if idx < len(self.index_dict["layer"]) - 1:
                    # -> find corresponding indices
                    # -> get offset to beginning of next layer
                    for kernel_dx_next in range(self.index_dict["kernel_no"][idx + 1]):
                        kernel_next_start = (
                            # get start of next layer
                            self.index_dict["idx_start"][idx + 1]
                            # offset by current kernel*dim of kernel_size (columns)
                            + kernel_dx * self.index_dict["kernel_size"][idx + 1]
                            # offset by rows: overall parameters per channel out
                            + kernel_dx_next
                            * self.index_dict["channels_in"][idx + 1]
                            * self.index_dict["kernel_size"][idx + 1]
                        )
                        kernel_next_end = (
                            kernel_next_start + self.index_dict["kernel_size"][idx + 1]
                        )

                        # extend
                        kernel_next_idx = list(
                            range(kernel_next_start, kernel_next_end)
                        )
                        index_kernel.extend(kernel_next_idx)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(
        self,
    ):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        y_lst = []
        # loop over layers
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                y_tmp = self.layer_lst[idx](x[:, kernel_index])
                y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        return y


class DebedderNeuron(nn.Module):
    def __init__(self, index_dict, d_model, seed=22, layers=1, dropout=0.1):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict
        self.get_kernel_slices()

        for idx, kernel_lst in enumerate(self.slice_lst):
            i_dim = len(kernel_lst[0])
            # check sanity of slices
            for slice in kernel_lst:
                assert (
                    len(slice) == i_dim
                ), f"layer-wise slices are not of the same lenght: {i_dim} vs {len(slice)}"
            # get layers
            if layers == 1:
                self.layer_lst.append(nn.Linear(d_model, i_dim))
            else:
                from model_definitions.def_net import MLP

                layertmp = MLP(
                    i_dim=d_model,
                    h_dim=[d_model for _ in range(layers - 2)],
                    o_dim=i_dim,
                    nlin="leakyrelu",
                    dropout=dropout,
                    init_type="kaiming_normal",
                    use_bias=True,
                )
                self.layer_lst.append(layertmp)
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")
            # self.layer_lst.append(nn.Linear(d_model, i_dim))

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                # get current kernel index
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                # get next_layers connected weights
                if idx < len(self.index_dict["layer"]) - 1:
                    # -> find corresponding indices
                    # -> get offset to beginning of next layer
                    for kernel_dx_next in range(self.index_dict["kernel_no"][idx + 1]):
                        kernel_next_start = (
                            # get start of next layer
                            self.index_dict["idx_start"][idx + 1]
                            # offset by current kernel*dim of kernel_size (columns)
                            + kernel_dx * self.index_dict["kernel_size"][idx + 1]
                            # offset by rows: overall parameters per channel out
                            + kernel_dx_next
                            * self.index_dict["channels_in"][idx + 1]
                            * self.index_dict["kernel_size"][idx + 1]
                        )
                        kernel_next_end = (
                            kernel_next_start + self.index_dict["kernel_size"][idx + 1]
                        )

                        # extend
                        kernel_next_idx = list(
                            range(kernel_next_start, kernel_next_end)
                        )
                        index_kernel.extend(kernel_next_idx)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(
        self,
    ):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        device = x.device
        # get last value of last layer last kernel last index - zero based -> +1
        i_dim = self.slice_lst[-1][-1][-1] + 1
        y = torch.zeros((x.shape[0], i_dim)).to(device)

        # loop over layers
        embed_dx = 0
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                # get values for this embedding
                y_tmp = self.layer_lst[idx](x[:, embed_dx])
                # !!add!! values in right places
                y[:, kernel_index] += y_tmp
                # raise counter
                embed_dx += 1

        # first layer and last layer get only embedded once,
        # while all middle layers overlap.
        # -> get index list for beginning of second and ending of second to last layer
        # -> devide embedded values by 2
        if len(self.index_dict["idx_start"]) > 2:
            index_start = self.index_dict["idx_start"][1]
            index_end = self.index_dict["idx_start"][-1]
            idx = list(range(index_start, index_end))
            # create tensor of same shape with 0.5 values put it on device
            factor = torch.ones(y[:, idx].shape) * 0.5
            factor = factor.to(y.device)
            # multiply with 0.5
            y[:, idx] = y[:, idx] * factor
        return y


# # Tranformer Encoder
class EmbedderNeuronGroup_index(nn.Module):
    def __init__(self, index_dict, d_model, seed=22, split_kernels_threshold=0):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict

        self.split_kernels_threshold = split_kernels_threshold

        for idx, layer in enumerate(index_dict["layer"]):
            i_dim = index_dict["kernel_size"][idx] * index_dict["channels_in"][idx] + 1
            if (self.split_kernels_threshold != 0) and (
                i_dim > self.split_kernels_threshold
            ):
                i_dim = self.split_kernels_threshold
            self.layer_lst.append(nn.Linear(i_dim, d_model))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

        self.get_kernel_slices()

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(
        self,
    ):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        y_lst = []
        # loop over layers
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                if (self.split_kernels_threshold != 0) and (
                    len(kernel_index) > self.split_kernels_threshold
                ):
                    from math import ceil

                    no_tokens = ceil(len(kernel_index) / self.split_kernels_threshold)
                    for idx in range(no_tokens):
                        idx_token_start = idx * self.split_kernels_threshold
                        idx_token_end = idx_token_start + self.split_kernels_threshold
                        kernel_tmp = kernel_index[idx_token_start:idx_token_end]
                        if idx == no_tokens - 1:  # last
                            x_tmp = torch.zeros(
                                size=[x.shape[0], self.split_kernels_threshold]
                            )  # pad
                            x_tmp[:, : len(kernel_index)] = x[:, kernel_tmp]
                        else:
                            x_tmp = x[:, kernel_tmp]
                        y_tmp = self.layer_lst[idx](x_tmp)
                        y_lst.append(y_tmp)
                else:
                    y_tmp = self.layer_lst[idx](x[:, kernel_index])
                    y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        return y



class EmbedderNeuronGroup(nn.Module):
    def __init__(self, d_model, seed=22):
        super().__init__()
        print("EmbedderNeuroneGroup")
        self.neuron_l1 = nn.Linear(16, d_model)
        self.neuron_l2 = nn.Linear(5, d_model)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        #print(v.shape)
        # Hardcoded position for easy-fast integration
        l = []
        # l1
        if v.shape[1]==100:
            for ndx in range(5):
                idx_start = ndx * 16
                idx_end = idx_start + 16
                l.append(self.neuron_l1(v[:, idx_start:idx_end]))
                
            # l2
            for ndx in range(4):
                idx_start = 5 * 16 + ndx * 5
                idx_end = idx_start + 5
                l.append(self.neuron_l2(v[:, idx_start:idx_end]))

        if v.shape[1]==2464:
            for g in range(25):
                if g==24:
                    for ndx in range(4):
                        idx_start = g*100 + ndx * 16
                        idx_end = idx_start + 16
                        l.append(self.neuron_l1(v[:, idx_start:idx_end]))
                        #print(v[:, idx_start:idx_end].shape)
                else:     
                    for ndx in range(5):
                        idx_start =g*100 + ndx * 16
                        idx_end = idx_start + 16
                        l.append(self.neuron_l1(v[:, idx_start:idx_end]))
                        #print(v[:, idx_start:idx_end].shape)
                    # l2
                    for ndx in range(4):
                        idx_start = g*100 + 5 * 16 + ndx * 5
                        idx_end = idx_start + 5
                        l.append(self.neuron_l2(v[:, idx_start:idx_end]))
                        #print(v[:, idx_start:idx_end].shape)
        #print(len(l))
        #print(len(l[0]))
        final = torch.stack(l, dim=1)

        # print(final.shape)
        return final

class DebedderNeuronGroup(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.linear1 = nn.Linear(d_model, 16)
        self.linear2 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, x):
        segments = []
        for i in range(x.size(1)):
            if i < self.max_seq_len // 2:
                segment = self.linear1(x[:, i])
            else:
                segment = self.linear2(x[:, i])
            segments.append(segment)

        reconstructed = torch.cat(segments, dim=1)
        return reconstructed

class Seq2Vec(nn.Module):
    def __init__(self, d_model, max_seq_len, num_neurons=20):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.neurons = nn.ModuleList([
            nn.Linear(d_model, 16) if i < num_neurons // 2 else nn.Linear(d_model, 5)
            for i in range(num_neurons)
        ])
        
class DebedderNeuronGroup(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 16)
        self.linear2 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, x):
        segments = []
        # Assuming x is of shape (batch_size, num_segments, d_model)
        for i in range(x.size(1)):
            if i < 480:
                segment = self.linear1(x[:, i])
            else:
                segment = self.linear2(x[:, i])
            segments.append(segment)

        # Concatenate segments along the second dimension
        reconstructed = torch.cat(segments, dim=1)
        return reconstructed
    
class Neck2Seq(nn.Module):
    def __init__(self, d_model, neck):
        super().__init__()

        self.neurons = nn.ModuleList([nn.Linear(neck, d_model) for _ in range(9)])

    def forward(self, x):
        l = [neuron(x) for neuron in self.neurons]
        final = torch.stack(l, dim=1)
        return final


class Seq2Vec(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.num_neurons = 20  # Number of linear transformations
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Define linear layers
        self.neurons = nn.ModuleList([
            nn.Linear(d_model, 16) if i < 5 else nn.Linear(d_model, 5)
            for i in range(self.num_neurons)
        ])

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        l = []
        for i in range(self.max_seq_len):
            # Apply the appropriate linear transformation
            neuron_output = self.neurons[i % self.num_neurons](v[:, i])
            l.append(neuron_output)
        
        final = torch.cat(l, dim=1)  # Concatenate tensors along the feature dimension (dim=1)
        return final
'''    
class Seq2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.neuron11 = nn.Linear(d_model, 16)
        self.neuron12 = nn.Linear(d_model, 16)
        self.neuron13 = nn.Linear(d_model, 16)
        self.neuron14 = nn.Linear(d_model, 16)
        self.neuron15 = nn.Linear(d_model, 16)
        self.neuron21 = nn.Linear(d_model, 5)
        self.neuron22 = nn.Linear(d_model, 5)
        self.neuron23 = nn.Linear(d_model, 5)
        self.neuron24 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        l = []
        l.append(self.neuron11(v[:, 0]))
        l.append(self.neuron12(v[:, 1]))
        l.append(self.neuron13(v[:, 2]))
        l.append(self.neuron14(v[:, 3]))
        l.append(self.neuron15(v[:, 4]))
        l.append(self.neuron21(v[:, 5]))
        l.append(self.neuron22(v[:, 6]))
        l.append(self.neuron23(v[:, 7]))
        l.append(self.neuron24(v[:, 8]))
        final = torch.cat(l, dim=1)

        # print(final.shape)
        return final
'''
class EncoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff):
        super().__init__()
        self.N = N
        self.embed = EmbedderNeuronGroup(d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        print("decoder droupout init",dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, normalize=True,dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        #print("scores variable shape is:",scores[0][0].shape," norm variable shape is:",self.norm(x).shape)
        #print(scores[0][0])
        return self.norm(x), scores

class DecoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff, neck):
        super().__init__()
        self.N = N
        self.embed = Neck2Seq(d_model, neck)
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




class TransformerAE(nn.Module):
    def __init__(
        self,
        max_seq_len=9,
        N=1,
        heads=1,
        d_model=100,
        d_ff=100,
        neck=20,
        dropout=0.1,
        **kwargs,
    ):

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
        # Addition Approach
        #print("Addition Approach!")
        self.vec2neck = nn.Linear(self.d_ff, self.neck)

        # Stacking Approach
        #print("Stack Approach!")
        #self.vec2neck = nn.Linear(2*self.d_ff * self.max_seq_len, self.neck)

        self.tanh = nn.Tanh()

        # Xavier Uniform Initialitzation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inp1,inp2):

        # First Approach
        out1, scEnc1 = self.enc1(inp1)
        #print(out1.shape)
        out2, scEnc2 = self.enc2(inp2)
        out3=torch.cat([out1,out2], dim=2)
        #print("out3",out3.shape,"d_ff *2",self.d_ff*2)
        #print("dim 0",torch.sum(out3, dim=0, keepdim=False).shape)
        #print("dim 1",torch.sum(out3, dim=1, keepdim=False).shape)
        #print("dim 2",torch.sum(out3, dim=2, keepdim=False).shape)
        #print("neck",self.neck)
        # Addition
        neck = self.tanh(self.vec2neck(torch.sum(out3, dim=1, keepdim=False)))

        # Stacking
        #out3 = out3.view(out3.shape[0], out3.shape[1] * out3.shape[2])
        #neck = self.tanh(self.vec2neck(out3))

        out, scDec = self.dec(neck)

        return out, neck, scEnc1,scEnc2, scDec

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


    

# # Tranformer Encoder
class AttnEmbedder(nn.Module):
    def __init__(self, index_dict, d_model, n_heads, d_embed, seed=22):
        super().__init__()

        self.index_dict = index_dict

        self.get_kernel_slices()

        assert d_model % d_embed == 0, "d_model and d_embed need to be divisible"
        self.output_dim = d_model
        self.embed_dim = d_embed
        self.heads = n_heads
        self.d_ff = int(1.5 * self.embed_dim)
        self.dropout = 0.1
        self.N = 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="relu",
        )
        tra_norm = None
        # if self.normalize is not None:
        # tra_norm = Norm(d_model=self.embed_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.N, norm=tra_norm
        )

        # position encoding
        self.max_seq_len = self.get_max_weights_per_neuron()
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)
        # weights-to-seqenence embedder
        self.comp_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def get_max_weights_per_neuron(
        self,
    ):
        weights_list = []
        for idx, layer in enumerate(self.index_dict["layer"]):
            weights_size = (
                self.index_dict["kernel_size"][idx]
                * self.index_dict["channels_in"][idx]
            )
            weights_list.append(int(weights_size) + 1)
        # get max number of weights
        max_no_weights = max(weights_list)
        # print(max_no_weights)
        return max_no_weights

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(
        self,
    ):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        y_lst = []
        # loop over layers
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # slice weights
                w_tmp = x[:, kernel_index]
                # unsqueeze -> last dimension to one
                w_tmp = w_tmp.unsqueeze(dim=-1)
                # repeat weights multiple times
                w_tmp = repeat(w_tmp, "b n () -> b n d", d=self.embed_dim)
                # apply position encoding
                # embedd positions
                b, n, d = w_tmp.shape
                # print(w_tmp.shape)
                positions = torch.arange(
                    # self.max_seq_len, device=w_tmp.device
                    n,
                    device=w_tmp.device,
                ).unsqueeze(0)
                w_tmp = w_tmp + self.position_embeddings(positions).expand_as(w_tmp)
                # compression token
                b, n, _ = w_tmp.shape
                copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
                w_tmp = torch.cat((copm_tokens, w_tmp), dim=1)
                # pass through attn
                y_tmp = self.transformer(w_tmp)
                # get compression tokens
                y_tmp = y_tmp[:, 0, :]
                y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        # repeat to get output dimensions
        # print(y.shape)
        repeat_factor = int(self.output_dim / self.embed_dim)
        y = y.repeat([1, 1, repeat_factor])
        # print(y.shape)
        return y


class ResBlock(nn.Module):
    def __init__(self, dim, nlayers, nlin, dropout):
        super().__init__()

        self.resblockList = nn.ModuleList()

        for ldx in range(nlayers - 1):
            self.resblockList.append(nn.Linear(dim, dim, bias=True))
            # add nonlinearity
            if nlin == "elu":
                self.resblockList.append(nn.ELU())
            if nlin == "celu":
                self.resblockList.append(nn.CELU())
            if nlin == "gelu":
                self.resblockList.append(nn.GELU())
            if nlin == "leakyrelu":
                self.resblockList.append(nn.LeakyReLU())
            if nlin == "relu":
                self.resblockList.append(nn.ReLU())
            if nlin == "tanh":
                self.resblockList.append(nn.Tanh())
            if nlin == "sigmoid":
                self.resblockList.append(nn.Sigmoid())
            if dropout > 0:
                self.resblockList.append(nn.Dropout(dropout))
        # init output layer
        self.resblockList.append(nn.Linear(dim, dim, bias=True))
        # add output nonlinearity
        if nlin == "elu":
            self.nonlin_out = nn.ELU()
        if nlin == "celu":
            self.nonlin_out = nn.CELU()
        if nlin == "gelu":
            self.nonlin_out = nn.GELU()
        if nlin == "leakyrelu":
            self.nonlin_out = nn.LeakyReLU()
        if nlin == "tanh":
            self.nonlin_out = nn.Tanh()
        if nlin == "sigmoid":
            self.nonlin_out = nn.Sigmoid()
        else:  # relu
            self.nonlin_out = nn.ReLU()

    def forward(self, x):
        # clone input
        x_inp = x.clone()
        # forward prop through res block
        for m in self.resblockList:
            x = m(x)
        # add input and new x together
        y = self.nonlin_out(x + x_inp)
        return y


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        print(f"init regular encoder")
        # load config
        res_blocks = config.get("model::res_blocks", 0)
        res_block_lays = config.get("model::res_block_lays", 0)
        h_layers = config.get("model::h_layers", 1)
        i_dim = config.get("model::i_dim", (14 * 14) * 10 + 10 * 10)
        latent_dim = config.get("model::latent_dim", 10)
        transition = config.get("model::transition", "lin")
        nlin = config.get("model::nlin", "leakyrelu")
        dropout = config.get("model::dropout", 0.2)
        init_type = config.get("model::init_type", "uniform")
        self.init_type = init_type

        # set flag for residual blocks
        self.res = False
        if res_blocks > 0 and res_block_lays > 0:
            self.res = True

        if self.res:
            # start with encoder resblock
            self.resEncoder = nn.ModuleList()
            for _ in range(res_blocks):
                self.resEncoder.append(
                    ResBlock(
                        dim=i_dim, nlayers=res_block_lays, nlin=nlin, dropout=dropout
                    )
                )
        # get array of dimensions (encoder, decoder is reverse)
        if transition == "lin":
            dimensions = np.linspace(i_dim, latent_dim, h_layers + 2).astype("int")
        else:
            raise NotImplementedError

        # init encoder
        self.encoder = nn.ModuleList()
        # compose layers
        for idx, _ in enumerate(dimensions[:-2]):
            self.encoder.append(nn.Linear(dimensions[idx], dimensions[idx + 1]))
            # add nonlinearity
            if nlin == "elu":
                self.encoder.append(nn.ELU())
            if nlin == "celu":
                self.encoder.append(nn.CELU())
            if nlin == "gelu":
                self.encoder.append(nn.GELU())
            if nlin == "leakyrelu":
                self.encoder.append(nn.LeakyReLU())
            if nlin == "relu":
                self.encoder.append(nn.ReLU())
            if nlin == "tanh":
                self.encoder.append(nn.Tanh())
            if nlin == "sigmoid":
                self.encoder.append(nn.Sigmoid())
            if dropout > 0:
                self.encoder.append(nn.Dropout(dropout))
        # init output layer
        self.encoder.append(nn.Linear(dimensions[-2], dimensions[-1]))

        # normalize outputs between 0 and 1
        if config.get("model::normalize_latent", True):
            self.encoder.append(nn.Tanh())

        # initialize weights with se methods
        print("initialze encoder")
        self.encoder = self.initialize_weights(self.encoder)
        if self.res:
            self.resEncoder = self.initialize_weights(self.resEncoder)

    def initialize_weights(self, module_list):
        for m in module_list:
            if type(m) == nn.Linear:
                if self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform(m.weight)
                if self.init_type == "xavier_normal":
                    nn.init.xavier_normal(m.weight)
                if self.init_type == "uniform":
                    nn.init.uniform(m.weight)
                if self.init_type == "normal":
                    nn.init.normal(m.weight)
                if self.init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                if self.init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)
        return module_list

    def forward(self, x):
        # forward prop through resEncoder
        if self.res:
            for resblock in self.resEncoder:
                x = resblock(x)
        # forward prop through encoder
        for layer in self.encoder:
            x = layer(x)
        return x

    
class EncoderTransformer(nn.Module):
    """
    Encoder Transformer
    """

    def __init__(self, config):
        super(EncoderTransformer, self).__init__()

        # get config
        self.N = config["model::N_attention_blocks"]
        self.input_dim = config["model::i_dim"]
        self.embed_dim = config["model::dim_attention_embedding"]
        self.normalize = config["model::normalize"]
        self.heads = config["model::N_attention_heads"]
        self.dropout = config["model::dropout"]
        self.d_ff = config["model::attention_hidden_dim"]
        self.latent_dim = config["model::latent_dim"]
        self.device = config["device"]
        self.compression = config.get("model::compression", "linear")
        # catch deprecated stuff.
        compression_token = config.get("model::compression_token", "NA")
        if not compression_token == "NA":
            if compression_token == True:
                self.compression == "token"
            elif compression_token == False:
                self.compression == "linear"

        print(f"init attn encoder")

        ### get token embeddings / config
        if config.get("model::encoding", "weight") == "weight":
            # encode each weight separately
            self.max_seq_len = self.input_dim
            self.token_embeddings = Embedder(self.input_dim, self.embed_dim)
        elif config.get("model::encoding", "weight") == "neuron":
            # encode weights of one neuron together
            if config.get("model::encoder") == "attn":
                # use attn embedder (attn of tokens for individual weights)
                print("## attention encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                d_embed = config.get("model::attn_embedder_dim")
                n_heads = config.get("model::attn_embedder_nheads")
                self.token_embeddings = AttnEmbedder(
                    index_dict,
                    d_model=int(self.embed_dim),
                    d_embed=d_embed,
                    n_heads=n_heads,
                )
                self.max_seq_len = self.token_embeddings.__len__()
            else:
                # encode weights of a neuron linearly
                print("## encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                self.token_embeddings = EmbedderNeuronGroup_index(
                    index_dict, self.embed_dim
                )
                self.max_seq_len = self.token_embeddings.__len__()
        elif config.get("model::encoding", "weight") == "neuron_in_out":
            # embed ingoing + outgoing weights together
            index_dict = config.get("model::index_dict", None)
            self.token_embeddings = EmbedderNeuron(index_dict, self.embed_dim)
            self.max_seq_len = self.token_embeddings.__len__()

        ### set compression token embedding
        if self.compression == "token":
            self.comp_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            # add sequence length of 1
            self.max_seq_len += 1

        #### get learned position embedding
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)

        ### compose transformer layers
        self.transformer_type = config.get("model::transformer_type", "pol")
        if self.transformer_type == "pol":
            self.layers = get_clones(
                EncoderLayer(
                    d_model=self.embed_dim,
                    heads=self.heads,
                    normalize=self.normalize,
                    dropout=self.dropout,
                    d_ff=self.d_ff,
                ),
                self.N,
            )
        elif self.transformer_type == "pytorch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation="relu",
            )
            tra_norm = None
            if self.normalize is not None:
                tra_norm = Norm(d_model=self.embed_dim)
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.N, norm=tra_norm
            )

        ### mapping from tranformer output to latent space
        # full, average, or compression token
        bottleneck = config.get("model::bottleneck", "linear")
        if self.compression == "token" or self.compression == "average":
            bottleneck_input = self.embed_dim
        else:  # self.compression=="linear"
            bottleneck_input = self.embed_dim * self.max_seq_len
        # get mapping: linear, linear bounded (with tanh) or mlp
        if bottleneck == "linear":
            self.vec2neck = nn.Sequential(nn.Linear(bottleneck_input, self.latent_dim))
        elif bottleneck == "linear_bounded":
            self.vec2neck = nn.Sequential(
                nn.Linear(bottleneck_input, self.latent_dim), nn.Tanh()
            )
        elif bottleneck == "mlp":
            h_layers_mlp = config.get("model::bottleneck::h_lays", 3)
            config_mlp = {
                "model::res_blocks": 0,
                "model::res_block_lays": 0,
                "model::h_layers": h_layers_mlp,
                "model::i_dim": bottleneck_input,
                "model::latent_dim": self.latent_dim,
                "model::transition": "lin",
                "model::nlin": "leakyrelu",
                "model::dropout": self.dropout,
                "model::init_type": "kaiming_normal",
                "model::normalize_latent": True,
            }
            self.vec2neck = Encoder(config_mlp)

    def forward(self, x, mask=None):
        """
        forward function: get token embeddings, add position encodings, pass through transformer, map to bottleneck
        """
        attn_scores = []  # not yet implemented, to prep interface
        # embedd weights
        x = self.token_embeddings(x)
        # add a compression token to the beginning of each sequence (dim = 1)
        if self.compression == "token":
            b, n, _ = x.shape
            copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
            x = torch.cat((copm_tokens, x), dim=1)
        # embedd positions
        positions = torch.arange(self.max_seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions).expand_as(x)

        # pass through encoder
        # x = self.encoder(x, mask)
        if self.transformer_type == "pol":
            for ndx in range(self.N):
                x, scores = self.layers[ndx](x, mask)
                attn_scores.append(scores)
        elif self.transformer_type == "pytorch":
            x = self.transformer(x)

        # compress to bottleneck
        if self.compression == "token":
            # take only first part of the sequence / token
            x = x[:, 0, :]
        elif self.compression == "average":
            # take only first part of the sequence / token
            x = torch.mean(x, dim=1)
        else:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x = self.vec2neck(x)
        #
        return x, attn_scores
    

class AE_attn(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE_attn, self).__init__()

        self.encoder = EncoderTransformer(config)
        self.decoder = DecoderTransformer(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z, _ = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y, _ = self.decoder(z)
        return y




class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.0,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()
        if nlin == "ORU":
            return ORU()
        if nlin == "ERU":
            return ERU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
                or isinstance(layer, ORU)
                or isinstance(layer, ERU)
            ):
                activations.append(x)
        return x, activations
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()   
        
        
class ReconLoss(nn.Module):
    """
    Regular MSE w/ normalization
    """

    def __init__(self, reduction, normalization_var=None):
        super(ReconLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.normalization_var = normalization_var
        self.loss_mean = None

    def forward(self, output, target):
        assert (
            output.shape == target.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape}"
        if self.normalization_var is not None:
            output /= self.normalization_var
            target /= self.normalization_var
        loss = self.criterion(output, target)
        # init rsq
        rsq = -999
        if self.loss_mean:
            rsq = torch.tensor(1 - loss.item() / self.loss_mean)
        return {"loss_recon": loss, "rsq": rsq}

    def set_normalization(self, reference_weights, index_dict):
        """set normalization koefficinet at init"""
        # compute variance of the weights __per layer__
        variances = []
        for start, length in zip(index_dict["idx_start"], index_dict["idx_length"]):
            idx_start = start
            idx_end = start + length
            sliced = reference_weights[:, idx_start:idx_end]
            tmp = sliced.flatten()
            var_tmp = torch.var(tmp)
            var = torch.ones(sliced.shape[1]) * var_tmp
            variances.append(var)
        variances = torch.cat(variances, dim=0)
        # set norm in recon loss
        self.normalization_var = variances

    def set_mean_loss(self, weights: torch.Tensor):
        # check that weights are tensor..
        assert isinstance(weights, torch.Tensor)
        w_mean = weights.mean(dim=0)  # compute over samples (dim0)
        # scale up to same size as weights
        weights_mean = repeat(w_mean, "d -> n d", n=weights.shape[0])
        out_mean = self.forward(weights_mean, weights)

        # compute mean
        print(f" mean loss: {out_mean['loss_recon']}")

        self.loss_mean = out_mean["loss_recon"]


class MSELossClipped(nn.Module):
    """
    implementation of MSE with error clipping
    thresholds the error term of MSE loss at value threshold.
    Idea: limit maximum influence of data points with large error to prevent them from dominating the entire error term
    """

    def __init__(self, reduction, threshold):

        super(MSELossClipped, self).__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, x, y):
        # compure raw error
        error = self.mse(x, y)
        # clip values
        if self.threshold:
            error = -torch.nn.functional.threshold(
                -error, -self.threshold, -self.threshold
            )
        error = torch.sum(error)
        if self.reduction == "mean":
            nsamples = torch.numel(error)
            error /= nsamples
        return error


class LayerWiseReconLoss(nn.Module):
    """
    MSE w/ layer-wise normalization
    """

    def __init__(self, reduction, index_dict, normalization_koeff=None, threshold=None):
        super(LayerWiseReconLoss, self).__init__()
        self.threshold = threshold
        if self.threshold:
            self.criterion = MSELossClipped(reduction="sum", threshold=self.threshold)
        else:
            self.criterion = nn.MSELoss(reduction="sum")
        self.reduction = reduction
        self.normalization_koeff = normalization_koeff
        self.get_index_idx(index_dict)
        self.loss_mean = None

    def forward(self, output, target):
        # check validity
        assert (
            output.shape == target.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape}"
        # normalize outputs and targets
        if self.normalization_koeff is not None:
            dev = output.device
            self.normalization_koeff = self.normalization_koeff.to(dev)
            output = torch.clone(output) / self.normalization_koeff
            target = torch.clone(target) / self.normalization_koeff

        # compute layer-wise loss / rsq
        out = {}
        loss = torch.tensor(0.0, device=output.device).float()
        # iterate over layers
        for layer, idx_start, idx_end, loss_weight_idx in self.layer_index:
            # slice weight vector
            out_tmp = output[:, idx_start:idx_end]
            tar_tmp = target[:, idx_start:idx_end]
            # compute loss
            loss_tmp = self.criterion(out_tmp, tar_tmp)
            # reduction
            if self.reduction == "global_mean":
                # scale with overall number of paramters. each weight has the same contribution to loss
                loss_tmp /= output.shape[0] * output.shape[1]
            elif self.reduction == "layer_mean":
                # scale with layer number of paramters. each layer has the same contribution to loss
                loss_tmp /= output.shape[0] * out_tmp.shape[1]
            else:
                raise NotImplementedError
            # reweight with # of weights in this layer
            loss += loss_weight_idx * loss_tmp
            out[f"loss_recon_l{layer[0]}"] = loss_tmp.detach()
            # if loss_mean exists: compute rsq for this layer
            if self.loss_mean:
                out[f"rsq_l{layer[0]}"] = torch.tensor(
                    1
                    - loss_tmp.item() / self.loss_mean[f"loss_recon_l{layer[0]}"].item()
                )
        # pass loss_recon to output
        out["loss_recon"] = loss
        # of loss_mean exists: compute overall rsq
        if self.loss_mean:
            out["rsq"] = torch.tensor(
                1 - loss.item() / self.loss_mean["loss_recon"].item()
            )
        return out
    
    
class SimCLRAEModule(nn.Module):
    """
    Main Hyper-Representation Model Class.
    Implements forward, backwards pass, steps
    Handles device, precision, normalization, etc.
    """

    def __init__(self, config):
        super(SimCLRAEModule, self).__init__()

        self.verbosity = config.get("verbosity", 0)

        if self.verbosity > 0:
            print("Initialize Model")

        # set deivce
        self.device = config.get("device", torch.device("cpu"))
        if type(self.device) is not torch.device:
            self.device = torch.device(self.device)
        if self.verbosity > 0:
            print(f"device: {self.device}")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # if not CPU -> GPU: set cuda seeds
        if self.device is not torch.device("cpu"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # initialize backbone architecture
        self.type = config.get("model::type", "vanilla")
        if self.type == "vanilla":
            model = AE(config)
        elif self.type == "transformer":
            model = AE_attn(config)
        elif self.type == "perceiver":
            model = AE_perceiver(config)

        # initialize projection head (for contrastive learning)
        self.model = model
        projection_head = (
            True if config.get("model::projection_head_layers", None) > 0 else False
        )

        # set loss combination of MSE and InfoNCE
        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction=config.get("training::reduction", "global_mean"),
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            device=self.device,
            contrast=config.get("training::contrast", "simclr"),
            projection_head=projection_head,
            threshold=config.get("training::error_threshold", None),
            z_var_penalty=config.get("training::z_var_penalty", 0.0),
            config=config,
        )

        # send model and criterion to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # initialize model in eval mode
        self.model.eval()

        # gather model parameters and projection head parameters
        # params_lst = [self.model.parameters(), self.criterion.parameters()]
        self.params = self.parameters()

        # set optimizer
        self.set_optimizer(config)

        ### precision
        # half precision
        self.use_half = (
            True if config.get("training::precision", "full") == "half" else False
        )
        if self.use_half:
            print(f"++++++ USE HALF PRECISION +++++++")
            self.model = self.model.half()
            self.criterion = self.criterion.half()

        # automatic mixed precision
        self.use_amp = (
            True if config.get("training::precision", "full") == "amp" else False
        )
        if self.use_amp:
            print(f"++++++ USE AUTOMATIC MIXED PRECISION +++++++")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # set trackers (deprecated)
        self.best_epoch = None
        self.loss_best = None
        self.best_checkpoint = None

        # initialize mean loss for r^2
        self.loss_mean = None

        # initialize scheduler
        self.set_scheduler(config)

        # initialize whether to save the checkpoint
        self._save_model_checkpoint = True

        # initialize feature normalization coefficients
        self.feature_normalization_koeff = None

        # init gradien clipping
        if config.get("training::gradient_clipping", None) == "norm":
            self.clip_grads = self.clip_grad_norm
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        elif config.get("training::gradient_clipping", None) == "value":
            self.clip_grads = self.clip_grad_value
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        else:
            self.clip_grads = None

    def set_feature_normalization(self, reference_weights, index_dict):
        """
        computes std of weights __per layer__ for end-to-end normalization
        """
        # compute std of the weights __per layer__
        norm_std = []
        for start, length in zip(index_dict["idx_start"], index_dict["idx_length"]):
            idx_start = start
            idx_end = start + length
            sliced = reference_weights[:, idx_start:idx_end]
            tmp = sliced.flatten()
            std_tmp = torch.std(tmp)
            # apply thresholding to prevent division by zero
            epsilon = 1e-4
            if std_tmp.item() < epsilon:
                std_tmp = torch.tensor(1) * epsilon
            std = torch.ones(sliced.shape[1]) * std_tmp
            norm_std.append(std)
        norm_std = torch.cat(norm_std, dim=0)
        # set norm in recon loss
        self.feature_normalization_koeff = norm_std

    def clip_grad_norm(
        self,
    ):
        # print(f"clip grads by norm")
        nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

    def clip_grad_value(
        self,
    ):
        # print(f"clip grads by value")
        nn.utils.clip_grad_value_(self.parameters(), self.clipping_value)

    def forward(self, x):
        # pass forward call through to model
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        # normalize input features
        if self.feature_normalization_koeff is not None:
            dev = x.device
            self.feature_normalization_koeff = self.feature_normalization_koeff.to(dev)
            x /= self.feature_normalization_koeff
        z = self.model.forward_encoder(x)
        return z

    def forward_decoder(self, z):
        y = self.model.forward_decoder(z)
        # map output features back to original feature space
        if self.feature_normalization_koeff is not None:
            dev = y.device
            self.feature_normalization_koeff = self.feature_normalization_koeff.to(dev)
            y *= self.feature_normalization_koeff
        return y

    def forward_embeddings(self, x):
        z = self.forward_encoder(x)
        return z

    def set_optimizer(self, config):
        if config.get("optim::optimizer", "adamw") == "sgd":
            self.optimizer = torch.optim.SGD(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adam":
            self.optimizer = torch.optim.Adam(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adamw":
            self.optimizer = torch.optim.AdamW(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "lamb":
            self.optimizer = torch.optim.Lamb(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        else:
            raise NotImplementedError(
                f'the optimizer {config.get("optim::optimizer", "adam")} is not implemented. break'
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "ReduceLROnPlateau":
            mode = config.get("optim::scheduler_mode", "min")
            factor = config.get("optim::scheduler_factor", 0.1)
            patience = config.get("optim::scheduler_patience", 10)
            threshold = config.get("optim::scheduler_threshold", 1e-4)
            threshold_mode = config.get("optim::scheduler_threshold_mode", "rel")
            cooldown = config.get("optim::scheduler_cooldown", 0)
            min_lr = config.get("optim::scheduler_min_lr", 0.0)
            eps = config.get("optim::scheduler_eps", 1e-8)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
                verbose=False,
            )

    def set_normalization(self, reference_weights, index_dict):
        """
        sets normalization for layer-wise loss norm. get's passed on to criterion
        """
        self.criterion.loss_recon.set_normalization(reference_weights, index_dict)

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            if self._save_model_checkpoint:
                # save model state-dict
                perf_dict["state_dict"] = self.model.state_dict()
                if self.criterion.loss_contrast is not None:
                    perf_dict[
                        "projection_head"
                    ] = self.criterion.loss_contrast.projection_head.state_dict()
                # save optimizer state-dict
                perf_dict["optimizer_state"] = self.optimizer.state_dict()
            torch.save(perf_dict, fname)
        return None

    # ##########################
    # one training step / batch
    # ##########################
    def train_step(self, x_i, x_j):
        # zero grads before training steps
        self.optimizer.zero_grad()
        if self.use_half:
            x_i, x_j = x_i.half(), x_j.half()
        # forward pass with both views
        z_i, y_i = self.forward(x_i)
        z_j, y_j = self.forward(x_j)
        # cat y_i, y_j and x_i, x_j
        x = torch.cat([x_i, x_j], dim=0)
        y = torch.cat([y_i, y_j], dim=0)
        # compute loss
        perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
        # prop loss backwards to
        loss = perf["loss"]
        loss.backward()
        # gradient clipping
        if self.clip_grads is not None:
            self.clip_grads()
        # update parameters
        self.optimizer.step()
        # compute embedding properties
        z_norm = torch.linalg.norm(z_i, ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i, dim=0))
        perf["z_norm"] = z_norm
        perf["z_var"] = z_var
        return perf

    # ##########################
    # one training step / batch with automatic mixed precision
    # ##########################
    def train_step_amp(self, x_i, x_j):
        with torch.cuda.amp.autocast(enabled=True):
            # forward pass with both views
            z_i, y_i = self.forward(x_i)
            z_j, y_j = self.forward(x_j)
            # cat y_i, y_j and x_i, x_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            # compute loss
            perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            # prop loss backwards to
            loss = perf["loss"]
        # backward
        # technically, there'd need to be a scaler for each loss individually.
        self.scaler.scale(loss).backward()
        # if gradient clipping is to be used...
        if self.clip_grads is not None:
            # # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
            self.clip_grads()
        # update parameters
        self.scaler.step(self.optimizer)
        # update scaler
        self.scaler.update()
        # zero grads
        self.optimizer.zero_grad()
        # compute embedding properties
        z_norm = torch.linalg.norm(z_i, ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i, dim=0))
        perf["z_norm"] = z_norm
        perf["z_var"] = z_var
        return perf

    # ##########################
    # one full training epoch
    # ##########################
    def train_epoch(self, trainloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()
        self.criterion.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(trainloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step

            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            # compute loss
            if self.use_amp:
                perf = self.train_step_amp(x_i, x_j)
            else:
                perf = self.train_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * len(l_i)
                else:
                    perf_out[key] += perf[key] * len(l_i)
            n_data += len(l_i)

        self.model.eval()
        self.criterion.eval()
        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            perf_out[key] = perf_out[key].item()
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step(perf_out["loss"])

        return perf_out

    # ##########################
    # one training step / batch
    # ##########################
    def test_step(self, x_i, x_j):
        with torch.no_grad():
            if self.use_half:
                x_i, x_j = x_i.half(), x_j.half()
            # forward pass with both views
            z_i, y_i = self.forward(x_i)
            z_j, y_j = self.forward(x_j)
            # cat y_i, y_j and x_i, x_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            # compute loss
            perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            return perf

    # ##########################
    # one training step / batch with automatic mixed precision
    # ##########################
    def test_step_amp(self, x_i, x_j):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                # forward pass with both views
                z_i, y_i = self.forward(x_i)
                z_j, y_j = self.forward(x_j)
                # cat y_i, y_j and x_i, x_j
                x = torch.cat([x_i, x_j], dim=0)
                y = torch.cat([y_i, y_j], dim=0)
                # compute loss
                perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            return perf

    # ##########################
    # one full test epoch
    # ##########################
    def test_epoch(self, testloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"test at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        self.criterion.eval()
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(testloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step
            # compute loss
            if self.use_amp:
                perf = self.test_step_amp(x_i, x_j)
            else:
                perf = self.test_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * len(l_i)
                else:
                    perf_out[key] += perf[key] * len(l_i)
            n_data += len(l_i)

        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            perf_out[key] = perf_out[key].item()

        return perf_out
if __name__ == "__main__":
    #from pathlib import Path
    # set which hyper-representation to load

    #PATH_ROOT = Path("./")
    # load config
    #config_path = PATH_ROOT.joinpath('config_ae.json')
    config = json.load('./config_ae.json'.open('r'))
    #config['dataset::dump'] = PATH_ROOT.joinpath('dataset.pt').absolute()
    # set resources
    gpus = 1 if torch.cuda.is_available() else 0
    cpus = 8
    resources_per_trial = {"cpu": cpus, "gpu": gpus}

    device = torch.device('cuda:0') if gpus>0 else torch.device('cpu')
    config['device'] = device
    config['model::type'] = 'transformer'
    # Instanciate model
    module = SimCLRAEModule(config)
    # load checkpoint
    #checkpoint_path = PATH_ROOT.joinpath('checkpoint_ae.pt')
    #checkpoint = torch.load(checkpoint_path,map_location=device)
    # load checkpoint to model
    #module.model.load_state_dict(checkpoint)
