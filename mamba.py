# some of the code is modified from https://github.com/johnma2006/mamba-minimal

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from dataclasses import dataclass
import json
import math
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class RMSNorm(nn.Module):
    def __init__(self,
                 d: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class Mamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        D = args.d_model
        E = args.d_inner
        N = args.d_state
        D_delta = args.dt_rank
        D_conv = args.d_conv
        V = args.vocab_size
        
        self.embedding = nn.Embedding(V, D)
        self.layers = nn.ModuleList([MambaLayer(args=args) for _ in range(args.n_layer)])
        self.norm = RMSNorm(D)
        self.lm_head  = nn.Linear(D, V, bias=False)
    
    def forward(self, input_ids):
        args = self.args
        D = args.d_model
        E = args.d_inner
        N = args.d_state
        D_delta = args.dt_rank
        D_conv = args.d_conv
        V = args.vocab_size
        
        Batch,L = input_ids.size()

        # [B,L,D]                         [B,L]
        resid         = self.embedding(input_ids)
        
        for layer in self.layers:
            # [B,L,D]         [B,L,D]
            resid     += layer(resid)
         
        # [B,L,D]              [B,L,D]
        resid     = self.norm( resid )
        
        # [B,L,V]          [D->V] [B,L,D]
        logits    = self.lm_head( resid ) # no bias
        
        return logits
    
    @staticmethod
    def from_pretrained(pretrained_model_name):
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
            
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        D = args.d_model
        E = args.d_inner
        N = args.d_state
        D_delta = args.dt_rank
        D_conv = args.d_conv
        V = args.vocab_size
        
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace("backbone.", "").replace("mixer.", "")
            # we split in_proj into two seperate things
            if 'in_proj' in key:
                new_state_dict[key] = value[:E]
                new_state_dict[key.replace("in_proj", "skip_proj")] = value[E:]
            # we renamed these
            elif 'dt_proj' in key:
                new_state_dict[key.replace("dt_proj", "W_delta_2")] = value
            elif 'norm_f' in key:
                new_state_dict[key.replace("norm_f", "norm")] = value
            # we split this into three seperate things
            elif 'x_proj' in key:
                W = value
                # pull them out
                new_state_dict[key.replace("x_proj", "W_delta_1")] = W[:D_delta]
                new_state_dict[key.replace("x_proj", "W_B")] = W[D_delta:D_delta+N]
                new_state_dict[key.replace("x_proj", "W_C")] = W[D_delta+N:]
            # we call this W_D
            elif '.D' in key:
                new_state_dict[key.replace(".D", ".W_D")] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        return model

class MambaLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        ## Variables
        D = args.d_model
        E = args.d_inner
        N = args.d_state
        D_delta = args.dt_rank
        D_conv = args.d_conv
        V = args.vocab_size
        
        ## Process inputs
        self.norm      = RMSNorm(D)
        self.skip_proj = nn.Linear(D, E, bias=False)
        self.in_proj   = nn.Linear(D, E, bias=False)
        
        ## Conv
        self.conv1d    = nn.Conv1d(
            in_channels=E,
            out_channels=E,
            bias=True,
            kernel_size=D_conv,
            groups=E,
            padding=D_conv - 1,
        )
        
        ## SSM Params
        self.W_delta_1 = nn.Linear(E, D_delta, bias=False)
        self.W_delta_2 = nn.Linear(D_delta, E, bias=True)
        self.W_B = nn.Linear(E, N, bias=False)
        self.W_C = nn.Linear(E, N, bias=False)
        
        self.A_log     = nn.Parameter(torch.log(torch.randn([E,N])))
        self.W_D = nn.Parameter(torch.ones(E))
        
        ## Project back out
        self.out_proj  = nn.Linear(E, D, bias=False)
    
    def forward(self, x):
        
        args = self.args
        D = args.d_model
        E = args.d_inner
        N = args.d_state
        D_delta = args.dt_rank
        D_conv = args.d_conv
        V = args.vocab_size
        
        Batch,L,D = x.size()
        
        ###### Process inputs ######
        # [B,L,D]             [B,L,D]
        x         = self.norm(  x  )
        # [B,L,E]         [D->E]  [B,L,D]
        skip      = self.skip_proj(  x  ) # no bias
        # [B,L,E]         [D->E] [B,L,D]
        x         = self.in_proj(  x  ) # no bias
        
        ###### Conv ######
        # [B,E,L]
        x         = rearrange(x, 'B L E -> B E L')
        # [B E L]                [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
        x         = self.conv1d(   x   )[:, :, :L]
        # [B,L,E]
        x         = rearrange(x, 'B E L -> B L E')

        ###### Nonlinearity  ######
        # [B,L,E]          [B,L,E]
        x         = F.silu(  x   )
        
        ###### SSM ######
        
        # W_delta is factored into two matrices W_delta_1 and W_delta_2, combine them back
        # [E,E] =          [E,D_delta]         [D_delta, E]
        W_delta = self.W_delta_1.weight.T @ self.W_delta_2.weight.T
       
        self.A = -torch.exp(self.A_log)
       
        ys = []
        for b in range(Batch):
            ys_b = []
            
            # latent state, init to zeros
            h = torch.zeros([E,N])
            for l in range(L):
                #### First, discretization: A and B -> A_bar and B_bar ####
                ## Compute Delta ##
                # [E]                   [E]  x  [E,E]  +         [E]
                delta    = F.softplus(self.W_delta_2(self.W_delta_1(x[b,l])))
                
                ## Discretize A -> A_bar ##
                # (note [E,N]*[E,1] will first repeat the [E,1] N times so its like [E,N])
                # [E,N]             (     [E,1]      *  [E,N] ) 
                A_bar    = torch.exp(delta.view(E,1) * self.A)
                
                ## Discretize B -> B_bar ##
                # [N]        [E->N]   [E]
                B        = self.W_B(x[b,l]) # no bias
                # [E,N]        [E,1]       x    [1,N]
                B_bar    = delta.view(E,1) @ B.view(1,N)
                
                #### Update latent vector h ####
                ## input floats for the ssm at time l
                # [E]       [E]
                x_l      = x[b,l]
                
                ## Move ahead by one step
                # (note, [E,N]*[E,1] will first repeat the [E,1] N times so its like [E,N])
                # [E,N]    [E,N]  [E,N]   [E,N]      [E,1]
                h        = A_bar *  h   + B_bar  *  x_l.view(E,1)
                
                #### Compute output float y ####
                ## (C matrix needed for computing y)
                # [N]        [E->N]   [E]
                C        = self.W_C(x[b,l]) # no bias
                
                ## Output floats y at time l
                # [E,1]      [E,N]  x   [N,1]
                y_l      =     h    @ C.view(N,1)
                
                ys_b.append([y.float() for y in y_l.flatten()])
            ys.append(ys_b)
        # [B,L,E]
        y = torch.tensor(ys)
        
        ###### Finish layer ######
        
        # [B,L,E]  [B,L,E]    [B,L,E]       [E]
        y         =   y      +   x     *  self.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      * F.silu(  skip  )
        
        # [B,L,D]          [E->D]  [B,L,E]
        y         = self.out_proj(   y   ) # no bias
        
        return y
