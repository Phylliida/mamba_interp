# some of the code is modified from https://github.com/johnma2006/mamba-minimal

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, overload
from jaxtyping import Float, Int
from contextlib import contextmanager
from typing_extensions import Literal

import re
import time
import logging
from dataclasses import dataclass
import json
import math
import copy
import itertools
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import AutoTokenizer
from transformer_lens.utils import USE_DEFAULT_VALUE

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "B L-1"]
Loss = Union[SingleLoss, LossPerToken]


MODEL_TOKENIZER = 'EleutherAI/gpt-neox-20b'


def get_converted_model_from_hf(pretrained_model_name, device='cuda'):
    
    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))

    def load_state_dict_hf(model_name, device=None, dtype=None):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True, map_location='cpu')
        
    config_data = load_config_hf(pretrained_model_name)
    cfg = ModelCfg(
        d_model=config_data['d_model'],
        n_layers=config_data['n_layer'],
        vocab_size=config_data['vocab_size'],
        device=device
    )
    
    state_dict = load_state_dict_hf(pretrained_model_name)
    
    converted_state_dict = convert_state_dict_to_our_format(cfg, state_dict=state_dict)
        
    return cfg, converted_state_dict


def convert_state_dict_to_our_format(cfg, state_dict):
    D = cfg.d_model
    E = cfg.d_inner
    N = cfg.d_state
    D_delta = cfg.dt_rank
    D_conv = cfg.d_conv
    V = cfg.vocab_size
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("backbone.", "").replace("mixer.", "")
        key = key.replace("layers.", "blocks.")
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
    return new_state_dict


def convert_state_dict_to_original_format(cfg, state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        original_key = key
        if not 'lm_head' in key: # everything except lm_head has backbone in front
            key = 'backbone.' + key
        # replace blocks.*.stuff with layers.*.mixer.stuff
        key = re.sub(r"blocks\.(\d+)\.", r"layers.\1.mixer.", key)

        # they don't put the norm in mixer
        if 'mixer.norm' in key:
            key = key.replace("mixer.norm", "norm")
        
        if "in_proj" in key:
            # we split this into in_proj and skip_proj
            in_proj = value
            skip_proj = state_dict[original_key.replace("in_proj", "skip_proj")]
            new_state_dict[key] = torch.cat([in_proj, skip_proj], dim=0)
        elif "skip_proj" in key:
            pass # we do this above
        elif "W_delta_2" in key:
            new_state_dict[key.replace("W_delta_2", "dt_proj")] = value
        # we renamed these
        elif 'dt_proj' in key:
            new_state_dict[key.replace("dt_proj", "W_delta_2")] = value
        elif 'norm' in key and not 'layers' in key: # the base norm is called norm_f
            new_state_dict[key.replace("norm", "norm_f")] = value
        # we split x_proj into three seperate things
        elif 'W_delta_1' in key:
            W_delta_1 = value
            W_B = state_dict[original_key.replace("W_delta_1", "W_B")]
            W_C = state_dict[original_key.replace("W_delta_1", "W_C")]
            new_state_dict[key.replace("W_delta_1", "x_proj")] = torch.cat([W_delta_1, W_B, W_C], dim=0)
        elif 'W_B' in key or 'W_C' in key:
            pass # we do this above
        # we call this W_D
        elif "W_D" in key:
            new_state_dict[key.replace("W_D", "D")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def test_state_dict_convert(device='cuda'):

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    cfg, converted_state_dict = get_converted_model_from_hf("state-spaces/mamba-370m", device=device)

    their_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m")
    their_model = their_model.to(cfg.device)
    test_inputs = torch.tensor([[1,2]], device=cfg.device)

    #their_logits = their_model.forward(test_inputs)
    our_model = HookedMamba(cfg)
    our_model.load_state_dict(converted_state_dict)
    our_model = our_model.to(cfg.device)
    our_logits = our_model.forward(test_inputs)
    
    for i in range(10):
        original = convert_state_dict_to_original_format(cfg, converted_state_dict)
        converted = convert_state_dict_to_our_format(cfg, original)
    our_model_again = HookedMamba(cfg)
    our_model_again.load_state_dict(converted)
    our_model_again = our_model_again.to(cfg.device)
    our_logits_again = our_model_again.forward(test_inputs)


    their_model.load_state_dict(original)
    #their_model = their_model.to(cfg.device)
    #their_again_logits = their_model.forward()

    #assert(torch.allclose(their_logits, our_logits))
    #assert(torch.allclose(their_logits, their_again_logits))
    
    assert(torch.allclose(our_logits, our_logits_again))
    
    print("all tests passed")



@dataclass
class ModelCfg:
    d_model: int
    n_layers: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    default_prepend_bos: bool = True
    tokenizer_prepends_bos: bool = False
    n_ctx: int = 2048
    device: Union[torch.device,str] = 'cuda'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
    
    @property
    def D(self):
        return self.d_model
    @property
    def E(self):
        return self.d_inner
    @property
    def N(self):
        return self.d_state
    @property
    def D_delta(self):
        return self.dt_rank
    @property
    def D_conv(self):
        return self.d_conv
    @property
    def V(self):
        return self.vocab_size
        
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

# reference implementation
# modified from https://github.com/johnma2006/mamba-minimal
class Mamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        self.embedding = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([MambaBlock(cfg=cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(D)
        self.lm_head  = nn.Linear(D, V, bias=False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    

    def forward(self, input):
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L = input.size()

        if type(input) is str:
            input = self.tokenizer(input, return_tensors='pt')['input_ids'] # they passed in a prompt and not input_ids

        # [B,L,D]                         [B,L]
        resid         = self.embedding(input)
        
        for block in self.blocks:
            # [B,L,D]         [B,L,D]
            resid     = block(resid)
         
        # [B,L,D]              [B,L,D]
        resid     = self.norm( resid )
        
        # [B,L,V]          [D->V] [B,L,D]
        logits    = self.lm_head( resid ) # no bias
        
        return logits
    
    @staticmethod
    def from_pretrained(pretrained_model_name, device='cuda'):
        cfg, state_dict = get_converted_model_from_hf(pretrained_model_name=pretrained_model_name, device=device)
        model = Mamba(cfg)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model

class MambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ## Variables
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
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
    
    def forward(self, resid):
        
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L,D = resid.size()
        
        ###### Process inputs ######
        # [B,L,D]             [B,L,D]
        x         = self.norm(  resid  )
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
        
        self.A = -torch.exp(self.A_log)
       
        ys = []
        for b in range(Batch):
            ys_b = []
            
            # latent state, init to zeros
            h = torch.zeros([E,N], device=self.cfg.device)
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
        y = torch.tensor(ys, device=self.cfg.device)
        
        ###### Finish layer ######
        
        # [B,L,E]  [B,L,E]    [B,L,E]       [E]
        y         =   y      +   x     *  self.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      * F.silu(  skip  )
        
        # [B,L,D]         [E->D]  [B,L,E]
        y         = self.out_proj(   y   ) # no bias
    
        # [B,L,D]     [B,L,D]
        resid     +=     y
        
        return resid

    
    def forward_e(self, resid):
        
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L,D = resid.size()
        
        ###### Process inputs ######
        # [B,L,D]             [B,L,D]
        x         = self.norm(  resid  )
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
        
        self.A = -torch.exp(self.A_log)
       
        ys = []
        for b in range(Batch):
            ys_b = []
            
            # latent state, init to zeros
            h = torch.zeros([E,N], device=self.cfg.device)
            for l in range(L):
                ## Compute Delta ##
                # [E]                     [D_delta->E] + [E->D_delta] [E]
                delta_l    = F.softplus(self.W_delta_2(self.W_delta_1(x[b,l])))
                
                for n in range(N):
                    ## Discretize A -> A_bar ##
                    # [E]                   ( [E]    *  [E] ) 
                    A_bar_l_n    = torch.exp(delta_l * self.A[:,n])

                    ## Discretize B -> B_bar ##
                    # [1]                [E]    dot      [E]
                    B_l_n     = self.W_B.weight[n,:].dot(x[b,l]) # no bias
                    # [E]       [E]       x     [1]
                    B_bar_l_n = delta_l    *    B_l_n

                    #### Update latent vector h ####
                    ## Move ahead by one step
                    # [E]      [E]     [E]     [E]       [E]
                    h[:,n]  = A_bar * h[:,n]   + B_bar  *  x[b,l]

                #### Compute output float y ####
                y_l = torch.zeros([E], device=self.cfg.device)
                for n in range(N):
                    # [1]               [E]         dot   [E]
                    C        = self.W_C.weight[n,:].dot(x[b,l]) # no bias
                    # [E]      [1]   [E]
                    y_l      += C * h[:,n]
            
                ys_b.append([y.float() for y in y_l.flatten()])
            ys.append(ys_b)
        # [B,L,E]
        y = torch.tensor(ys, device=self.cfg.device)
        
        ###### Finish layer ######
        
        # [B,L,E]  [B,L,E]    [B,L,E]       [E]
        y         =   y      +   x     *  self.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      * F.silu(  skip  )
        
        # [B,L,D]         [E->D]  [B,L,E]
        y         = self.out_proj(   y   ) # no bias
    
        # [B,L,D]     [B,L,D]
        resid     +=     y
        
        return resid
        
class InputDependentHookPoint(HookPoint):
    def __init__(self, input_dependent_postfixes_func):
        super().__init__()
        self.hooks = {}
        self.input_dependent_postfixes_func = input_dependent_postfixes_func
    
    def add_input_dependent_hooks(self, input):
        for postfix in self.input_dependent_postfixes_func(input=input):
            if not postfix in self.hooks:
                postfix_hook = HookPoint()
                postfix_hook.name = self.name + postfix
                self.hooks[postfix] = postfix_hook
                yield self.hooks[postfix].name, self.hooks[postfix]
    
class InputDependentHookedRootModule(HookedTransformer):
    
    def setup(self):
        # setup hooks
        super().setup()

        # remove input_dependent hooks (multiple hooks for each of these will be added below during each call)
        for name, hook in self.input_dependent_hooks():
            if name in self.mod_dict:
                del self.mod_dict[name]
            if name in self.hook_dict:
                del self.hook_dict[name]
            hook.name = name
        
        self.did_setup_input_dependent_hooks = False
    

    def get_all_hooks(self, *model_args, **model_kwargs):
        # fwd_hooks=None expands all input_dependent hooks
        # we need this context to get the input dependent hooks
        with self.input_dependent_hooks_context(fwd_hooks=None, bwd_hooks=None, model_args=model_args, model_kwargs=model_kwargs):
            for name, hp in self.hook_dict.items():
                yield name, hp


    def hook_points(self):
        if self.did_setup_input_dependent_hooks:
            return super().hook_points()
        # we need to also include the input_dependent ones
        else:
            return list(super().hook_points()) + [x[1] for x in self.input_dependent_hooks()]
        
    # we need to wrap these calls with a call to setup_input_dependent_hooks
    def run_with_hooks (
            self,
            *model_args,
            fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
            bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
            reset_hooks_end=True,
            clear_contexts=False,
            **model_kwargs,
        ):
                with self.input_dependent_hooks_context(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, model_args=model_args, model_kwargs=model_kwargs):
                    res = super().run_with_hooks(
                        *model_args,
                        fwd_hooks=fwd_hooks,
                        bwd_hooks=bwd_hooks,
                        reset_hooks_end=reset_hooks_end,
                        clear_contexts=clear_contexts,
                        **model_kwargs
                    )
                    return res
    
    def run_with_cache(
            self,
            *model_args,
            names_filter: NamesFilter = None,
            device=None,
            remove_batch_dim=False,
            incl_bwd=False,
            reset_hooks_end=True,
            clear_contexts=False,
            **model_kwargs,
        ):
            model_kwargs = dict(list(model_kwargs.items()))
            fwd_hooks = None
            if 'fwd_hooks' in model_kwargs:
                fwd_hooks = model_kwargs['fwd_hooks']
                del model_kwargs['fwd_hooks']
            bwd_hooks = None
            if 'bwd_hooks' in model_kwargs:
                bwd_hooks = model_kwargs['bwd_hooks']
                del model_kwargs['bwd_hooks']
            with self.input_dependent_hooks_context(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, model_args=model_args, model_kwargs=model_kwargs):
                res = super().run_with_cache(
                    *model_args,
                    names_filter=names_filter,
                    device=device,
                    remove_batch_dim=remove_batch_dim,
                    incl_bwd=incl_bwd,
                    reset_hooks_end=reset_hooks_end,
                    clear_contexts=clear_contexts,
                    **model_kwargs)
                return res
        
    def input_dependent_hooks(self):
        for name, module in self.named_modules():
            if name == "":
                continue
            if "InputDependentHookPoint" in str(type(module)):
                yield name, module
    
    @contextmanager
    def input_dependent_hooks_context(self, fwd_hooks, bwd_hooks, model_args, model_kwargs):
        try:
            self.setup_input_dependent_hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, model_args=model_args, model_kwargs=model_kwargs)
            yield self
        finally:
            self.done_input_dependent_hooks()

    def setup_input_dependent_hooks(self, fwd_hooks, bwd_hooks, model_args, model_kwargs):
        if 'input' in model_kwargs:
            input = model_kwargs['input']
        elif len(model_args) > 0:
            input = model_args[0]
        else:
            raise Exception(f"Could not find input in args {model_args} and kwargs {model_kwargs}")
        
        
        # make sure input is ids and not a str
        input = tokenize_if_str(tokenizer=self.tokenizer, input=input)
        input_dependent_lookup = {}
        for name, hook in self.input_dependent_hooks():
            input_dependent_lookup[name] = hook
        expand_all = False
        input_dependent = []
        if fwd_hooks is None:
            expand_all = True # used for run with cache
        else:
            for name, hook in fwd_hooks:
                if type(name) == str and not name in self.mod_dict:
                    input_dependent.append(name)
                else:
                    expand_all = True # we don't know what things make this eval to true, so expand them all
          
        if bwd_hooks is None:
            expand_all = True
        else:
            for name, hook in bwd_hooks:
                if type(name) == str and not name in self.mod_dict:
                    input_dependent.append(name)
                else:
                    expand_all = True # we don't know what things make this eval to true, so expand them all
        
        hooks_to_expand = []
        if expand_all:
            hooks_to_expand = list(self.input_dependent_hooks())
        else:
            for name in input_dependent:
                # look for any prefix-matches, if we have them we need to expand them 
                for input_dependent_name, input_dependent_hook in self.input_dependent_hooks():
                    if name.startswith(input_dependent_name):
                        hooks_to_expand.append((input_dependent_name, input_dependent_hook))
        for name, hook in hooks_to_expand:
            for added_hook_name, added_hook in hook.add_input_dependent_hooks(input=input):
                self.mod_dict[added_hook_name] = added_hook
                self.hook_dict[added_hook_name] = added_hook
        self.did_setup_input_dependent_hooks = True
    
    def done_input_dependent_hooks(self):
        for name, hook in self.input_dependent_hooks():
            for input_dependent_hook_postfix, input_dependent_hook in hook.hooks.items():
                input_dependent_hook_name = input_dependent_hook.name
                if input_dependent_hook_name in self.mod_dict:
                    del self.mod_dict[input_dependent_hook_name]
                if input_dependent_hook_name in self.hook_dict:
                    del self.hook_dict[input_dependent_hook_name]
            hook.hooks = {}
        self.did_setup_input_dependent_hooks = False

        
def tokenize_if_str(tokenizer, input):
    if type(input) is str:
        input = tokenizer(input, return_tensors='pt')['input_ids']
    return input

def make_postfix_batch_split(b, l):
    return f".{b}.{l}"
    
def make_postfix(l):
    return f".{l}"

def input_dependent_postfixes_batch_split(input):
    Batch, L = input.size()
    for b,l in itertools.product(range(Batch), range(L)):
        postfix = make_postfix_batch_split(b=b, l=l)
        yield postfix
        
def input_dependent_postfixes(input):
    Batch, L = input.size()
    for b,l in itertools.product(range(Batch), range(L)):
        postfix = make_postfix(l=l)
        yield postfix

class HookedMamba(InputDependentHookedRootModule):
    def __init__(self, cfg):
        super(HookedTransformer, self).__init__()
        self.cfg = cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        self.embedding = nn.Embedding(V, D)
        self.hook_embed = HookPoint()
        
        self.blocks = nn.ModuleList([HookedMambaBlock(cfg=cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(D)
        self.hook_norm = HookPoint() # [B,L,D]
        self.lm_head  = nn.Linear(D, V, bias=False)
        self.hook_logits = HookPoint() # [B,L,V]
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        super().setup()

        
    def forward(self, 
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "B L"],
            Float[torch.Tensor, "B L E"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: Optional[bool] = False,
        tokens: Optional[Int[torch.Tensor, "B L"]] = None,
        fast_conv=False,
        fast_ssm=False,
        warn_disabled_hooks=True,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    ) -> Union[
        None,
        Float[torch.Tensor, "B L V"],
        Loss,
        Tuple[Float[torch.Tensor, "B L V"], Loss],
    ]:
        fast_ssm = True
        fast_conv = True
        # make sure input is ids and not a str
        if type(input) is str:
            input = self.to_tokens(input=input, prepend_bos=prepend_bos, padding_side=padding_side)
        
        input = input.to(self.cfg.device)
        
        given_tokens = len(input.size()) <= 2
        
        if given_tokens:
            tokens = input
        
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L = input.size()

        if given_tokens:
            # [B,L,D]                         [B,L]
            input_embed         = self.embedding(input)
        else: #[B,L,D]      [B,L,D]
            input_embed         = input
        resid         = self.hook_embed(input_embed)
        
        stopping = False
        if start_at_layer is None:
            start_at_layer = 0
        else:
            stopping = True
        if stop_at_layer is None:
            stop_at_layer = self.cfg.n_layers
        else:
            stopping = True
        
        for layer in self.blocks[start_at_layer:stop_at_layer]:
            # [B,L,D]         [B,L,D]
            resid     = layer(resid, fast_conv=fast_conv, fast_ssm=fast_ssm, warn_disabled_hooks=warn_disabled_hooks)
        
        # we stop early, just return the resid
        if stopping:
            return resid
        
        # [B,L,D]                   [B,L,D]
        resid_normed     = self.norm( resid )
        resid_normed     = self.hook_norm(resid_normed) # [B,L,D]
        
        # [B,L,V]          [D->V]    [B,L,D]
        logits    = self.lm_head( resid_normed ) # no bias
        logits    = self.hook_logits(logits) # [B,L,V]
        
        if return_type is None:
            return None
        else:
            if return_type == "logits":
                return logits
            else:
                assert (
                    tokens is not None
                ), "tokens must be passed in if return_type is 'loss' or 'both'"
                loss = self.loss_fn(logits, tokens, per_token=loss_per_token)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
    
    @staticmethod
    def from_pretrained(pretrained_model_name, device='cuda'):
        cfg, state_dict = get_converted_model_from_hf(pretrained_model_name=pretrained_model_name, device=device)
        model = HookedMamba(cfg)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model


class HookedMambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ## Variables
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        
        self.hook_resid_pre = HookPoint() # [B,L,D]
        
        ## Process inputs
        self.norm      = RMSNorm(D)
        self.hook_normalized_input = HookPoint() # [B,L,D]
        
        self.skip_proj = nn.Linear(D, E, bias=False)
        self.hook_skip = HookPoint() # [B,L,E]
        self.in_proj   = nn.Linear(D, E, bias=False)
        self.hook_in_proj = HookPoint() # [B,L,E]
        
        ## Conv
        self.conv1d    = nn.Conv1d(
            in_channels=E,
            out_channels=E,
            bias=True,
            kernel_size=D_conv,
            groups=E,
            padding=D_conv - 1,
        )
        self.hook_conv = HookPoint()  # [B,L+D_conv-1,E]
        self.hook_conv_after_cutoff = HookPoint() # [B,L,E]
        self.hook_ssm_input = HookPoint() # [B,L,E]
        
        ## SSM Params
        self.W_delta_1 = nn.Linear(E, D_delta, bias=False)
        self.W_delta_2 = nn.Linear(D_delta, E, bias=True)
        self.W_B = nn.Linear(E, N, bias=False)
        self.W_C = nn.Linear(E, N, bias=False)
        
        self.A_log     = nn.Parameter(torch.log(torch.randn([E,N])))
        
        
        self.hook_h_start = HookPoint()     # [B,E,N]
        
        self.hook_delta_1 = HookPoint() # [B,L,D_delta]
        self.hook_delta_2 = HookPoint() # [B,L,E]
        self.hook_delta = HookPoint() # [B,L,E]
        
        self.hook_A_bar = HookPoint() # [B,L,E,N]
        self.hook_B = HookPoint()     # [B,L,N]
        self.hook_B_bar = HookPoint() # [B,L,E,N]
        
        self.hook_C = HookPoint()     # [B,L,N]
        
        self.hook_h = InputDependentHookPoint(input_dependent_postfixes)     # [B,E,N], with l param
        
        self.hook_y = HookPoint() # [B,L,E]

        self.W_D = nn.Parameter(torch.ones(E))
        self.hook_ssm_output = HookPoint() # [B,L,E]
        
        self.hook_after_skip = HookPoint() # [B,L,E]
        
        
        ## Project back out
        self.out_proj  = nn.Linear(E, D, bias=False)
        self.hook_out_proj = HookPoint() # [B,L,D]
        self.hook_resid_post = HookPoint() # [B,L,D]
        
    def has_hooks_in_inner_loop(self):
        for name, module in self.named_modules():
            if "InputDependentHookPoint" in str(type(module)):
                if len(module.fwd_hooks) > 0 or len(module.bwd_hooks) > 0:
                    return True
        return False
    
    def forward(self, resid, fast_conv=False, fast_ssm=False, warn_disabled_hooks=True):
        
        cfg = self.cfg
        D = cfg.d_model
        E = cfg.d_inner
        N = cfg.d_state
        D_delta = cfg.dt_rank
        D_conv = cfg.d_conv
        V = cfg.vocab_size
        
        Batch,L,D = resid.size()
        
        ###### Process inputs ######
        resid      = self.hook_resid_pre(resid) # [B,L,D]
        
        # [B,L,D]             [B,L,D]
        resid_norm = self.norm(  resid  )
        resid_norm = self.hook_normalized_input(resid_norm) # [B,L,E]
        
        # [B,L,E]          [D->E]     [B,L,D]
        skip       = self.skip_proj( resid_norm ) # no bias
        skip       = self.hook_skip(skip) # [B,L,E]
        
        # [B,L,E]          [D->E]   [B,L,D]
        x_in       = self.in_proj( resid_norm ) # no bias
        x_in       = self.hook_in_proj(x_in) # [B,L,E]
        
        ###### Conv ######
        # [B,E,L]
        x_conv     = rearrange(x_in, 'B L E -> B E L')
        if fast_conv:
            if warn_disabled_hooks:
                for hook in [self.hook_conv, self.hook_conv_after_cutoff]:
                    if len(hook.fwd_hooks) > 0 or len(hook.bwd_hooks) > 0:
                        print(f"warning: hook {hook.name} will not be called because fast_conv=True, pass warn_disabled_hooks=True to disable this warning")

            from causal_conv1d import causal_conv1d_fn
            # this does the silu and conv at same time
            # so sadly we miss some hooks if we do this
            # [B,E,L]
            x_conv_out = causal_conv1d_fn(
                x=x_conv,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )
            # [B,L,E]
            x         = rearrange(x_conv_out, 'B E L -> B L E')
        else:
            # [B,E,L+3]                 [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
            x_conv_out = self.conv1d(   x_conv   )
            # [B,L+3,E]            [B,E,L+3]
            x_conv_out = rearrange(x_conv_out, 'B E L -> B L E')
            x_conv_out = self.hook_conv(x_conv_out) # [B,L+3,E] 
            # [B,L,E]
            x_conv_out_cutoff = x_conv_out[:,:L,:]
            x_conv_out_cutoff = self.hook_conv_after_cutoff(x_conv_out_cutoff) # [B,L,E]

            ###### Nonlinearity  ######
            # [B,L,E]               [B,L,E]
            x         = F.silu( x_conv_out_cutoff )
        x         = self.hook_ssm_input(x) # [B,L,E]
        
        ###### SSM ######
       
        self.A = -torch.exp(self.A_log)
       
        ys = []
       
        # latent state, init to zeros
        h = torch.zeros([Batch,E,N], device=self.cfg.device)
        h = self.hook_h_start(h) 
        
        ### Compute the delta, A_bar, B_bar, and C ahead of time,
        ### since none of them depend on h
        
        ## Compute Delta ##
        # [B,L,D_delta] [E->D_delta]  [B,E]
        delta_1        = self.W_delta_1( x ) # no bias
        delta_1        = self.hook_delta_1(delta_1) # [B,L,D_delta]
        
        # [B,L,E]         [D_delta->E] [B,L,D_delta] 
        delta_2        = self.W_delta_2(  delta_1  ) # with bias
        delta_2        = self.hook_delta_2(delta_2) # [B,L,E]

        # [B,L,N]     [E->N]   [B,L,E]
        B           = self.W_B(   x   )
        B           = self.hook_B(B) # [B,L,N]

        ## C
        # this just applies E->N projection to each E-sized vector
        # [B,L,N]      [E->N]  [B,L,E]     
        C           = self.W_C(   x   ) # no bias
        C           = self.hook_C(C) # [B,L,N]

        inner_loop_hooks = [
                self.hook_delta,
                self.hook_A_bar,
                self.hook_B_bar,
                self.hook_y,
        ] + list(self.hook_h.hooks.values())
        

        if fast_ssm:
            import selective_scan_cuda

            if warn_disabled_hooks:
                for hook in inner_loop_hooks:
                    if len(hook.fwd_hooks) > 0 or len(hook.bwd_hooks) > 0:
                        print(f"warning: hook {hook.name} will not be called because fast_ssm=True, pass warn_disabled_hooks=True to disable this warning")

            # the cuda kernel is picky about shapes, rearrange things to make it happy
            
            # [B,E,L]
            skip_ssm_input = rearrange(skip, "B L E -> B E L")
            # [B,E,L]
            x_ssm_input = rearrange(x, "B L E -> B E L")
            # [B,E,L]
            delta_2_ssm_input = rearrange(delta_2, 'B L E -> B E L')
            # [B,1,N,L]
            B_ssm_input = rearrange(B, 'B L N -> B 1 N L')
            # [B,1,N,L]
            C_ssm_input = rearrange(C, "B L N -> B 1 N L")

            # hack because we applied bias above
            # it's a little slower but that's ok
            if not hasattr(self, "empty_bias"):
                self.empty_bias = torch.zeros(self.W_delta_2.bias.size(), device=self.cfg.device)

            # this does softplus(delta), discretization, inner loop, add x*D, and multiply softplus(skip)
            # all the stuff you see in the else clause below 
            y_apply_D_ssm_output, scan_intermediates, y_skip_ssm_output = selective_scan_cuda.fwd(
                                    x_ssm_input.contiguous(), # u
                                    delta_2_ssm_input.contiguous(), # delta
                                    self.A.contiguous(), # A 
                                    B_ssm_input.contiguous(), # B
                                    C_ssm_input.contiguous(), # C
                                    self.W_D.float(), # D
                                    skip_ssm_input.contiguous(), # z
                                    self.empty_bias, # delta_bias
                                    True) # delta_softplus
            
            ssm_output_has_hooks = len(self.hook_ssm_output.fwd_hooks) > 0 or len(self.hook_ssm_output.bwd_hooks) > 0
           
            # recompute this if there was a hook,
            # this is in case the hook modifies it
            if ssm_output_has_hooks:
                # [B,L,E]
                y_apply_D = rearrange(y_apply_D_ssm_output, "B E L -> B L E")
                y_apply_D = self.hook_ssm_output(y_apply_D) # [B,L,E]

                # [B,L,E]   [B,L,E]             [B,L,E]
                y_skip    = y_apply_D * F.silu(  skip  )
            else:
                # [B,L,E]
                y_skip = rearrange(y_skip_ssm_output, "B E L -> B L E")
        else:
           
            # [B,L,E]           [B,L,E]
            delta  = F.softplus(delta_2) 
            delta  = self.hook_delta(delta) # [B,L,E]
            
            ## Discretize A
            # [B,L,E,N]                    [B,L,E] [E,N]
            A_bar       = torch.exp(einsum(delta, self.A, 'b l e, e n -> b l e n'))
            A_bar       = self.hook_A_bar(A_bar) # [B,L,E,N]
            
            ## Discretize B (also, multiply by x ahead of time)
            # [B,L,E,N]          [B,L,E]  [B,L,N] 
            B_bar       = einsum( delta,    B,     'b l e, b l n -> b l e n')
            B_bar       = self.hook_B_bar(B_bar) # [B,L,E,N]
            
            # Now we do the recurrence
            ys = []
            
            h = torch.zeros([Batch,E,N], device=self.cfg.device)
            for l in range(L):
                
                def apply_hook(hook, value):
                    postfix = make_postfix(l=l)
                    if postfix in hook.hooks:
                        input_dependent_hook = hook.hooks[postfix]
                        input_dependent_hook.l = l
                        return input_dependent_hook(value)
                    else: # not setup, maybe called forward by itself?
                        return value
                
                # [B,E,N]   [B,E,N]     [B,E,N]          [B,E,N]          [B,E]
                h        =    h    *  A_bar[:,l,:,:]  + B_bar[:,l,:,:] * x[:,l].view(Batch, E, 1)
                h        = apply_hook(self.hook_h, h) # [B,E,N]
                
                # [B,E]    [B,E,N]       [B,N,1]   # this is like [E,N] x [N,1] for each batch
                y_l       =   h     @   C[:,l,:].view(Batch,N,1)
                # [B,E]              [B,E,1]
                y_l      =    y_l.view(Batch,E)
                ys.append(y_l)
                
            # we have lots of [B,E]
            # we need to stack them along the 1 dimension to get [B,L,E]
            y = torch.stack(ys, dim=1)
            y = self.hook_y(y) # [B,L,E]
            
            ###### Finish block ######
            
            # [B,L,E]  [B,L,E]    [B,L,E]       [E]
            y_apply_D =   y      +   x     *  self.W_D
            y_apply_D =  self.hook_ssm_output(y_apply_D) # [B,L,E]
                
            # [B,L,E]   [B,L,E]             [B,L,E]
            y_skip    = y_apply_D * F.silu(  skip  )

        y_skip    =  self.hook_after_skip(y_skip) # [B,L,E]
            
        # [B,L,D]         [E->D]   [B,L,E]
        y_out     = self.out_proj( y_skip ) # no bias
        y_out     = self.hook_out_proj(y_out) # [B,L,D]
    
        # [B,L,D]   [B,L,D]   [B,L,D]
        resid     = resid +  y_out
        resid     = self.hook_resid_post(resid) # [B,L,D]
        
        return resid




