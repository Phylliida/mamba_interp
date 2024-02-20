
# What hooks are available?

As a reminder:

```python
B = Batch = batch size
L = context len
D = d_model = 1024
E = d_inner = d_in = 2048
N = d_state = 16
D_delta = dt_rank = 64
D_conv = d_conv = 4
V = vocab_size = 50280
```


<details>
  <summary>Pre-reqs (silu, softplus, rmsnorm)</summary>

### Silu
$$\text{silu}(x) = x*\text{sigmoid}(x)$$

![silu](https://github.com/Phylliida/mamba_interp/blob/main/graphs/silu.png?raw=true)

### Sigmoid

$$\text{sigmoid}(x) = \frac{1}{1+e^{-x}}$$

![sigmoid](https://github.com/Phylliida/mamba_interp/blob/main/graphs/sigmoid.png?raw=true)

### Softplus

$$\text{softplus}(x) = \log(1+e^{x})$$

![softplus](https://github.com/Phylliida/mamba_interp/blob/main/graphs/softplus.png?raw=true)

Note: as softplus is basically linear for large x, after `x>20` implementations usually just turn it into $\text{softplus}(x) = x$

### RMSNorm

```python
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
```

</details>


It may be useful to just look at hooked_mamba.py. Still, here's each hook with a breif summary, in the order they are encountered:

## hook_embed : `[B,L,D]`

The embedded tokens

```
# [B,L,D]                         [B,L]
input_embed         = self.embedding(input)
resid         = self.hook_embed(input_embed)
```

## Hooks for each layer

This loop is ran over all the layers:

```python
for layer in self.blocks:
    resid = layer(resid)
```

Here are the hooks for each layer.

Replace `{layer}` with layer index, for example, `blocks.3.hook_skip` is the `hook_skip` for the 4th layer (they are zero-indexed)

### `blocks.{layer}.hook_resid_pre : [B,L,D]`

Layer input

```python
resid = self.hook_resid_pre(resid)
```

### `blocks.{layer}.hook_normalized_input : [B,L,D]`

Layer input after normalization (RMSNorm)

```python
resid_norm = self.norm(  resid  ) # RMSNorm
resid_norm = self.hook_normalized_input(  resid_norm  ) # [B,L,D]
```

### `blocks.{layer}.hook_skip : [B,L,E]`

Skip vector

```python
skip = self.skip_proj(resid_norm) # no bias
skip = self.hook_skip(skip) # [B,L,E]
```

### `blocks.{layer}.hook_in_proj : [B,L,E]`

Input, projected into d_inner (E) space

```python
x_in = self.in_proj(resid_norm) # no bias
x_in = self.hook_in_proj(x_in) # [B,L,E]
```

Note: You may be familiar with `in_proj` returning a `[B,L,2*E]` sized vector, which is then split into `x_in` and `skip`. In our implementation we split the `in_proj` from the original implementation into `skip_proj` and `in_proj`


### `blocks.{layer}.hook_conv : [B,L,E]`

Output of conv

```python
x_conv     = rearrange(x_in, 'B L E -> B E L')
# [B,E,L+d_conv-1]
x_conv_out = self.conv1d(   x_conv   )
# [B,L+d_conv-1,E]
x_conv_out = rearrange(x_conv_out, 'B E L -> B L E')
# Say d_conv is 4, this clipping makes the [:,l,:] pos a conv over (l-3, l-2, l-1, l) positions
# [B,L,E]
x_conv_out_cutoff = x_conv_out[:,:L,:]
x_conv_out_cutoff = self.hook_conv(x_conv_out_cutoff) # [B,L,E]
```

### `blocks.{layer}.hook_ssm_input : [B,L,E]`

The input to the ssm, this is the output of conv after applying silu (smooth relu)

```python
x = F.silu( x_conv_out_cutoff )
x = self.hook_ssm_input(x) # [B,L,E]
```

### `blocks.{layer}.hook_h_start : [B,E,N]`

The initial hidden state (always initialized to zero vector)

```python
h = torch.zeros([Batch,E,N], device=self.cfg.device)
h = self.hook_h_start(h) 
```

### `blocks.{layer}.hook_h_start : [B,E,N]`

The initial hidden state (always initialized to zero vector)

### `blocks.{layer}.hook_delta_1 : [B,L,D_delta]`

Delta is computed by projecting x into a `D_delta` sized space, and the projecting back into a `E` sized space. 

`delta_1` is that intermediate `D_delta` space

```python
# [B,L,D_delta] [E->D_delta]  [B,E]
delta_1        = self.W_delta_1( x ) # no bias
delta_1        = self.hook_delta_1(delta_1) # [B,L,D_delta]
```

### `blocks.{layer}.hook_delta_2 : [B,L,D_delta]`

`delta_2` is `delta_1` projected back into a `E` sized space.

```python
# [B,L,E]         [D_delta->E] [B,L,D_delta] 
delta_2        = self.W_delta_2(  delta_1  ) # with bias
delta_2        = self.hook_delta_2(delta_2) # [B,L,E]
```

Note: In the original implementation, `W_delta_2` is called `dt_proj`.

### `blocks.{layer}.hook_delta : [B,L,D_delta]`

This is `delta_2` after applying softplus.

Note,

$$softplus(x) = log(1+e^x)$$

softplus is basically a smooth relu.

```python
# [B,L,E]           [B,L,E]
delta  = F.softplus(delta_2) 
delta  = self.hook_delta(delta) # [B,L,E]
```

### `blocks.{layer}.hook_A : [E,N]`

This is just -exp of the learned parameter `A_log`

Note, this doesn't depend on the input

```python
A = -torch.exp(self.A_log)
A = self.hook_A(A) # [E,N]
```

### `blocks.{layer}.hook_A_bar : [B,L,E,N]`

Discretized `A`

```python
# [B,L,E,N]                    [B,L,E] [E,N]
A_bar       = torch.exp(einsum(delta, self.A, 'b l e, e n -> b l e n'))
A_bar       = self.hook_A_bar(A_bar) # [B,L,E,N]
```

### `blocks.{layer}.hook_B : [B,L,N]`

Input-dependent `B`

```python
# [B,L,N]     [E->N]   [B,L,E]
B           = self.W_B(   x   ) # no bias
B           = self.hook_B(B) # [B,L,N]
```

### `blocks.{layer}.hook_B_bar : [B,L,E,N]`

`B_bar` is Discretized `B`

```python
## Discretize B
# [B,L,E,N]          [B,L,E]  [B,L,N] 
B_bar       = einsum( delta,    B,     'b l e, b l n -> b l e n')
B_bar       = self.hook_B_bar(B_bar) # [B,L,E,N]
```


### `blocks.{layer}.hook_C : [B,L,N]`

Input-dependent `C` from the ssm

```python
# [B,L,N]      [E->N]  [B,L,E]     
C           = self.W_C(   x   ) # no bias
C           = self.hook_C(C) # [B,L,N]
```

### What is `W_delta_1`, `W_B`, and `W_C`?

In the original implementation, all three of these are put into one matrix called `x_proj`. In our implementation, we split this apart into those three matrices (this is numerically equivalent).

### `blocks.{layer}.hook_h.{position} : [B,E,N]`

The hidden state of the ssm after processing token at position {position}

Note, there is a seperate hook for each position.

For example, `blocks.3.hook_h.2` is a hook for the hidden state after processing the 0th token, 1th token, and 2nd token.

You can use this just like any other hook, see Activation Patching for an example of patching on `hook_h` at a specific position in the sequence.

```python
ys = []
h = torch.zeros([Batch,E,N], device=self.cfg.device)
for l in range(L):
    # [B,E,N]   [B,E,N]     [B,E,N]          [B,E,N]          [B,E]
    h        =    h    *  A_bar[:,l,:,:]  + B_bar[:,l,:,:] * x[:,l].view(Batch, E, 1)
    
    # hook_h is input-dependent
    # that means it has one child hook for each l
    # thus we need to pass it a postfix ".4" that it'll look up
    # the child hook for
    postfix = make_postfix(l=l)
    h        = self.hook_h(h, postfix=postfix) # [B,E,N]
    
    # [B,E]    [B,E,N]       [B,N,1]   # this is like [E,N] x [N,1] for each batch
    y_l       =   h     @   C[:,l,:].view(Batch,N,1)
    # [B,E]              [B,E,1]
    y_l      =    y_l.view(Batch,E)
    ys.append(y_l)
```

### `blocks.{layer}.hook_y : [B,L,E]`

Output of the ssm (before adding $x*D$)

```python
# we have lots of [B,E]
# we need to stack them along the 1 dimension to get [B,L,E]
y = torch.stack(ys, dim=1)
y = self.hook_y(y) # [B,L,E]
```

### `blocks.{layer}.hook_ssm_output : [B,L,E]`

Output of the ssm (after adding $x*D$)

```python
# [B,L,E]     [B,L,E]    [B,L,E]       [E]
y_ssm_output =   y      +   x     *  self.W_D
y_ssm_output =  self.hook_ssm_output(y_ssm_output) # [B,L,E]
```

Note: In the original implementation, `W_D` is called `D`.

### `blocks.{layer}.hook_after_skip : [B,L,E]`

Output of the layer after mulitplying by silu(skip vector)

(see above for definition of skip vector)

Note: 

$$silu(x) = x*\text{sigmoid}(x)$$

Silu is like a soft relu, though it can go negative.

```python
# [B,L,E]   [B,L,E]                     [B,L,E]
y_after_skip    = y_ssm_output * F.silu(  skip  )
y_after_skip    =  self.hook_after_skip(y_after_skip) # [B,L,E]
```

### `blocks.{layer}.hook_out_proj : [B,L,D]`

Output of this layer, before adding to residual

```python
# [B,L,D]         [E->D]       [B,L,E]
y_out     = self.out_proj( y_after_skip ) # no bias
y_out     = self.hook_out_proj(y_out) # [B,L,D]
```

### `blocks.{layer}.hook_resid_post : [B,L,D]`

Resulting residual after adding output from this layer

```python
# [B,L,D]   [B,L,D]   [B,L,D]
resid     = resid +  y_out
resid     = self.hook_resid_post(resid) # [B,L,D]
return resid
```

## Final model output hooks

As mentioned above, after going through all the layers via:

```python
for layer in self.blocks:
    resid = layer(resid)
```

We can finally do

### `hook_norm : [B,L,D]`

Resulting residual stream after applying the norm

```
# [B,L,D]                   [B,L,D]
resid_normed     = self.norm( resid )
resid_normed     = self.hook_norm(resid_normed) # [B,L,D]
```

### `hook_logits : [B,L,V]`

The output model logits

```
# [B,L,V]          [D->V]    [B,L,D]
logits    = self.lm_head( resid_normed ) # no bias
logits    = self.hook_logits(logits) # [B,L,V]
```

# Loading a model

## From a pretrained model on huggingface:

```python
from hooked_mamba import HookedMamba

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
model = HookedMamba.from_pretrained("state-spaces/mamba-370m", device='cuda', tokenizer=tokenizer)
```

## From a config and state dict:

```python
import hooked_mamba

state_dict = old_model.state_dict() # your state dict from a model using https://github.com/state-spaces/mamba
cfg = { # your config from a model using https://github.com/state-spaces/mamba
    "d_model": 1024,
    "n_layer": 48,
    "vocab_size": 50277,
    "ssm_cfg": {
        "d_state": 16,
        "d_conv": 2,
        "expand": 2,
    },
    "rms_norm": true,
    "residual_in_fp32": true,
    "fused_add_norm": true,
    "pad_vocab_size_multiple": 8
}

# we need to convert to the format used by hooked mamba
# this does:
# unpacking of combined matrices:
#            in_proj -> [in_proj, skip_proj]
#            x_proj  -> [W_delta_1, W_B, W_C]
# renaming:
#            dt_proj -> W_delta_2
#            D       -> W_D
#            norm_f  -> norm
# it also does some moving around to make it look like HookedTransformer:
# the model is lm_head

hooked_mamba_cfg = hooked_mamba.convert_original_config_to_hooked_mamba_config(cfg, device=device)
hooked_mamba_state_dict = hooked_mamba.convert_original_state_dict_to_hooked_format(state_dict)

# Note: tokenizer is optional, it's only used if you pass in a string
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

model = hooked_mamba.HookedMamba(cfg=hooked_mamba_cfg, device='cuda', tokenizer=tokenizer)
model.load_state_dict(hooked_mamba_state_dict)
```

## Initialize a model from scratch (with correct parameter initialization)

```python
cfg = MambaCfg(
    d_model=1024,
    n_layer=48,
    vocab_size=50277,
    d_state=16,
    d_conv=4,
    expand=2
    initializer_config=MambaInitCfg(
        initializer_range = 0.02, # Used for embedding layer
        rescale_prenorm_residual = True,
        n_residuals_per_layer = 1,  # Change to 2 if we have MLP
        dt_init = 'random', # other option is "constant"
        dt_scale = 1.0,
        dt_min = 0.001,
        dt_max = 0.1,
        dt_init_floor = 1e-4
    )
)

model = hooked_mamba.HookedMamba(cfg=cfg, device='cuda', initialize_params=True)
```

## Port HookedMamba to other libraries using the original format:

```python
# model is a HookedMamba
cfg_dict = hooked_mamba.convert_hooked_mamba_config_to_original_config(hooked_mamba_cfg=model.cfg)
state_dict = hooked_mamba.convert_hooked_state_dict_to_original_format(cfg=model.cfg, state_dict=model.state_dict())
```

# Activation Patching 

## Activation Patching on the hidden state

There is a seperate hook for each position, so we can simply write a hook that will substitute the patched `h` at the target position only:

```python
def h_patching_hook(
    h: Float[torch.Tensor, "B E N"],
    hook: HookPoint,
    layer: int,
    position: int,
) -> Float[torch.Tensor, "B E N"]:
    return corrupted_activations[hook.name]
```

This will result in modified `h` after that position, as desired.

Here is the full code:

```python
from tqdm.notebook import tqdm
from functools import partial
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import torch
import plotly.express as px

import hooked_mamba
model = hooked_mamba.HookedMamba.from_pretrained("state-spaces/mamba-370m", device='cuda')

prompt_uncorrupted = 'Lately, Emma and Shelby had fun at school. Shelby gave an apple to'
prompt_corrupted = 'Lately, Emma and Shelby had fun at school. Emma gave an apple to'
uncorrupted_answer = ' Emma' # note the space in front is important
corrupted_answer = ' Shelby'

prompt_uncorrupted_tokens = model.to_tokens(prompt_uncorrupted)
prompt_corrupted_tokens = model.to_tokens(prompt_corrupted)

L = len(prompt_uncorrupted_tokens[0])
if len(prompt_corrupted_tokens[0]) != len(prompt_uncorrupted_tokens[0]):
    raise Exception("Prompts are not the same length") # feel free to comment this out, you can patch for different sized prompts its just a lil sus

# logits should be [B,L,V] 
def uncorrupted_logit_minus_corrupted_logit(logits, uncorrupted_answer, corrupted_answer):
    uncorrupted_index = model.to_single_token(uncorrupted_answer)
    corrupted_index = model.to_single_token(corrupted_answer)
    return logits[0, -1, uncorrupted_index] - logits[0, -1, corrupted_index]

# [B,L,V]
corrupted_logits, corrupted_activations = model.run_with_cache(prompt_corrupted_tokens)
corrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=corrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# [B,L,V]
uncorrupted_logits = model(prompt_uncorrupted_tokens)
uncorrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=uncorrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# diff is logit of uncorrupted_answer - logit of corrupted_answer
# we expect corrupted_diff to have a negative value (as corrupted should put high pr on corrupted_answer)
# we expect uncorrupted to have a positive value (as uncorrupted should put high pr on uncorrupted_answer)
# thus we can treat these as (rough) min and max possible values
min_logit_diff = corrupted_logit_diff
max_logit_diff = uncorrupted_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)

# 'blocks.{layer}.hook_h.{pos}' is the recurrent state of that layer after processing tokens at and before pos position
def h_patching_hook(
    h: Float[torch.Tensor, "B E N"],
    hook: HookPoint,
    layer: int,
    position: int,
) -> Float[torch.Tensor, "B E N"]:
    return corrupted_activations[hook.name]
    
patching_result_logits = torch.zeros((model.cfg.n_layers, L), device=model.cfg.device)
for layer in tqdm(range(model.cfg.n_layers)):
    for position in range(L):
        patching_hook_name = f'blocks.{layer}.hook_h.{position}'
        patching_hook = partial(h_patching_hook, layer=layer, position=position)
        # [B,L,V]
        patched_logits = model.run_with_hooks(prompt_uncorrupted_tokens, fwd_hooks=[
            (patching_hook_name, patching_hook)
        ])
        
        patched_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=patched_logits,
                                                                     uncorrupted_answer=uncorrupted_answer,
                                                                     corrupted_answer=corrupted_answer)
        # normalize it so
        # 0 means min_logit_diff (so 0 means that it is acting like the corrupted model)
        # 1 means max_logit_diff (so 1 means that it is acting like the uncorrupted model)
        normalized_patched_logit_diff = (patched_logit_diff-min_logit_diff)/(max_logit_diff - min_logit_diff)
        # now flip them, since most interventions will do nothing and thus act like uncorrupted model, visually its better to have that at 0
        # so now
        # 0 means that it is acting like the uncorrupted model
        # 1 means that it is acting like the corrupted model
        normalized_patched_logit_diff = 1.0 - normalized_patched_logit_diff
        patching_result_logits[layer, position] = normalized_patched_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)
token_labels = []
for index, (corrupted_token, uncorrupted_token) in enumerate(zip(corrupted_str_tokens, uncorrupted_str_tokens)):
    if corrupted_token == uncorrupted_token:
        token_labels.append(f"{corrupted_token}_{index}")
    else:
        token_labels.append(f"{uncorrupted_token}->{corrupted_token}_{index}")

# display outputs
px.imshow(utils.to_numpy(patching_result_logits), x=token_labels, color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":"Position", "y":"Layer"}).show()
```

## Patching on the ssm outputs 

Because there is a single hook called for all positions, we now need to only intervene on the specific position

```python
# 'blocks.{layer}.hook_ssm_output' is after the ssm and after doing  y   +   x     *  self.W_D
def ssm_output_patching_hook(
    ssm_output: Float[torch.Tensor, "B L E"],
    hook: HookPoint,
    position: int,
    layer: int
) -> Float[torch.Tensor, "B L E"]:
    # only intervene on the specific pos
    corrupted_ssm_output = corrupted_activations[hook.name]
    ssm_output[:, position, :] = corrupted_ssm_output[:, position, :]
    return ssm_output
```

Here is the full code

```python
from tqdm.notebook import tqdm
from functools import partial
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import torch
import plotly.express as px

import hooked_mamba
model = hooked_mamba.HookedMamba.from_pretrained("state-spaces/mamba-370m", device='cuda')

prompt_uncorrupted = 'Lately, Emma and Shelby had fun at school. Shelby gave an apple to'
prompt_corrupted = 'Lately, Emma and Shelby had fun at school. Emma gave an apple to'
uncorrupted_answer = ' Emma' # note the space in front is important
corrupted_answer = ' Shelby'

prompt_uncorrupted_tokens = model.to_tokens(prompt_uncorrupted)
prompt_corrupted_tokens = model.to_tokens(prompt_corrupted)

L = len(prompt_uncorrupted_tokens[0])
if len(prompt_corrupted_tokens[0]) != len(prompt_uncorrupted_tokens[0]):
    raise Exception("Prompts are not the same length") # feel free to comment this out, you can patch for different sized prompts its just a lil sus

# logits should be [B,L,V] 
def uncorrupted_logit_minus_corrupted_logit(logits, uncorrupted_answer, corrupted_answer):
    uncorrupted_index = model.to_single_token(uncorrupted_answer)
    corrupted_index = model.to_single_token(corrupted_answer)
    return logits[0, -1, uncorrupted_index] - logits[0, -1, corrupted_index]

# [B,L,V]
corrupted_logits, corrupted_activations = model.run_with_cache(prompt_corrupted_tokens)
corrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=corrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# [B,L,V]
uncorrupted_logits = model(prompt_uncorrupted_tokens)
uncorrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=uncorrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# diff is logit of uncorrupted_answer - logit of corrupted_answer
# we expect corrupted_diff to have a negative value (as corrupted should put high pr on corrupted_answer)
# we expect uncorrupted to have a positive value (as uncorrupted should put high pr on uncorrupted_answer)
# thus we can treat these as (rough) min and max possible values
min_logit_diff = corrupted_logit_diff
max_logit_diff = uncorrupted_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)

token_labels = []
for index, (corrupted_token, uncorrupted_token) in enumerate(zip(corrupted_str_tokens, uncorrupted_str_tokens)):
    if corrupted_token == uncorrupted_token:
        token_labels.append(f"{corrupted_token}_{index}")
    else:
        token_labels.append(f"{uncorrupted_token}->{corrupted_token}_{index}")

# 'blocks.{layer}.hook_h.{pos}' is the recurrent state of that layer after processing tokens at and before pos position
def ssm_output_patching_hook(
    ssm_output: Float[torch.Tensor, "B L E"],
    hook: HookPoint,
    position: int,
    layer: int
) -> Float[torch.Tensor, "B L E"]:
    # only intervene on the specific pos
    corrupted_ssm_output = corrupted_activations[hook.name]
    ssm_output[:, position, :] = corrupted_ssm_output[:, position, :]
    return ssm_output
    
patching_result_logits = torch.zeros((model.cfg.n_layers, L), device=model.cfg.device)
for layer in tqdm(range(model.cfg.n_layers)):
    for position in range(L):
        patching_hook_name = f'blocks.{layer}.hook_ssm_output'
        patching_hook = partial(ssm_output_patching_hook, layer=layer, position=position)
        # [B,L,V]
        patched_logits = model.run_with_hooks(prompt_uncorrupted_tokens, fwd_hooks=[
            (patching_hook_name, patching_hook)
        ])
        
        patched_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=patched_logits,
                                                                     uncorrupted_answer=uncorrupted_answer,
                                                                     corrupted_answer=corrupted_answer)
        # normalize it so
        # 0 means min_logit_diff (so 0 means that it is acting like the corrupted model)
        # 1 means max_logit_diff (so 1 means that it is acting like the uncorrupted model)
        normalized_patched_logit_diff = (patched_logit_diff-min_logit_diff)/(max_logit_diff - min_logit_diff)
        # now flip them, since most interventions will do nothing and thus act like uncorrupted model, visually its better to have that at 0
        # so now
        # 0 means that it is acting like the uncorrupted model
        # 1 means that it is acting like the corrupted model
        normalized_patched_logit_diff = 1.0 - normalized_patched_logit_diff
        patching_result_logits[layer, position] = normalized_patched_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)
token_labels = []
for index, (corrupted_token, uncorrupted_token) in enumerate(zip(corrupted_str_tokens, uncorrupted_str_tokens)):
    if corrupted_token == uncorrupted_token:
        token_labels.append(f"{corrupted_token}_{index}")
    else:
        token_labels.append(f"{uncorrupted_token}->{corrupted_token}_{index}")

# display outputs
px.imshow(utils.to_numpy(patching_result_logits), x=token_labels, color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":"Position", "y":"Layer"}).show()
```

# Optimizations:

If the above is too slow, you can pass in

```python
model.run_with_hooks(prompt_uncorrupted_tokens, fwd_hooks=[
            (patching_hook_name, patching_hook)
    ], fast_ssm=True, fast_conv=True)
```

## `fast_conv=True`

`fast_conv=True` uses the cuda kernel from [https://github.com/Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). To install it you can do

```
pip install causal_conv1d
```

Note that using `fast_conv=True` will disable the `hook_conv`, because the cuda kernel does the `silu` and `conv1d` at the same time.

As a reminder, here's the pure pytorch version:

### `blocks.{layer}.hook_conv : [B,L,E]`

Output of conv

```python
x_conv     = rearrange(x_in, 'B L E -> B E L')
# [B,E,L+d_conv-1]
x_conv_out = self.conv1d(   x_conv   )
# [B,L+d_conv-1,E]
x_conv_out = rearrange(x_conv_out, 'B E L -> B L E')
# Say d_conv is 4, this clipping makes the [:,l,:] pos a conv over (l-3, l-2, l-1, l) positions
# [B,L,E]
x_conv_out_cutoff = x_conv_out[:,:L,:]
x_conv_out_cutoff = self.hook_conv(x_conv_out_cutoff) # [B,L,E]
```

### `blocks.{layer}.hook_ssm_input : [B,L,E]`

The input to the ssm, this is the output of conv after applying silu (smooth relu)

```python
x = F.silu( x_conv_out_cutoff )
x = self.hook_ssm_input(x) # [B,L,E]
```

Wheras here's what happens if `fast_conv=True`:

### `blocks.{layer}.hook_conv : [B,L,E]`

(not available)

### `blocks.{layer}.hook_ssm_input : [B,L,E]`

Same as above, this is input to the ssm, which is the output of conv after applying silu (smooth relu)

```python
from causal_conv1d import causal_conv1d_fn
x_conv     = rearrange(x_in, 'B L E -> B E L')
# this does the silu and conv at same time
# [B,E,L]
x_conv_out = causal_conv1d_fn(
    x=x_conv,
    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
    bias=self.conv1d.bias,
    activation="silu",
)
# [B,L,E]
x         = rearrange(x_conv_out, 'B E L -> B L E')  
x = self.hook_ssm_input(x) # [B,L,E]
```

## `fast_ssm=True`

This uses the cuda kernel `selective_scan_cuda` from [the official mamba repo](https://github.com/state-spaces/mamba). To install it:

```
git clone https://github.com/state-spaces/mamba
cd mamba
pip install -e .
```

I recommend looking at hooked_mamba.py if you are interested in how this is called.

Using this will disable a few hooks:


### `blocks.{layer}.hook_delta : [B,L,D_delta]`

This is `delta_2` after applying softplus.

### `blocks.{layer}.hook_A_bar : [B,L,E,N]`

`A_bar` is Discretized `A`

### `blocks.{layer}.hook_B_bar : [B,L,E,N]`

`B_bar` is Discretized `B`

### `blocks.{layer}.hook_y : [B,L,E]`

Output of the ssm (before adding $x*D$)

