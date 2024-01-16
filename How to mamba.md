
<details>
  <summary>Background</summary>

The inspiration for mamba (and state space models in general) is mapping a 1D function $x(t) \in \mathbb{R}$ to a 1D function $y(t) \in \mathbb{R}$ via a N-dimensional latent space $h \in \mathbb{R}^N$.

Specifically, we have the following:

$$\stackrel{[N]}{\dot{h}(t)} = \stackrel{[N,N]}{A}\stackrel{[N]}{h(t)} + \stackrel{[N,1]}{B}\stackrel{[1]}{x(t)}$$
$$\stackrel{[1]}{y(t)} = \stackrel{[1,N]}{C}\stackrel{[N]}{h(t)}$$

(the $[_,_]$ stuff above the variables is just specifying the dimensions)

This is a diffeq, where $\dot{h}(t)$ is the derivitave of $\dot{h}(t)$ with respect to time.

If we have an initial $h_0$, we can approximate our diffeq this way:

$$\stackrel{[N]}{h_t} = \stackrel{[1]}{\Delta}\stackrel{[N,N]}{A}\stackrel{[N]}{h_{i-1}} + \stackrel{[1]}{\Delta}\stackrel{[N,1]}{B}\stackrel{[1]}{x_t}$$
$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

Where $\Delta$ is a small timestep, like $0.001$.

This approximation is like, if a character has a velocity $v$ and a position $p_0$, to find the position after $\Delta$ time we do $p_1 = \Delta v + p_0$, then we do $p_2 = \Delta v + p_1$, etc. In general:

$$p_t = \Delta v + p_{t-1}$$

We are doing the same sort of thing for $h(t)$.

</details>

<details>
  <summary>Discretization</summary>

Above, we have:

$$\stackrel{[N]}{h_t} = \stackrel{[1]}{\Delta}\stackrel{[N,N]}{A}\stackrel{[N]}{h_{i-1}} + \stackrel{[1]}{\Delta}\stackrel{[N,1]}{B}\stackrel{[1]}{x_t}$$
$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

We can write this as

$$\bar{A} = \Delta A$$

$$\bar{B} = \Delta B$$

So we get

$$\stackrel{[N]}{h_t} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_t}$$
$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

This process of turning $A$ and $B$ into $\bar{A}$ and $\bar{B}$ is called **Discretization**.

It turns out there are lots of ways to do this! Here are some options for **discretization rules**:

#### Zero-Order Hold (ZOH)

$$\bar{A} = \exp(\Delta A)$$

$$\bar{B} = (\Delta A)^{-1} (\exp(\Delta A)-I) \Delta B$$

#### Generalized Bilinear Transform (GBT)

$$\bar{A} = (I-\alpha \Delta A)^{-1}(I+(1-\alpha)\Delta A)$$

$$\bar{B} = \Delta (I-\alpha \Delta A)^{-1} B$$

If $\alpha=0$, this is called the **Euler Method** or the **Forward Euler Method**:

$$\bar{A} = I+\Delta A$$

$$\bar{B} = \Delta B$$

If $\alpha=\frac{1}{2}$ this is known as the **Bilinear Method**

$$\bar{A} = (I-\frac{1}{2}\Delta A)^{-1}(I+\frac{1}{2}\Delta A)$$

$$\bar{B} = \Delta (I-\frac{1}{2} \Delta A)^{-1} B$$

If $\alpha=1$ this is known as the **Backward Euler Method**

$$\bar{A} = (I-\Delta A)^{-1}$$

$$\bar{B} = \Delta (I-\Delta A)^{-1} B$$

## Discretization rule used in Mamba

Mamba uses a discretization rule that's a mix of Zero-Order Hold and Euler Method:

$$\bar{A} = \exp(\Delta A)$$

(element-wise exp, *not* a matrix exponential)

$$\bar{B} = \Delta B$$

</details>

<details>
  <summary>But x is multi-dimensional?</summary>

To summarize, in mamba we have

$$\bar{A} = exp(\Delta A)$$

$$\bar{B} = \Delta B$$

And then we get our output $y_t$ via:

$$\stackrel{[N]}{h_t} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_t}$$

$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

To handle language, each term $x_i$ corresponds to a token in our context. For example, if our inner dim is 5 and our context is "eat apple bees", we will get

``` python
[0.86,  -0.27, 1.65, 0.05,  2.34] "eat"
[-1.84, -1.79, 1.10, 2.38,  1.76] "apple"
[1.05,  -1.78, 0.16, -0.30, 1.91] "bees"
```

However, these are multi-dimensional, wheras our $x_t$ from above is one-dimensional.

To address this, mamba has a seperate state space model occuring for each component. For example, we will start with the first component:

```
x=[0.86, -1.84, 1.05]
```

Given these we can use

$$\stackrel{[N]}{h_t} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_t}$$

To find the N-dimensional $h_1, h_2, h_3$. (note, by convention we always start with $h_0=$ the zero vector). Say they are (let N=3):

```python
h_1=[1.0, -0.45, 2.0]
h_2=[4.3, -2.3,  4.4]
h_3=[0.2, -4.1, -0.2]
```

Now we can use 

$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

To find $y_1, y_2, y_3$.

Once we are done, we do this again for the next component:

```
x=[-0.27, -1.79, -1.78]
```

Given these we can use

$$\stackrel{[N]}{h_t} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_t}$$

To find the N-dimensional $h_1, h_2, h_3$. (note, by convention we always start with $h_0=$ the zero vector). Say they are (let N=3):

etc.

This might seem strange, and it is. However, it's not entirely unreasonable because due to selection (see the Selection section below) $\Delta, A, B, C$ are a function of the entire vector, not just the current component being used.
  
</details>

<details>
  <summary>Expanded vs Vectorized</summary>

Below, I wrote out the inner loop of Mamba in two ways. Both are equivalent, they are just different ways of looking at it.

"Expanded" does a seperate state space model for each component of the $E$-sized vectors. This is what's actually happening, so I think it's useful to see it like this first.

"Vectorized" computes all $E$ state space models at the same time. Numerically it's the same as "Expanded", but might be useful for reference (plus it's much faster)

</details>

<details>
  <summary>Selection</summary>

Above, we have 

$$\bar{A} = exp(\Delta A)$$

$$\bar{B} = \Delta B$$

And then we get our output $y_t$ via:

$$\stackrel{[N]}{h_t} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_t}$$

$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t}$$

Really, we do this seperately for each component $e$, so I'll write this

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N,N]}{\bar{A}}\stackrel{[N]}{h_{i-1,e}} + \stackrel{[N,1]}{\bar{B}}\stackrel{[1]}{x_{t,e}}$$

$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_{t,e}}$$

The way this is specified, $\Delta, A, B$, and $C$ are fixed. The idea behind Selection is to let these vary over time, by making them dependent on $x_t$. Specifically, let:

$$\stackrel{[1]}{\Delta_{t,e}} = \text{softplus}(\stackrel{[E]}{x_{t}} \cdot \stackrel{[E]}{W_{\Delta}[:,e]} + \stackrel{[1]}{B_{\Delta}[e]})$$

$$\stackrel{[N]}{\bar{A_{t,e}}} = \exp(\stackrel{[1]}{\Delta_{t,e}} \stackrel{[N]}{A[e]})$$

$$\stackrel{[N]}{B_{t}} = \stackrel{[N,E]}{W_B}\stackrel{[E]}{x_t}$$

$$\stackrel{[N]}{\bar{B_{t,e}}} = \stackrel{[1]}{\Delta_{t,e}}\stackrel{[N]}{B_{t}}$$

$$\stackrel{[N]}{C_t} = \stackrel{[N,E]}{W_C}\stackrel{[E]}{x_t}$$

Where $\stackrel{[E,E]}{W_{\Delta}}, \stackrel{[E]}{B_{\Delta}}, \stackrel{[E,N]}{A}, \stackrel{[N,E]}{W_B}, \stackrel{[N,E]}{W_C}$ are learned parameters, and $\text{softplus}(x) = \log(1+e^{x})$

This gives us

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N]}{\bar{A_{t,e}}}\stackrel{[N]}{h_{t-1,e}} + \stackrel{[N,1]}{\bar{B_{t,e}}}\stackrel{[1]}{x_{t,e}}$$

$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C_t}\stackrel{[N]}{h_{t,e}}$$

You may have noticed that $\bar{A}$ is now a vector $([N])$ instead of a matrix ($[N,N]$). I'm not sure why they do it that way, but that's what they do. This means that $$\stackrel{[N]}{\bar{A_{t,e}}}\stackrel{[N]}{h_{t-1,e}}$$ is just an element-wise product (hadamard product)

Anyway, expanded out, this gives us

$$\stackrel{[N]}{h_{t,e}} = \exp(\stackrel{[1]}{\Delta_{t,e}} \stackrel{[N]}{A[e]})\stackrel{[N]}{h_{t-1,e}} + (\stackrel{[1]}{\Delta_{t,e}}\stackrel{[N,E]}{W_B}\stackrel{[E]}{x_t})\stackrel{[1]}{x_{t,e}}$$

$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C_t}\stackrel{[N]}{h_{t,e}}$$
  
</details>


<details>
  <summary>Imports</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
```

</details>

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

## Mamba

### Variables (example values from mamba-370m)
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

### Params
```python
mamba = object()

mamba.embedding = nn.Embedding(V, D)
mamba.layers = [object() for _ in range(n_layer)]
mamba.norm = RMSNorm(D)
mamba.lm_head   = nn.Linear(D, V, bias=False)

## Params for each Layer/MambaBlock ##
for layer in range(n_layer):
    ## Process inputs
    layer.norm      = RMSNorm(D)
    layer.skip_proj = nn.Linear(D, E, bias=False)
    layer.in_proj   = nn.Linear(D, E, bias=False)
    
    ## Conv
    layer.conv1d    = nn.Conv1d(
        in_channels=E,
        out_channels=E,
        bias=True,
        kernel_size=D_conv,
        groups=E,
        padding=D_conv - 1,
    )
    
    ## SSM Params
    layer.W_delta_1 = nn.Linear(E, D_delta, bias=False)
    layer.W_delta_2 = nn.Linear(D_delta, E, bias=True)
    layer.W_B = nn.Linear(E, N, bias=False)
    layer.W_C = nn.Linear(E, N, bias=False)
    
    layer.A_log     = nn.Parameter(torch.log(torch.randn([E,N])))
    layer.W_D = nn.Parameter(torch.ones(E))
    
    ## Project back out
    layer.out_proj  = nn.Linear(E, D, bias=False)
```


<details>
<summary> Note: W_delta_1, W_B, and W_C are stored together in x_proj </summary>

Here's how you extract them:
```python
# maps [B,L,E] -> [B,L,D_delta+2*N], then we split into [B,L,D_delta], [B,L,N], [B,L,N]
W = layer.x_proj.weight.T
# pull them out
W_delta_1 = W[:,:D_delta]
W_B = W[:,D_delta:D_delta+N]
W_C = W[:,D_delta+N:]
```
</details>


### Implementation

```python
#           [B,L]
def run_mamba(mamba, input_ids):
    # [B,L,D]                         [B,L]
    resid         = mamba.embedding(input_ids)
    
    for layer in mamba.layers:
        ## Process inputs ##
        # [B,L,D]  [B,L,D]
        x         = resid
        # [B,L,D]             [B,L,D]
        x         = layer.norm(  x  )
        # [B,L,E]         [D->E]  [B,L,D]
        skip      = layer.skip_proj(  x  ) # no bias
        # [B,L,E]         [D->E] [B,L,D]
        x         = layer.in_proj(  x  ) # no bias
        
        ## Conv ##
        # [B,E,L]
        x         = rearrange(x, 'B L E -> B E L')
        # [B E L]                [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
        x         = layer.conv1d(   x   )[:, :, :L]
        # [B,L,E]
        x         = rearrange(x, 'B E L -> B L E')
```
<details>
  <summary>Conv Explanation</summary>

<details>
  <summary>General 1D Conv Explanation</summary>

The basic unit of a Conv1D is applying a kernel to a sequence.

For example, say my kernel is `[-1,2,3]` and my sequence is `[4,5,6,7,8,9]`.

Then to apply that kernel, I move it across my sequence like this:
```python
[*4,5,6*, 7,8,9]
-1*4 + 2*5 + 3*6 = 24

[4, *5,6,7*, 8,9]
-1*5 + 6*2 + 3*7 = 28

[4,5, *6,7,8*, 9]
-1*6 + 2*7 + 3*8 = 32

[4,5,6, *7,8,9*]
-1*7 + 2*8 + 3*9 = 36
```

So our resulting vector would be `[24, 28, 32, 36]`

It's annoying that our output is smaller than our input, so we can pad our input first:

`[0,0,4,5,6,7,8,9,0,0]`

Now we get
```python
[*0,0,4* ,5,6,7,8,9,0,0]
-1*0 + 2*0 + 3*4 = 12

[0, *0,4,5*, 6,7,8,9,0,0]
-1*0 + 2*4 + 3*5 = 23

[0,0, *4,5,6*, 7,8,9,0,0]
-1*4 + 2*5 + 3*6 = 24

[0,0,4, *5,6,7*, 8,9,0,0]
-1*5 + 6*2 + 3*7 = 28

[0,0,4,5, *6,7,8*, 9,0,0]
-1*6 + 2*7 + 3*8 = 32

[0,0,4,5,6, *7,8,9*, 0,0]
-1*7 + 2*8 + 3*9 = 36

[0,0,4,5,6,7, *8,9,0*, 0]
-1*8 + 2*9 + 3*0 = 10

[0,0,4,5,6,7,8, *9,0,0*]
-1*9 + 2*0 + 3*0 = -9
```

So our result is `[12, 23, 24, 28, 32, 36, 10, -9]`

Now this is longer than we need, so we'll cut off the last two, giving us

`[12, 23, 24, 28, 32, 36]`

</details>

<details>
  <summary>Worked Conv Example</summary>

Mamba conv is defined as
```python
layer.conv1d = nn.Conv1d(
        in_channels=E,
        out_channels=E,
        bias=True,
        kernel_size=D_conv,
        groups=E,
        padding=D_conv - 1,
    )
```
In this example, I will set:
```python
E = d_inner = 5 (for large models this is 2048-5012)
D_conv = kernel_size = 4 (for large models this is 4)
L = context size = 3
```
In practice, `D_conv=4` and `E` is around `2048-5012`.

Our input to to mamba's conv1d is of size [B, E, L]. I'll do a single batch.

Because `groups = E = 5`, we have `5` filters:

```python
[ 0.4,  0.7, -2.1,  1.1] filter 0 with bias [0.2]
[ 0.1, -0.7, -0.3,  0.0] filter 1 with bias [-4.3]
[-0.7,  0.9,  1.0,  0.9] filter 2 with bias [-0.3]
[-0.5, -0.8, -0.1,  1.5] filter 3 with bias [0.1]
[-0.9, -0.1,  0.2,  0.1] filter 4 with bias [0.2]
```

Let our context be:
```python
"eat" "apple" "bees"
```

Represented as embedding vectors
```python
[0.86,  -0.27, 1.65, 0.05,  2.34] "eat"
[-1.84, -1.79, 1.10, 2.38,  1.76] "apple"
[1.05,  -1.78, 0.16, -0.30, 1.91] "bees"
```

First we pad

```python
[0.00,  0.00,  0.00, 0.00,  0.00]
[0.00,  0.00,  0.00, 0.00,  0.00]
[0.00,  0.00,  0.00, 0.00,  0.00]
[0.86,  -0.27, 1.65, 0.05,  2.34] "eat"
[-1.84, -1.79, 1.10, 2.38,  1.76] "apple"
[1.05,  -1.78, 0.16, -0.30, 1.91] "bees"
[0.00,  0.00,  0.00, 0.00,  0.00]
[0.00,  0.00,  0.00, 0.00,  0.00]
[0.00,  0.00,  0.00, 0.00,  0.00]
```

Now to apply our first filter, we grab the first component of every vector

```python
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
[* 0.86*,  -0.27, 1.65, 0.05,  2.34] "eat"
[*-1.84*,  -1.79, 1.10, 2.38,  1.76] "apple"
[* 1.05*,  -1.78, 0.16, -0.30, 1.91] "bees"
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
[* 0.00*,  0.00,  0.00, 0.00,  0.00]
```

Giving us

```python
[0,0,0,0.86,-1.84,1.05,0,0,0]
```

Now we apply `filter 0 [ 0.4,  0.7, -2.1,  1.1]` with bias `[0.2]`
```python
[*0,0,0,0.86*,-1.84,1.05,0,0,0]
0.4*0     + 0.7*0     + -2.1*0     + 1.1*0.86  = 0.946  +  0.2 = 1.146

[0,*0,0,0.86,-1.84*,1.05,0,0,0]
0.4*0     + 0.7*0     + -2.1*0.86  + 1.1*-1.84 = -3.83  +  0.2 = -3.63

[0,0,*0,0.86,-1.84,1.05*,0,0,0]
0.4*0     + 0.7*0.86  + -2.1*-1.84 + 1.1*1.05  = 5.621  +  0.2 = 5.821

[0,0,0,*0.86,-1.84,1.05,0*,0,0]
0.4*0.86  + 0.7*-1.84 + -2.1*1.05  + 1.1*0     = -3.149 +  0.2 = -2.949

[0,0,0,0.86,*-1.84,1.05,0,0*,0]
0.4*-1.84 + 0.7*1.05  + -2.1*0     + 1.1*0     = -0.001 +  0.2 = 0.199

[0,0,0,0.86,-1.84,*1.05,0,0,0*]
0.4*1.05  + 0.7*0     + -2.1*0     + 1.1*0     = 0.42   +  0.2 = 0.62
```

So our output of `filter 0` is

```python
[1.146, -3.63, 5.821, -2.949, 0.199, 0.62]
```

Now we cut off the last two (to give us same size output as L), giving us

```python
[1.146, -3.63, 5.821, -2.949]
```

For `filter 1`, we grab the second component
```python
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
[0.86,  *-0.27*, 1.65, 0.05,  2.34] "eat"
[-1.84, *-1.79*, 1.10, 2.38,  1.76] "apple"
[1.05,  *-1.78*, 0.16, -0.30, 1.91] "bees"
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
[0.00,  * 0.00*,  0.00, 0.00,  0.00]
```

Giving us

```python
[0,0,0,-0.27,-1.79,-1.78,0,0,0]
```

Now we apply `filter 1 [ 0.1, -0.7, -0.3,  0.0]` with bias `[0.2]`

etc.

</details>

<details>
  <summary>Conv1D in Code</summary>
  
```python
def mamba_conv1d(x, conv):
    # x is [B, E, L]
    filters = conv.weight # filters is [E, 1, 4]
    bias = conv.bias # bias is [E]
    with torch.no_grad():
        # first we pad x to [B, E, 3+L+3]
        B, E, L = x.size()
        x = torch.nn.functional.pad(x, (3,3), mode='constant', value=0)
        res = torch.zeros([B, E, 3+L])
        for b in range(B):
            # one filter for each component of the E-sized vectors
            for filter_i in range(E):
                # filter is 4 values, go across words
                filter = filters[filter_i, 0]\
                # scan across all the places
                for starting_pos in range(3+L):
                    output = 0.0
                    for i, f in enumerate(filter):
                        output += x[b, filter_i, starting_pos+i]*f
                    res[b, filter_i, starting_pos] = output+bias[filter_i]
        return res
```
</details>
</details>

```python
        # [B,L,E]       [B,L,E]
        x         = F.silu(  x  ) # silu(x) = x * sigmoid(x) = x * 1/(1+exp(-x))
        
        # [B,L,E]               [B,L,E]
        y         = ssm( layer,    x    )
```

<details>
<summary>SSM Explanation</summary>

<details>
<summary>Expanded SSM</summary>

```python
def ssm(layer, x):

    # W_delta is factored into two matrices W_delta_1 and W_delta_2, combine them back
    # [E,E] =          [E,D_delta]         [D_delta, E]
    W_delta = layer.W_delta_1.weight.T @ layer.W_delta_2.weight.T
        
    ys = []
    # every pair (b,e) has a 1-D ssm
    for b in range(Batch):
        ys_b = []
        for e in range(E):
            ys_e_b = []
            
            # latent state, init to zeros
            h = torch.zeros(N)
            for l in range(L):
                #### First, discretization: A and B -> Abar and Bbar ####
                ## Compute Delta ##
                # [1]                 ([E]  dot  [E])                  [1]
                delta =  F.softplus(x[b,l].dot(W_delta[:,e]) + layer.W_delta_2.bias[e])
                
                ## Discretize A ##
                # [N]                ( [1]  *    [N]    ) 
                A_bar     = torch.exp(delta * layer.A[e])
                
                ## Discretize B ##
                # [N]         [E->N]    [E]
                B         = layer.W_B(x[b,l]) # no bias
                # [N]        [1]   [N]
                B_bar     = delta * B
                
                #### Update latent vector h ####
                ## input float for the ssm at time l
                # [1]         [1]
                x_l       = x[b,l,e]
                
                ## move ahead by one step
                # [N]        [N]   [N]   [N]    [1]
                h         = A_bar * h + B_bar * x_l
                
                #### Compute output float y ####
                ## (C matrix needed for computing y)
                # [N]         [E->N]    [E]
                C_l       = layer.W_C(x[b,l]) # no bias
                
                ## Output a float y at time l
                # [1]      [N]    [N]
                y_l       = h.dot(C_l)
                
                ys_e_b.append(y_l)
            ys_b.append(ys_e_b)
        ys.append(ys_b)

    ## Code expects this transposed a bit
    # [B,E,L]
    ys         = torch.tensor(ys)
    # [B,L,E]              [B,E,L]
    ys         = rearrange(  ys   , "B E L -> B L E")
    
    return ys
```
</details>


<details>
<summary>Vectorized</summary>

```python
def ssm(layer, x):
    ys = []
    for b in range(Batch):
        ys_b = []
        
        # latent state, init to zeros
        h = torch.zeros([E,N])
        for l in range(L):
            #### First, discretization: A and B -> A_bar and B_bar ####
            ## Compute Delta ##
            # [E]                   [E]  x  [E,E]  +         [E]
            delta    = F.softplus(layer.W_delta_2(layer.W_delta_1(x[b,l])))
            
            ## Discretize A -> A_bar ##
            # (note [E,N]*[E,1] will first repeat the [E,1] N times so its like [E,N])
            # [E,N]             (     [E,1]      *  [E,N] ) 
            A_bar    = torch.exp(delta.view(E,1) * layer.A)
            
            ## Discretize B -> B_bar ##
            # [N]        [E->N]   [E]
            B        = layer.W_B(x[b,l]) # no bias
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
            C        = layer.W_C(x[b,l]) # no bias
            
            ## Output floats y at time l
            # [E,1]      [E,N]  x   [N,1]
            y_l      =     h    @ C.view(N,1)
            
            ys_b.append([y.float() for y in y_l.flatten()])
        ys.append(ys_b)
    return torch.tensor(ys)
```

</details>



</details>


```python
        
        # [B,L,E]  [B,L,E]    [B,L,E]    [E]
        y         =   y      +   x     *  layer.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      + F.silu(  skip  )
        
        # [B,L,D]          [E->D]  [B,L,E]
        y         = layer.out_proj(   y   ) # no bias
        
        # [B,L,D]   [B,L,D]
        resid    +=    y
    
    # [B,L,D]              [B,L,D]
    resid     = mamba.norm( resid )
    
    # [B,L,V]          [D->V] [B,L,D]
    logits    = mamba.lm_head( resid ) # no bias
    return logits
```

<details>
  <summary>Full code</summary>
  
  
<details>
  <summary>Setup</summary>
  
```python
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

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


#### Params ####

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
```
</details>

<details>
<summary>Expanded</summary>

```python
def run_mamba(mamba, input_ids):

    args = mamba.args
    D = args.d_model
    E = args.d_inner
    N = args.d_state
    D_delta = args.dt_rank
    D_conv = args.d_conv
    V = args.vocab_size
    
    Batch,L = input_ids.size()

    # [B,L,D]                         [B,L]
    resid         = mamba.embedding(input_ids)
    
    for layer in mamba.layers:
        ###### Process inputs ######
        # [B,L,D]  [B,L,D]
        x         = resid
        # [B,L,D]             [B,L,D]
        x         = layer.norm(  x  )
        # [B,L,E]         [D->E]  [B,L,D]
        skip      = layer.skip_proj(  x  ) # no bias
        # [B,L,E]         [D->E] [B,L,D]
        x         = layer.in_proj(  x  ) # no bias
        
        ###### Conv ######
        # [B,E,L]
        x         = rearrange(x, 'B L E -> B E L')
        # [B E L]                [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
        x         = layer.conv1d(   x   )[:, :, :L]
        # [B,L,E]
        x         = rearrange(x, 'B E L -> B L E')

        ###### Nonlinearity  ######
        # [B,L,E]          [B,L,E]
        x         = F.silu(  x   )

        ###### SSM ######
        
        layer.A = -torch.exp(layer.A_log)
        
        # W_delta is factored into two matrices W_delta_1 and W_delta_2, combine them back
        # [E,E] =          [E,D_delta]         [D_delta, E]
        W_delta = layer.W_delta_1.weight.T @ layer.W_delta_2.weight.T
        
        ys = []
        # every pair (b,e) has a 1-D ssm
        for b in range(Batch):
            ys_b = []
            for e in range(E):
                ys_e_b = []
                
                # latent state, init to zeros
                h = torch.zeros(N)
                for l in range(L):
                    #### First, discretization: A and B -> Abar and Bbar ####
                    ## Compute Delta ##
                    # [1]                 ([E]  dot  [E])                  [1]
                    delta =  F.softplus(x[b,l].dot(W_delta[:,e]) + layer.W_delta_2.bias[e])
                    
                    ## Discretize A ##
                    # [N]                ( [1]  *    [N]    ) 
                    A_bar     = torch.exp(delta * layer.A[e])
                    
                    ## Discretize B ##
                    # [N]         [E->N]    [E]
                    B         = layer.W_B(x[b,l]) # no bias
                    # [N]        [1]   [N]
                    B_bar     = delta * B
                    
                    #### Update latent vector h ####
                    ## input float for the ssm at time l
                    # [1]         [1]
                    x_l       = x[b,l,e]
                    
                    ## move ahead by one step
                    # [N]        [N]   [N]   [N]    [1]
                    h         = A_bar * h + B_bar * x_l
                    
                    #### Compute output float y ####
                    ## (C matrix needed for computing y)
                    # [N]         [E->N]    [E]
                    C_l       = layer.W_C(x[b,l]) # no bias
                    
                    ## Output a float y at time l
                    # [1]      [N]    [N]
                    y_l       = h.dot(C_l)
                    
                    ys_e_b.append(y_l)
                ys_b.append(ys_e_b)
            ys.append(ys_b)

        ## Code expects this transposed a bit
        # [B,E,L]
        y         = torch.tensor(ys)
        # [B,L,E]              [B,E,L]
        y         = rearrange(  y   , "B E L -> B L E")
        
        ###### Finish layer ######
        
        # [B,L,E]  [B,L,E]    [B,L,E]       [E]
        y         =   y      +   x     *  layer.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      * F.silu(  skip  )
        
        # [B,L,D]          [E->D]  [B,L,E]
        y         = layer.out_proj(   y   ) # no bias
        
        # [B,L,D]   [B,L,D]
        resid    +=    y
    
    ###### Final processing ######
    # [B,L,D]              [B,L,D]
    resid     = mamba.norm( resid )
    
    # [B,L,V]          [D->V] [B,L,D]
    logits    = mamba.lm_head( resid ) # no bias
    return logits
```
  
</details>




<details>
<summary>Vectorized</summary>

```python
def run_mamba(mamba, input_ids):

    args = mamba.args
    D = args.d_model
    E = args.d_inner
    N = args.d_state
    D_delta = args.dt_rank
    D_conv = args.d_conv
    V = args.vocab_size
    
    Batch,L = input_ids.size()

    # [B,L,D]                         [B,L]
    resid         = mamba.embedding(input_ids)
    
    for layer in mamba.layers:
        ###### Process inputs ######
        # [B,L,D]  [B,L,D]
        x         = resid
        # [B,L,D]             [B,L,D]
        x         = layer.norm(  x  )
        # [B,L,E]         [D->E]  [B,L,D]
        skip      = layer.skip_proj(  x  ) # no bias
        # [B,L,E]         [D->E] [B,L,D]
        x         = layer.in_proj(  x  ) # no bias
        
        ###### Conv ######
        # [B,E,L]
        x         = rearrange(x, 'B L E -> B E L')
        # [B E L]                [B,E,L]  conv1d outputs [B,E,3+L], cut off last 3
        x         = layer.conv1d(   x   )[:, :, :L]
        # [B,L,E]
        x         = rearrange(x, 'B E L -> B L E')

        ###### Nonlinearity  ######
        # [B,L,E]          [B,L,E]
        x         = F.silu(  x   )
        
        ###### SSM ######
        
        # W_delta is factored into two matrices W_delta_1 and W_delta_2, combine them back
        # [E,E] =          [E,D_delta]         [D_delta, E]
        W_delta = layer.W_delta_1.weight.T @ layer.W_delta_2.weight.T
       
        layer.A = -torch.exp(layer.A_log)
       
        ys = []
        for b in range(Batch):
            ys_b = []
            
            # latent state, init to zeros
            h = torch.zeros([E,N])
            for l in range(L):
                #### First, discretization: A and B -> A_bar and B_bar ####
                ## Compute Delta ##
                # [E]                   [E]  x  [E,E]  +         [E]
                delta    = F.softplus(layer.W_delta_2(layer.W_delta_1(x[b,l])))
                
                ## Discretize A -> A_bar ##
                # (note [E,N]*[E,1] will first repeat the [E,1] N times so its like [E,N])
                # [E,N]             (     [E,1]      *  [E,N] ) 
                A_bar    = torch.exp(delta.view(E,1) * layer.A)
                
                ## Discretize B -> B_bar ##
                # [N]        [E->N]   [E]
                B        = layer.W_B(x[b,l]) # no bias
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
                C        = layer.W_C(x[b,l]) # no bias
                
                ## Output floats y at time l
                # [E,1]      [E,N]  x   [N,1]
                y_l      =     h    @ C.view(N,1)
                
                ys_b.append([y.float() for y in y_l.flatten()])
            ys.append(ys_b)
        # [B,L,E]
        y = torch.tensor(ys)
        
        ###### Finish layer ######
        
        # [B,L,E]  [B,L,E]    [B,L,E]       [E]
        y         =   y      +   x     *  layer.W_D
        # [B,L,E]  [B,L,E]          [B,L,E]
        y         =   y      * F.silu(  skip  )
        
        # [B,L,D]          [E->D]  [B,L,E]
        y         = layer.out_proj(   y   ) # no bias
        
        # [B,L,D]   [B,L,D]
        resid    +=    y
    
    ###### Final processing ######
    # [B,L,D]              [B,L,D]
    resid     = mamba.norm( resid )
    
    # [B,L,V]          [D->V] [B,L,D]
    logits    = mamba.lm_head( resid ) # no bias
    return logits
```
  
</details>

<details>
<summary>Load model</summary>

```python
from dataclasses import dataclass
import json
import math

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

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

def load_mamba(pretrained_model_name):

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
        
    for key, value in new_state_dict.items():
        print(key)
    model.load_state_dict(new_state_dict)
    return model
    
```


</details>


Sources:

- Softplus image from [pytorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- Much of this code is modified from [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
