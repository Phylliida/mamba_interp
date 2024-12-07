The following post was made as part of Danielle's MATS work on doing circuit-based mech interp on Mamba, mentored by Adrià Garriga-Alonso. It's the first in a sequence of posts about finding an IOI circuit in Mamba/applying [ACDC](https://arxiv.org/abs/2304.14997) to Mamba.

This introductory post was also made in collaboration with Gonçalo Paulo.

# A new challenger arrives!

Why Mamba?

## Promising Scaling

Mamba [^gu2023mamba] is a type of recurrent neural network based on state-space models, and is being proposed as an alternative architecture to transformers. It is the result of years of capability research [^gu2020hippo] [^gu2022efficiently] [^fu2023hungry] and likely not the final iteration of architectures based on state-space models.

In its current form, Mamba has been scaled up to 2.8B parameters on The Pile and on Slimpj, having similar scaling laws when compared to Llama-like architectures.

![From Mamba paper, Mamba scaling compared to Llama (Transformer++), previous state space models (S3++), convolutions (Hyena), and a transformer inspired RNN (RWKV)](https://github.com/Phylliida/mamba_interp/blob/main/graphs/scaling%20mamba.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scaling curves from Mamba paper: Mamba scaling compared to Llama (Transformer++), previous state space models (S3++), convolutions (Hyena), and a transformer inspired RNN (RWKV)

More recently, ai21labs [^lieber2024jamba] trained a 52B parameter MOE Mamba-Transformer hybrid called Jamba. At inference, this model has 12B active parameters and has benchmark scores comparable to Llama-2 70B and Mixtral.

![Jamba benchmark scores, from Jamba paper](https://github.com/Phylliida/mamba_interp/blob/main/graphs/jamba%20metrics.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jamba benchmark scores, from Jamba paper [^lieber2024jamba]

## Efficient Inference

One advantage of RNNs, and in particular of Mamba, is that the memory required to store the context length is constant, as you only need to store the past state of the SSM and of the convolution layers, while it grows linearly for transformers. The same happens with the generation time, where predicting each token scales as $O(1)$ instead of $O(\text{context length})$.

![Jamba throughput (tokens/second), from Jamba paper](https://github.com/Phylliida/mamba_interp/blob/main/graphs/Throughput%20Jamba.png?raw=true)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jamba throughput (tokens/second), from Jamba paper[^lieber2024jamba]

# What are Space-state models?

The inspiration for Mamba (and similar models) is an established technique used in control theory called state space models (SSM). SSMs are normally used to represent linear systems that have p inputs, q outputs and n state variables. To keep the notation concise, we will consider the input as E-dimensional vector $x(t) \in \mathbb{R}^E$, an E-dimensional output $y(t) \in \mathbb{R}^E$ and a N-dimensional latent space $h \in \mathbb{R}^N$. In Mamba 2.8b, E=5120 and N=16. In the following, we will note the dimensions of new variables using the notation [X,Y].

Specifically, we have the following:

$$\stackrel{[N]}{\dot{h}(t)} = \stackrel{[N,N]}{A}\stackrel{[N]}{h(t)} + \stackrel{[N,E]}{B}\stackrel{[E]}{x(t)}$$
$$\stackrel{[E]}{y(t)} = \stackrel{[E,N]}{C}\stackrel{[N]}{h(t)} + \stackrel{[E,E]} D\stackrel{[E]} x(t)$$

This is an ordinary differential equation (ODE), where $\dot{h}(t)$ is the derivative of $h(t)$ with respect to time, t. This ODE can be solved in various ways, which will be described below.

In state space models, $A$ is called the \emph{state matrix}, $B$ is called the \emph{input matrix}, $C$ is called the \emph{output matrix}, and $D$ is called the \emph{feedthrough matrix}.


# Solving the ODE

We can write the ODE from above as a recurrence, using discrete timesteps:

$$\stackrel{[N]}{h_{t}} = \stackrel{[N,N]}{\overline{A}}\stackrel{[N]}{h_{t-1}} + \stackrel{[N,E]}{\overline{B}}\stackrel{[E]}{x_{t}}$$
$$\stackrel{[E]}{y_t} = \stackrel{[E,N]}{C}\stackrel{[N]}{h_t} + \stackrel{[E,E]} D\stackrel{[E]} x_t$$

where $\overline{A}$ and $\overline{B}$ are our \emph{discretization matrices}. Different ways of integrating the original ODE will give different $\overline{A}$ and $\overline{B}$, but will still preserve this overall form.

In a context with discretization matrices, $t$ corresponds to a discrete time. In language modeling, $t$ refers to the
token position.

## Euler method

The simplest way to numerically integrate ODE is by using the Euler method, which consists in approximating the derivative by considering the ratio between a small variation in h and a small variation in time, $\dot{h}=\frac{dh}{dt}\approx\frac{\Delta h}{\Delta t}$. This allows us to write:

$$\frac{h_{t+1}-h_t}{\Delta t}= A h_t + B x_t $$
$$h_{t+1}= \Delta t(A h_t + B x_t) + h_t $$

Where the index t, of $h_t$, represents the discretized time. This is the same thing that is done when considering a character's position and velocity in a video game, for instance. If a character has a velocity $v$ and a position $x_0$, to find the position after $\Delta t$ time we can do $x_1 = \Delta t v + x_0$. In general:

$$x_t = \Delta t  v_{t} + x_{t-1}$$
$$x_t = (\Delta t  v_{t}+1)x_{t-1}$$

Turning back to the above example, we can rewrite

$$h_{t+1}= \Delta t(A h_t + B x_t) + h_t $$

as

$$ h_t = (\Delta A + I ) h_{t-1} + \Delta B x_{t}$$.

which means that, for the Euler Method, $\overline{A} = (\Delta A + I )$ and $\overline{B} = \Delta B$.
Here, $\Delta$ is an abbreviation of $\Delta t$, the discretization size in time.

[comment]: # (What are \Delta A and \Delta B here? We haven't introduced A and B as time-varying, so it should just be A and B. Or, we properly introduce A and B as time-varying.)

## Zero-Order Hold (ZOH)

Another way to integrate the ODE is to consider that the input $x(t)$ remains fixed during a time interval $\Delta$, and to integrate the differential equation from time $t$ to $t+\Delta$. This gives us an expression for $x(t+\Delta)$:

$$x(t+\Delta) = e^{\Delta A} x(t) + u(t+1)\int_t^{t+\Delta} e^{(t+\Delta-\tau) A} B  d\tau$$

With some [algebra](https://faculty.washington.edu/chx/teaching/me547/1-8_zohSS_slides.pdf) we finally get:

$$\overline{A} = \exp(\Delta A) \quad \overline{B} = (\Delta A)^{-1} (\exp(\Delta A)-I) \Delta B$$

## Discretization rule used in Mamba

Mamba uses a mix of Zero-Order Hold and the Euler Method:

$$\overline{A} = \exp(\Delta A) \quad \overline{B} = \Delta B$$

Why is this justified? Consider the ZOH $\overline{B}$:

$$\overline{B} = (\Delta A)^{-1} (\exp(\Delta A)-I) \Delta B$$

In Mamba, $A$ is diagonal, as we will see later, so we can write

$$\big((\Delta A)^{-1} (\exp(\Delta A)-I)\big)_{i,i}=\frac{\exp(\Delta A_{i,i}) - 1}{\Delta A_{i,i}}$$

If we consider that $\Delta A_{i,i}$ is small and we expand the exponential to just first order [^expexpand], this expression reduces to 1 which means that:

$$\overline{B} = (\Delta A)^{-1} (\exp(\Delta A)-I) \Delta B \approx \Delta B$$

for small enough $\Delta A_{i,i}$. Using the same approximation for $\overline{A}$ recovers the Euler method:

$$\overline{A} = \exp(\Delta A) \approx I+\Delta A.$$

In the original work, the authors argued that while ZOH was necessary for the modeling of $\overline{A}$, using the Euler Method for $\overline{B}$ gave reasonable results, without having to compute $(\Delta A)^{-1}$.

# Specific Quirks to Mamba

## The structured SSM

Mamba takes an interesting approach to the SSM equation. As previously mentioned, each timestep in Mamba represents a token position, and each token is represented (by the time it arrives to the SSM) by a E dimensional vector. The authors chose to represent the SSM as:

$$\stackrel{[E,N]}{h_t} = \stackrel{[E,N]}{\overline{A}}\stackrel{[E,N]}{h_{t-1}} + \stackrel{[E,N]}{\overline{B}}\stackrel{[E]}{x_t}$$

$$\stackrel{[E]}{y_t} = \stackrel{[N]}{C}\stackrel{[E,N]}{h_t} + \stackrel{[E]}{E}\stackrel{[E]}{x_t} $$

### The case of a 1-Dimensional input

When trying to understand Mamba, I find it's easiest to start with each $x_{t}$ being a single value first, and then working up from there. The standard SSM equation is, then:

$$\stackrel{[N]}{h_{t}} = \stackrel{[N,N]}{\overline{A}}\stackrel{[N]}{h_{t-1}} + \stackrel{[N,1]}{\overline{B}}\stackrel{[1]}{x_{t}}$$
$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t} + \stackrel{[1,1]}D \stackrel{[1]} x_t$$

The authors of the original Mamba paper were working on top of previous results on Structured SSMs. Because of this, in this work, A is a diagonal matrix. This means that A can be represented as a set of N numbers instead of a $NxN$ matrix. That gives us:

$$\stackrel{[N]}{h_{t}} = \stackrel{[N]}{\overline{A}}\stackrel{[N]}{h_{t-1}} + \stackrel{[N,1]}{\stackrel{[1]}{x_{t}}}$$
$$\stackrel{[1]}{y_t} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_t} + \stackrel{[1,1]} D\stackrel{[1]} x_t$$

Where $\stackrel{[N]}{\overline{A}}\stackrel{[N]}{h_{t-1}}$ is an element-wise product. In this example we are mapping a $1$-dimensional input to a $n$-dimensional hidden state, then mapping the $n$-dimensional hidden state back to a $1$ dimensional output.

### The Mamba implementation

In practice, $x_t$ and $y_t$ are not one dimensional, but $E$-dimensional vectors. Mamba simply maps each of these elements separately to a $N$ dimensional hidden space. So we can write a set of E equations:

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N]}{\overline{A}}\stackrel{[N]}{h_{t-1,e}} + \stackrel{[N,1]}{\overline{B}}\stackrel{[1]}{x_{t,e}}$$
$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C}\stackrel{[N]}{h_{t,e}} + \stackrel{[1,1]} D\stackrel{[1]} x_{t,e}$$

Where $e$ ranges from $[1,E]$. This means that each dimension of input to the SSM block is modeled by its own, independent, SSM. We will see that, due to the selection mechanism (see below) $\Delta, \overline{A}, \overline{B}, C$ are a function of all the dimensions of the input, not just the dimension e.

One thing to note: In practice, $A$ has a separate value for each $e$, and is encoded as an $[E,N]$ matrix. We can denote $\overline{A_e}=\Delta A_e$ as the $N$-sized entry for stream $e$, giving us,

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N]}{\overline{A_e}}\stackrel{[N]}{h_{t-1,e}} + \stackrel{[N,1]}{\overline{B}}\stackrel{[1]}{x_{t,e}}$$

## Selection mechanism

Mamba deviates from the simplest SSM approaches, and from the previous work of the authors, by making matrices B and C
dependent on the input, x(t). Not only that, but the time discretization $\Delta$ is also input dependent. This replaces
the equations shown above, with one which takes the form:

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N]}{\overline{A_{t,e}}}\stackrel{[N]}{h_{t-1,e}} + \stackrel{[N,1]}{\overline{B_{t,e}}}\stackrel{[1]}{x_{t,e}}$$
$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C_t}\stackrel{[N]}{h_{t,e}} + \stackrel{[1,1]} D\stackrel{[1]} x_{t,e}$$

Where the new matrices are given by:

$$\stackrel{[N]}{\overline{A_{t,e}}} = \exp(\stackrel{[1]}{\Delta_{t,e}} \stackrel{[N]}{A_e})$$


$$\stackrel{[N]}{\overline{B_{t,e}}} = \stackrel{[1]}{\Delta_{t,e}}\stackrel{[N]}{B_{t}}, \quad \text{with} \stackrel{[N]}{B_{t}} = \stackrel{[N,E]}{W_B}\stackrel{[E]}{x_t}$$

$$\stackrel{[N]}{C_t} = \stackrel{[N,E]}{W_C}\stackrel{[E]}{x_t}$$

$$\stackrel{[1]}{\Delta_{t,e}} = \text{softplus}(\stackrel{[E]}{x_{t}} \cdot \stackrel{[E]}{W^{\Delta}_e} + \stackrel{[1]}{B^{\Delta}_e})$$

with $\stackrel{[E,E]}{W^{\Delta}}, \stackrel{[E]}{B^{\Delta}}, \stackrel{[N,E]}{W_B}, \stackrel{[N,E]}{W_C}$ being learned parameters, and $\text{softplus}(x) = \log(1+e^{x})$

![softplus](https://github.com/Phylliida/mamba_interp/blob/main/graphs/softplus.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;softplus

One final thing to note: A is not a trainable parameter, and what is actually trained is $\stackrel{[E,N]}A_{\text{log}}$. $A$ is then computed as $A = -exp(A_{\text{log}})$ (using element-wise exp). This ensures $A$ is a strictly negative number

In turn, this implies that $\exp(\Delta A)$ is between 0 and 1. This is important for stable training: it ensures that
the elements of $h(t)$ do not grow exponentially with token position $t$, and the gradients do not explode. It is long
known [^pascanu2012rnn] that the explosion and vanishing of gradients are obstacles to training RNNs, and successful
architectures (LSTM, GRU) minimize these.


## $W_{\Delta}$ is low rank

In Mamba, they don't encode $\stackrel{[E,E]}{W_{\Delta}}$ as an $[E,E]$ matrix. Instead, it is encoded as two smaller matrices:

$$\stackrel{[E,E]}{W_{\Delta}}=\stackrel{[E,D_{\Delta}]}{W_{\Delta_1}}\stackrel{[D_{\Delta},E]}{W_{\Delta_2}}$$

Where, for example, $E=2048$, $D_{\Delta}=64$

This makes this term

$$\stackrel{[1]}{\Delta_{t,e}} = \text{softplus}(\stackrel{[E]}{x_{t}} \cdot \stackrel{[E]}{W_{\Delta}[:,e]} + \stackrel{[1]}{B_{\Delta}[e]})$$

Be instead

$$\stackrel{[1]}{\Delta_{t,e}} = \text{softplus}(\overset{[1]}{\overbrace{\stackrel{[E]}{x_{t}} \cdot \overset{[E]}{{{\overbrace{\Big(\stackrel{[E,D_\Delta]}{W_{\Delta_1}}  {{\stackrel{[D_\Delta]}{W_{\Delta_2}[:,e]}}}\Big)}}}}}} + \stackrel{[1]}{B_{\Delta}[e]})$$

## RMSNorm

This normalization is not unique to Mamba. It's defined as

$$RMSNorm(\stackrel{[B,L,D]}{x}) = \frac{x}{\sqrt{\text{mean}(x^2, \text{dim=-1})}}\text{weight}$$

If $\text{mean}$ was instead $\text{sum}$, this first term would be normalizing $x$ along the $D$ dimension. Because it's $\text{mean}$ there's an extra $D$ term, and we can rewrite this as:

$$RMSNorm(\stackrel{[B,L,D]}{x}) = \sqrt{D}\frac{x}{\sqrt{\text{sum}(x^2, \text{dim=-1})}}\text{weight}$$

The reason we want to do this is so that each _element_'s value is on average 1, as opposed to the whole activation's
vector. Since the introduction of the He initialization [^resnet2016], deep learning weights have been initialized so
the activation variance is 1 assuming the input variance is 1, thus keeping gradients stable throughout training.

# Full Architecture

Now that we know how the SSM works, here is the full architecture.

## Dimensions

(Example values from state-spaces/mamba-370m)

- $B$ is the batch size
- $L$ is the context length
- $D=d_\text{model}=1024$ is the dimension of the residual stream
- $E=d_\text{inner}=2048$ is the dimension of the embed size
- $N=d_\text{state}=16$ is the dimension of the state space
- $D_\text{delta}=dt_\text{rank}=64$ is the low rank size used when calculating delta, see section 4.4

## Notes on reading these graphs

- Text not in circles/squares are variable names/size annotations
- Rounded, white rectangles are mathmatical operations
- Shaded triangles are learned params
- Shaded squares are learned params that are projections
- Shaded circles are conv or rms norm (other operations that have some learned params)

## Overview

Mamba has:

- Embedding
- Residual stream that each layer adds to
- RMSNorm
- Project to logits


![High level overview of Mamba](https://github.com/Phylliida/mamba_interp/blob/main/graphs/mamba%20layer%20add%20to%20resid.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;High level overview of Mamba

```python
# [B,L,D]                 [B,L]
resid  = mamba.embedding(input_ids)
for layer in mamba.layers:
   # [B,L,D]     [B,L,D]
   resid += layer(resid)
# [B,L,D]              [B,L,D]
resid     = mamba.norm( resid )
# [B,L,V]           [D->V] [B,L,D]
logits    = mamba.lm_head( resid ) # no bias
return logits
```

## Layer contents

Each layer does:

- Project input $\overset{[B,L,D]}{\text{resid}}$ to $\overset{[B,L,E]}{x}$
- Project input $\overset{[B,L,D]}{\text{resid}}$ to $\overset{[B,L,E]}{\text{skip}}$
- Conv over the $E$ dimension[^conv] ($x = \text{conv}(x)$)
- Apply non-linearity (silu) ($x = \text{silu}(x)$)
- $y = SSM(x)$
- Gating: $y = y * \text{silu}(\text{skip})$
- Project $\overset{[B,L,E]}{y}$ to $\overset{[B,L,D]}{\text{output}}$

![Mamba layer overview](https://github.com/Phylliida/mamba_interp/blob/main/graphs/mamba%20layer%20view.png?raw=true)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mamba layer overview

![silu](https://github.com/Phylliida/mamba_interp/blob/main/graphs/silu.png?raw=true)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;silu

```python
def forward(layer, resid):
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

   ## Non-linearity ##
   # silu(x) = x * sigmoid(x)
   # silu(x) = x * 1/(1+exp(-x))
   # [B,L,E]         [B,L,E]
   x         = F.silu(  x  ) 
   
   ## SSM ##
   # [B,L,E]               [B,L,E]
   y         = ssm( layer,    x    )

   ## Gating ##
   # [B,L,E]  [B,L,E]          [B,L,E]
   y         =   y      * F.silu(  skip  )

   ## Project out ##
   # [B,L,D]          [E->D]  [B,L,E]
   y         = layer.out_proj(   y   ) # no bias
   return y
```


## SSM

From above:

$$\stackrel{[1]}{\Delta_{t,e}} = \text{softplus}(\stackrel{[E]}{x_{t}} \cdot \Big(\stackrel{[E,D_\Delta]}{W_{\Delta_1}}  \stackrel{[D_\Delta,1]}{W_{\Delta_2}[:,e].\text{view}(D_\Delta,1)}\Big).\text{view}(E) + \stackrel{[1]}{B_{\Delta}[e]})$$

$$\stackrel{[E,N]}{A}=\exp(\stackrel{[E,N]}{A_{log}})$$

$$\stackrel{[N]}{\overline{A_{t,e}}} = \exp(\stackrel{[1]}{\Delta_{t,e}} \stackrel{[N]}{A[e]})$$

$$\stackrel{[N]}{B_{t}} = \stackrel{[N,E]}{W_B}\stackrel{[E]}{x_t}$$

$$\stackrel{[N]}{\overline{B_{t,e}}} = \stackrel{[1]}{\Delta_{t,e}}\stackrel{[N]}{B_{t}}$$

$$\stackrel{[N]}{C_t} = \stackrel{[N,E]}{W_C}\stackrel{[E]}{x_t}$$

$$\stackrel{[N]}{h_{t,e}} = \stackrel{[N]}{\overline{A_{t,e}}}\stackrel{[N]}{h_{t-1,e}} + \stackrel{[N,1]}{\overline{B_{t,e}}}\stackrel{[1]}{x_{t,e}}$$
$$\stackrel{[1]}{y_{t,e}} = \stackrel{[1,N]}{C_t}\stackrel{[N]}{h_{t,e}} + \stackrel{[1,1]} D\stackrel{[1]} x_{t,e}$$

where $\stackrel{[E,E]}{W_{\Delta}}, \stackrel{[E]}{B_{\Delta}}, \stackrel{[N,E]}{A_{log}}, \stackrel{[N,E]}{W_B}, \stackrel{[N,E]}{W_C}$ are learned parameters, and $\text{softplus}(x) = \log(1+e^{x})$

```python
def ssm(layer, x):
       
   # stored as A_log
   layer.A = -torch.exp(layer.A_log)
   
   ys = []
   # every (e) has a 1-D ssm
   for e in range(E):
       ys_e = []
       
       # latent state, init to zeros
       h_e = torch.zeros(Batch,N)
       for l in range(L):
           #### First, discretization: A and B -> Abar and Bbar ####
           ## Compute Delta ##
           # [E,1]  [E,D_Delta]                                [D_delta,1]
           inner_term = layer.W_delta_1.weight.T@layer.W_delta_2.weight.T[:,e].view(D_Delta,1)
           # [1]                 [E]             [E]                   [1]
           delta = F.softplus(x[:,e].dot(inner_term.view(E)) + layer.W_delta_2.bias[e])
           
           ## Discretize A ##
           # [B,N]             ( [B,1]  *   [N]    ) 
           A_bar     = torch.exp(delta * layer.A[e])
           
           ## Discretize B ##
           # [B,N]         [E->N] [B,E]
           B         = layer.W_B(x[b,l]) # no bias
           # [B,N]     [B,1]  [B,N]
           B_bar     = delta * B
           
           #### Update latent vector h ####
           ## input float for the ssm at time l
           # [B]          [B]
           x_l       = x[:,l,e]
           
           ## move ahead by one step
           # [B,N]     [B,N]  [B,N]  [B,N]   [B,1]
           h_e       = A_bar * h   + B_bar * x_l.view(B,1)
           
           #### Compute output float y ####
           ## (C matrix needed for computing y)
           # [B,N]         [E->N]  [B,E]
           C_l       = layer.W_C(x[:,l]) # no bias
           
           ## Output a float y at time l
           # [B]      [B,N] [B,N]
           y_l       = (h*C_l).sum(dim=-1) # dot prod
           
           ys_e.append(y_l)
       # list of [L,B]
       ys.append(ys_e)
   
   ## Code expects this transposed a bit
   # [E,L,B]
   y          = torch.tensor(ys)
   # [B,L,E]             [B,E,L]
   y          = rearrange(  y   , "E L B -> B L E")
   ## Add the D term (we can do this outside the loop)
   # [B,L,E]   [B,L,E]   [B,L,E] [E]
   y          =   y     +   x  *  D
   return y
```

Or, vectorized, and computing non-h terms ahead of time (since they don't depend on the recurrence)

![Selective SSM](https://github.com/Phylliida/mamba_interp/blob/main/graphs/ssm%20light.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Selective SSM

```python
def ssm(layer, x):
    # [E,N]
   self.A = -torch.exp(self.A_log)
   
   ## Compute Delta ##
   # [B,L,D_delta] [E->D_delta]  [B,E]
   delta_1        = self.W_delta_1( x ) # no bias
   
   # [B,L,E]         [D_delta->E] [B,L,D_delta] 
   delta_2        = self.W_delta_2(  delta_1  ) # with bias
   
   # [B,L,E]           [B,L,E]
   delta  = F.softplus(delta_2)

   ## B
   # [B,L,N]     [E->N]   [B,L,E]
   B           = self.W_B(   x   )
   
   ## C
   # this just applies E->N projection to each E-sized vector
   # [B,L,N]      [E->N]  [B,L,E]     
   C           = self.W_C(   x   ) # no bias
   
   ## Discretize A
   # [B,L,E,N]                    [B,L,E] [E,N]
   A_bar       = torch.exp(einsum(delta, self.A, 'b l e, e n -> b l e n'))
   
   ## Discretize B
   # [B,L,E,N]          [B,L,E]  [B,L,N] 
   B_bar       = einsum( delta,    B,     'b l e, b l n -> b l e n')
   
   # Now we do the recurrence
   ys = []
   
   # latent state, init to zeros
   h = torch.zeros([Batch,E,N])
   for l in range(L):
       # [B,E,N]   [B,E,N]     [B,E,N]          [B,E,N]          [B,E]
       h        =    h    *  A_bar[:,l,:,:]  + B_bar[:,l,:,:] * x[:,l].view(Batch, E, 1)
       
       # this is like [E,N] x [N,1] for each batch
       # [B,E]    [B,E,N]       [B,N,1]  
       y_l       =   h     @   C[:,l,:].view(Batch,N,1)
       
       # [B,E]              [B,E,1]
       y_l      =    y_l.view(Batch,E)
       ys.append(y_l)
       
   # we have lots of [B,E]
   # we need to stack them along the 1 dimension to get [B,L,E]
   y = torch.stack(ys, dim=1)
   ## Add the D term
   # [B,L,E] [B,L,E]    [B,L,E]       [E]
   y =         y      +   x     *  self.W_D
   return y
```

Also keep in mind: In the official implementation,  $W_{\Delta_2}$ is called $dt_{\text{proj}}$, and some matrices are concatenated together (this is numerically equivalent, but helps performance as it's a fused operation):
- $\text{in}_{\text{proj}}$ and $\text{skip}_{\text{proj}}$ $\mapsto$ $\text{in}_{\text{proj}}$
- $W_{\Delta_1}$ $W_B$, $W_C$ $\mapsto$ $x_{\text{proj}}$

# Appendix

Here's some further info on how Mamba's 1D conv works, for those unfamiliar. This is not unique to Mamba, conv is a standard operation usually used in image processing.

## Conv1D Explanation

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

## Worked Conv Example

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

Now to apply our first filter, we grab the first element of every vector

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

For `filter 1`, we grab the second element

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

## Conv1D in code

 Here's what that means in code:

```python
def mamba_conv1d(x, conv):
    # x is [B, E, L]
    CONV = D_Conv-1 # D_conv=4 for mamba-370m
    filters = conv.weight # filters is [E, 1, D_conv]
    bias = conv.bias # bias is [E]
    with torch.no_grad():
        # first we pad x to [B, E, CONV+L+CONV]
        B, E, L = x.size()
        x = torch.nn.functional.pad(x, (CONV,CONV), mode='constant', value=0)
        res = torch.zeros([B, E, CONV+L])
        for b in range(B):
            # one filter for each element of the E-sized vectors
            for filter_i in range(E):
                # filter is 4 values, go across words
                filter = filters[filter_i, 0]
                # scan across all the places
                for starting_pos in range(CONV+L):
                    output = 0.0
                    for i, f in enumerate(filter):
                        output += x[b, filter_i, starting_pos+i]*f
                    res[b, filter_i, starting_pos] = output+bias[filter_i]
        return res
```


[^expexpand]: The Taylor series expansion of $\exp(x)$ at $x=0$ is $$\exp(x) = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + ...$$ And if we just consider the first-order terms, then we get $$\exp(x) \approx 1 + x$$

[^fu2023hungry]: Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher R ́e. Hungry hungry hippos: Towards language modeling with state space models, 2023. [https://arxiv.org/abs/2212.14052](https://arxiv.org/abs/2212.14052)

[^gu2023mamba]: Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

[^gu2020hippo]: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Re. Hippo: Recurrent memory with optimal polynomial projections, 2020. [https://arxiv.org/abs/2008.07669](https://arxiv.org/abs/2008.07669)

[^gu2022efficiently]: Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces, 2022. [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396)

[^lieber2024jamba]: Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Gloz-man, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model, 2024. [https://arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887)

[^pascanu2012rnn]: Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training recurrent neural networks." International Conference on Machine Learning, 2013. [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)

[^resnet2016]: He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1026-1034. 2015. [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)

