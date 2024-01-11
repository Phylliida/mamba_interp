import torch
from einops import einsum, rearrange
import torch.nn.functional as F

def mamba_conv1d(x, filters, bias):
    with torch.no_grad():
        # x is [B, D, L]
        # filters is [D, 1, 4]
        # bias is [D]
        # first we pad x to [B, D, 3+L+3]
        B, D, L = x.size()
        x = torch.nn.functional.pad(x, (3,3), mode='constant', value=0)
        res = torch.zeros([B, D, 3+L])
        for b in range(B):
            # one filter for each component
            for filter_i in range(D):
                # filter is 4 values, go across words
                filter = filters[filter_i, 0]\
                # scan across all the places
                for starting_pos in range(3+L):
                    output = 0.0
                    for i, f in enumerate(filter):
                        output += x[b, filter_i, starting_pos+i]*f
                    res[b, filter_i, starting_pos] = output+bias[filter_i]
        return res
 
def test_a_bar():
    B, L, E, N = 2,3,5,7
    delta = torch.randn([B, L, E])
    A = torch.randn([E, N])
    r, custom = a_bar(delta, A)
    print(r[1,2,3], custom[1,2,3])
    assert(torch.all(torch.abs(r-custom)<0.01))
 
def a_bar(delta, A):
    B, L, E = delta.size()
    E, N = A.size()
    with torch.no_grad():
        r = einsum(delta, A, 'B L E, E N -> B L E N')
        custom = torch.zeros([B, L, E, N])
        for b in range(B):
            for l in range(L):
                for e in range(E):
                    for n in range(N):
                        custom[b,l,e,n] = delta[b, l, e]*A[e,n]
        return r, custom

def test_b_bar_x():
    B_, L, E, N = 2,3,5,7
    delta = torch.randn([B_, L, E])
    B = torch.randn([B_, L, N])
    x = torch.randn([B_, L, E])
    r, custom = b_bar_x(delta, B, x)
    print(r[1,2,3], custom[1,2,3])
    assert(torch.all(torch.abs(r-custom)<0.01))
 
 
def b_bar_x(delta, B, x):
    B_, L, E = delta.size()
    B_, L, N = B.size()
    B_, L, E = x.size()
    r = einsum(delta, B, x, 'B L E, B L N, B L E -> B L E N')
    custom = torch.zeros([B_, L, E, N])
    for b in range(B_):
        for l in range(L):
            for e in range(E):
                for n in range(N):
                    custom[b,l,e,n] = delta[b,l,e]*x[b,l,e]*B[b,l,n]
    return r, custom
    
# todo: if ssm is just a bunch of 1D things, does this make it privlidged?
def test_y():
    B, E, N = 2,3,5
    h = torch.randn([B, E, N])
    C = torch.randn([B, N])
    r, custom = y(h, C)
    print(r[1], custom[1])
    assert(torch.all(torch.abs(r-custom)<0.01))

def y(h, C):
    B, E, N = h.size()
    B, N = C.size()
    r = einsum(h, C, "B E N, B N -> B E")
    custom = torch.zeros([B, E])
    for b in range(B):
        for e in range(E):
            vec1 = h[b,e]
            vec2 = C[b]
            custom[b,e] = vec1.dot(vec2)
    return r, custom



def test_simple_mamba_ssm(mamba):
    for i, layer in enumerate(mamba.layers):
        print("layer", i)
        B = 2
        L = 5
        E = mamba.args.d_inner
        x = torch.randn([B,L,E])
        output1 = layer.mixer.ssm(x)
        output2 = simple_ssm(layer.mixer, x)
        print(output1[1,2], output2[1,2])
        assert(torch.all(torch.abs(output1-output2)<0.01))
    
def simple_ssm(self, x):
    # B = batch size
    # L = context len
    # D = d_model = 1024
    # E = d_inner/d_in = 2048
    # N = d_state = 16
    # D_delta = dt_rank = 64
    
    B, L, E = x.size()
    D = self.args.d_model
    E = self.args.d_inner
    N = self.args.d_state
    D_delta = self.args.dt_rank
    
    # maps [B,L,E] -> [B,L,D_delta+2*N] -> [B,L,D_delta], [B,L,N], [B,L,N]
    W = self.x_proj.weight.T
    # they are all shoved in the same weight, pull them out
    W_delta_1 = W[:,:D_delta]
    W_B = W[:,D_delta:D_delta+N]
    W_C = W[:,D_delta+N:]
    
    # maps [B,L,D_delta] -> [B,L,E]
    W_delta_2 = self.dt_proj.weight.T
    B_delta = self.dt_proj.bias
    
    # W_delta is factored into two matrices W_delta_1 and W_delta_2, combine them back
    # maps [B,L,E] -> [B,L,E]
    W_delta = W_delta_1@W_delta_2
    
    # [E,N] learned parameter
    A = -torch.exp(self.A_log.float())
               
    ys = []
    
    for b in range(B):
        print(b)
        ys_b = []
        for e in range(E):
            ys_e_b = []
            # every pair (b,e) has a 1-D ssm
            h = torch.zeros(N)
            
            for l in range(L):
                #### First, discretization: A and B -> Abar and Bbar ####
                ## Compute Delta ##
                #                x is [B,L,E]  W_delta is [E,E]
                #  [1]                  ([E]  dot   [E])      +  [1]
                delta_ble = F.softplus(x[b,l].dot(W_delta[:,e])+B_delta[e])
                
                ## Discretize A ##
                #  [N]             ( [1]  * [N]) 
                A_bar = torch.exp(delta_ble*A[e])
                
                ## Discretize B ##
                #  [N]       [E]  [E,N]
                B_l      = (x[b,l]@W_B)
                #  [N]    [1]     [N]
                B_bar = delta_ble*B_l
                
                #### Update latent vector h ####
                ## input float for the ssm at time l
                #  [1]     [1]
                x_l    = x[b,l,e]
                
                ## move ahead by one step
                #  [N]     [N]   [N]    [N]   [1]
                h       = A_bar * h + B_bar * x_l
                
                #### Compute output float y ####
                ## (C matrix needed for computing y)
                #  [N]     [E]  [E,N]
                C_l     = x[b,l]@W_C
                
                ## Output a float y at time l
                # [1]      [N]    [N]
                y_l       = h.dot(C_l)
                
                ys_e_b.append(y_l)
            ys_b.append(ys_e_b)
        ys.append(ys_b)
    
    ## Code expects this transposed a bit
    # B E L
    ys = torch.tensor(ys)
    # B L E
    ys = rearrange(ys, "B E L -> B L E")
    
    ## Extra bonus thing they do
    # D is just a learned parameter of size [E]
    D     = self.D.float()
    ys = ys + x * D
    
    return ys
    