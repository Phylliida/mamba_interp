Initializing hooked mamba:

From a pretrained model on huggingface:

```
from hooked_mamba import HookedMamba

model = HookedMamba.from_pretrained("state-spaces/mamba-370m", device='cuda')
```

From a config and state dict:
```
import hooked_mamba

state_dict = old_model.state_dict() # your state dict from a model using https://github.com/state-spaces/mamba
cfg = { # your config from a model using https://github.com/state-spaces/mamba
    "d_model": 1024,
    "n_layer": 48,
    "vocab_size": 50277,
    "ssm_cfg": {
        "d_state": 16
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
# it also does:

hooked_mamba_cfg = hooked_mamba.convert_original_config_to_hooked_mamba_config(cfg, device=device)
hooked_mamba_state_dict = hooked_mamba.convert_original_state_dict_to_hooked_format(state_dict)

model = hooked_mamba.HookedMamba(cfg=hooked_mamba_cfg, device='cuda')
model.load_state_dict(hooked_mamba_state_dict)
```

Initialize a model from scratch (with correct parameter initialization)

```


```
