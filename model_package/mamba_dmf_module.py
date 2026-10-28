import torch
import torch.nn as nn

try:
    import model_package.Mamba as Mamba
except ImportError:
    from . import Mamba


def _conf_get(conf, key, default=None):
    if isinstance(conf, dict):
        return conf.get(key, default)
    return getattr(conf, key, default)


class MambaDmf(nn.Module):
    def __init__(self, conf):
        super(MambaDmf, self).__init__()

        z_size = _conf_get(conf, "z_size")
        num_slices = _conf_get(conf, "num_slices")
        d_state = _conf_get(conf, "d_state", 16)
        d_conv = _conf_get(conf, "d_conv", 4)
        expand = _conf_get(conf, "expand", 2)
        dmf_hidden_size = _conf_get(conf, "dmf_hidden_size")
        num_locations = _conf_get(conf, "num_locations")
        dropout_rate = _conf_get(conf, "dropout_rate", 0.0)
        z_init = _conf_get(conf, "mamba_z_init", "randn")

        if z_init == "zeros":
            z_parameter = torch.zeros((z_size, num_slices))
        elif z_init == "randn":
            z_parameter = torch.randn((z_size, num_slices))
        else:
            raise ValueError("mamba_z_init must be 'randn' or 'zeros'")

        self.z = torch.nn.parameter.Parameter(
            z_parameter,
            requires_grad=True)

        self.mamba_residual = bool(_conf_get(conf, "mamba_residual", False))
        self.mamba_residual_scale = float(_conf_get(conf, "mamba_residual_scale", 1.0))
        self.mamba_input_norm = nn.LayerNorm(z_size) if bool(_conf_get(conf, "mamba_layer_norm", False)) else nn.Identity()
        head_activation_name = _conf_get(conf, "mamba_head_activation", "elu")
        if head_activation_name == "elu":
            head_activation = nn.ELU()
        elif head_activation_name == "tanh":
            head_activation = nn.Tanh()
        else:
            raise ValueError("mamba_head_activation must be 'elu' or 'tanh'")

        self.mamba = Mamba.Mamba(z_size, d_state, d_conv, expand, use_fast_path=False)

        self.dmf = nn.Sequential(
            nn.Linear(
                in_features=z_size,
                out_features=dmf_hidden_size,
                bias=True),
            head_activation,
            nn.Dropout(dropout_rate),
            nn.Linear(
                in_features=dmf_hidden_size,
                out_features=num_locations,
                bias=True
            )
        )

    def forward(self, delta_t):
        if delta_t is not None:
            delta_t = delta_t.to(device=self.z.device, dtype=torch.float32).flatten()

        z_input = self.z.T.unsqueeze(0)
        z = self.mamba(self.mamba_input_norm(z_input), delta_t)
        if self.mamba_residual:
            z = z_input + self.mamba_residual_scale * z
        z = z.squeeze(0)
        completion = self.dmf(z)
        return completion.T, None, None, None
