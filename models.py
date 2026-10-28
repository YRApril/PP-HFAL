import torch.nn as nn
import torch

try:
    from model_package.mamba_dmf_module import MambaDmf
except ImportError:
    from .model_package.mamba_dmf_module import MambaDmf


def get_model(conf):
    if conf["model_name"] == 'DMF':
        return Dmf(conf)
    if conf["model_name"] == 'RNN_DMF':
        return RnnDmf(conf)
    if conf["model_name"] == 'TIME_DMF':
        return TimeDmf(conf)
    if conf["model_name"] == 'Mamba_DMF':
        return MambaDmf(conf)
    raise ValueError("Unknown model_name")


class Dmf(nn.Module):

    def __init__(self, conf):
        super(Dmf, self).__init__()

        self.z = nn.parameter.Parameter(
            torch.zeros((conf["z_size"], conf["num_slices"])),
            requires_grad=True)

        self.dmf = nn.Sequential(
            nn.Linear(
                in_features=conf["z_size"],
                out_features=conf["dmf_hidden_size"],
                bias=True
            ),
            nn.Tanh(),
            nn.Dropout(conf["dropout_rate"]),
            nn.Linear(
                in_features=conf["dmf_hidden_size"],
                out_features=conf["num_locations"],
                bias=True
            )
        )

    def forward(self):
        return self.dmf(self.z.T)


class RnnDmf(nn.Module):

    def __init__(self, conf):
        super(RnnDmf, self).__init__()

        self.z = torch.nn.parameter.Parameter(
            torch.randn((conf["z_size"], conf["num_slices"])),
            requires_grad=True)

        self.delta = conf["delta"]

        self.rnn_ih = nn.Sequential(
            nn.Linear(
                in_features=conf["h_size"],
                out_features=conf["z_size"],
                bias=True),
            nn.Tanh()
        )
        self.activate = nn.Tanh()
        self.dmf = nn.Sequential(
            nn.Linear(
                in_features=conf["z_size"],
                out_features=conf["dmf_hidden_size"],
                bias=True),
            nn.Tanh(),
            nn.Dropout(conf["dropout_rate"]),
            nn.Linear(
                in_features=conf["dmf_hidden_size"],
                out_features=conf["num_locations"],
                bias=True
            )
        )

    def forward(self, z_steps, h):
        output = []
        for z in z_steps.T:
            delta = self.delta
            h = delta * h + (1 - delta) * self.rnn_ih(z)

            y = self.dmf(h).T
            output.append(y)

        completion = torch.cat(output, dim=1)

        return completion, h.data


class TimeDmf(nn.Module):

    def __init__(self, conf):
        super(TimeDmf, self).__init__()

        self.z = torch.nn.parameter.Parameter(
            torch.randn((conf["z_size"], conf["num_slices"])),
            requires_grad=True)

        self.delta = nn.Sequential(
            nn.Linear(conf["z_size"] * 2, conf["z_size"]),
            nn.Sigmoid()
        )

        self.rnn_ih = nn.Sequential(
            nn.Linear(
                in_features=conf["z_size"],
                out_features=conf["z_size"],
                bias=True),
        )

        self.xh = nn.Linear(conf["z_size"], conf["z_size"])
        self.ch = nn.Linear(conf["c_size"], conf["z_size"])

        self.T1 = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        self.T2 = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        self.activate = nn.Tanh()

        self.dmf = nn.Sequential(
            nn.Linear(
                in_features=conf["z_size"],
                out_features=conf["dmf_hidden_size"],
                bias=True),
            nn.Tanh(),
            nn.Dropout(conf["dropout_rate"]),
            nn.Linear(
                in_features=conf["dmf_hidden_size"],
                out_features=conf["num_locations"],
                bias=True
            )
        )

    def forward(self, z_steps, h, c_long, t_steps, last_t, delta):
        output = []

        for (z, t) in zip(z_steps.mT, t_steps.T):
            delta_t = (t - last_t).reshape(1, 1).to(device=z.device, dtype=torch.float32).detach()
            delta_t.requires_grad = False
            last_t = t.detach()

            t1 = self.T1(delta_t)
            t2 = self.T2(delta_t)

            x = delta * h + (1 - delta) * self.rnn_ih(z)

            c_long = c_long * t1 + x * (1 - t1)
            c_short = c_long * t2 + x * (1 - t2)


            y = self.dmf(c_short).T

            output.append(y)
        completion = torch.cat(output, dim=1)
        return completion, c_short.data, c_long.data, last_t
