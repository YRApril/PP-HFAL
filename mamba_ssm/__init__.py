__version__ = "2.3.2.post1"

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "Mamba3",
    "MambaLMHeadModel",
]


def __getattr__(name):
    if name in {"selective_scan_fn", "mamba_inner_fn"}:
        from mamba_ssm.ops import selective_scan_interface

        return getattr(selective_scan_interface, name)
    if name == "Mamba":
        from mamba_ssm.modules.mamba_simple import Mamba

        return Mamba
    if name == "Mamba2":
        from mamba_ssm.modules.mamba2 import Mamba2

        return Mamba2
    if name == "Mamba3":
        from mamba_ssm.modules.mamba3 import Mamba3

        return Mamba3
    if name == "MambaLMHeadModel":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
