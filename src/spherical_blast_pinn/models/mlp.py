"""Shared building blocks for the network stack.

Everything is deterministic: given the same seed, the same configuration and
the same device, the initial parameters are bit-identical, which is what makes
the three-seed repeat runs comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_ACTIVATIONS = {
    "tanh": "tanh",
    "sin": "sin",
    "gelu": "gelu",
    "silu": "silu",
    "softplus": "softplus",
}


class ModelError(ValueError):
    """Raised when a network is misconfigured."""


def resolve_activation(name: str):
    """Map an activation name to a torch module factory."""
    if nn is None:  # pragma: no cover
        raise ModelError("PyTorch is required")
    key = (name or "tanh").strip().lower()
    if key not in _ACTIVATIONS:
        raise ModelError(f"unknown activation {name!r}; known: {sorted(_ACTIVATIONS)}")
    return {
        "tanh": nn.Tanh,
        "sin": lambda: _Sin(),
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
    }[key]()


class _Sin(nn.Module if nn is not None else object):  # type: ignore[misc]
    """``sin`` activation, used only for Fourier-style feature layers."""

    def forward(self, x):  # pragma: no cover - trivial
        return torch.sin(x)


@dataclass(frozen=True)
class MLPConfig:
    """Configuration of one multi-layer perceptron."""

    input_dim: int
    output_dim: int
    width: int = 128
    depth: int = 8
    activation: str = "tanh"
    fourier_features: int = 0
    fourier_scale: float = 1.0
    output_bias: float = 0.0
    #: Multiplier on the Xavier initialisation of the *output* layer's weights.
    #: ``0.0`` makes the network's initial output exactly ``output_bias`` at
    #: every input, which is what a trajectory front needs: its initial state
    #: should be the one the configuration asks for, not that value plus an
    #: unmeasured scatter from the hidden layers.
    output_weight_scale: float = 1.0

    def validate(self) -> None:
        if self.input_dim < 1 or self.output_dim < 1:
            raise ModelError("input_dim and output_dim must both be positive")
        if self.width < 1 or self.depth < 1:
            raise ModelError("width and depth must both be positive")
        if self.fourier_features < 0:
            raise ModelError("fourier_features must be non-negative")

    @classmethod
    def from_mapping(cls, section: dict[str, Any], *, input_dim: int, output_dim: int) -> "MLPConfig":
        return cls(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            width=int(section.get("width", 128)),
            depth=int(section.get("depth", 8)),
            activation=str(section.get("activation", "tanh")),
            fourier_features=int(section.get("fourier_features", 0) or 0),
            fourier_scale=float(section.get("fourier_scale", 1.0)),
        )


class FourierFeatures(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Random Fourier features with a *fixed* seed.

    The frequencies are drawn from a generator seeded explicitly rather than
    from the global RNG, so that changing the training seed does not silently
    change the input encoding.  The matrix is a registered buffer, so it follows
    the module to a device and is excluded from the optimiser.
    """

    def __init__(self, input_dim: int, n_features: int, scale: float, seed: int = 0, dtype=None):
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        dtype = dtype or torch.float64
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        matrix = torch.randn(input_dim, n_features, generator=generator, dtype=torch.float64)
        self.register_buffer("frequencies", (matrix * float(scale)).to(dtype=dtype))

    def forward(self, x):
        projected = x @ self.frequencies
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    @property
    def output_dim(self) -> int:
        return int(self.frequencies.shape[1]) * 2


class MLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    """A plain MLP with optional Fourier input encoding.

    Xavier (Glorot) initialisation on the ``tanh`` layers keeps the pre-activation
    variance roughly constant with depth, which matters at the default depth of
    eight hidden layers.
    """

    def __init__(self, config: MLPConfig, *, seed: int = 0, dtype=None):
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        config.validate()
        self.config = config
        self.dtype_ = dtype or torch.float64

        encoder = None
        encoded_dim = config.input_dim
        if config.fourier_features > 0:
            encoder = FourierFeatures(
                config.input_dim,
                config.fourier_features,
                config.fourier_scale,
                seed=seed,
                dtype=self.dtype_,
            )
            encoded_dim = encoder.output_dim
        self.encoder = encoder

        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        layers: list[Any] = []
        previous = encoded_dim
        for _ in range(config.depth):
            linear = nn.Linear(previous, config.width, dtype=self.dtype_)
            _xavier_uniform_(linear.weight, generator)
            with torch.no_grad():
                linear.bias.zero_()
            layers.append(linear)
            layers.append(resolve_activation(config.activation))
            previous = config.width
        output = nn.Linear(previous, config.output_dim, dtype=self.dtype_)
        _xavier_uniform_(output.weight, generator)
        if config.output_weight_scale != 1.0:
            with torch.no_grad():
                output.weight.mul_(float(config.output_weight_scale))
        with torch.no_grad():
            output.bias.fill_(config.output_bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.network(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "width": self.config.width,
            "depth": self.config.depth,
            "activation": self.config.activation,
            "fourier_features": self.config.fourier_features,
            "parameters": self.parameter_count(),
        }


def _xavier_uniform_(tensor, generator) -> None:
    """Xavier/Glorot uniform initialisation with an explicit generator."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    with torch.no_grad():
        uniform = torch.empty(tensor.shape, dtype=tensor.dtype).uniform_(
            -limit, limit, generator=generator
        )
        tensor.copy_(uniform)


def set_seed(seed: int) -> None:
    """Seed every RNG the training loop touches."""
    if torch is None:  # pragma: no cover
        raise ModelError("PyTorch is required")
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


#: Set by :func:`resolve_device` so the run provenance can record why a
#: fallback happened instead of leaving it invisible.
DEVICE_NOTES: list[str] = []


def cuda_is_usable(index: int = 0) -> tuple[bool, str]:
    """Whether CUDA can actually execute a kernel, not merely whether it is reported.

    ``torch.cuda.is_available()`` returns ``True`` on machines whose GPU is too
    old for the installed CUDA build; the failure only appears at the first
    kernel launch, usually several frames into a run.  A one-element probe with
    an explicit synchronise turns that into an immediate, reportable answer.
    """
    if torch is None:  # pragma: no cover
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    if torch.cuda.device_count() <= index:
        return False, f"only {torch.cuda.device_count()} CUDA device(s) present"
    try:
        probe = torch.zeros(1, dtype=torch.float64, device=f"cuda:{index}")
        probe = probe + 1.0
        torch.cuda.synchronize(index)
        del probe
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, f"CUDA probe failed: {type(exc).__name__}: {exc}"
    return True, "CUDA probe succeeded"


def resolve_device(name: str, *, probe: bool = True) -> Any:
    """``auto`` picks CUDA when it is usable, else the CPU.

    ``probe=True`` runs a one-element kernel before committing to CUDA.  The
    reason for any fallback is appended to :data:`DEVICE_NOTES` and ends up in
    the run provenance.
    """
    if torch is None:  # pragma: no cover
        raise ModelError("PyTorch is required")
    key = (name or "auto").strip().lower()
    if key == "auto":
        usable, reason = cuda_is_usable() if probe else (torch.cuda.is_available(), "")
        if usable:
            DEVICE_NOTES.append("device=auto selected cuda; " + reason)
            return torch.device("cuda")
        DEVICE_NOTES.append(f"device=auto fell back to cpu ({reason})")
        return torch.device("cpu")
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ModelError(f"device {name!r} was requested but CUDA is not available")
        if probe:
            usable, reason = cuda_is_usable()
            if not usable:
                raise ModelError(
                    f"device {name!r} was requested explicitly but {reason}. "
                    "Use device: cpu or device: auto."
                )
        return torch.device(key)
    return torch.device(key)


def resolve_dtype(name: str) -> Any:
    if torch is None:  # pragma: no cover
        raise ModelError("PyTorch is required")
    key = (name or "float64").strip().lower()
    table = {"float64": torch.float64, "double": torch.float64,
             "float32": torch.float32, "single": torch.float32}
    if key not in table:
        raise ModelError(f"unknown dtype {name!r}; use float64 or float32")
    return table[key]


def as_column(x, *, dtype=None, device=None):
    """Reshape a 1-D tensor to ``(n, 1)`` for evaluation.

    A 2-D input is flattened to a column, which is what the networks expect:
    they always see one trailing axis.  Passing a genuinely two-dimensional grid
    (say ``(n_batch, n_quadrature)``) therefore requires flattening it first and
    reshaping the result — see ``losses.conservation_loss._batched_integral``.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=dtype or torch.float64, device=device)
    return x.reshape(-1, 1)


def as_grad_leaf(x, *, dtype=None, device=None):
    """Detach ``x`` and mark it as a leaf that requires grad.

    The sampler hands out constants.  Any loss that needs a derivative with
    respect to a *coordinate* — the Euler residual with respect to ``(r, tau)``,
    or a front speed with respect to ``tau`` — must first turn that coordinate
    into a leaf, otherwise ``torch.autograd.grad`` fails with a message that
    points at the wrong place entirely.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=dtype or torch.float64, device=device)
    x = as_column(x)
    return x.detach().requires_grad_(True)


__all__ = [
    "FourierFeatures",
    "MLP",
    "MLPConfig",
    "ModelError",
    "DEVICE_NOTES",
    "as_column",
    "cuda_is_usable",
    "resolve_activation",
    "resolve_device",
    "resolve_dtype",
    "set_seed",
]
