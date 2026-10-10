r"""网络层：两个场网络 + 一个前沿网络。

.. math::

    \texttt{ProductNet}(s, \xi^2) &\to (\rho_p, u_p, e_p), \quad p_p = p_{JWL} \\
    \texttt{AirNet}(s, \eta)     &\to (\rho_a, u_a, e_a), \quad p_a = (\gamma_a - 1)\rho_a e_a \\
    \texttt{FrontNet}(s)         &\to (\hat R_c, \hat h)

两个场网络**不共享权重**：它们遵守不同的状态方程，通过接触面与主激波条件
耦合，而不是通过共享参数耦合。
"""
from __future__ import annotations

from .air_net import AirFields, AirNet
from .coupled_model import (
    GROUP_AIR,
    GROUP_FRONT,
    GROUP_PRODUCT,
    PARAMETER_GROUPS,
    CoupledModel,
    UndisturbedState,
    build_model,
)
from .front_net import (
    MINIMUM_GAP_HAT,
    SOFTPLUS_UNITY_ARGUMENT,
    FrontNet,
    FrontPositions,
    FrontState,
)
from .mlp import (
    MLP,
    MLPConfig,
    ModelError,
    as_column,
    resolve_activation,
    resolve_device,
    resolve_dtype,
)
from .product_net import ProductFields, ProductNet

__all__ = [
    "GROUP_AIR",
    "GROUP_FRONT",
    "GROUP_PRODUCT",
    "MINIMUM_GAP_HAT",
    "MLP",
    "MLPConfig",
    "PARAMETER_GROUPS",
    "SOFTPLUS_UNITY_ARGUMENT",
    "AirFields",
    "AirNet",
    "CoupledModel",
    "FrontNet",
    "FrontPositions",
    "FrontState",
    "ModelError",
    "ProductFields",
    "ProductNet",
    "UndisturbedState",
    "as_column",
    "build_model",
    "resolve_activation",
    "resolve_device",
    "resolve_dtype",
]
