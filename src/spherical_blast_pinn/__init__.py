"""自由空气中球形 TNT 爆炸的 1D 球对称双场 PINN。

两个场网络全程并发求解：``ProductNet`` 在 ``0 <= r < R_c`` 上给出爆轰产物，
``AirNet`` 在 ``R_c < r < R_s`` 上给出受扰动空气；``FrontNet`` 由统一时间 ``s``
直接预测两条前沿。``r >= R_s`` 是解析的环境态，没有网络也没有标签。

时间约定是 ``t_PINN = t_phys - epsilon(R_0)``，其中
``epsilon(R_0) = (R_0/R_ref) epsilon_ref``，只依赖半径。参考初始场由
:mod:`spherical_blast_pinn.physics.precursor` 独立求解，不由任何仿真帧改名而来。

快速开始
--------

.. code-block:: bash

    python -m spherical_blast_pinn.cli.audit_dyna   --config configs/experiments/smoke.yaml
    python -m spherical_blast_pinn.cli.reference_field --config configs/experiments/smoke.yaml
    python -m spherical_blast_pinn.cli.train        --config configs/experiments/smoke.yaml
    python -m spherical_blast_pinn.cli.predict      --config configs/base.yaml --radius 0.08
    python -m spherical_blast_pinn.cli.compare      --config configs/base.yaml --radius 0.08
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
