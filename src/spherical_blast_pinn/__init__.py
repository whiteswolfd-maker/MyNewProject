"""自由空气中球形 TNT 爆炸的 1D 球对称双场 PINN。

两个场网络全程并发求解：``ProductNet`` 在 ``0 <= r < R_c`` 上给出爆轰产物，
``AirNet`` 在 ``R_c < r < R_s`` 上给出受扰动空气；``FrontNet`` 由统一时间 ``s``
直接预测两条前沿。``r >= R_s`` 是解析的环境态，没有网络也没有标签。

初值只有一个来源（2026-10-21 决策、2026-10-25 改为局部平面黎曼启动）：
``t_phys = 0`` 时均匀静止的 JWL 产物球 + 静止空气，经 JWL 等熵释放与空气激波
RH 的局部平面黎曼匹配，在小的正时间 ``epsilon`` 得到
:class:`~spherical_blast_pinn.physics.two_medium_startup_planar.PlanarStartupField`。
新链路的时间约定是 ``tau = t_phys - epsilon``，``s = 0`` 对应该场自己的
``epsilon``（:class:`~spherical_blast_pinn.physics.similitude.PlanarStartupScales`）。
旧的爆轰产物 + 拉格朗日双材料前置参考场已经删除。

快速开始
--------

.. code-block:: bash

    python -m spherical_blast_pinn.cli.startup                 # 写出 handoff.npz/json
    python -m spherical_blast_pinn.cli.train --config configs/experiments/startup_diagnostic.yaml \
        --diagnostic --max-steps 10                            # 程序诊断训练
    python -m spherical_blast_pinn.cli.predict --checkpoint ... --handoff-json ... --tau 1e-7
    python -m spherical_blast_pinn.cli.compare --prediction ... --reference ...

启动场不再以 ``training_qualified`` 作为运行门槛；该字段只作为来源记录保留。
程序正确性检查（数值、EOS、格式、解析/恢复绑定、隔离）仍然硬失败。
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
