"""独立预测（提示词 §14）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.predict --config configs/base.yaml \
        --checkpoint outputs/full/last.pt --radius 0.08 --output outputs/predict/80mm.npz

**这条命令看不到任何目标半径的真值。** 它加载检查点里的物理参数与尺度，不读
目标半径的真实场量、初态或 ``R_c``/``R_s`` 轨迹，也不重跑目标半径的完整 CFD。
允许使用验证文件的时间戳与位置**作为查询坐标**，但真值不参与预测。

输出按**预测的** ``R_c``/``R_s`` 分区：``r < R_c`` 返回产物，``R_c < r < R_s``
返回空气，``r > R_s`` 返回环境态。界面处不给平均值，而是明确标出左右极限。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..models.coupled_model import CoupledModel
from ..models.product_net import ProductNet
from ..models.air_net import AirNet
from ..models.front_net import FrontNet
from ..models.mlp import resolve_device, resolve_dtype
from ..physics.similitude import (
    SimilitudeScales,
    physical_time_from_scaled,
    scaled_time_coverage,
)
from ._common import (
    SetupError,
    TRAINING_NOT_CONNECTED_MESSAGE,
    TrainingNotConnected,
    add_common_arguments,
    configure_stdout,
    load_config_from_args,
    write_json,
)


def load_model_from_checkpoint(path: Path, *, charge_radius: float, device=None) -> tuple[CoupledModel, dict[str, Any]]:
    """预测入口尚未接入新启动场检查点。

    旧的参考场检查点（带爆速与 ``T_h``）已经退出；新格式按 ``startup`` /
    ``scales`` 绑定启动场，而把预测网格、覆盖判据与前沿重建迁到
    ``PlanarStartupScales`` 属于后续工作。这里明确拒绝，不猜测旧数据的时间原点。
    """
    raise SetupError(
        "predict 尚未接入新启动场检查点：旧的参考场检查点不再读取，"
        "新的 startup/scales 契约还没有接到预测网格与前沿重建。"
    )

def predict_grid(
    model: CoupledModel,
    *,
    physical_times: np.ndarray,
    radii: np.ndarray,
    scaled_time_coverage: float | None = None,
) -> dict[str, Any]:
    """在**物理时间**网格 ``(t_phys, r)`` 上给出预测。

    输入的 ``physical_times`` 是秒，物理时间。``t_phys -> t_PINN -> s`` 的换算
    **只在这里做一次**：物理时间减一次 ``T_h = R_0/D + epsilon(R_0)``，得到 ``t_PINN``
    除以 ``R_0/U_*`` 再除以 ``hat T_*`` 得到 ``s``。输出里同时给出
    ``physical_time`` 与 ``pinn_time``，两者始终满足
    ``physical_time = pinn_time + T_h``，所以下游不需要再猜哪一个已经
    含过零点。这正是"在一处加、在另一处又加一次"最容易发生的地方：两个数组都
    只有秒的量纲，多平移一次不会报错，只会把轨迹整体错开。

    按预测前沿分区；域外为 ``nan``，不填任何替代值。超出训练**尺度化**时间
    覆盖（``s_max``）的查询标记为外推——判断在 ``s`` 上做，因此与目标半径无关。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = model.scales
    dtype = model.product_net.dtype_
    physical_times = np.asarray(physical_times, dtype=np.float64).reshape(-1)
    start = float(scales.start_time())
    pinn_times = physical_times - start
    if np.any(pinn_times < -1.0e-15):
        earliest = float(physical_times[np.argmin(pinn_times)])
        raise SetupError(
            f"a query at t_phys = {earliest:.6g} s precedes the PINN time origin "
            f"T_h = R_0/D + epsilon(R_0) = {start:.6g} s; the trained domain starts there"
        )
    pinn_times = np.maximum(pinn_times, 0.0)

    n_t = pinn_times.size
    n_r = radii.size

    s_values = scales.network_time(torch.as_tensor(pinn_times, dtype=dtype))
    s_values = s_values.reshape(-1, 1)
    radius = torch.as_tensor(radii, dtype=dtype).reshape(1, -1).expand(n_t, n_r)

    with torch.no_grad():
        positions = model.front_positions(s_values)
        contact_column = scales.charge_radius * positions.contact_hat
        shock_column = scales.charge_radius * positions.shock_hat
        # 前沿只随时间变，半径网格是另一维；掩码是 (n_t, n_r)，所以前沿必须先
        # 广播到同一形状再参与比较。
        contact = contact_column.expand(n_t, n_r)
        shock = shock_column.expand(n_t, n_r)

        product_mask = radius < contact
        air_mask = (radius >= contact) & (radius < shock)
        undisturbed = ~(product_mask | air_mask)

        out = {
            "density": torch.full((n_t, n_r), float("nan"), dtype=dtype),
            "velocity": torch.full((n_t, n_r), float("nan"), dtype=dtype),
            "specific_energy": torch.full((n_t, n_r), float("nan"), dtype=dtype),
            "pressure": torch.full((n_t, n_r), float("nan"), dtype=dtype),
        }
        for mask, fields in (
            (
                product_mask,
                model.product_fields(
                    s_values.expand(n_t, n_r)[product_mask].reshape(-1, 1),
                    (radius[product_mask] / contact[product_mask]).reshape(-1, 1),
                ),
            ),
            (
                air_mask,
                model.air_fields(
                    s_values.expand(n_t, n_r)[air_mask].reshape(-1, 1),
                    (
                        (radius[air_mask] - contact[air_mask])
                        / (shock[air_mask] - contact[air_mask])
                    ).reshape(-1, 1),
                ),
            ),
        ):
            if not bool(mask.any()):
                continue
            for name, value in (
                ("density", fields.density),
                ("velocity", fields.velocity),
                ("specific_energy", fields.specific_energy),
                ("pressure", fields.pressure),
            ):
                out[name][mask] = value.reshape(-1)
        for name, value in (
            ("density", model.undisturbed.density),
            ("velocity", model.undisturbed.velocity),
            ("specific_energy", model.undisturbed.specific_energy),
            ("pressure", model.undisturbed.pressure),
        ):
            out[name][undisturbed] = value

    # 前沿速度要 autograd 对 ``s`` 求导，所以**不能**包在 ``torch.no_grad()``
    # 里——那里面建不出计算图，``autograd.grad`` 会直接报"不 requires_grad"。
    # 场量是纯前向的，继续留在 no_grad 里省内存。
    s_grad = s_values.detach().clone().requires_grad_(True)
    state = model.front_net.state(s_grad, create_graph=False)
    contact_speed = state.contact_speed.detach().reshape(-1).numpy()
    shock_speed = state.shock_speed.detach().reshape(-1).numpy()

    # 外推判断在 ``s`` 上做，不在秒数上做：同样的 s 在大半径上对应更长的物理
    # 时间，拿秒数比大小会把覆盖内的点判成外推（或反过来）。
    s_flat = s_values.detach().reshape(-1).numpy()
    if scaled_time_coverage is None:
        extrapolated = np.zeros(n_t, dtype=bool)
    else:
        extrapolated = s_flat > float(scaled_time_coverage) + 1.0e-12

    return {
        "physical_time": physical_times,
        "pinn_time": pinn_times,
        "network_time": s_flat,
        "radius": np.asarray(radii),
        "density": out["density"].numpy(),
        "velocity": out["velocity"].numpy(),
        "specific_energy": out["specific_energy"].numpy(),
        "pressure": out["pressure"].numpy(),
        "overpressure": out["pressure"].numpy() - model.undisturbed.pressure,
        "contact_radius": contact_column.detach().reshape(-1).numpy(),
        "shock_radius": shock_column.detach().reshape(-1).numpy(),
        "contact_speed": contact_speed,
        "shock_speed": shock_speed,
        "region": np.where(
            product_mask.numpy(), 0, np.where(air_mask.numpy(), 1, 2)
        ).astype(np.int64),
        "extrapolated": extrapolated,
        "charge_radius": float(scales.charge_radius),
        "start_time": start,
        "time_origin": "physical_time = pinn_time + start_time, applied exactly once here",
        "scaled_time_coverage": (
            float("nan") if scaled_time_coverage is None else float(scaled_time_coverage)
        ),
        # 固定的物理尺度，供对照使用。**不能**由参考数据现算：那样分母会随被
        # 比较的那份数据变，误差就不再是同一个量。
        "scale_product_density": float(model.scales.product_density),
        "scale_air_density": float(model.scales.air_density),
        "scale_velocity": float(model.scales.reference_velocity),
        "scale_specific_energy": float(model.scales.reference_specific_energy),
        "ambient_pressure": float(model.undisturbed.pressure),
        "note": (
            "region: 0=product, 1=air, 2=ambient. NaN means outside every region. "
            "No value is ever filled in by clamping. physical_time = pinn_time + start_time."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """正式入口：新启动场的训练接口尚未接入，明确失败而不是回退旧方案。"""
    try:
        return _run(argv)
    except TrainingNotConnected as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 3


def _run(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--radius", type=float, required=True, help="目标装药半径，m")
    parser.add_argument("--output", default=None, help="预测结果 npz 路径")
    parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="查询终止的**物理**时间，s（省略时取训练尺度化覆盖的末端）",
    )
    parser.add_argument("--n-times", type=int, default=200)
    parser.add_argument("--radius-max-factor", type=float, default=12.0)
    parser.add_argument("--n-radii", type=int, default=600)
    args = parser.parse_args(argv)
    configure_stdout()
    raise TrainingNotConnected(TRAINING_NOT_CONNECTED_MESSAGE)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
