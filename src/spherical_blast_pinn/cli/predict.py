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
from ..training.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    refuse_if_uncommitted,
    require_detonation_velocity,
)
from ._common import (
    SetupError,
    add_common_arguments,
    configure_stdout,
    load_config_from_args,
    write_json,
)


def load_model_from_checkpoint(path: Path, *, charge_radius: float, device=None) -> tuple[CoupledModel, dict[str, Any]]:
    """由检查点在**新半径**上重建模型。

    两个场网络与前沿网络都不把 ``R_0`` 作为输入——半径的作用已经被尺度化吸收
    ——所以同一个 ``state_dict`` 可以直接装进为另一个半径建立的模型里。这不是
    近似：这正是"共用同一套尺度化网络"这句话的实现方式。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    path = Path(path)
    # **发布完整性：与 ``Trainer.load`` 用同一条规则。** 预测同样会拿一套"一部分新、
    # 一部分旧"的产物当真值使用，所以"训练恢复被拒绝"不能是唯一的防线——以前只有
    # 训练入口查这个标记，复核实测：一个被恢复拒绝的未提交检查点，预测照样读进去。
    refuse_if_uncommitted(path, error=SetupError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SetupError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise SetupError(
            f"checkpoint version {payload.get('version')} is not supported by this "
            f"build, which reads version {CHECKPOINT_VERSION}; retrain rather than "
            "reinterpreting an older field layout"
        )

    time_scale = float(payload["time_scale"])
    # 爆速随检查点冻结：时间原点 T_h = R_0/D + epsilon 必须与训练时完全一致，
    # 否则推理的整条轨迹会平移一个 T_b。读取与校验走**共用**的那条规则
    # （``training.checkpoint``），保存端写什么、这边要求什么只有一个定义。
    detonation_velocity = require_detonation_velocity(payload, path, error=SetupError)
    scales = SimilitudeScales.for_radius(
        charge_radius,
        time_scale=time_scale,
        detonation_velocity=detonation_velocity,
    )
    state = payload["model"]

    # 架构从检查点清单里读，**不从 ``state_dict`` 的键名反推**。``nn.Sequential``
    # 把每个 Linear 后面接一个激活，于是 Linear 落在偶数下标上；按 "最大下标除以
    # 2 加一" 去猜会得到 depth+1，然后 state_dict 加载失败。清单里本来就记着
    # width 与 depth，读它比猜它可靠。
    provenance = payload["provenance"]
    architecture = provenance["model"]
    dtype = resolve_dtype(str(payload.get("config", {}).get("networks", {}).get("dtype", "float64")))

    product = ProductNet(
        scales=scales,
        width=int(architecture["product"]["width"]),
        depth=int(architecture["product"]["depth"]),
        activation=str(architecture["product"]["activation"]),
        dtype=dtype,
    )
    air = AirNet(
        scales=scales,
        width=int(architecture["air"]["width"]),
        depth=int(architecture["air"]["depth"]),
        activation=str(architecture["air"]["activation"]),
        gamma_air=float(provenance["material"]["gamma_air"]),
        dtype=dtype,
    )
    front = FrontNet(
        scales=scales,
        width=int(architecture["front"]["width"]),
        depth=int(architecture["front"]["depth"]),
        activation=str(architecture["front"]["activation"]),
        initial_contact_hat=float(architecture["front"]["initial_contact_hat"]),
        initial_gap_hat=float(architecture["front"]["initial_gap_hat"]),
        anchor_contact_at_charge_surface=bool(
            architecture["front"].get("anchored_contact", False)
        ),
        anchor_until_s=architecture["front"].get("anchor_until_s", None),
        dtype=dtype,
    )
    model = CoupledModel(
        product=product,
        air=air,
        front=front,
        scales=scales,
        gamma_air=float(payload["provenance"]["material"]["gamma_air"]),
        ambient_density=float(payload["provenance"]["material"]["ambient"]["rho_ambient"]),
        ambient_pressure=float(payload["provenance"]["material"]["ambient"]["p_ambient"]),
    )
    model.load_state_dict(state)
    target = device if device is not None else resolve_device("cpu")
    model.to(target)
    model.eval()
    return model, payload


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
    config = load_config_from_args(args)

    model, payload = load_model_from_checkpoint(Path(args.checkpoint), charge_radius=args.radius)

    # 训练覆盖：在 ``s`` 上是**一个与半径无关的数**，由检查点里冻结的 ``hat T_*``
    # 与各训练算例的最后有效 ``t_PINN`` 得到。目标半径的真值不参与。
    time_scale = float(payload["time_scale"])
    # 同一个冻结爆速，走同一条读取规则。默认 ``--time-end`` 要把 ``s_max`` 映回
    # 物理时间，而那个映射里含 ``T_h = R_0/D + epsilon``，所以它也要这个值。
    detonation_velocity = require_detonation_velocity(
        payload, args.checkpoint, error=SetupError
    )
    valid = dict(payload.get("provenance", {}).get("valid_pinn_time_max", {}) or {})
    # 检查点里 ``valid_pinn_time_max`` 以 case_id 为键；覆盖判据需要的是半径，
    # 因为 hat_t = U_* t_PINN / R_0。半径从同一份检查点的 scales 里读，不猜。
    radius_of_case = {
        str(case_id): float(entry["charge_radius"])
        for case_id, entry in (payload.get("scales") or {}).items()
    }
    coverage_provenance = None
    if valid:
        missing = [case_id for case_id in valid if case_id not in radius_of_case]
        if missing:
            raise SetupError(
                f"the checkpoint records valid times for {missing} but no charge radius "
                "for them; the scaled-time coverage cannot be formed"
            )
        s_max, coverage_provenance = scaled_time_coverage(
            {radius_of_case[case_id]: float(value) for case_id, value in valid.items()},
            time_scale=time_scale,
        )
    else:
        s_max = None

    if args.time_end is not None:
        end_physical = float(args.time_end)
        end_source = "--time-end (physical seconds, given explicitly)"
        if end_physical < model.scales.start_time():
            raise SetupError(
                f"--time-end = {end_physical:.6g} s precedes the PINN time origin "
                f"T_h = R_0/D + epsilon(R_0) = {model.scales.start_time():.6g} s for this radius"
            )
    elif s_max is not None:
        # ``detonation_velocity`` 是必填的：``t_phys = s hat T_* R_0 / U_* + T_h``
        # 里的 ``T_h`` 就是 ``R_0/D + epsilon``。以前这里没传它，于是**默认**
        # ``--time-end`` 这条路一走到就 ``TypeError``——而它从来没被测过，因为
        # 没有一条测试真的从训练检查点走到预测（复核指出的断链）。
        end_physical = physical_time_from_scaled(
            s_max,
            args.radius,
            time_scale=time_scale,
            detonation_velocity=detonation_velocity,
        )
        end_source = (
            "default: the end of the training scaled-time coverage, mapped to this "
            "radius through t_phys = s hat T_* R_0 / U_* + T_h"
        )
    else:
        raise SetupError(
            "the checkpoint records no valid t_PINN range, so there is no training "
            "coverage to default --time-end from; pass --time-end explicitly"
        )

    physical_times = np.linspace(
        model.scales.start_time(), end_physical, int(args.n_times)
    )
    radii = np.linspace(
        args.radius * 1.0e-3, args.radius * float(args.radius_max_factor), int(args.n_radii)
    )
    result = predict_grid(
        model,
        physical_times=physical_times,
        radii=radii,
        scaled_time_coverage=s_max,
    )
    out_dir = Path(args.output) if args.output else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "predict"
    out_path = Path(out_dir)
    if out_path.suffix != ".npz":
        out_path = out_path / f"predict_R{int(round(args.radius * 1000)):04d}mm.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **result)
    write_json(out_path.with_suffix(".json"), {
        "charge_radius_m": args.radius,
        "start_time_s": result["start_time"],
        "time_origin": result["time_origin"],
        "time_end_physical_s": end_physical,
        "time_end_source": end_source,
        "n_times": int(physical_times.size),
        "n_radii": int(radii.size),
        "scaled_time_coverage_s_max": s_max,
        "scaled_time_coverage_provenance": coverage_provenance,
        "n_extrapolated_times": int(result["extrapolated"].sum()),
        "checkpoint": str(args.checkpoint),
        "physics_parameters_source": "checkpoint provenance, not the target radius data",
        "target_radius_truth_used": False,
    })
    print(f"predicted on {physical_times.size} times x {radii.size} radii")
    print(f"physical time range: [{physical_times[0]:.6g}, {physical_times[-1]:.6g}] s")
    print(f"R_c at t_end = {result['contact_radius'][-1]:.6g} m")
    print(f"R_s at t_end = {result['shock_radius'][-1]:.6g} m")
    if s_max is not None:
        n_extra = int(result["extrapolated"].sum())
        print(f"training scaled-time coverage: s_max = {s_max:.6g}"
              + (f"; {n_extra} query time(s) beyond it are flagged extrapolated"
                 if n_extra else "; all query times are inside it"))
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
