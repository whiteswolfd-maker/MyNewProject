"""独立预测（新启动场时间轴）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.predict \
        --checkpoint outputs/run/last.pt \
        --handoff-json outputs/planar_startup/<model>/<key>/handoff.json \
        --tau 0,1e-7,2e-7 --output outputs/predict/run.npz

时间参数**必须显式**：``--t-phys``（物理时间，秒）与 ``--tau``（从启动场时刻
算起的时长，秒）二选一，不接受"默认到训练末端"这种含糊参数。
``tau = t_phys - epsilon_field``，``s = U_* tau / (R_0 hat T_*)``。

输出按**预测的**前沿分区：产物侧 ``0 <= r < Rc``、空气侧 ``Rc < r < Rs``、
环境侧 ``r >= Rs``。两侧界面各取自己的单侧极限，不跨界面插值，也不在界面处
做平均。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..models.coupled_model import CoupledModel, build_model_from_description
from ..physics.similitude import PlanarStartupScales
from ..training.checkpoint import (
    SCALES_KEY,
    TrainingError,
    inspect_startup_checkpoint,
    load_frozen_model,
    load_startup_checkpoint,
    planar_scales_from_record,
)
from ..physics.two_medium_startup_handoff import load_planar_startup_handoff
from ._common import SetupError, configure_stdout

#: 界面单侧极限的偏移（在相似坐标上）。它只表示"取左/右极限"，不改变物理。
SIDE_EPS = 1.0e-9


def _parse_time_list(text: str, label: str) -> np.ndarray:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise SetupError(f"{label}: {part!r} is not a number") from exc
        if not math.isfinite(value):
            raise SetupError(f"{label}: {part!r} is not finite")
        values.append(value)
    if not values:
        raise SetupError(f"{label} contains no times")
    return np.asarray(values, dtype=np.float64)


@dataclass
class _SelectedCase:
    record: dict[str, Any]
    case_id: str
    field: Any
    scales: PlanarStartupScales
    model: CoupledModel
    mode: str
    trained_case_ids: list[str]
    tau_end: float
    window_fraction: float


def _matching_record(
    cases: Sequence[dict[str, Any]], *, radius: float
) -> dict[str, Any] | None:
    matching = [
        record for record in cases
        if abs(
            float(record[SCALES_KEY]["charge_radius"]) - float(radius)
        ) <= 1.0e-12 * max(abs(float(radius)), 1.0)
    ]
    if len(matching) > 1:
        raise SetupError(
            f"more than one checkpoint case has R_0 = {radius!r} m; pass --case-id"
        )
    return None if not matching else matching[0]


def _build_case(
    checkpoint: Path, *, handoff_json: Path, case_id: str | None, radius: float | None
) -> _SelectedCase:
    inspected = inspect_startup_checkpoint(checkpoint)
    cases = inspected["cases"]
    trained_ids = [str(record["case_id"]) for record in cases]
    field = load_planar_startup_handoff(handoff_json)
    record = None
    if case_id is not None:
        matched = [r for r in cases if str(r["case_id"]) == case_id]
        if not matched:
            raise SetupError(
                f"no trained case {case_id!r}; the checkpoint binds {trained_ids}. "
                "For an untrained compatible radius use --radius without --case-id."
            )
        record = matched[0]
        if radius is not None and abs(
            float(record[SCALES_KEY]["charge_radius"]) - float(radius)
        ) > 1.0e-12 * max(abs(float(radius)), 1.0):
            raise SetupError(
                f"--case-id {case_id!r} has R_0="
                f"{float(record[SCALES_KEY]['charge_radius'])!r} m, not {radius!r} m"
            )
    elif radius is not None:
        record = _matching_record(cases, radius=float(radius))
    elif len(cases) == 1:
        record = cases[0]
    else:
        raise SetupError(
            "this checkpoint binds several cases; pass --case-id (trained) or "
            "--radius (trained or compatible untrained)"
        )

    reference = record if record is not None else cases[0]
    time_scale = float(reference[SCALES_KEY]["time_scale"])
    if record is None:
        # 未训练半径：用冻结的共享参数；只要求材料/环境/delta/方法/时间尺度
        # 与检查点兼容，cache_key 可以与训练算例不同。
        scales = PlanarStartupScales.for_startup_field(field, time_scale=time_scale)
        model = build_model_from_description(
            inspected["model"]["model_structure"], scales=scales, device="cpu"
        )
        load_frozen_model(
            checkpoint,
            handoff_json=handoff_json,
            scales=scales,
            model=model,
        )
        mode = "frozen_shared_parameters"
        selected_id = f"untrained:{field.charge_radius:g}"
    else:
        scales = planar_scales_from_record(
            record[SCALES_KEY], where="checkpoint.scales"
        )
        model = build_model_from_description(
            inspected["model"]["model_structure"], scales=scales, device="cpu"
        )
        load_startup_checkpoint(
            checkpoint,
            handoff_json=handoff_json,
            model=model,
            scales=scales,
            restore_rng=False,
            weights_only=True,
            select_case_id=str(record["case_id"]),
        )
        mode = "trained_case_reproduction"
        selected_id = str(record["case_id"])

    tau_end = float(reference["sampling"]["tau_end_s"])
    window_fraction = float(reference["sampling"]["window"]["fraction"])
    return _SelectedCase(
        record=reference,
        case_id=selected_id,
        field=field,
        scales=scales,
        model=model,
        mode=mode,
        trained_case_ids=trained_ids,
        tau_end=tau_end,
        window_fraction=window_fraction,
    )


def _profile_blocks(
    case: _SelectedCase,
    *,
    tau: float,
    n_product: int,
    n_air: int,
    radius_max_factor: float,
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = case.scales
    model = case.model
    dtype = model.product_net.dtype_
    s_value = float(scales.network_time(torch.as_tensor(tau, dtype=dtype)))
    s = torch.tensor([[s_value]], dtype=dtype, requires_grad=True)
    state = model.front_net.state(
        s,
        charge_radius=scales.charge_radius,
        create_graph=False,
        scales=scales,
    )
    contact = float(state.contact_radius.detach())
    shock = float(state.shock_radius.detach())
    contact_speed = float(state.contact_speed.detach())
    shock_speed = float(state.shock_speed.detach())
    if not (math.isfinite(contact) and math.isfinite(shock) and 0.0 < contact < shock):
        raise SetupError(
            f"the model predicts Rc={contact!r}, Rs={shock!r} at tau={tau!r}; "
            "a valid profile needs 0 < Rc < Rs"
        )
    n_product = max(2, int(n_product))
    n_air = max(2, int(n_air))
    xi = np.linspace(0.0, 1.0 - SIDE_EPS, n_product)
    eta = np.linspace(SIDE_EPS, 1.0 - SIDE_EPS, n_air)
    product_radius = contact * xi
    air_radius = contact + eta * (shock - contact)
    ambient_radius = np.asarray(
        [shock, shock * (1.0 + 1.0e-6), max(shock * 1.05, scales.charge_radius * float(radius_max_factor))]
    )
    with torch.no_grad():
        product = model.product_fields(
            torch.full((n_product, 1), s_value, dtype=dtype),
            torch.as_tensor(xi.reshape(-1, 1), dtype=dtype),
            scales=scales,
        )
        air = model.air_fields(
            torch.full((n_air, 1), s_value, dtype=dtype),
            torch.as_tensor(eta.reshape(-1, 1), dtype=dtype),
            scales=scales,
        )
        ambient = model.undisturbed
        product_values = {
            "density": product.density.detach().numpy().reshape(-1),
            "velocity": product.velocity.detach().numpy().reshape(-1),
            "specific_energy": product.specific_energy.detach().numpy().reshape(-1),
            "pressure": product.pressure.detach().numpy().reshape(-1),
        }
        air_values = {
            "density": air.density.detach().numpy().reshape(-1),
            "velocity": air.velocity.detach().numpy().reshape(-1),
            "specific_energy": air.specific_energy.detach().numpy().reshape(-1),
            "pressure": air.pressure.detach().numpy().reshape(-1),
        }
    ambient_values = {
        "density": np.full(ambient_radius.size, ambient.density),
        "velocity": np.full(ambient_radius.size, ambient.velocity),
        "specific_energy": np.full(ambient_radius.size, ambient.specific_energy),
        "pressure": np.full(ambient_radius.size, ambient.pressure),
    }
    regions = (
        np.zeros(product_radius.size, dtype=np.int8),
        np.ones(air_radius.size, dtype=np.int8),
        np.full(ambient_radius.size, 2, dtype=np.int8),
    )
    radius = np.concatenate([product_radius, air_radius, ambient_radius])
    values = {
        name: np.concatenate([product_values[name], air_values[name], ambient_values[name]])
        for name in ("density", "velocity", "specific_energy", "pressure")
    }
    return {
        "tau": float(tau),
        "physical_time": float(scales.physical_time_from_network(s_value)),
        "network_time": s_value,
        "radius": radius,
        "region": np.concatenate(regions),
        **values,
        "overpressure": values["pressure"] - float(ambient.pressure),
        "contact_radius": contact,
        "shock_radius": shock,
        "contact_speed": contact_speed,
        "shock_speed": shock_speed,
        "product_points": int(product_radius.size),
        "air_points": int(air_radius.size),
        "ambient_points": int(ambient_radius.size),
        "contact_left": {
            "radius": contact * (1.0 - SIDE_EPS),
            "density": float(product_values["density"][-1]),
            "velocity": float(product_values["velocity"][-1]),
            "specific_energy": float(product_values["specific_energy"][-1]),
            "pressure": float(product_values["pressure"][-1]),
        },
        "contact_right": {
            "radius": contact + (shock - contact) * SIDE_EPS,
            "density": float(air_values["density"][0]),
            "velocity": float(air_values["velocity"][0]),
            "specific_energy": float(air_values["specific_energy"][0]),
            "pressure": float(air_values["pressure"][0]),
        },
        "shock_post": {
            "radius": shock - (shock - contact) * SIDE_EPS,
            "density": float(air_values["density"][-1]),
            "velocity": float(air_values["velocity"][-1]),
            "specific_energy": float(air_values["specific_energy"][-1]),
            "pressure": float(air_values["pressure"][-1]),
        },
        "ambient": {
            "radius": float(ambient_radius[0]),
            "density": float(ambient.density),
            "velocity": float(ambient.velocity),
            "specific_energy": float(ambient.specific_energy),
            "pressure": float(ambient.pressure),
        },
    }


def predict_grid(
    model: CoupledModel,
    *,
    physical_times: np.ndarray,
    radii: np.ndarray,
    scaled_time_coverage: float | None = None,
) -> dict[str, Any]:
    """在固定 ``(t_phys, r)`` 网格上给出分区预测（不跨界面插值）。

    保留这个接口给独立调用；CLI 的重建流程用 :func:`_profile_blocks`，因为它
    输出的是按材料侧分段的点，界面两侧各取单侧极限。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = model.scales
    dtype = model.product_net.dtype_
    physical_times = np.asarray(physical_times, dtype=np.float64).reshape(-1)
    epsilon = float(scales.start_time())
    tau = physical_times - epsilon
    if np.any(tau < -1.0e-15):
        earliest = float(physical_times[np.argmin(tau)])
        raise SetupError(
            f"a query at t_phys = {earliest:.6g} s precedes the startup origin "
            f"epsilon = {epsilon:.6g} s"
        )
    tau = np.maximum(tau, 0.0)
    s_values = scales.network_time(torch.as_tensor(tau, dtype=dtype)).reshape(-1, 1)
    radius = torch.as_tensor(radii, dtype=dtype).reshape(1, -1).expand(
        tau.size, np.asarray(radii).size
    )
    with torch.no_grad():
        positions = model.front_positions(s_values)
        contact = scales.charge_radius * positions.contact_hat
        shock = scales.charge_radius * positions.shock_hat
        contact = contact.expand_as(radius)
        shock = shock.expand_as(radius)
        product_mask = radius < contact
        air_mask = (radius >= contact) & (radius < shock)
        undisturbed = ~(product_mask | air_mask)
        out = {
            name: torch.full(radius.shape, float("nan"), dtype=dtype)
            for name in ("density", "velocity", "specific_energy", "pressure")
        }
        if bool(product_mask.any()):
            fields = model.product_fields(
                s_values.expand_as(radius)[product_mask].reshape(-1, 1),
                (radius[product_mask] / contact[product_mask]).reshape(-1, 1),
                scales=model.scales,
            )
            for name in out:
                out[name][product_mask] = getattr(fields, name).reshape(-1)
        if bool(air_mask.any()):
            fields = model.air_fields(
                s_values.expand_as(radius)[air_mask].reshape(-1, 1),
                (
                    (radius[air_mask] - contact[air_mask])
                    / (shock[air_mask] - contact[air_mask])
                ).reshape(-1, 1),
                scales=model.scales,
            )
            for name in out:
                out[name][air_mask] = getattr(fields, name).reshape(-1)
        for name in out:
            out[name][undisturbed] = getattr(model.undisturbed, name if name != "specific_energy" else "specific_energy")
    s_flat = s_values.detach().reshape(-1).numpy()
    extrapolated = (
        np.zeros(tau.size, dtype=bool)
        if scaled_time_coverage is None
        else s_flat > float(scaled_time_coverage) + 1.0e-12
    )
    return {
        "physical_time": physical_times,
        # ``pinn_time`` 是旧名字，等价于 ``tau = t_phys - epsilon``；两者同时给出
        # 且只平移一次，便于旧对照代码读取而不需要再猜时间零点。
        "pinn_time": tau,
        "tau": tau,
        "start_time": epsilon,
        "network_time": s_flat,
        "radius": np.asarray(radii),
        "density": out["density"].numpy(),
        "velocity": out["velocity"].numpy(),
        "specific_energy": out["specific_energy"].numpy(),
        "pressure": out["pressure"].numpy(),
        "overpressure": out["pressure"].numpy() - model.undisturbed.pressure,
        "contact_radius": contact.detach().reshape(-1).numpy(),
        "shock_radius": shock.detach().reshape(-1).numpy(),
        "region": np.where(product_mask.numpy(), 0, np.where(air_mask.numpy(), 1, 2)),
        "extrapolated": extrapolated,
        "epsilon_s": epsilon,
        "time_convention": scales.time_convention,
    }


def _write_npz(path: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value for key, value in payload.items()
        if isinstance(value, np.ndarray)
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, default=str))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (SetupError, TrainingError) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--handoff-json", required=True,
        help="该检查点绑定的 handoff.json（显式给出，不依赖检查点里的路径）",
    )
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--radius", type=float, default=None, help="按 R_0 选算例，m")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--t-phys", default=None, help="物理时间，秒（逗号分隔，可多个）")
    group.add_argument("--tau", default=None, help="从 epsilon 起算的时长，秒")
    parser.add_argument("--output", default=None, help="输出 .npz（缺省写到检查点旁）")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--n-product", type=int, default=128)
    parser.add_argument("--n-air", type=int, default=256)
    parser.add_argument("--radius-max-factor", type=float, default=8.0)
    args = parser.parse_args(argv)
    configure_stdout()

    checkpoint = Path(args.checkpoint)
    handoff = Path(args.handoff_json)
    case = _build_case(
        checkpoint, handoff_json=handoff, case_id=args.case_id, radius=args.radius
    )
    epsilon = float(case.field.epsilon)
    tau_end = float(case.tau_end)
    window_fraction = float(case.window_fraction)
    trained_tau_max = tau_end * window_fraction

    if args.tau is not None:
        tau = _parse_time_list(args.tau, "--tau")
        if np.any(tau < 0.0):
            raise SetupError("--tau must be non-negative (it starts at epsilon)")
        physical = epsilon + tau
    else:
        physical = _parse_time_list(args.t_phys, "--t-phys")
        tau = physical - epsilon
        if np.any(tau < -1.0e-15):
            earliest = float(physical[np.argmin(tau)])
            raise SetupError(
                f"--t-phys {earliest:.6g} s precedes epsilon = {epsilon:.6g} s; "
                "the trained domain starts at the startup field's own time"
            )
        tau = np.maximum(tau, 0.0)

    profiles = [
        _profile_blocks(
            case,
            tau=float(value),
            n_product=args.n_product,
            n_air=args.n_air,
            radius_max_factor=args.radius_max_factor,
        )
        for value in tau
    ]
    time_index = np.concatenate([
        np.full(profile["radius"].size, index, dtype=np.int64)
        for index, profile in enumerate(profiles)
    ])
    stacked = {
        "radius": np.concatenate([p["radius"] for p in profiles]),
        "time_index": time_index,
        "region": np.concatenate([p["region"] for p in profiles]),
        "density": np.concatenate([p["density"] for p in profiles]),
        "velocity": np.concatenate([p["velocity"] for p in profiles]),
        "specific_energy": np.concatenate([p["specific_energy"] for p in profiles]),
        "pressure": np.concatenate([p["pressure"] for p in profiles]),
        "overpressure": np.concatenate([p["overpressure"] for p in profiles]),
        "times_phys": physical,
        "tau": tau,
        "network_time": np.asarray([p["network_time"] for p in profiles]),
        "contact_radius": np.asarray([p["contact_radius"] for p in profiles]),
        "shock_radius": np.asarray([p["shock_radius"] for p in profiles]),
        "contact_speed": np.asarray([p["contact_speed"] for p in profiles]),
        "shock_speed": np.asarray([p["shock_speed"] for p in profiles]),
        "extrapolated": np.asarray([float(t) > trained_tau_max for t in tau]),
    }
    metadata = {
        "format": "spherical_blast_pinn.startup_prediction",
        "version": 1,
        "checkpoint": str(checkpoint),
        "case_id": case.case_id,
        "mode": case.mode,
        "trained_case_ids": list(case.trained_case_ids),
        "frozen_parameters": True,
        "charge_radius_m": float(case.scales.charge_radius),
        "epsilon_s": epsilon,
        "time_convention": case.scales.time_convention,
        "time_mapping": "tau = t_phys - epsilon; s = network_time(tau)",
        "tau_end_s": tau_end,
        "trained_window_fraction": window_fraction,
        "trained_tau_max_s": trained_tau_max,
        "ambient_pressure": float(case.model.undisturbed.pressure),
        "product_density_scale": float(case.scales.product_density),
        "air_density_scale": float(case.scales.air_density),
        "velocity_scale": float(case.scales.reference_velocity),
        "energy_scale": float(case.scales.reference_specific_energy),
        "extrapolated": [bool(t > trained_tau_max) for t in tau],
        "profiles": [
            {
                key: profile[key]
                for key in (
                    "tau", "physical_time", "network_time", "contact_radius",
                    "shock_radius", "contact_speed", "shock_speed",
                    "product_points", "air_points", "ambient_points",
                    "contact_left", "contact_right", "shock_post", "ambient",
                )
            }
            for profile in profiles
        ],
        "region_codes": {"0": "product", "1": "air", "2": "ambient"},
        "rules": (
            "product: 0 <= r < Rc with xi in [0, 1-eps]; air: Rc < r < Rs with "
            "eta in [eps, 1-eps]; ambient: r >= Rs. No cross-interface interpolation."
        ),
    }
    output = Path(args.output) if args.output else checkpoint.with_name(
        f"{checkpoint.stem}.predict.npz"
    )
    _write_npz(output, stacked, metadata)
    json_path = (
        Path(args.output_json)
        if args.output_json
        else output.with_suffix(".json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"case        : {case.case_id}  R0={case.scales.charge_radius:g} m  "
        f"mode={case.mode}"
    )
    if case.mode == "frozen_shared_parameters":
        print(
            f"frozen      : shared parameters trained on {case.trained_case_ids}; "
            "no parameter update, no supervision read"
        )
    print(f"time origin : epsilon={epsilon:.6e} s; {case.scales.time_convention}")
    for profile in profiles:
        flag = (
            "  [extrapolated]"
            if profile["tau"] > trained_tau_max + 1.0e-15
            else ""
        )
        print(
            f"  tau={profile['tau']:.6e} s  t_phys={profile['physical_time']:.6e} s  "
            f"s={profile['network_time']:.6e}  Rc={profile['contact_radius']:.6e} m  "
            f"Rs={profile['shock_radius']:.6e} m{flag}"
        )
    print(f"written     : {output}")
    print(f"              {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
