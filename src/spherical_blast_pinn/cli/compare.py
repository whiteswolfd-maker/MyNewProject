"""预测与验证仿真的对照（提示词 §14）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.compare --config configs/base.yaml \
        --prediction outputs/predict/predict_R0080mm.npz \
        --case tnt_80mm --output outputs/compare/80mm

先核对单位、球心、半径和时间零点，再比较。三件事**不做**：

* **不做时间对齐寻优。** 时间零点由物理定义给出（``t_PINN = t_phys - T_h``），
  不通过事后平移压力曲线把到达时间误差抹掉。两支曲线的到达时间差就是要报告的量
  本身，不是要消除的残差。
* **不把界面附近算成一个数。** 前沿位置误差与间断附近的场误差分开显示；保护带
  依据参考数据的分辨率**预先**设定，不因为结果不好而事后扩大。
* **不只在共同区域算分。** 两个模型前沿不一致时，材料不匹配的区域单独报告。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.extracted_case import load_extracted_case
from ..evaluation.waveform import compare_gauge, gauge_metrics
from ..physics import material as mat
from ._common import (
    SetupError,
    add_common_arguments,
    configure_stdout,
    load_config_from_args,
    write_json,
)


#: 预测网格上的时间插值容差。超出这个范围就**不插值**，返回 ``nan``——
#: 静默外推会把"这里没有预测"读成一个具体的数。
_TIME_TOLERANCE_FRACTION = 1.0e-9


def _time_bracket(times: np.ndarray, pinn_time: float) -> tuple[int, float] | None:
    """``(下限下标, 权重)``；超出网格覆盖时返回 ``None``。

    ``np.searchsorted`` 之后把下标钳到 ``[0, n-2]`` 会让权重跑出 ``[0, 1]``：
    查询时间在网格末端之外时，权重是大于 1 的数，插值结果是一个**外推值**，
    却和一个真正的插值长得一模一样。所以这里先判定覆盖，再决定权重。
    """
    if times.size < 2:
        return None
    span = float(times[-1] - times[0])
    tolerance = _TIME_TOLERANCE_FRACTION * max(abs(span), 1.0e-300)
    if pinn_time < times[0] - tolerance or pinn_time > times[-1] + tolerance:
        return None
    index = int(np.clip(np.searchsorted(times, pinn_time) - 1, 0, times.size - 2))
    t0, t1 = float(times[index]), float(times[index + 1])
    weight = (pinn_time - t0) / (t1 - t0) if t1 > t0 else 0.0
    return index, float(min(max(weight, 0.0), 1.0))


def _pressure_at(prediction: dict[str, Any], radius: float, physical_time: float) -> float:
    """在预测网格上取 ``(r, t_phys)`` 处的压力，同介质内双线性插值。

    预测网格的时间轴是 ``t_PINN``，所以这里做**一次**换算
    ``t_PINN = t_phys - T_h``。调用方给的是物理时间。
    """
    times = prediction["pinn_time"]
    pinn_time = float(physical_time) - float(prediction["start_time"])
    bracket = _time_bracket(times, pinn_time)
    if bracket is None:
        return float("nan")
    i, wt = bracket
    return _field_corner_average(
        prediction, "pressure", radius, i, i + 1, wt
    )


def _field_corner_average(
    prediction, name: str, radius: float, i: int, j_time: int, wt: float
) -> float:
    """在 ``(i, radius)`` 与 ``(j_time, radius)`` 两点上取同介质插值再按时间加权。

    半径方向不做双线性跨单元插值：预测的每一个网格点上都有自己的区域标签，
    跨接触面或激波线性插值会把一个间断抹成一个中间状态，而那个状态不属于任何
    一侧。四个采样点区域不一致时返回 ``nan``，并把这件事记在调用方的统计里。
    """
    radii = prediction["radius"]
    field = prediction[name]
    region = prediction["region"]
    if radius < float(radii[0]) or radius > float(radii[-1]):
        return float("nan")
    index = int(np.clip(np.searchsorted(radii, radius) - 1, 0, radii.size - 2))
    r0, r1 = float(radii[index]), float(radii[index + 1])
    span = r1 - r0
    if span <= 0.0:
        return float("nan")
    wr = (radius - r0) / span
    values = []
    for row in (i, j_time):
        left, right = float(field[row, index]), float(field[row, index + 1])
        labels = {int(region[row, index]), int(region[row, index + 1])}
        if len(labels) != 1 or not np.isfinite([left, right]).all():
            return float("nan")
        values.append((1.0 - wr) * left + wr * right)
    return float((1.0 - wt) * values[0] + wt * values[1])


def _reference_at(case, radius: float, physical_time: float) -> float:
    """参考压力；**只读物理时间在数据覆盖内的点**，域外返回 ``nan``。

    用最近帧是静默外推：请求到数据末端之后，它会给出最后一帧的值，读起来和
    一个真实采样一样。
    """
    times = case.times
    if physical_time < times[0] or physical_time > times[-1]:
        return float("nan")
    frame = case.frame_index(physical_time)
    value = float(case.sample_at(frame, np.array([radius]))["pressure"][0])
    return value if np.isfinite(value) else float("nan")


def compare_trajectories(prediction: dict[str, Any], case, *, ambient: float) -> dict[str, Any]:
    """``R_c``/``R_s`` 轨迹相对 RMS 误差，**按统一物理时间加权**。

    逐点等权会让输出帧密的那一段主导结论；这里把两条轨迹都插值到同一个均匀
    物理时间网格上再比较，帧间隔的不均匀就不再改变结论。
    """
    # ``physical_time`` 已经从 predict 里出来就是物理秒，``epsilon(R_0)`` 在那边
    # 加过一次。这里再加一次会让两条轨迹整体错开一个 ``epsilon``——R_0 = 50 mm
    # 时那是 1 µs，在 3000/4000 m/s 的前沿速度下就是 3/4 mm 的假误差。
    pred_phys = np.asarray(prediction["physical_time"], dtype=np.float64)
    grid = np.linspace(float(pred_phys[0]), float(pred_phys[-1]), 400)

    pred_contact = np.interp(grid, pred_phys, prediction["contact_radius"])
    pred_shock = np.interp(grid, pred_phys, prediction["shock_radius"])

    usable = case.supervisable_frames()
    frames = np.flatnonzero(usable)
    if frames.size < 4:
        return {"available": False, "reason": "参考数据没有足够的有效帧"}
    ref_times = case.times[frames]
    keep = (grid >= ref_times[0]) & (grid <= ref_times[-1])
    if np.count_nonzero(keep) < 4:
        return {"available": False, "reason": "预测与参考在物理时间上没有重叠区间"}

    ref_contact = np.interp(grid[keep], ref_times, case.contact_radius[frames])
    ref_shock = np.interp(grid[keep], ref_times, case.shock_radius[frames])
    scale = float(case.charge_radius)

    def rms(model, reference):
        return float(np.sqrt(np.mean(((model - reference) / scale) ** 2)))

    # 到达时间：两条轨迹第一次越过某个固定 ``r/R_0`` 的时刻之差，由轨迹本身
    # 给出，不做任何时间对齐寻优。
    arrival = {}
    for factor in (2.0, 5.0, 10.0):
        target = factor * scale
        arrival[f"r_over_R0_{factor:g}"] = {
            "model_s": _first_crossing(pred_phys, prediction["shock_radius"], target),
            "reference_s": _first_crossing(ref_times, case.shock_radius[frames], target),
        }
    for entry in arrival.values():
        model_time, reference_time = entry["model_s"], entry["reference_s"]
        if model_time is None or reference_time is None:
            entry["absolute_error_s"] = None
            entry["relative_error"] = None
        else:
            entry["absolute_error_s"] = model_time - reference_time
            entry["relative_error"] = (
                (model_time - reference_time) / reference_time if reference_time > 0.0 else None
            )

    return {
        "available": True,
        "normalisation": "charge radius R_0 (fixed, from the prediction's own scales)",
        "n_overlap_points": int(np.count_nonzero(keep)),
        "overlap_physical_time_s": [float(grid[keep][0]), float(grid[keep][-1])],
        "contact_relative_rms": rms(pred_contact[keep], ref_contact),
        "shock_relative_rms": rms(pred_shock[keep], ref_shock),
        "shock_arrival": arrival,
    }


def _first_crossing(times, radii, target: float) -> float | None:
    """``radii`` 第一次达到 ``target`` 的**线性插值**时刻；从未达到时 ``None``。"""
    times = np.asarray(times, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    finite = np.isfinite(radii) & np.isfinite(times)
    if not np.any(finite):
        return None
    times, radii = times[finite], radii[finite]
    above = np.flatnonzero(radii >= target)
    if above.size == 0 or above[0] == 0:
        return None
    index = int(above[0])
    r0, r1 = float(radii[index - 1]), float(radii[index])
    t0, t1 = float(times[index - 1]), float(times[index])
    if r1 == r0:
        return t0
    return t0 + (t1 - t0) * (target - r0) / (r1 - r0)


def _frozen_scales(prediction: dict[str, Any]) -> dict[str, float]:
    """每个场量的**固定物理尺度**，全部来自检查点，不来自被比较的数据。

    用参考数据的局部最大值当分母，分母会随"这次比了哪些点"变化，于是两次
    报告里的"相对误差"不是同一个量。这里用的是乘积里就有的尺度：
    ``rho_{*,k}``、``U_*``、``U_*^2``。
    """
    return {
        "density_product": float(prediction["scale_product_density"]),
        "density_air": float(prediction["scale_air_density"]),
        "velocity": float(prediction["scale_velocity"]),
        "specific_energy": float(prediction["scale_specific_energy"]),
    }


def compare_fields(prediction: dict[str, Any], case, *, buffer_cells: int = 3) -> dict[str, Any]:
    """两材料光滑区域内的场量相对 L2，**按球壳体积加权**，**分材料报告**。

    四条规则，每一条都对应一个会改变结论的失误：

    1. **保护带按参考数据的分辨率预先设置**（``buffer_cells`` 个单元），不因为
       结果不好而事后扩大。前沿附近的结果单独统计，不并入光滑区。
    2. **只比较材料身份一致的单元。** 参考说产物、预测说空气的点，既不属于
       产物误差也不属于空气误差，必须单独计数。把它们混进"共同区域"会让一个
       前沿错位的模型得到一个不错的光滑区误差。
    3. **未受扰动标签（2）不当作空气。** 数据里 ``material == 2`` 是主激波之外
       的静止空气；把它记为"空气"会让一个完全正确的环境态预测被判为材料错配。
    4. **分介质报告**，并保留错配统计与界面被排除的统计。
    """
    physical_time_grid = np.asarray(prediction["physical_time"], dtype=np.float64)
    radii_grid = np.asarray(prediction["radius"], dtype=np.float64)
    usable = case.supervisable_frames()
    frames = np.flatnonzero(usable)
    if frames.size == 0:
        return {"available": False, "reason": "参考数据没有有效帧"}

    cell = float(np.median(np.diff(case.radius)))
    scales = _frozen_scales(prediction)
    field_names = ("density", "velocity", "specific_energy")
    medium_scale = {
        "density": {"product": scales["density_product"], "air": scales["density_air"]},
        "velocity": {"product": scales["velocity"], "air": scales["velocity"]},
        "specific_energy": {"product": scales["specific_energy"], "air": scales["specific_energy"]},
    }

    totals = {
        f"{medium}_{name}": {"num": 0.0, "den": 0.0, "n": 0, "max_abs": 0.0}
        for medium in ("product", "air")
        for name in field_names
    }
    mismatch = {medium: 0 for medium in ("product", "air", "ambient")}
    excluded_inside_prediction_domain = 0
    compared_cells = 0

    for frame in frames:
        physical_time = float(case.times[frame])
        if physical_time < physical_time_grid[0] or physical_time > physical_time_grid[-1]:
            continue
        # 预测自己的分区标签，用于判定"材料身份是否一致"。
        region = _region_at(prediction, case.radius, physical_time)
        if not np.any(region >= 0):
            continue
        for label, medium in ((0, "product"), (1, "air"), (2, "ambient")):
            reference_here = case.material[frame] == label
            if not np.any(reference_here):
                continue
            # 错配：参考说这是 `label`，而预测的前沿把它划进了别的区域。
            mismatched = reference_here & (region >= 0) & (region != label)
            mismatch[medium] += int(np.count_nonzero(mismatched))
            compared_cells += int(np.count_nonzero(reference_here & (region >= 0)))
            if label == 2:
                # 环境态不参与场误差：那里是几千个常数单元，拿它评分对所有模型
                # 一样地"好看"，衡量不出任何东西（见 README 第八节）。
                continue

            valid = case.valid[frame] & reference_here & (region == label)
            contact = case.contact_radius[frame]
            shock = case.shock_radius[frame]
            near = (np.abs(case.radius - contact) <= buffer_cells * cell) | (
                np.abs(case.radius - shock) <= buffer_cells * cell
            )
            valid = valid & ~near
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                continue
            radii_sample = case.radius[indices]
            values = case.sample_at(int(frame), radii_sample)
            weights = 4.0 * np.pi * radii_sample**2 * cell

            for name in field_names:
                reference = values[name]
                model = _field_at(
                    prediction, name, radii_sample, physical_time
                )
                good = np.isfinite(reference) & np.isfinite(model)
                if not np.any(good):
                    continue
                w = weights[good]
                difference = model[good] - reference[good]
                scale = medium_scale[name][medium]
                key = f"{medium}_{name}"
                totals[key]["num"] += float(np.sum(w * (difference / scale) ** 2))
                totals[key]["den"] += float(np.sum(w))
                totals[key]["n"] += int(np.count_nonzero(good))
                totals[key]["max_abs"] = max(
                    totals[key]["max_abs"], float(np.max(np.abs(difference)))
                )
        # 预测域外的参考点单独统计，不当作误差。
        excluded_inside_prediction_domain += int(np.count_nonzero(region < 0))

    result: dict[str, Any] = {
        "available": True,
        "metric": (
            "sqrt( sum_shell w (model - reference)^2 / sum_shell w ) / scale, "
            "w = 4 pi r^2 dr (spherical shell volume), scale = fixed physical scale "
            "from the checkpoint"
        ),
        "buffer_cells": int(buffer_cells),
        "buffer_width_m": float(buffer_cells * cell),
        "buffer_basis": "reference data radial resolution, fixed before the comparison",
        "volume_weighted": True,
        "normalisation_scales": scales,
        "reference_label_semantics": "0=product, 1=air, 2=undisturbed ambient",
        "ambient_cells_excluded": True,
        "compared_cells": int(compared_cells),
        "region_mismatch_cells": mismatch,
        "region_mismatch_total": int(sum(mismatch.values())),
        "cells_outside_prediction_grid": int(excluded_inside_prediction_domain),
    }
    for key, entry in totals.items():
        if entry["den"] > 0.0:
            result[f"{key}_relative_l2"] = float(np.sqrt(entry["num"] / entry["den"]))
            result[f"{key}_n_points"] = entry["n"]
            result[f"{key}_max_abs_error"] = entry["max_abs"]
    for medium in ("product", "air"):
        keys = [f"{medium}_{name}_relative_l2" for name in field_names]
        present = [result[k] for k in keys if k in result]
        result[f"{medium}_relative_l2_over_fields"] = (
            float(np.mean(present)) if present else float("nan")
        )
    return result


def _field_at(prediction, name: str, radii, physical_time: float) -> np.ndarray:
    """在预测网格上按 ``(r, t_phys)`` 取一个场量。

    * 时间换算只做一次（``t_PINN = t_phys - T_h``）；网格覆盖之外不给值。
    * **不跨间断插值**：参与插值的两个半径网格点与两个时间切片必须落在同一个
      区域标签里，否则返回 ``nan``。跨接触面或激波做线性插值会造出一个两侧都不
      属于的中间状态，然后再拿它去和一个真实状态比。
    """
    times = prediction["pinn_time"]
    grid_radii = np.asarray(prediction["radius"], dtype=np.float64)
    field = prediction[name]
    region = prediction["region"]
    radii = np.asarray(radii, dtype=np.float64)
    out = np.full(radii.shape, np.nan, dtype=np.float64)

    bracket = _time_bracket(times, physical_time - float(prediction["start_time"]))
    if bracket is None:
        return out
    i, wt = bracket
    rows = (i, min(i + 1, times.size - 1))

    inside = (radii >= grid_radii[0]) & (radii <= grid_radii[-1])
    if not np.any(inside):
        return out
    positions = np.flatnonzero(inside)
    query = radii[positions]
    index = np.clip(np.searchsorted(grid_radii, query, side="right") - 1, 0, grid_radii.size - 2)
    left = grid_radii[index]
    right = grid_radii[index + 1]
    span = right - left
    weight = np.where(span > 0.0, (query - left) / np.where(span > 0.0, span, 1.0), 0.0)

    values = []
    for row in rows:
        a = field[row, index]
        b = field[row, index + 1]
        same_region = region[row, index] == region[row, index + 1]
        same_region &= region[row, index] >= 0
        blended = a * (1.0 - weight) + b * weight
        values.append(np.where(same_region & np.isfinite(a) & np.isfinite(b), blended, np.nan))
    combined = values[0] * (1.0 - wt) + values[-1] * wt if len(values) == 2 else values[0]
    out[positions] = combined
    return out


def _region_at(prediction, radii, physical_time: float) -> np.ndarray:
    """预测在 ``(r, t_phys)`` 处的区域标签：0 产物、1 空气、2 环境、-1 网格外。"""
    times = prediction["pinn_time"]
    radii = np.asarray(radii)
    bracket = _time_bracket(times, physical_time - float(prediction["start_time"]))
    if bracket is None:
        return np.full(radii.shape, -1, dtype=np.int64)
    i, _ = bracket
    contact = float(prediction["contact_radius"][i])
    shock = float(prediction["shock_radius"][i])
    return np.where(radii < contact, 0, np.where(radii < shock, 1, 2)).astype(np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--prediction", required=True, help="predict 命令写出的 npz")
    parser.add_argument("--case", required=True, help="验证算例的 case_id")
    parser.add_argument("--gauge-factors", type=float, nargs="*", default=[2.0, 5.0, 10.0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)

    section = config.section("data")
    specification = next(
        (s for s in (section.get("cases") or []) if str(s.get("case_id")) == args.case), None
    )
    if specification is None:
        raise SetupError(f"case {args.case!r} is not declared under data.cases")
    case = load_extracted_case(Path(str(specification["directory"])), case_id=args.case)

    # 同上：显式取出并释放句柄。
    with np.load(args.prediction, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}
    prediction = {key: data[key] for key in data.files}

    radius = float(prediction["charge_radius"])
    if abs(radius - case.charge_radius) > 1.0e-9:
        raise SetupError(
            f"预测的装药半径 {radius} m 与参考算例的 {case.charge_radius} m 不一致；"
            "半径、单位和时间零点必须先核对再比较"
        )

    ambient = float(prediction.get("ambient_pressure", mat.AIR_PRESSURE))
    report: dict[str, Any] = {
        "case_id": args.case,
        "charge_radius_m": radius,
        "checks": {
            "units": "两侧均为 SI",
            "charge_radius_match": True,
            "time_origin_s": float(prediction["start_time"]),
            "reference_time_origin_s": float(case.start_time),
            "time_origin_match": abs(
                float(prediction["start_time"]) - float(case.start_time)
            )
            < 1.0e-15,
            "time_origin_rule": (
                "t_PINN = t_phys - T_h; the prediction's physical_time already "
                "carries the shift, so nothing is shifted again here"
            ),
            "time_alignment_optimised": False,
            "scaled_time_coverage_s_max": float(
                prediction.get("scaled_time_coverage", float("nan"))
            ),
        },
        "trajectories": compare_trajectories(prediction, case, ambient=ambient),
        "fields": compare_fields(prediction, case),
    }

    # -- 测点压力时程 -----------------------------------------------------
    gauges = []
    for factor in args.gauge_factors:
        gauge_radius = float(factor) * radius
        if gauge_radius >= float(prediction["radius"][-1]):
            continue
        physical = np.asarray(prediction["physical_time"], dtype=np.float64)
        model_pressure = np.array(
            [_pressure_at(prediction, gauge_radius, float(t)) for t in physical]
        )
        reference_pressure = np.array(
            [_reference_at(case, gauge_radius, float(t)) for t in physical]
        )
        peak = float(np.nanmax(model_pressure - ambient)) if model_pressure.size else 1.0
        comparison = compare_gauge(
            times=physical,
            model_pressure=model_pressure,
            reference_pressure=reference_pressure,
            ambient=ambient,
            pressure_scale=max(abs(peak), ambient),
            time_scale=max(float(physical[-1] - physical[0]), 1.0e-12),
            impulse_scale=max(abs(peak), ambient) * max(float(physical[-1] - physical[0]), 1.0e-12),
            radius=gauge_radius,
        )
        gauges.append(comparison.as_dict())
    report["gauges"] = gauges

    out_dir = Path(args.output) if args.output else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "compare"
    write_json(out_dir / f"compare_{args.case}.json", report)
    print(_render(report))
    print(f"\nwritten: {out_dir / f'compare_{args.case}.json'}")
    return 0


def _render(report: dict[str, Any]) -> str:
    lines = ["# 预测—验证对照", "", f"算例：{report['case_id']}", ""]
    checks = report["checks"]
    lines.append("## 单位、半径与时间零点")
    lines.append("")
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    traj = report["trajectories"]
    lines.append("## 前沿轨迹")
    lines.append("")
    if traj.get("available"):
        lines.append(f"- 归一化：{traj['normalisation']}")
        lines.append(f"- R_c 相对 RMS：{traj['contact_relative_rms']:.4e}")
        lines.append(f"- R_s 相对 RMS：{traj['shock_relative_rms']:.4e}")
        lines.append(f"- 重叠物理时间：{traj['overlap_physical_time_s']}")
        lines.append("")
        lines.append("| 测点 r/R_0 | 主激波到达（模型） | 主激波到达（参考） | 相对误差 |")
        lines.append("|---|---:|---:|---:|")
        for key, entry in traj.get("shock_arrival", {}).items():
            model_time = entry["model_s"]
            reference_time = entry["reference_s"]
            relative = entry["relative_error"]
            lines.append(
                f"| {key.replace('r_over_R0_', '')} | "
                f"{'—' if model_time is None else f'{model_time:.4e}'} | "
                f"{'—' if reference_time is None else f'{reference_time:.4e}'} | "
                f"{'—' if relative is None else f'{relative:.3e}'} |"
            )
    else:
        lines.append(f"- 不可评价：{traj.get('reason')}")
    fields = report["fields"]
    lines.append("")
    lines.append("## 场量（球壳体积加权相对 L2，分介质）")
    lines.append("")
    if fields.get("available"):
        lines.append(f"- 保护带：{fields['buffer_cells']} 个单元"
                     f"（{fields['buffer_width_m']:.4g} m，{fields['buffer_basis']}）")
        lines.append(f"- 归一化：{fields['metric']}")
        lines.append(f"- 固定尺度：{fields['normalisation_scales']}")
        lines.append("")
        lines.append("| 介质 | 场量 | 相对 L2 | 点数 | 最大绝对误差 |")
        lines.append("|---|---|---:|---:|---:|")
        for medium in ("product", "air"):
            for name in ("density", "velocity", "specific_energy"):
                key = f"{medium}_{name}_relative_l2"
                if key not in fields:
                    lines.append(f"| {medium} | {name} | 不可评价（无共同区域） | 0 | — |")
                    continue
                lines.append(
                    f"| {medium} | {name} | {fields[key]:.4e} | "
                    f"{fields[f'{medium}_{name}_n_points']} | "
                    f"{fields[f'{medium}_{name}_max_abs_error']:.4e} |"
                )
        lines.append("")
        lines.append(f"- 材料不匹配单元数：{fields['region_mismatch_cells']}"
                     "（分介质单独报告，不并入共同区域）")
        lines.append(f"- 环境态单元数（不参与场误差）：{fields['ambient_cells_excluded']} 标记启用")
    else:
        lines.append(f"- 不可评价：{fields.get('reason')}")
    lines.append("")
    lines.append("## 测点压力时程")
    lines.append("")
    lines.append("| r/R_0 | 峰值超压误差 | 到达时间误差 | 正压时长误差 | 正相冲量误差 | 正压完整 |")
    lines.append("|---|---|---|---|---|---|")
    for gauge in report["gauges"]:
        errors = gauge["relative_errors"]
        def fmt(key):
            value = errors.get(key)
            return "—" if value is None else f"{value:.3e}"
        lines.append(
            f"| {gauge['radius_m'] / report['charge_radius_m']:.3g} | "
            f"{fmt('peak_overpressure_relative')} | {fmt('arrival_time_relative')} | "
            f"{fmt('positive_duration_relative')} | {fmt('positive_impulse_relative')} | "
            f"{'是' if gauge['reference'] and gauge['reference']['positive_phase_complete'] else '否'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
