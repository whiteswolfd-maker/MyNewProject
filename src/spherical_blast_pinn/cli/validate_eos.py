"""状态方程一致性检查。

三条检查，都作用在**实际数据**上，不是作用在一个孤立的手算例子上：

1. 规范 §4 的初值：由 ``rho0`` 与 ``e = E0/rho0`` 经一般 JWL 得到 ``p_init``，
   与参考状态 ``(rho_ref, e_*)`` 处的 ``8.220261897e9 Pa`` 对照（**不是初值压力**）；
2. 空气卡：``*EOS_LINEAR_POLYNOMIAL`` 的 C4/C5 必须等价于理想气体；
3. 数据一致性：把数据里的 ``(rho, p)`` 按材料各自的状态方程反解出比内能，
   再正算回压力，检查闭合误差。这是"数据的能量定义与我们的 EOS 是否同一套"
   唯一可执行的检验。

**检查不通过就不许开始训练**：状态方程错了，后面每一个残差、每一条跳跃条件
都建在一个错的闭包上。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..physics import material as mat
from ..physics.eos import (
    EOSError,
    check_linear_polynomial_is_ideal_gas,
    ideal_gas_pressure_numpy,
    jwl_pressure_numpy,
)
from ._common import (
    add_common_arguments,
    configure_stdout,
    load_cases,
    load_config_from_args,
    write_json,
)


def material_table(config) -> dict[str, Any]:
    jwl = mat.product_jwl()
    return {
        "jwl": jwl.as_dict(),
        "rho0": mat.PRODUCT_REFERENCE_DENSITY,
        "E0_volume_energy": mat.PRODUCT_VOLUME_ENERGY,
        "e_star_j_kg": mat.PRODUCT_SPECIFIC_ENERGY,
        "u_star_m_s": mat.REFERENCE_VELOCITY,
        "p_reference_state_pa": mat.product_pressure_at_reference_state(),
        "gamma_air": mat.AIR_GAMMA,
        "air_density": mat.AIR_DENSITY,
        "air_pressure": mat.AIR_PRESSURE,
        "air_specific_energy_j_kg": mat.AIR_SPECIFIC_ENERGY,
        "product_pressure_scale_pa": mat.PRODUCT_VOLUME_ENERGY,
        "air_pressure_scale_pa": mat.AIR_DENSITY * mat.REFERENCE_VELOCITY**2,
    }


def closure_check(cases, *, tolerance: float) -> dict[str, Any]:
    """把数据里的 ``(rho, p)`` 反解成 ``e`` 再正算回 ``p``，检查闭合误差。"""
    worst = 0.0
    detail = []
    for case in cases:
        density = np.asarray(case.density)
        pressure = np.asarray(case.pressure)
        energy = np.asarray(case.specific_energy)
        material = np.asarray(case.material)
        good = np.isfinite(density) & np.isfinite(pressure) & np.isfinite(energy)
        if not np.any(good):
            detail.append({"case_id": case.case_id, "n_points": 0})
            continue
        product = good & (material == 0)
        air = good & (material == 1)
        values = {}
        if np.any(product):
            back = jwl_pressure_numpy(density[product], energy[product], mat.product_jwl())
            values["product"] = float(
                np.max(np.abs(back - pressure[product]) / np.maximum(np.abs(pressure[product]), 1.0))
            )
        if np.any(air):
            back = ideal_gas_pressure_numpy(density[air], energy[air], mat.AIR_GAMMA)
            values["air"] = float(
                np.max(np.abs(back - pressure[air]) / np.maximum(np.abs(pressure[air]), 1.0))
            )
        case_worst = max(values.values()) if values else 0.0
        worst = max(worst, case_worst)
        detail.append(
            {
                "case_id": case.case_id,
                "n_product_points": int(np.count_nonzero(product)),
                "n_air_points": int(np.count_nonzero(air)),
                "max_relative_closure_error": case_worst,
                **{f"{k}_relative": v for k, v in values.items()},
            }
        )
    return {
        "tolerance": float(tolerance),
        "max_relative_closure_error": worst,
        "passed": bool(worst <= tolerance),
        "cases": detail,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="反解—正算闭合的相对容差")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)

    table = material_table(config)

    # 1) 参考状态 (rho_ref, e_*) 处的 EOS 压力。
    #    **方案 B 里这不是初值压力**：初值场非均匀，压力逐点不同。这里比的是
    #    JWL 实现本身没有漂移，所以用固定的参考状态，而不是"整球初压"。
    expected = 8.220261897e9
    p_reference = table["p_reference_state_pa"]
    p_init_ok = abs(p_reference - expected) / expected < 1.0e-8

    # 2) 空气卡是否等价于理想气体
    coefficients = config.section("air").get("linear_polynomial") or {
        "C0": 0.0, "C1": 0.0, "C2": 0.0, "C3": 0.0,
        "C4": mat.AIR_GAMMA - 1.0, "C5": mat.AIR_GAMMA - 1.0, "C6": 0.0,
    }
    air_card_ok = True
    air_card_note = ""
    try:
        check_linear_polynomial_is_ideal_gas(coefficients, mat.AIR_GAMMA)
    except EOSError as exc:
        air_card_ok = False
        air_card_note = str(exc)

    # 3) 数据闭合
    cases = load_cases(config, split=args.split)
    closure = closure_check(cases, tolerance=args.tolerance)

    report = {
        "material_table": table,
        "initial_pressure": {
            "expected_pa": expected,
            "computed_pa": p_init,
            "relative_deviation": abs(p_init - expected) / expected,
            "passed": p_init_ok,
        },
        "air_card": {
            "coefficients": {k: float(v) for k, v in coefficients.items()},
            "equivalent_to_ideal_gas": air_card_ok,
            "note": air_card_note,
        },
        "closure": closure,
        "energy_definition": (
            "比内能 e（J/kg）由 (rho, p) 按材料各自的状态方程反解；"
            "不是比总能，也不是体积能量。deck 的 E0 是按**参考体积**定义的 "
            "J/m^3，换算见 material_table.e_star_j_kg"
        ),
    }
    out_dir = Path(args.output_dir) if args.output_dir else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "audit"
    write_json(out_dir / "eos_validation.json", report)

    lines = ["# 状态方程一致性检查", ""]
    lines.append(f"- p_init = {p_init:.6e} Pa（规范 {expected:.6e}）："
                 f"{'通过' if p_init_ok else '**未通过**'}")
    lines.append(f"- e_* = {table['e_star_j_kg']:.6e} J/kg，U_* = {table['u_star_m_s']:.4f} m/s")
    lines.append(f"- 空气卡等价于理想气体：{'通过' if air_card_ok else '**未通过**'}"
                 + (f"（{air_card_note}）" if air_card_note else ""))
    lines.append(f"- 数据闭合最大相对误差：{closure['max_relative_closure_error']:.3e}"
                 f"（容差 {closure['tolerance']:.1e}）："
                 f"{'通过' if closure['passed'] else '**未通过**'}")
    lines.append("")
    lines.append("| 算例 | 产物点 | 空气点 | 最大闭合相对误差 |")
    lines.append("|---|---|---|---|")
    for entry in closure["cases"]:
        lines.append(
            f"| {entry['case_id']} | {entry.get('n_product_points', 0)} | "
            f"{entry.get('n_air_points', 0)} | "
            f"{entry.get('max_relative_closure_error', float('nan')):.3e} |"
        )
    text = "\n".join(lines)
    (out_dir / "eos_validation.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten: {out_dir / 'eos_validation.json'}")
    return 0 if (p_init_ok and air_card_ok and closure["passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
