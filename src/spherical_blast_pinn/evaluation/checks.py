"""固定的独立物理检查集与逐方程验收指标（提示词 §12、§13、§16）。

"每 500 步保存一个检查点"不等于"每 500 步完成一次物理验收"。训练循环里的损失
是在**当步新采的**样本上算的，它随采样一起抖动；用一个抖动的数去判断"方程是否
已经满足"，看到的其实是采样噪声。所以这里另建一套**固定**的检查点：

* 一次采样、全程复用，不参与任何梯度；
* 由独立的随机种子生成，与训练采样器无关；
* 定义（帧号、点数、种子、半径）随检查点和运行清单一同保存，事后可复现。

检查集还承担另一件事：**选模型**。§13 要求在查看最终验证结果之前固定选择标准，
并且优先满足有效状态与物理条件，再比较开发集预测误差。``CheckReport.score``
就是那个固定的偏序，见 :data:`SELECTION_CRITERIA`。

数据项用的是**开发帧**（§11 的 10% 帧组留出），所以"开发集误差"真的是一组训练
梯度没有见过的帧，而不是同一批点的另一次采样。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..physics import material as mat
from ..physics.moving_frame import AIR, PRODUCT
from ..physics.similitude import PlanarStartupScales
from ..physics.two_medium_startup_planar import PlanarStartupField
from ..losses.objective import contact_pressure_scale
from ..sampling.batch import SampleBatch, restrict_to_time_window
from ..sampling.sampler import RadiusSampler, SamplingConfig

#: 选模型标准的逐字定义。写进检查点，避免事后换一套说法。
SELECTION_CRITERIA = (
    "lexicographic, fixed before the final test set is seen: "
    "(1) non-physical / inadmissible state count, (2) squared hinge of the physics "
    "and interface residuals against the section 16 tolerances, (3) development-set "
    "prediction error, (4) training loss. Lower is better at every level."
)


class CheckError(RuntimeError):
    """检查集无法建立或无法评价。"""


@dataclass
class CheckSet:
    """一套固定的检查点。"""

    label: str
    batches: list[SampleBatch]
    definition: dict[str, Any]
    #: 与训练损失同源的 ``Pc_*``（启动场的 ``p_*``）。两个不同的分母会让检查
    #: 指标与训练损失不可比，所以在建立检查集时就从场里取定、随检查集一起走。
    contact_pressure: float = float("nan")

    @property
    def case_ids(self) -> list[str]:
        return [batch.case_id for batch in self.batches]

    @property
    def radii(self) -> list[float]:
        return [float(batch.charge_radius) for batch in self.batches]

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, **self.definition}

    def windowed(self, fraction: float) -> "CheckSet":
        """把检查集裁到尺度化时间窗口 ``[0, fraction * s_max]``（§14.2）。

        检查点**不重采**：裁的是同一批点，只是把窗口外的点排除在外。这样阶段判据
        与阶段结束后的验收用的是同一套仪器，区别只在评价区间——而这是不可避免的，
        因为阶段 1 的窗口里本来就没有窗口外的训练数据，拿窗口外的点判它不合格
        等于用没训过的区域给它打分。
        """
        value = float(fraction)
        if not (0.0 < value <= 1.0):
            raise CheckError(f"time window fraction must lie in (0, 1], got {fraction!r}")
        # ``pinn_time_max`` 是 tau（秒），窗口判据是 s（无量纲）：先经批次自己
        # 的尺度换算，不把两个不同量纲的数直接相乘后当成 s 上界。
        bounds = {
            batch.case_id: float(
                batch.scales.network_time(value * float(batch.pinn_time_max))
            )
            for batch in self.batches
        }
        batches = [
            restrict_to_time_window(batch, bounds[batch.case_id])
            for batch in self.batches
        ]
        definition = dict(self.definition)
        definition["window"] = {
            "fraction": value,
            "s_max": bounds,
            "note": "same fixed points, restricted to the stage window; never re-sampled",
        }
        return CheckSet(
            label=f"{self.label}@w{value:g}",
            batches=batches,
            definition=definition,
            contact_pressure=self.contact_pressure,
        )


def build_check_set(
    *,
    field: PlanarStartupField,
    scales: PlanarStartupScales,
    tau_end: float,
    config: SamplingConfig | None = None,
    seed: int = 0,
    dtype=None,
    device=None,
    r_min_fraction: float = 1.0e-2,
    case_id: str | None = None,
    label: str = "independent-check",
) -> CheckSet:
    """建立一个固定检查集：一次采样，之后一直用这一套。

    采样器**不参与梯度**，用独立种子；它从新启动场采样，没有数据监督。
    检查点的 ``Pc_*`` 在建立时从同一个启动场取定，因此训练与检查共用一条
    归一化尺度。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    sampler = RadiusSampler(
        field=field,
        scales=scales,
        tau_end=tau_end,
        case_id=case_id,
        config=config,
        seed=int(seed),
        dtype=dtype,
        device=device,
        r_min_fraction=r_min_fraction,
    )
    batch = sampler.batch()
    contact_pressure = contact_pressure_scale(field)
    definition = {
        "seed": int(seed),
        "purpose": "fixed physics verification and model selection; never in a gradient",
        "sampler": {
            "case_id": batch.case_id,
            "tau_end_s": float(tau_end),
            "points": batch.as_dict(),
        },
        "contact_pressure_pa": float(contact_pressure),
        "selection_criteria": SELECTION_CRITERIA,
    }
    return CheckSet(
        label=label,
        batches=[batch],
        definition=definition,
        contact_pressure=contact_pressure,
    )


@dataclass
class CheckReport:
    """一次独立检查的结果。"""

    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_radius: dict[str, dict[str, float]] = field(default_factory=dict)
    conservation: dict[str, Any] = field(default_factory=dict)
    selection_criteria: str = SELECTION_CRITERIA

    # -- 选模型 -----------------------------------------------------------
    @property
    def score(self) -> tuple[float, float, float, float]:
        """固定的偏序，越低越好。前两级是物理条件，后两级才是误差。"""
        return (
            float(self.metrics.get("nonphysical_violation_count", 0.0)),
            float(self.metrics.get("physics_violation_score", float("inf"))),
            float(self.metrics.get("development_prediction_error", float("inf"))),
            float(self.metrics.get("total_loss", 0.0)),
        )

    @property
    def admissible(self) -> bool:
        """是否满足 §16 的"无 NaN/Inf、负密度、负比内能或无效 EOS 状态"。"""
        return float(self.metrics.get("nonphysical_violation_count", 1.0)) == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "metrics": self.metrics,
            "per_radius": self.per_radius,
            "conservation": self.conservation,
            "score": list(self.score),
            "admissible": self.admissible,
            "selection_criteria": self.selection_criteria,
        }


# --------------------------------------------------------------------------
# 指标工具
# --------------------------------------------------------------------------


def _tensor_stats(values) -> dict[str, float]:
    """一组残差的 RMS / P95 / 最大值，以及最大值的位置（样本下标）。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return {"rms": float("nan"), "p95": float("nan"), "max": float("nan"),
                "argmax": -1, "n": 0}
    absolute = flat.abs()
    return {
        "rms": float(torch.sqrt((flat**2).mean())),
        "p95": float(absolute.quantile(0.95)),
        "max": float(absolute.max()),
        "argmax": int(absolute.argmax()),
        "n": int(flat.numel()),
    }


def _validity_rates(batch: SampleBatch, model) -> dict[str, float]:
    """非物理状态的比例。

    §16 要求"无 NaN/Inf、负密度、负比内能或无效 EOS 状态"。密度与比内能由输出
    构造保证为正，所以真正需要检查的是**压力的符号**与**声速平方的符号**：

    * 产物：``c^2`` 用**未裁剪**的公式，先把负值裁成 0 再判断等于没判断；
    * 空气：``c^2 = gamma p / rho``，与压力的符号一致。

    每一项都单独回报，并且**用原始值判断**，不用任何数值保护之后的值。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    out: dict[str, float] = {}
    if batch.s_product is not None and batch.s_product.numel():
        fields = model.product_fields(batch.s_product.detach(), batch.xi_product.detach())
        out["product_nonfinite_rate"] = float(
            (~torch.isfinite(fields.pressure)).to(torch.float64).mean()
        )
        out["product_negative_pressure_rate"] = float(
            (fields.pressure < 0.0).to(torch.float64).mean()
        )
        c2 = model.product_net.sound_speed_squared(fields)
        out["product_invalid_sound_speed_rate"] = float(
            (c2 < 0.0).to(torch.float64).mean()
        )
    if batch.s_air is not None and batch.s_air.numel():
        fields = model.air_fields(batch.s_air.detach(), batch.eta_air.detach())
        out["air_nonfinite_rate"] = float(
            (~torch.isfinite(fields.pressure)).to(torch.float64).mean()
        )
        out["air_negative_pressure_rate"] = float(
            (fields.pressure < 0.0).to(torch.float64).mean()
        )
    if batch.xi_ic_product is not None and batch.xi_ic_product.numel():
        fields = model.product_fields(
            torch.zeros_like(batch.xi_ic_product), batch.xi_ic_product
        )
        out["ic_product_negative_pressure_rate"] = float(
            (fields.pressure < 0.0).to(torch.float64).mean()
        )
    return out


def _prefix(stats: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in stats.items()}


def _guarded_mean(values: Sequence[float]) -> float:
    """空集合或全 ``nan`` 时返回 ``nan``，不返回 0。

    返回 0 会让"这一项没能评价"读起来像"这一项误差为零"。
    """
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _hinge_excess(value: float, tolerance: float) -> float:
    """``max(value/tolerance - 1, 0)^2``；容忍度为 0 时不计入。"""
    if not math.isfinite(value):
        return 1.0e6
    if tolerance <= 0.0:
        return 0.0
    return max(value / tolerance - 1.0, 0.0) ** 2


# --------------------------------------------------------------------------
# 评价
# --------------------------------------------------------------------------


def run_checks(
    model,
    check_set: CheckSet,
    *,
    pde_weight: float = 1.0,
    total_loss: float | None = None,
    tolerances: Mapping[str, float] | None = None,
    with_conservation: bool = True,
    steps: int = 0,
) -> CheckReport:
    """在固定检查集上评价物理指标。

    全部在 ``torch.no_grad()`` 之外完成的一阶导数除外——PDE 残差需要 ``s`` 的
    一阶导，所以这里仍然建图，但**不**调用 ``backward``，也不写进优化器。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    from ..losses.terms import (
        contact_loss,
        data_loss,
        front_floor_monitor,
        initial_condition_loss,
        medium_pde_residual,
        shock_loss,
    )

    limits = dict(tolerances or {})
    pde_rms_limit = float(limits.get("pde_rms", 1.0e-2))
    pde_p95_limit = float(limits.get("pde_absolute_p95", 5.0e-2))
    field_l2_limit = float(limits.get("field_relative_l2", 0.10))

    metrics: dict[str, float] = {}
    per_radius: dict[str, dict[str, float]] = {}
    physics_excess = 0.0
    nonphysical = 0.0

    if not (math.isfinite(check_set.contact_pressure) and check_set.contact_pressure > 0.0):
        raise CheckError(
            "the check set carries no finite positive contact-pressure scale; the "
            "startup field's p_star must be captured when the set is built"
        )
    contact_pressure = float(check_set.contact_pressure)

    for batch in check_set.batches:
        entry: dict[str, float] = {}

        # -- PDE：逐材料、逐方程 ------------------------------------------
        for medium in (PRODUCT, AIR):
            residual = medium_pde_residual(batch, model, medium)
            if residual is None:
                entry[f"pde_{medium}_points"] = 0.0
                continue
            mass, momentum, energy = residual.scaled()
            entry[f"pde_{medium}_points"] = float(residual.n_points)
            for name, values in (("mass", mass), ("momentum", momentum), ("energy", energy)):
                stats = _tensor_stats(values)
                entry.update(_prefix(stats, f"pde_{medium}_{name}"))
                physics_excess += _hinge_excess(stats["rms"], pde_rms_limit)
                physics_excess += _hinge_excess(stats["p95"], pde_p95_limit)

        # -- 接触面与主激波 -----------------------------------------------
        contact = contact_loss(batch, model, contact_pressure=contact_pressure)
        entry.update({f"contact_{k}": float(v) for k, v in contact.parts.items()})
        entry.update({f"contact_{k}": float(v) for k, v in contact.monitors.items()
                      if isinstance(v, (int, float))})
        shock = shock_loss(batch, model)
        entry.update({f"shock_{k}": float(v) for k, v in shock.parts.items()})
        entry.update({f"shock_{k}": float(v) for k, v in shock.monitors.items()
                      if isinstance(v, (int, float))})
        for key in ("contact_pressure", "contact_speed_product", "contact_speed_air"):
            if key in contact.parts:
                physics_excess += _hinge_excess(math.sqrt(float(contact.parts[key])), 1.0e-2)
        for key in ("shock_mass", "shock_momentum", "shock_energy"):
            if key in shock.parts:
                physics_excess += _hinge_excess(math.sqrt(float(shock.parts[key])), 1.0e-2)

        # -- 初值 ----------------------------------------------------------
        initial = initial_condition_loss(batch, model)
        entry.update({f"ic_{k}": float(v) for k, v in initial.parts.items()})
        entry.update({f"ic_{k}": float(v) for k, v in initial.monitors.items()
                      if isinstance(v, (int, float))})

        # -- 数据（开发帧） -------------------------------------------------
        data = data_loss(batch, model)
        entry.update({f"data_{k}": float(v) for k, v in data.parts.items()})
        entry.update({f"data_{k}": float(v) for k, v in data.monitors.items()
                      if isinstance(v, (int, float))})
        development_error = _guarded_mean(
            [math.sqrt(float(v)) for v in data.parts.values()]
        )
        entry["data_development_rms"] = development_error
        if math.isfinite(development_error):
            physics_excess += _hinge_excess(development_error, field_l2_limit)

        # -- 有效状态 ------------------------------------------------------
        entry.update(front_floor_monitor(batch, model))
        entry.update(_validity_rates(batch, model))
        for key, value in entry.items():
            if key.endswith("_rate") and value > 0.0:
                nonphysical += float(value)

        per_radius[batch.case_id] = entry
        metrics.update({f"{batch.case_id}.{key}": value for key, value in entry.items()})

    per_radius_values = list(per_radius.values())
    metrics["pde_rms_max"] = max(
        (value for entry in per_radius_values for key, value in entry.items()
         if key.startswith("pde_") and key.endswith("_rms")),
        default=float("nan"),
    )
    development_values = [
        float(value)
        for entry in per_radius_values
        if math.isfinite(
            value := float(entry.get("data_development_rms", float("nan")))
        )
    ]
    # 没有数据监督时这一项**没有定义**，不是 0：把"没有观测"写成 0 会让选模型
    # 偏序在一个不存在的指标上比较。
    metrics["development_prediction_error"] = (
        float(np.mean(development_values)) if development_values else float("nan")
    )
    metrics["physics_violation_score"] = float(physics_excess)
    metrics["nonphysical_violation_count"] = float(nonphysical)
    metrics["total_loss"] = float(total_loss if total_loss is not None else float("nan"))
    metrics["pde_weight"] = float(pde_weight)

    report = CheckReport(step=int(steps), metrics=metrics, per_radius=per_radius)
    if with_conservation:
        report.conservation = conservation_diagnostics(model, check_set)
        for key in (
            "product_mass_max_relative_drift",
            "box_energy_balance_relative_error",
        ):
            value = report.conservation.get(key)
            if isinstance(value, float) and math.isfinite(value):
                report.metrics[key] = value
                physics_excess += _hinge_excess(
                    value,
                    float(limits.get(
                        "product_mass_drift" if "mass" in key else "box_energy_balance",
                        0.01 if "mass" in key else 0.02,
                    )),
                )
        report.metrics["physics_violation_score"] = float(physics_excess)
    return report


# --------------------------------------------------------------------------
# 守恒诊断（§16）
# --------------------------------------------------------------------------


def conservation_diagnostics(
    model,
    check_set: CheckSet,
    *,
    n_radial: int = 400,
    box_radius_factor: float = 8.0,
    n_times: int = 12,
) -> dict[str, Any]:
    """产物质量漂移与控制体总能量平衡。

    固定控制体：外半径 ``R_box = box_radius_factor * R_0``，全程不变。能量平衡用

    .. math::

        E_{\\rm box}(t) - E_{\\rm box}(t_0)
          + \\int_{t_0}^{t} 4\\pi R_{\\rm box}^2\\, u(\\rho E + p)\\big|_{R_{\\rm box}}\\,d\\tau = 0

    误差用**物理零点**的初始产物能量 ``(4\\pi/3) R_0^3 E_0`` 归一化——它是一个
    冻结常数，不是 PINN 启动时刻的能量，两者不是一回事（§16 明确要求分清）。

    这两项**只是诊断**，不进梯度。提示词禁止把全局积分守恒做成训练项。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    from ..evaluation.metrics import trapezoid

    rows: list[dict[str, float]] = []
    mass_drift = 0.0
    for batch in check_set.batches:
        radius_0 = float(batch.charge_radius)
        box = float(box_radius_factor) * radius_0
        network_time_max = float(batch.scales.network_time(batch.pinn_time_max))
        radii = torch.linspace(
            radius_0 * 1.0e-4, box, int(n_radial),
            dtype=model.product_net.dtype_, device=next(model.parameters()).device,
        )
        shell = 4.0 * math.pi * radii**2
        # 固定控制体上的梯形权重（非均匀网格也要）
        edges = torch.cat([radii[:1], 0.5 * (radii[1:] + radii[:-1]), radii[-1:]])
        widths = edges[1:] - edges[:-1]
        weights = shell * widths

        s_values = np.linspace(0.0, network_time_max, int(n_times))
        mass_series, energy_series, flux_series, times = [], [], [], []
        for s_value in s_values:
            s = torch.as_tensor([[float(s_value)]], dtype=radii.dtype, device=radii.device)
            # ``piecewise_profile`` 返回 ``(1, n_r)``；压成一维，否则 ``[-1]`` 取到
            # 的是整行而不是最外侧那一点。
            profile = model.piecewise_profile(radii, s, charge_radius=radius_0)
            density = profile["density"].reshape(-1)
            velocity = profile["velocity"].reshape(-1)
            energy = profile["specific_energy"].reshape(-1)
            pressure = profile["pressure"].reshape(-1)
            contact = float(profile["contact_radius"].reshape(-1)[0])
            product = radii < contact

            internal = density * energy
            kinetic = 0.5 * density * velocity**2
            total = internal + kinetic
            finite = torch.isfinite(density) & torch.isfinite(energy)
            mass_series.append(
                float((torch.where(product & finite, density, torch.zeros_like(density))
                       * weights).sum())
            )
            energy_series.append(
                float((torch.where(finite, total, torch.zeros_like(total)) * weights).sum())
            )
            # 外边界通量：固定控制体上的 4 pi R_box^2 u (rho E + p)。
            flux_series.append(
                float(
                    (
                        velocity[-1]
                        * (density[-1] * (energy[-1] + 0.5 * velocity[-1] ** 2) + pressure[-1])
                    ).detach()
                )
                * 4.0
                * math.pi
                * box**2
            )
            times.append(float(s_value))

        if not times:
            continue
        mass_array = np.asarray(mass_series)
        energy_array = np.asarray(energy_series)
        flux_array = np.asarray(flux_series)
        times_array = np.asarray(times)

        # t_PINN = s * hat T_* * R_0 / U_*
        physical = times_array * float(batch.scales.time_scale) * radius_0 / mat.REFERENCE_VELOCITY
        initial_product_energy = (4.0 / 3.0) * math.pi * radius_0**3 * mat.PRODUCT_VOLUME_ENERGY

        drift = float(np.max(np.abs(mass_array / max(mass_array[0], 1.0e-300) - 1.0)))
        flux_integral = np.concatenate([[0.0], np.cumsum(
            [
                trapezoid(flux_array[i:i + 2], physical[i:i + 2])
                for i in range(physical.size - 1)
            ]
        )])
        balance = energy_array - energy_array[0] + flux_integral
        balance_error = float(np.max(np.abs(balance)) / initial_product_energy)
        mass_drift = max(mass_drift, drift)
        rows.append({
            "case_id": batch.case_id,
            "charge_radius_m": radius_0,
            "box_radius_m": box,
            "n_radial_points": int(n_radial),
            "n_times": int(n_times),
            "product_mass_first_kg": float(mass_array[0]),
            "product_mass_max_relative_drift": drift,
            "box_energy_balance_relative_error": balance_error,
            "reference_energy_j": float(initial_product_energy),
            "reference_energy_definition": "(4 pi / 3) R_0^3 E_0 at the physical time origin",
        })

    if not rows:
        return {"available": False, "reason": "检查集里没有算例"}
    return {
        "available": True,
        "per_case": rows,
        "product_mass_max_relative_drift": float(mass_drift),
        "box_energy_balance_relative_error": float(
            max(row["box_energy_balance_relative_error"] for row in rows)
        ),
        "note": (
            "diagnostics only, never a training term; the box is fixed and the outer "
            "flux is integrated rather than assumed zero"
        ),
    }


__all__ = [
    "CheckError",
    "CheckReport",
    "CheckSet",
    "SELECTION_CRITERIA",
    "build_check_set",
    "conservation_diagnostics",
    "run_checks",
]
