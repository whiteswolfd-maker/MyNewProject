"""已提取算例的读取适配器与材料分区识别。

数据来源
--------

三个半径都来自真实的 LS-DYNA 求解，以「沿半径线的时空剖面」形式保存：

* ``raw_line.npz`` —— 键 ``t``、``x``、``rho``、``u``、``P``，形状
  ``t(N_t,)``、``x(N_x,)``、其余 ``(N_t, N_x)``，SI 单位；
* ``metadata.json`` —— ``R_0``、``x_end``、``t_end``、网格数等。

字段映射**不写死**：:class:`FieldMapping` 把逻辑字段名映射到 npz 的键，
默认值只是本数据集的当前约定。换一份导出不需要改代码。

**没有输出比内能。** 算例卡片不写 ``specific_internal_energy``，所以它由
``(rho, p)`` 按材料各自的状态方程反解：

* 空气： ``e = p / [(gamma - 1) rho]``；
* 产物： ``e = (p - p_cold(V)) / (omega rho)``，``V = rho_0/rho``。

反解出的非物理值保持 ``nan``，**不做截断**——用一个"保护"后的值冒充有效状态，
会让下游的 EOS 一致性检查全部失去意义。

材料分区
--------

提示词 §11 要求"监督样本的材料归属与有效掩码来自数据预处理"，并且"不根据
当前预测前沿随意删除难拟合样本"。所以分区在这里一次性确定，并随算例一起
保存，训练时只读不改。

本适配器用**密度最陡下降**定位接触面：接触面两侧压力与速度连续、密度间断，
因此它是内侧剖面上密度下降最快的位置。主激波用压力超出环境态的比例定位。

这两条都是**启发式**判据，不是从算例的体积分数直接读出来的。它们的不确定
性由 :meth:`ExtractedCase.quality_report` 给出，验收表里也据此标注哪些帧
"参考数据不足以判断"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..physics import material as mat
from ..physics.eos import jwl_pressure_numpy
from ..physics.similitude import handover_time

#: 逻辑字段 -> ``raw_line.npz`` 的键。
DEFAULT_FIELD_MAP: dict[str, str] = {
    "time": "t",
    "radius": "x",
    "density": "rho",
    "velocity": "u",
    "pressure": "P",
}

#: 反解比内能所需的最少字段。
REQUIRED_FIELDS = ("time", "radius", "density", "velocity", "pressure")


class DataError(RuntimeError):
    """数据不可用或字段缺失。"""


def specific_energy_from_pressure(
    density: np.ndarray,
    pressure: np.ndarray,
    material: np.ndarray,
    *,
    jwl=None,
    gamma_air: float = mat.AIR_GAMMA,
) -> np.ndarray:
    """由 ``(rho, p)`` 按材料各自的状态方程反解比内能。

    产物走一般 JWL 的逆：``p = p_cold(V) + omega rho e`` 解出 ``e``；空气走理想气体。
    两条都是**除法**，不是求根：有限输入、``rho > 0``、``omega > 0`` 时结果必然是
    有限实数。所以这里没有"解不出来"的分支。

    ``nan`` 只来自两个地方，且都不是数学上的无解：

    * 输入本身无效（``omega <= 0``、密度非正、压力非有限）；
    * 反解出**非正**比内能，被下面的有效性判据当成"不是该 EOS 的合法状态"剔除。
      产物上这等价于 ``p < p_cold(V)``，即该单元的化学能还没放出来（未起爆、或正
      被爆轰波扫过的装药），套产物 JWL 本来就不成立。

    **这是最后手段。** 正常情况下比内能随剖面一起写出（``raw_line.npz`` 的 ``e``），
    由提取器在**单元级**先反解再加权平均，比拿壳层平均后的 ``(rho, p)`` 再解一次准。
    只有旧的、没有 ``e`` 的导出才会走到这条路径。
    """
    params = jwl or mat.product_jwl()
    density = np.asarray(density, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    material = np.asarray(material)

    out = np.full(density.shape, np.nan, dtype=np.float64)
    air = material == 1
    if np.any(air):
        rho = density[air]
        with np.errstate(divide="ignore", invalid="ignore"):
            value = pressure[air] / ((gamma_air - 1.0) * rho)
        out[air] = np.where(np.isfinite(value) & (value > 0.0), value, np.nan)

    product = material == 0
    if np.any(product):
        rho = density[product]
        rho_safe = np.maximum(rho, 1.0e-30)
        v = params.rho0 / rho_safe
        p_cold = jwl_pressure_numpy(rho, np.zeros_like(rho), params)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (pressure[product] - p_cold) / (params.omega * rho_safe)
        out[product] = np.where(np.isfinite(value) & (value > 0.0), value, np.nan)
    return out


@dataclass
class FieldMapping:
    """逻辑字段名到数据键的映射。"""

    mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_MAP))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "FieldMapping":
        if not payload:
            return cls()
        merged = dict(DEFAULT_FIELD_MAP)
        merged.update({str(k): str(v) for k, v in payload.items()})
        return cls(mapping=merged)

    def key(self, logical: str) -> str:
        try:
            return self.mapping[logical]
        except KeyError as exc:  # pragma: no cover - guarded by validation
            raise DataError(f"no data key mapped for logical field {logical!r}") from exc


@dataclass
class DetectorSettings:
    """前沿识别参数。每个都在质量报告里回报，便于事后审查。"""

    #: 主激波判据：压力超过环境态的比例。
    shock_pressure_ratio: float = 1.0e-3
    #: 接触面判据：在内侧剖面上密度最陡下降位置；越过主激波即停止搜索。
    contact_smoothing_cells: int = 5
    #: 产物密度必须至少是环境密度的这个倍数，否则该帧的接触面判为不可靠。
    contact_density_ratio_floor: float = 1.02
    #: 需要的最少纯材料单元数。
    minimum_cells_per_medium: int = 3

    def as_dict(self) -> dict[str, float]:
        return {
            "shock_pressure_ratio": self.shock_pressure_ratio,
            "contact_smoothing_cells": self.contact_smoothing_cells,
            "contact_density_ratio_floor": self.contact_density_ratio_floor,
            "minimum_cells_per_medium": self.minimum_cells_per_medium,
        }


@dataclass
class ExtractedCase:
    """一个半径的时空剖面，附带识别出的前沿与有效掩码。"""

    case_id: str
    charge_radius: float
    times: np.ndarray               # 物理时间，s
    radius: np.ndarray              # 固定的径向网格，m
    density: np.ndarray             # (N_t, N_r)
    velocity: np.ndarray
    pressure: np.ndarray
    specific_energy: np.ndarray
    contact_radius: np.ndarray      # (N_t,)
    shock_radius: np.ndarray
    contact_valid: np.ndarray       # (N_t,) bool
    shock_valid: np.ndarray
    material: np.ndarray            # (N_t, N_r) int：0 产物、1 空气、2 未受扰动
    valid: np.ndarray               # (N_t, N_r) bool
    source: dict[str, Any] = field(default_factory=dict)
    detector: DetectorSettings = field(default_factory=DetectorSettings)

    #: 逐字段有效性（§11.1）。键是 ``density`` / ``velocity`` / ``specific_energy``
    #: / ``pressure``，值是 ``(N_t, N_r)`` 的布尔掩码。
    #:
    #: ``None`` 表示**这份导出没有记录**，不表示"都有效"。两者必须能分开：一个
    #: 缺字段的旧导出会让采样器退回联合掩码，而那是要显式告诉调用方的退化。
    field_validity: dict[str, np.ndarray] | None = None

    #: 前沿监督资格（§10）：检测状态与"够不够格当标签"分开，两条前沿各一份。
    #: 同样是 ``None`` 表示未记录，而不是"都够格"。
    front_supervision_eligibility: dict[str, Any] | None = None

    # -- 时间（方案 B）----------------------------------------------------
    @property
    def detonation_velocity(self) -> float:
        """卡片上的爆速 ``D``，m/s，来自提取元数据。

        **没有它就不能算时间原点。** 方案 B 的原点是 ``T_h = R_0/D + epsilon``，
        缺少 ``D`` 时唯一"能用"的做法是退回只有 ``epsilon`` 的方案 A 时间轴——而两条
        时间轴都只有秒的量纲，退回去不会报错，只会把整条轨迹平移一个 ``T_b``。
        所以这里直接失败，而不是给一个默认值。
        """
        velocity = float(self.source.get("detonation_velocity_m_s") or 0.0)
        if not (velocity > 0.0):
            raise DataError(
                f"{self.case_id}: the extraction metadata carries no usable detonation "
                "velocity, so the scheme-B time origin T_h = R_0/D + epsilon cannot be "
                "formed. A cache written before the scheme-B migration must be "
                "re-extracted, not silently read with the scheme-A clock."
            )
        return velocity

    @property
    def pinn_times(self) -> np.ndarray:
        """``tau = T - T_h``，``T_h = R_0/D + epsilon``。"""
        return self.times - self.start_time

    @property
    def start_time(self) -> float:
        """``T_h``，s：PINN 的启动时刻（方案 B）。"""
        return handover_time(
            self.charge_radius, detonation_velocity=self.detonation_velocity
        )

    def _after_start(self) -> np.ndarray:
        """时间落在 PINN 启动时刻之后的帧。"""
        return self.pinn_times >= 0.0

    # -- 位置不确定性（§11.2）--------------------------------------------
    @property
    def grid_spacing(self) -> float:
        """径向网格间距，m。探测器的分辨率下限。"""
        if self.radius.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.radius)))

    def contact_position_uncertainty(self) -> float:
        """接触面位置的冻结不确定性，m。

        接触面是"内侧剖面上密度最陡下降"定位的，用的是
        ``detector.contact_smoothing_cells`` 个单元的平滑窗口——所以它自己的
        分辨率就是那条窗口，不是单个单元。**这个值由数据与探测器设置决定，
        训练过程中不更新**：位置约束的 delta 一旦随训练自动扩大，就会把错误预测
        一起包容进去，约束就失去意义了。
        """
        return float(self.detector.contact_smoothing_cells) * self.grid_spacing

    def shock_position_uncertainty(self) -> float:
        """主激波位置的冻结不确定性，m。

        激波是压力越过阈值的位置，落在网格上，所以不确定性是一个单元。真实数据
        若记录了跃迁带宽度，应按那个宽度取——这里用分辨率下界，不夸大。
        """
        return self.grid_spacing

    def _air_gap_cells(self) -> np.ndarray:
        """``Rs`` 与 ``Rc`` 之间的单元数。任一侧前沿不可用时为 0。"""
        gap = np.zeros(self.times.size, dtype=np.int64)
        both = self.contact_valid & self.shock_valid
        for index in np.flatnonzero(both):
            gap[index] = int(
                np.count_nonzero(
                    (self.radius > self.contact_radius[index])
                    & (self.radius < self.shock_radius[index])
                )
            )
        return gap

    def supervisable_frames(self, *, minimum_air_cells: int = 1) -> np.ndarray:
        """可以用于监督的帧：时间在启动之后、且**至少一个前沿**识别可靠。

        提示词 §11.1 要求两套前沿**独立**。这里以前写的是
        ``contact_valid & shock_valid``：只要激波那一天没识别出来，整帧就从训练
        窗口里消失——**接触面明明可靠，也被一起丢掉**，而且丢得完全无声（表现为
        有效时间范围莫名其妙地短）。

        现在改成"至少一侧可靠"。空气间隙要求只在**两侧都可用**时才施加：只有
        单侧前沿时算不出间隙，用"算不出"去否决整帧就是拿缺失当不合格。

        提示词要求**每个训练半径取自己的最后有效仿真帧作为终止时间**，不要求
        三个半径的终止时间同比例缩放。这里给出的就是这个集合，终止时间由
        :meth:`last_valid_pinn_time` 给出。
        """
        front_ok = self.contact_valid | self.shock_valid
        gap_ok = (self._air_gap_cells() >= int(minimum_air_cells)) | ~(
            self.contact_valid & self.shock_valid
        )
        return self._after_start() & front_ok & gap_ok

    def contact_supervisable_frames(self) -> np.ndarray:
        """接触面**单独**可监督的帧（§11.1）。不要求激波有效。"""
        return self._after_start() & self.contact_valid

    def shock_supervisable_frames(self, *, minimum_air_cells: int = 1) -> np.ndarray:
        """激波**单独**可监督的帧（§11.1）。不要求接触面有效。"""
        return self._after_start() & self.shock_valid

    def air_supervisable_frames(self, *, minimum_air_cells: int = 1) -> np.ndarray:
        """空气区（``Rc`` 与 ``Rs`` 之间）可监督的帧。

        这一项**必须**两侧前沿都可靠——空气间隙是两套前沿共同定义的。它单独成
        一个方法，而不是混进 :meth:`supervisable_frames` 去否决整帧。
        """
        return (
            self._after_start()
            & self.contact_valid
            & self.shock_valid
            & (self._air_gap_cells() >= int(minimum_air_cells))
        )

    def frame_split(
        self,
        *,
        holdout_fraction: float = 0.1,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把可监督帧按**连续帧组**划分成 ``(train, development)``。

        提示词 §11 要求"优先按完整时间段/帧组留出，而非将同一帧相邻网格点随机
        拆分后宣称独立验证"。同一帧相邻网格点高度相关，随机拆分会让两个集合
        互相看得见对方，于是开发集误差只是训练误差的一个更贵的写法。

        这里按位置把可用帧切成 ``round(1/holdout_fraction)`` 个连续块，整块留一
        块作开发检查；留哪一块由 ``seed`` 决定，因此同一份数据在同一个种子上
        永远给出同一个划分（划分本身要可复现，才谈得上"在看最终结果之前固定"）。

        返回值是**真实帧号**，不是位置——见 :meth:`ExtractedCase.sample_at`
        的调用约定。
        """
        if not (0.0 < float(holdout_fraction) < 0.5):
            raise DataError(
                f"holdout_fraction must lie in (0, 0.5), got {holdout_fraction}"
            )
        usable = np.flatnonzero(self.supervisable_frames())
        if usable.size < 2:
            return usable, np.zeros(0, dtype=np.int64)

        blocks = max(2, int(round(1.0 / float(holdout_fraction))))
        blocks = min(blocks, usable.size)
        edges = np.linspace(0, usable.size, blocks + 1).astype(int)
        held = int(seed) % blocks
        # 避开第一块：它承载启动时刻附近最陡的一段，也是 IC 之外唯一能看到早期
        # 演化的监督；把它整个留出去会让开发集与训练集的时间分布差得太远。
        if blocks > 1 and held == 0:
            held = 1
        positions = np.arange(usable.size)
        development = usable[(positions >= edges[held]) & (positions < edges[held + 1])]
        train = usable[(positions < edges[held]) | (positions >= edges[held + 1])]
        return train, development

    def last_valid_pinn_time(self) -> float:
        usable = self.supervisable_frames()
        if not np.any(usable):
            raise DataError(
                f"case {self.case_id!r} has no supervisable frame: the front detectors "
                "never produced a reliable contact/shock pair after the start time"
            )
        return float(self.pinn_times[usable].max())

    def last_valid_physical_time(self) -> float:
        return self.last_valid_pinn_time() + self.start_time

    # -- 查询 -------------------------------------------------------------
    def frame_index(self, physical_time: float) -> int:
        return int(np.argmin(np.abs(self.times - float(physical_time))))

    def sample_at(self, frame: int, radii: np.ndarray) -> dict[str, np.ndarray]:
        """在一帧的给定半径上取值；**线性插值，不跨间断**。

        调用方负责只在同一材料的区间内查询。落在两个单元之间时，只要有一侧是
        ``nan``，结果就是 ``nan``——插值到无效单元不会得到一个看起来合理的数。

        **恰好落在网格点上时必须返回该点的值本身。** ``a * (1 - w) + b * w`` 在
        ``w = 0`` 处并不等于 ``a``：``nan * 0`` 仍然是 ``nan``。采样器查的正是
        单元中心，也就是**网格点本身**，所以只要相邻单元有一个 ``nan``，那个位置
        上的有效标签就会被这一项污染成 ``nan``——位置选对了，取值又把它毁了。

        所以这里对 ``w = 0`` 与 ``w = 1`` 各走一条短路：权重为零的一侧**不参与
        运算**，而不是"参与运算但乘 0"。
        """
        radii = np.asarray(radii, dtype=np.float64)
        clipped = np.clip(radii, self.radius[0], self.radius[-1])
        index = np.searchsorted(self.radius, clipped, side="right") - 1
        index = np.clip(index, 0, self.radius.size - 2)
        left = self.radius[index]
        right = self.radius[index + 1]
        weight = np.where(right > left, (clipped - left) / np.maximum(right - left, 1e-300), 0.0)
        on_left = weight <= 0.0
        on_right = weight >= 1.0
        mixed = ~(on_left | on_right)

        def blend(field: np.ndarray) -> np.ndarray:
            row = field[frame]
            left_value = row[index]
            right_value = row[index + 1]
            out = np.full(left_value.shape, np.nan, dtype=np.float64)
            # 恰好命中左/右网格点：直接取该点的值，另一侧完全不参与
            out[on_left] = left_value[on_left]
            out[on_right] = right_value[on_right]
            # 真正的区间内查询：两侧都得有效，否则结果无定义
            if np.any(mixed):
                w = weight[mixed]
                out[mixed] = (
                    left_value[mixed] * (1.0 - w) + right_value[mixed] * w
                )
            return out

        return {
            "density": blend(self.density),
            "velocity": blend(self.velocity),
            "specific_energy": blend(self.specific_energy),
            "pressure": blend(self.pressure),
        }

    # -- 质量报告 ---------------------------------------------------------
    def quality_report(self) -> dict[str, Any]:
        usable = self.supervisable_frames()
        air_cells = np.zeros(self.times.size, dtype=np.int64)
        for index in range(self.times.size):
            if self.contact_valid[index] and self.shock_valid[index]:
                air_cells[index] = int(
                    np.count_nonzero(
                        (self.radius > self.contact_radius[index])
                        & (self.radius < self.shock_radius[index])
                    )
                )
        gaps = np.where(
            self.contact_valid & self.shock_valid,
            self.shock_radius - self.contact_radius,
            np.nan,
        )
        return {
            "case_id": self.case_id,
            "charge_radius_m": self.charge_radius,
            "n_frames": int(self.times.size),
            "n_frames_supervisable": int(np.count_nonzero(usable)),
            "time_start_s": self.start_time,
            "t_phys_first_s": float(self.times[0]),
            "t_phys_last_s": float(self.times[-1]),
            "t_pinn_last_valid_s": (
                float(self.pinn_times[usable].max()) if np.any(usable) else None
            ),
            "contact_valid_fraction": float(np.mean(self.contact_valid)),
            "shock_valid_fraction": float(np.mean(self.shock_valid)),
            "min_air_cells": int(air_cells[usable].min()) if np.any(usable) else 0,
            "min_gap_m": float(np.nanmin(gaps)) if np.any(np.isfinite(gaps)) else None,
            "energy_source": self.source.get(
                "energy_source", "unknown — this object was not built from a directory"
            ),
            "front_resolution": self.source.get(
                "front_resolution", "not recorded by this extraction"
            ),
            "material_masks_from": "density-steepest-descent contact detector (heuristic)",
            "detector": self.detector.as_dict(),
            "frames_rejected_contact_below_charge_radius": int(
                self.source.get("frames_rejected_contact_below_charge_radius", 0)
            ),
            "detector_reliability": (
                "该判据是启发式的，不是从材料体积分数读出来的。本数据集上它给出的"
                "接触面位置存在量化跳变，且大量帧落在装药半径以内（物理上不可能）；"
                "这些帧已按物理约束剔除，但**探测器本身仍不可靠**——用它做的前沿监督"
                "只能作为弱标签，权威的材料分区需要从 d3plot 的材料体积分数重新提取。"
            ),
            **{k: v for k, v in self.source.items() if k != "raw"},
        }

    def describe(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "charge_radius": self.charge_radius,
            "n_frames": int(self.times.size),
            "n_radii": int(self.radius.size),
            "radius_min": float(self.radius[0]),
            "radius_max": float(self.radius[-1]),
            "time_min": float(self.times[0]),
            "time_max": float(self.times[-1]),
        }


# --------------------------------------------------------------------------
# 前沿识别
# --------------------------------------------------------------------------


def detect_shock_radius(
    radius: np.ndarray,
    pressure: np.ndarray,
    *,
    ambient_pressure: float = mat.AIR_PRESSURE,
    settings: DetectorSettings | None = None,
) -> float:
    """主激波半径：压力仍明显高于环境态的最外侧位置。"""
    settings = settings or DetectorSettings()
    excess = np.asarray(pressure) - float(ambient_pressure)
    reference = float(np.max(np.abs(excess))) if excess.size else 0.0
    if reference <= 0.0:
        return float("nan")
    active = np.flatnonzero(excess > settings.shock_pressure_ratio * reference)
    if active.size == 0:
        return float("nan")
    index = int(active[-1])
    if index + 1 >= radius.size:
        return float("nan")
    return float(radius[index + 1])


def detect_contact_radius(
    radius: np.ndarray,
    density: np.ndarray,
    shock_radius: float,
    *,
    ambient_density: float = mat.AIR_DENSITY,
    settings: DetectorSettings | None = None,
) -> tuple[float, float]:
    """接触面半径：内侧剖面上密度最陡下降的位置。

    返回 ``(R_c, 判据余量)``。余量是接触面两侧的密度对比度，用来判断这一帧
    是否可靠；低于 :attr:`DetectorSettings.contact_density_ratio_floor` 时
    调用方应把该帧标为无效。
    """
    settings = settings or DetectorSettings()
    radius = np.asarray(radius, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if not np.isfinite(shock_radius):
        return float("nan"), 0.0

    inside = radius <= shock_radius
    if np.count_nonzero(inside) < 2 * settings.minimum_cells_per_medium:
        return float("nan"), 0.0

    indices = np.flatnonzero(inside)
    row = density[indices]
    if not np.all(np.isfinite(row)):
        return float("nan"), 0.0

    window = max(1, int(settings.contact_smoothing_cells))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(row, kernel, mode="same")
    else:
        smoothed = row

    gradient = np.gradient(smoothed, radius[indices])
    drop_index = int(np.argmin(gradient))
    position = int(indices[drop_index])
    if position + 1 >= radius.size:
        position = radius.size - 2

    outer_density = float(density[position + 1])
    inner_density = float(density[position])
    contrast = inner_density / max(outer_density, 1.0e-30)
    if contrast < settings.contact_density_ratio_floor:
        return float(radius[position + 1]), contrast
    return float(radius[position + 1]), contrast


def partition_materials(
    radius: np.ndarray,
    density: np.ndarray,
    pressure: np.ndarray,
    contact_radius: float,
    shock_radius: float,
    *,
    settings: DetectorSettings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按识别出的前沿把剖面分成产物 / 空气 / 未受扰动三区。

    返回 ``(material, valid)``。``material`` 用 ``0/1/2`` 编码，``valid`` 标记
    哪些点可以作为监督标签。紧邻接触面与主激波的一个单元被判为混合或展宽带，
    标为无效——提示词要求"剔除数值激波展宽区和无法可靠区分材料的单元标签"。
    """
    settings = settings or DetectorSettings()
    radius = np.asarray(radius, dtype=np.float64)
    n = radius.size
    material = np.full(n, 2, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)

    if not (np.isfinite(contact_radius) and np.isfinite(shock_radius)):
        return material, valid
    if shock_radius <= contact_radius:
        return material, valid

    product = radius < contact_radius
    air = (radius >= contact_radius) & (radius < shock_radius)
    material[product] = 0
    material[air] = 1

    valid[product] = True
    valid[air] = True

    # 接触面与主激波附近各剔除一个单元。
    buffer = max(1, int(settings.minimum_cells_per_medium) // 2)
    near_contact = np.abs(radius - contact_radius) <= buffer * _cell_width(radius)
    near_shock = np.abs(radius - shock_radius) <= buffer * _cell_width(radius)
    valid &= ~(near_contact | near_shock)

    # 非有限值一律无效。
    for field in (density, pressure):
        valid &= np.isfinite(np.asarray(field, dtype=np.float64))
    return material, valid


def _cell_width(radius: np.ndarray) -> float:
    if radius.size < 2:
        return 1.0
    return float(np.median(np.diff(radius)))


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------


#: 前沿三态码的取值域。**0/1/2 之外的码不是"另一档无效"，是格式错误**：
#: 读取器不认识它，说明这份导出与当前约定不是同一套。把它当成普通无效帧处理，
#: 会把"我们对不上这份数据"说成"这一帧没测到"。
FRONT_STATE_CODES = (0, 1, 2)


def _check_axis(values, *, name: str, path) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise DataError(f"{path}: {name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise DataError(f"{path}: {name} is empty")
    if not np.all(np.isfinite(array)):
        raise DataError(f"{path}: {name} contains non-finite values")
    return array


def validate_case_arrays(
    *,
    times,
    radius,
    profiles: Mapping[str, Any],
    fronts: Mapping[str, Any] | None = None,
    states: Mapping[str, Any] | None = None,
    eligibility: Mapping[str, Any] | None = None,
    path: Any = "<memory>",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """**所有数组与两条坐标轴的关系，一次核完。**

    以前这件事是分散的：启发式路径核了 ``density`` 的形状与两条轴的单调性，
    体积分数路径两条都没核，而两条路径都没核 ``velocity`` / ``pressure`` /
    ``energy`` 的形状、状态码的取值域、以及资格数组的**维度**。分散的后果不是
    "少查了一项"，而是**两份数据用两套规则**：同一份导出换条路径读，一个报错、
    一个放行。

    具体的漏法（都实际复现过）：

    * ``density (2, 3)`` 配 ``velocity (1, 3)``、时间两帧 —— 读取成功，然后查
      第二帧抛 ``IndexError``。也就是说**被判就绪的数据在正常消费时会崩**。
    * 时间轴倒序或重复 —— 直接读取成功。排序、``searchsorted``、分层取帧全都
      建立在"时间严格递增"上，而它们一个都不报错。
    * 资格数组形状 ``(1, 2)`` —— 被 ``reshape(-1)`` 展平成 ``(2,)`` 通过。
      长度相同不代表格式正确。
    * 状态码 ``999`` —— 当成普通无效帧放行。不认识的码是**格式**问题。

    全部在这里一次性拦下，两条读取路径共用。
    """
    times = _check_axis(times, name="the time axis", path=path)
    radius = _check_axis(radius, name="the radial axis", path=path)
    if times.size < 1:
        raise DataError(f"{path}: the time axis needs at least one frame")
    if radius.size < 2:
        raise DataError(f"{path}: the radial axis needs at least two points")
    if not np.all(np.diff(times) > 0.0):
        # 严格递增：相等与倒序都不行。相等会让"第几帧"失去意义，倒序会让所有
        # 基于顺序的切片取反。
        raise DataError(
            f"{path}: the time axis must be strictly increasing; got "
            f"{times[:4].tolist()}{'...' if times.size > 4 else ''}"
        )
    if not np.all(np.diff(radius) > 0.0):
        raise DataError(f"{path}: the radial axis must be strictly increasing")

    expected = (times.size, radius.size)
    normalised: dict[str, np.ndarray] = {}
    for name, values in profiles.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1 and array.size == radius.size and times.size == 1:
            array = array.reshape(1, -1)
        if array.shape != expected:
            raise DataError(
                f"{path}: {name} has shape {array.shape}, expected {expected} "
                f"(frames, radial points). Every profile must share one grid; a "
                "field whose frame count differs from the time axis will fail later, "
                "at a query, where the reason is no longer visible"
            )
        normalised[name] = array

    for name, values in (fronts or {}).items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != times.size:
            raise DataError(
                f"{path}: front {name!r} has {array.size} entries but the time axis "
                f"has {times.size} frames"
            )
        if np.any(np.isinf(array)):
            raise DataError(f"{path}: front {name!r} contains infinite values")
        normalised[name] = array

    for name, values in (states or {}).items():
        array = np.asarray(values).reshape(-1)
        if array.size != times.size:
            raise DataError(
                f"{path}: state code {name!r} has {array.size} entries but the time "
                f"axis has {times.size} frames"
            )
        unknown = sorted(set(np.unique(array).tolist()) - set(FRONT_STATE_CODES))
        if unknown:
            raise DataError(
                f"{path}: state code {name!r} contains {unknown}, which this reader "
                f"does not define (known: {list(FRONT_STATE_CODES)}). An unrecognised "
                "code is a format mismatch, not an unusable frame"
            )
        normalised[name] = array.astype(np.int64)

    for name, values in (eligibility or {}).items():
        array = np.asarray(values)
        if array.dtype != np.bool_:
            raise DataError(
                f"{path}: eligibility {name!r} has dtype {array.dtype}, not bool"
            )
        if array.ndim != 1 or array.shape != (times.size,):
            # 这里**不能** ``reshape(-1)``：``(1, 2)`` 会被展平成 ``(2,)`` 然后
            # 长度检查通过。长度相同不代表格式正确——一个二维的资格表说明它本来
            # 不是逐帧的量。
            raise DataError(
                f"{path}: eligibility {name!r} has shape {array.shape}, expected "
                f"{(times.size,)} (one boolean per frame)"
            )
        normalised[name] = array

    return times, radius, normalised


def load_extracted_case(
    directory: Path | str,
    *,
    case_id: str | None = None,
    field_mapping: FieldMapping | None = None,
    settings: DetectorSettings | None = None,
    gamma_air: float = mat.AIR_GAMMA,
) -> ExtractedCase:
    """读取 ``raw_line.npz`` + ``metadata.json``，识别前沿并分区。"""
    directory = Path(directory)
    npz_path = directory / "raw_line.npz"
    meta_path = directory / "metadata.json"
    if not npz_path.exists():
        raise DataError(f"missing {npz_path}")
    if not meta_path.exists():
        raise DataError(f"missing {meta_path}")

    mapping = field_mapping or FieldMapping()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path, allow_pickle=True)

    # 如果有配套的 ``material.npz``（由 ``cli.extract_profiles`` 从 d3plot 的 ALE
    # 材料体积分数写出），材料分区就是**读数**而不是启发式判据。两条路径在这里
    # 分岔，并且把用的是哪一条如实记进 ``source``。
    companion = directory / "material.npz"
    if companion.exists():
        return _load_with_volume_fraction_partition(
            directory, npz_path, meta_path, companion, metadata, data, mapping,
            case_id=case_id, gamma_air=gamma_air,
        )

    for logical in REQUIRED_FIELDS:
        key = mapping.key(logical)
        if key not in data:
            raise DataError(
                f"{npz_path} has no key {key!r} for logical field {logical!r}; "
                f"available keys are {sorted(data.files)}"
            )

    times = np.asarray(data[mapping.key("time")], dtype=np.float64).reshape(-1)
    radius = np.asarray(data[mapping.key("radius")], dtype=np.float64).reshape(-1)
    density = np.atleast_2d(np.asarray(data[mapping.key("density")], dtype=np.float64))
    velocity = np.atleast_2d(np.asarray(data[mapping.key("velocity")], dtype=np.float64))
    pressure = np.atleast_2d(np.asarray(data[mapping.key("pressure")], dtype=np.float64))

    # 统一的形状 / 坐标轴校验（两条读取路径共用同一套规则）。
    times, radius, checked = validate_case_arrays(
        times=times, radius=radius,
        profiles={
            "density": density, "velocity": velocity, "pressure": pressure,
        },
        path=npz_path,
    )
    density, velocity, pressure = (
        checked["density"], checked["velocity"], checked["pressure"]
    )

    charge_radius = float(metadata.get("R_0", np.nan))
    if not np.isfinite(charge_radius) or charge_radius <= 0.0:
        raise DataError(f"{meta_path} does not declare a usable R_0")

    settings = settings or DetectorSettings()
    n_t = times.size
    contact = np.full(n_t, np.nan)
    shock = np.full(n_t, np.nan)
    contact_contrast = np.zeros(n_t)
    material = np.full((n_t, radius.size), 2, dtype=np.int64)
    valid = np.zeros((n_t, radius.size), dtype=bool)

    for frame in range(n_t):
        shock[frame] = detect_shock_radius(
            radius, pressure[frame], settings=settings
        )
        contact[frame], contact_contrast[frame] = detect_contact_radius(
            radius, density[frame], shock[frame], settings=settings
        )
        material[frame], valid[frame] = partition_materials(
            radius,
            density[frame],
            pressure[frame],
            contact[frame],
            shock[frame],
            settings=settings,
        )

    # 接触面**不可能**跑到装药半径以内。这不是启发式过滤，而是一条物理约束：
    # 接触面就是原装药表面所在的那个物质界面，它只能随产物向外走。一条 R_c < R_0
    # 的标签一定会把前沿网络往内拉，最后把一个本来合法的移动域变成非法域。
    #
    # 沿用阈值形式的启发式判据而不加这条约束，会让 50 mm 的 101 个"有效"帧里
    # 有 86 个带着 R_c < R_0——被剔除的帧数因此也写进质量报告，剔除原因不隐藏。
    below_charge = np.isfinite(contact) & (contact < charge_radius)
    contact_valid = (
        np.isfinite(contact)
        & (contact_contrast >= settings.contact_density_ratio_floor)
        & np.isfinite(shock)
        & (shock > contact)
        & ~below_charge
    )
    shock_valid = np.isfinite(shock)

    # 比内能由 (rho, p) 按材料反解，逐帧进行。
    energy = np.full_like(density, np.nan)
    for frame in range(n_t):
        energy[frame] = specific_energy_from_pressure(
            density[frame], pressure[frame], material[frame], gamma_air=gamma_air
        )

    return ExtractedCase(
        case_id=case_id or str(metadata.get("case_id", directory.name)),
        charge_radius=charge_radius,
        times=times,
        radius=radius,
        density=density,
        velocity=velocity,
        pressure=pressure,
        specific_energy=energy,
        contact_radius=contact,
        shock_radius=shock,
        contact_valid=contact_valid,
        shock_valid=shock_valid,
        material=material,
        valid=valid,
        source={
            "directory": str(directory),
            "npz": str(npz_path),
            "metadata": str(meta_path),
            "field_map": mapping.mapping,
            "metadata_raw": metadata,
            "contact_contrast_median": float(np.median(contact_contrast)),
            "frames_rejected_contact_below_charge_radius": int(np.count_nonzero(below_charge)),
            "frames_with_detected_contact_below_charge_radius": int(
                np.count_nonzero(below_charge)
            ),
            "detonation_velocity_m_s": metadata.get("detonation_velocity_m_s"),
        },
        detector=settings,
    )


def _load_with_volume_fraction_partition(
    directory: Path,
    npz_path: Path,
    meta_path: Path,
    companion: Path,
    metadata: Mapping[str, Any],
    data,
    mapping: FieldMapping,
    *,
    case_id: str | None,
    gamma_air: float,
):
    """由 ALE 材料体积分数直接给出材料分区的那条路径。

    与启发式路径的区别不是精度，是**依据**：这里的材料归属来自求解器自己写出的
    体积分数，不需要从密度梯度上猜接触面在哪。所以 ``material_partition`` 这一项
    可以按"已核实"处理。
    """
    for logical in REQUIRED_FIELDS:
        key = mapping.key(logical)
        if key not in data:
            raise DataError(
                f"{npz_path} has no key {key!r} for logical field {logical!r}; "
                f"available keys are {sorted(data.files)}"
            )
    block = np.load(companion, allow_pickle=True)

    times = np.asarray(data[mapping.key("time")], dtype=np.float64).reshape(-1)
    radius = np.asarray(data[mapping.key("radius")], dtype=np.float64).reshape(-1)
    density = np.atleast_2d(np.asarray(data[mapping.key("density")], dtype=np.float64))
    velocity = np.atleast_2d(np.asarray(data[mapping.key("velocity")], dtype=np.float64))
    pressure = np.atleast_2d(np.asarray(data[mapping.key("pressure")], dtype=np.float64))

    charge_radius = float(metadata.get("R_0", np.nan))
    if not np.isfinite(charge_radius) or charge_radius <= 0.0:
        raise DataError(f"{meta_path} does not declare a usable R_0")

    # 统一的形状 / 坐标轴校验。**与启发式路径同一套规则**——以前这条路径既没核
    # 时间轴的单调性，也没核 velocity/pressure 的形状。
    times, radius, checked = validate_case_arrays(
        times=times, radius=radius,
        profiles={"density": density, "velocity": velocity, "pressure": pressure},
        path=npz_path,
    )
    density, velocity, pressure = (
        checked["density"], checked["velocity"], checked["pressure"]
    )

    material = np.atleast_2d(np.asarray(block["material"], dtype=np.int64))
    valid = np.atleast_2d(np.asarray(block["valid"], dtype=bool))
    if material.shape != density.shape or valid.shape != density.shape:
        raise DataError(
            f"{companion} has material {material.shape} / valid {valid.shape}, but the "
            f"profiles are {density.shape}; the companion file belongs to another extraction"
        )

    contact_radius = np.asarray(block["contact_radius"], dtype=np.float64).reshape(-1)
    shock_radius = np.asarray(block["shock_radius"], dtype=np.float64).reshape(-1)
    # 前沿的有效性判据在这里是"数出来了"，不是"猜出来了"。
    #
    # **两条前沿必须解耦**（§11.1）。以前 ``contact_valid`` 里带着
    # ``isfinite(shock_radius) & (shock_radius > contact_radius)``：激波那天没识别
    # 出来，接触面明明可靠也被一起撤标。"接触面在激波内侧"是**空气层**的要求
    # （``h = R_s - R_c`` 要有意义），不是接触面标签的前提；把它挂在这里，等于让
    # 一个前沿的缺失去惩罚另一个。
    geometric_contact = (
        np.isfinite(contact_radius)
        & (contact_radius > 0.0)
        & (contact_radius >= charge_radius - 1.0e-9)
    )
    geometric_shock = np.isfinite(shock_radius)
    # 单独记录、单独使用：只有空气组与"两前沿都在"的判据才需要它。
    front_ordering_ok = (
        np.isfinite(contact_radius)
        & np.isfinite(shock_radius)
        & (shock_radius > contact_radius)
    )

    def _frame_array(key: str):
        value = block.get(key) if hasattr(block, "get") else None
        return None if value is None else np.asarray(value)

    # 前沿、状态码、资格数组**一起**交给统一的校验：它们的维度、长度、取值域
    # 与两条坐标轴的关系在那里一次核完。以前每个数组各查各的，于是"资格数组被
    # ``reshape(-1)`` 展平"和"状态码 999 被当成普通无效帧"都能溜过去。
    _, _, checked_fronts = validate_case_arrays(
        times=times, radius=radius,
        profiles={
            "density": density, "velocity": velocity, "pressure": pressure,
        },
        fronts={
            "contact_radius": contact_radius, "shock_radius": shock_radius,
        },
        states={
            key: array for key, array in (
                ("contact_state", _frame_array("contact_state")),
                ("shock_state", _frame_array("shock_state")),
            ) if array is not None
        },
        eligibility={
            key: array for key, array in (
                ("contact_has_supervision", _frame_array("contact_has_supervision")),
                ("shock_has_supervision", _frame_array("shock_has_supervision")),
            ) if array is not None
        },
        path=companion,
    )
    contact_radius = checked_fronts["contact_radius"]
    shock_radius = checked_fronts["shock_radius"]
    contact_state = checked_fronts.get("contact_state")
    shock_state = checked_fronts.get("shock_state")
    contact_eligibility = checked_fronts.get("contact_has_supervision")
    shock_eligibility = checked_fronts.get("shock_has_supervision")

    # **每条前沿各自决定用什么来源。**
    #
    # 以前这里是"两侧资格数组都在，才用资格数组；否则两侧一起退回三态码"。于是
    # 接触侧的资格数组**明确写着 False**、激波侧的数组缺失时，接触侧会跟着退回
    # 三态码——一个被显式记录下来的否决，被**另一侧字段的缺失**覆盖掉。这既违反
    # "资格优先"，也违反"两条前沿独立"：A 的缺失改变了 B 的判据。
    #: 资格证据的档位。**分档而不是只分"有/没有"**：三态码是真的证据，但它不是
    #: 当前契约要求的那一档（§10 的逐帧资格），两者混在一起会让"旧格式"读起来像
    #: "已就绪"。
    LEVEL_ARRAYS = "arrays"
    LEVEL_LEGACY_CODE = "legacy_state_code"
    LEVEL_NONE = "none"

    def _front(eligibility, state, geometric, name: str):
        if eligibility is not None:
            return geometric & np.asarray(eligibility, dtype=bool), (
                f"per-frame supervision eligibility array ({name}_has_supervision)"
            ), LEVEL_ARRAYS
        if state is not None:
            # 三态码是**较弱的**证据：它只说"分辨得出吗"，不说"够不够格当标签"。
            # 比"半径有限即可"强，所以这一侧单独用它——但它**不是**当前契约要的
            # 那一档，所以档位记成 legacy，契约会据此判未就绪（可诊断、不可训练）。
            return geometric & (state == 2), (
                f"three-state detection code ({name}_state == 2; no eligibility array)"
            ), LEVEL_LEGACY_CODE
        return geometric, f"none recorded for {name} — UNVERIFIED", LEVEL_NONE

    contact_valid, contact_source, contact_level = _front(
        contact_eligibility, contact_state, geometric_contact, "contact"
    )
    shock_valid, shock_source, shock_level = _front(
        shock_eligibility, shock_state, geometric_shock, "shock"
    )
    front_eligibility = {"contact": contact_source, "shock": shock_source}
    contact_eligibility_level = contact_level
    shock_eligibility_level = shock_level
    # 整体档位取两侧里**最弱**的那一档：一侧是数组、另一侧只有三态码时，后者仍然
    # 是正式训练的缺口，不能被前者的强度盖住。
    _order = {LEVEL_ARRAYS: 2, LEVEL_LEGACY_CODE: 1, LEVEL_NONE: 0}
    eligibility_level = min(
        (contact_level, shock_level), key=lambda value: _order[value]
    )
    front_resolution = (
        "each front uses its own strongest available source, independently: "
        f"contact -> {contact_source}; shock -> {shock_source}. A missing record on one "
        "side never changes the other side's verdict, and an explicitly recorded False is "
        "never overridden by a fallback"
    )
    if eligibility_level == LEVEL_LEGACY_CODE:
        front_resolution += (
            ". At least one front falls back to the legacy three-state code, which is "
            "weaker evidence than the per-frame eligibility arrays the current contract "
            "requires; the data contract refuses it for formal training"
        )
    elif eligibility_level == LEVEL_NONE:
        front_resolution += (
            ". At least one front has no eligibility evidence at all, so every finite "
            "radius is being treated as a label — the data contract refuses this"
        )

    # 比内能优先用提取时写下的那一列。它是**单元级**反解之后再按体积加权的，比拿
    # 壳层平均的 ``(rho, p)`` 再解一次准；而且产物膨胀到低密度、压力转负之后那个
    # 反解没有实数解，重新解会给一批已被标为有效的壳层填上 ``nan``。
    if "e" in data.files:
        energy = np.atleast_2d(np.asarray(data["e"], dtype=np.float64))
        if energy.shape != density.shape:
            raise DataError(
                f"{npz_path} carries 'e' with shape {energy.shape}, expected {density.shape}"
            )
        energy_source = "written by the extractor from the per-cell EOS inverse"
    else:
        energy = np.full_like(density, np.nan)
        for frame in range(times.size):
            energy[frame] = specific_energy_from_pressure(
                density[frame], pressure[frame], material[frame], gamma_air=gamma_air
            )
        energy_source = "reconstructed from (rho, p) via the material EOS"

    mixed = (
        np.asarray(block["mixed_cell_fraction"], dtype=np.float64)
        if "mixed_cell_fraction" in block.files
        else np.zeros(times.size)
    )
    return ExtractedCase(
        case_id=case_id or str(metadata.get("case_id", directory.name)),
        charge_radius=charge_radius,
        times=times,
        radius=radius,
        density=density,
        velocity=velocity,
        pressure=pressure,
        specific_energy=energy,
        contact_radius=contact_radius,
        shock_radius=shock_radius,
        contact_valid=contact_valid,
        shock_valid=shock_valid,
        material=material,
        valid=valid,
        source={
            "directory": str(directory),
            "npz": str(npz_path),
            "metadata": str(meta_path),
            "material_companion": str(companion),
            "field_map": mapping.mapping,
            "metadata_raw": dict(metadata),
            "material_masks_from": (
                "ALE material volume fraction (d3plot history variables 1..7); "
                "pure product alpha >= 0.95, pure air alpha <= 0.05, "
                "mixed band excluded from every label"
            ),
            # **机器可判的标志，取自提取器的元数据，不由读取器盖章。**
            #
            # 契约按这个布尔量判断分区是否权威（不按字符串匹配——"not from volume
            # fraction" 里也含 "volume fraction"，子串匹配会把否认读成肯定）。但
            # 读取器自己写一个 `True` 只是把判据从"信任说明文字"换成"信任读取器"：
            # 它并没有核实数据是怎么来的，只是照着"有这个文件"推了一句。
            # 提取器没留下这个标志时记 `None`，契约会因此拒绝——那是**未核实**，
            # 不是"通过"。
            "material_partition_is_volume_fraction_based": metadata.get(
                "material_partition_is_volume_fraction_based"
            ),
            "material_partition_source_recorded": metadata.get("material_partition_source"),
            "mixed_cell_fraction_max": float(np.max(mixed)) if mixed.size else 0.0,
            "energy_source": energy_source,
            "front_resolution": front_resolution,
            "front_eligibility_source": front_eligibility,
            # 方案 B 的时间原点 T_h = R_0/D + epsilon 需要卡片爆速。
            "detonation_velocity_m_s": metadata.get("detonation_velocity_m_s"),
            "frames_rejected_contact_below_charge_radius": int(
                np.count_nonzero(np.isfinite(contact_radius) & (contact_radius < charge_radius))
            ),
            # §8：球对称份额。写不出来就是写不出来——契约会因此判未就绪，而不是
            # 让下游自己猜一个 8。
            #
            # 三个别名**原样带出去**，一个都不合并：契约要能看见"两个别名互相矛盾"
            # 这件事。读取器在这里挑一个"胜出者"的话，冲突就永远到不了契约那一层。
            **{
                key: metadata.get(key)
                for key in ("symmetry_factor", "symmetry_share",
                            "symmetry_share_inverse", "full_sphere_factor")
            },
        },
        detector=DetectorSettings(),
        field_validity=_field_validity_from_companion(block, density.shape),
        front_supervision_eligibility={
            "source": front_eligibility,
            "level": eligibility_level,
            "contact_source": contact_source,
            "shock_source": shock_source,
            "contact_level": contact_eligibility_level,
            "shock_level": shock_eligibility_level,
            "contact_status": contact_state,
            "shock_status": shock_state,
            "contact_eligible": (
                None if contact_eligibility is None
                else np.asarray(contact_eligibility, dtype=bool)
            ),
            "shock_eligible": (
                None if shock_eligibility is None
                else np.asarray(shock_eligibility, dtype=bool)
            ),
            "ordering_ok": front_ordering_ok,
            "metadata": metadata.get("front_supervision_eligibility"),
        },
    )


def _field_validity_from_companion(block, shape) -> dict[str, np.ndarray] | None:
    """从配套文件里读逐字段有效性（§11.1）。

    缺任何一项就整体返回 ``None``：只给一半的字段有效性，会让调用方以为另一半
    也各自独立，而实际上它又要退回联合掩码——两种语义混在一起比完全没有更糟。

    **掩码必须是真正的布尔量。** 以前这里写的是
    ``np.asarray(block[key], dtype=bool)``——那一句会把 ``nan`` 变成 ``True``
    （``bool(nan)`` 为真），于是一张**全是 NaN** 的掩码读进来是**全有效**，而
    形状检查对此毫无办法。所以先验 dtype 与取值域，再转换。
    """
    names = {
        "density": "field_valid_density",
        "velocity": "field_valid_velocity",
        "specific_energy": "field_valid_energy",
        "pressure": "field_valid_pressure",
    }
    available = set(getattr(block, "files", ()))
    if not set(names.values()) <= available:
        return None
    out: dict[str, np.ndarray] = {}
    for logical, key in names.items():
        raw = np.atleast_2d(np.asarray(block[key]))
        if raw.shape != tuple(shape):
            raise DataError(
                f"the companion file carries {key} with shape {raw.shape}, expected "
                f"{tuple(shape)}; the per-field validity does not belong to these profiles"
            )
        if raw.dtype != np.bool_:
            raise DataError(
                f"the companion array {key!r} has dtype {raw.dtype}, not bool. Converting "
                "a float array with `dtype=bool` would turn every NaN into True, so an "
                "all-NaN mask would read back as 'everything valid'"
            )
        out[logical] = raw
    return out


def load_cases(specifications: Sequence[Mapping[str, Any]], **kwargs) -> list[ExtractedCase]:
    """按 ``[{case_id, directory, split}, ...]`` 批量读取。"""
    cases: list[ExtractedCase] = []
    for specification in specifications:
        case = load_extracted_case(
            specification["directory"],
            case_id=specification.get("case_id"),
            **kwargs,
        )
        case.source["split"] = specification.get("split")
        cases.append(case)
    return cases


__all__ = [
    "DEFAULT_FIELD_MAP",
    "DataError",
    "DetectorSettings",
    "ExtractedCase",
    "FieldMapping",
    "REQUIRED_FIELDS",
    "detect_contact_radius",
    "detect_shock_radius",
    "load_cases",
    "load_extracted_case",
    "partition_materials",
    "specific_energy_from_pressure",
]
