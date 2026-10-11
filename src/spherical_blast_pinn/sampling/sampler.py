"""按提示词 §12 采样。

三条规则，逐条对应规范里的条目。

**时间**：半数样本覆盖该半径的整个有效时间区间 ``[0, T_last_valid]``，
半数在 ``log(1 + hat_t / hat_t_eps)`` 上均匀采样，其中

.. math::  \\hat t_\\epsilon = \\frac{U_* \\epsilon(R_0)}{R_0}

是一个**与半径无关的固定值**——把 ``epsilon(R_0) = k epsilon_ref`` 代进去就
看到 ``k`` 约掉了。早期时间对数分布把采样密度放在激波刚形成、梯度最陡的
那一段，而不是均匀地浪费在整个区间上。

**空间**：产物与空气各 70% 均匀覆盖移动坐标内部，15% 取靠近左端的 10% 区间，
15% 取靠近右端的 10% 区间。PDE 点只在**开区间**内采样。

**不依赖模型**：所有 PDE 点直接采在 ``xi ∈ (0,1)``、``eta ∈ (0,1)`` 上。
好处是采样永远落在合法域内：一个还没训练好的前沿网络不可能把配置点采到域外，
因而也不可能出现"空域上的残差恰好是零、读起来像方程已满足"那种假收敛。
物理半径 ``r = R_c(s) xi`` 在损失里用模型自己的前沿算出来。
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

from ..data.extracted_case import ExtractedCase
from ..physics import material as mat
from ..physics.precursor import ReferenceField
from ..physics.similitude import SimilitudeScales
from .batch import SampleBatch

PRODUCT = "product"
AIR = "air"

#: 数据监督的字段：``(采样批次里的键, 算例上的逻辑字段名)``。
#: 逻辑字段名同时也用作 ``field_validity`` 的键，所以这一张表把"取数"、
#: "逐字段有效性"和"损失里的标签名"三处绑在同一个来源上。
_DATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("rho", "density"),
    ("u", "velocity"),
    ("e", "specific_energy"),
)


class SamplingError(ValueError):
    """采样配置或数据不合法。"""


@dataclass
class SamplingConfig:
    """每次参数更新、每个半径的样本数（提示词 §12 的表格）。"""

    n_pde_product: int = 1024
    n_pde_air: int = 1024
    n_ic_product: int = 256
    n_ic_air: int = 256
    n_contact: int = 128
    n_shock: int = 128
    n_data_product: int = 256
    n_data_air: int = 256
    n_data_front: int = 64

    #: 空间分层的比例：内侧 10% 区间、外侧 10% 区间各占多少。
    near_fraction: float = 0.15
    near_span: float = 0.10
    #: ``xi`` 的下界，对应轴心保护半径。``r = R_c xi >= R_0 xi_floor`` 时不会
    #: 触到 ``1/r`` 的奇点，因为 ``R_c >= R_0`` 本应由前沿网络保证。
    xi_floor: float = 1.0e-3

    def validate(self) -> None:
        for name in (
            "n_pde_product", "n_pde_air", "n_ic_product", "n_ic_air",
            "n_contact", "n_shock", "n_data_product", "n_data_air", "n_data_front",
        ):
            if int(getattr(self, name)) < 0:
                raise SamplingError(f"{name} must be non-negative")
        if not (0.0 <= self.near_fraction <= 1.0):
            raise SamplingError(f"near_fraction must lie in [0, 1], got {self.near_fraction}")
        if not (0.0 < self.near_span < 0.5):
            raise SamplingError(f"near_span must lie in (0, 0.5), got {self.near_span}")
        if not (0.0 < self.xi_floor < 0.5):
            raise SamplingError(f"xi_floor must lie in (0, 0.5), got {self.xi_floor}")

    @classmethod
    def from_config(cls, config) -> "SamplingConfig":
        section = config.section("sampling")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in section.items() if k in known}
        settings = cls(**payload)
        settings.validate()
        return settings


class RadiusSampler:
    """一个半径的采样器。"""

    def __init__(
        self,
        *,
        case: ExtractedCase,
        scales: SimilitudeScales,
        reference: ReferenceField | "_ScaledReference",
        config: SamplingConfig | None = None,
        seed: int = 0,
        dtype=None,
        device=None,
        r_min_fraction: float = 1.0e-2,
        use_development_frames: bool = False,
        dev_holdout_fraction: float = 0.1,
        split_seed: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        self.config = config or SamplingConfig()
        self.config.validate()
        self.case = case
        self.scales = scales
        self.reference = reference
        self.dtype = dtype or torch.float64
        self.device = device
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        self.r_min = float(r_min_fraction) * float(scales.charge_radius)
        self.use_development_frames = bool(use_development_frames)
        #: 数据监督用的是哪一张有效性掩码。逐字段记录缺失时退回联合掩码，这件事
        #: 必须能被读出来——"退化"和"等价"在报表上不能长得一样。
        self.validity_source = "unknown (no data sampled yet)"

        if abs(float(scales.charge_radius) - float(case.charge_radius)) > 1.0e-12:
            raise SamplingError(
                f"case {case.case_id!r} has R_0 = {case.charge_radius}, but the scales "
                f"were built for R_0 = {scales.charge_radius}"
            )

        usable = case.supervisable_frames()
        if not np.any(usable):
            raise SamplingError(
                f"case {case.case_id!r} has no supervisable frame; it cannot supervise "
                "a model"
            )
        # 数据划分只影响**数据监督**抽哪些帧：时间范围仍然由全部可监督帧给出
        # （提示词要求每个半径取自己的最后有效帧作为终止时间，那与开发集划分
        # 是两件事）。
        self.all_supervisable_frames = np.flatnonzero(usable)
        train_frames, dev_frames = case.frame_split(
            holdout_fraction=dev_holdout_fraction, seed=split_seed
        )
        chosen = dev_frames if self.use_development_frames else train_frames
        if chosen.size == 0:
            raise SamplingError(
                f"case {case.case_id!r} has no {'development' if self.use_development_frames else 'training'} "
                "frames after the data split; the data/development partition removed "
                "everything usable"
            )
        self.usable_frames = chosen
        self.development_frames = dev_frames
        # **完整**目标时段：本半径最后一个有效帧。它在构造时冻结，是这台采样器
        # 能覆盖的最大范围，也是阶段推进的上限。
        self.full_pinn_time_max = float(case.pinn_times[usable].max())
        self.pinn_time_min = 0.0
        # 当前**阶段窗口**（§14.2）。区间恒为 ``[0, window_pinn_time_max]``：
        # 扩大窗口只把右端往外推，**不动左端**，所以早期采样点永远留在池子里
        # ——这正是"保留早期采样"的可执行含义。左端若也随之移动，就等于把早期
        # 区间丢掉了，而表面上"时间范围在扩大"看不出问题。
        # ``window_fraction`` 的初值是完整时段，但**尚未提交**：采样器刚建好时
        # 不属于任何阶段。第一次 :meth:`set_time_window` 是"设定"，可以往任一
        # 方向；此后的每一次才是"推进"，只许扩大。若不区分，训练一开始把窗口设成
        # 阶段 1 的 25 % 会被当成"从 100 % 收窄"而拒绝。
        self.window_fraction = 1.0
        self.window_pinn_time_max = self.full_pinn_time_max
        self._window_committed = False
        self._window_history: list[float] = [1.0]
        #: 最近一次采样里落在窗口内的可监督帧数（诊断用）。
        self._window_frame_count = int(usable.size)

        # ``hat t_epsilon`` 与半径无关（见模块文档）。
        #
        # 这里要的是**交接延迟 epsilon** 的无量纲形式，所以必须调 ``start_delay()``。
        # 方案 A 时代没有 ``start_delay``，``start_time()`` 返回的就是 epsilon；
        # 改成方案 B 之后 ``start_time()`` 返回 ``T_h = T_b + epsilon``，在 50 mm 上
        # 是 epsilon 的 8.215 倍。两者都是正数、都是秒，写错了不会报错，只会把
        # 早期时间的对数分层刻度整体挪走。
        self.hat_time_epsilon = (
            scales.reference_velocity * scales.start_delay() / scales.charge_radius
        )
        if not math.isfinite(self.hat_time_epsilon) or self.hat_time_epsilon <= 0.0:
            raise SamplingError("hat_t_epsilon must be positive")

    # -- 逐字段有效性（§11.1）----------------------------------------------
    def _field_valid(self, case, frame: int, logical: str):
        """某个字段在第 ``frame`` 帧上可用的位置。

        有逐字段记录时用它；没有时退回联合掩码，并把这件事记在
        ``self.validity_source`` 里——退回是**退化**，不是等价。
        """
        validity = getattr(case, "field_validity", None)
        if validity is None:
            self.validity_source = "joint `valid` mask (per-field validity not recorded)"
            return np.asarray(case.valid[frame], dtype=bool)
        if logical not in validity:
            # 加载器要么给全四个字段、要么给 None；只给一半的映射说明它是别处
            # 拼出来的，此时退回联合掩码会让缺失的那一半静默变成"都有效"。
            raise SamplingError(
                f"case {case.case_id!r} carries per-field validity without {logical!r}; "
                f"it has {sorted(validity)}. A partial mapping is not a usable record"
            )
        self.validity_source = "per-field validity (field_validity)"
        return np.asarray(validity[logical][frame], dtype=bool)

    # -- 随机状态 ---------------------------------------------------------
    def rng_state(self) -> dict[str, Any]:
        """采样器自己的发生器的状态。

        ``default_rng()`` 建的生成器**不**受 ``np.random.get_state()`` 管，所以
        只保存全局 NumPy 状态再恢复，并不能让采样序列从断点继续——恢复出来的
        是一个不同的采样轨迹。这里把请求做成"中断后恢复到第 N 步"与"连续跑
        N 步"完全相同。
        """
        return {"seed": self.seed, "bit_generator": self.rng.bit_generator.state}

    def validate_rng_state(self, state: Any) -> None:
        """检查一份采样器随机状态**结构完整**，不修改任何东西。

        以前 ``set_rng_state`` 是"没有就返回"：空字典、缺 ``bit_generator``、
        类型不对——全都静默跳过。于是恢复出来的采样序列是**另一条轨迹**，而"恢复
        到第 N 步"与"连跑 N 步"的差别在损失曲线上完全看不出来。
        """
        if not isinstance(state, Mapping):
            raise SamplingError(
                f"case {self.case.case_id!r}: sampler RNG state must be a mapping, got "
                f"{type(state).__name__}"
            )
        if "seed" not in state:
            raise SamplingError(
                f"case {self.case.case_id!r}: sampler RNG state has no 'seed'"
            )
        payload = state.get("bit_generator")
        if not isinstance(payload, Mapping):
            raise SamplingError(
                f"case {self.case.case_id!r}: sampler RNG state has no 'bit_generator' "
                f"mapping (got {type(payload).__name__})"
            )
        if "state" not in payload or "bit_generator" not in payload:
            raise SamplingError(
                f"case {self.case.case_id!r}: the bit-generator state must carry both "
                f"'bit_generator' and 'state'; it carries {sorted(payload)}"
            )

    def set_rng_state(self, state: Mapping[str, Any] | None) -> None:
        self.validate_rng_state(state)
        self.rng.bit_generator.state = state["bit_generator"]

    # -- 时间窗口（§14.2）--------------------------------------------------
    def set_time_window(self, fraction: float) -> dict[str, float]:
        """把尺度化时间窗口设为完整有效时段的 ``fraction``。

        **只允许扩大**。窗口区间恒为 ``[0, W]``，所以 W 增大不会丢掉任何早期
        区间——从阶段 1 到阶段 3，早期采样点一直在池子里，这正是 §14 说的
        "保留早期采样"。

        收窄在这里被拒绝，而不是"照做但记一笔"：收窄会让已经用过的时段重新退出
        采样，早期样本被丢掉，而所有报表里只会看到一个更小的百分比，看起来一切
        正常。
        """
        value = float(fraction)
        if not math.isfinite(value) or not (0.0 < value <= 1.0):
            raise SamplingError(f"time window fraction must lie in (0, 1], got {fraction!r}")
        if self._window_committed and value < self.window_fraction - 1.0e-12:
            raise SamplingError(
                f"time window would shrink from {self.window_fraction} to {value}; "
                "§14 requires the window to expand while keeping early samples"
            )
        self._window_committed = True
        self.window_fraction = value
        self.window_pinn_time_max = value * self.full_pinn_time_max
        if self._window_history[-1] != value:
            self._window_history.append(value)
        return self.window_state()

    @property
    def window_frames(self) -> np.ndarray:
        """当前窗口内的可监督帧号。

        窗口只扩大不缩小、且区间恒为 ``[0, W]``，所以这个集合随阶段**只增不减**
        ——这正是"保留早期采样"可以逐帧核对的形式。
        """
        usable = self.usable_frames
        return usable[
            self.case.pinn_times[usable] <= self.window_pinn_time_max + 1.0e-12
        ]

    @property
    def window_history(self) -> list[float]:
        """窗口依次取过的值。回滚要连它一起还原（见 :meth:`restore_time_window`）。"""
        return list(self._window_history)

    def restore_time_window(
        self, fraction: float, *, history: Sequence[float] | None = None
    ) -> None:
        """**无条件**把窗口设成 ``fraction``，用于回滚。

        与 :meth:`set_time_window` 的区别是它**允许收窄**——这正是回滚需要的：
        一次失败的恢复要把窗口还原成加载前的值，而那个值可能比现在的小。所以它
        只给回滚用，调用点必须能说清楚"这是在撤销一次未完成的修改"。

        ``history`` 给出时连窗口历史一起还原。**回滚不得制造历史事件**：恢复流程
        先按检查点里的值调一次 :meth:`set_time_window`（它会往历史里追一条），
        随后某个后续步骤失败、再回滚——只还原窗口值而不还原历史，会在
        ``[1.0, 0.25]`` 后面留下一个第三项，而那次变化从来没有真正生效过。
        历史是"窗口实际取过哪些值"的记录，不是"曾经尝试过哪些值"。
        """
        value = float(fraction)
        if not math.isfinite(value) or not (0.0 < value <= 1.0):
            raise SamplingError(f"time window fraction must lie in (0, 1], got {fraction!r}")
        self._window_committed = True
        self.window_fraction = value
        self.window_pinn_time_max = value * self.full_pinn_time_max
        if history is not None:
            restored = [float(entry) for entry in history]
            if not restored:
                raise SamplingError("the window history to restore is empty")
            self._window_history = restored

    def window_state(self) -> dict[str, Any]:
        return {
            "fraction": float(self.window_fraction),
            "pinn_time_max": float(self.window_pinn_time_max),
            "full_pinn_time_max": float(self.full_pinn_time_max),
            "history": list(self._window_history),
        }

    # -- 时间 -------------------------------------------------------------
    def sample_network_time(self, count: int) -> np.ndarray:
        """``count`` 个 ``s`` 值。半数均匀、半数在 ``log(1 + hat t/hat t_eps)`` 上均匀。"""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        hat_max = self.scales.hat_time(self.window_pinn_time_max)
        half = count // 2
        uniform = self.rng.random(count - half) * hat_max
        span = math.log1p(hat_max / self.hat_time_epsilon)
        log_part = self.hat_time_epsilon * np.expm1(self.rng.random(half) * span)
        hat = np.concatenate([uniform, log_part])
        self.rng.shuffle(hat)
        return hat / self.scales.time_scale

    # -- 空间 -------------------------------------------------------------
    def _stratified_coordinate(self, count: int) -> np.ndarray:
        """在开区间 ``(xi_floor, 1)`` 上按 70/15/15 分层。"""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        lo = self.config.xi_floor
        span = 1.0 - lo
        n_near = int(round(count * self.config.near_fraction))
        n_near = min(n_near, count // 2)
        n_uniform = count - 2 * n_near
        parts = [
            lo + span * self.rng.random(n_uniform),
            lo + span * self.config.near_span * self.rng.random(n_near),
            lo + span * (1.0 - self.config.near_span)
            + span * self.config.near_span * self.rng.random(n_near),
        ]
        values = np.concatenate(parts)
        self.rng.shuffle(values)
        return np.clip(values, lo, 1.0 - 1.0e-9)

    # -- 张量 -------------------------------------------------------------
    def _column(self, values) -> Any:
        array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    # -- 初始条件 ---------------------------------------------------------
    def _initial_condition(self) -> dict[str, Any]:
        """由参考场在 ``t_PINN = 0`` 取值。

        ``t_PINN = tau = T - T_h``，所以这一点对应
        ``t_phys = T_h = R_0/D + epsilon(R_0)``，**不是**方案 A 的 ``epsilon``。
        """
        reference = self.reference
        start = self.scales.start_time()

        # ``reference`` 已经是缩放到本半径的视图，所以这里**不再**传 ``k``：
        # 传两次会把长度和时间各缩放一遍，而结果看起来仍然"像那么回事"。
        contact, shock = reference.contact_and_shock_at(start)
        contact = float(np.asarray(contact).reshape(-1)[0])
        shock = float(np.asarray(shock).reshape(-1)[0])
        if not (shock > contact > 0.0):
            raise SamplingError(
                f"the reference field gives R_c = {contact} m and R_s = {shock} m at "
                f"t = {start} s for R_0 = {self.scales.charge_radius} m; the air layer "
                "is not resolved"
            )

        rng = self.rng

        def draw(lower: float, upper: float, count: int) -> np.ndarray:
            """在 ``(lower, upper)`` 上按同样的分层规则取半径。"""
            if count <= 0 or not (upper > lower):
                return np.zeros(0)
            lo_fraction = self.config.xi_floor
            width = upper - lower
            n_near = min(int(round(count * self.config.near_fraction)), count // 2)
            n_uniform = count - 2 * n_near
            parts = [
                lower + width * (lo_fraction + (1.0 - lo_fraction) * rng.random(n_uniform)),
                lower + width * lo_fraction + width * self.config.near_span * rng.random(n_near),
                upper - width * self.config.near_span * rng.random(n_near),
            ]
            return np.concatenate(parts)

        r_product = draw(self.r_min, contact, self.config.n_ic_product)
        r_air = draw(contact, shock, self.config.n_ic_air)

        rho_p, u_p, e_p = reference.sample(r_product, np.full_like(r_product, start))
        rho_a, u_a, e_a = reference.sample(r_air, np.full_like(r_air, start))

        return {
            "contact_ic": contact,
            "shock_ic": shock,
            "xi_ic_product": self._column(r_product / contact),
            "rho_ic_product": self._column(rho_p),
            "u_ic_product": self._column(u_p),
            "e_ic_product": self._column(e_p),
            "eta_ic_air": self._column((r_air - contact) / (shock - contact)),
            "rho_ic_air": self._column(rho_a),
            "u_ic_air": self._column(u_a),
            "e_ic_air": self._column(e_a),
        }

    # -- 数据监督 ---------------------------------------------------------
    def _data_points(self) -> dict[str, Any]:
        """先按物理时间分层抽实际帧，再按材料抽点。

        §14.2：数据监督也只看**当前窗口**内的帧。窗口只扩大不缩小，所以早期帧
        一直留在池子里。窗口里一个可监督帧都没有时不是错误——阶段 1 是初值预
        训练，本来就可以没有数据项；此时 ``data_loss`` 返回一个不带梯度的零，
        有界梯度平衡会把它判为"本步不存在"并保持权重（§14）。
        """
        case = self.case
        frames = self.usable_frames[
            case.pinn_times[self.usable_frames] <= self.window_pinn_time_max + 1.0e-12
        ]
        self._window_frame_count = int(frames.size)
        rng = self.rng

        def pick_frames(count: int) -> np.ndarray:
            """在有效帧上分层抽样，返回**真实帧号**。

            把有效帧等分为 ``count`` 段、每段随机取一个**位置**，然后映射回
            ``frames[position]``。返回位置本身是一个不会报错的错误：位置 0 与
            帧号 0 都是合法的整数，于是它会静默地读第 0、1、2 帧——在前期帧被
            过滤掉、或中间有无效帧的数据上，这会把无效的早期帧放进监督，同时
            漏掉真正有效的后期帧。
            """
            if count <= 0 or frames.size == 0:
                return np.zeros(0, dtype=np.int64)
            count = min(int(count), int(frames.size))
            edges = np.linspace(0, frames.size, count + 1).astype(int)
            positions = []
            for index in range(count):
                low, high = int(edges[index]), int(edges[index + 1])
                if high <= low:
                    high = low + 1
                positions.append(int(rng.integers(low, high)))
            chosen = frames[np.asarray(positions, dtype=np.int64)]
            if not np.all(np.isin(chosen, frames)):
                # 结构上不可能：``frames`` 是自己的子集。留着是因为一旦有人把
                # 这里换回"位置"，这条断言就是唯一会发现它的地方。
                raise SamplingError("frame selection produced indices outside the usable set")
            self._sampled_frames.extend(int(f) for f in chosen)
            return np.asarray(chosen, dtype=np.int64)

        #: 径向网格的半宽。观测的**支持区**就是被采样单元本身，不是它中心那个点。
        half_width = 0.5 * float(np.median(np.diff(case.radius)))

        def collect(medium: int, count: int) -> dict[str, Any]:
            empty = {name: self._column(np.zeros(0)) for name in
                     ("s", "r", "lo", "hi", "rho", "u", "e", "count")}
            if count <= 0:
                return empty
            chosen = pick_frames(count)
            per_frame = max(1, int(math.ceil(count / max(chosen.size, 1))))
            s_values: list[float] = []
            r_values: list[float] = []
            lo_values: list[float] = []
            hi_values: list[float] = []
            count_values: list[float] = []
            targets: dict[str, list[float]] = {"rho": [], "u": [], "e": []}
            for frame in chosen:
                # **逐字段**的有效性（§11.1）。以前这里用的是
                # ``(case.material[frame] == medium) & case.valid[frame]``——那张
                # ``valid`` 是"四项场量联合可用"的掩码，于是"某个字段缺失不丢掉其他
                # 字段"这句承诺在**取数的地方**根本没兑现：一个位置只要 e 不可用就
                # 整点消失，rho 与 u 一起被丢。提取器把逐字段掩码写出来了，这里必须
                # 真的用它，否则那几张掩码只是让契约检查变绿。
                medium_mask = case.material[frame] == medium
                allowed = {
                    key: medium_mask & self._field_valid(case, frame, logical)
                    for key, logical in _DATA_FIELDS
                }
                union = np.zeros_like(medium_mask)
                for mask in allowed.values():
                    union |= mask
                indices = np.flatnonzero(union)
                if indices.size == 0:
                    continue
                take = min(per_frame, indices.size)
                picked = rng.choice(indices, size=take, replace=False)
                radii = case.radius[picked]
                values = case.sample_at(int(frame), radii)
                # 逐字段掩码：某个字段在这个位置上不可用时只把它置 NaN，位置与其余
                # 字段照常保留。损失里的按字段掩码均值会跳过这些 NaN。
                for key, logical in _DATA_FIELDS:
                    here = np.asarray(allowed[key][picked], dtype=bool)
                    if not here.all():
                        values[logical] = np.where(
                            here, np.asarray(values[logical], dtype=np.float64), np.nan
                        )
                physical_time = float(case.times[frame])
                if physical_time < self.scales.start_time() - 1.0e-15:
                    raise SamplingError(
                        f"case {case.case_id!r} frame {int(frame)} is at t = "
                        f"{physical_time:.6g} s, before the start time "
                        f"{self.scales.start_time():.6g} s; a supervision sample may "
                        "not precede the PINN time origin"
                    )
                s_here = self.scales.network_time(
                    physical_time - self.scales.start_time()
                )
                n_here = radii.size
                s_values.extend([s_here] * n_here)
                r_values.extend(radii.tolist())
                # 支持区 = 被采样单元本身（§12）。标签是**单元平均值**，不是中心点
                # 值；模型侧必须在同一段支持上做同样的几何加权平均，否则两边比的
                # 不是同一个观测。
                lo_values.extend((radii - half_width).clip(case.radius[0]).tolist())
                hi_values.extend((radii + half_width).clip(case.radius[-1]).tolist())
                # 该观测由几个原始单元贡献。覆盖率用它，不拿"点了几个点"冒充。
                count_values.extend([1.0] * n_here)
                for key, field_name in _DATA_FIELDS:
                    targets[key].extend(
                        np.asarray(values[field_name], dtype=np.float64).tolist()
                    )

            if not r_values:
                return empty

            # 每个字段**独立**保留（§11.1）：一个字段缺失（NaN）不丢掉同一位置
            # 其他可靠的字段。位置 (s, r) 保留"至少一个字段有限"的点，rho/u/e 各自
            # 只保证自己的有限性——缺失处留 NaN，由损失里的按字段掩码均值跳过。
            # 以前是 ``keep &= isfinite`` 对 rho/u/e **全部**做与，任一字段 NaN 就
            # 把整点丢掉（"一个 nan 标签在均值里会被吞掉"的旧注释反过来把它当成了
            # 需要整点删除的理由）。
            rho = np.asarray(targets["rho"], dtype=np.float64)
            u = np.asarray(targets["u"], dtype=np.float64)
            e = np.asarray(targets["e"], dtype=np.float64)
            s_values = np.asarray(s_values, dtype=np.float64)
            r_values = np.asarray(r_values, dtype=np.float64)
            keep = (
                np.isfinite(s_values)
                & np.isfinite(r_values)
                & (np.isfinite(rho) | np.isfinite(u) | np.isfinite(e))
            )
            return {
                "s": self._column(s_values[keep]),
                "r": self._column(r_values[keep]),
                "lo": self._column(np.asarray(lo_values)[keep]),
                "hi": self._column(np.asarray(hi_values)[keep]),
                "count": self._column(np.asarray(count_values)[keep]),
                "rho": self._column(rho[keep]),
                "u": self._column(u[keep]),
                "e": self._column(e[keep]),
            }

        product = collect(0, self.config.n_data_product)
        air = collect(1, self.config.n_data_air)

        # 前沿标签：contact 与 shock **独立**采样（§11.1）。一个前沿有效不要求
        # 另一个也有效——以前 ``contact_valid AND shock_valid`` 才留帧，等于把
        # "只有接触面可用"或"只有激波可用"的整帧都丢掉，两套前沿被绑死。
        contact_frames = pick_frames(self.config.n_data_front)
        s_contact, contact = [], []
        for frame in contact_frames:
            if not case.contact_valid[frame]:
                continue
            physical_time = float(case.times[frame])
            s_contact.append(
                self.scales.network_time(physical_time - self.scales.start_time())
            )
            contact.append(float(case.contact_radius[frame]))
        shock_frames = pick_frames(self.config.n_data_front)
        s_shock, shock = [], []
        for frame in shock_frames:
            if not case.shock_valid[frame]:
                continue
            physical_time = float(case.times[frame])
            s_shock.append(
                self.scales.network_time(physical_time - self.scales.start_time())
            )
            shock.append(float(case.shock_radius[frame]))

        # §11.2 的冻结位置不确定性。先算再验：delta 必须是**有限正数**，否则
        # 由它定义的几何约束根本无从谈起。
        contact_delta = float(case.contact_position_uncertainty())
        shock_delta = float(case.shock_position_uncertainty())
        for label, value in (("contact", contact_delta), ("shock", shock_delta)):
            if not (math.isfinite(value) and value > 0.0):
                raise SamplingError(
                    f"case {case.case_id!r}: the {label} position uncertainty is {value!r}, "
                    "so L_geometry (§11.2) has no defined tolerance. This comes from the "
                    "radial grid spacing; a case with fewer than two usable radii cannot "
                    "define it. Refusing rather than silently dropping the constraint."
                )

        return {
            "s_data_product": product["s"],
            "r_data_product": product["r"],
            "rho_data_product": product["rho"],
            "u_data_product": product["u"],
            "e_data_product": product["e"],
            # §12 观测支持区：标签是这一段半径上的**平均值**，不是中心点值。
            "lo_data_product": product["lo"],
            "hi_data_product": product["hi"],
            "coverage_data_product": product["count"],
            "s_data_air": air["s"],
            "r_data_air": air["r"],
            "rho_data_air": air["rho"],
            "u_data_air": air["u"],
            "e_data_air": air["e"],
            "lo_data_air": air["lo"],
            "hi_data_air": air["hi"],
            "coverage_data_air": air["count"],
            "s_data_contact": self._column(s_contact),
            "contact_data": self._column(contact),
            "s_data_shock": self._column(s_shock),
            "shock_data": self._column(shock),
            # §11.2 的 delta：来自冻结的位置不确定性，不随训练变化。
            # 这里**当场校验**，不留到损失里再判。理由是：``geometry_loss`` 拿不
            # 到有限 delta 时只能返回 0（它没有别的选择），于是一个被单独加权、
            # 本该生效的约束会**无声消失**——正是提示词反复禁止的那种"约束被吞掉"。
            # 而在源头它是有明确错误的：算例没有可用径向网格，位置不确定性无从
            # 定义。宁可在这里硬失败。
            "geometry_delta_contact": contact_delta,
            "geometry_delta_shock": shock_delta,
        }

    # -- 组装 -------------------------------------------------------------
    def batch(self) -> SampleBatch:
        config = self.config
        self._sampled_frames: list[int] = []
        s_pde_product = self.sample_network_time(config.n_pde_product)
        s_pde_air = self.sample_network_time(config.n_pde_air)
        s_contact = self.sample_network_time(config.n_contact)
        s_shock = self.sample_network_time(config.n_shock)

        payload = {
            "case_id": self.case.case_id,
            "charge_radius": float(self.scales.charge_radius),
            "detonation_velocity": float(self.case.detonation_velocity),
            "scales": self.scales,
            "pinn_time_min": self.pinn_time_min,
            # 这是**本阶段窗口**的上端，不是完整时段：检查集与守恒诊断都用它，
            # 若在这里写完整时段，窗口内的验收会被拿到窗口外的点上评价。
            "pinn_time_max": self.window_pinn_time_max,
            "s_product": self._column(s_pde_product),
            "xi_product": self._column(self._stratified_coordinate(config.n_pde_product)),
            "s_air": self._column(s_pde_air),
            "eta_air": self._column(self._stratified_coordinate(config.n_pde_air)),
            "s_contact": self._column(s_contact),
            "s_shock": self._column(s_shock),
        }
        payload.update(self._initial_condition())
        payload.update(self._data_points())
        payload["diagnostics"] = {
            "r_min": self.r_min,
            "hat_time_epsilon": self.hat_time_epsilon,
            "n_supervisable_frames": int(self.all_supervisable_frames.size),
            "n_sampled_frames": int(self.usable_frames.size),
            "n_development_frames": int(self.development_frames.size),
            "sampled_frames": sorted({int(f) for f in self._sampled_frames}),
            "pinn_time_max": self.window_pinn_time_max,
            "full_pinn_time_max": self.full_pinn_time_max,
            "time_window_fraction": self.window_fraction,
            "n_window_frames": int(getattr(self, "_window_frame_count", 0)),
            "data_split": "development" if self.use_development_frames else "train",
            "data_validity_source": self.validity_source,
        }
        return SampleBatch(**payload)


class GroupedSampler:
    """三个半径各一个 :class:`RadiusSampler`，每次更新返回三个 batch。"""

    def __init__(
        self,
        samplers: Sequence[RadiusSampler],
        *,
        base_seed: int = 0,
    ) -> None:
        if not samplers:
            raise SamplingError("GroupedSampler needs at least one radius sampler")
        self.samplers = list(samplers)
        self.base_seed = int(base_seed)
        self.step = 0

    @classmethod
    def build(
        cls,
        *,
        cases: Sequence[ExtractedCase],
        scales_by_radius: Mapping[float, SimilitudeScales],
        reference: ReferenceField,
        config: SamplingConfig | None = None,
        base_seed: int = 0,
        dtype=None,
        device=None,
        r_min_fraction: float = 1.0e-2,
        use_development_frames: bool = False,
        dev_holdout_fraction: float = 0.1,
        split_seed: int = 0,
    ) -> "GroupedSampler":
        samplers = []
        for case in cases:
            scales = scales_by_radius[float(case.charge_radius)]
            samplers.append(
                RadiusSampler(
                    case=case,
                    scales=scales,
                    reference=reference.scale_to(case.charge_radius),
                    config=config,
                    seed=base_seed + 1000 * (len(samplers) + 1),
                    dtype=dtype,
                    device=device,
                    r_min_fraction=r_min_fraction,
                    use_development_frames=use_development_frames,
                    dev_holdout_fraction=dev_holdout_fraction,
                    split_seed=split_seed,
                )
            )
        return cls(samplers, base_seed=base_seed)

    @property
    def case_ids(self) -> list[str]:
        return [sampler.case.case_id for sampler in self.samplers]

    def batch(self) -> list[SampleBatch]:
        self.step += 1
        return [sampler.batch() for sampler in self.samplers]

    # -- 时间窗口（§14.2）--------------------------------------------------
    def set_time_window(self, fraction: float) -> dict[str, Any]:
        """把**全部**半径的窗口一起设置。§14 要求"同一套网络同步扩展"。"""
        per_case = {s.case.case_id: s.set_time_window(fraction) for s in self.samplers}
        return {"fraction": float(fraction), "per_case": per_case}

    def time_window_state(self) -> dict[str, Any]:
        return {
            "fraction": (
                float(self.samplers[0].window_fraction) if self.samplers else 1.0
            ),
            "per_case": {
                s.case.case_id: s.window_state() for s in self.samplers
            },
        }

    # -- 随机状态 ---------------------------------------------------------
    def rng_state(self) -> dict[str, Any]:
        """每个采样器的发生器状态。检查点必须连它一起存，否则恢复训练会换掉
        采样轨迹（见 :meth:`RadiusSampler.rng_state`）。"""
        return {
            "version": 1,
            "samplers": {
                sampler.case.case_id: sampler.rng_state() for sampler in self.samplers
            },
        }

    #: 采样器随机状态的结构版本。
    RNG_STATE_VERSION = 1

    def validate_rng_state(self, payload: Any) -> None:
        """整份采样器随机状态的**结构**校验，不改任何东西。

        要求：是一个带版本的映射；带 ``samplers`` 子映射；**每一个**本组的算例
        都在里面。缺一个算例就等于那台采样器会从别处继续——而它同样不会报错。
        """
        if not isinstance(payload, Mapping):
            raise SamplingError(
                f"the sampler RNG state must be a mapping, got {type(payload).__name__}"
            )
        if int(payload.get("version", -1)) != self.RNG_STATE_VERSION:
            raise SamplingError(
                f"the sampler RNG state has version {payload.get('version')!r}, "
                f"expected {self.RNG_STATE_VERSION}"
            )
        stored = payload.get("samplers")
        if not isinstance(stored, Mapping):
            raise SamplingError(
                "the sampler RNG state has no 'samplers' mapping; restoring from it "
                "would leave every sampler on a different trajectory without saying so"
            )
        missing = [
            sampler.case.case_id for sampler in self.samplers
            if sampler.case.case_id not in stored
        ]
        if missing:
            raise SamplingError(
                f"the sampler RNG state is missing {missing}; every case that will be "
                "sampled must carry its own state"
            )
        for sampler in self.samplers:
            sampler.validate_rng_state(stored[sampler.case.case_id])

    def set_rng_state(self, payload: Mapping[str, Any] | None) -> None:
        self.validate_rng_state(payload)
        stored = payload["samplers"]
        for sampler in self.samplers:
            sampler.set_rng_state(stored[sampler.case.case_id])

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_ids": self.case_ids,
            "n_radii": len(self.samplers),
            "pinn_time_max": {
                sampler.case.case_id: sampler.window_pinn_time_max
                for sampler in self.samplers
            },
            "full_pinn_time_max": {
                sampler.case.case_id: sampler.full_pinn_time_max
                for sampler in self.samplers
            },
            "time_window_fraction": (
                float(self.samplers[0].window_fraction) if self.samplers else 1.0
            ),
            "data_split": (
                "development" if self.samplers and self.samplers[0].use_development_frames
                else "train"
            ),
            "frames": {
                sampler.case.case_id: {
                    "n_supervisable": int(sampler.all_supervisable_frames.size),
                    "n_sampled": int(sampler.usable_frames.size),
                    "n_development": int(sampler.development_frames.size),
                    "sampled_frame_numbers": [int(f) for f in sampler.usable_frames],
                    "development_frame_numbers": [int(f) for f in sampler.development_frames],
                }
                for sampler in self.samplers
            },
        }


__all__ = [
    "AIR",
    "PRODUCT",
    "GroupedSampler",
    "RadiusSampler",
    "SamplingConfig",
    "SamplingError",
]
