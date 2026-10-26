"""按提示词 §12 采样：新启动场上的物理点批次。

采样器的输入只有三样：一个通过校验的 :class:`PlanarStartupField`、它的
:class:`PlanarStartupScales`，以及从网络启动时刻算起的持续时间 ``tau_end``
（秒）。**没有参考场参数，也没有数据监督。**

**时间**：半数样本覆盖 ``[0, tau_end]``，半数在
``log(1 + hat_t / hat_t_eps)`` 上均匀采样，其中

.. math::  \\hat t_\\epsilon = \\frac{U_* \\epsilon_{\\rm field}}{R_0}

``epsilon_field`` 是启动场自己的物理时刻，经**新尺度** ``hat_time`` 换算；
它不是时间原点，不再与 ``T_h`` 相加。``tau_end`` 是相对网络启动时刻的时长，
所以 ``pinn_time_max`` 与批次的 ``s`` 上限都从 ``tau = 0`` 起算。

**空间**：产物与空气各 70% 均匀覆盖移动坐标内部，15% 取靠近左端的 10% 区间，
15% 取靠近右端的 10% 区间。PDE 点只在**开区间**内采样，物理半径
``r = R_c(s) xi`` 在损失里用模型自己的前沿算出来。

**初始条件**：复用 :func:`build_planar_startup_initial_batch`，不复制解析采样
和单侧边界规则。产物/空气的初值半径由本采样器的分层规则给出，锚点（球心、
接触面两侧、波后）由适配器补齐。

**数据监督**：尚未迁移到 ``tau = t_phys - epsilon``。传入 ``case`` 会在构造时
明确报未支持，而不是继续沿用 ``case.pinn_times``（旧 ``T_h`` 时间轴）。不构造
假算例、不伪造观测：无数据时数据与几何监督批次为空，其它物理点照常生成。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..data.extracted_case import ExtractedCase
from ..physics.similitude import PlanarStartupScales
from ..physics.two_medium_startup_planar import PlanarStartupField
from .batch import SampleBatch
from .planar_startup_initial import build_planar_startup_initial_batch

PRODUCT = "product"
AIR = "air"


class SamplingError(ValueError):
    """采样配置、启动场或输入不合法。"""


@dataclass
class SamplingConfig:
    """每次参数更新、每个半径的样本数（提示词 §12 的表格）。"""

    n_pde_product: int = 1024
    n_pde_air: int = 1024
    n_ic_product: int = 256
    n_ic_air: int = 256
    n_contact: int = 128
    n_shock: int = 128

    #: 空间分层的比例：内侧 10% 区间、外侧 10% 区间各占多少。
    near_fraction: float = 0.15
    near_span: float = 0.10
    #: ``xi`` 的下界，对应轴心保护半径。``r = R_c xi >= R_0 xi_floor`` 时不会
    #: 触到 ``1/r`` 的奇点，因为 ``R_c >= R_0`` 本应由前沿网络保证。
    xi_floor: float = 1.0e-3

    def validate(self) -> None:
        for name in (
            "n_pde_product", "n_pde_air", "n_ic_product", "n_ic_air",
            "n_contact", "n_shock",
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


def _positive_finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SamplingError(f"{name} must be finite and positive, got {value!r}")
    return number


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-12 * max(
        abs(float(left)), abs(float(right)), 1.0e-30
    )


def validate_time_window_record(state: Any) -> dict[str, Any]:
    """校验一份窗口记录**内部自洽**，返回归一化后的字段。

    只查记录本身：分数在 ``(0, 1]``、窗口上端等于分数乘完整时段、历史非空且
    只增不减、最后一项等于当前分数、"未提交"时仍是完整时段。它**不**知道
    ``tau_end`` 是多少——那是采样器的属性，由 :meth:`RadiusSampler.validate_window_state`
    再核对。检查点的静态配置与动态窗口都用这一份定义。
    """
    if not isinstance(state, Mapping):
        raise SamplingError(
            f"the time window state must be a mapping, got {type(state).__name__}"
        )
    for key in ("fraction", "pinn_time_max", "full_pinn_time_max", "history", "committed"):
        if key not in state:
            raise SamplingError(f"the time window state has no {key!r}")

    def number(key: str) -> float:
        value = state[key]
        if isinstance(value, (bool, np.bool_)):
            raise SamplingError(f"time window {key!r} is {value!r}, not a number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise SamplingError(
                f"time window {key!r} is {value!r}, not a number"
            ) from exc
        if not math.isfinite(result):
            raise SamplingError(f"time window {key!r} is not finite ({value!r})")
        return result

    fraction = number("fraction")
    if not (0.0 < fraction <= 1.0):
        raise SamplingError(
            f"time window fraction must lie in (0, 1], got {fraction!r}"
        )
    full = number("full_pinn_time_max")
    if full <= 0.0:
        raise SamplingError(f"the full time window must be positive, got {full!r}")
    pinn_time_max = number("pinn_time_max")
    if not _close(pinn_time_max, fraction * full):
        raise SamplingError(
            f"the saved window top is {pinn_time_max!r} s, but fraction {fraction!r} "
            f"of {full!r} s is {fraction * full!r} s"
        )

    committed = state["committed"]
    if not isinstance(committed, (bool, np.bool_)):
        raise SamplingError(
            f"time window 'committed' is {committed!r}, not a boolean"
        )
    history = state["history"]
    if isinstance(history, (str, bytes)) or not isinstance(history, (list, tuple)):
        raise SamplingError(
            f"the time window history is {type(history).__name__}, not a list"
        )
    if not history:
        raise SamplingError("the time window history is empty")
    values: list[float] = []
    for index, entry in enumerate(history):
        if isinstance(entry, (bool, np.bool_)):
            raise SamplingError(f"time window history[{index}] is {entry!r}, not a number")
        try:
            value = float(entry)
        except (TypeError, ValueError) as exc:
            raise SamplingError(
                f"time window history[{index}] is {entry!r}, not a number"
            ) from exc
        if not math.isfinite(value) or not (0.0 < value <= 1.0):
            raise SamplingError(
                f"time window history[{index}] is {entry!r}, outside (0, 1]"
            )
        values.append(value)
    # 首项是 `[1.0]` 的占位（"尚未设定"），第一次 ``set_time_window`` 允许把它
    # 改成阶段 1 的较小值；**从第二项起**只许扩大。
    if any(later < earlier - 1.0e-12 for earlier, later in zip(values[1:], values[2:])):
        raise SamplingError(
            f"the time window history {values!r} shrinks after it was committed; §14 "
            "requires the window to expand while keeping early samples"
        )
    if not _close(values[-1], fraction):
        raise SamplingError(
            f"the last window history entry is {values[-1]!r}, but the current fraction "
            f"is {fraction!r}"
        )
    if not bool(committed) and (
        len(values) != 1 or not _close(values[0], 1.0) or not _close(fraction, 1.0)
    ):
        raise SamplingError(
            "an uncommitted time window must still be the full period (fraction 1.0, "
            f"history [1.0]), got fraction {fraction!r} and history {values!r}"
        )
    return {
        "fraction": fraction,
        "pinn_time_max": pinn_time_max,
        "full_pinn_time_max": full,
        "history": values,
        "committed": bool(committed),
    }


class RadiusSampler:
    """一个半径的物理点采样器（新启动场，无数据监督）。

    参数
    ----
    field:
        通过一致性校验的 :class:`PlanarStartupField`。
    scales:
        与 ``field`` 同半径、同 ``epsilon`` 的 :class:`PlanarStartupScales`。
    tau_end:
        从网络启动时刻（``s = 0``，``t_phys = epsilon_field``）算起的持续时间，
        秒。必须是有限正数；**不**从任何数据帧隐式推导。
    case:
        只有 ``None`` 受支持。真实数据的时间轴还没有迁移到
        ``tau = t_phys - epsilon``，传入算例会明确报未支持。
    """

    def __init__(
        self,
        *,
        field: PlanarStartupField,
        scales: PlanarStartupScales,
        tau_end: float,
        case: ExtractedCase | None = None,
        case_id: str | None = None,
        config: SamplingConfig | None = None,
        seed: int = 0,
        dtype=None,
        device=None,
        r_min_fraction: float = 1.0e-2,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        if not isinstance(field, PlanarStartupField):
            raise SamplingError(
                f"field must be a PlanarStartupField, got {type(field).__name__}"
            )
        if not isinstance(scales, PlanarStartupScales):
            raise SamplingError(
                "scales must be PlanarStartupScales (time origin = the field's own "
                f"epsilon), got {type(scales).__name__}"
            )
        if field.checks.get("passed") is not True:
            raise SamplingError(
                "the startup field has no passed consistency report; refusing to sample it"
            )
        if case is not None:
            raise SamplingError(
                "real-data supervision is not supported yet: the data time axis is still "
                "t_phys, not tau = t_phys - epsilon, so case.pinn_times cannot be reused. "
                "Pass case=None for the physics-only batch; re-extraction onto the new "
                "origin is a later task."
            )

        self.field = field
        self.scales = scales
        self.config = config or SamplingConfig()
        self.config.validate()
        if self.config.n_ic_product < 1 or self.config.n_ic_air < 1:
            raise SamplingError(
                "n_ic_product and n_ic_air must both be >= 1: the initial condition is "
                "taken from the startup field and is always part of the batch"
            )
        self.case_id = self._case_label(case_id)
        self.tau_end = _positive_finite("tau_end", tau_end)
        self.dtype = dtype or torch.float64
        self.device = device
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        fraction = float(r_min_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise SamplingError(
                f"r_min_fraction must lie in (0, 1), got {r_min_fraction!r}"
            )
        self.r_min_fraction = fraction
        self.r_min = fraction * float(scales.charge_radius)

        if abs(float(scales.charge_radius) - float(field.charge_radius)) > 1.0e-12 * max(
            abs(float(field.charge_radius)), 1.0
        ):
            raise SamplingError(
                f"the scales belong to R_0 = {scales.charge_radius!r} m, but the startup "
                f"field has R_0 = {field.charge_radius!r} m"
            )
        if abs(float(scales.startup_epsilon) - float(field.epsilon)) > 1.0e-12 * max(
            abs(float(field.epsilon)), 1.0e-30
        ):
            raise SamplingError(
                f"the scales start at epsilon = {scales.startup_epsilon!r} s, but the "
                f"startup field is at epsilon = {field.epsilon!r} s"
            )

        # 阶段窗口（§14.2）：区间恒为 ``[0, W]``，扩大只把右端往外推。
        # ``window_fraction`` 的初值是完整时段，但**尚未提交**；第一次
        # ``set_time_window`` 是"设定"，此后的每一次才是"推进"，只许扩大。
        self.full_pinn_time_max = self.tau_end
        self.pinn_time_min = 0.0
        self.window_fraction = 1.0
        self.window_pinn_time_max = self.tau_end
        self._window_committed = False
        self._window_history: list[float] = [1.0]

        # 早期对数分层用的特征时间：启动场自己的 epsilon。``hat_time`` 只做
        # ``U_* tau / R_0``，不加任何时间原点，所以这里不会重复平移。
        self.hat_time_epsilon = float(self.scales.hat_time(float(field.epsilon)))
        if not math.isfinite(self.hat_time_epsilon) or self.hat_time_epsilon <= 0.0:
            raise SamplingError("hat_t_epsilon must be finite and positive")

    def _case_label(self, case_id: str | None) -> str:
        if case_id is None:
            return f"planar_startup:{self.field.cache_key()[:16]}"
        if not isinstance(case_id, str) or not case_id:
            raise SamplingError("case_id must be a non-empty string")
        return case_id

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
        """检查一份采样器随机状态**结构完整**，不修改任何东西。"""
        if not isinstance(state, Mapping):
            raise SamplingError(
                f"case {self.case_id!r}: sampler RNG state must be a mapping, got "
                f"{type(state).__name__}"
            )
        if "seed" not in state:
            raise SamplingError(
                f"case {self.case_id!r}: sampler RNG state has no 'seed'"
            )
        payload = state.get("bit_generator")
        if not isinstance(payload, Mapping):
            raise SamplingError(
                f"case {self.case_id!r}: sampler RNG state has no 'bit_generator' "
                f"mapping (got {type(payload).__name__})"
            )
        if "state" not in payload or "bit_generator" not in payload:
            raise SamplingError(
                f"case {self.case_id!r}: the bit-generator state must carry both "
                f"'bit_generator' and 'state'; it carries {sorted(payload)}"
            )

    def set_rng_state(self, state: Mapping[str, Any] | None) -> None:
        self.validate_rng_state(state)
        # ``seed`` 是状态记录的一部分：只恢复位发生器而留下另一个种子，会让
        # ``rng_state()`` 与保存的那份不再相等，检查点往返就核对不上。
        self.seed = int(state["seed"])
        self.rng.bit_generator.state = state["bit_generator"]

    # -- 时间窗口（§14.2）--------------------------------------------------
    def set_time_window(self, fraction: float) -> dict[str, float]:
        """把尺度化时间窗口设为完整时段的 ``fraction``。

        **只允许扩大**。窗口区间恒为 ``[0, W]``，所以 W 增大不会丢掉任何早期
        区间——从阶段 1 到阶段 3，早期采样点一直在池子里。
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
    def window_history(self) -> list[float]:
        """窗口依次取过的值。回滚要连它一起还原（见 :meth:`restore_time_window`）。"""
        return list(self._window_history)

    def restore_time_window(
        self, fraction: float, *, history: Sequence[float] | None = None
    ) -> None:
        """**无条件**把窗口设成 ``fraction``，用于回滚。

        与 :meth:`set_time_window` 的区别是它**允许收窄**——这正是回滚需要的：
        一次失败的恢复要把窗口还原成加载前的值。``history`` 给出时连窗口历史
        一起还原；**回滚不得制造历史事件**。
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
            # "还没设定过窗口"与"窗口恰好是 1.0"不是同一件事：前者第一次
            # ``set_time_window`` 只是设定，后者之后只许扩大。恢复时必须保留。
            "committed": bool(self._window_committed),
        }

    def _validated_window_record(self, state: Any) -> dict[str, Any]:
        record = validate_time_window_record(state)
        if not _close(record["full_pinn_time_max"], float(self.tau_end)):
            raise SamplingError(
                f"case {self.case_id!r}: the saved full window is "
                f"{record['full_pinn_time_max']!r} s, but this sampler's tau_end is "
                f"{self.tau_end!r} s; the same window fraction would cover a different "
                "physical duration"
            )
        return record

    def validate_window_state(self, state: Any) -> None:
        """检查一份窗口状态**自洽**且属于这个采样器，不修改任何东西。

        窗口是**动态状态**：它决定接下来采哪些时间点。只恢复 RNG 而不恢复窗口，
        "从第 N 步继续"会换成另一段时间范围；只恢复窗口而不核对 ``tau_end``，
        同一个分数会对应另一个物理时长。
        """
        self._validated_window_record(state)

    def set_window_state(self, state: Mapping[str, Any]) -> None:
        """按检查点**无条件**恢复窗口（允许相对当前值收窄）。

        与 :meth:`set_time_window` 的区别：这里恢复的是"当时的状态"，不是推进
        阶段。校验先做；校验失败时窗口保持原样。
        """
        record = self._validated_window_record(state)
        self.restore_time_window(
            record["fraction"], history=record["history"]
        )
        self._window_committed = record["committed"]

    # -- 采样配置（检查点绑定用）------------------------------------------
    def sampling_config_state(self) -> dict[str, Any]:
        """**实际影响采样点**的配置，加上当前窗口。

        检查点用它把"这一份是在什么采样设置下跑出来的"记录并核对。``tau_end``、
        点数与分层比例任何一项不同，采样分布就不同；拿着另一套设置继续跑，不能
        称为精确续训。
        """
        config = self.config
        return {
            "tau_end_s": float(self.tau_end),
            "r_min_fraction": float(self.r_min_fraction),
            "counts": {
                "n_pde_product": int(config.n_pde_product),
                "n_pde_air": int(config.n_pde_air),
                "n_ic_product": int(config.n_ic_product),
                "n_ic_air": int(config.n_ic_air),
                "n_contact": int(config.n_contact),
                "n_shock": int(config.n_shock),
            },
            "stratification": {
                "near_fraction": float(config.near_fraction),
                "near_span": float(config.near_span),
                "xi_floor": float(config.xi_floor),
            },
            "window": self.window_state(),
        }

    # -- 时间 -------------------------------------------------------------
    def sample_network_time(self, count: int) -> np.ndarray:
        """``count`` 个 ``s`` 值。半数均匀、半数在 ``log(1 + hat t/hat t_eps)`` 上均匀。

        ``hat_t`` 对应 ``tau``（从网络启动时刻算起的时长），上限是当前窗口；
        ``hat_t_eps`` 来自启动场的 ``epsilon``，两者都是时长，不与时间原点相加。
        """
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
        """复用初值适配器，只提供本采样器的分层半径。"""
        contact = float(self.field.contact_radius)
        shock = float(self.field.shock_radius)

        def draw(lower: float, upper: float, count: int) -> np.ndarray:
            """在 ``(lower, upper)`` 上按同样的分层规则取半径。"""
            width = upper - lower
            lo_fraction = self.config.xi_floor
            n_near = min(int(round(count * self.config.near_fraction)), count // 2)
            n_uniform = count - 2 * n_near
            parts = [
                lower + width * (lo_fraction + (1.0 - lo_fraction) * self.rng.random(n_uniform)),
                lower + width * lo_fraction + width * self.config.near_span * self.rng.random(n_near),
                upper - width * self.config.near_span * self.rng.random(n_near),
            ]
            return np.concatenate(parts)

        r_product = draw(self.r_min, contact, self.config.n_ic_product)
        r_air = draw(contact, shock, self.config.n_ic_air)
        ic = build_planar_startup_initial_batch(
            self.field,
            scales=self.scales,
            product_radii=r_product,
            air_radii=r_air,
            case_id=self.case_id,
            dtype=self.dtype,
            device=self.device,
        )
        names = (
            "xi_ic_product", "rho_ic_product", "u_ic_product", "e_ic_product",
            "eta_ic_air", "rho_ic_air", "u_ic_air", "e_ic_air",
            "contact_ic", "shock_ic",
        )
        return {name: getattr(ic, name) for name in names}

    # -- 组装 -------------------------------------------------------------
    def batch(self) -> SampleBatch:
        config = self.config
        s_pde_product = self.sample_network_time(config.n_pde_product)
        s_pde_air = self.sample_network_time(config.n_pde_air)
        s_contact = self.sample_network_time(config.n_contact)
        s_shock = self.sample_network_time(config.n_shock)

        payload = {
            "case_id": self.case_id,
            "charge_radius": float(self.scales.charge_radius),
            # 新原点不含 R_0/D：显式记“不适用”，不填任何伪造爆速。
            "detonation_velocity": None,
            "scales": self.scales,
            "pinn_time_min": self.pinn_time_min,
            # 本阶段窗口的上端（tau，秒），不是完整时段。
            "pinn_time_max": self.window_pinn_time_max,
            "s_product": self._column(s_pde_product),
            "xi_product": self._column(self._stratified_coordinate(config.n_pde_product)),
            "s_air": self._column(s_pde_air),
            "eta_air": self._column(self._stratified_coordinate(config.n_pde_air)),
            "s_contact": self._column(s_contact),
            "s_shock": self._column(s_shock),
        }
        payload.update(self._initial_condition())
        payload["diagnostics"] = {
            "time_convention": self.scales.time_convention,
            "startup_epsilon_s": float(self.field.epsilon),
            "initial_physical_time_s": float(self.scales.physical_time_from_network(0.0)),
            "tau_end_s": float(self.tau_end),
            "window_tau_max_s": float(self.window_pinn_time_max),
            "time_window_fraction": self.window_fraction,
            "hat_time_epsilon": self.hat_time_epsilon,
            "r_min": self.r_min,
            "n_pde_product": int(config.n_pde_product),
            "n_pde_air": int(config.n_pde_air),
            "n_ic_product": int(config.n_ic_product),
            "n_ic_air": int(config.n_ic_air),
            "n_contact": int(config.n_contact),
            "n_shock": int(config.n_shock),
            "planar_startup_cache_key": self.field.cache_key(),
            "data_supervision": "disabled: real-data time axis is not migrated",
        }
        return SampleBatch(**payload)


__all__ = [
    "AIR",
    "PRODUCT",
    "RadiusSampler",
    "SamplingConfig",
    "SamplingError",
]
