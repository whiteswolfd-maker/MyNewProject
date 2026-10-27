"""数据监督契约（新启动场时间轴）。

这条链路只接受**明确标注**的监督数据，并且**时间原点必须由调用方给出**：

.. code-block:: text

    t_phys = T_DYNA - T_origin
    tau    = t_phys - epsilon_field
    s      = clip(tau, 0, tau_end) 经 PlanarStartupScales.network_time

``T_origin`` 是"DYNA 的哪个时刻对应模型的 ``t_phys = 0``"。它**没有默认值**：
不写就是 ``unknown``，此时数据可以加载、审查与画图，但**不能**进入监督，也
不能做同时间对比。对已确认的中心起爆算例，
:mod:`spherical_blast_pinn.data.dyna_supervision` 按 ``T_origin = LT + R0/D``
从卡片参数构造并记录 ``LT``、``R0``、``D``；这不是曲线拟合，也不适用于
非中心起爆或缺少参数的 deck。其他来源必须由调用方明确给出原点。

数据字段一律 SI（m、s、kg/m^3、m/s、J/kg、Pa）。逐字段有效性是**显式布尔
数组**，不由 ``value != 0`` 推断——零速度是合法值。前沿的接触面/激波各自
有有效性与位置不确定性；缺一侧不影响另一侧。

文件格式：单个 ``.npz``，其中 ``metadata`` 是一个 JSON 字符串（0 维数组）。
数组清单见 :data:`ARRAY_KEYS`。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: 数据文件格式与版本。字段语义改变时 +1。
SUPERVISION_FORMAT = "spherical_blast_pinn.supervision"
SUPERVISION_VERSION = 1

#: 介质码。``-1`` 表示"不属于任何可监督介质"（环境或未分类）。
PRODUCT_CODE = 0
AIR_CODE = 1
NO_MEDIUM = -1

#: 时间原点状态的合法值。
ORIGIN_CONFIRMED = "confirmed"
ORIGIN_UNKNOWN = "unknown"
ORIGIN_STATUSES = (ORIGIN_CONFIRMED, ORIGIN_UNKNOWN)

#: 数据来源标记。真实 DYNA 数据必须写 ``real``；合成样例必须写 ``synthetic``。
SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL = "real"
SOURCE_KINDS = (SOURCE_SYNTHETIC, SOURCE_REAL)

#: 文件里必须存在的数组。
ARRAY_KEYS = (
    "times_dyna",
    "radius",
    "density",
    "velocity",
    "specific_energy",
    "pressure",
    "valid_density",
    "valid_velocity",
    "valid_specific_energy",
    "valid_pressure",
    "medium",
    "contact_radius",
    "contact_valid",
    "shock_radius",
    "shock_valid",
)

#: 可选数组：支持区边界。缺省时由相邻半径推出一个单元宽度。
OPTIONAL_ARRAY_KEYS = ("r_lo", "r_hi", "contact_delta_m", "shock_delta_m")

#: SI 单位表。写进文件，读取端逐项核对。
UNITS = {
    "times_dyna": "s",
    "radius": "m",
    "density": "kg/m^3",
    "velocity": "m/s",
    "specific_energy": "J/kg",
    "pressure": "Pa",
    "contact_radius": "m",
    "shock_radius": "m",
    "r_lo": "m",
    "r_hi": "m",
    "contact_delta_m": "m",
    "shock_delta_m": "m",
}


class SupervisionError(ValueError):
    """监督数据、时间原点或字段有效性不成立。"""


@dataclass(frozen=True)
class TimeOrigin:
    """DYNA 时间到模型 ``t_phys`` 的显式映射。

    ``status="confirmed"`` 时才允许监督与同时间对比。``method`` / ``evidence``
    是**给人看的依据**；它们不能由程序从曲线里自动生成后当作证据。
    """

    status: str
    origin_time_s: float | None
    method: str
    evidence: str
    source: str = "caller"

    def __post_init__(self) -> None:
        if self.status not in ORIGIN_STATUSES:
            raise SupervisionError(
                f"time-origin status must be one of {ORIGIN_STATUSES}, got "
                f"{self.status!r}"
            )
        if self.status == ORIGIN_CONFIRMED:
            value = _finite_or_none(self.origin_time_s)
            if value is None:
                raise SupervisionError(
                    "a confirmed time origin needs a finite origin_time_s; refusing to "
                    "default T_origin to 0"
                )
            object.__setattr__(self, "origin_time_s", value)
        else:
            if self.origin_time_s is not None:
                raise SupervisionError(
                    "an unknown time origin must carry origin_time_s=None; a number "
                    "here would silently become the mapping"
                )
            if not str(self.method).strip() or not str(self.evidence).strip():
                raise SupervisionError(
                    "an unknown time origin must still record why it is unknown "
                    "(method and evidence strings)"
                )

    @property
    def ready(self) -> bool:
        return self.status == ORIGIN_CONFIRMED

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "origin_time_s": self.origin_time_s,
            "method": self.method,
            "evidence": self.evidence,
            "source": self.source,
            "mapping": (
                "t_phys = T_DYNA - T_origin; tau = t_phys - epsilon_field"
                if self.status == ORIGIN_CONFIRMED
                else "not usable: T_origin is unknown"
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TimeOrigin":
        if not isinstance(payload, Mapping):
            raise SupervisionError("time_origin must be a mapping")
        status = payload.get("status")
        origin = payload.get("origin_time_s")
        method = str(payload.get("method", "") or "")
        evidence = str(payload.get("evidence", "") or "")
        source = str(payload.get("source", "caller") or "caller")
        if status == ORIGIN_UNKNOWN and origin is not None:
            raise SupervisionError(
                "time_origin.status='unknown' must not carry a numeric origin_time_s"
            )
        return cls(
            status=str(status),
            origin_time_s=None if origin is None else float(origin),
            method=method,
            evidence=evidence,
            source=source,
        )


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool_mask(array: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(array)
    if mask.dtype != np.bool_:
        raise SupervisionError(f"{name} must be a boolean array, got dtype {mask.dtype}")
    return mask


def _require_finite_where_valid(
    values: np.ndarray, valid: np.ndarray, name: str
) -> None:
    if values.shape != valid.shape:
        raise SupervisionError(
            f"{name} has shape {values.shape} but its validity mask has {valid.shape}"
        )
    if bool(np.any(valid & ~np.isfinite(values))):
        count = int(np.sum(valid & ~np.isfinite(values)))
        raise SupervisionError(
            f"{name} has {count} valid point(s) with a non-finite value; mark them "
            "invalid instead of leaving NaN behind a true flag"
        )


@dataclass(frozen=True)
class FrameSplit:
    """按**连续帧组**划分训练帧与开发帧。

    一行数据里的所有径向点属于同一帧，因此训练/开发归属按帧决定，不能按单个
    空间点随机拆开。``groups`` 是原始时间顺序上的连续组；开发组整组留出。
    记录实际帧号、组号与比例，供检查点核对。
    """

    group_frames: int
    holdout_fraction: float
    seed: int
    frames: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    development_groups: tuple[int, ...]
    train_frames: tuple[int, ...]
    development_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if int(self.group_frames) < 1:
            raise SupervisionError("development group_frames must be >= 1")
        fraction = float(self.holdout_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise SupervisionError(
                f"development holdout_fraction must lie in (0, 1), got {fraction!r}"
            )
        frames = tuple(int(v) for v in self.frames)
        if not frames or any(a >= b for a, b in zip(frames, frames[1:])):
            raise SupervisionError("split frames must be a non-empty increasing sequence")
        groups = tuple(tuple(int(v) for v in group) for group in self.groups)
        if not groups or any(not group for group in groups):
            raise SupervisionError("frame groups must be a non-empty list of non-empty groups")
        flattened = tuple(v for group in groups for v in group)
        if flattened != frames:
            raise SupervisionError("frame groups do not partition the ordered frames")
        development_groups = tuple(int(v) for v in self.development_groups)
        if not development_groups:
            raise SupervisionError("a development split needs at least one held-out group")
        if any(index < 0 or index >= len(groups) for index in development_groups):
            raise SupervisionError("development group index is out of range")
        if len(set(development_groups)) != len(development_groups):
            raise SupervisionError("development group indices must be unique")
        development = tuple(
            v for index in development_groups for v in groups[index]
        )
        development = tuple(sorted(development))
        train = tuple(v for v in frames if v not in set(development))
        if development != self.development_frames or train != self.train_frames:
            raise SupervisionError(
                "train/development frame lists do not agree with the selected groups"
            )
        if not train:
            raise SupervisionError("the development split would leave no training frames")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "development_groups", development_groups)
        object.__setattr__(self, "train_frames", train)
        object.__setattr__(self, "development_frames", development)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_train_frames(self) -> int:
        return len(self.train_frames)

    @property
    def n_development_frames(self) -> int:
        return len(self.development_frames)

    @property
    def actual_holdout_fraction(self) -> float:
        return self.n_development_frames / self.n_frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_frames": int(self.group_frames),
            "holdout_fraction": float(self.holdout_fraction),
            "seed": int(self.seed),
            "frames": list(self.frames),
            "groups": [list(group) for group in self.groups],
            "development_groups": list(self.development_groups),
            "train_frames": list(self.train_frames),
            "development_frames": list(self.development_frames),
            "n_frames": int(self.n_frames),
            "n_groups": int(self.n_groups),
            "n_train_frames": int(self.n_train_frames),
            "n_development_frames": int(self.n_development_frames),
            "actual_holdout_fraction": float(self.actual_holdout_fraction),
        }


@dataclass
class SupervisionDataset:
    """一份按材料侧监督的时空剖面（SI，显式时间原点与逐字段有效性）。"""

    case_id: str
    charge_radius_m: float
    synthetic: bool
    source_kind: str
    time_origin: TimeOrigin
    times_dyna: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    valid_density: np.ndarray
    valid_velocity: np.ndarray
    valid_specific_energy: np.ndarray
    valid_pressure: np.ndarray
    medium: np.ndarray
    contact_radius: np.ndarray
    contact_valid: np.ndarray
    shock_radius: np.ndarray
    shock_valid: np.ndarray
    r_lo: np.ndarray | None = None
    r_hi: np.ndarray | None = None
    contact_delta_m: float = float("nan")
    shock_delta_m: float = float("nan")
    source: dict[str, Any] | None = None
    notes: list[str] | None = None
    #: 由装配阶段按配置设置的连续帧组划分；``None`` 表示不设开发集。
    frame_split: FrameSplit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise SupervisionError("case_id must be a non-empty string")
        radius = _finite_or_none(self.charge_radius_m)
        if radius is None or radius <= 0.0:
            raise SupervisionError(
                f"charge_radius_m must be finite and positive, got {self.charge_radius_m!r}"
            )
        if not isinstance(self.synthetic, bool):
            raise SupervisionError("synthetic must be an explicit boolean")
        if self.source_kind not in SOURCE_KINDS:
            raise SupervisionError(
                f"source_kind must be one of {SOURCE_KINDS}, got {self.source_kind!r}"
            )
        if self.synthetic and self.source_kind != SOURCE_SYNTHETIC:
            raise SupervisionError(
                "synthetic=True requires source_kind='synthetic'; refusing a mixed label"
            )
        if not self.synthetic and self.source_kind != SOURCE_REAL:
            raise SupervisionError(
                "synthetic=False requires source_kind='real'; refusing a mixed label"
            )
        if not isinstance(self.time_origin, TimeOrigin):
            raise SupervisionError("time_origin must be a TimeOrigin")

        times = np.asarray(self.times_dyna, dtype=np.float64).reshape(-1)
        if times.size == 0:
            raise SupervisionError("times_dyna is empty")
        if not np.all(np.isfinite(times)):
            raise SupervisionError("times_dyna contains non-finite values")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise SupervisionError("times_dyna must be strictly increasing")
        self.times_dyna = times

        radius_array = np.asarray(self.radius, dtype=np.float64)
        if radius_array.ndim == 1:
            radius_array = np.broadcast_to(
                radius_array.reshape(1, -1), (times.size, radius_array.size)
            ).copy()
        if radius_array.ndim != 2 or radius_array.shape[0] != times.size:
            raise SupervisionError(
                f"radius must be (n_t, n_r) or (n_r,), got {radius_array.shape} with "
                f"{times.size} times"
            )
        if radius_array.shape[1] < 2:
            raise SupervisionError(
                "at least two radial points are needed to define a cell support"
            )
        if not np.all(np.isfinite(radius_array)):
            raise SupervisionError("radius contains non-finite values")
        if np.any(np.diff(radius_array, axis=1) <= 0.0):
            raise SupervisionError("radius must be strictly increasing within each frame")
        self.radius = radius_array

        for name in ("density", "velocity", "specific_energy", "pressure"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != radius_array.shape:
                raise SupervisionError(
                    f"{name} has shape {values.shape}, expected {radius_array.shape}"
                )
            setattr(self, name, values)
        for name in (
            "valid_density", "valid_velocity", "valid_specific_energy", "valid_pressure"
        ):
            setattr(
                self, name,
                _as_bool_mask(getattr(self, name), name).reshape(radius_array.shape),
            )
            field = name[len("valid_"):]
            _require_finite_where_valid(
                getattr(self, field), getattr(self, name), field
            )

        medium = np.asarray(self.medium)
        if medium.shape != radius_array.shape:
            raise SupervisionError(
                f"medium has shape {medium.shape}, expected {radius_array.shape}"
            )
        if not np.all(np.isin(medium, (PRODUCT_CODE, AIR_CODE, NO_MEDIUM))):
            raise SupervisionError(
                "medium must contain only 0 (product), 1 (air) and -1 (unclassified)"
            )
        self.medium = medium.astype(np.int8)
        # 材料归属是**类型**，不是观测值：标成产物却没有任何场量有效性，说明提取
        # 或标记出了问题，而不是"这一帧没有标签"。
        for code, label in ((PRODUCT_CODE, "product"), (AIR_CODE, "air")):
            selected = self.medium == code
            if selected.any():
                any_valid = (
                    self.valid_density | self.valid_velocity
                    | self.valid_specific_energy | self.valid_pressure
                )
                bad = selected & ~any_valid
                if bad.any():
                    raise SupervisionError(
                        f"{int(bad.sum())} point(s) are labelled {label} but carry no "
                        "valid field at all"
                    )

        for name in ("contact_radius", "shock_radius"):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.size != times.size:
                raise SupervisionError(
                    f"{name} must have one value per frame ({times.size}), got {values.size}"
                )
            setattr(self, name, values)
        for name in ("contact_valid", "shock_valid"):
            mask = _as_bool_mask(getattr(self, name), name).reshape(-1)
            if mask.size != times.size:
                raise SupervisionError(
                    f"{name} must have one value per frame ({times.size}), got {mask.size}"
                )
            front = name.replace("_valid", "_radius")
            _require_finite_where_valid(getattr(self, front), mask, front)
            setattr(self, name, mask)

        lo = self.r_lo
        hi = self.r_hi
        if lo is None or hi is None:
            bounds = np.empty((radius_array.shape[0], radius_array.shape[1] + 1))
            width = np.diff(radius_array, axis=1)
            bounds[:, 1:-1] = 0.5 * (radius_array[:, :-1] + radius_array[:, 1:])
            bounds[:, 0] = np.maximum(
                radius_array[:, 0] - 0.5 * width[:, 0], 0.0
            )
            bounds[:, -1] = radius_array[:, -1] + 0.5 * width[:, -1]
            lo = bounds[:, :-1]
            hi = bounds[:, 1:]
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        if lo.shape != radius_array.shape or hi.shape != radius_array.shape:
            raise SupervisionError(
                f"r_lo/r_hi must match radius shape {radius_array.shape}"
            )
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise SupervisionError("r_lo/r_hi contain non-finite values")
        if np.any(lo < 0.0) or np.any(hi <= lo):
            raise SupervisionError("support intervals must satisfy 0 <= r_lo < r_hi")
        self.r_lo = lo
        self.r_hi = hi

        for name in ("contact_delta_m", "shock_delta_m"):
            value = _finite_or_none(getattr(self, name))
            setattr(self, name, float("nan") if value is None else value)
        self.source = dict(self.source or {})
        self.notes = list(self.notes or [])
        if self.frame_split is not None:
            if not isinstance(self.frame_split, FrameSplit):
                raise SupervisionError(
                    "frame_split must be a FrameSplit or None, got "
                    f"{type(self.frame_split).__name__}"
                )
            if any(frame < 0 or frame >= times.size for frame in self.frame_split.frames):
                raise SupervisionError("frame_split refers to a frame outside this dataset")

    # -- 时间 -------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.times_dyna.size)

    def absolute_physical_time(self) -> np.ndarray:
        """``T_DYNA -> t_phys``。原点未知时**拒绝**，不默认 ``T_origin=0``。"""
        if not self.time_origin.ready:
            raise SupervisionError(
                f"case {self.case_id!r}: the DYNA-to-model time origin is unknown "
                f"({self.time_origin.evidence!r}). Supervision and same-time comparison "
                "are refused until the caller states which DYNA time is t_phys = 0"
            )
        return self.times_dyna - float(self.time_origin.origin_time_s)

    def tau_time(self, field_epsilon: float) -> np.ndarray:
        """``tau = T_DYNA - T_origin - epsilon_field``。"""
        epsilon = _finite_or_none(field_epsilon)
        if epsilon is None or epsilon < 0.0:
            raise SupervisionError(
                f"field epsilon must be finite and non-negative, got {field_epsilon!r}"
            )
        return self.absolute_physical_time() - epsilon

    # -- 连续帧组划分 -----------------------------------------------------
    def supervisable_frames(self) -> np.ndarray:
        """至少有一个可用标签的帧号（按原始时间顺序）。

        只要帧内有任意字段或任一条前沿有效，这一帧就属于划分候选。字段是否有效
        仍由各自的 ``valid_*`` 掩码逐点决定，不会因为整帧进入训练集就把缺失值
        填零。
        """
        any_field = (
            self.valid_density
            | self.valid_velocity
            | self.valid_specific_energy
            | self.valid_pressure
        ).any(axis=1)
        return np.flatnonzero(any_field | self.contact_valid | self.shock_valid)

    def split_frames(
        self,
        *,
        group_frames: int,
        holdout_fraction: float,
        seed: int,
    ) -> FrameSplit:
        """按连续帧组建立训练/开发划分，返回记录对象（不修改数据集）。"""
        group_size = int(group_frames)
        if group_size < 1:
            raise SupervisionError(
                f"development group_frames must be >= 1, got {group_frames!r}"
            )
        fraction = float(holdout_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise SupervisionError(
                "development holdout_fraction must be finite and lie in (0, 1), got "
                f"{holdout_fraction!r}"
            )
        usable = self.supervisable_frames()
        if usable.size < 2:
            raise SupervisionError(
                f"case {self.case_id!r}: development split needs at least two "
                f"supervisable frames, found {usable.size}"
            )
        groups = tuple(
            tuple(int(v) for v in usable[start:start + group_size])
            for start in range(0, usable.size, group_size)
        )
        if len(groups) < 2:
            raise SupervisionError(
                f"case {self.case_id!r}: development split needs at least two frame "
                f"groups, but {usable.size} frame(s) with group_frames={group_size} "
                "produce one group"
            )
        # 不把第一组留出：它包含最早的可监督演化，整组拿掉会让开发集与训练集的
        # 时间分布相差过远。随机只决定其余连续组中留哪几组。
        eligible = np.arange(1, len(groups), dtype=np.int64)
        target_groups = max(1, int(round(fraction * len(groups))))
        target_groups = min(target_groups, eligible.size)
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(eligible, size=target_groups, replace=False))
        development_groups = tuple(int(v) for v in chosen)
        development = tuple(
            v for index in development_groups for v in groups[index]
        )
        train = tuple(v for v in usable if v not in set(development))
        return FrameSplit(
            group_frames=group_size,
            holdout_fraction=fraction,
            seed=int(seed),
            frames=tuple(int(v) for v in usable),
            groups=groups,
            development_groups=development_groups,
            train_frames=train,
            development_frames=tuple(sorted(development)),
        )

    def assign_frame_split(
        self, *, group_frames: int, holdout_fraction: float, seed: int
    ) -> FrameSplit | None:
        """设置连续帧组划分并返回它。``holdout_fraction=0`` 清除划分。"""
        fraction = float(holdout_fraction)
        if not math.isfinite(fraction) or not (0.0 <= fraction < 1.0):
            raise SupervisionError(
                "development holdout_fraction must be finite and in [0, 1), got "
                f"{holdout_fraction!r}"
            )
        if fraction == 0.0:
            self.frame_split = None
            return None
        split = self.split_frames(
            group_frames=group_frames, holdout_fraction=fraction, seed=seed
        )
        self.frame_split = split
        return split

    def training_frame_mask(self) -> np.ndarray:
        mask = np.ones(self.n_frames, dtype=bool)
        if self.frame_split is not None:
            mask[:] = False
            mask[list(self.frame_split.train_frames)] = True
        return mask

    def development_frame_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_frames, dtype=bool)
        if self.frame_split is not None:
            mask[list(self.frame_split.development_frames)] = True
        return mask

    # -- 身份 -------------------------------------------------------------
    def identity(self) -> dict[str, Any]:
        """内容身份，随检查点保存；用于拒绝"同一路径换了数据"的恢复。"""
        digest = hashlib.sha256()
        for name in ARRAY_KEYS + OPTIONAL_ARRAY_KEYS:
            value = getattr(self, name, None)
            if value is None:
                digest.update(f"{name}|missing".encode("utf-8"))
                continue
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(f"{name}|{array.dtype.str}|{array.shape}".encode("utf-8"))
            digest.update(array.tobytes())
        return {
            "case_id": self.case_id,
            "charge_radius_m": float(self.charge_radius_m),
            "synthetic": bool(self.synthetic),
            "source_kind": self.source_kind,
            "time_origin": self.time_origin.as_dict(),
            "content_sha256": digest.hexdigest(),
            "time_origin_deltas": {
                "contact_delta_m": self.contact_delta_m,
                "shock_delta_m": self.shock_delta_m,
            },
            "development_split": (
                None if self.frame_split is None else self.frame_split.as_dict()
            ),
        }

    # -- 读写 -------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": SUPERVISION_FORMAT,
            "version": SUPERVISION_VERSION,
            "case_id": self.case_id,
            "charge_radius_m": float(self.charge_radius_m),
            "synthetic": bool(self.synthetic),
            "source_kind": self.source_kind,
            "time_origin": self.time_origin.as_dict(),
            "units": dict(UNITS),
            "contact_delta_m": self.contact_delta_m,
            "shock_delta_m": self.shock_delta_m,
            "source": dict(self.source or {}),
            "notes": list(self.notes or []),
        }
        payload = {name: getattr(self, name) for name in ARRAY_KEYS}
        payload.update({name: getattr(self, name) for name in OPTIONAL_ARRAY_KEYS})
        payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        temporary = target.with_suffix(target.suffix + ".tmp")
        # ``np.savez`` 会自行追加 ``.npz``；用文件对象写，避免实际文件名与
        # ``temporary`` 不一致、随后 ``replace`` 找不到文件。
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "SupervisionDataset":
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as archive:
                if "metadata" not in archive:
                    raise SupervisionError(
                        f"{source}: no 'metadata' entry; this is not a supervision file "
                        f"({SUPERVISION_FORMAT} v{SUPERVISION_VERSION})"
                    )
                raw = archive["metadata"].item()
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                metadata = json.loads(text)
                missing = [key for key in ARRAY_KEYS if key not in archive]
                if missing:
                    raise SupervisionError(f"{source}: missing arrays {missing}")
                arrays = {key: archive[key] for key in ARRAY_KEYS}
                for key in OPTIONAL_ARRAY_KEYS:
                    # 两个 delta 是**元数据标量**，不从数组通道再传一遍；否则
                    # 构造器会收到两个同名关键字。
                    if key in archive and key not in (
                        "contact_delta_m", "shock_delta_m"
                    ):
                        arrays[key] = archive[key]
        except SupervisionError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the file's actual problem
            raise SupervisionError(f"{source}: cannot read the supervision file: {exc}") from exc

        if metadata.get("format") != SUPERVISION_FORMAT:
            raise SupervisionError(
                f"{source}: format {metadata.get('format')!r} is not "
                f"{SUPERVISION_FORMAT!r}"
            )
        if int(metadata.get("version", -1)) != SUPERVISION_VERSION:
            raise SupervisionError(
                f"{source}: version {metadata.get('version')!r} is not "
                f"{SUPERVISION_VERSION}; no migration is performed"
            )
        units = metadata.get("units") or {}
        mismatched = {
            key: value for key, value in UNITS.items()
            if key in arrays and units.get(key) != value
        }
        if mismatched:
            raise SupervisionError(
                f"{source}: units differ from SI ({mismatched}); refusing to reinterpret "
                "the numbers"
            )
        return cls(
            case_id=str(metadata.get("case_id", "")),
            charge_radius_m=float(metadata.get("charge_radius_m", float("nan"))),
            synthetic=metadata.get("synthetic"),
            source_kind=str(metadata.get("source_kind", "")),
            time_origin=TimeOrigin.from_mapping(metadata.get("time_origin") or {}),
            **arrays,
            contact_delta_m=metadata.get("contact_delta_m", float("nan")),
            shock_delta_m=metadata.get("shock_delta_m", float("nan")),
            source=metadata.get("source") or {},
            notes=metadata.get("notes") or [],
        )

    # -- 可用性 -----------------------------------------------------------
    def supervision_blockers(self) -> list[str]:
        """为什么这份数据现在**不能**进入监督。空列表表示可以。"""
        blockers: list[str] = []
        if not self.time_origin.ready:
            blockers.append(
                "the DYNA-to-model time origin is unknown; supervision requires an "
                "explicit confirmed T_origin and tau = t_phys - epsilon_field"
            )
        if not math.isfinite(self.contact_delta_m) or self.contact_delta_m <= 0.0:
            blockers.append(
                "contact_delta_m is missing; L_geometry must not silently drop"
            )
        if not math.isfinite(self.shock_delta_m) or self.shock_delta_m <= 0.0:
            blockers.append(
                "shock_delta_m is missing; L_geometry must not silently drop"
            )
        return blockers

    def require_supervision_ready(self) -> None:
        blockers = self.supervision_blockers()
        if blockers:
            raise SupervisionError(
                f"case {self.case_id!r} cannot be used for supervision: "
                + "; ".join(blockers)
            )


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------


@dataclass
class SupervisionSample:
    """一次批次里的数据监督点。所有数组已是 ``(N, 1)`` 或 ``(N,)`` 约定。"""

    fields: dict[str, Any]
    counts: dict[str, int]
    tau_range_s: tuple[float, float] | None


def _finite_tau_mask(
    tau: np.ndarray, window_tau_max: float
) -> np.ndarray:
    return np.isfinite(tau) & (tau >= -1.0e-15) & (tau <= float(window_tau_max) + 1.0e-15)


def sample_supervision(
    dataset: SupervisionDataset,
    *,
    scales,
    field_epsilon: float,
    window_tau_max: float,
    rng: np.random.Generator,
    n_product: int,
    n_air: int,
    n_contact: int,
    n_shock: int,
    dtype,
    device=None,
    frame_mask: np.ndarray | Sequence[bool] | None = None,
) -> SupervisionSample:
    """从一份数据集里采一批监督点，**只取有效点**。

    * 时间：``tau`` 必须在 ``[0, window_tau_max]`` 内；窗口外不是"标签为零"，
      而是这一步没有那个点。
    * 材料：由数据自己的 ``medium`` 码选择；产物与空气分开采样，不跨界面。
    * 字段：逐字段有效性写进标签——无效字段写 ``NaN``，损失按字段跳过；不会把
      一个无效观测当成 0。
    * 前沿：接触面与激波各自用自己的有效性；缺一侧不影响另一侧。
    """
    import torch

    dataset.require_supervision_ready()
    if frame_mask is None:
        selected_frames = np.ones(dataset.n_frames, dtype=bool)
    else:
        selected_frames = np.asarray(frame_mask)
        if selected_frames.dtype != np.bool_:
            raise SupervisionError("frame_mask must be a boolean array")
        if selected_frames.shape != (dataset.n_frames,):
            raise SupervisionError(
                f"frame_mask has shape {selected_frames.shape}, expected "
                f"({dataset.n_frames},)"
            )
    tau = dataset.tau_time(field_epsilon)
    in_window = _finite_tau_mask(tau, window_tau_max) & selected_frames
    if not in_window.any():
        return SupervisionSample(
            fields={}, counts={"product": 0, "air": 0, "contact": 0, "shock": 0},
            tau_range_s=None,
        )
    s_values = np.asarray(scales.network_time(
        torch.as_tensor(
            np.clip(tau, 0.0, None), dtype=dtype
        )
    ).detach().cpu(), dtype=np.float64)

    def sample_group(count: int, candidate_mask: np.ndarray, label: str):
        if count <= 0:
            return None
        rows, cols = np.nonzero(candidate_mask)
        if rows.size == 0:
            return None
        take = int(min(int(count), rows.size))
        choice = rng.choice(rows.size, size=take, replace=False)
        chosen_rows = rows[choice]
        chosen_cols = cols[choice]
        return chosen_rows, chosen_cols

    def field_labels(rows, cols, density_key, velocity_key, energy_key, valid_prefix):
        density = np.asarray(dataset.density[rows, cols], dtype=np.float64)
        velocity = np.asarray(dataset.velocity[rows, cols], dtype=np.float64)
        energy = np.asarray(dataset.specific_energy[rows, cols], dtype=np.float64)
        density[~getattr(dataset, valid_prefix + "density")[rows, cols]] = np.nan
        velocity[~getattr(dataset, valid_prefix + "velocity")[rows, cols]] = np.nan
        energy[~getattr(dataset, valid_prefix + "specific_energy")[rows, cols]] = np.nan
        return density, velocity, energy

    fields: dict[str, Any] = {}
    counts = {"product": 0, "air": 0, "contact": 0, "shock": 0}
    finite_fields = (
        np.isfinite(dataset.density)
        | np.isfinite(dataset.velocity)
        | np.isfinite(dataset.specific_energy)
    )
    product_valid = (dataset.medium == PRODUCT_CODE) & in_window[:, None]
    air_valid = (dataset.medium == AIR_CODE) & in_window[:, None]
    # 有效点由**有效性掩码**决定，不由字段是否有限决定；无效点本来就可能带 NaN。
    product_usable = product_valid & (
        dataset.valid_density | dataset.valid_velocity | dataset.valid_specific_energy
    )
    air_usable = air_valid & (
        dataset.valid_density | dataset.valid_velocity | dataset.valid_specific_energy
    )
    del finite_fields

    product = sample_group(n_product, product_usable, "product")
    air = sample_group(n_air, air_usable, "air")
    for medium, group in (("product", product), ("air", air)):
        if group is None:
            continue
        rows, cols = group
        prefix = "valid_"
        rho, u, e = field_labels(
            rows, cols, "density", "velocity", "specific_energy", prefix
        )
        fields[f"s_data_{medium}"] = _column(s_values[rows], dtype, device)
        fields[f"r_data_{medium}"] = _column(dataset.radius[rows, cols], dtype, device)
        fields[f"lo_data_{medium}"] = _column(dataset.r_lo[rows, cols], dtype, device)
        fields[f"hi_data_{medium}"] = _column(dataset.r_hi[rows, cols], dtype, device)
        fields[f"coverage_data_{medium}"] = _column(
            np.ones(rows.size), dtype, device
        )
        fields[f"rho_data_{medium}"] = _column(rho, dtype, device)
        fields[f"u_data_{medium}"] = _column(u, dtype, device)
        fields[f"e_data_{medium}"] = _column(e, dtype, device)
        counts[medium] = int(rows.size)

    # 前沿：各用各的有效性与时间点。无效的那一侧不产生样本，也不补零。
    contact_mask = dataset.contact_valid & in_window
    shock_mask = dataset.shock_valid & in_window
    for name, mask, count, field_s, field_label in (
        ("contact", contact_mask, n_contact, "s_data_contact", "contact_data"),
        ("shock", shock_mask, n_shock, "s_data_shock", "shock_data"),
    ):
        if count <= 0:
            continue
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            continue
        take = int(min(int(count), indices.size))
        chosen = rng.choice(indices, size=take, replace=False)
        values = (
            dataset.contact_radius[chosen]
            if name == "contact"
            else dataset.shock_radius[chosen]
        )
        fields[field_s] = _column(s_values[chosen], dtype, device)
        fields[field_label] = _column(values, dtype, device)
        counts[name] = int(take)

    fields["geometry_delta_contact"] = float(dataset.contact_delta_m)
    fields["geometry_delta_shock"] = float(dataset.shock_delta_m)
    active_tau = tau[in_window]
    return SupervisionSample(
        fields=fields,
        counts=counts,
        tau_range_s=(float(active_tau.min()), float(active_tau.max())),
    )


def _column(values, dtype, device):
    import torch

    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return torch.as_tensor(array, dtype=dtype, device=device)


# --------------------------------------------------------------------------
# 合成监督样例（只用于接口验证）
# --------------------------------------------------------------------------


def synthetic_dataset(
    *,
    case_id: str,
    charge_radius_m: float,
    times_dyna: Sequence[float],
    radius: Sequence[float] | np.ndarray,
    origin_time_s: float,
    fields: Mapping[str, np.ndarray],
    valid: Mapping[str, np.ndarray] | None = None,
    medium: np.ndarray | None = None,
    contact_radius: Sequence[float] | None = None,
    contact_valid: Sequence[bool] | None = None,
    shock_radius: Sequence[float] | None = None,
    shock_valid: Sequence[bool] | None = None,
    contact_delta_m: float = 1.0e-4,
    shock_delta_m: float = 1.0e-4,
    origin_method: str = "synthetic prescribed offset",
    origin_evidence: str = "constructed with a known offset for interface testing only",
    notes: Sequence[str] | None = None,
) -> SupervisionDataset:
    """构造一份**明确标记 synthetic** 的数据集。

    这是接口测试工具，不是 DYNA 数据，也不构成物理验证。所有字段 SI；``fields``
    必须给出 ``density`` / ``velocity`` / ``specific_energy`` / ``pressure``。
    """
    times = np.asarray(times_dyna, dtype=np.float64)
    radius_array = np.asarray(radius, dtype=np.float64)
    if radius_array.ndim == 1:
        radius_array = np.broadcast_to(
            radius_array.reshape(1, -1), (times.size, radius_array.size)
        ).copy()
    shape = radius_array.shape
    if shape[0] != times.size:
        raise SupervisionError("radius/time shapes are inconsistent")
    values = {}
    for name in ("density", "velocity", "specific_energy", "pressure"):
        if name not in fields:
            raise SupervisionError(f"synthetic data is missing the {name!r} field")
        array = np.asarray(fields[name], dtype=np.float64)
        if array.shape != shape:
            raise SupervisionError(
                f"synthetic field {name!r} has shape {array.shape}, expected {shape}"
            )
        values[name] = array
    if medium is None:
        medium_array = np.full(shape, NO_MEDIUM, dtype=np.int8)
    else:
        medium_array = np.asarray(medium, dtype=np.int8)
        if medium_array.shape != shape:
            raise SupervisionError("medium shape is inconsistent")
    valid_masks = {}
    for name in ("density", "velocity", "specific_energy", "pressure"):
        if valid is None or name not in valid:
            valid_masks[f"valid_{name}"] = np.isfinite(values[name])
        else:
            valid_masks[f"valid_{name}"] = np.asarray(valid[name], dtype=bool)
    if contact_radius is None:
        contact_radius = np.full(times.size, float("nan"))
        contact_valid = np.zeros(times.size, dtype=bool)
    else:
        contact_radius = np.asarray(contact_radius, dtype=np.float64)
        contact_valid = (
            np.zeros(times.size, dtype=bool)
            if contact_valid is None
            else np.asarray(contact_valid, dtype=bool)
        )
    if shock_radius is None:
        shock_radius = np.full(times.size, float("nan"))
        shock_valid = np.zeros(times.size, dtype=bool)
    else:
        shock_radius = np.asarray(shock_radius, dtype=np.float64)
        shock_valid = (
            np.zeros(times.size, dtype=bool)
            if shock_valid is None
            else np.asarray(shock_valid, dtype=bool)
        )
    return SupervisionDataset(
        case_id=case_id,
        charge_radius_m=float(charge_radius_m),
        synthetic=True,
        source_kind=SOURCE_SYNTHETIC,
        time_origin=TimeOrigin(
            status=ORIGIN_CONFIRMED,
            origin_time_s=float(origin_time_s),
            method=str(origin_method),
            evidence=str(origin_evidence),
            source="synthetic-builder",
        ),
        times_dyna=times,
        radius=radius_array,
        density=values["density"],
        velocity=values["velocity"],
        specific_energy=values["specific_energy"],
        pressure=values["pressure"],
        medium=medium_array,
        contact_radius=contact_radius,
        contact_valid=contact_valid,
        shock_radius=shock_radius,
        shock_valid=shock_valid,
        contact_delta_m=float(contact_delta_m),
        shock_delta_m=float(shock_delta_m),
        source={"origin": "synthetic builder", "purpose": "interface testing only"},
        notes=list(notes or [
            "SYNTHETIC: this is not DYNA data and does not validate the physics"
        ]),
        **valid_masks,
    )


__all__ = [
    "AIR_CODE",
    "ARRAY_KEYS",
    "NO_MEDIUM",
    "OPTIONAL_ARRAY_KEYS",
    "ORIGIN_CONFIRMED",
    "ORIGIN_UNKNOWN",
    "PRODUCT_CODE",
    "SOURCE_KINDS",
    "SOURCE_REAL",
    "SOURCE_SYNTHETIC",
    "SUPERVISION_FORMAT",
    "SUPERVISION_VERSION",
    "FrameSplit",
    "SupervisionDataset",
    "SupervisionError",
    "SupervisionSample",
    "TimeOrigin",
    "sample_supervision",
    "synthetic_dataset",
]
