"""检查点格式与读写规则的**唯一定义**（新启动场，共享模型 + 多半径）。

这份检查点只服务一条链路：

.. code:: text

    cli.startup -> handoff.json/npz -> PlanarStartupField
    -> PlanarStartupScales -> 每半径采样
    -> 一个共享 CoupledModel（ProductNet/AirNet/前沿）-> 联合训练

一个模型、一个优化器；每个算例只保留自己的启动场身份、尺度、采样状态与
监督数据。时间约定固定为 ``t_phys = epsilon + tau``，``s = 0`` 对应
``t_phys = epsilon``。

检查点**不复制**初场数组：加载时用显式提供的 ``handoff.json``（或调用方
自己的场与尺度）核对身份即可。未训练半径的预测走
:func:`load_frozen_model`：允许不同的 ``cache_key``/半径，但材料、环境、
``delta``、方法、时间尺度与无量纲前沿初值必须一致。

旧检查点（reference-field / 方案 B，``format = "spherical_blast_pinn.checkpoint"``
且带 ``detonation_velocity``、``start_time = T_h = R_0/D + epsilon``）在这里
**明确拒绝**：格式与版本都不同，没有迁移路径，也不猜测旧数据的时间原点。
新链路重新生成检查点。

保留的通用读取规则：模型状态结构/形状/有限性
（:func:`validate_model_state`）、优化器状态
（:func:`validate_optimizer_state`）、步号（:func:`require_step_value`）与
模型参数指纹（:func:`parameter_fingerprint_of`）。旧训练器的"发布标记"与
"待发布最佳候选"事务校验已随独立模型路径一起删除。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..physics.material_state import (
    MaterialConditionsError,
    condition_mismatches,
    field_material_conditions,
    model_material_conditions,
    validate_material_record,
)
from ..physics.similitude import (
    TIME_CONVENTION_PLANAR_STARTUP,
    PlanarStartupScales,
)
from ..physics.two_medium_startup_handoff import load_planar_startup_handoff
from ..physics.two_medium_startup_planar import PlanarStartupField
from ..sampling.rar import RARConfig, RARError, validate_rar_state

#: 检查点格式标识与版本。旧格式的标识是
#: ``"spherical_blast_pinn.checkpoint"``（版本 ≤ 8，带爆速与 ``T_h``），
#: 读取端按格式与版本直接拒绝，不做迁移。
#:
#: **v5（2026-10-27）**：每个 case 绑定连续帧组开发划分（经数据身份）、RAR
#: 配置/点池/触发进度与采样器 RNG。v4 没有 RAR 与开发划分状态，无法精确续训。
#:
#: **v4（2026-10-27）**：``material`` 算例记录绑定完整 JWL、空气 ``gamma``、
#: 环境密度/压力以及初始产物/空气热力学状态。模型与检查点必须与启动场使用
#: 同一组物理条件；只有数值归一化尺度允许逐半径不同。
#:
#: **v3（2026-10-28）**：多半径共用**一套** ProductNet/AirNet/前沿参数；模型与
#: 优化器只保存一份，``cases`` 只保存各算例的场/尺度/采样/数据。v3 的
#: ``material`` 记录只比较 JWL 与空气三项，未绑定初始热力学状态；该记录不足以
#: 证明材料一致，不能安全迁移。
#: **v2（2026-10-27）** 是"每个半径一份模型与优化器"的独立模型格式，
#: 不能拆出一个半径的参数冒充共享模型续训，因此 v2 被明确拒绝并提示重新生成。
CHECKPOINT_FORMAT = "spherical_blast_pinn.startup_checkpoint"
CHECKPOINT_VERSION = 5

#: 用途标记。它不是版本号：同一版本下换用途必须换一个新值，否则读取端会把
#: 另一种结构的文件当成这一种。
CHECKPOINT_PURPOSE = "planar_startup_field_training_shared"

#: 顶层字段名。v5 的顶层是格式、用途、``model``、``cases`` 与 ``training``。
MODEL_KEY = "model"
CASES_KEY = "cases"
TRAINING_KEY = "training"
STARTUP_KEY = "startup"
SCALES_KEY = "scales"
MODEL_STRUCTURE_KEY = "model_structure"
MODEL_STATE_KEY = "model_state"
OPTIMIZER_KEY = "optimizer"
STEP_KEY = "step"
RNG_KEY = "rng"
SAMPLING_KEY = "sampling"
SAMPLER_RNG_KEY = "sampler_rng"
RAR_KEY = "rar"
DATA_KEY = "data"
CASE_ID_KEY = "case_id"
CASE_SPLIT_KEY = "split"
INITIAL_FRONT_KEY = "initial_front"
MATERIAL_KEY = "material"
CONTACT_PRESSURE_KEY = "contact_pressure_pa"
FINGERPRINT_KEY = "parameter_fingerprint"
CODE_VERSION_KEY = "code_version"
PURPOSE_KEY = "purpose"

#: 共享模型块。参数只有一份，optimizer 里每个共享参数只出现一次。
MODEL_BLOCK_KEYS = (
    MODEL_STRUCTURE_KEY,
    MODEL_STATE_KEY,
    OPTIMIZER_KEY,
    FINGERPRINT_KEY,
)

#: ``cases`` 列表中每一项必须带的东西。它**不**含模型/优化器：那些是共享的。
CASE_RECORD_KEYS = (
    CASE_ID_KEY,
    CASE_SPLIT_KEY,
    STARTUP_KEY,
    SCALES_KEY,
    INITIAL_FRONT_KEY,
    MATERIAL_KEY,
    SAMPLING_KEY,
    SAMPLER_RNG_KEY,
    RAR_KEY,
    CONTACT_PRESSURE_KEY,
    DATA_KEY,
)

#: ``training`` 块必须带的东西。没有它们只能"加载权重"，不能"续训"。
TRAINING_KEYS = (
    STEP_KEY,
    "stage",
    "schedule",
    "balancing",
    "rng",
    "history",
    "stage_history",
    "check_history",
    "best",
    "run_origin",
    "diagnostic",
    "settings",
    "resumable",
)

#: 启动场的身份字段。这里**不**重复记录半径/epsilon/time_scale；那些只在
#: ``scales`` 记录里保存一份，避免两份可相互矛盾的参数。
STARTUP_MODEL_ID_KEY = "model_id"
STARTUP_METHOD_VERSION_KEY = "method_version"
STARTUP_CACHE_KEY = "cache_key"
STARTUP_DELTA_KEY = "delta"

#: ``PlanarStartupScales`` 的完整参数。重建后的对象必须保留类型、epsilon 原点、
#: ``time_scale`` 与 ``s <-> tau <-> t_phys`` 的换算。
SCALES_RECORD_KEYS = (
    "charge_radius",
    "reference_velocity",
    "reference_specific_energy",
    "product_density",
    "air_density",
    "time_scale",
    "startup_epsilon",
    "time_convention",
    "detonation_velocity",
)

#: 比较尺度记录时的相对容差。写入端与读取端是同一批 ``float``，往返逐位不变，
#: 这里只放过浮点表示本身的舍入。
SCALES_RELATIVE_TOLERANCE = 1.0e-9

class TrainingError(RuntimeError):
    """训练无法继续。"""


# --------------------------------------------------------------------------
# 多文件发布的读取规则
# --------------------------------------------------------------------------
















# --------------------------------------------------------------------------
# 冻结的物理参数
# --------------------------------------------------------------------------


def _real_number(value: Any) -> float | None:
    """取"合法的实数"；布尔、字符串、列表、``None`` 等一律返回 ``None``。

    ``bool`` 必须**显式**排除：``isinstance(True, int)`` 为真，而 ``float(True)``
    会安静地得到 ``1.0``——一个"看起来合法"的爆速。``"6930"`` 同理：
    ``float("6930")`` 成功，于是错误类型被转成了合法输入。
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


# --------------------------------------------------------------------------
# 通用归一化与数字检查
# --------------------------------------------------------------------------


def _normalise(value: Any) -> Any:
    """把结构里的 NumPy / Python 标量归一成可比较的普通类型。"""
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _same_number(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= SCALES_RELATIVE_TOLERANCE * max(
        abs(float(left)), abs(float(right)), 1.0e-30
    )


def _require_field(field: Any) -> PlanarStartupField:
    if not isinstance(field, PlanarStartupField):
        raise TrainingError(
            f"a startup checkpoint needs a PlanarStartupField, got "
            f"{type(field).__name__}"
        )
    if field.checks.get("passed") is not True:
        raise TrainingError(
            "the startup field has no passed consistency report; refusing to bind a "
            "checkpoint to it"
        )
    if field.reference_field is not False:
        raise TrainingError(
            "the startup field claims reference_field=True; this checkpoint format is "
            "only for the local planar startup field"
        )
    return field


# --------------------------------------------------------------------------
# 尺度、启动场身份
# --------------------------------------------------------------------------


def startup_scale_record(scales: PlanarStartupScales) -> dict[str, Any]:
    """把 ``PlanarStartupScales`` 的**全部**参数写成一条记录。"""
    if not isinstance(scales, PlanarStartupScales):
        raise TrainingError(
            f"scales must be PlanarStartupScales, got {type(scales).__name__}"
        )
    record = {
        "charge_radius": float(scales.charge_radius),
        "reference_velocity": float(scales.reference_velocity),
        "reference_specific_energy": float(scales.reference_specific_energy),
        "product_density": float(scales.product_density),
        "air_density": float(scales.air_density),
        "time_scale": float(scales.time_scale),
        "startup_epsilon": float(scales.startup_epsilon),
        "time_convention": str(scales.time_convention),
        "detonation_velocity": None,
    }
    for key in SCALES_RECORD_KEYS[:-2]:
        value = _real_number(record[key])
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise TrainingError(
                f"the startup scale {key!r} is {record[key]!r}; it must be finite and "
                "positive"
            )
    if record["time_convention"] != TIME_CONVENTION_PLANAR_STARTUP:
        raise TrainingError(
            f"the startup scales carry time_convention={record['time_convention']!r}; "
            f"this checkpoint format requires {TIME_CONVENTION_PLANAR_STARTUP!r}"
        )
    if record["detonation_velocity"] is not None:
        raise TrainingError(
            "the planar-startup scales must carry detonation_velocity=None; the new time "
            "origin is epsilon_field and has no R_0/D term"
        )
    return record


def planar_scales_from_record(
    record: Any, *, where: str = "scales", error: type[Exception] = TrainingError
) -> PlanarStartupScales:
    """从检查点记录**重建** ``PlanarStartupScales``（类型、epsilon 原点与换算）。"""
    if not isinstance(record, Mapping):
        raise error(f"{where}: {type(record).__name__}, not a scale record")
    missing = [key for key in SCALES_RECORD_KEYS if key not in record]
    if missing:
        raise error(f"{where}: missing {missing}")
    numbers: dict[str, float] = {}
    for key in SCALES_RECORD_KEYS[:-2]:
        value = _real_number(record[key])
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise error(
                f"{where}.{key}: {record[key]!r} is not a finite positive number"
            )
        numbers[key] = value
    if record["time_convention"] != TIME_CONVENTION_PLANAR_STARTUP:
        raise error(
            f"{where}.time_convention: {record['time_convention']!r} is not "
            f"{TIME_CONVENTION_PLANAR_STARTUP!r}"
        )
    if record["detonation_velocity"] is not None:
        raise error(
            f"{where}.detonation_velocity must be None for the planar-startup time origin"
        )
    try:
        scales = PlanarStartupScales(
            charge_radius=numbers["charge_radius"],
            reference_velocity=numbers["reference_velocity"],
            reference_specific_energy=numbers["reference_specific_energy"],
            product_density=numbers["product_density"],
            air_density=numbers["air_density"],
            time_scale=numbers["time_scale"],
            detonation_velocity=None,
            startup_epsilon=numbers["startup_epsilon"],
        )
    except Exception as exc:  # noqa: BLE001 - re-raise with the record's context
        raise error(f"{where}: the recorded scale parameters are inconsistent: {exc}") from exc
    if not _same_number(scales.start_time(), numbers["startup_epsilon"]):
        raise error(
            f"{where}: the rebuilt scale starts at {scales.start_time()!r} s, not the "
            f"recorded epsilon {numbers['startup_epsilon']!r} s"
        )
    if scales.time_convention != TIME_CONVENTION_PLANAR_STARTUP:
        raise error(f"{where}: the rebuilt scale changed its time convention")
    return scales


def _check_scales_match(
    record: Any, scales: PlanarStartupScales, *, where: str
) -> PlanarStartupScales:
    rebuilt = planar_scales_from_record(record, where=where)
    if not isinstance(scales, PlanarStartupScales):
        raise TrainingError(f"{where}: the receiving scales are {type(scales).__name__}")
    left = _normalise(rebuilt.as_dict())
    right = _normalise(scales.as_dict())
    if left != right:
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise TrainingError(
            f"{where}: the checkpoint's scales differ from the receiving scales in "
            f"{differing}; resuming would change the network's time input or units"
        )
    return rebuilt


def startup_identity(field: PlanarStartupField) -> dict[str, Any]:
    return {
        STARTUP_MODEL_ID_KEY: str(field.model_id),
        STARTUP_METHOD_VERSION_KEY: str(field.method_version),
        STARTUP_CACHE_KEY: str(field.cache_key()),
        STARTUP_DELTA_KEY: float(field.delta),
    }


def _check_startup_identity(
    identity: Any,
    field: PlanarStartupField,
    scales: PlanarStartupScales,
    *,
    where: str,
) -> None:
    if not isinstance(identity, Mapping):
        raise TrainingError(f"{where}: the startup identity is not a mapping")
    for key in (
        STARTUP_MODEL_ID_KEY,
        STARTUP_METHOD_VERSION_KEY,
        STARTUP_CACHE_KEY,
    ):
        value = identity.get(key)
        if not isinstance(value, str) or not value:
            raise TrainingError(f"{where}: the startup identity has no usable {key!r}")
    delta = _real_number(identity.get(STARTUP_DELTA_KEY))
    checks = (
        (identity[STARTUP_MODEL_ID_KEY], field.model_id, "model_id"),
        (identity[STARTUP_METHOD_VERSION_KEY], field.method_version, "method_version"),
        (identity[STARTUP_CACHE_KEY], field.cache_key(), "cache_key"),
    )
    for recorded, actual, label in checks:
        if recorded != actual:
            raise TrainingError(
                f"{where}: the checkpoint was written for startup {label} "
                f"{recorded!r}, but this field has {actual!r}; the initial condition is "
                "not the one this checkpoint was trained on"
            )
    if delta is None or abs(delta - float(field.delta)) > SCALES_RELATIVE_TOLERANCE * max(
        abs(float(field.delta)), 1.0e-30
    ):
        raise TrainingError(
            f"{where}: the checkpoint's delta is {identity.get(STARTUP_DELTA_KEY)!r}, "
            f"but this field has {field.delta!r}"
        )
    if not _same_number(field.epsilon, scales.startup_epsilon):
        raise TrainingError(
            f"{where}: the field epsilon is {field.epsilon!r} s but the recorded scales "
            f"start at {scales.startup_epsilon!r} s"
        )
    if not _same_number(field.charge_radius, scales.charge_radius):
        raise TrainingError(
            f"{where}: the field charge radius is {field.charge_radius!r} m but the "
            f"recorded scales belong to {scales.charge_radius!r} m"
        )


# --------------------------------------------------------------------------
# 采样配置（静态设置 + 动态窗口）
# --------------------------------------------------------------------------

SAMPLING_TAU_END_KEY = "tau_end_s"
SAMPLING_R_MIN_FRACTION_KEY = "r_min_fraction"
SAMPLING_COUNTS_KEY = "counts"
SAMPLING_STRATIFICATION_KEY = "stratification"
SAMPLING_WINDOW_KEY = "window"
SAMPLING_RAR_KEY = "rar"
SAMPLING_RECORD_KEYS = (
    SAMPLING_TAU_END_KEY,
    SAMPLING_R_MIN_FRACTION_KEY,
    SAMPLING_COUNTS_KEY,
    SAMPLING_STRATIFICATION_KEY,
    SAMPLING_RAR_KEY,
    SAMPLING_WINDOW_KEY,
)
SAMPLING_COUNT_KEYS = (
    "n_pde_product",
    "n_pde_air",
    "n_ic_product",
    "n_ic_air",
    "n_contact",
    "n_shock",
    "n_data_product",
    "n_data_air",
    "n_data_contact",
    "n_data_shock",
)
SAMPLING_STRATIFICATION_KEYS = ("near_fraction", "near_span", "xi_floor")


def _validate_sampling_record(
    record: Any, *, where: str, error: type[Exception] = TrainingError
) -> None:
    from ..sampling.sampler import SamplingError, validate_time_window_record

    if not isinstance(record, Mapping):
        raise error(
            f"{where}: the sampling record is {type(record).__name__}, not a mapping"
        )
    missing = [key for key in SAMPLING_RECORD_KEYS if key not in record]
    if missing:
        raise error(f"{where}: the sampling record is missing {missing}")

    tau_end = _real_number(record[SAMPLING_TAU_END_KEY])
    if tau_end is None or not math.isfinite(tau_end) or tau_end <= 0.0:
        raise error(
            f"{where}.{SAMPLING_TAU_END_KEY}: {record[SAMPLING_TAU_END_KEY]!r} is not a "
            "finite positive duration"
        )
    r_min_fraction = _real_number(record[SAMPLING_R_MIN_FRACTION_KEY])
    if r_min_fraction is None or not (0.0 < r_min_fraction < 1.0):
        raise error(
            f"{where}.{SAMPLING_R_MIN_FRACTION_KEY}: "
            f"{record[SAMPLING_R_MIN_FRACTION_KEY]!r} is not in (0, 1)"
        )
    counts = record[SAMPLING_COUNTS_KEY]
    if not isinstance(counts, Mapping):
        raise error(
            f"{where}.{SAMPLING_COUNTS_KEY}: {type(counts).__name__}, not a mapping"
        )
    for key in SAMPLING_COUNT_KEYS:
        value = counts.get(key)
        if not _is_int_like(value) or int(value) < 0:
            raise error(
                f"{where}.{SAMPLING_COUNTS_KEY}.{key}: {value!r} is not a "
                "non-negative integer"
            )
    for key in ("n_ic_product", "n_ic_air"):
        if int(counts[key]) < 1:
            raise error(
                f"{where}.{SAMPLING_COUNTS_KEY}.{key}: {counts[key]!r}; the initial "
                "condition is always part of the batch, so it cannot be zero"
            )
    stratification = record[SAMPLING_STRATIFICATION_KEY]
    if not isinstance(stratification, Mapping):
        raise error(
            f"{where}.{SAMPLING_STRATIFICATION_KEY}: "
            f"{type(stratification).__name__}, not a mapping"
        )
    for key in SAMPLING_STRATIFICATION_KEYS:
        if key not in stratification:
            raise error(f"{where}.{SAMPLING_STRATIFICATION_KEY}: missing {key!r}")
        value = _real_number(stratification[key])
        if value is None or not math.isfinite(value):
            raise error(
                f"{where}.{SAMPLING_STRATIFICATION_KEY}.{key}: "
                f"{stratification[key]!r} is not a finite number"
            )
    if not (0.0 <= float(stratification["near_fraction"]) <= 1.0):
        raise error(f"{where}: near_fraction is outside [0, 1]")
    for key in ("near_span", "xi_floor"):
        if not (0.0 < float(stratification[key]) < 0.5):
            raise error(f"{where}: {key} is outside (0, 0.5)")

    try:
        rar_config = RARConfig.from_mapping(record[SAMPLING_RAR_KEY])
    except RARError as exc:
        raise error(f"{where}.{SAMPLING_RAR_KEY}: {exc}") from exc
    if rar_config.as_dict() != _normalise(record[SAMPLING_RAR_KEY]):
        raise error(
            f"{where}.{SAMPLING_RAR_KEY}: normalized RAR settings do not round-trip"
        )

    window = record[SAMPLING_WINDOW_KEY]
    try:
        validate_time_window_record(window)
    except SamplingError as exc:
        raise error(f"{where}.{SAMPLING_WINDOW_KEY}: {exc}") from exc
    full = float(window["full_pinn_time_max"])
    if not _same_number(full, tau_end):
        raise error(
            f"{where}.{SAMPLING_WINDOW_KEY}: the window's full duration is {full!r} s, "
            f"but the recorded tau_end is {tau_end!r} s"
        )


def sampling_record(sampler: Any) -> dict[str, Any]:
    if not hasattr(sampler, "sampling_config_state"):
        raise TrainingError(
            f"the checkpoint needs a RadiusSampler-like object, got "
            f"{type(sampler).__name__}"
        )
    record = _normalise(sampler.sampling_config_state())
    _validate_sampling_record(record, where="save_startup_checkpoint: sampling")
    return record


def _check_sampling_record(record: Any, sampler: Any, *, where: str) -> None:
    _validate_sampling_record(record, where=where)
    expected = _normalise(sampler.sampling_config_state())
    for key in (SAMPLING_TAU_END_KEY, SAMPLING_R_MIN_FRACTION_KEY):
        if not _same_number(record[key], expected[key]):
            raise TrainingError(
                f"{where}.{key}: the checkpoint recorded {record[key]!r}, but the "
                f"receiving sampler uses {expected[key]!r}; the sampled time range or "
                "radial floor would differ, so this is not an exact resume"
            )
    saved_counts = record[SAMPLING_COUNTS_KEY]
    receiving_counts = expected[SAMPLING_COUNTS_KEY]
    differing = [
        key for key in SAMPLING_COUNT_KEYS
        if int(saved_counts[key]) != int(receiving_counts[key])
    ]
    if differing:
        raise TrainingError(
            f"{where}.counts: the sampling point counts differ in {differing}"
        )
    saved_stratification = record[SAMPLING_STRATIFICATION_KEY]
    receiving_stratification = expected[SAMPLING_STRATIFICATION_KEY]
    differing = [
        key for key in SAMPLING_STRATIFICATION_KEYS
        if not _same_number(saved_stratification[key], receiving_stratification[key])
    ]
    if differing:
        raise TrainingError(
            f"{where}.stratification: the layered sampling settings differ in "
            f"{differing}"
        )
    if _normalise(record[SAMPLING_RAR_KEY]) != _normalise(
        expected[SAMPLING_RAR_KEY]
    ):
        raise TrainingError(
            f"{where}.{SAMPLING_RAR_KEY}: the RAR settings differ from the receiving "
            "sampler; the candidate pool or replacement rule would change"
        )
    try:
        sampler.validate_window_state(record[SAMPLING_WINDOW_KEY])
    except Exception as exc:  # noqa: BLE001 - keep the sampler's actual reason
        raise TrainingError(f"{where}.window: {exc}") from exc


def _check_sampler_startup_binding(
    sampler: Any,
    identity: Mapping[str, Any],
    scale_record: Any,
    rebuilt_scales: PlanarStartupScales,
    field: PlanarStartupField | None,
    *,
    where: str,
) -> None:
    sampler_field = getattr(sampler, "field", None)
    try:
        _require_field(sampler_field)
    except TrainingError as exc:
        raise TrainingError(f"{where}.field: {exc}") from exc
    _check_startup_identity(
        identity, sampler_field, rebuilt_scales, where=f"{where}.startup"
    )
    sampler_scales = getattr(sampler, "scales", None)
    if not isinstance(sampler_scales, PlanarStartupScales):
        raise TrainingError(
            f"{where}.scales: the sampler's scales are "
            f"{type(sampler_scales).__name__}, not PlanarStartupScales"
        )
    _check_scales_match(scale_record, sampler_scales, where=f"{where}.scales")
    if field is not None and sampler_field.cache_key() != field.cache_key():
        raise TrainingError(
            f"{where}.field: the sampler samples startup field "
            f"{sampler_field.cache_key()!r}, but the handoff provided is "
            f"{field.cache_key()!r}"
        )


# --------------------------------------------------------------------------
# 模型结构、RNG
# --------------------------------------------------------------------------


def _model_structure(model: Any) -> Any:
    if not hasattr(model, "describe") or not hasattr(model, "state_dict"):
        raise TrainingError(
            f"the checkpoint needs a CoupledModel-like object, got {type(model).__name__}"
        )
    parameters = list(model.parameters())
    if not parameters:
        raise TrainingError("the model has no trainable parameters")
    describe = dict(model.describe())
    # 顶层 ``scales`` 含 charge_radius/epsilon，是**算例**身份，不是共享结构。
    # 共享网络的材料/参考尺度与前沿无量纲初值仍在 describe 里，会被核对。
    describe.pop("scales", None)
    return _normalise({"describe": describe, "dtype": str(parameters[0].dtype)})


def _optimizer_reference(optimizer: Any) -> tuple[list[list[int]], dict[int, tuple[int, ...]]]:
    state_dict = optimizer.state_dict()
    groups = [list(group["params"]) for group in state_dict.get("param_groups", [])]
    tensors = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    ids = [int(pid) for group in groups for pid in group]
    if len(ids) != len(tensors):
        raise TrainingError(
            "the optimizer's parameter-group ids do not line up with its parameters"
        )
    shapes: dict[int, tuple[int, ...]] = {}
    for pid, parameter in zip(ids, tensors):
        shapes[pid] = tuple(parameter.shape)
    return [[int(pid) for pid in group] for group in groups], shapes


def collect_rng_state(*, sampler: Any = None) -> dict[str, Any]:
    import random

    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": None if sampler is None else sampler.rng_state(),
    }


def _validate_rng_state(
    state: Any, *, where: str, error: type[Exception] = TrainingError
) -> None:
    if not isinstance(state, Mapping):
        raise error(f"{where}: the RNG state is {type(state).__name__}, not a mapping")
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise error(f"{where}: the RNG state has no {key!r}")
    if not isinstance(state["python"], tuple) or len(state["python"]) != 3:
        raise error(f"{where}: the Python RNG state is not a 3-tuple")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise error(f"{where}: the NumPy RNG state is not a 5-tuple")
    if not _is_tensor_like(state["torch"]):
        raise error(f"{where}: the Torch RNG state is not a tensor")
    sampler_state = state.get("sampler")
    if sampler_state is not None and not isinstance(sampler_state, Mapping):
        raise error(f"{where}: the sampler RNG state is not a mapping")


def _apply_rng_state(state: Mapping[str, Any], sampler: Any) -> None:
    import random

    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("sampler") is not None:
        if sampler is None:
            raise TrainingError(
                "the checkpoint carries a sampler RNG state but no sampler was provided"
            )
        sampler.set_rng_state(state["sampler"])


def _package_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _require_new_checkpoint(payload: Any, path: Path) -> None:
    if not isinstance(payload, Mapping):
        raise TrainingError(f"{path}: the checkpoint is not a mapping")
    stored_format = payload.get("format")
    if stored_format != CHECKPOINT_FORMAT:
        raise TrainingError(
            f"{path}: unsupported checkpoint format {stored_format!r}. This build reads "
            f"{CHECKPOINT_FORMAT!r} (version {CHECKPOINT_VERSION}). Old reference-field "
            "checkpoints (format 'spherical_blast_pinn.checkpoint', version <= 8, with "
            "detonation velocity and T_h = R_0/D + epsilon) are not migrated: their time "
            "origin cannot be rebuilt without guessing. Regenerate the checkpoint with "
            "the new startup chain."
        )
    stored_version = payload.get("version")
    if not _is_int_like(stored_version) or int(stored_version) != CHECKPOINT_VERSION:
        raise TrainingError(
            f"{path}: checkpoint version {stored_version!r} is not supported; this "
            f"build reads version {CHECKPOINT_VERSION}. Version 2 stored one model and "
            "one optimizer per radius; selecting one radius's weights would not be the "
            "shared model these runs use. Version 3 compared only JWL and three air "
            "values, not the initial product/air thermodynamic state, so its material "
            "record cannot prove that a resume or frozen prediction uses the same "
            "physical conditions. Version 4 has no development split or RAR pool, so "
            "it cannot be resumed exactly. None of these formats is migrated; "
            "regenerate with the current shared-model chain."
        )
    if payload.get(PURPOSE_KEY) != CHECKPOINT_PURPOSE:
        raise TrainingError(
            f"{path}: checkpoint purpose {payload.get(PURPOSE_KEY)!r} is not "
            f"{CHECKPOINT_PURPOSE!r}; it belongs to a different checkpoint contract"
        )
    model_block = payload.get(MODEL_KEY)
    if not isinstance(model_block, Mapping):
        raise TrainingError(
            f"{path}: the checkpoint has no 'model' block; a shared-model checkpoint "
            "must carry exactly one set of trainable parameters"
        )
    missing_model = [key for key in MODEL_BLOCK_KEYS if key not in model_block]
    if missing_model:
        raise TrainingError(f"{path}: the model block is missing {missing_model}")
    if not isinstance(payload.get(CASES_KEY), list) or not payload[CASES_KEY]:
        raise TrainingError(
            f"{path}: the checkpoint has no non-empty 'cases' list; it cannot bind a "
            "training radius"
        )
    if TRAINING_KEY not in payload:
        raise TrainingError(
            f"{path}: the checkpoint has no {TRAINING_KEY!r} block. Version 2 always "
            "writes it (a model-only checkpoint writes resumable=False); a file without "
            "it cannot say whether it may be resumed from"
        )


# --------------------------------------------------------------------------
# 训练动态状态
# --------------------------------------------------------------------------


def _stage_windows(schedule: Sequence[Any] | None) -> dict[str, float]:
    return {
        str(stage.name): float(stage.time_window_fraction)
        for stage in (schedule or [])
    }


def _validate_training_state(
    state: Any,
    *,
    where: str,
    schedule: Sequence[Any] | None = None,
    error: type[Exception] = TrainingError,
) -> None:
    """训练状态**只读**校验；不修改任何对象。

    ``schedule`` 给出时，步号、阶段与窗口必须与这份日程一致；配置改变不能
    冒充精确续训。``schedule=None`` 时只做内部自洽检查（模型权重读取不需要
    日程）。
    """
    if not isinstance(state, Mapping):
        raise error(f"{where}: the training state is {type(state).__name__}")
    missing = [key for key in TRAINING_KEYS if key not in state]
    if missing:
        raise error(f"{where}: the training state is missing {missing}")
    step = require_step_value(state.get(STEP_KEY), field="step", where=where, error=error)
    if not isinstance(state.get("resumable"), (bool, np.bool_)):
        raise error(f"{where}: 'resumable' must be a boolean")
    stage = state.get("stage")
    if not isinstance(stage, Mapping):
        raise error(f"{where}: no 'stage' block")
    for key in ("current", "window_fraction", "blocked", "pending"):
        if key not in stage:
            raise error(f"{where}: the stage block has no {key!r}")
    current = stage["current"]
    pending = stage["pending"]
    if current is not None and not isinstance(current, str):
        raise error(f"{where}: stage.current is {current!r}, neither None nor a name")
    if pending is not None and not isinstance(pending, str):
        raise error(f"{where}: stage.pending is {pending!r}, neither None nor a name")
    if not isinstance(stage["blocked"], (bool, np.bool_)):
        raise error(f"{where}: stage.blocked is not a boolean")
    window = _real_number(stage["window_fraction"])
    if window is None or not (0.0 < window <= 1.0):
        raise error(f"{where}: stage.window_fraction is {stage['window_fraction']!r}")

    recorded_schedule = state.get("schedule")
    if not isinstance(recorded_schedule, list):
        raise error(f"{where}: 'schedule' is not a list")
    if schedule is not None:
        expected = [_normalise(item.as_dict()) for item in schedule]
        actual = [_normalise(item) for item in recorded_schedule]
        if actual != expected:
            raise error(
                f"{where}: the checkpoint's schedule differs from this run's schedule; "
                "a different stage budget, window or weight ramp is not an exact resume"
            )
        windows = _stage_windows(schedule)
        names = set(windows)
        if current is not None and current not in names:
            raise error(f"{where}: stage.current={current!r} is not in the schedule")
        if pending is not None and pending not in names:
            raise error(f"{where}: stage.pending={pending!r} is not in the schedule")
        if bool(stage["blocked"]):
            if pending is None or current != pending:
                raise error(
                    f"{where}: a blocked stage must record current == pending, got "
                    f"current={current!r}, pending={pending!r}"
                )
            if abs(window - windows[pending]) > 1.0e-12:
                raise error(
                    f"{where}: the blocked stage window is {window}, but stage "
                    f"{pending!r} uses {windows[pending]}"
                )
        else:
            if pending is not None:
                raise error(
                    f"{where}: stage.pending={pending!r} without blocked=True"
                )
            expected_stage, _ = _stage_and_fraction(step, schedule)
            expected_name = expected_stage.name
            if current is None:
                if step != 0:
                    raise error(
                        f"{where}: stage.current is None at step {step}; only a "
                        "not-yet-started run may omit it"
                    )
                expected_name = schedule[0].name
            elif current != expected_name:
                raise error(
                    f"{where}: stage.current={current!r} does not match step {step} "
                    f"of this schedule ({expected_name!r})"
                )
            if abs(window - windows[expected_name]) > 1.0e-12:
                raise error(
                    f"{where}: stage.window_fraction={window} does not match the "
                    f"window of stage {expected_name!r} ({windows[expected_name]})"
                )

    balancing = state.get("balancing")
    if not isinstance(balancing, Mapping):
        raise error(f"{where}: 'balancing' is not a mapping")
    if not isinstance(balancing.get("config"), Mapping):
        raise error(f"{where}: the balancing block has no config mapping")
    if not isinstance(balancing.get("state"), Mapping):
        raise error(f"{where}: the balancing block has no state mapping")
    _validate_rng_state(state.get("rng"), where=f"{where}: rng", error=error)
    for key in ("history", "stage_history", "check_history"):
        if not isinstance(state.get(key), list):
            raise error(f"{where}: {key!r} is not a list")
    best = state.get("best")
    if best is not None:
        if not isinstance(best, Mapping):
            raise error(f"{where}: 'best' is neither None nor a mapping")
        require_step_value(
            best.get("step"), field="best.step", where=where, error=error
        )
        if not isinstance(best.get("score"), (list, tuple)):
            raise error(f"{where}: best.score is not a list")
        if not isinstance(best.get("parameter_fingerprint"), str):
            raise error(f"{where}: best.parameter_fingerprint is not a string")
        best_model = best.get("model_state")
        if best_model is not None:
            if not isinstance(best_model, Mapping) or not best_model:
                raise error(f"{where}: best.model_state is not a non-empty mapping")
            if parameter_fingerprint_of(best_model) != best.get(
                "parameter_fingerprint"
            ):
                raise error(
                    f"{where}: the best model weights do not hash to the recorded "
                    "parameter fingerprint"
                )
    origin = state.get("run_origin")
    if not isinstance(origin, Mapping):
        raise error(f"{where}: 'run_origin' is not a mapping")
    for key in ("explicit_diagnostic", "refined"):
        if not isinstance(origin.get(key), (bool, np.bool_)):
            raise error(f"{where}: run_origin[{key!r}] is not a boolean")
    diagnostic = state.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise error(f"{where}: 'diagnostic' is not a mapping")
    if not isinstance(diagnostic.get("enabled"), (bool, np.bool_)):
        raise error(f"{where}: diagnostic.enabled is not a boolean")
    cap = diagnostic.get("max_updates")
    if cap is not None and (not _is_int_like(cap) or int(cap) < 0):
        raise error(f"{where}: diagnostic.max_updates is {cap!r}")
    settings = state.get("settings")
    if not isinstance(settings, Mapping):
        raise error(f"{where}: no 'settings' block")
    clip = _real_number(settings.get("gradient_clip"))
    if clip is None or clip < 0.0:
        raise error(f"{where}: settings.gradient_clip is {settings.get('gradient_clip')!r}")
    if not isinstance(settings.get("tolerances"), Mapping):
        raise error(f"{where}: settings.tolerances is not a mapping")
    if not isinstance(settings.get("balancing_config"), Mapping):
        raise error(f"{where}: settings.balancing_config is not a mapping")
    if not _is_int_like(settings.get("diagnostic_max_updates")):
        raise error(
            f"{where}: settings.diagnostic_max_updates is not an integer"
        )


def _stage_and_fraction(step: int, schedule: Sequence[Any]):
    remaining = int(step)
    for stage in schedule:
        if remaining < stage.updates:
            return stage, (remaining / stage.updates if stage.updates else 0.0)
        remaining -= stage.updates
    return schedule[-1], 1.0


# --------------------------------------------------------------------------
# 保存
# --------------------------------------------------------------------------


def _require_case_attribute(case: Any, name: str, where: str) -> Any:
    if not hasattr(case, name):
        raise TrainingError(
            f"{where}: a training case must expose {name!r}; got {type(case).__name__}"
        )
    return getattr(case, name)


def _initial_front(field: PlanarStartupField) -> dict[str, float]:
    return {
        "contact_hat": float(field.contact_radius / field.charge_radius),
        "gap_hat": float(
            (field.shock_radius - field.contact_radius) / field.charge_radius
        ),
    }


def _material_signature(field: PlanarStartupField) -> dict[str, Any]:
    """The one authoritative material/environment/initial-state record.

    The definition lives in :mod:`...physics.material_state`; the checkpoint
    does not maintain a second, weaker "same material" predicate.
    """
    try:
        return _normalise(field_material_conditions(field))
    except MaterialConditionsError as exc:
        raise TrainingError(f"the startup field has no usable material state: {exc}") from exc


def _check_model_material(
    model: Any, record: Any, *, where: str
) -> None:
    """Require the shared model's EOS/ambient state to match a case record."""
    try:
        actual = model_material_conditions(model)
    except MaterialConditionsError as exc:
        raise TrainingError(f"{where}: {exc}") from exc
    problems = condition_mismatches(
        record,
        actual,
        reference_label="checkpoint material record",
        actual_label="shared model",
    )
    if problems:
        raise TrainingError(
            f"{where}: the shared model's EOS/ambient state differs from the "
            "checkpoint's case material record: " + "; ".join(problems)
        )


def _check_record_materials(records: Sequence[Mapping[str, Any]], *, where: str) -> None:
    """All cases in one shared-model checkpoint must carry the same conditions."""
    if not records:
        return
    reference = records[0].get(MATERIAL_KEY)
    if not isinstance(reference, Mapping):
        raise TrainingError(f"{where}: the first case has no material record")
    for index, record in enumerate(records[1:], 1):
        material = record.get(MATERIAL_KEY)
        if not isinstance(material, Mapping):
            raise TrainingError(f"{where}: case {index} has no material record")
        problems = condition_mismatches(
            reference,
            material,
            reference_label="case 0 material record",
            actual_label=f"case {index} material record",
        )
        if problems:
            raise TrainingError(
                f"{where}: cases do not share the same material/environment/initial "
                "state: " + "; ".join(problems)
            )


def _structure_material_conditions(structure: Any, *, where: str) -> dict[str, Any]:
    """Read the EOS/ambient conditions out of the saved model structure."""
    if not isinstance(structure, Mapping):
        raise TrainingError(f"{where}: the model structure is not a mapping")
    describe = structure.get("describe")
    if not isinstance(describe, Mapping):
        raise TrainingError(f"{where}: the model structure has no describe mapping")
    product = describe.get("product")
    air = describe.get("air")
    ambient = describe.get("undisturbed")
    if not isinstance(product, Mapping) or not isinstance(air, Mapping) or not isinstance(
        ambient, Mapping
    ):
        raise TrainingError(
            f"{where}: the model structure lacks product/air/undisturbed conditions"
        )
    try:
        gamma = float(air["gamma_air"])
        density = float(ambient["rho_ambient"])
        pressure = float(ambient["p_ambient"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError(
            f"{where}: the model structure has no usable air/ambient state: {exc}"
        ) from exc
    if gamma <= 1.0 or density <= 0.0 or pressure <= 0.0:
        raise TrainingError(
            f"{where}: the model structure's air state is inadmissible "
            f"(gamma={gamma!r}, rho={density!r}, p={pressure!r})"
        )
    return {
        "jwl": dict(product.get("jwl") or {}),
        "air_gamma": gamma,
        "initial_state": {
            "air_density": density,
            "air_specific_energy": pressure / ((gamma - 1.0) * density),
            "air_pressure": pressure,
            "air_sound_speed": math.sqrt(gamma * pressure / density),
        },
    }


def _check_structure_material(
    structure: Any, record: Any, *, where: str
) -> None:
    """Bind the case material record to the saved model structure itself.

    This runs even when no live model is supplied (inspection / frozen
    prediction construction), so a checkpoint cannot pair one material record
    with a model structure that uses another EOS or ambient state.
    """
    actual = _structure_material_conditions(structure, where=where)
    problems = condition_mismatches(
        record,
        actual,
        reference_label="checkpoint material record",
        actual_label="checkpoint model structure",
    )
    if problems:
        raise TrainingError(
            f"{where}: the model structure's EOS/ambient state differs from the "
            "case material record: " + "; ".join(problems)
        )


def _case_payload(case: Any, *, where: str) -> dict[str, Any]:
    """只保存**算例上下文**；可训练参数是共享的，不在这里。"""
    field = _require_case_attribute(case, "field", where)
    scales = _require_case_attribute(case, "scales", where)
    sampler = getattr(case, "sampler", None)
    _require_field(field)
    record = startup_scale_record(scales)
    if not _same_number(record["startup_epsilon"], field.epsilon):
        raise TrainingError(
            f"{where}: the scales start at epsilon {record['startup_epsilon']!r} s but "
            f"the field is at {field.epsilon!r} s"
        )
    if not _same_number(record["charge_radius"], field.charge_radius):
        raise TrainingError(
            f"{where}: the scales belong to R_0 = {record['charge_radius']!r} m but the "
            f"field has {field.charge_radius!r} m"
        )
    if sampler is not None:
        _check_sampler_startup_binding(
            sampler, startup_identity(field), record, scales, field,
            where=f"{where}: sampler",
        )
    pressure = float(getattr(case, "contact_pressure", float("nan")))
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise TrainingError(
            f"{where}: contact pressure scale is {pressure!r}; must be finite positive"
        )
    data = getattr(case, "data", None)
    data_identity = None if data is None else _normalise(data.identity())
    if sampler is None:
        sampling_state = None
        sampler_rng = None
        rar_state = None
    else:
        sampling_state = sampling_record(sampler)
        sampler_rng = sampler.rng_state()
        sampler.validate_rng_state(sampler_rng)
        if not hasattr(sampler, "rar_state") or not hasattr(sampler, "validate_rar_state"):
            raise TrainingError(
                f"{where}: the sampler has no RAR state interface; a v5 checkpoint "
                "cannot record a resumable point pool"
            )
        rar_state = sampler.rar_state()
        sampler.validate_rar_state(rar_state)
    return {
        CASE_ID_KEY: str(_require_case_attribute(case, "case_id", where)),
        CASE_SPLIT_KEY: str(getattr(case, "split", "train")),
        STARTUP_KEY: startup_identity(field),
        SCALES_KEY: record,
        INITIAL_FRONT_KEY: _initial_front(field),
        MATERIAL_KEY: _material_signature(field),
        SAMPLING_KEY: sampling_state,
        SAMPLER_RNG_KEY: sampler_rng,
        RAR_KEY: rar_state,
        CONTACT_PRESSURE_KEY: pressure,
        DATA_KEY: data_identity,
    }


def _model_payload(model: Any, optimizer: Any = None) -> dict[str, Any]:
    """共享模型块：参数、优化器、系统一且**只有一份**。"""
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    validate_model_state(
        model_state, model.state_dict(), where="checkpoint: model_state"
    )
    optimizer_state = None
    if optimizer is not None:
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        reference_groups, parameter_shapes = _optimizer_reference(optimizer)
        validate_optimizer_state(
            optimizer_state,
            reference_param_groups=reference_groups,
            parameter_shapes=parameter_shapes,
            where="checkpoint: optimizer",
        )
    return {
        MODEL_STRUCTURE_KEY: _model_structure(model),
        MODEL_STATE_KEY: model_state,
        OPTIMIZER_KEY: optimizer_state,
        FINGERPRINT_KEY: parameter_fingerprint_of(model_state),
    }


def _model_only_training_state(step: int, *, sampler: Any = None) -> dict[str, Any]:
    return {
        STEP_KEY: int(step),
        "stage": {
            "current": None,
            "window_fraction": 1.0,
            "blocked": False,
            "pending": None,
        },
        "schedule": [],
        "balancing": {"config": {}, "state": {}},
        "rng": collect_rng_state(sampler=sampler),
        "history": [],
        "stage_history": [],
        "check_history": [],
        "best": None,
        "run_origin": {"explicit_diagnostic": False, "refined": False},
        "diagnostic": {"enabled": False, "max_updates": None, "reason": None},
        "settings": {
            "gradient_clip": 0.0,
            "tolerances": {},
            "balancing_config": {},
            "diagnostic_max_updates": 1,
        },
        "resumable": False,
    }


def save_startup_checkpoint(
    path: Path | str,
    *,
    field: PlanarStartupField | None = None,
    scales: PlanarStartupScales | None = None,
    model: Any = None,
    optimizer: Any = None,
    step: int = 0,
    sampler: Any = None,
    cases: Sequence[Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """写出 v5 共享模型检查点（一个模型 + 一个或多个算例上下文）。

    训练入口应传 ``cases``、``model``、``optimizer`` 与 ``training_state``；
    只给 ``field/scales/model`` 时写一个不可续训的模型检查点。
    """
    import torch

    if model is None:
        raise TrainingError("a shared-model checkpoint needs one model")
    if cases is None:
        if field is None or scales is None:
            raise TrainingError(
                "save_startup_checkpoint needs either cases=[...] or field= and scales="
            )
        case_id = getattr(sampler, "case_id", None) or (
            "planar_startup:" + field.cache_key()[:16]
        )
        minimal = type("_SingleCase", (), {})()
        minimal.case_id = case_id
        minimal.split = "train"
        minimal.field = field
        minimal.scales = scales
        minimal.sampler = sampler
        minimal.data = None
        from ..losses.objective import contact_pressure_scale

        minimal.contact_pressure = contact_pressure_scale(field)
        cases = [minimal]
    payload_cases = [
        _case_payload(case, where=f"save_startup_checkpoint: case {index}")
        for index, case in enumerate(cases)
    ]
    ids = [case[CASE_ID_KEY] for case in payload_cases]
    if len(set(ids)) != len(ids):
        raise TrainingError(f"duplicate case ids in checkpoint: {ids}")
    _check_record_materials(
        payload_cases, where="save_startup_checkpoint: material consistency"
    )
    _check_model_material(
        model,
        payload_cases[0][MATERIAL_KEY],
        where="save_startup_checkpoint: model/material",
    )
    if training_state is None:
        training_state = _model_only_training_state(int(step), sampler=sampler)
    else:
        _validate_training_state(
            training_state, where="save_startup_checkpoint: training"
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        PURPOSE_KEY: CHECKPOINT_PURPOSE,
        MODEL_KEY: _model_payload(model, optimizer),
        CASES_KEY: payload_cases,
        TRAINING_KEY: training_state,
        CODE_VERSION_KEY: _package_version(),
    }
    for key, value in dict(extra or {}).items():
        if key in payload:
            raise TrainingError(
                f"extra metadata {key!r} would overwrite a reserved checkpoint field"
            )
        payload[key] = value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(target)
    return target


def save_training_checkpoint(
    path: Path | str,
    *,
    cases: Sequence[Any],
    model: Any,
    optimizer: Any,
    training_state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """训练入口的保存接口：一个共享模型 + 全部算例上下文。"""
    return save_startup_checkpoint(
        path,
        cases=cases,
        model=model,
        optimizer=optimizer,
        training_state=training_state,
        extra=extra,
    )


def _require_case_record(record: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TrainingError(f"{where}: {type(record).__name__}, not a case record")
    missing = [key for key in CASE_RECORD_KEYS if key not in record]
    if missing:
        raise TrainingError(f"{where}: missing {missing}")
    try:
        validate_material_record(record[MATERIAL_KEY], where=f"{where}: material")
    except MaterialConditionsError as exc:
        raise TrainingError(str(exc)) from exc
    return dict(record)


def _check_initial_and_material(
    record: Mapping[str, Any], field: PlanarStartupField, *, where: str
) -> None:
    front = record.get(INITIAL_FRONT_KEY)
    if not isinstance(front, Mapping):
        raise TrainingError(f"{where}: no initial-front record")
    actual = _initial_front(field)
    for key in ("contact_hat", "gap_hat"):
        value = _real_number(front.get(key))
        if value is None or not _same_number(value, actual[key]):
            raise TrainingError(
                f"{where}: the field's initial {key} is {actual[key]!r}, but the "
                f"checkpoint records {front.get(key)!r}; the shared front bias was "
                "built for a different startup geometry"
            )
    material = record.get(MATERIAL_KEY)
    if not isinstance(material, Mapping):
        raise TrainingError(f"{where}: no material signature")
    problems = condition_mismatches(
        material,
        _material_signature(field),
        reference_label="checkpoint material record",
        actual_label="startup field",
    )
    if problems:
        raise TrainingError(
            f"{where}: the field's JWL/air/environment or initial product/air state "
            "differs from the checkpoint's shared model; refusing to mix materials: "
            + "; ".join(problems)
        )


def _validate_case_record(
    record: Mapping[str, Any],
    *,
    where: str,
    target: Any = None,
    handoff_field: PlanarStartupField | None = None,
    weights_only: bool = False,
    allow_new_radius: bool = False,
) -> dict[str, Any]:
    """检查一个算例上下文；不涉及共享模型参数。"""
    rebuilt = planar_scales_from_record(record[SCALES_KEY], where=f"{where}: scales")
    identity = record[STARTUP_KEY]
    if not isinstance(identity, Mapping):
        raise TrainingError(f"{where}: startup identity is not a mapping")
    for key in (STARTUP_MODEL_ID_KEY, STARTUP_METHOD_VERSION_KEY, STARTUP_CACHE_KEY):
        if not isinstance(identity.get(key), str) or not identity.get(key):
            raise TrainingError(f"{where}: startup identity has no usable {key!r}")
    if _real_number(identity.get(STARTUP_DELTA_KEY)) is None:
        raise TrainingError(f"{where}: startup identity has no usable delta")
    rar_state = record.get(RAR_KEY)
    if record[SAMPLING_KEY] is not None:
        _validate_sampling_record(record[SAMPLING_KEY], where=f"{where}: sampling")
        try:
            rar_config = RARConfig.from_mapping(
                record[SAMPLING_KEY][SAMPLING_RAR_KEY]
            )
            validate_rar_state(
                rar_state, config=rar_config, where=f"{where}: RAR state"
            )
        except RARError as exc:
            raise TrainingError(str(exc)) from exc
    elif rar_state is not None:
        raise TrainingError(
            f"{where}: a model-only case record carries RAR state without sampling state"
        )
    pressure = _real_number(record[CONTACT_PRESSURE_KEY])
    if pressure is None or not math.isfinite(pressure) or pressure <= 0.0:
        raise TrainingError(
            f"{where}: contact pressure {record[CONTACT_PRESSURE_KEY]!r} is not finite "
            "positive"
        )
    sampler_state = record[SAMPLER_RNG_KEY]
    if sampler_state is not None and not isinstance(sampler_state, Mapping):
        raise TrainingError(f"{where}: sampler RNG state is not a mapping")

    sampler = getattr(target, "sampler", None) if target is not None and not weights_only else None
    if sampler is not None:
        if record[SAMPLING_KEY] is None:
            raise TrainingError(
                f"{where}: the checkpoint carries no sampler state, but a sampler was "
                "supplied"
            )
        _check_sampling_record(record[SAMPLING_KEY], sampler, where=f"{where}: sampling")
        if sampler_state is None:
            raise TrainingError(f"{where}: sampler RNG state is missing")
        try:
            sampler.validate_rng_state(sampler_state)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"{where}: sampler RNG state: {exc}") from exc
        try:
            sampler.validate_rar_state(rar_state)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"{where}: RAR state: {exc}") from exc
        _check_sampler_startup_binding(
            sampler,
            record[STARTUP_KEY],
            record[SCALES_KEY],
            rebuilt,
            handoff_field if allow_new_radius else None,
            where=f"{where}: sampler",
        )

    target_field = getattr(target, "field", None) if target is not None else None
    if handoff_field is not None:
        if allow_new_radius:
            # 未训练半径：允许 cache_key 不同，但材料/方法/delta/无量纲初值必须相同。
            for key in (STARTUP_MODEL_ID_KEY, STARTUP_METHOD_VERSION_KEY):
                if identity.get(key) != getattr(handoff_field, {
                    STARTUP_MODEL_ID_KEY: "model_id",
                    STARTUP_METHOD_VERSION_KEY: "method_version",
                }[key]):
                    raise TrainingError(
                        f"{where}: new-radius handoff has {key}="
                        f"{getattr(handoff_field, {STARTUP_MODEL_ID_KEY: 'model_id', STARTUP_METHOD_VERSION_KEY: 'method_version'}[key])!r}, "
                        f"checkpoint has {identity.get(key)!r}"
                    )
            if not _same_number(identity.get(STARTUP_DELTA_KEY), handoff_field.delta):
                raise TrainingError(
                    f"{where}: new-radius delta={handoff_field.delta!r}, checkpoint "
                    f"delta={identity.get(STARTUP_DELTA_KEY)!r}"
                )
            _check_initial_and_material(record, handoff_field, where=where)
        else:
            if target_field is not None and target_field.cache_key() != handoff_field.cache_key():
                raise TrainingError(
                    f"{where}: the receiving case uses startup "
                    f"{target_field.cache_key()!r}, but the handoff provided is "
                    f"{handoff_field.cache_key()!r}"
                )
            target_field = handoff_field
    if target_field is not None:
        _require_field(target_field)
        _check_startup_identity(
            identity, target_field, rebuilt, where=f"{where}: startup"
        )
        if allow_new_radius:
            _check_initial_and_material(record, target_field, where=where)
    target_scales = getattr(target, "scales", None) if target is not None else None
    if target_scales is not None:
        _check_scales_match(record[SCALES_KEY], target_scales, where=f"{where}: scales")

    data = getattr(target, "data", None) if target is not None and not weights_only else None
    recorded_data = record.get(DATA_KEY)
    if not weights_only:
        if data is not None and recorded_data is not None:
            if _normalise(data.identity()) != _normalise(recorded_data):
                raise TrainingError(
                    f"{where}: the supervision dataset differs from the one this "
                    "checkpoint was trained on (content, time origin or validity changed)"
                )
        elif data is not None and recorded_data is None:
            raise TrainingError(
                f"{where}: the checkpoint was trained without supervision, but this "
                "case carries a supervision dataset"
            )
        elif data is None and recorded_data is not None:
            raise TrainingError(
                f"{where}: the checkpoint was trained with supervision, but this case "
                "has no supervision dataset"
            )
    return {
        "case_id": str(record[CASE_ID_KEY]),
        "split": str(record.get(CASE_SPLIT_KEY, "train")),
        "scales": rebuilt,
        "startup": dict(identity),
        "sampling": record[SAMPLING_KEY],
        "contact_pressure_pa": float(pressure),
        "sampler_state": sampler_state,
        "rar_state": rar_state,
        "data": recorded_data,
        "initial_front": dict(record[INITIAL_FRONT_KEY]),
    }


def _case_targets(cases: Sequence[Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for case in cases:
        case_id = str(_require_case_attribute(case, "case_id", "cases="))
        if case_id in targets:
            raise TrainingError(f"duplicate receiving case_id {case_id!r}")
        targets[case_id] = case
    return targets


def _validate_model_block(
    block: Any,
    *,
    model: Any,
    optimizer: Any,
    weights_only: bool,
    where: str,
) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        raise TrainingError(f"{where}: the model block is not a mapping")
    missing = [key for key in MODEL_BLOCK_KEYS if key not in block]
    if missing:
        raise TrainingError(f"{where}: the model block is missing {missing}")
    model_state = block[MODEL_STATE_KEY]
    if not isinstance(model_state, Mapping) or not model_state:
        raise TrainingError(f"{where}: model_state is empty or not a mapping")
    if block.get(FINGERPRINT_KEY) != parameter_fingerprint_of(model_state):
        raise TrainingError(
            f"{where}: the recorded parameter fingerprint does not match the stored "
            "parameters; the checkpoint is damaged"
        )
    if model is not None:
        if _normalise(block[MODEL_STRUCTURE_KEY]) != _model_structure(model):
            raise TrainingError(
                f"{where}: the checkpoint's shared network structure or EOS "
                "configuration does not match this model"
            )
        validate_model_state(
            model_state, model.state_dict(), where=f"{where}: model_state"
        )
    optimizer_state = block.get(OPTIMIZER_KEY)
    if optimizer is not None and not weights_only:
        if model is None:
            raise TrainingError(f"{where}: restoring an optimizer needs its model")
        if optimizer_state is None:
            raise TrainingError(
                f"{where}: the checkpoint carries no optimizer state, but an optimizer "
                "was supplied"
            )
        reference_groups, parameter_shapes = _optimizer_reference(optimizer)
        validate_optimizer_state(
            optimizer_state,
            reference_param_groups=reference_groups,
            parameter_shapes=parameter_shapes,
            where=f"{where}: optimizer",
        )
    return {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "fingerprint": block.get(FINGERPRINT_KEY),
        "structure": block.get(MODEL_STRUCTURE_KEY),
    }


def inspect_startup_checkpoint(path: Path | str) -> dict[str, Any]:
    """只读地取出一份检查点的共享模型块、case 记录与训练状态。"""
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require_new_checkpoint(payload, source)
    records = [
        _require_case_record(record, where=f"{source}: case {index}")
        for index, record in enumerate(payload[CASES_KEY])
    ]
    return {
        "path": str(source),
        "model": payload.get(MODEL_KEY),
        "cases": records,
        "training": payload.get(TRAINING_KEY),
        "code_version": payload.get(CODE_VERSION_KEY),
    }


def load_startup_checkpoint(
    path: Path | str,
    *,
    handoff_json: Path | str | None = None,
    model: Any = None,
    scales: PlanarStartupScales | None = None,
    optimizer: Any = None,
    sampler: Any = None,
    restore_rng: bool = True,
    cases: Sequence[Any] | None = None,
    schedule: Sequence[Any] | None = None,
    require_resumable: bool = False,
    weights_only: bool = False,
    select_case_id: str | None = None,
    expected_training_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """读取并核对 v5 共享模型检查点；给出目标对象时执行恢复。

    共享模型/优化器只校验、只应用**一次**；``cases`` 逐项核对场、尺度、采样
    配置、窗口、RNG 与监督数据。先完成全部校验，再修改任何对象。
    """
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require_new_checkpoint(payload, source)
    records = [
        _require_case_record(record, where=f"{source}: case {index}")
        for index, record in enumerate(payload[CASES_KEY])
    ]
    _check_record_materials(records, where=f"{source}: material consistency")
    _check_structure_material(
        payload[MODEL_KEY].get(MODEL_STRUCTURE_KEY),
        records[0][MATERIAL_KEY],
        where=f"{source}: structure/material",
    )
    if model is not None:
        _check_model_material(
            model, records[0][MATERIAL_KEY], where=f"{source}: model/material"
        )
    if weights_only:
        restore_rng = False
        require_resumable = False
    handoff_field = None
    if handoff_json is not None:
        handoff_field = _require_field(load_planar_startup_handoff(handoff_json))

    training_state = payload.get(TRAINING_KEY)
    if training_state is not None:
        _validate_training_state(
            training_state, where=f"{source}: training", schedule=schedule
        )
        if require_resumable and not bool(training_state.get("resumable")):
            raise TrainingError(
                f"{source}: this checkpoint is a model-only checkpoint "
                "(resumable=False); it cannot be used to resume a training trajectory"
            )
        if expected_training_settings is not None:
            actual = _normalise(training_state.get("settings"))
            expected = _normalise(expected_training_settings)
            if actual != expected:
                differing = sorted(
                    key for key in set(actual) | set(expected)
                    if actual.get(key) != expected.get(key)
                )
                raise TrainingError(
                    f"{source}: the checkpoint's training settings differ from this "
                    f"run's in {differing}; gradient clipping, acceptance tolerances or "
                    "the adaptive-weight configuration changed, so this is not an exact "
                    "resume"
                )

    if cases is not None:
        targets = _case_targets(cases)
        recorded_ids = [record[CASE_ID_KEY] for record in records]
        if len(set(recorded_ids)) != len(recorded_ids):
            raise TrainingError(f"{source}: duplicate case ids in the checkpoint")
        if set(recorded_ids) != set(targets):
            missing = sorted(set(recorded_ids) - set(targets))
            extra = sorted(set(targets) - set(recorded_ids))
            raise TrainingError(
                f"{source}: the checkpoint binds cases {recorded_ids}, but this run "
                f"has {sorted(targets)} (missing {missing}, extra {extra}); a "
                "multi-radius resume needs exactly the same case set"
            )
        record_by_id = {record[CASE_ID_KEY]: record for record in records}
        validations = {
            case_id: _validate_case_record(
                record_by_id[case_id],
                where=f"{source}: case {case_id!r}",
                target=target,
                handoff_field=handoff_field,
                weights_only=weights_only,
            )
            for case_id, target in targets.items()
        }
        ordered = [(case_id, targets[case_id]) for case_id in recorded_ids]
    else:
        if select_case_id is not None:
            selected = [
                record for record in records
                if str(record[CASE_ID_KEY]) == str(select_case_id)
            ]
            if not selected:
                raise TrainingError(
                    f"{source}: no case {select_case_id!r}; the checkpoint binds "
                    f"{[record[CASE_ID_KEY] for record in records]}"
                )
            record = selected[0]
        else:
            if len(records) != 1:
                raise TrainingError(
                    f"{source}: this checkpoint binds {len(records)} cases; pass cases= "
                    "with one target per case, or select_case_id= for a single case"
                )
            record = records[0]
        small = type("_Target", (), {})()
        small.case_id = record[CASE_ID_KEY]
        small.field = handoff_field
        small.scales = scales
        small.sampler = sampler
        small.data = None
        validations = {
            small.case_id: _validate_case_record(
                record,
                where=f"{source}: case {small.case_id!r}",
                target=small,
                handoff_field=handoff_field,
                weights_only=weights_only,
            )
        }
        ordered = [(small.case_id, small)]
        targets = {small.case_id: small}

    model_info = _validate_model_block(
        payload[MODEL_KEY],
        model=model,
        optimizer=optimizer,
        weights_only=weights_only,
        where=f"{source}: model",
    )

    if restore_rng and training_state is not None:
        _validate_rng_state(training_state.get(RNG_KEY), where=f"{source}: rng")
    model_backup = None
    optimizer_backup = None
    window_backups: dict[str, Any] = {}
    sampler_rng_backups: dict[str, Any] = {}
    rar_state_backups: dict[str, Any] = {}
    global_rng_backup = collect_rng_state() if restore_rng else None
    try:
        if model is not None:
            model_backup = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            model.load_state_dict(model_info["model_state"], strict=True)
        if optimizer is not None and not weights_only:
            optimizer_backup = copy.deepcopy(optimizer.state_dict())
            optimizer.load_state_dict(
                copy.deepcopy(model_info["optimizer_state"])
            )
        for case_id, target in ordered:
            info = validations[case_id]
            if target.sampler is not None and not weights_only:
                window_backups[case_id] = target.sampler.window_state()
                sampler_rng_backups[case_id] = target.sampler.rng_state()
                rar_state_backups[case_id] = target.sampler.rar_state()
                target.sampler.set_window_state(
                    info["sampling"][SAMPLING_WINDOW_KEY]
                )
                target.sampler.set_rng_state(info["sampler_state"])
                target.sampler.set_rar_state(info["rar_state"])
        if restore_rng and training_state is not None:
            state = dict(training_state[RNG_KEY])
            state["sampler"] = None
            _apply_rng_state(state, None)
    except Exception:
        if model is not None and model_backup is not None:
            model.load_state_dict(model_backup, strict=True)
        if optimizer is not None and optimizer_backup is not None:
            optimizer.load_state_dict(optimizer_backup)
        for case_id, target in ordered:
            if target.sampler is not None and case_id in window_backups:
                target.sampler.set_window_state(window_backups[case_id])
                target.sampler.set_rng_state(sampler_rng_backups[case_id])
                target.sampler.set_rar_state(rar_state_backups[case_id])
        if restore_rng and global_rng_backup is not None:
            _apply_rng_state(global_rng_backup, None)
        raise

    case_infos = [validations[case_id] for case_id, _ in ordered]
    first = case_infos[0]
    return {
        "path": str(source),
        "cases": case_infos,
        "case_ids": [info["case_id"] for info in case_infos],
        "scales": first["scales"],
        "step": (
            int(training_state[STEP_KEY])
            if training_state is not None
            else 0
        ),
        "sampling": first["sampling"],
        "contact_pressure_pa": first["contact_pressure_pa"],
        "startup": first["startup"],
        "training": training_state,
        "model_fingerprint": model_info["fingerprint"],
        "handoff_checked": handoff_field is not None,
        "restored_model": model is not None,
        "restored_optimizer": optimizer is not None and not weights_only,
        "restored_sampling": sampler is not None or cases is not None,
        "restored_rng": bool(restore_rng),
    }


def load_frozen_model(
    path: Path | str,
    *,
    handoff_json: Path | str,
    scales: PlanarStartupScales,
    model: Any,
) -> dict[str, Any]:
    """用**冻结**共享参数预测一个兼容的未训练半径。

    这不是续训：不读优化器/训练状态，不恢复 RNG，不修改任何参数。允许
    ``cache_key`` 与训练算例不同，但材料、环境、delta、方法、时间尺度与无量纲
    前沿初值必须一致。
    """
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require_new_checkpoint(payload, source)
    records = [
        _require_case_record(record, where=f"{source}: case {index}")
        for index, record in enumerate(payload[CASES_KEY])
    ]
    _check_record_materials(records, where=f"{source}: material consistency")
    _check_structure_material(
        payload[MODEL_KEY].get(MODEL_STRUCTURE_KEY),
        records[0][MATERIAL_KEY],
        where=f"{source}: structure/material",
    )
    field = _require_field(load_planar_startup_handoff(handoff_json))
    if not isinstance(scales, PlanarStartupScales):
        raise TrainingError(
            f"frozen prediction needs PlanarStartupScales, got {type(scales).__name__}"
        )
    if not _same_number(scales.charge_radius, field.charge_radius):
        raise TrainingError(
            f"the new field has R_0={field.charge_radius!r} m but the scales belong to "
            f"{scales.charge_radius!r} m"
        )
    if not _same_number(scales.startup_epsilon, field.epsilon):
        raise TrainingError(
            f"the scales start at {scales.startup_epsilon!r} s but the field is at "
            f"{field.epsilon!r} s"
        )
    first_record = records[0]
    trained_scales = planar_scales_from_record(
        first_record[SCALES_KEY], where=f"{source}: trained scales"
    )
    for key in (
        "reference_velocity",
        "reference_specific_energy",
        "product_density",
        "air_density",
        "time_scale",
    ):
        actual = float(getattr(scales, key))
        expected = float(getattr(trained_scales, key))
        if abs(actual - expected) > SCALES_RELATIVE_TOLERANCE * max(
            abs(actual), abs(expected), 1.0e-30
        ):
            raise TrainingError(
                f"new radius is not scale-compatible: {key}={actual!r}, checkpoint "
                f"uses {expected!r}"
            )
    _check_model_material(
        model, first_record[MATERIAL_KEY], where=f"{source}: frozen model/material"
    )
    # 允许新的 cache_key/半径，但其它身份与无量纲初值必须一致。
    _validate_case_record(
        first_record,
        where=f"{source}: frozen compatibility",
        handoff_field=field,
        allow_new_radius=True,
        weights_only=True,
    )
    model_info = _validate_model_block(
        payload[MODEL_KEY],
        model=model,
        optimizer=None,
        weights_only=True,
        where=f"{source}: model",
    )
    model.load_state_dict(model_info["model_state"], strict=True)
    return {
        "path": str(source),
        "mode": "frozen_shared_parameters",
        "trained_case_ids": [record[CASE_ID_KEY] for record in records],
        "model_fingerprint": model_info["fingerprint"],
        "charge_radius_m": float(field.charge_radius),
        "epsilon_s": float(field.epsilon),
        "scales": scales,
    }


def load_training_checkpoint(
    path: Path | str,
    *,
    cases: Sequence[Any],
    model: Any,
    optimizer: Any,
    schedule: Sequence[Any] | None = None,
    restore_rng: bool = True,
    expected_training_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """训练入口的读取接口：共享模型 + 完整 case 集合 + 训练状态。"""
    return load_startup_checkpoint(
        path,
        cases=cases,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        restore_rng=restore_rng,
        require_resumable=True,
        expected_training_settings=expected_training_settings,
    )




def parameter_fingerprint_of(state_dict: Mapping[str, Any]) -> str:
    """模型参数的**内容**指纹。**唯一实现**。

    ``Trainer.parameter_fingerprint()`` 调它。候选校验必须"从候选里的实际参数
    重算"，而"重算"用的必须是同一个算法——另写一套（哪怕看起来等价）会让两份
    指纹在某个 dtype 或遍历顺序的细节上分道扬镳，而那种分歧恰恰是这一项要防的。
    """
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "detach")


def validate_model_state(
    model: Any,
    expected: Mapping[str, Any],
    *,
    where: str,
    error: type[Exception] = TrainingError,
) -> None:
    """一份 ``state_dict`` 与**当前模型结构**相容吗。

    与"数值必须和当前参数相同"无关：候选本来就可能来自更早的一步，要求两者相等
    会把正常情形判成损坏。这里只问结构——键集合、张量类型、形状、dtype，以及
    **参数是否有限**。

    非有限参数是真损坏：更新前就有非有限拦截（``NonFiniteUpdate``），一段正常的
    轨迹不该产出 NaN/Inf 的权重。把这种候选发布成 ``best.pt`` 等于把一个坏模型
    写进"最佳"。
    """
    if not isinstance(model, Mapping):
        raise error(
            f"{where}: the model state is {type(model).__name__}, not a mapping of "
            "parameter name to tensor"
        )
    if not model:
        raise error(
            f"{where}: the model state is empty. An empty state dict loads without "
            "error into any model and then leaves the parameters untouched, so it reads "
            "exactly like a valid checkpoint while carrying no parameters at all"
        )
    missing = sorted(key for key in expected if key not in model)
    unexpected = sorted(key for key in model if key not in expected)
    if missing or unexpected:
        raise error(
            f"{where}: the model state does not match this model — missing "
            f"{missing[:4]}, unexpected {unexpected[:4]}"
        )
    for key in sorted(model):
        value = model[key]
        reference = expected[key]
        if not _is_tensor_like(value):
            raise error(
                f"{where}: the parameter {key!r} is {type(value).__name__}, not a tensor"
            )
        if tuple(value.shape) != tuple(reference.shape):
            raise error(
                f"{where}: the parameter {key!r} has shape {tuple(value.shape)}, but "
                f"this model's is {tuple(reference.shape)}"
            )
        if value.dtype != reference.dtype:
            raise error(
                f"{where}: the parameter {key!r} has dtype {value.dtype}, but this "
                f"model's is {reference.dtype}"
            )
        array = value.detach().cpu().numpy()
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            count = int((~np.isfinite(array)).sum())
            raise error(
                f"{where}: the parameter {key!r} holds {count} non-finite value(s); a "
                "candidate with NaN/Inf weights cannot be published as the best model"
            )


def _is_int_like(value: Any) -> bool:
    """``True`` 不是参数标识。``bool`` 是 ``int`` 的子类，直接 ``isinstance`` 会
    把它算成 ``1`` 号参数。"""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def require_step_value(
    value: Any,
    *,
    field: str,
    where: str,
    error: type[Exception] = TrainingError,
) -> int:
    """**一个**步号字段：先验原始类型与范围，再谈数值。**唯一的实现。**

    调用方（快照校验与最佳候选校验）各自只决定
    "哪些字段必须有、要和谁相等"；"什么才算一个步号"只有这一处说了算。

    **不先 ``int()``。** ``int("1")``、``int(1.0)``、``int(True)`` 全都得到 ``1``，
    那样一个字符串、一个浮点数、一个布尔值都会变成"看起来合法"的整数，而恢复端
    读到的仍是 ``"1"`` / ``1.0`` / ``True``——差别恰好被抹掉。接受的是契约本来就
    支持的 Python ``int`` 与 NumPy 整数；``bool`` 是 ``int`` 的子类，必须显式排除
    （Python 与 NumPy 的布尔值都不行）。

    合法范围沿用既有契约：非负。
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise error(
            f"{where}: {field} is {value!r} ({type(value).__name__}), not an integer. "
            "Step numbers are plain Python/NumPy integers — a missing key, None, a "
            "string, a float or a boolean is not coerced with int(): int('1'), int(1.0) "
            "and int(True) all yield 1 and would hide the difference between a valid "
            "record and a corrupt one"
        )
    if int(value) < 0:
        raise error(f"{where}: {field} is negative ({value!r})")
    return int(value)


#: 恢复端真正会读的优化器字段。``state`` 可以是空的（一次更新都没做过），
#: ``param_groups`` 不行——空字典 ``{}`` 缺的正是它，而 ``load_state_dict`` 第一
#: 个去读的也是它（复核实测的报错是 ``KeyError: 'param_groups'``）。次序照着真实
#: 读取路径排，诊断才会指向真正先炸的那一个。
STATE_OPTIMIZER_KEYS = ("param_groups", "state")

#: 优化器状态里必须**逐元素非负**的字段。
#:
#: 这两个都是**梯度的平方**（时间常数加权）。Adam 的更新是
#: ``exp_avg / (sqrt(exp_avg_sq) + eps)``：负值开平方得 NaN，**恢复之后的第一次
#: 更新**就把带这份状态的参数写成 NaN。``-1`` 是完全有限的值，所以"张量有限性"
#: 那条检查放它过去（复核实测）。
#:
#: ``max_exp_avg_sq`` 只在 ``amsgrad=True`` 时存在，语义与 ``exp_avg_sq`` 相同——
#: 按**字段名**判断，而不是先去读参数组里的 ``amsgrad`` 开关：开关与字段不一致
#: 时该信哪个是没有依据的，而"这个字段是平方量"是确定的。
#:
#: ``exp_avg`` **不在**此列：它是一阶矩，本来就可以为负。把它一起要求非负会把
#: 合法的训练历史判成损坏。
OPTIMIZER_NON_NEGATIVE_FIELDS = ("exp_avg_sq", "max_exp_avg_sq")


def validate_optimizer_state(
    optimizer: Any,
    *,
    reference_param_groups: Sequence[Sequence[int]],
    parameter_shapes: Mapping[int, Sequence[int]],
    where: str,
    error: type[Exception] = TrainingError,
) -> None:
    """优化器的 ``state_dict`` 与**这套参数组**相容吗。只看结构，不动任何状态。

    "合法初始状态允许 ``state`` 为空"**不等于**"整个优化器可以是空字典"。``{}``
    能通过"是不是 Mapping"那一关，然后在恢复时炸在 ``KeyError: 'param_groups'``
    ——而那时接收实例已经被改了一半（复核实测：候选换成 ``{}`` 之后，
    ``Trainer.load`` 接受、``run()`` 照样把 ``best.pt`` 写出去，只有**下一个**
    实例恢复 ``best.pt`` 时才报错）。

    只要求**候选自己带的**状态与它自己的参数相容，**不**要求它和当前优化器的状态
    一样多：候选可能来自更早的一步，那时还没有动量。反向是错的——"用当前的动量
    补齐一份损坏的记录"会把损坏洗掉。

    ``parameter_shapes`` 里没有的参数只跳过**形状**一项，标识与参数组的对应关系
    照样要过。形状规则是"要么与参数同形，要么是标量（``step`` 这类计数）"；
    标量那一档由调用方的探针优化器补上——``(1,)`` 的 ``exp_avg`` 能骗过宽播，
    但真实一步会抛错。
    """
    if not isinstance(optimizer, Mapping):
        raise error(
            f"{where}: the optimizer state is {type(optimizer).__name__}, not a "
            "mapping of 'state'/'param_groups'"
        )
    for key in STATE_OPTIMIZER_KEYS:
        if key not in optimizer:
            raise error(
                f"{where}: the optimizer state carries no {key!r} key. Restoring it "
                f"fails with KeyError({key!r}) — but only after the receiving instance "
                "has already been modified, so a corrupt checkpoint would be accepted "
                "at load time and only explode in whichever process resumes next"
            )

    state = optimizer["state"]
    if not isinstance(state, Mapping):
        raise error(
            f"{where}: the optimizer 'state' is {type(state).__name__}, not a mapping. "
            "An empty mapping is a legal initial state (no update has run yet); this is "
            "not one"
        )

    groups = optimizer["param_groups"]
    if isinstance(groups, (str, bytes)) or not isinstance(groups, (list, tuple)):
        raise error(
            f"{where}: the optimizer 'param_groups' is {type(groups).__name__}, not a "
            "list of parameter groups"
        )
    if len(groups) != len(reference_param_groups):
        raise error(
            f"{where}: the optimizer state has {len(groups)} parameter group(s), but "
            f"this model's optimizer has {len(reference_param_groups)}; restoring it "
            "would reassign the adaptive moments to the wrong parameters"
        )

    known_ids: list[int] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise error(
                f"{where}: optimizer parameter group {index} is "
                f"{type(group).__name__}, not a mapping"
            )
        params = group.get("params")
        if (
            isinstance(params, (str, bytes))
            or not isinstance(params, (list, tuple))
            or not params
            or not all(_is_int_like(value) for value in params)
        ):
            raise error(
                f"{where}: optimizer parameter group {index} carries "
                f"params={params!r}, which is not a non-empty list of parameter ids"
            )
        ids = [int(value) for value in params]
        expected = [int(value) for value in reference_param_groups[index]]
        if ids != expected:
            raise error(
                f"{where}: optimizer parameter group {index} refers to parameters "
                f"{ids[:6]}, but this model's optimizer group {index} refers to "
                f"{expected[:6]}; the state does not belong to this model's parameters"
            )
        known_ids.extend(ids)

    known = set(known_ids)
    for key in state:
        if not _is_int_like(key) or int(key) not in known:
            raise error(
                f"{where}: the optimizer carries state for {key!r}, which is not one of "
                "its parameter ids"
            )

    for key in sorted(state, key=lambda value: int(value)):
        entry = state[key]
        parameter = int(key)
        if not isinstance(entry, Mapping):
            raise error(
                f"{where}: the optimizer state for parameter {parameter} is "
                f"{type(entry).__name__}, not a mapping"
            )
        shape = tuple(parameter_shapes.get(parameter, ()))
        for name in sorted(entry, key=str):
            value = entry[name]
            if not _is_tensor_like(value):
                raise error(
                    f"{where}: the optimizer state {name!r} for parameter {parameter} is "
                    f"{type(value).__name__}, not a tensor"
                )
            array = value.detach().cpu().numpy()
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                count = int((~np.isfinite(array)).sum())
                raise error(
                    f"{where}: the optimizer state {name!r} for parameter {parameter} "
                    f"holds {count} non-finite value(s); the first update after "
                    "restoring would turn the parameters themselves non-finite"
                )
            # **有限不等于有效。** ``-1`` 是有限值，而它是梯度的平方的估计：
            # Adam 要开它的平方根。这四种"修法"都不许：取绝对值、截断到零、
            # 清空动量、``nan_to_num``——它们都把一份损坏的记录**改成能过**，
            # 而不是把它认出来；而且会静默换掉一段训练历史。
            if name in OPTIMIZER_NON_NEGATIVE_FIELDS and np.issubdtype(
                array.dtype, np.floating
            ):
                negative = int((array < 0).sum())
                if negative:
                    worst = float(array.min())
                    raise error(
                        f"{where}: the optimizer state {name!r} for parameter "
                        f"{parameter} holds {negative} negative value(s), down to "
                        f"{worst:.6g}. This field estimates a **squared** gradient and "
                        "Adam takes its square root before dividing by it, so the first "
                        "update after restoring yields NaN — for this parameter and "
                        "then for the whole model, since the loss built from it is not "
                        "finite either. Refusing rather than clamping to zero, taking "
                        "the absolute value or rebuilding the moments: each of those "
                        "would silently substitute a different training history"
                    )
            if shape and array.size > 1 and tuple(array.shape) != shape:
                raise error(
                    f"{where}: the optimizer state {name!r} for parameter {parameter} has "
                    f"shape {tuple(array.shape)}, but that parameter's shape is {shape}; "
                    "restoring it would misalign the adaptive moments with the weights"
                )


__all__ = [
    "CASES_KEY",
    "CASE_ID_KEY",
    "INITIAL_FRONT_KEY",
    "MATERIAL_KEY",
    "MODEL_BLOCK_KEYS",
    "MODEL_KEY",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_PURPOSE",
    "CHECKPOINT_VERSION",
    "CONTACT_PRESSURE_KEY",
    "DATA_KEY",
    "FINGERPRINT_KEY",
    "MODEL_STATE_KEY",
    "MODEL_STRUCTURE_KEY",
    "OPTIMIZER_KEY",
    "RAR_KEY",
    "RNG_KEY",
    "SAMPLER_RNG_KEY",
    "SAMPLING_KEY",
    "SAMPLING_RAR_KEY",
    "SAMPLING_RECORD_KEYS",
    "SCALES_KEY",
    "SCALES_RECORD_KEYS",
    "SCALES_RELATIVE_TOLERANCE",
    "STARTUP_KEY",
    "STEP_KEY",
    "TRAINING_KEY",
    "TrainingError",
    "collect_rng_state",
    "inspect_startup_checkpoint",
    "load_frozen_model",
    "load_startup_checkpoint",
    "load_training_checkpoint",
    "parameter_fingerprint_of",
    "planar_scales_from_record",
    "sampling_record",
    "save_startup_checkpoint",
    "save_training_checkpoint",
    "startup_identity",
    "startup_scale_record",
    "validate_model_state",
    "validate_optimizer_state",
]
