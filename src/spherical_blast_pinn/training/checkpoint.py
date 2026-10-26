"""检查点格式与读写规则的**唯一定义**（新启动场，单半径）。

这份检查点只服务一条链路：

.. code:: text

    cli.startup -> handoff.json/npz -> PlanarStartupField
    -> PlanarStartupScales -> 单半径物理批次 -> CoupledModel

它绑定的初始化身份是**启动场**：``model_id`` / ``method_version`` /
``cache_key``，以及 ``PlanarStartupScales`` 的完整参数（半径、``epsilon``、
``time_scale`` 与各材料尺度）。时间约定固定为
``t_phys = epsilon + tau``，``s = 0`` 对应 ``t_phys = epsilon``。

检查点**不复制**初场数组：加载时用显式提供的 ``handoff.json``（或调用方
自己的场与尺度）核对身份即可。

旧检查点（reference-field / 方案 B，``format = "spherical_blast_pinn.checkpoint"``
且带 ``detonation_velocity``、``start_time = T_h = R_0/D + epsilon``）在这里
**明确拒绝**：格式与版本都不同，没有迁移路径，也不猜测旧数据的时间原点。
新链路重新生成检查点。

保留的通用读取规则：发布标记（:func:`refuse_if_uncommitted` 及结构契约）、
模型状态结构/形状/有限性（:func:`validate_model_state`）、优化器状态
（:func:`validate_optimizer_state`）、步号（:func:`require_step_value`）与
模型参数指纹（:func:`parameter_fingerprint_of`）。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np

from ..physics.similitude import (
    TIME_CONVENTION_PLANAR_STARTUP,
    PlanarStartupScales,
)
from ..physics.two_medium_startup_handoff import load_planar_startup_handoff
from ..physics.two_medium_startup_planar import PlanarStartupField

#: 检查点格式标识与版本。旧格式的标识是
#: ``"spherical_blast_pinn.checkpoint"``（版本 ≤ 8，带爆速与 ``T_h``），
#: 读取端按格式与版本直接拒绝，不做迁移。
CHECKPOINT_FORMAT = "spherical_blast_pinn.startup_checkpoint"
CHECKPOINT_VERSION = 1

#: 用途标记。它不是版本号：同一版本下换用途（例如以后的多半径）必须换一个新值，
#: 否则读取端会把另一种结构的文件当成这一种。
CHECKPOINT_PURPOSE = "planar_startup_field_resume_single_radius"

#: 顶层字段名。
STARTUP_KEY = "startup"
SCALES_KEY = "scales"
MODEL_STRUCTURE_KEY = "model_structure"
MODEL_STATE_KEY = "model_state"
OPTIMIZER_KEY = "optimizer"
STEP_KEY = "step"
RNG_KEY = "rng"
SAMPLING_KEY = "sampling"
CONTACT_PRESSURE_KEY = "contact_pressure_pa"
FINGERPRINT_KEY = "parameter_fingerprint"
CODE_VERSION_KEY = "code_version"
PURPOSE_KEY = "purpose"

#: 启动场的身份字段。这里**不**重复记录半径/epsilon/time_scale；那些只在
#: ``scales`` 记录里保存一份，避免两份可相互矛盾的参数。
STARTUP_MODEL_ID_KEY = "model_id"
STARTUP_METHOD_VERSION_KEY = "method_version"
STARTUP_CACHE_KEY = "cache_key"
STARTUP_DELTA_KEY = "delta"

#: 采样配置与动态窗口。``tau_end``、点数与分层比例**实际改变样本分布**；窗口
#: （当前分数与历史）决定接下来采哪段时间。拿着另一套设置继续，不是精确续训。
SAMPLING_TAU_END_KEY = "tau_end_s"
SAMPLING_R_MIN_FRACTION_KEY = "r_min_fraction"
SAMPLING_COUNTS_KEY = "counts"
SAMPLING_STRATIFICATION_KEY = "stratification"
SAMPLING_WINDOW_KEY = "window"
SAMPLING_RECORD_KEYS = (
    SAMPLING_TAU_END_KEY,
    SAMPLING_R_MIN_FRACTION_KEY,
    SAMPLING_COUNTS_KEY,
    SAMPLING_STRATIFICATION_KEY,
    SAMPLING_WINDOW_KEY,
)
SAMPLING_COUNT_KEYS = (
    "n_pde_product",
    "n_pde_air",
    "n_ic_product",
    "n_ic_air",
    "n_contact",
    "n_shock",
)
SAMPLING_STRATIFICATION_KEYS = ("near_fraction", "near_span", "xi_floor")

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

#: 未提交标记的文件名。发布多份产物**不是**原子的：进程在两次写入之间被杀掉，
#: 目录里会留下"一部分新、一部分旧"的状态，而每一份单看都是合法文件。
PUBLISH_MARKER = ".publish_in_progress.json"

#: 标记里列出"这次发布正在写哪些产物"的键。值必须是**非空的名字列表**：
#: 它不是"缺少时的默认为空"，而是"读不懂就不能当已提交"。空列表/字符串/字典
#: 都被下面那条契约明确拒绝，见 :func:`require_publish_marker_contract`。
PUBLISH_ARTIFACTS_KEY = "artifacts"

#: 标记里记录"这次发布属于哪个事务"的键。没有它，一个后来的事务就无法判断
#: 标记是不是自己写的，只能**无条件**删除——那会把上一次中断的痕迹一起抹掉，
#: 让一套从未提交成功的产物变成"看起来已提交"（复核实测）。
#:
#: 它是**必需**字段：缺了它的老标记一律保持阻断，不当成"没有标记"，也不当成
#: "本次事务的标记"。把"读不懂归属"当放行，等于把上面那个洞重新打开一次。
PUBLISH_OWNER_KEY = "owner"


class TrainingError(RuntimeError):
    """训练无法继续。"""


# --------------------------------------------------------------------------
# 多文件发布的读取规则
# --------------------------------------------------------------------------


def publish_marker_path(output_dir: Path | str) -> Path:
    """标记总在**它描述的那批产物旁边**。"""
    return Path(output_dir) / PUBLISH_MARKER


def _illegal_artifact_name(name: Any) -> str | None:
    """``PUBLISH_ARTIFACTS_KEY`` 的一个列表项不合法时给出理由，否则 ``None``。

    "合法名字"按**它在标记目录里的作用**定义：标记旁边的一个**文件**。所以路径
    分隔符、"``.``/``..``"、空串都不算名字——它们要么指向别的地方，要么什么都没
    指向。类型必须**原样**是字符串：以前这里靠 ``str(name)`` 兜底，结果是任何
    错误类型都被强行转成合法输入（``5`` → ``"5"``、``None`` → ``"None"``），
    错误类型永远不会被判为错。
    """
    if not isinstance(name, str):
        return f"is {type(name).__name__}, not a string"
    if not name.strip():
        return "is empty"
    if name in (".", ".."):
        return "is a directory reference, not a file name"
    if "/" in name or "\\" in name:
        return "contains a path separator, so it does not name a file in this directory"
    if "\x00" in name:
        return "contains a NUL byte"
    return None


def require_publish_marker_contract(
    payload: Any,
    path: Path | str,
    *,
    error: type[Exception] = TrainingError,
) -> list[str]:
    """校验未提交标记的**结构**，合法则返回它的产物名字列表。

    这是标记结构的**唯一定义**。契约只有三条，但每一条都是"读不懂就拒绝"：

    1. 顶层是 JSON 对象；
    2. ``artifacts`` 存在，是一个**非空**列表，每项是合法、非空的文件名字符串；
    3. ``owner`` 存在，是一个非空字符串。

    为什么要这么严：这几种输入**都是合法的 JSON 对象**，所以"不是对象就拒绝"
    拦不住它们，而每一种都会让"未提交"这件事静默消失——

    * ``{}`` / ``{"artifacts": []}``：取不到或取到空的 ``artifacts``，被
      ``or ()`` 兜成空集合，读起来和"全部已提交"一模一样；
    * ``{"artifacts": "last.pt"}``：字符串被**逐字符**遍历成
      ``{'l','a','s','t','.','p'}``，``"last.pt" in artifacts`` 为假；
    * 缺 ``owner``：没人能说清这个标记属于哪个事务。

    三者修前都会让训练恢复与预测**双双放行**（复核实测）。所以这里不做任何
    "缺省即无害"的解读：标记存在就必须能按上面的契约读懂。
    """

    def fail(reason: str) -> NoReturn:
        raise error(
            f"{path} is not a usable publish marker ({reason}); a publish was "
            "interrupted, so the artifacts beside it cannot be trusted as committed. "
            "Refusing to read them rather than assuming the marker says nothing"
        )

    if not isinstance(payload, dict):
        fail(f"it is {type(payload).__name__}, not a JSON object")

    if PUBLISH_ARTIFACTS_KEY not in payload:
        fail(f"it carries no {PUBLISH_ARTIFACTS_KEY!r} field")
    artifacts = payload[PUBLISH_ARTIFACTS_KEY]
    if not isinstance(artifacts, list):
        fail(
            f"its {PUBLISH_ARTIFACTS_KEY!r} is {type(artifacts).__name__}, not a list; a "
            "string here would be walked character by character and match no artifact"
        )
    if not artifacts:
        fail(
            f"its {PUBLISH_ARTIFACTS_KEY!r} list is empty, so it names no artifact at "
            "all: that reads exactly like 'everything is committed'"
        )
    for index, name in enumerate(artifacts):
        reason = _illegal_artifact_name(name)
        if reason is not None:
            fail(f"its {PUBLISH_ARTIFACTS_KEY}[{index}] = {name!r} {reason}")

    if PUBLISH_OWNER_KEY not in payload:
        fail(
            f"it carries no {PUBLISH_OWNER_KEY!r} field, so no transaction can be shown "
            "to own it; without an owner a marker cannot be told apart from an earlier "
            "interrupted publish"
        )
    owner = payload[PUBLISH_OWNER_KEY]
    if not isinstance(owner, str) or not owner.strip():
        fail(
            f"its {PUBLISH_OWNER_KEY!r} is {owner!r}, not a non-empty string; the owner "
            "is what tells this transaction's marker apart from an earlier one"
        )

    supersedes = payload.get("supersedes")
    if supersedes is not None and not isinstance(supersedes, dict):
        fail(f"its 'supersedes' is {type(supersedes).__name__}, not a JSON object")

    return list(artifacts)


def _read_marker_json(marker: Path, *, error: type[Exception]) -> Any:
    """解析标记文件。读不出来（权限、编码、目录、损坏的 JSON）**一律拒绝**。

    返回解析结果本身，``None`` 不在这里被当成"没有标记"：JSON 的 ``null`` 也是
    一个合法文档，把它和"文件不存在"混为一谈就会放过一个读不懂的标记。
    """
    try:
        text = marker.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - 读不懂就拒绝，不做任何猜测
        raise error(
            f"{marker} exists but cannot be read ({exc.__class__.__name__}: {exc}); a "
            "publish was interrupted, so the artifacts beside it cannot be trusted as "
            "committed"
        ) from exc
    try:
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001 - 同上
        raise error(
            f"{marker} exists but cannot be read ({exc.__class__.__name__}: {exc}); a "
            "publish was interrupted, so the artifacts beside it cannot be trusted as "
            "committed"
        ) from exc


def read_publish_marker(
    output_dir: Path | str, *, error: type[Exception] = TrainingError
) -> dict[str, Any] | None:
    """读 ``output_dir`` 里的未提交标记。没有标记时返回 ``None``。

    参数是**标记所在目录**，不是某个检查点：调用方有时按目录问（"这个目录里
    有没有没发布的产物"），有时按检查点问（"这一份能不能读"）。后者
    :func:`refuse_if_uncommitted` 会自己取父目录。

    标记存在时**必须**满足 :func:`require_publish_marker_contract`。读不出来、
    或者结构不合法，都**保守拒绝**：宁可让一个可能完好的目录多拒绝一次，也不能
    把"读不懂标记"当成"没有标记"。
    """
    marker = publish_marker_path(output_dir)
    if not marker.exists():
        return None
    payload = _read_marker_json(marker, error=error)
    require_publish_marker_contract(payload, marker, error=error)
    return payload


def uncommitted_artifacts(
    output_dir: Path | str, *, error: type[Exception] = TrainingError
) -> set[str] | None:
    """``output_dir`` 里那批**未提交**产物的名字；没有标记时 ``None``。

    校验走的是**同一个** :func:`require_publish_marker_contract`，不做第二套
    判断。返回的集合直接来自校验通过的那个列表，**不再经过 ``str()``**——那一步
    正是让错误类型冒充合法输入的地方。
    """
    marker = publish_marker_path(output_dir)
    if not marker.exists():
        return None
    payload = _read_marker_json(marker, error=error)
    return set(require_publish_marker_contract(payload, marker, error=error))


def refuse_if_uncommitted(
    checkpoint_path: Path | str, *, error: type[Exception] = TrainingError
) -> None:
    """加载端必须**落实**拒绝规则，而不是只在日志里提一句。

    训练恢复与预测入口都调它。只在一处检查时，"恢复被拒绝"与"可以用来预测"
    会同时成立——而预测同样会拿一套新旧混合的产物当真值使用。

    标记在**被加载检查点的父目录**里找，不是调用方自己的输出目录：读它的往往是
    一个新实例，写出那次运行的输出目录与它不同（规则 §6）。

    标记存在却读不懂、或结构不合法时，:func:`uncommitted_artifacts` 已经抛错，
    这里不再有第二条路径——"读不懂"与"列出了这一份"都落进同一个拒绝。
    """
    artifacts = uncommitted_artifacts(Path(checkpoint_path).parent, error=error)
    name = Path(checkpoint_path).name
    if artifacts and name in artifacts:
        marker = publish_marker_path(Path(checkpoint_path).parent)
        raise error(
            f"{checkpoint_path} was left uncommitted by an interrupted publish (see "
            f"{marker}). The directory holds a mixture of new and old artifacts; "
            "refusing to read it rather than treating it as a valid checkpoint"
        )


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
# 新启动场检查点：尺度记录、身份、保存与恢复
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


# --------------------------------------------------------------------------
# 采样配置与时间窗口
# --------------------------------------------------------------------------


def _validate_sampling_record(
    record: Any, *, where: str, error: type[Exception] = TrainingError
) -> None:
    """采样记录的**结构**校验，不依赖任何接收对象。

    覆盖四类实际改变样本分布的量：``tau_end``、点数、分层比例、``r_min_fraction``；
    以及动态窗口（用采样器模块的同一份窗口定义校验，不另写一套）。类型与范围都
    按原样检查：``bool`` 不算整数，字符串不转成数字。
    """
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
        if not _is_int_like(value):
            raise error(
                f"{where}.{SAMPLING_COUNTS_KEY}.{key}: {value!r} is not an integer"
            )
        if int(value) < 0:
            raise error(
                f"{where}.{SAMPLING_COUNTS_KEY}.{key}: {value!r} is negative"
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
    near_fraction = float(stratification["near_fraction"])
    if not (0.0 <= near_fraction <= 1.0):
        raise error(
            f"{where}.{SAMPLING_STRATIFICATION_KEY}.near_fraction: {near_fraction!r} "
            "is outside [0, 1]"
        )
    for key in ("near_span", "xi_floor"):
        value = float(stratification[key])
        if not (0.0 < value < 0.5):
            raise error(
                f"{where}.{SAMPLING_STRATIFICATION_KEY}.{key}: {value!r} is outside "
                "(0, 0.5)"
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
    """把采样器的静态配置与当前窗口写成一条记录。"""
    if not hasattr(sampler, "sampling_config_state"):
        raise TrainingError(
            f"the checkpoint needs a RadiusSampler-like object, got "
            f"{type(sampler).__name__}"
        )
    record = _normalise(sampler.sampling_config_state())
    _validate_sampling_record(record, where="save_startup_checkpoint: sampling")
    return record


def _check_sampling_record(
    record: Any, sampler: Any, *, where: str
) -> None:
    """核对接收采样器与检查点的采样设置**逐项相同**，不作任何修改。"""
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
            f"{where}.counts: the sampling point counts differ in {differing} "
            f"(checkpoint { {key: saved_counts[key] for key in differing} }, "
            f"receiving sampler { {key: receiving_counts[key] for key in differing} }); "
            "a different sampling configuration is not an exact resume"
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
            f"{differing} (checkpoint "
            f"{ {key: saved_stratification[key] for key in differing} }, receiving "
            f"sampler { {key: receiving_stratification[key] for key in differing} })"
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
    """接收采样器必须站在检查点的**同一个启动场与尺度**上。"""
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
            f"{field.cache_key()!r}; resuming would continue on a different initial "
            "condition"
        )


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


def _model_structure(model: Any) -> Any:
    if not hasattr(model, "describe") or not hasattr(model, "state_dict"):
        raise TrainingError(
            f"the checkpoint needs a CoupledModel-like object, got {type(model).__name__}"
        )
    parameters = list(model.parameters())
    if not parameters:
        raise TrainingError("the model has no trainable parameters")
    return _normalise({"describe": model.describe(), "dtype": str(parameters[0].dtype)})


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
    """全局 Python/NumPy/Torch 随机状态，加上可选的采样器状态。"""
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
                "the checkpoint carries a sampler RNG state but no sampler was provided "
                "to restore it into"
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
            f"build reads version {CHECKPOINT_VERSION}. There is no migration from the "
            "old reference-field format — regenerate with the new startup chain."
        )
    if payload.get(PURPOSE_KEY) != CHECKPOINT_PURPOSE:
        raise TrainingError(
            f"{path}: checkpoint purpose {payload.get(PURPOSE_KEY)!r} is not "
            f"{CHECKPOINT_PURPOSE!r}; it belongs to a different checkpoint contract"
        )


def save_startup_checkpoint(
    path: Path | str,
    *,
    field: PlanarStartupField,
    scales: PlanarStartupScales,
    model: Any,
    optimizer: Any = None,
    step: int = 0,
    sampler: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """写出新启动场检查点（模型 + 可选优化器/随机状态）。"""
    import torch

    from ..losses.objective import contact_pressure_scale

    _require_field(field)
    record = startup_scale_record(scales)
    if not _same_number(record["startup_epsilon"], field.epsilon):
        raise TrainingError(
            f"the scales start at epsilon {record['startup_epsilon']!r} s but the field "
            f"is at {field.epsilon!r} s"
        )
    if not _same_number(record["charge_radius"], field.charge_radius):
        raise TrainingError(
            f"the scales belong to R_0 = {record['charge_radius']!r} m but the field has "
            f"{field.charge_radius!r} m"
        )
    step_value = require_step_value(step, field="step", where="save_startup_checkpoint")
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    validate_model_state(
        model_state,
        model.state_dict(),
        where="save_startup_checkpoint: model_state",
    )
    optimizer_state = None
    if optimizer is not None:
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        reference_groups, parameter_shapes = _optimizer_reference(optimizer)
        validate_optimizer_state(
            optimizer_state,
            reference_param_groups=reference_groups,
            parameter_shapes=parameter_shapes,
            where="save_startup_checkpoint: optimizer",
        )
    pressure = float(contact_pressure_scale(field))
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise TrainingError(
            f"the contact pressure scale is {pressure!r} Pa; it must be finite and positive"
        )
    rng_state = collect_rng_state(sampler=sampler)
    sampling_state = None if sampler is None else sampling_record(sampler)
    if sampler is not None:
        # 写出的采样器必须站在**这次保存的**启动场与尺度上；否则文件一写出来就
        # 自相矛盾，而读取端要到很久以后才会发现。
        _check_sampler_startup_binding(
            sampler,
            startup_identity(field),
            record,
            scales,
            field,
            where="save_startup_checkpoint: sampler",
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        PURPOSE_KEY: CHECKPOINT_PURPOSE,
        STARTUP_KEY: startup_identity(field),
        SCALES_KEY: record,
        MODEL_STRUCTURE_KEY: _model_structure(model),
        MODEL_STATE_KEY: model_state,
        OPTIMIZER_KEY: optimizer_state,
        STEP_KEY: step_value,
        RNG_KEY: rng_state,
        SAMPLING_KEY: sampling_state,
        CONTACT_PRESSURE_KEY: pressure,
        FINGERPRINT_KEY: parameter_fingerprint_of(model_state),
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


def load_startup_checkpoint(
    path: Path | str,
    *,
    handoff_json: Path | str | None = None,
    model: Any = None,
    scales: PlanarStartupScales | None = None,
    optimizer: Any = None,
    sampler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """读取并核对新启动场检查点；给出目标对象时执行恢复。

    先完成**全部**检查，再修改任何对象。恢复过程中的异常会把模型、优化器与随机
    状态回滚到调用前的值，避免留下"一半加载"的目标。
    """
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require_new_checkpoint(payload, source)

    record = payload.get(SCALES_KEY)
    rebuilt = planar_scales_from_record(record, where=f"{source}: scales")
    identity = payload.get(STARTUP_KEY)
    if not isinstance(identity, Mapping):
        raise TrainingError(f"{source}: the startup identity is not a mapping")
    for key in (STARTUP_MODEL_ID_KEY, STARTUP_METHOD_VERSION_KEY, STARTUP_CACHE_KEY):
        if not isinstance(identity.get(key), str) or not identity.get(key):
            raise TrainingError(f"{source}: the startup identity has no usable {key!r}")
    if _real_number(identity.get(STARTUP_DELTA_KEY)) is None:
        raise TrainingError(f"{source}: the startup identity has no usable delta")

    field = None
    if handoff_json is not None:
        field = _require_field(load_planar_startup_handoff(handoff_json))
        _check_startup_identity(identity, field, rebuilt, where=f"{source}: startup")
    if scales is not None:
        _check_scales_match(record, scales, where=f"{source}: scales")

    step_value = require_step_value(
        payload.get(STEP_KEY), field="step", where=f"{source}: step"
    )
    pressure = _real_number(payload.get(CONTACT_PRESSURE_KEY))
    if pressure is None or not math.isfinite(pressure) or pressure <= 0.0:
        raise TrainingError(
            f"{source}: the contact pressure scale {payload.get(CONTACT_PRESSURE_KEY)!r} "
            "is not a finite positive number"
        )

    model_state = payload.get(MODEL_STATE_KEY)
    if model is not None:
        if not hasattr(model, "state_dict"):
            raise TrainingError(f"{source}: the receiving model is not a torch module")
        model_scales = getattr(model, "scales", None)
        if not isinstance(model_scales, PlanarStartupScales):
            raise TrainingError(
                f"{source}: the receiving model's scales are "
                f"{type(model_scales).__name__}, not PlanarStartupScales"
            )
        _check_scales_match(record, model_scales, where=f"{source}: model scales")
        if _normalise(payload.get(MODEL_STRUCTURE_KEY)) != _model_structure(model):
            raise TrainingError(
                f"{source}: the checkpoint's network structure or EOS configuration does "
                "not match this model; refusing to load parameters into a different model"
            )
        validate_model_state(
            model_state,
            model.state_dict(),
            where=f"{source}: model_state",
        )
        stored_fingerprint = payload.get(FINGERPRINT_KEY)
        if stored_fingerprint != parameter_fingerprint_of(model_state):
            raise TrainingError(
                f"{source}: the recorded parameter fingerprint does not match the stored "
                "parameters; the checkpoint is damaged"
            )

    optimizer_state = payload.get(OPTIMIZER_KEY)
    if optimizer is not None:
        if model is None:
            raise TrainingError(
                f"{source}: restoring an optimizer needs the receiving model as well"
            )
        if optimizer_state is None:
            raise TrainingError(
                f"{source}: this checkpoint carries no optimizer state, but an optimizer "
                "was supplied to restore; refusing to continue from a different update history"
            )
        reference_groups, parameter_shapes = _optimizer_reference(optimizer)
        validate_optimizer_state(
            optimizer_state,
            reference_param_groups=reference_groups,
            parameter_shapes=parameter_shapes,
            where=f"{source}: optimizer",
        )

    if SAMPLING_KEY not in payload:
        raise TrainingError(
            f"{source}: the checkpoint has no {SAMPLING_KEY!r} field. It was written "
            "by an earlier contract that did not record tau_end, point counts or the "
            "time window, so an exact resume cannot be reconstructed. Regenerate it "
            "with the current startup chain"
        )
    sampling_state = payload[SAMPLING_KEY]
    if sampler is not None:
        if sampling_state is None:
            raise TrainingError(
                f"{source}: this checkpoint carries no sampling configuration, but a "
                "sampler was supplied. Without it the sampler's tau_end, point counts "
                "and stratification cannot be shown to match, so the run cannot be "
                "resumed exactly"
            )
        _check_sampling_record(sampling_state, sampler, where=f"{source}: sampling")
        _check_sampler_startup_binding(
            sampler, identity, record, rebuilt, field, where=f"{source}: sampler"
        )
    elif sampling_state is not None:
        _validate_sampling_record(sampling_state, where=f"{source}: sampling")

    rng_state = payload.get(RNG_KEY)
    if restore_rng:
        _validate_rng_state(rng_state, where=f"{source}: rng")
        if sampler is not None:
            if rng_state.get("sampler") is None:
                raise TrainingError(
                    f"{source}: the checkpoint has no sampler RNG state, but a sampler "
                    "was supplied; a resume without it would sample a different trajectory"
                )
            sampler.validate_rng_state(rng_state["sampler"])

    model_backup = (
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if model is not None
        else None
    )
    optimizer_backup = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    window_backup = (
        sampler.window_state()
        if sampler is not None and sampling_state is not None
        else None
    )
    rng_backup = collect_rng_state(sampler=sampler) if restore_rng else None
    try:
        if model is not None:
            model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        if sampler is not None and sampling_state is not None:
            sampler.set_window_state(sampling_state[SAMPLING_WINDOW_KEY])
        if restore_rng:
            _apply_rng_state(rng_state, sampler)
    except Exception:
        if model is not None and model_backup is not None:
            model.load_state_dict(model_backup, strict=True)
        if optimizer is not None and optimizer_backup is not None:
            optimizer.load_state_dict(optimizer_backup)
        if sampler is not None and window_backup is not None:
            sampler.set_window_state(window_backup)
        if restore_rng and rng_backup is not None:
            _apply_rng_state(rng_backup, sampler)
        raise

    return {
        "path": str(source),
        "startup": dict(identity),
        "scales": rebuilt,
        "step": step_value,
        "contact_pressure_pa": float(pressure),
        "parameter_fingerprint": payload.get(FINGERPRINT_KEY),
        "handoff_checked": field is not None,
        "restored_model": model is not None,
        "restored_optimizer": optimizer is not None,
        "restored_sampling": bool(sampler is not None and sampling_state is not None),
        "sampling": copy.deepcopy(sampling_state) if sampling_state is not None else None,
        "restored_rng": bool(restore_rng),
    }


def same_selection_value(left: Any, right: Any) -> bool:
    """结构相等，**且把 NaN 当作等于自己**。

    ``best_check.score`` 里带 NaN 是**合法**的（某个分量没有可测值，例如没有
    对照数据时的开发集误差），而 ``nan != nan``——直接比会把一份完全正常的候选
    判成"选择依据自相矛盾"。实测就是这么被误伤的：合法对照被自己新增的检查
    拒掉，报的却是"score 不一致"，两边打印出来一模一样。
    """
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            same_selection_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_selection_value(a, b) for a, b in zip(left, right)
        )
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return bool(left == right)
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


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

    调用方（``validate_state_snapshot`` 与 :func:`_require_one_moment`）各自只决定
    "哪些字段必须有、要和谁相等"；"什么才算一个步号"只有这一处说了算。以前三处
    各写各的，于是候选内部与顶层最佳记录的 ``step`` 根本没人验——删掉、写成
    ``"broken"``，都能一路加载并发布（复核实测）。

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


#: 一份状态快照里必须是**列表**的字段。三项都是恢复端逐条读的：
#: ``history``/``checks`` 进轨迹，``refinement`` 进精修记录。
STATE_SNAPSHOT_LIST_FIELDS = ("history", "checks", "refinement")

#: ``stage`` 块必须带的键（第 4 版起）。缺 ``blocked`` 会被读成"没有待决判据"，
#: 而它存在的意义正是把那一档和"没有待决判据"分开。
STATE_SNAPSHOT_STAGE_KEYS = ("current", "window_fraction", "pending", "blocked")


def validate_state_snapshot(
    state: Any,
    *,
    where: str,
    expected_model_state: Mapping[str, Any],
    sampler_rng_validator: Callable[[Any], None],
    optimizer_validator: Callable[[Any, str], None],
    rng_state_validator: Callable[[Any], None],
    stage_windows: Mapping[str, float],
    window_at_step: Callable[[int], float],
    error: type[Exception] = TrainingError,
) -> None:
    """一份"训练状态快照"的**只读**校验。顶层载荷与候选快照共用这一份。

    以前只有顶层载荷有一套逐条的检查、候选只查了"是不是 Mapping、步号对不对"，
    于是候选的模型、优化器、阶段、采样器 RNG、平衡器、历史全都没有人验——清空
    ``model``、改参数留旧指纹、删 ``optimizer``、``optimizer = {}``、
    ``rng['torch'] = 0``、改 ``stage.window_fraction`` 都能**加载成功**（复核实测）。

    字段清单的依据是**两端的实际行为**，不是想象：写入端
    :meth:`trainer.Trainer._checkpoint_state` 产生它们，恢复端
    :meth:`trainer.Trainer._commit_checkpoint` 消费它们。写而不读的字段
    （``learning_rates``）**不在此列**——把"写过"当成"必需"会让规则比实际契约
    更严，而漏掉一个真被消费的字段会让它在损坏时静默缺省。

    只查"拿到它之后能不能正确恢复"：

    * **不**要求数值与当前参数一致（候选可能来自更早的一步）；
    * **不**要求字段非空或所有数都有限——尚未更新时的空优化器状态、按契约允许的
      未测损失标记（``nan``）都是**合法初始状态**，把它们判成损坏等于禁止恢复
      一个刚开始的训练。

    调用方必须提供 :func:`validate_state_snapshot` 自己看不到的那几样东西——
    每个子系统"能不能被接受"要问那个子系统本身，另写一套判断会在细节上分道
    扬镳。**都要显式给**：给默认值就等于"调用方忘了传时这几项静默不查"，而这一
    项要修的正是那种静默。

    :param optimizer_validator: ``(optimizer_state, where)``；做结构检查与一次
        **隔离的**探针更新。
    :param rng_state_validator: ``(global_rng_state,)``；用独立的生成器验证这份
        状态真能被接受，**不改动全局 RNG 再恢复**。
    :param stage_windows: 阶段名 → 该阶段的时间窗口。它的键同时是"这套日程里
        真有的阶段名"——``stage.current``/``stage.pending`` 必须落在其中。
        只传名字集合不足以判断下面那些**关系**。
    :param window_at_step: 步号 → 该步所属阶段的窗口。
    """
    if not isinstance(state, Mapping):
        raise error(f"{where}: the snapshot is {type(state).__name__}, not a mapping")

    step = require_step_value(state.get("step"), field="step", where=where, error=error)

    # **快照自己带的最佳记录也要验。** 已发布的 ``best.pt`` 没有待发布候选
    # （``best_dirty=False``），``require_best_candidate_consistency`` 会在第一道
    # 判据就 return——于是这份文件**自身**的 ``best_check.step`` 以前完全没人看：
    # 删掉或写成 ``"broken"`` 照样加载成功（复核实测）。它必须在这里查，因为这是
    # 顶层载荷与候选**共用**的那一份只读校验。
    #
    # 合法初始状态保留：还没产生过最佳记录时 ``best_check`` 是 ``None``；写入端
    # （``Trainer._checkpoint_state``）在那种情况下也是写 ``None``，不是缺这个键。
    best_record = state.get("best_check")
    if best_record is not None:
        if not isinstance(best_record, Mapping):
            raise error(
                f"{where}: 'best_check' is {type(best_record).__name__}, not a mapping "
                "or None; a best record that cannot be read cannot be checked against "
                "the step it claims"
            )
        require_step_value(
            best_record.get("step"),
            field="best_check.step",
            where=where,
            error=error,
        )

    validate_model_state(
        state.get("model"),
        expected_model_state,
        where=f"{where}: model",
        error=error,
    )

    # 优化器：**结构**要够恢复用。``state`` 为空是合法的——一次更新都没做过的
    # 训练就是那样，把它判成损坏会禁止恢复第一份检查点；但"空 state"与"整个
    # 优化器是空字典"是两回事，后者恢复时才会炸（复核实测）。
    optimizer_validator(state.get("optimizer"), f"{where}: optimizer")

    if not isinstance(state.get("balancing"), Mapping):
        raise error(
            f"{where}: the adaptive-weight state is "
            f"{type(state.get('balancing')).__name__}, not a mapping; restoring without "
            "it would silently fall back to the base weights"
        )

    rng = state.get("rng")
    if not isinstance(rng, Mapping):
        raise error(
            f"{where}: carries no global RNG state ({type(rng).__name__}, not a "
            "mapping); resuming from it would continue the global random sequence from "
            "somewhere else, so 'resume to step N' would not reproduce 'run N steps'"
        )
    missing_rng = [key for key in ("numpy", "torch") if rng.get(key) is None]
    if missing_rng:
        raise error(
            f"{where}: the global RNG state has no {missing_rng}; a partial record "
            "cannot restore the sequence it claims to"
        )
    # **"非空"不等于"能用"。** 整数 ``0`` 不是 ``None``，于是 ``rng['torch'] = 0``
    # 通过了原来那条检查，然后在恢复时抛 ``TypeError: expected a torch.ByteTensor``
    # ——同样是"加载成功、下一个实例才炸"（复核实测）。
    try:
        rng_state_validator(rng)
    except Exception as exc:  # noqa: BLE001 - 保留实际原因
        raise error(f"{where}: the global RNG state is not usable: {exc}") from exc

    if "sampler_rng" not in state:
        raise error(
            f"{where}: no sampler RNG state; 'resume to step N' would not reproduce "
            "'run N steps'"
        )
    try:
        sampler_rng_validator(state["sampler_rng"])
    except Exception as exc:  # noqa: BLE001 - 保留采样器给出的**实际**原因
        raise error(f"{where}: the sampler RNG state is not usable: {exc}") from exc

    stage = state.get("stage")
    if not isinstance(stage, Mapping):
        raise error(f"{where}: no 'stage' block; the stage, its window and any pending verdict cannot be reconstructed")
    for key in STATE_SNAPSHOT_STAGE_KEYS:
        if key not in stage:
            raise error(
                f"{where}: the 'stage' block has no {key!r}. A missing record is not a "
                "pass: an older writer that omitted 'blocked' would be read as 'no "
                "pending verdict', which is exactly the state this field exists to "
                "distinguish"
            )
    window = stage["window_fraction"]
    if not isinstance(window, (int, float, np.floating)) or not np.isfinite(float(window)):
        raise error(f"{where}: the stage window is {window!r}, not a finite number")
    # 阶段**名字**必须与这份日程相容：``current``/``pending`` 要么是 ``None``，要么
    # 是这套日程里真有的阶段名。只查"键在不在"会放过 ``current = 123`` 这种，
    # 而它到了下游会让"这一份是哪个阶段的结论"变成一句无从判断的话。
    for key in ("current", "pending"):
        value = stage[key]
        if value is not None and not isinstance(value, str):
            raise error(f"{where}: the stage {key!r} is {value!r}, neither None nor a stage name")
        if value is not None and value not in stage_windows:
            raise error(
                f"{where}: the stage {key!r} is {value!r}, which this schedule does not "
                "define; refusing rather than guessing a stage"
            )
    if not isinstance(stage["blocked"], (bool, np.bool_)):
        raise error(
            f"{where}: the stage 'blocked' flag is {stage['blocked']!r}, not a boolean; "
            "a string 'False' would be read as blocked, the opposite of what it says"
        )

    # ---- 阶段、窗口与步号必须**互相**成立 --------------------------------
    #
    # 允许的窗口**不是**"按步数算出的、或者任何一个记录过的阶段窗口"——那太松了：
    # 任何阶段窗口都会被接受一次，等于把"窗口对不上"这件事放过去。真正的条件按
    # **阻塞与否**分成两条，两条都是精确等值：
    #
    # * 未阻塞：步数已经推进到下一阶段，窗口必须就是该步数所属阶段的窗口；
    # * 已阻塞：步数停在边界上、阶段没切，窗口必须是**那个未切走的阶段**的窗口，
    #   而且被阻塞的阶段必须正是它。
    #
    # **一律用这份快照自己的步号**，不能用当前顶层实例的：候选来自更早的一步是
    # 合法情形，照着接收实例的步号去算，会把"稍早一步的候选"整批判成损坏。
    blocked = bool(stage["blocked"])
    pending = stage["pending"]
    recorded_stage = stage["current"]
    if blocked:
        if pending not in stage_windows:
            raise error(
                f"{where}: the snapshot is marked blocked on stage {pending!r}, which "
                "this schedule does not define"
            )
        if recorded_stage != pending:
            raise error(
                f"{where}: the snapshot is blocked on {pending!r} but records its "
                f"current stage as {recorded_stage!r}; a blocked snapshot must still be "
                "sitting in the stage whose criteria failed"
            )
        expected_window = float(stage_windows[pending])
    else:
        if pending is not None:
            raise error(
                f"{where}: the snapshot records a pending stage {pending!r} but is not "
                "marked blocked; the two fields must agree"
            )
        expected_window = float(window_at_step(int(step)))
    if abs(float(window) - expected_window) > 1.0e-12:
        raise error(
            f"{where}: stage.window_fraction is {window}, but "
            f"{'the blocked stage' if blocked else 'step'} "
            f"{recorded_stage if blocked else int(step)} of this schedule requires "
            f"{expected_window}; this is not the same schedule"
        )

    origin = state.get("run_origin")
    if not isinstance(origin, Mapping):
        raise error(
            f"{where}: no 'run_origin'; without it a checkpoint written by an explicit "
            "diagnostic run or by a refinement would come back looking like a plain "
            "completed run"
        )
    for key in ("explicit_diagnostic", "refined"):
        if not isinstance(origin.get(key), (bool, np.bool_)):
            raise error(f"{where}: run_origin[{key!r}] is {origin.get(key)!r}, not a boolean")

    # 未测损失是**合法**的初始状态：没更新过就是 ``nan``。所以只查"是不是数"，
    # 不查有限性。
    for key in ("last_update_loss", "last_update_loss_step"):
        if key not in state:
            raise error(
                f"{where}: carries no {key!r}. Without it a resumed run that performs "
                "no update would report a fabricated NaN loss at step 0"
            )
    loss = state["last_update_loss"]
    if not isinstance(loss, (int, float, np.floating)) or isinstance(loss, bool):
        raise error(f"{where}: the last update loss is {loss!r}, not a number")
    loss_step = state["last_update_loss_step"]
    if not isinstance(loss_step, (int, np.integer)) or isinstance(loss_step, bool):
        raise error(f"{where}: the last update loss step is {loss_step!r}, not an integer")
    if int(loss_step) > int(step):
        raise error(
            f"{where}: the last measured loss is recorded at step {loss_step}, beyond "
            f"the snapshot's own step {step}"
        )

    for key in STATE_SNAPSHOT_LIST_FIELDS:
        if not isinstance(state.get(key), list):
            raise error(
                f"{where}: {key!r} is {type(state.get(key)).__name__}, not a list"
            )


def _require_one_moment(
    candidate: Mapping[str, Any],
    inner: Mapping[str, Any],
    best_check: Mapping[str, Any],
    path: Path | str,
    error: type[Exception],
) -> None:
    """三处步号**各自**必须是合法整数，并且表示同一步。

    以前这里在类型不对时直接 ``return`` / ``continue``，注释说"后续状态校验会负责
    类型检查"——但后续只查了快照自身的 ``step``，**最佳记录里的那两处没人查**。
    于是删掉 ``pending_best_candidate.best_check.step``、或把它写成 ``"broken"``，
    都能通过共享校验、加载、发布（复核实测）。

    现在：三处各走一遍 :func:`require_step_value`（缺失/None/字符串/浮点/布尔都
    明确拒绝，**不**先 ``int()``），再比数值。**不修补**：缺失的不拿顶层步号补，
    不一致的不拿当前训练步号覆盖。
    """
    steps = [
        require_step_value(
            candidate.get("step"),
            field="pending_best_candidate.step",
            where=str(path),
            error=error,
        ),
        require_step_value(
            inner.get("step"),
            field="pending_best_candidate.best_check.step",
            where=str(path),
            error=error,
        ),
        require_step_value(
            best_check.get("step"),
            field="best_check.step",
            where=str(path),
            error=error,
        ),
    ]
    if steps[0] != steps[1] or steps[0] != steps[2]:
        raise error(
            f"{path}: the pending best candidate, the best record it carries inside "
            "itself and the checkpoint's best record must all describe the same "
            f"moment, but their steps are {steps[0]}, {steps[1]} and {steps[2]} "
            "(pending_best_candidate.step / pending_best_candidate.best_check.step / "
            "best_check.step); they describe different moments"
        )


def require_best_candidate_consistency(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    expected_model_state: Mapping[str, Any],
    sampler_rng_validator: Callable[[Any], None],
    optimizer_validator: Callable[[Any, str], None],
    rng_state_validator: Callable[[Any], None],
    stage_windows: Mapping[str, float],
    window_at_step: Callable[[int], float],
    error: type[Exception] = TrainingError,
) -> None:
    """``best_dirty=True`` 必须对应一份**完整可用**且**自证一致**的候选快照。

    四件事，缺一件都会让"最佳模型"这件事变成一句空话：

    1. **候选本身是一份完整的状态快照**（走 :func:`validate_state_snapshot`，
       与顶层载荷同一份规则）——模型结构、优化器、阶段、采样器与全局 RNG、
       平衡器、历史、来源。
    2. **指纹从候选里的实际参数重算**，并等于记录里的值。以前只检查"记录里那个
       指纹是非空字符串"，于是"改一个参数、指纹不动"这种改法无人能发现——而
       ``best.pt`` 的 ``candidate.parameter_fingerprint`` 正是靠它声称"这一份
       就是被选中的那组参数"。**不把记录改成重算值来"修复"**：那会让一个坏文件
       看起来是好的，而不是被拒绝。
    3. **步号三处一致**：候选自己的 ``step``、候选内部 ``best_check.step``、
       顶层 ``best_check.step``。三者描述的是同一个时刻。
    4. **选择依据不矛盾**：候选内部的 ``best_check`` 与顶层最佳记录的指纹、
       评分与检查集身份必须相同——「谁选出了它」与「它是什么」必须对得上。

    **执行次序是 3 → 1 → 2 → 4**，与上面的编号不同：一个被抬高的步号会同时违反
    第 3 条与第 1 条里的日程规则，而"这份候选不是那条记录选中的"才是直接原因，
    "它与日程对不上"是同一个改动的下游症状。最直接的那条要先说出口。

    上下文（模型结构、采样器/优化器/全局随机的验证器、日程窗口）全部**必填**：
    给它们默认值就等于"调用方忘了传时这几项就静默不查"，而这一项要修的正是那种
    静默。它们同时原样转给 :func:`validate_state_snapshot`，好让候选与顶层载荷
    **共用同一条**规则——两份规则各写各的，正是漏掉候选那一路的原因。
    """
    candidate = payload.get("pending_best_candidate")
    dirty = bool(payload.get("best_dirty"))
    best_check = payload.get("best_check")

    if not dirty:
        return
    if not isinstance(candidate, Mapping):
        raise error(
            f"{path} has best_dirty=True but pending_best_candidate="
            f"{type(candidate).__name__ if candidate is not None else None}; the "
            "checkpoint claims an unpublished best candidate that it does not carry. "
            "Refusing to load rather than clearing the flag, which would silently lose "
            "the best model"
        )
    if not isinstance(best_check, Mapping):
        raise error(
            f"{path} has best_dirty=True but no 'best_check' record; the candidate "
            "cannot be matched against the score that selected it"
        )

    inner = candidate.get("best_check")
    if not isinstance(inner, Mapping):
        raise error(
            f"{path}: pending_best_candidate.best_check is "
            f"{type(inner).__name__}, not a mapping; the candidate cannot be matched "
            "against the score that selected it"
        )

    # 3) 步号三处一致。**先比这个，再走下面的完整状态校验。**
    #
    # 一个被抬高的步号会同时违反两条：与最佳记录对不上、与日程窗口对不上。哪一条
    # 先说出口决定了诊断指不指向真正的改动——"这份候选不是那条记录选中的"是直接
    # 原因，"它与日程对不上"是同一个改动的下游症状。既有测试钉的就是前者。
    #
    # 步号**不是整数**时这里保持沉默：那是类型问题，交给下面的状态校验按类型报。
    _require_one_moment(candidate, inner, best_check, path, error)

    # 1) 候选是一份完整的状态快照。**用的是候选自己的步号与阶段**——候选来自更早
    #    一步是合法情形，拿当前实例的步号去算窗口等于把正常候选判成损坏。
    validate_state_snapshot(
        candidate,
        where=f"{path}: pending_best_candidate",
        expected_model_state=expected_model_state,
        sampler_rng_validator=sampler_rng_validator,
        optimizer_validator=optimizer_validator,
        rng_state_validator=rng_state_validator,
        stage_windows=stage_windows,
        window_at_step=window_at_step,
        error=error,
    )

    # 2) 指纹从候选的实际参数重算。
    recorded = inner.get("parameter_fingerprint")
    if not isinstance(recorded, str) or not recorded:
        raise error(
            f"{path}: the best record carries no parameter fingerprint, so the "
            "candidate cannot be shown to be the parameters that were selected"
        )
    actual = parameter_fingerprint_of(candidate["model"])
    if actual != recorded:
        raise error(
            f"{path}: the pending best candidate's recorded parameter fingerprint is "
            f"{recorded}, but the parameters it actually carries hash to {actual}. The "
            "candidate is not the parameter set that was selected. This is refused "
            "rather than repaired by rewriting the record: a rewritten fingerprint "
            "would make a corrupt checkpoint look consistent"
        )

    # 4) 选择依据不矛盾。
    if best_check.get("parameter_fingerprint") != recorded:
        raise error(
            f"{path}: the checkpoint's best record fingerprint is "
            f"{best_check.get('parameter_fingerprint')!r}, but the candidate inside it "
            f"carries {recorded!r}; the two records describe different selections"
        )
    for key in ("score", "check_set_identity"):
        if key not in inner or key not in best_check:
            raise error(
                f"{path}: the best record and the candidate's copy of it do not both "
                f"carry {key!r}; the selection basis cannot be compared"
            )
        if not same_selection_value(inner[key], best_check[key]):
            raise error(
                f"{path}: the best record and the candidate's copy of it disagree on "
                f"{key!r} ({inner[key]!r} vs {best_check[key]!r}); the candidate cannot "
                "be shown to be the one this record selected"
            )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_PURPOSE",
    "CHECKPOINT_VERSION",
    "CONTACT_PRESSURE_KEY",
    "FINGERPRINT_KEY",
    "MODEL_STATE_KEY",
    "MODEL_STRUCTURE_KEY",
    "OPTIMIZER_KEY",
    "PUBLISH_ARTIFACTS_KEY",
    "PUBLISH_MARKER",
    "PUBLISH_OWNER_KEY",
    "RNG_KEY",
    "SAMPLING_KEY",
    "SAMPLING_RECORD_KEYS",
    "SCALES_KEY",
    "SCALES_RECORD_KEYS",
    "SCALES_RELATIVE_TOLERANCE",
    "STARTUP_KEY",
    "STEP_KEY",
    "TrainingError",
    "collect_rng_state",
    "load_startup_checkpoint",
    "parameter_fingerprint_of",
    "planar_scales_from_record",
    "publish_marker_path",
    "read_publish_marker",
    "refuse_if_uncommitted",
    "require_best_candidate_consistency",
    "require_publish_marker_contract",
    "sampling_record",
    "save_startup_checkpoint",
    "startup_identity",
    "startup_scale_record",
    "uncommitted_artifacts",
    "validate_model_state",
    "validate_optimizer_state",
    "validate_state_snapshot",
]
