"""检查点格式、冻结参数与"多文件发布"读取规则的**唯一定义**。

保存端（``training.trainer``）与读取端（``cli.predict``）都从这里取格式标识、
版本号、字段名与读取规则。以前两边各维护一份清单，"保存端写了什么"与"读取端
要求什么"就此脱节：``cli.predict`` 要求一个 :meth:`Trainer._build_payload`
从来没写过的顶层字段 ``detonation_velocity``，于是一份**方案 B** 训练出来的
检查点被预测入口当成方案 A 的旧文件拒绝（复核实测）。定义放在一处，这种
不一致就没有藏身之处。

三条读取规则：

* :func:`require_detonation_velocity` —— 爆速是**冻结的物理参数**，从检查点里
  读出来、校验之后使用。缺失即拒绝，**不用默认常量补齐**：``D`` 决定方案 B 的
  时间原点 ``T_h = R_0/D + epsilon``，补一个默认值不会报错，只会把整条轨迹平移
  一个爆轰穿越时间 ``T_b``（50 mm 上是 7.2 µs）。
  **三层**都要过：顶层是合法有限正**实数**（不许 ``float()`` 吞掉 ``True`` 或
  ``"6930"``）；顶层与 ``case_ids`` 里**每一个**算例的 ``scales`` 记录一致
  （记录里没有 ``D``，按 ``T_h`` 反解）；恢复端还要与**接收实例实际使用的**
  尺度一致。以前只查第一层、且只有预测端调它，于是"同一份检查点被两个入口
  解释成两条时间轴"（复核实测，见测试文件）。
* :func:`refuse_if_uncommitted` —— 多文件发布不是原子的。发布前写下标记、成功后
  删除，**两个入口都必须落实**这条规则。以前只有 ``Trainer.load`` 查它，于是
  "恢复训练被拒绝的文件，仍能被预测入口读进去"（复核实测）。
  标记的**结构**由 :func:`require_publish_marker_contract` 一处定义并强制：
  存在就必须是一个带非空 ``artifacts`` 名字列表与 ``owner`` 的 JSON 对象。
  以前只问"顶层是不是字典"，于是 ``{}``、``{"artifacts": []}`` 被当成"没有未提交
  产物"，``{"artifacts": "last.pt"}`` 被逐字符遍历——**三个坏标记两个入口都放行**
  （复核实测）。
* :func:`require_best_candidate_consistency` —— ``best_dirty`` 与候选快照必须
  互相印证。一个 ``best_dirty=True`` 却没有任何候选的检查点不能"加载成功并把
  标志悄悄清掉"：那会让一次中断之后的最佳模型无声消失。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

# ``epsilon(R_0)`` 是反解逐算例冻结爆速时要减掉的那一项。定义只有一份，在
# ``physics/similitude.py``（``physics`` 不反向依赖 ``training``，无循环导入）。
from ..physics.similitude import start_delay

#: 检查点格式标识与版本。加载时两者都必须匹配，否则拒绝。
CHECKPOINT_FORMAT = "spherical_blast_pinn.checkpoint"

#: 第 3 版起检查点必须带**采样器自己的** RNG 状态（第 2 版只存全局 NumPy 状态，
#: 恢复出来的不是一个连续的采样轨迹），并且带固定检查集与兼容性闸门的结论。
#:
#: 第 4 版起 ``stage`` 块必须同时带 ``pending`` 与 ``blocked``，且加载端**要求
#: 它们存在**（缺失即拒绝）。这两项记录的是"更新跑完了、判据没过、尚不允许推进"
#: 这个中间状态；旧写法没有它们，读出来会被当成"没有待决判据"——那正是这两个
#: 字段要区分的那件事。所以旧检查点按版本号直接拒绝，而不是"缺字段就默认通过"。
#:
#: 第 8 版起顶层带 ``detonation_velocity``（冻结的卡片爆速），供预测入口在**新
#: 半径**上重建时间原点。旧版本没有它，方案 B 的时间轴无法重建——同样按版本号
#: 拒绝，不做迁移：写入端当时保存的是另一套字段，读出来的东西我们无法核对。
CHECKPOINT_VERSION = 8

#: 冻结爆速的字段名。保存端与读取端共用这一个名字。
DETONATION_VELOCITY_FIELD = "detonation_velocity"

#: 逐算例 ``scales`` 记录里用来**反解**冻结爆速的两个键。名字来自
#: :meth:`..physics.similitude.SimilitudeScales.as_dict`——它**不写**
#: ``detonation_velocity``，写的是 ``start_time = T_h = R_0/D + epsilon(R_0)``，
#: 而 ``(charge_radius, start_time)`` 这对值把 ``D`` 唯一确定下来：
#: ``D = R_0 / (T_h - epsilon(R_0))``。所以"逐算例记录里的冻结值"就是它。
SCALES_CHARGE_RADIUS_KEY = "charge_radius"
SCALES_START_TIME_KEY = "start_time"

#: 比较两个爆速时的**相对**容差。
#:
#: 反解要经过 ``T_h - epsilon`` 这一步（两个约 1e-5 秒的数相减），float64 上的
#: 往返误差量级在 1e-16，远小于这里。而任何**真实**的不一致——换了爆速、换了
#: 材料卡——相对差都在 1e-3 以上。这条容差只放过"同一个数的舍入"，不放过
#: "另一个数"。
DETONATION_VELOCITY_RELATIVE_TOLERANCE = 1.0e-9

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


def _scales_records(
    payload: Mapping[str, Any], path: Path | str, error: type[Exception]
) -> list[tuple[str, Mapping[str, Any]]]:
    """取出 ``case_ids`` 声明的每个算例及其 ``scales`` 记录，缺一不可。

    返回的顺序与 ``case_ids`` 一致——**不排序、不取第一个**：调用方要检查的是
    每一个算例，任何"只比第一个"的写法都会漏掉"只有某个算例对不上"这一档。
    """
    case_ids = payload.get("case_ids")
    if isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, Sequence) or not case_ids:
        raise error(
            f"{path} carries no usable 'case_ids' ({case_ids!r}); without the list of "
            "cases this checkpoint is for, the per-case frozen scales cannot be matched "
            "against the top-level detonation velocity"
        )
    for case_id in case_ids:
        if not isinstance(case_id, str) or not case_id:
            raise error(
                f"the checkpoint's 'case_ids' contains {case_id!r}, which is not a "
                "case name"
            )
    scales = payload.get("scales")
    if not isinstance(scales, Mapping):
        raise error(
            f"{path} carries no 'scales' block ({type(scales).__name__}); the per-case "
            "frozen scales are what the training run actually used"
        )
    missing = [case_id for case_id in case_ids if case_id not in scales]
    if missing:
        raise error(
            f"{path} records no scales for {missing}, which its own 'case_ids' lists. A "
            "missing record is not a pass: those cases are part of the run"
        )
    unexpected = [key for key in scales if key not in set(case_ids)]
    if unexpected:
        raise error(
            f"{path} carries scales for {unexpected}, which its own 'case_ids' does not "
            "list; the two describe different sets of cases"
        )
    return [(case_id, scales[case_id]) for case_id in case_ids]


def require_detonation_velocity(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    expected: float | None = None,
    error: type[Exception] = TrainingError,
) -> float:
    """读出检查点里**冻结的**爆速并校验，返回它。

    保存端写什么、读取端要求什么只有这一份定义。``Trainer.load`` 与
    ``cli.predict.load_model_from_checkpoint`` 都调它——以前**只有预测端**调，
    于是删掉这个字段或把它改成 ``-1`` 时"恢复训练放行、预测拒绝"（复核实测）。

    校验分三层，任何一层不成立都**硬失败**：

    1. **顶层字段**必须存在，且是合法的**有限正实数**（``True``、``"6930"``、
       ``[6930]`` 都不算——不许用 ``float()`` 把错误类型转成合法输入）；
    2. **文件内部**：顶层值必须与 ``case_ids`` 里**每一个**算例的
       ``scales[case_id]`` 记录一致。记录里没有 ``detonation_velocity`` 键，
       ``SimilitudeScales.as_dict()`` 写的是 ``start_time = T_h``，所以按
       ``D = R_0 / (T_h - epsilon(R_0))`` 反解再比。缺记录、缺键、值不可用一律
       拒绝：**不得忽略缺少记录的算例，也不得无条件取第一个算例**；
    3. ``expected``（恢复端传入接收实例**实际使用**的爆速）不为 ``None`` 时，
       顶层值还必须与它一致。只查"文件内部自洽"是不够的：一份自洽的检查点照样
       能被一个不同爆速的实例恢复，恢复出来是另一条时间轴——实测同一份文件在
       恢复端给出 ``T_h = 8.21500721501 µs``、在预测端给出 ``41.5186385737 µs``。

    ``D`` 进 ``T_h = R_0/D + epsilon(R_0)``，所以任何一层都不允许回落到默认常量：
    补一个默认值换来的不是错误，而是整条轨迹平移一个爆轰穿越时间。
    """
    if DETONATION_VELOCITY_FIELD not in payload:
        raise error(
            f"{path} carries no {DETONATION_VELOCITY_FIELD!r}, but checkpoint version "
            f"{CHECKPOINT_VERSION} requires it. That field is the charge's detonation "
            "velocity, a frozen physical parameter: the scheme-B time origin "
            "T_h = R_0/D + epsilon(R_0) cannot be reconstructed without it, and "
            "substituting a default would shift the whole trajectory by the detonation "
            "transit time. A version-8 checkpoint without it is damaged, not an older "
            "scheme's file"
        )
    raw = payload[DETONATION_VELOCITY_FIELD]
    velocity = _real_number(raw)
    if velocity is None:
        raise error(
            f"the checkpoint's {DETONATION_VELOCITY_FIELD} is {raw!r} "
            f"({type(raw).__name__}), not a real number"
        )
    if not math.isfinite(velocity) or velocity <= 0.0:
        raise error(
            f"the checkpoint's {DETONATION_VELOCITY_FIELD} must be finite and positive, "
            f"got {velocity!r}; it sets the time origin T_h = R_0/D + epsilon(R_0)"
        )

    # -- 第二层：与逐算例 scales 记录里的冻结值一致 ------------------------
    for case_id, record in _scales_records(payload, path, error):
        if not isinstance(record, Mapping):
            raise error(
                f"the checkpoint's scales for case {case_id!r} are "
                f"{type(record).__name__}, not a record"
            )
        radius = _real_number(record.get(SCALES_CHARGE_RADIUS_KEY))
        if radius is None or not math.isfinite(radius) or radius <= 0.0:
            raise error(
                f"the checkpoint's scales for case {case_id!r} record "
                f"{SCALES_CHARGE_RADIUS_KEY} = "
                f"{record.get(SCALES_CHARGE_RADIUS_KEY)!r}, which is not a finite "
                "positive radius"
            )
        start_time = _real_number(record.get(SCALES_START_TIME_KEY))
        if start_time is None or not math.isfinite(start_time):
            raise error(
                f"the checkpoint's scales for case {case_id!r} record "
                f"{SCALES_START_TIME_KEY} = {record.get(SCALES_START_TIME_KEY)!r}, "
                "which is not a finite number; the frozen detonation velocity of that "
                "case cannot be recovered from it"
            )
        transit = start_time - start_delay(radius)
        if transit <= 0.0:
            raise error(
                f"the checkpoint's scales for case {case_id!r} put the time origin "
                f"{SCALES_START_TIME_KEY} = {start_time!r} s at or before the handover "
                f"delay epsilon({radius!r}) = {start_delay(radius)!r} s, so no positive "
                "detonation velocity produces it"
            )
        implied = radius / transit
        if abs(implied - velocity) > DETONATION_VELOCITY_RELATIVE_TOLERANCE * max(
            abs(implied), abs(velocity)
        ):
            raise error(
                f"the checkpoint's top-level {DETONATION_VELOCITY_FIELD} is {velocity!r} "
                f"m/s, but its scales for case {case_id!r} imply {implied!r} m/s: "
                f"{SCALES_START_TIME_KEY} = {start_time!r} s at "
                f"{SCALES_CHARGE_RADIUS_KEY} = {radius!r} m is T_h = R_0/D + epsilon(R_0) "
                "only for that other value. The two describe different time axes, so "
                "neither can be trusted"
            )

    # -- 第三层：与**接收实例实际使用的**尺度一致（恢复端传入 expected） ----
    if expected is not None:
        reference = _real_number(expected)
        if reference is None or not math.isfinite(reference) or reference <= 0.0:
            raise error(
                f"cannot compare the checkpoint's {DETONATION_VELOCITY_FIELD} against "
                f"{expected!r}, which is not a finite positive detonation velocity"
            )
        if abs(velocity - reference) > DETONATION_VELOCITY_RELATIVE_TOLERANCE * max(
            abs(velocity), abs(reference)
        ):
            raise error(
                f"the checkpoint was trained with {DETONATION_VELOCITY_FIELD} = "
                f"{velocity!r} m/s, but this trainer's scales use {reference!r} m/s. "
                "Resuming would rebuild the time origin T_h = R_0/D + epsilon(R_0) from "
                "a different D, silently shifting the trajectory by the difference in "
                "detonation transit time. Refusing rather than adopting either value"
            )
    return velocity


# --------------------------------------------------------------------------
# 候选与"待发布"标志的一致性
# --------------------------------------------------------------------------


def require_best_candidate_consistency(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    error: type[Exception] = TrainingError,
) -> None:
    """``best_dirty=True`` 必须对应一份**完整可用**的候选快照。

    只把标志读进来、发现没有候选就**悄悄清成 False**，是这一项要禁止的做法：
    目录里那份最佳模型会无声消失（复核实测：加载成功、``best_dirty`` 被改成
    ``False``、再次运行不产出 ``best.pt``），而调用方拿到的是一份看不出异常的
    实例。这是**输入自相矛盾**，不是可以自动修正的缺省值，所以拒绝加载。
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
    step = candidate.get("step")
    if not isinstance(step, (int,)) or isinstance(step, bool):
        raise error(
            f"{path}: the pending best candidate records step {step!r}, not an integer"
        )
    if int(step) != int(best_check.get("step", -1)):
        raise error(
            f"{path}: the pending best candidate is from step {step}, but the best "
            f"record it is supposed to correspond to is from step "
            f"{best_check.get('step')!r}; the two describe different moments"
        )
    if not isinstance(candidate.get("model"), Mapping):
        raise error(
            f"{path}: the pending best candidate carries no model state, so it cannot "
            "be published as a checkpoint"
        )
    fingerprint = best_check.get("parameter_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise error(
            f"{path}: the best record carries no parameter fingerprint, so the "
            "candidate cannot be shown to be the parameters that were selected"
        )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "DETONATION_VELOCITY_FIELD",
    "DETONATION_VELOCITY_RELATIVE_TOLERANCE",
    "PUBLISH_ARTIFACTS_KEY",
    "SCALES_CHARGE_RADIUS_KEY",
    "SCALES_START_TIME_KEY",
    "PUBLISH_MARKER",
    "PUBLISH_OWNER_KEY",
    "TrainingError",
    "publish_marker_path",
    "read_publish_marker",
    "refuse_if_uncommitted",
    "require_best_candidate_consistency",
    "require_detonation_velocity",
    "require_publish_marker_contract",
    "uncommitted_artifacts",
]
