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
  互相印证：候选要走**与顶层载荷同一份** :func:`validate_state_snapshot`，
  指纹要**从候选的实际参数重算**，步号与选择依据要与最佳记录对得上。
  一个 ``best_dirty=True`` 却没有任何候选（或候选不完整）的检查点不能"加载成功
  并把标志悄悄清掉"：那会让一次中断之后的最佳模型无声消失。

:func:`validate_state_snapshot` 覆盖的是**模型以外的**那部分状态：优化器结构
（:func:`validate_optimizer_state`）、全局随机状态能不能真被生成器接受、阶段与
窗口跟**这份快照自己的**步号对不对得上。这几项此前只查了"键在不在、是不是
Mapping"，于是 ``optimizer = {}``、``rng['torch'] = 0``、
``stage.window_fraction = 0.123`` 都能**加载成功、正常发布**，直到**下一个**
实例恢复 ``best.pt`` 时才抛错（复核实测）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np

# 时间原点的定义只有一份，在 ``physics/similitude.py``：``T_h = R_0/D + epsilon``。
# 校验逐算例 ``start_time`` 时直接用这个函数，不在这里重写一遍公式
# （``physics`` 不反向依赖 ``training``，无循环导入）。
from ..physics.similitude import handover_time

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

#: 逐算例记录的键名。名字全部来自**实际写入端**，只有一份定义：
#: ``scales`` 来自 :meth:`..physics.similitude.SimilitudeScales.as_dict`，
#: ``data_identity`` 与 ``provenance`` 来自 :meth:`..training.trainer.Trainer._data_identity`
#: 与 :meth:`..training.trainer.Trainer._provenance`。
#:
#: ``as_dict()`` **不写** ``detonation_velocity``，写的是
#: ``start_time = T_h = R_0/D + epsilon(R_0)``；``D`` 由 ``(charge_radius, start_time)``
#: 唯一确定。所以"逐算例记录里的冻结值"就是这一对。
SCALES_CHARGE_RADIUS_KEY = "charge_radius"
SCALES_START_TIME_KEY = "start_time"
SCALES_TIME_SCALE_KEY = "time_scale"

#: 算例身份在各处记录中的键。同一个 ``case_id`` 必须串起这四份记录。
CASE_IDS_KEY = "case_ids"
SCALES_KEY = "scales"
DATA_IDENTITY_KEY = "data_identity"
PROVENANCE_KEY = "provenance"

#: ``data_identity`` 里逐算例的半径。
DATA_IDENTITY_CHARGE_RADIUS_KEY = "charge_radius_m"

#: ``provenance`` 里的逐算例记录。``charge_radii`` 与 ``case_ids`` 是**平行列表**
#: （同一个 ``self.cases`` 顺序），``start_times`` 与 ``valid_pinn_time_max``
#: 按 ``case_id`` 取键。
PROVENANCE_CASE_IDS_KEY = "case_ids"
PROVENANCE_CHARGE_RADII_KEY = "charge_radii"
PROVENANCE_START_TIMES_KEY = "start_times"
PROVENANCE_VALID_TIME_KEY = "valid_pinn_time_max"

#: 比较两个**尺度**（半径、起始时间、``hat T_*``）时的相对容差。
#:
#: 这些值在写入端与读取端是同一个 ``float``，经 ``torch.save`` 往返**逐位不变**，
#: 所以容差只需要覆盖浮点表示本身的舍入。它同时必须**远小于**任何真实矛盾：
#: 本项复现的"半径 0.05 → 0.10"是 100% 的相对差。用 1e-9 而不是更松的值，
#: 就是为了不靠"放宽容差"把明显矛盾掩盖过去。
SCALES_RELATIVE_TOLERANCE = 1.0e-9

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


def _case_ids(payload: Mapping[str, Any], path: Path | str, error: type[Exception]) -> list[str]:
    """``case_ids``：非空、不重复、每项是算例名的列表。"""
    case_ids = payload.get(CASE_IDS_KEY)
    if isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, Sequence) or not case_ids:
        raise error(
            f"{path} carries no usable {CASE_IDS_KEY!r} ({case_ids!r}); without the list "
            "of cases this checkpoint is for, the per-case records cannot be matched to "
            "one another"
        )
    for case_id in case_ids:
        if not isinstance(case_id, str) or not case_id:
            raise error(
                f"the checkpoint's {CASE_IDS_KEY!r} contains {case_id!r}, which is not a "
                "case name"
            )
    seen: dict[str, int] = {}
    for case_id in case_ids:
        seen[case_id] = seen.get(case_id, 0) + 1
    duplicated = sorted(name for name, count in seen.items() if count > 1)
    if duplicated:
        raise error(
            f"the checkpoint's {CASE_IDS_KEY!r} lists {duplicated} more than once; a case "
            "appearing twice cannot be matched to a single set of frozen scales"
        )
    return list(case_ids)


def _associated_record(
    mapping: Any,
    key: str,
    case_id: str,
    *,
    path: Path | str,
    where: str,
    error: type[Exception],
) -> Mapping[str, Any]:
    """从 ``mapping`` 里按 ``case_id`` 取一条逐算例记录，缺了就是硬失败。"""
    if not isinstance(mapping, Mapping):
        raise error(
            f"{path} carries no {where} block ({type(mapping).__name__}); the per-case "
            "records it should hold cannot be matched"
        )
    if case_id not in mapping:
        raise error(
            f"{path} records no {where} entry for case {case_id!r}, which its own "
            f"{CASE_IDS_KEY!r} lists. A missing record is not a pass: that case is part "
            "of the run"
        )
    record = mapping[case_id]
    if not isinstance(record, Mapping):
        raise error(
            f"the checkpoint's {where} entry for case {case_id!r} is "
            f"{type(record).__name__}, not a record"
        )
    return record


def require_case_scales_association(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    expected_cases: Mapping[str, tuple[float, float]] | None = None,
    error: type[Exception] = TrainingError,
) -> dict[str, dict[str, float]]:
    """**逐算例关联校验**：四份记录必须指同一批算例、同一个半径。

    一份检查点里，同一个算例被写了四遍：

    ==================  ==========================  ===============================
    位置                键                          写入方
    ==================  ==========================  ===============================
    ``case_ids``        算例名列表                  ``Trainer._build_payload``
    ``scales``          ``{case_id: as_dict()}``    ``_build_payload``
    ``data_identity``   ``{case_id: {...}}``        ``Trainer._data_identity``
    ``provenance``      ``{case_id: ...}`` 与平行列表  ``Trainer._provenance``
    ==================  ==========================  ===============================

    以前它们之间**没有任何核对**：把 ``scales[case_id].charge_radius`` 从 0.05 改成
    0.10、同时把 ``start_time`` 乘 2，反算出的爆速仍然是 6930，于是共享校验、
    ``Trainer.load()``、预测加载**全部通过**，实际预测 CLI 也返回 0——只是默认
    结束时间从 300 µs 变成 154.108 µs、``s_max`` 从 5.59815 变成 2.79908
    （复核实测）。尺度记录已经不属于那个算例了，却没人发现。

    校验按 ``case_id`` 关联，逐项如下（**不排序、不取第一个、不跳过**）：

    1. ``case_ids`` 非空、不重复；
    2. ``scales`` 与 ``data_identity`` 的键集都**恰好等于** ``case_ids``
       （缺记录与来源不明的多记录都是硬失败）；
    3. ``scales[case_id].charge_radius`` == ``data_identity[case_id].charge_radius_m``；
    4. ``scales[case_id].time_scale`` == 顶层 ``time_scale``；
    5. ``provenance['case_ids']`` 与 ``case_ids`` 是同一批算例，
       ``provenance['charge_radii']`` 按下标与它平行、且等于对应算例
       ``scales`` 里的半径；
    6. ``provenance['start_times'][case_id]`` 与
       ``provenance['valid_pinn_time_max']`` 都按 ``case_id`` 覆盖同一批算例；
    7. ``expected_cases``（恢复端传入接收实例的 ``{case_id: (半径, 起始时间)}``）
       不为 ``None`` 时，算例集合、每个算例的半径与起始时间都要与它一致。

    "起始时间符合当前时间定义"（``T_h = R_0/D + epsilon(R_0)``，用顶层冻结 ``D``）
    由 :func:`require_detonation_velocity` 用反解出的半径与它自己的 ``D`` 检查——
    这里只知道半径，不知道 ``D``。

    返回 ``{case_id: {"charge_radius": ..., "start_time": ...}}``。
    """

    def fail(reason: str) -> NoReturn:
        raise error(f"{path}: {reason}")

    def same(left: float, right: float) -> bool:
        return abs(left - right) <= SCALES_RELATIVE_TOLERANCE * max(abs(left), abs(right))

    def require_scale(value: Any, case_id: str, where: str, field: str) -> float:
        number = _real_number(value)
        if number is None or not math.isfinite(number) or number <= 0.0:
            fail(
                f"its {where} for case {case_id!r} records {field} = {value!r}, which is "
                "not a finite positive number"
            )
        return number

    case_ids = _case_ids(payload, path, error)
    scales = payload.get(SCALES_KEY)
    identity = payload.get(DATA_IDENTITY_KEY)
    if not isinstance(scales, Mapping):
        fail(
            f"it carries no {SCALES_KEY!r} block ({type(scales).__name__}); the per-case "
            "frozen scales are what the training run actually used"
        )
    if not isinstance(identity, Mapping):
        fail(
            f"it carries no {DATA_IDENTITY_KEY!r} block ({type(identity).__name__}); "
            "nothing ties the case names to the data they came from"
        )
    expected_ids = set(case_ids)
    for where, mapping in ((SCALES_KEY, scales), (DATA_IDENTITY_KEY, identity)):
        missing = [case_id for case_id in case_ids if case_id not in mapping]
        if missing:
            fail(
                f"its {where!r} has no entry for {missing}, which its own "
                f"{CASE_IDS_KEY!r} lists; a missing record is not a pass"
            )
        unexpected = sorted(str(key) for key in mapping if key not in expected_ids)
        if unexpected:
            fail(
                f"its {where!r} carries entries for {unexpected}, which its own "
                f"{CASE_IDS_KEY!r} does not list; the two describe different sets of cases"
            )

    declared_time_scale = _real_number(payload.get(SCALES_TIME_SCALE_KEY))
    if declared_time_scale is None or not math.isfinite(declared_time_scale):
        fail(
            f"it records no usable top-level {SCALES_TIME_SCALE_KEY!r} "
            f"({payload.get(SCALES_TIME_SCALE_KEY)!r}) to match the per-case scales against"
        )

    resolved: dict[str, dict[str, float]] = {}
    for case_id in case_ids:
        record = _associated_record(
            scales, case_id, case_id, path=path, where=SCALES_KEY, error=error
        )
        radius = require_scale(
            record.get(SCALES_CHARGE_RADIUS_KEY), case_id, SCALES_KEY, SCALES_CHARGE_RADIUS_KEY
        )
        start_time = require_scale(
            record.get(SCALES_START_TIME_KEY), case_id, SCALES_KEY, SCALES_START_TIME_KEY
        )
        time_scale = require_scale(
            record.get(SCALES_TIME_SCALE_KEY), case_id, SCALES_KEY, SCALES_TIME_SCALE_KEY
        )
        if not same(time_scale, declared_time_scale):
            fail(
                f"its scales for case {case_id!r} carry {SCALES_TIME_SCALE_KEY} = "
                f"{time_scale!r} but the checkpoint declares {SCALES_TIME_SCALE_KEY} = "
                f"{declared_time_scale!r}; the same case is described with two different "
                "hat T_*, so the network's time input is ambiguous"
            )

        case_identity = _associated_record(
            identity, case_id, case_id, path=path, where=DATA_IDENTITY_KEY, error=error
        )
        identity_radius = require_scale(
            case_identity.get(DATA_IDENTITY_CHARGE_RADIUS_KEY),
            case_id,
            DATA_IDENTITY_KEY,
            DATA_IDENTITY_CHARGE_RADIUS_KEY,
        )
        if not same(radius, identity_radius):
            fail(
                f"its scales for case {case_id!r} record {SCALES_CHARGE_RADIUS_KEY} = "
                f"{radius!r} m, but its {DATA_IDENTITY_KEY!r} for the same case records "
                f"{DATA_IDENTITY_CHARGE_RADIUS_KEY} = {identity_radius!r} m. The scale "
                "record no longer belongs to that case's data"
            )
        resolved[case_id] = {
            SCALES_CHARGE_RADIUS_KEY: radius,
            SCALES_START_TIME_KEY: start_time,
        }

    # -- provenance 里的同一批逐算例记录 ------------------------------------
    provenance = payload.get(PROVENANCE_KEY)
    if not isinstance(provenance, Mapping):
        fail(
            f"it carries no {PROVENANCE_KEY!r} block ({type(provenance).__name__}); the "
            "per-case records there are another description of the same run"
        )
    provenance_ids = provenance.get(PROVENANCE_CASE_IDS_KEY)
    if (
        isinstance(provenance_ids, (str, bytes))
        or not isinstance(provenance_ids, Sequence)
        or not provenance_ids
    ):
        fail(
            f"its {PROVENANCE_KEY}[{PROVENANCE_CASE_IDS_KEY!r}] is "
            f"{provenance_ids!r}, not a non-empty list of case names"
        )
    if set(provenance_ids) != expected_ids:
        fail(
            f"its top-level {CASE_IDS_KEY!r} is {sorted(expected_ids)} but "
            f"{PROVENANCE_KEY}[{PROVENANCE_CASE_IDS_KEY!r}] says "
            f"{sorted(str(name) for name in provenance_ids)}; the run is described with "
            "two different sets of cases"
        )

    radii = provenance.get(PROVENANCE_CHARGE_RADII_KEY)
    if (
        isinstance(radii, (str, bytes))
        or not isinstance(radii, Sequence)
        or len(radii) != len(provenance_ids)
    ):
        fail(
            f"its {PROVENANCE_KEY}[{PROVENANCE_CHARGE_RADII_KEY!r}] is not a list running "
            f"parallel to [{PROVENANCE_CASE_IDS_KEY!r}] "
            f"({len(provenance_ids)} case names, radii = {radii!r})"
        )
    for index, case_id in enumerate(provenance_ids):
        recorded = require_scale(
            radii[index], str(case_id), PROVENANCE_KEY, PROVENANCE_CHARGE_RADII_KEY
        )
        if case_id not in resolved:
            fail(
                f"its {PROVENANCE_KEY}[{PROVENANCE_CASE_IDS_KEY!r}] names case "
                f"{case_id!r}, which its top-level {CASE_IDS_KEY!r} does not"
            )
        if not same(recorded, resolved[case_id][SCALES_CHARGE_RADIUS_KEY]):
            fail(
                f"its {PROVENANCE_KEY}[{PROVENANCE_CHARGE_RADII_KEY!r}][{index}] = "
                f"{recorded!r} m is for case {case_id!r}, whose scales record "
                f"{SCALES_CHARGE_RADIUS_KEY} = "
                f"{resolved[case_id][SCALES_CHARGE_RADIUS_KEY]!r} m; the parallel list no "
                "longer lines up with the scales"
            )

    start_times = provenance.get(PROVENANCE_START_TIMES_KEY)
    if not isinstance(start_times, Mapping):
        fail(
            f"its {PROVENANCE_KEY}[{PROVENANCE_START_TIMES_KEY!r}] is "
            f"{type(start_times).__name__}, not a per-case record"
        )
    valid_times = provenance.get(PROVENANCE_VALID_TIME_KEY)
    if not isinstance(valid_times, Mapping):
        fail(
            f"its {PROVENANCE_KEY}[{PROVENANCE_VALID_TIME_KEY!r}] is "
            f"{type(valid_times).__name__}, not a per-case record"
        )
    for where, mapping in (
        (PROVENANCE_START_TIMES_KEY, start_times),
        (PROVENANCE_VALID_TIME_KEY, valid_times),
    ):
        missing = [case_id for case_id in case_ids if case_id not in mapping]
        if missing:
            fail(
                f"its {PROVENANCE_KEY}[{where!r}] has no entry for {missing}, which its "
                f"own {CASE_IDS_KEY!r} lists"
            )
        unexpected = sorted(str(key) for key in mapping if key not in expected_ids)
        if unexpected:
            fail(
                f"its {PROVENANCE_KEY}[{where!r}] carries entries for {unexpected}, which "
                f"its own {CASE_IDS_KEY!r} does not list"
            )

    # ``start_times`` 是同一件事的**第二份记录**：它与该算例 ``scales`` 里的
    # ``start_time`` 必须逐位一致。只核对键集是不够的——那个值决定整条时间轴，
    # 改掉它同样会让同一份检查点在两处被读出两个原点。
    for case_id in case_ids:
        recorded = _real_number(start_times[case_id])
        if recorded is None or not math.isfinite(recorded) or recorded <= 0.0:
            fail(
                f"its {PROVENANCE_KEY}[{PROVENANCE_START_TIMES_KEY!r}][{case_id!r}] is "
                f"{start_times[case_id]!r}, not a finite positive time"
            )
        if not same(recorded, resolved[case_id][SCALES_START_TIME_KEY]):
            fail(
                f"its {PROVENANCE_KEY}[{PROVENANCE_START_TIMES_KEY!r}][{case_id!r}] = "
                f"{recorded!r} s disagrees with its scales for the same case, which "
                f"record {SCALES_START_TIME_KEY} = "
                f"{resolved[case_id][SCALES_START_TIME_KEY]!r} s: one case, two time "
                "origins"
            )
    # ``valid_pinn_time_max`` 是数据里量出来的 ``t_PINN`` 上界，没有第二份记录可
    # 对，但它进 ``s_max``——所以至少必须是一个有限的非负数，不能是 ``None``/字符串
    # /负数那种"读起来像覆盖范围"的东西。
    for case_id in case_ids:
        value = _real_number(valid_times[case_id])
        if value is None or not math.isfinite(value) or value < 0.0:
            fail(
                f"its {PROVENANCE_KEY}[{PROVENANCE_VALID_TIME_KEY!r}][{case_id!r}] is "
                f"{valid_times[case_id]!r}, not a finite non-negative PINN time"
            )

    # -- 恢复端：与接收实例**实际使用的**算例半径和起始时间对照 ------------
    if expected_cases is not None:
        if set(expected_cases) != expected_ids:
            fail(
                f"the checkpoint is for cases {sorted(expected_ids)}, but the receiving "
                f"trainer holds {sorted(expected_cases)}; resuming across different cases "
                "is not a resume"
            )
        for case_id in case_ids:
            radius, start_time = expected_cases[case_id]
            if not same(resolved[case_id][SCALES_CHARGE_RADIUS_KEY], float(radius)):
                fail(
                    f"the checkpoint's scales for case {case_id!r} use "
                    f"{SCALES_CHARGE_RADIUS_KEY} = "
                    f"{resolved[case_id][SCALES_CHARGE_RADIUS_KEY]!r} m, but this "
                    f"trainer's case {case_id!r} has radius {float(radius)!r} m"
                )
            if not same(resolved[case_id][SCALES_START_TIME_KEY], float(start_time)):
                fail(
                    f"the checkpoint's scales for case {case_id!r} put the time origin at "
                    f"{resolved[case_id][SCALES_START_TIME_KEY]!r} s, but this trainer's "
                    f"case {case_id!r} starts at {float(start_time)!r} s; resuming would "
                    "shift that case's trajectory"
                )
    return resolved


def require_detonation_velocity(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    expected: float | None = None,
    expected_cases: Mapping[str, tuple[float, float]] | None = None,
    error: type[Exception] = TrainingError,
) -> float:
    """读出检查点里**冻结的**爆速并校验，返回它。

    保存端写什么、读取端要求什么只有这一份定义。``Trainer.load`` 与
    ``cli.predict.load_model_from_checkpoint`` 都调它——以前**只有预测端**调，
    于是删掉这个字段或把它改成 ``-1`` 时"恢复训练放行、预测拒绝"（复核实测）。

    校验分四层，任何一层不成立都**硬失败**：

    1. **顶层字段**必须存在，且是合法的**有限正实数**（``True``、``"6930"``、
       ``[6930]`` 都不算——不许用 ``float()`` 把错误类型转成合法输入）；
    2. **逐算例关联**：``case_ids`` / ``scales`` / ``data_identity`` / ``provenance``
       必须指同一批算例、同一个半径。这一层由
       :func:`require_case_scales_association` 一处实现；
    3. **时间定义**：每个算例的 ``scales[case_id][start_time]`` 必须等于
       ``T_h = R_0/D + epsilon(R_0)``——用顶层那个 ``D`` 与**该算例自己的半径**
       正向算一遍。用不了反解那一套来兜底：半径与起始时间**同时**翻倍时反解出的
       ``D`` 一模一样，只有正向定义对得上才对得上；
    4. ``expected`` / ``expected_cases``（恢复端传入接收实例**实际使用**的爆速，
       以及每个算例的半径与起始时间）不为 ``None`` 时，还要与**接收实例**一致。
       只查"文件内部自洽"是不够的：一份自洽的检查点照样能被另一个爆速、另一个
       半径的实例恢复——实测同一份文件在恢复端给出 ``T_h = 8.21500721501 µs``、
       在预测端给出 ``41.5186385737 µs``。

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

    # -- 第二层：逐算例关联（case_ids / scales / data_identity / provenance）--
    #
    # 这一层要回答的是"每条记录说的是不是**同一个算例、同一个半径**"。只比爆速
    # 远远不够：把 ``scales[case_id].charge_radius`` 从 0.05 改成 0.10、同时把
    # ``start_time`` 乘 2，反算出的爆速**仍然是** 6930，于是"冻结参数一致"这条
    # 检查完全拦不住，而尺度记录已经不属于那个算例（复核实测：默认预测结束时间
    # 从 300 µs 变成 154.108 µs、``s_max`` 从 5.59815 变成 2.79908）。
    cases = require_case_scales_association(
        payload, path, expected_cases=expected_cases, error=error
    )

    # -- 第三层：起始时间必须**符合当前时间定义** ``T_h = R_0/D + epsilon`` --
    #
    # 用顶层那个 ``D`` 正向算一遍，与记录里的 ``start_time`` 比——这比"反解出 D
    # 再比"更直接地表达了契约，而且同一份公式只有一份定义（``physics.similitude``）。
    for case_id in cases:
        radius = cases[case_id][SCALES_CHARGE_RADIUS_KEY]
        recorded = cases[case_id][SCALES_START_TIME_KEY]
        defined = handover_time(radius, detonation_velocity=velocity)
        if abs(recorded - defined) > DETONATION_VELOCITY_RELATIVE_TOLERANCE * max(
            abs(recorded), abs(defined)
        ):
            raise error(
                f"the checkpoint's scales for case {case_id!r} record "
                f"{SCALES_START_TIME_KEY} = {recorded!r} s, but the checkpoint's own "
                f"{DETONATION_VELOCITY_FIELD} = {velocity!r} m/s at "
                f"{SCALES_CHARGE_RADIUS_KEY} = {radius!r} m gives "
                f"T_h = R_0/D + epsilon(R_0) = {defined!r} s. The case's time origin and "
                "its frozen detonation velocity describe different time axes"
            )

    # -- 第四层：与**接收实例实际使用的**爆速一致（恢复端传入 expected） ----
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
    "CHECKPOINT_VERSION",
    "CASE_IDS_KEY",
    "DATA_IDENTITY_CHARGE_RADIUS_KEY",
    "DATA_IDENTITY_KEY",
    "DETONATION_VELOCITY_FIELD",
    "DETONATION_VELOCITY_RELATIVE_TOLERANCE",
    "PROVENANCE_CASE_IDS_KEY",
    "PROVENANCE_CHARGE_RADII_KEY",
    "PROVENANCE_KEY",
    "PROVENANCE_START_TIMES_KEY",
    "PROVENANCE_VALID_TIME_KEY",
    "PUBLISH_ARTIFACTS_KEY",
    "SCALES_CHARGE_RADIUS_KEY",
    "SCALES_KEY",
    "SCALES_RELATIVE_TOLERANCE",
    "SCALES_START_TIME_KEY",
    "SCALES_TIME_SCALE_KEY",
    "PUBLISH_MARKER",
    "PUBLISH_OWNER_KEY",
    "TrainingError",
    "parameter_fingerprint_of",
    "publish_marker_path",
    "read_publish_marker",
    "refuse_if_uncommitted",
    "require_best_candidate_consistency",
    "validate_model_state",
    "validate_optimizer_state",
    "validate_state_snapshot",
    "require_case_scales_association",
    "require_detonation_velocity",
    "require_publish_marker_contract",
    "uncommitted_artifacts",
]
