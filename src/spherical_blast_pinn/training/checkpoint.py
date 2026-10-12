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
* :func:`refuse_if_uncommitted` —— 多文件发布不是原子的。发布前写下标记、成功后
  删除，**两个入口都必须落实**这条规则。以前只有 ``Trainer.load`` 查它，于是
  "恢复训练被拒绝的文件，仍能被预测入口读进去"（复核实测）。
* :func:`require_best_candidate_consistency` —— ``best_dirty`` 与候选快照必须
  互相印证。一个 ``best_dirty=True`` 却没有任何候选的检查点不能"加载成功并把
  标志悄悄清掉"：那会让一次中断之后的最佳模型无声消失。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

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

#: 未提交标记的文件名。发布多份产物**不是**原子的：进程在两次写入之间被杀掉，
#: 目录里会留下"一部分新、一部分旧"的状态，而每一份单看都是合法文件。
PUBLISH_MARKER = ".publish_in_progress.json"

#: 标记里记录"这次发布属于哪个事务"的键。没有它，一个后来的事务就无法判断
#: 标记是不是自己写的，只能**无条件**删除——那会把上一次中断的痕迹一起抹掉，
#: 让一套从未提交成功的产物变成"看起来已提交"（复核实测）。
PUBLISH_OWNER_KEY = "owner"


class TrainingError(RuntimeError):
    """训练无法继续。"""


# --------------------------------------------------------------------------
# 多文件发布的读取规则
# --------------------------------------------------------------------------


def publish_marker_path(output_dir: Path | str) -> Path:
    """标记总在**它描述的那批产物旁边**。"""
    return Path(output_dir) / PUBLISH_MARKER


def read_publish_marker(
    output_dir: Path | str, *, error: type[Exception] = TrainingError
) -> dict[str, Any] | None:
    """读 ``output_dir`` 里的未提交标记。没有标记时返回 ``None``。

    参数是**标记所在目录**，不是某个检查点：调用方有时按目录问（"这个目录里
    有没有没发布的产物"），有时按检查点问（"这一份能不能读"）。后者
    :func:`refuse_if_uncommitted` 会自己取父目录。

    标记存在却读不出来时**保守拒绝**：宁可让一个可能完好的目录多拒绝一次，
    也不能把"读不懂标记"当成"没有标记"。
    """
    marker = publish_marker_path(output_dir)
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 读不懂就拒绝，不做任何猜测
        raise error(
            f"{marker} exists but cannot be read ({exc.__class__.__name__}: {exc}); a "
            "publish was interrupted, so the artifacts beside it cannot be trusted as "
            "committed"
        ) from exc
    if not isinstance(payload, dict):
        raise error(
            f"{marker} is not a JSON object ({type(payload).__name__}); a publish was "
            "interrupted, so the artifacts beside it cannot be trusted as committed"
        )
    return payload


def uncommitted_artifacts(
    output_dir: Path | str, *, error: type[Exception] = TrainingError
) -> set[str] | None:
    """``output_dir`` 里那批**未提交**产物的名字；没有标记时 ``None``。"""
    payload = read_publish_marker(output_dir, error=error)
    if payload is None:
        return None
    return {str(name) for name in (payload.get("artifacts") or ())}


def refuse_if_uncommitted(
    checkpoint_path: Path | str, *, error: type[Exception] = TrainingError
) -> None:
    """加载端必须**落实**拒绝规则，而不是只在日志里提一句。

    训练恢复与预测入口都调它。只在一处检查时，"恢复被拒绝"与"可以用来预测"
    会同时成立——而预测同样会拿一套新旧混合的产物当真值使用。
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


def require_detonation_velocity(
    payload: Mapping[str, Any],
    path: Path | str,
    *,
    error: type[Exception] = TrainingError,
) -> float:
    """读出检查点里**冻结的**爆速并校验。

    三个拒绝理由各自报出来：**没有这个字段**（旧写入端根本没保存它）、
    **不是数**、**不是有限正数**。三者都不允许回落到默认常量——``D`` 进
    ``T_h = R_0/D + epsilon``，默认值换来的是整条轨迹的平移，而不是一个错误。
    """
    if DETONATION_VELOCITY_FIELD not in payload:
        raise error(
            f"{path} carries no {DETONATION_VELOCITY_FIELD!r}. That field is the "
            "charge's detonation velocity, a frozen physical parameter: the scheme-B "
            "time origin T_h = R_0/D + epsilon cannot be reconstructed without it, and "
            "substituting a default would shift the whole predicted trajectory by the "
            "detonation transit time. Checkpoints written before this field existed "
            "must be retrained, not reinterpreted"
        )
    raw = payload[DETONATION_VELOCITY_FIELD]
    try:
        velocity = float(raw)
    except (TypeError, ValueError) as exc:
        raise error(
            f"the checkpoint's {DETONATION_VELOCITY_FIELD} is {raw!r}, not a number"
        ) from exc
    if not math.isfinite(velocity) or velocity <= 0.0:
        raise error(
            f"the checkpoint's {DETONATION_VELOCITY_FIELD} must be finite and positive, "
            f"got {velocity!r}; it sets the time origin T_h = R_0/D + epsilon"
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
    "PUBLISH_MARKER",
    "PUBLISH_OWNER_KEY",
    "TrainingError",
    "publish_marker_path",
    "read_publish_marker",
    "refuse_if_uncommitted",
    "require_best_candidate_consistency",
    "require_detonation_velocity",
    "uncommitted_artifacts",
]
