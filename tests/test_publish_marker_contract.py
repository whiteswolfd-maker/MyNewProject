"""未提交发布标记（``.publish_in_progress.json``）的**严格**结构与读取契约。

第十轮 §2 点名的漏洞：标记的结构**没有被校验**。``uncommitted_artifacts()`` 里
那句 ``{str(name) for name in (payload.get("artifacts") or ())}`` 一次犯了三件事：

* ``{}`` 与 ``{"artifacts": []}`` —— 取不到/取到空的 ``artifacts`` 被 ``or ()``
  兜成"没有未提交产物"，于是**放行**；
* ``{"artifacts": "last.pt"}`` —— 字符串被逐字符遍历成
  ``{'l', 'a', 's', 't', '.', 'p'}``，``"last.pt" in artifacts`` 为假，同样**放行**；
* ``str(name)`` —— 任何类型都被强行转成字符串，错误类型永远不会被判为错。

于是"在真实合法检查点旁边放一个坏标记"这种输入，**训练恢复与预测两个入口都会
放行**。这些标记本身是合法的 JSON（一个对象），所以"不是 JSON 对象就拒绝"那条
检查也拦不住它们。

这个文件钉住修后的契约：标记存在就必须能按**明确契约**读懂，读不懂一律拒绝，
而且拒绝发生在**两个入口**上、发生在**任何状态被改动之前**。

夹具是真实的一份检查点（真实参数更新、真实落盘、真实检查点格式）。反向用例
只改标记，不动检查点内容。合法已提交的检查点保持原有推理语义。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.cli import predict as predict_module
from spherical_blast_pinn.cli._common import SetupError
from spherical_blast_pinn.training.checkpoint import (
    PUBLISH_MARKER,
    PUBLISH_OWNER_KEY,
    publish_marker_path,
)
from spherical_blast_pinn.training.trainer import TrainingError

from test_best_candidate_and_transaction import _tiny

OWNER = "some-other-transaction"


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def _trained(monkeypatch, tmp_path, name: str = "contract"):
    """真实跑一次 ``run()``，得到一份真实、合法、已提交的检查点。"""
    trainer, _ = _tiny(monkeypatch, tmp_path, name=name)
    trainer.run(max_updates=2, verbose=False)
    return trainer, trainer.output_dir / "last.pt"


def _fresh(monkeypatch, tmp_path, name: str):
    """另一个实例：输出目录与写出检查点的那次运行**不同**。"""
    trainer, _ = _tiny(monkeypatch, tmp_path, name=f"fresh_{name}")
    return trainer


def _marker_path(trainer) -> Path:
    return trainer.output_dir / PUBLISH_MARKER


def _place_marker(trainer, payload, *, raw: str | None = None) -> Path:
    """在检查点旁边放一个标记。``payload`` 是对象，``raw`` 直接写原文字节。"""
    marker = _marker_path(trainer)
    marker.parent.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps(payload)
    marker.write_text(text, encoding="utf-8")
    return marker


def _radius(trainer) -> float:
    return float(trainer.cases[0].charge_radius)


def _config(tmp_path: Path) -> Path:
    """预测入口要一个真实配置。``--output`` 另行给出，这里只保证配置可加载。"""
    base_yaml = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    path = tmp_path / "predict_config.yaml"
    path.write_text(
        f"inherits: {base_yaml.as_posix()}\n"
        f"training:\n  output_dir: {(tmp_path / 'run').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def _load_outcome(monkeypatch, tmp_path, trainer, checkpoint, name):
    """返回 ``(trainer_outcome, predict_outcome)``，各自是异常或 ``"accepted"``。

    **两个入口都必须被问到**：只查一处时，"恢复被拒绝"与"可以用来预测"会同时
    成立，而预测同样把一套新旧混合的产物当真值用。
    """
    fresh = _fresh(monkeypatch, tmp_path, name)
    try:
        fresh.load(checkpoint)
        train_outcome = "accepted"
    except TrainingError as exc:
        train_outcome = exc
    try:
        predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )
        predict_outcome = "accepted"
    except SetupError as exc:
        predict_outcome = exc
    return train_outcome, predict_outcome


# ==========================================================================
# 正向控制：没有标记就必须正常读进去
# ==========================================================================


def test_without_a_marker_both_readers_accept(monkeypatch, tmp_path):
    """规则 1：没有发布标记时，合法检查点两个入口都能加载。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "clean")
    assert not _marker_path(trainer).exists()

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "clean"
    )
    assert train_outcome == "accepted", train_outcome
    assert predict_outcome == "accepted", predict_outcome


# ==========================================================================
# 规则 2：合法标记 + 目标属于未提交产物 → 两个入口都拒绝
# ==========================================================================


def test_a_lawful_marker_blocks_both_readers(monkeypatch, tmp_path):
    """规则 2：结构合法的标记，列到了这一份检查点，两个入口都拒绝。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "lawful")
    marker = _place_marker(
        trainer, {"artifacts": ["last.pt"], PUBLISH_OWNER_KEY: OWNER}
    )

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "lawful"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome
        assert "uncommitted" in str(outcome)
        assert str(marker) in str(outcome), "错误信息里没有标记路径"


def test_a_lawful_marker_for_another_artifact_does_not_block(
    monkeypatch, tmp_path
):
    """合法标记只描述它列出的产物：没列到这一份，它仍然是已提交的。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "other_artifact")
    _place_marker(trainer, {"artifacts": ["best.pt"], PUBLISH_OWNER_KEY: OWNER})

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "other_artifact"
    )
    assert train_outcome == "accepted", train_outcome
    assert predict_outcome == "accepted", predict_outcome


# ==========================================================================
# 规则 3/4：三种已复现的坏标记 —— 本轮的核心回归
# ==========================================================================


@pytest.mark.parametrize(
    "payload, field",
    [
        pytest.param({}, "artifacts", id="empty-object"),
        pytest.param({"artifacts": []}, "artifacts", id="empty-list"),
        pytest.param({"artifacts": "last.pt"}, "artifacts", id="string-not-list"),
    ],
)
def test_the_three_reproduced_bad_markers_block_both_readers(
    monkeypatch, tmp_path, payload, field
):
    """**复核的三个原反例。** 修前：两个入口都**放行**。

    ``{}`` 与 ``{"artifacts": []}`` 被 ``or ()`` 兜成空集合；
    ``{"artifacts": "last.pt"}`` 被逐字符遍历成 ``{'l','a','s','t','.','p'}``。
    三者都是合法 JSON 对象，所以"不是对象就拒绝"拦不住它们。
    """
    trainer, checkpoint = _trained(monkeypatch, tmp_path, f"bad_{field}_{len(payload)}")
    _place_marker(trainer, payload)

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, f"bad_{field}"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), f"坏标记被放行了：{outcome!r}"
        assert str(checkpoint) in str(outcome) or str(_marker_path(trainer)) in str(outcome)
        assert field in str(outcome), f"错误信息没点名非法字段：{outcome}"


def test_a_corrupt_marker_blocks_both_readers(monkeypatch, tmp_path):
    """规则 3：JSON 损坏 → 明确拒绝，不能当成"没有标记"。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "corrupt")
    _place_marker(trainer, None, raw="{not json")

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "corrupt"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome
        assert "cannot be read" in str(outcome)


def test_a_marker_that_is_a_directory_blocks_both_readers(monkeypatch, tmp_path):
    """规则 3：标记存在但**读不出来**（这里是一个目录）→ 保守拒绝。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "unreadable")
    _marker_path(trainer).mkdir(parents=True, exist_ok=True)

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "unreadable"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome
        assert "cannot be read" in str(outcome)


@pytest.mark.parametrize(
    "payload", [[1, 2, 3], {"a": 1}, 7, None, True, "last.pt"]
)
def test_a_marker_that_is_not_a_json_object_blocks_both_readers(
    monkeypatch, tmp_path, payload
):
    """规则 3：顶层不是 JSON 对象 → 拒绝。"""
    trainer, checkpoint = _trained(
        monkeypatch, tmp_path, f"not_object_{abs(hash(json.dumps(payload))) % 1000}"
    )
    _place_marker(trainer, payload)

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "not_object"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome


# ==========================================================================
# 规则 4：artifacts 的类型与列表项
# ==========================================================================


@pytest.mark.parametrize(
    "artifacts",
    [
        pytest.param([""], id="empty-name"),
        pytest.param(["   "], id="blank-name"),
        pytest.param([None], id="none-item"),
        pytest.param([5], id="number-item"),
        pytest.param([True], id="bool-item"),
        pytest.param([["last.pt"]], id="nested-list-item"),
        pytest.param([{"name": "last.pt"}], id="object-item"),
        pytest.param(["sub/last.pt"], id="path-separator"),
        pytest.param(["sub\\last.pt"], id="windows-separator"),
        pytest.param([".."], id="parent-dir"),
        pytest.param(["."], id="current-dir"),
        # 下面两项故意**不**包含被加载的 ``last.pt``：修前 ``str()`` 会把坏项变成
        # 字符串，而名字对不上就不拒绝——只有真正校验列表项才会被拦下。
        pytest.param(["best.pt", ""], id="good-name-plus-empty"),
        pytest.param(["best.pt", None], id="good-name-plus-none"),
    ],
)
def test_illegal_artifact_entries_block_both_readers(monkeypatch, tmp_path, artifacts):
    """规则 4：每项必须是**合法、非空的文件名字符串**。

    修前 ``str(name)`` 把 ``5`` 变成 ``"5"``、把 ``None`` 变成 ``"None"``，
    错误类型永远不会被判为错。
    """
    tag = abs(hash(repr(artifacts))) % 10000
    trainer, checkpoint = _trained(monkeypatch, tmp_path, f"entry_{tag}")
    _place_marker(trainer, {"artifacts": artifacts, PUBLISH_OWNER_KEY: OWNER})

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, f"entry_{tag}"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), f"非法列表项被放行：{artifacts!r}"
        assert "artifacts" in str(outcome), outcome


@pytest.mark.parametrize(
    "artifacts",
    [
        pytest.param({}, id="object"),
        pytest.param("last.pt", id="string"),
        pytest.param(5, id="number"),
        pytest.param(None, id="none"),
        pytest.param(True, id="bool"),
        pytest.param([], id="empty-list"),
    ],
)
def test_illegal_artifacts_types_block_both_readers(monkeypatch, tmp_path, artifacts):
    """规则 4：``artifacts`` 必须是非空列表，字符串/字典/数字/None/空列表一律拒绝。"""
    tag = abs(hash(repr(artifacts))) % 10000
    trainer, checkpoint = _trained(monkeypatch, tmp_path, f"type_{tag}")
    _place_marker(trainer, {"artifacts": artifacts, PUBLISH_OWNER_KEY: OWNER})

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, f"type_{tag}"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), f"非法 artifacts 类型被放行：{artifacts!r}"
        assert "artifacts" in str(outcome), outcome


# ==========================================================================
# 规则 5：owner 等必需身份字段
# ==========================================================================


@pytest.mark.parametrize(
    "owner",
    [
        pytest.param(None, id="null-owner"),
        pytest.param("", id="empty-owner"),
        pytest.param("   ", id="blank-owner"),
        pytest.param(17, id="numeric-owner"),
        pytest.param(["a"], id="list-owner"),
        pytest.param({"id": "a"}, id="object-owner"),
        pytest.param(True, id="bool-owner"),
    ],
)
def test_a_marker_without_a_usable_owner_blocks_both_readers(
    monkeypatch, tmp_path, owner
):
    """规则 5：缺身份信息或身份信息非法 → 保持阻断，不得静默放行。

    这里列的是 ``best.pt``：被加载的 ``last.pt`` 名字对不上，所以**唯一的**
    拒绝理由只能是 owner 本身非法。修前 owner 根本不读，于是放行。
    """
    tag = abs(hash(repr(owner))) % 10000
    trainer, checkpoint = _trained(monkeypatch, tmp_path, f"owner_{tag}")
    _place_marker(trainer, {"artifacts": ["best.pt"], PUBLISH_OWNER_KEY: owner})

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, f"owner_{tag}"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), f"非法 owner 被放行：{owner!r}"
        assert PUBLISH_OWNER_KEY in str(outcome), outcome


def test_an_old_marker_without_an_owner_field_stays_blocking(monkeypatch, tmp_path):
    """规则 5：**老标记**（没有 ``owner`` 字段）不许被当成"本次事务的标记"放过。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "no_owner")
    _place_marker(trainer, {"artifacts": ["best.pt"]})

    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "no_owner"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome
        assert PUBLISH_OWNER_KEY in str(outcome), outcome


# ==========================================================================
# 规则 6：标记在**输入检查点所在目录**查找
# ==========================================================================


def test_the_marker_is_looked_up_beside_the_input_checkpoint(monkeypatch, tmp_path):
    """规则 6：新实例的输出目录干净，检查点旁边的标记照样拦住它。

    反过来也测：把标记放进**读取方的输出目录**（这里等同于"预测的输出目录"），
    输入检查点旁边没有标记，就不该被拦。
    """
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "beside")
    receiving = _fresh(monkeypatch, tmp_path, "beside")
    assert receiving.output_dir != trainer.output_dir

    # 标记放在读取方自己的输出目录里 —— 与输入检查点无关，不构成拒绝理由。
    _place_marker(receiving, {"artifacts": ["last.pt"], PUBLISH_OWNER_KEY: OWNER})
    receiving.load(checkpoint)  # 不该抛

    # 标记放进**输入检查点旁边** —— 两个入口拒绝，而且错误指向那个路径。
    marker = _place_marker(
        trainer, {"artifacts": ["last.pt"], PUBLISH_OWNER_KEY: OWNER}
    )
    train_outcome, predict_outcome = _load_outcome(
        monkeypatch, tmp_path, trainer, checkpoint, "beside2"
    )
    for outcome in (train_outcome, predict_outcome):
        assert isinstance(outcome, Exception), outcome
        assert str(marker) in str(outcome), outcome


def test_the_marker_path_helper_points_at_the_directory_not_the_checkpoint(
    tmp_path,
):
    """规则 6 的辅助证据：标记路径是 ``<目录>/.publish_in_progress.json``。"""
    directory = tmp_path / "run"
    assert publish_marker_path(directory) == directory / PUBLISH_MARKER
    assert publish_marker_path(directory / "last.pt") == directory / "last.pt" / PUBLISH_MARKER
    assert publish_marker_path(directory / "last.pt").parent != directory


# ==========================================================================
# 拒绝必须发生在任何状态被改动之前
# ==========================================================================


def _state(trainer) -> dict:
    rng = torch.get_rng_state()
    numpy_state = np.random.get_state()
    return {
        "step": int(trainer.step),
        "parameters": trainer.parameter_fingerprint(),
        "optimizer": repr(trainer.optimizer.state_dict()),
        "history": json.dumps(trainer.history, sort_keys=True, default=str),
        "sampler_rng": repr(trainer.sampler.rng_state()),
        "torch_rng": rng.numpy().tobytes(),
        "numpy_rng": numpy_state[1].tobytes(),
        "best_dirty": bool(trainer._best_dirty),
    }


def test_a_refused_resume_leaves_the_receiving_instance_untouched(
    monkeypatch, tmp_path
):
    """训练加载被拒绝之前，接收实例的模型、优化器、RNG、历史都不许变。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "untouched")
    _place_marker(trainer, {"artifacts": "last.pt"})

    receiving = _fresh(monkeypatch, tmp_path, "untouched")
    before = _state(receiving)
    with pytest.raises(TrainingError):
        receiving.load(checkpoint)
    assert _state(receiving) == before, "拒绝之前就把接收实例改掉了"


def test_a_refused_prediction_writes_nothing_and_keeps_existing_output(
    monkeypatch, tmp_path
):
    """预测被拒绝后，不产生新预测文件，也不覆盖已有结果。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "no_output")
    _place_marker(trainer, {"artifacts": []})

    config = _config(tmp_path)
    output_dir = tmp_path / "prediction"
    output_dir.mkdir()
    existing = output_dir / "80mm.npz"
    sentinel = b"a previous committed prediction"
    existing.write_bytes(sentinel)

    with pytest.raises(SetupError):
        predict_module.main([
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--radius", str(_radius(trainer)),
            "--output", str(existing),
            "--n-times", "5",
            "--n-radii", "20",
        ])

    assert existing.read_bytes() == sentinel, "被拒绝的预测覆盖了已有结果"
    assert sorted(p.name for p in output_dir.iterdir()) == ["80mm.npz"], (
        "被拒绝的预测写出了新文件"
    )


# ==========================================================================
# 合法已提交的检查点保持原有推理语义
# ==========================================================================


def test_a_committed_checkpoint_keeps_its_inference_semantics(monkeypatch, tmp_path):
    """反面：合法的 diagnostic 检查点照旧能推理，时间原点仍由冻结的 D 重建。

    这一份走 ``verify()`` + ``_save()``，是**诊断**来源（不是 ``run()`` 发布的
    那一套），确认新的标记契约没有把"合法的中间检查点"误伤成"未提交产物"。
    """
    import test_review_regressions as base
    from spherical_blast_pinn.physics.similitude import SimilitudeScales
    from spherical_blast_pinn.training.checkpoint import DETONATION_VELOCITY_FIELD

    trainer, _ = _tiny(monkeypatch, tmp_path, name="diagnostic")
    trainer.verify(verbose=False)
    checkpoint = trainer.output_dir / "pending.pt"
    trainer._save(checkpoint)
    assert not _marker_path(trainer).exists()

    config = _config(tmp_path)
    output = tmp_path / "diagnostic_prediction"
    radius = _radius(trainer)

    assert predict_module.main([
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--radius", str(radius),
        "--output", str(output / "out.npz"),
        "--n-times", "5",
        "--n-radii", "20",
    ]) == 0

    written = list(output.glob("*.npz"))
    assert written, "合法检查点没有产出预测文件"
    with np.load(written[0], allow_pickle=True) as handle:
        result = {key: handle[key] for key in handle.files}

    frozen_d = float(
        torch.load(checkpoint, map_location="cpu", weights_only=False)[
            DETONATION_VELOCITY_FIELD
        ]
    )
    expected = SimilitudeScales.for_radius(
        radius, time_scale=base.TIME_SCALE, detonation_velocity=frozen_d
    ).start_time()
    assert float(result["start_time"]) == pytest.approx(expected, rel=0, abs=1e-18)
    assert float(result["charge_radius"]) == pytest.approx(radius)
