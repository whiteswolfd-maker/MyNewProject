"""逐算例尺度记录与**真实算例身份、半径**的绑定（第 3 轮）。

第 3 轮点名的漏洞：同一份检查点里，一个算例被写了**四遍**——

==================  ==========================  ==============================
位置                键                          写入方
==================  ==========================  ==============================
``case_ids``        算例名列表                  ``Trainer._build_payload``
``scales``          ``{case_id: as_dict()}``    ``_build_payload``
``data_identity``   ``{case_id: {...}}``        ``Trainer._data_identity``
``provenance``      ``{case_id: ...}`` 与平行列表  ``Trainer._provenance``
==================  ==========================  ==============================

而它们之间**没有任何核对**。于是下面这份"矛盾副本"畅通无阻：

* 把 ``scales[case_id].charge_radius`` 从 ``0.05`` 改成 ``0.10``；
* 同时把同一条记录的 ``start_time`` **乘 2**；
* 顶层 ``detonation_velocity``、``data_identity``、``provenance`` 一律不动。

反算出的爆速**仍然是 6930**（``0.10 / (16.430µs - 2µs) == 6930``），所以第 2 轮
那条"冻结爆速一致"的检查完全拦不住它——但尺度记录已经不属于那个算例了。实测：

======================  ==============  ==============
                       正例（对照）      矛盾副本
======================  ==============  ==============
共享校验                通过            通过
``Trainer.load()``      通过            通过
预测加载                通过            通过
预测 CLI 退出码         0               0
默认结束物理时间        300 µs          154.108 µs
``s_max``               5.59815         2.79908
======================  ==============  ==============

同一关联问题的第二种形式：把 ``case_ids`` 与 ``scales`` 里的算例名一起改成
``ghost``，``data_identity`` 与 ``provenance`` 里仍是原名——共享校验与两个入口
**同样全部放行**，只有在后面算覆盖范围时才以另一条理由报错。

本文件钉住修后的契约。坏输入都是**主动构造**的（真实检查点上改指定字段，并先
断言变异生效），对照组与变异组不共享可变对象；正例从一致的算例配置构造。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.cli import predict as predict_module
from spherical_blast_pinn.cli._common import SetupError
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.similitude import REFERENCE_RADIUS, REFERENCE_START_TIME
from spherical_blast_pinn.training.checkpoint import (
    DATA_IDENTITY_CHARGE_RADIUS_KEY,
    DETONATION_VELOCITY_FIELD,
    PROVENANCE_CHARGE_RADII_KEY,
    PROVENANCE_START_TIMES_KEY,
    PROVENANCE_VALID_TIME_KEY,
    SCALES_CHARGE_RADIUS_KEY,
    SCALES_START_TIME_KEY,
    SCALES_TIME_SCALE_KEY,
    require_case_scales_association,
    require_detonation_velocity,
)
from spherical_blast_pinn.training.trainer import TrainingError

from test_best_candidate_and_transaction import _tiny
from test_frozen_detonation_velocity import _add_consistent_second_case

DEFAULT_D = 6930.0
TIME_SCALE = 2.0


def _expected_time_origin(radius: float, velocity: float) -> float:
    """按 AGENTS.md 的公式**独立**算 ``T_h = R_0/D + (R_0/R_ref) epsilon_ref``。"""
    return radius / velocity + (radius / REFERENCE_RADIUS) * REFERENCE_START_TIME


def _independent_coverage(payload) -> float:
    """独立算 ``s_max = max_cases (U_* t_PINN / R_0) / hat T_*``，不调被测函数。"""
    time_scale = float(payload[SCALES_TIME_SCALE_KEY])
    return max(
        mat.REFERENCE_VELOCITY
        * float(payload["provenance"][PROVENANCE_VALID_TIME_KEY][case_id])
        / float(payload["scales"][case_id][SCALES_CHARGE_RADIUS_KEY])
        for case_id in payload["case_ids"]
    ) / time_scale


def _independent_end_time(payload, target_radius: float) -> float:
    """``T = s_max hat T_* R_0/U_* + T_h``，用检查点自己的冻结参数。"""
    velocity = float(payload[DETONATION_VELOCITY_FIELD])
    return (
        _independent_coverage(payload)
        * float(payload[SCALES_TIME_SCALE_KEY])
        * target_radius
        / mat.REFERENCE_VELOCITY
        + _expected_time_origin(target_radius, velocity)
    )


# --------------------------------------------------------------------------
# 夹具：一份真实检查点（模块级，只训练一次），测试里只改**副本**
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    """真实训练出来的合法检查点：``(临时目录, Trainer, last.pt 路径)``。"""
    tmp = tmp_path_factory.mktemp("case_scales")
    patch = pytest.MonkeyPatch()
    try:
        trainer, _ = _tiny(patch, tmp, name="control")
        trainer.run(max_updates=2, verbose=False)
        yield tmp, trainer, Path(trainer.output_dir) / "last.pt"
    finally:
        patch.undo()


def _fingerprint(payload) -> bytes:
    import hashlib
    import pickle

    return hashlib.sha256(pickle.dumps(payload)).digest()


_COUNTER = [0]


def _variant(control_path: Path, tmp_path: Path, mutate=None):
    """复制一份对照检查点并改指定字段，**先断言变异生效**，返回 (路径, 载荷)。"""
    _COUNTER[0] += 1
    target = tmp_path / f"variant_{_COUNTER[0]}.pt"
    shutil.copy2(control_path, target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    before = _fingerprint(payload)
    if mutate is not None:
        mutate(payload)
    if mutate is None:
        return target, payload
    assert _fingerprint(payload) != before, "变异没有生效"
    torch.save(payload, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=False)
    assert _fingerprint(reloaded) != before, "变异没有写进文件"
    return target, reloaded


def _forge_radius_and_origin(payload, *, radius=0.10, factor=2.0):
    """本轮的核心矛盾副本：半径与起始时间一起改，反算出的 ``D`` **不变**。"""
    case_id = payload["case_ids"][0]
    record = dict(payload["scales"][case_id])
    record[SCALES_CHARGE_RADIUS_KEY] = radius
    record[SCALES_START_TIME_KEY] = float(record[SCALES_START_TIME_KEY]) * factor
    payload["scales"][case_id] = record
    return case_id


def _rename_case(payload, new_name="ghost"):
    case_id = payload["case_ids"][0]
    payload["case_ids"] = [new_name]
    payload["scales"] = {new_name: payload["scales"][case_id]}
    return new_name


def _fresh(monkeypatch, tmp_path, name):
    trainer, _ = _tiny(monkeypatch, tmp_path, name=name)
    return trainer


def _radius(trainer) -> float:
    return float(trainer.cases[0].charge_radius)


def _predict_config(tmp_path: Path) -> Path:
    base_yaml = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    path = tmp_path / "predict_config.yaml"
    path.write_text(
        f"inherits: {base_yaml.as_posix()}\n"
        f"training:\n  output_dir: {(tmp_path / 'run').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def _state(trainer) -> dict:
    return {
        "step": int(trainer.step),
        "parameters": trainer.parameter_fingerprint(),
        "optimizer": repr(trainer.optimizer.state_dict()),
        "history": json.dumps(trainer.history, sort_keys=True, default=str),
        "sampler_rng": repr(trainer.sampler.rng_state()),
        "stage": (trainer._current_stage, float(trainer._stage_window), trainer._stage_blocked),
        "torch_rng": torch.get_rng_state().numpy().tobytes(),
        "numpy_rng": np.random.get_state()[1].tobytes(),
        "frozen_d": trainer._frozen_detonation_velocity(),
        "case_scales": trainer._instance_case_scales(),
    }


# ==========================================================================
# §2 复核点名的两种复现
# ==========================================================================


def test_the_radius_and_time_origin_forgery_keeps_the_implied_velocity(
    control, tmp_path
):
    """先把**修前为什么拦不住**钉住：半径与起始时间一起翻倍，反算爆速不变。

    这条断言是本轮的前提，不是被测行为：它说明"只比爆速"必然放行。
    """
    _, _, control_path = control
    _, payload = _variant(
        control_path, tmp_path, lambda p: _forge_radius_and_origin(p)
    )
    record = payload["scales"][payload["case_ids"][0]]
    assert float(record[SCALES_CHARGE_RADIUS_KEY]) == pytest.approx(0.10)
    implied = float(record[SCALES_CHARGE_RADIUS_KEY]) / (
        float(record[SCALES_START_TIME_KEY]) - float(0.10 / 0.05) * REFERENCE_START_TIME
    )
    assert implied == pytest.approx(DEFAULT_D, rel=1e-12), (
        "这个副本的爆速已经不再一致，它就测不到'只比爆速会放行'这件事"
    )


def test_the_radius_and_time_origin_forgery_is_refused(control, tmp_path):
    """核心反例：共享校验必须拒绝，并点名路径、算例与两个冲突值。"""
    _, _, control_path = control
    target, payload = _variant(
        control_path, tmp_path, lambda p: _forge_radius_and_origin(p)
    )
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    message = str(info.value)
    assert str(target) in message, message
    assert "frame-check" in message, message
    assert "0.1" in message and "0.05" in message, message


def test_both_entries_refuse_the_radius_and_time_origin_forgery(
    monkeypatch, tmp_path, control
):
    """训练恢复与预测都必须拒绝——**预测加载**当场拒绝，不等到覆盖计算。"""
    _, _, control_path = control
    target, _ = _variant(
        control_path, tmp_path, lambda p: _forge_radius_and_origin(p)
    )
    receiving = _fresh(monkeypatch, tmp_path, "forge_recv")
    with pytest.raises(TrainingError):
        receiving.load(target)
    with pytest.raises(SetupError):
        predict_module.load_model_from_checkpoint(
            target, charge_radius=_radius(receiving)
        )


def test_the_rename_forgery_is_refused_at_every_entry(monkeypatch, tmp_path, control):
    """第二种复现：``case_ids`` + ``scales`` 改名，其余记录仍是原名。"""
    _, _, control_path = control
    target, payload = _variant(control_path, tmp_path, lambda p: _rename_case(p))
    assert payload["case_ids"] == ["ghost"]
    assert list(payload["data_identity"]) == ["frame-check"]

    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    assert "ghost" in str(info.value)

    receiving = _fresh(monkeypatch, tmp_path, "ghost_recv")
    with pytest.raises(TrainingError):
        receiving.load(target)
    # 预测端在**加载**阶段就拒绝，不生成任何产物。
    with pytest.raises(SetupError):
        predict_module.load_model_from_checkpoint(
            target, charge_radius=_radius(receiving)
        )


def test_predict_cli_writes_nothing_for_a_forged_checkpoint(monkeypatch, tmp_path, control):
    """实际 ``predict.main()``：拒绝时退出前就抛错，输出目录里不留任何文件。"""
    _, _, control_path = control
    target, _ = _variant(
        control_path, tmp_path, lambda p: _forge_radius_and_origin(p)
    )
    receiving = _fresh(monkeypatch, tmp_path, "cli_recv")
    config = _predict_config(tmp_path)
    output = tmp_path / "prediction"
    output.mkdir()
    existing = output / "80mm.npz"
    sentinel = b"a previous committed prediction"
    existing.write_bytes(sentinel)

    with pytest.raises(SetupError):
        predict_module.main([
            "--config", str(config),
            "--checkpoint", str(target),
            "--radius", str(_radius(receiving)),
            "--output", str(existing),
            "--n-times", "5",
            "--n-radii", "20",
        ])
    assert existing.read_bytes() == sentinel, "被拒绝的预测覆盖了已有结果"
    assert sorted(p.name for p in output.iterdir()) == ["80mm.npz"], "被拒绝的预测写出了新文件"


# ==========================================================================
# §3.1 逐算例关联：每一项都单独钉一条
# ==========================================================================


@pytest.mark.parametrize(
    "mutate, needle",
    [
        pytest.param(
            lambda p: p["data_identity"][p["case_ids"][0]].__setitem__(
                DATA_IDENTITY_CHARGE_RADIUS_KEY, 0.10
            ),
            "charge_radius_m",
            id="scale-radius-vs-data-identity",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_CHARGE_RADII_KEY].__setitem__(0, 0.10),
            "charge_radii",
            id="provenance-parallel-radii",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_START_TIMES_KEY].__setitem__(
                p["case_ids"][0], 8.215007215007216e-06 * 3.0
            ),
            "start_times",
            id="provenance-synonymous-start-time",
        ),
        pytest.param(
            lambda p: p["scales"][p["case_ids"][0]].__setitem__(SCALES_TIME_SCALE_KEY, 2.2),
            SCALES_TIME_SCALE_KEY,
            id="scales-time-scale-vs-top-level",
        ),
        pytest.param(
            lambda p: p["data_identity"].pop(p["case_ids"][0]),
            "data_identity",
            id="data-identity-record-missing",
        ),
        pytest.param(
            lambda p: p["scales"].pop(p["case_ids"][0]),
            "scales",
            id="scales-record-missing",
        ),
        pytest.param(
            lambda p: p["data_identity"].__setitem__("ghost", dict(next(iter(p["data_identity"].values())))),
            "ghost",
            id="data-identity-record-without-a-case",
        ),
        pytest.param(
            lambda p: p["provenance"].__setitem__("case_ids", ["ghost"]),
            "ghost",
            id="provenance-case-list-disagrees",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_START_TIMES_KEY].pop(p["case_ids"][0]),
            "start_times",
            id="provenance-start-time-record-missing",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_VALID_TIME_KEY].pop(p["case_ids"][0]),
            PROVENANCE_VALID_TIME_KEY,
            id="valid-time-record-missing",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_VALID_TIME_KEY].__setitem__("ghost", 1.0e-4),
            "ghost",
            id="valid-time-record-without-a-case",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_VALID_TIME_KEY].__setitem__(
                p["case_ids"][0], float("nan")
            ),
            PROVENANCE_VALID_TIME_KEY,
            id="valid-time-not-finite",
        ),
        pytest.param(
            lambda p: p["provenance"][PROVENANCE_VALID_TIME_KEY].__setitem__(
                p["case_ids"][0], -1.0
            ),
            PROVENANCE_VALID_TIME_KEY,
            id="valid-time-negative",
        ),
        pytest.param(
            lambda p: p.__setitem__("case_ids", [p["case_ids"][0], p["case_ids"][0]]),
            "more than once",
            id="duplicate-case-ids",
        ),
    ],
)
def test_every_association_link_is_enforced(control, tmp_path, mutate, needle):
    """规则 3.1 的逐项：每断开一条关联都必须拒绝，且错误信息指向冲突字段。"""
    _, _, control_path = control
    target, payload = _variant(control_path, tmp_path, mutate)
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    assert needle in str(info.value), info.value


def test_the_untouched_control_passes_the_association_check(control):
    """正向控制：没被改动的检查点必须通过，否则上面每一条都可能是"总是拒绝"。"""
    _, _, control_path = control
    payload = torch.load(control_path, map_location="cpu", weights_only=False)
    resolved = require_case_scales_association(payload, control_path)
    assert set(resolved) == set(payload["case_ids"])
    for case_id, record in resolved.items():
        assert record[SCALES_CHARGE_RADIUS_KEY] == pytest.approx(
            float(payload["data_identity"][case_id][DATA_IDENTITY_CHARGE_RADIUS_KEY])
        )
        assert record[SCALES_START_TIME_KEY] == pytest.approx(
            _expected_time_origin(record[SCALES_CHARGE_RADIUS_KEY], DEFAULT_D), rel=0, abs=1e-18
        )
    assert require_detonation_velocity(payload, control_path) == pytest.approx(DEFAULT_D)


# ==========================================================================
# §4 多算例：只坏一个也必须拒绝；两个都好必须通过
# ==========================================================================


def test_only_one_broken_case_among_several_is_refused(control, tmp_path):
    """**不得只检查第一项。** 第一个算例完全正常，坏的是第二个。"""
    _, _, control_path = control

    def mutate(payload):
        case_id = _add_consistent_second_case(payload)
        payload["data_identity"][case_id][DATA_IDENTITY_CHARGE_RADIUS_KEY] = 0.10
        return case_id

    target, payload = _variant(control_path, tmp_path, mutate)
    assert payload["case_ids"] == ["frame-check", "second-case"]
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    message = str(info.value)
    # 报的是**第二个**算例；第一个算例的名字不该出现在冲突描述里。
    assert "second-case" in message, message
    assert "frame-check" not in message, message


def test_two_consistent_cases_pass(control, tmp_path):
    """反面：两个算例四份记录全都一致时必须通过。"""
    _, _, control_path = control
    target, payload = _variant(
        control_path, tmp_path, lambda p: _add_consistent_second_case(p)
    )
    assert len(payload["case_ids"]) == 2
    resolved = require_case_scales_association(payload, target)
    assert len(resolved) == 2
    assert require_detonation_velocity(payload, target) == pytest.approx(DEFAULT_D)


# ==========================================================================
# §3.2 恢复端：与接收实例逐算例核对
# ==========================================================================


def test_a_refused_resume_leaves_the_receiving_instance_and_files_untouched(
    monkeypatch, tmp_path, control
):
    """规则 3.2/4：拒绝发生在改动模型、优化器、阶段状态、历史、RNG 与文件之前。"""
    _, _, control_path = control
    target, _ = _variant(
        control_path, tmp_path, lambda p: _forge_radius_and_origin(p)
    )
    before_bytes = target.read_bytes()

    receiving = _fresh(monkeypatch, tmp_path, "untouched_recv")
    before_state = _state(receiving)
    with pytest.raises(TrainingError):
        receiving.load(target)
    assert _state(receiving) == before_state, "拒绝之前就把接收实例改掉了"
    assert target.read_bytes() == before_bytes, "校验过程改写了检查点文件"
    # 也不许"为了让检查点通过"去改接收实例。
    assert receiving._instance_case_scales()[receiving.cases[0].case_id][1] == pytest.approx(
        _expected_time_origin(_radius(receiving), DEFAULT_D), rel=0, abs=1e-18
    )


def test_the_same_instance_resumes_its_own_control(monkeypatch, tmp_path, control):
    """反面：同一套算例/尺度的实例读这份检查点必须照常恢复。"""
    _, _, control_path = control
    receiving = _fresh(monkeypatch, tmp_path, "own_recv")
    receiving.load(control_path)
    assert receiving._instance_case_scales() == _fresh(
        monkeypatch, tmp_path, "own_recv2"
    )._instance_case_scales()


# ==========================================================================
# §4 正向控制：合法产物照旧，时间轴与覆盖范围符合独立计算
# ==========================================================================


@pytest.mark.parametrize("artifact", ["last.pt", "best.pt"])
def test_legal_artifacts_load_and_predict_consistently(
    monkeypatch, tmp_path, control, artifact
):
    """合法的 ``last.pt`` 与 ``best.pt`` 都能恢复、能预测，且两端时间原点一致。"""
    tmp, trainer, _ = control
    checkpoint = Path(trainer.output_dir) / artifact
    assert checkpoint.exists(), f"run() 没有产出 {artifact}"
    radius = _radius(trainer)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    receiving = _fresh(monkeypatch, tmp_path, f"legal_{artifact.split('.')[0]}")
    receiving.load(checkpoint)
    resume_origin = receiving.scales_by_radius[radius].start_time()

    model, _ = predict_module.load_model_from_checkpoint(checkpoint, charge_radius=radius)
    assert resume_origin == pytest.approx(model.scales.start_time(), rel=0, abs=1e-18)
    assert resume_origin == pytest.approx(
        _expected_time_origin(radius, float(payload[DETONATION_VELOCITY_FIELD])),
        rel=0,
        abs=1e-18,
    )


def test_predict_cli_is_legal_at_the_training_radius_and_at_another_one(
    monkeypatch, tmp_path, control
):
    """跨半径预测仍然合法：目标半径只影响还原，不影响覆盖判断。"""
    _, trainer, checkpoint = control
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_radius = _radius(trainer)
    config = _predict_config(tmp_path)

    for target in (training_radius, 0.08):
        output = tmp_path / f"legal_{int(round(target * 1000))}"
        assert predict_module.main([
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--radius", str(target),
            "--output", str(output / "out.npz"),
            "--n-times", "5",
            "--n-radii", "20",
        ]) == 0
        with np.load(next(output.glob("*.npz")), allow_pickle=True) as handle:
            result = {key: handle[key] for key in handle.files}
        assert float(result["charge_radius"]) == pytest.approx(target)
        # 时间轴与覆盖范围都按**独立**公式复核。
        assert float(result["start_time"]) == pytest.approx(
            _expected_time_origin(target, DEFAULT_D), rel=0, abs=1e-18
        )
        assert float(result["scaled_time_coverage"]) == pytest.approx(
            _independent_coverage(payload), rel=1e-12
        )
        assert float(result["physical_time"][-1]) == pytest.approx(
            _independent_end_time(payload, target), rel=1e-12
        )
