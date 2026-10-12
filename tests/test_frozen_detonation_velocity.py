"""冻结爆速 ``D`` 在**保存 / 恢复训练 / 预测**三端的一致性（第 2 轮）。

第 2 轮点名的漏洞：``detonation_velocity`` 这个冻结参数只有**预测端**在读。

* 顶层 ``detonation_velocity`` 只有 ``cli/predict.py`` 校验，``Trainer.load()``
  **一次都没读过它** —— 删掉它、或写成 ``-1``，恢复训练照常放行；
* 顶层值与逐算例 ``scales[case_id]`` 记录的冻结值之间**没有任何核对**；
* 恢复端也没有把检查点里的 ``D`` 与**接收实例实际使用的尺度**比过。

于是同一份检查点被解释成两条时间轴。实测（50 mm 算例，顶层改成 1234、
``scales`` 记录仍是 6930 那条）：

* 恢复训练取接收实例的尺度 → ``T_h = 8.21500721501e-06 s``；
* 预测取顶层的 1234 → ``T_h = 4.15186385737e-05 s``。

**为什么 ``scales`` 记录里没有 ``detonation_velocity`` 键**：``SimilitudeScales``
的 ``as_dict()`` 不写 ``D``，它写 ``start_time = T_h = R_0/D + epsilon(R_0)``。
``D`` 由 ``(charge_radius, start_time)`` **唯一确定**
（``D = R_0 / (T_h - epsilon(R_0))``），所以"逐算例记录里的对应冻结值"就是这一对。
本文件按这个关系核对，而不是新增字段、也不升级检查点版本。

坏输入都是**主动构造**的（真实检查点上改指定字段，并先断言变异生效）。这里
**不**声称正常保存器会生成这种文件。
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
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.physics.detonation import DetonationConfig
from spherical_blast_pinn.physics.precursor import PrecursorConfig, solve_reference_field
from spherical_blast_pinn.physics.similitude import (
    REFERENCE_RADIUS,
    REFERENCE_START_TIME,
    SimilitudeScales,
)
from spherical_blast_pinn.training.checkpoint import (
    DETONATION_VELOCITY_FIELD,
    require_detonation_velocity,
)
from spherical_blast_pinn.training.trainer import TrainingError

import test_review_regressions as base
from test_best_candidate_and_transaction import _tiny

#: 常用默认卡片爆速（材料卡值）。
DEFAULT_D = 6930.0
#: **测试专用**的另一个爆速：与默认值不同，但在这套夹具里从构造阶段就自洽。
#: 它只用来证明读取端没有把 6930 写死，**不**代表物理模型有效。
ALTERNATE_D = 7000.0
TIME_SCALE = 2.0


# --------------------------------------------------------------------------
# 独立的时间原点公式：T_h = R_0/D + (R_0/R_ref) epsilon_ref
# --------------------------------------------------------------------------


def _expected_time_origin(radius: float, velocity: float) -> float:
    """按 AGENTS.md 的公式**独立**算一遍，不调用被测的 ``start_time()``。"""
    return radius / velocity + (radius / REFERENCE_RADIUS) * REFERENCE_START_TIME


def test_the_independent_formula_reproduces_the_documented_reference_values():
    """先锚住这个独立公式本身：AGENTS.md 给的 50/80/100/500 mm 的 ``T_h``。"""
    documented = {
        0.05: 8.215007e-6,
        0.08: 13.144012e-6,
        0.10: 16.430014e-6,
        0.50: 82.150072e-6,
    }
    for radius, expected in documented.items():
        assert _expected_time_origin(radius, DEFAULT_D) == pytest.approx(
            expected, rel=2.0e-7
        ), radius


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def _trainer_with_velocity(monkeypatch, tmp_path, velocity: float, name: str):
    """**从构造阶段**就使用 ``velocity`` 的 Trainer。

    算例的输入元数据（``metadata.json`` 的 ``detonation_velocity_m_s``）、参考场、
    尺度集合、网络与采样器全部用同一个 ``D`` 建立。这是"构造阶段一致"，不是
    "先按默认值正常保存、再把字段改成看起来一致"——后者正是本轮要禁止的取证方式。
    """
    real_write_case = base._write_case

    def write_case(path):
        case = real_write_case(path)
        metadata_path = Path(path) / "case" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["detonation_velocity_m_s"] = velocity
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        rebuilt = load_extracted_case(Path(path) / "case", case_id="frame-check")
        # ``_write_case`` 会把前几帧标成无效，重建之后要照做。
        rebuilt.contact_valid[:4] = False
        rebuilt.shock_valid[:4] = False
        assert float(rebuilt.detonation_velocity) == pytest.approx(velocity)
        return rebuilt

    def scales(radius):
        return SimilitudeScales.for_radius(
            radius, time_scale=TIME_SCALE, detonation_velocity=velocity
        )

    def reference(radius=0.05):
        return solve_reference_field(
            charge_radius=radius,
            config=PrecursorConfig(cells_inside=120, growth=1.05, cfl=0.3),
            detonation_velocity=velocity,
            detonation_config=DetonationConfig(cells=120),
            sample_times=[0.0, 1.0e-6, 4.0e-6],
        )

    monkeypatch.setattr(base, "_write_case", write_case)
    monkeypatch.setattr(base, "_scales", scales)
    monkeypatch.setattr(base, "_tiny_reference", reference)
    trainer, _ = _tiny(monkeypatch, tmp_path, name=name)
    return trainer


def _trained(monkeypatch, tmp_path, name, *, velocity=DEFAULT_D):
    trainer = _trainer_with_velocity(monkeypatch, tmp_path, velocity, name)
    trainer.run(max_updates=2, verbose=False)
    return trainer, trainer.output_dir / "last.pt"


def _fresh(monkeypatch, tmp_path, name, *, velocity=DEFAULT_D):
    """另一个实例：它读到的就是盘上那一份文件。"""
    return _trainer_with_velocity(monkeypatch, tmp_path, velocity, name)


def _radius(trainer) -> float:
    return float(trainer.cases[0].charge_radius)


def _fingerprint(payload) -> bytes:
    """整份载荷的内容指纹，用来断言"变异确实生效"。"""
    import hashlib
    import pickle

    return hashlib.sha256(pickle.dumps(payload)).digest()


_MUTATION_COUNTER = [0]


def _mutated(control: Path, tmp_path: Path, mutate):
    """把控制检查点复制一份，改指定字段，**并断言变异确实生效**。"""
    _MUTATION_COUNTER[0] += 1
    target = tmp_path / f"mutated_{_MUTATION_COUNTER[0]}.pt"
    shutil.copy2(control, target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    before = _fingerprint(payload)
    value = mutate(payload)
    assert _fingerprint(payload) != before, "变异没有生效：载荷对象没有被改动"
    torch.save(payload, target)
    after = torch.load(target, map_location="cpu", weights_only=False)
    assert _fingerprint(after) != before, "变异没有生效：文件内容与原来一致"
    return target, value


def _predict_config(tmp_path: Path) -> Path:
    base_yaml = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    path = tmp_path / "predict_config.yaml"
    path.write_text(
        f"inherits: {base_yaml.as_posix()}\n"
        f"training:\n  output_dir: {(tmp_path / 'run').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def _outcomes(monkeypatch, tmp_path, trainer, checkpoint, slug, *, receiving_d=DEFAULT_D):
    """两个入口的结论：异常对象，或 ``"accepted"``（预测还会给出模型）。"""
    receiving = _fresh(monkeypatch, tmp_path, f"{slug}_recv", velocity=receiving_d)
    try:
        receiving.load(checkpoint)
        resume = "accepted"
    except TrainingError as exc:
        resume = exc
    try:
        model, _ = predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )
        predict = ("accepted", model)
    except SetupError as exc:
        predict = exc
    return receiving, resume, predict


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
    }


# ==========================================================================
# A. 字段合法性
# ==========================================================================

BAD_VALUES = [
    pytest.param(0.0, id="zero"),
    pytest.param(-1.0, id="negative"),
    pytest.param(-6930.0, id="negative-default"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="plus-inf"),
    pytest.param(float("-inf"), id="minus-inf"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("6930", id="numeric-string"),
    pytest.param("fast", id="plain-string"),
    pytest.param("", id="empty-string"),
    pytest.param([6930.0], id="list"),
    pytest.param((6930.0,), id="tuple"),
    pytest.param({"value": 6930.0}, id="dict"),
    pytest.param(None, id="null"),
    pytest.param(1 + 2j, id="complex"),
]


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_the_shared_validator_rejects_an_illegal_frozen_velocity(
    monkeypatch, tmp_path, bad
):
    """规则 1、2：必须是合法的有限正**实数**；布尔、字符串、列表等一律拒绝。

    修前 ``float(raw)`` 把 ``True`` 静默变成 ``1.0``、把 ``"6930"`` 变成 ``6930.0``
    ——错误类型被转成了合法输入。
    """
    _, control = _trained(monkeypatch, tmp_path, "bad_value")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.__setitem__(DETONATION_VELOCITY_FIELD, bad)
    )
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload[DETONATION_VELOCITY_FIELD] == bad or (
        isinstance(bad, float) and np.isnan(bad)
    )
    with pytest.raises(TrainingError, match=DETONATION_VELOCITY_FIELD):
        require_detonation_velocity(payload, target)


def test_the_shared_validator_rejects_a_missing_frozen_velocity(monkeypatch, tmp_path):
    """规则 1、8：缺失必须报"缺字段"，而且不能被说成"旧物理方案的文件"。

    版本 8 起这个字段是**必须**的，所以在一份版本 8 的检查点里缺它只能是损坏。
    """
    _, control = _trained(monkeypatch, tmp_path, "missing")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.pop(DETONATION_VELOCITY_FIELD)
    )
    payload = torch.load(target, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    message = str(info.value)
    assert DETONATION_VELOCITY_FIELD in message
    assert "scheme-A" not in message and "scheme A" not in message


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(None, id="null"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="bool"),
        pytest.param("6930", id="string"),
        pytest.param([6930.0], id="list"),
    ],
)
def test_both_readers_refuse_an_illegal_frozen_velocity(monkeypatch, tmp_path, bad):
    """规则 1、2：非法顶层值，**恢复训练与预测都必须拒绝**。"""
    slug = "entry_bad"
    trainer, control = _trained(monkeypatch, tmp_path, slug)
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.__setitem__(DETONATION_VELOCITY_FIELD, bad)
    )
    _, resume, predict = _outcomes(monkeypatch, tmp_path, trainer, target, slug)
    assert isinstance(resume, TrainingError), f"恢复训练放行了非法 D：{bad!r}"
    assert isinstance(predict, SetupError), f"预测放行了非法 D：{bad!r}"
    for outcome in (resume, predict):
        assert DETONATION_VELOCITY_FIELD in str(outcome), outcome


def test_both_readers_refuse_a_missing_frozen_velocity(monkeypatch, tmp_path):
    """**复核的原反例 (1)。** 删掉顶层字段：修前恢复训练放行、预测拒绝。"""
    trainer, control = _trained(monkeypatch, tmp_path, "entry_missing")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.pop(DETONATION_VELOCITY_FIELD)
    )
    _, resume, predict = _outcomes(monkeypatch, tmp_path, trainer, target, "entry_missing")
    assert isinstance(resume, TrainingError), "删掉冻结爆速之后恢复训练仍然放行"
    assert isinstance(predict, SetupError)
    assert DETONATION_VELOCITY_FIELD in str(resume)


def test_both_readers_refuse_a_negative_frozen_velocity(monkeypatch, tmp_path):
    """**复核的原反例 (2)。** 顶层 ``D = -1``：修前恢复训练放行、预测拒绝。"""
    trainer, control = _trained(monkeypatch, tmp_path, "entry_negative")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.__setitem__(DETONATION_VELOCITY_FIELD, -1.0)
    )
    _, resume, predict = _outcomes(monkeypatch, tmp_path, trainer, target, "entry_negative")
    assert isinstance(resume, TrainingError), "D = -1 时恢复训练仍然放行"
    assert isinstance(predict, SetupError)


# ==========================================================================
# B. 文件内部一致性：顶层值 vs 逐算例 scales 记录
# ==========================================================================


def test_the_control_checkpoint_is_internally_consistent(monkeypatch, tmp_path):
    """正向控制：未经改动的检查点本身满足新契约（否则下面所有反例都没意义）。"""
    trainer, control = _trained(monkeypatch, tmp_path, "consistent")
    payload = torch.load(control, map_location="cpu", weights_only=False)
    assert require_detonation_velocity(payload, control) == pytest.approx(DEFAULT_D)
    for case_id, record in payload["scales"].items():
        radius = float(record["charge_radius"])
        assert float(record["start_time"]) == pytest.approx(
            _expected_time_origin(radius, DEFAULT_D), rel=0, abs=1e-18
        ), case_id


def test_both_readers_refuse_a_top_level_value_that_contradicts_the_scales_record(
    monkeypatch, tmp_path
):
    """**复核的原反例 (3)。** 顶层改成 1234、``scales`` 记录仍是 6930 那条。

    修前两个入口都放行，而它们各自重建出**不同**的时间原点：
    恢复 8.21500721501 µs、预测 41.5186385737 µs。
    """
    trainer, control = _trained(monkeypatch, tmp_path, "entry_bogus")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.__setitem__(DETONATION_VELOCITY_FIELD, 1234.0)
    )
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload[DETONATION_VELOCITY_FIELD] == 1234.0
    # ``scales`` 记录**没有**被动过：它是 6930 那条。
    record = next(iter(payload["scales"].values()))
    assert float(record["start_time"]) == pytest.approx(
        _expected_time_origin(float(record["charge_radius"]), DEFAULT_D), rel=0, abs=1e-18
    )

    _, resume, predict = _outcomes(monkeypatch, tmp_path, trainer, target, "entry_bogus")
    assert isinstance(resume, TrainingError), "顶层与 scales 矛盾时恢复训练仍然放行"
    assert isinstance(predict, SetupError), "顶层与 scales 矛盾时预测仍然放行"
    for outcome in (resume, predict):
        assert "1234" in str(outcome) and str(payload["case_ids"][0]) in str(outcome)


def test_a_corrupted_per_case_start_time_is_refused(monkeypatch, tmp_path):
    """某个算例记录里的冻结值被改坏：顶层合法也必须拒绝。"""
    trainer, control = _trained(monkeypatch, tmp_path, "corrupt_scales")

    def corrupt(payload):
        case_id = payload["case_ids"][0]
        payload["scales"][case_id] = dict(payload["scales"][case_id])
        payload["scales"][case_id]["start_time"] = (
            float(payload["scales"][case_id]["start_time"]) * 1.01
        )
        return case_id

    target, case_id = _mutated(control, tmp_path, corrupt)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert float(payload["scales"][case_id]["start_time"]) == pytest.approx(
        _expected_time_origin(_radius(trainer), DEFAULT_D) * 1.01
    )

    _, resume, predict = _outcomes(monkeypatch, tmp_path, trainer, target, "corrupt_scales")
    assert isinstance(resume, TrainingError)
    assert isinstance(predict, SetupError)
    for outcome in (resume, predict):
        assert case_id in str(outcome), outcome


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda r: r.pop("start_time"), id="no-start-time"),
        pytest.param(lambda r: r.__setitem__("start_time", None), id="null-start-time"),
        pytest.param(lambda r: r.__setitem__("start_time", "8.2e-6"), id="string-start-time"),
        pytest.param(lambda r: r.__setitem__("start_time", float("nan")), id="nan-start-time"),
        pytest.param(lambda r: r.pop("charge_radius"), id="no-charge-radius"),
        pytest.param(lambda r: r.__setitem__("charge_radius", -0.05), id="negative-radius"),
        pytest.param(lambda r: r.__setitem__("charge_radius", 0.0), id="zero-radius"),
    ],
)
def test_a_per_case_record_without_a_usable_frozen_value_is_refused(
    monkeypatch, tmp_path, damage
):
    """规则 3：算例记录缺 ``start_time`` / ``charge_radius``，或值不可用 → 拒绝。"""
    _, control = _trained(monkeypatch, tmp_path, "record_damage")

    def corrupt(payload):
        case_id = payload["case_ids"][0]
        payload["scales"][case_id] = dict(payload["scales"][case_id])
        damage(payload["scales"][case_id])
        return case_id

    target, case_id = _mutated(control, tmp_path, corrupt)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    assert case_id in str(info.value)


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda p: p.pop("scales"), id="no-scales-block"),
        pytest.param(lambda p: p.__setitem__("scales", {"nope": {}}), id="missing-case-record"),
        pytest.param(lambda p: p.pop("case_ids"), id="no-case-ids"),
        pytest.param(lambda p: p.__setitem__("case_ids", []), id="empty-case-ids"),
        pytest.param(lambda p: p.__setitem__("case_ids", "frame-check"), id="string-case-ids"),
    ],
)
def test_a_missing_per_case_record_is_refused_not_skipped(
    monkeypatch, tmp_path, damage
):
    """规则 3：缺记录、缺 ``case_ids``、或 ``scales`` 不是对象 → 拒绝，**不跳过**。"""
    _, control = _trained(monkeypatch, tmp_path, "block_damage")
    target, _ = _mutated(control, tmp_path, damage)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError):
        require_detonation_velocity(payload, target)


def test_a_scales_record_for_a_case_the_checkpoint_does_not_claim_is_refused(
    monkeypatch, tmp_path
):
    """规则 3：``scales`` 里多出来的算例记录同样是"相互矛盾"。"""
    _, control = _trained(monkeypatch, tmp_path, "extra_case")

    def add_extra(payload):
        payload["scales"]["ghost-case"] = dict(next(iter(payload["scales"].values())))
        return "ghost-case"

    target, _ = _mutated(control, tmp_path, add_extra)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    assert "ghost-case" in str(info.value)


def test_a_second_case_with_a_different_frozen_value_is_refused(monkeypatch, tmp_path):
    """规则 3：**不得无条件取第一个算例**。多算例里只有一个对不上也要拒绝。"""
    _, control = _trained(monkeypatch, tmp_path, "second_case_bad")

    def add_second(payload):
        first_id = payload["case_ids"][0]
        payload["case_ids"] = [first_id, "second-case"]
        record = dict(payload["scales"][first_id])
        record["charge_radius"] = 0.08
        # 0.08 上用 7000 算出来的时间原点，与顶层的 6930 不符。
        record["start_time"] = _expected_time_origin(0.08, ALTERNATE_D)
        payload["scales"]["second-case"] = record
        return "second-case"

    target, case_id = _mutated(control, tmp_path, add_second)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["scales"][case_id]["start_time"] == pytest.approx(
        _expected_time_origin(0.08, ALTERNATE_D)
    )
    with pytest.raises(TrainingError) as info:
        require_detonation_velocity(payload, target)
    assert case_id in str(info.value)


def test_a_second_case_that_agrees_is_accepted(monkeypatch, tmp_path):
    """反面：两个算例都与顶层一致时不能被误伤（否则"检查每个算例"就退化成"总是拒绝"）。"""
    trainer, control = _trained(monkeypatch, tmp_path, "second_case_ok")

    def add_second(payload):
        first_id = payload["case_ids"][0]
        payload["case_ids"] = [first_id, "second-case"]
        record = dict(payload["scales"][first_id])
        record["charge_radius"] = 0.08
        record["start_time"] = _expected_time_origin(0.08, DEFAULT_D)
        payload["scales"]["second-case"] = record

    target, _ = _mutated(control, tmp_path, add_second)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert require_detonation_velocity(payload, target) == pytest.approx(DEFAULT_D)


# ==========================================================================
# C. 恢复实例的一致性
# ==========================================================================


def test_a_self_consistent_checkpoint_cannot_be_resumed_by_a_different_scale_instance(
    monkeypatch, tmp_path
):
    """规则 4：文件内部自洽**不够**——它还必须与接收实例实际使用的尺度一致。

    一份用 7000 写的、内部完全一致的检查点，交给一个按 6930 构造的实例恢复：
    修前放行，恢复出来的是一条 8.215 µs 起算的轨迹，而文件描述的是 8.1429 µs。
    """
    alternate, control = _trained(monkeypatch, tmp_path, "alt_d", velocity=ALTERNATE_D)
    payload = torch.load(control, map_location="cpu", weights_only=False)
    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(ALTERNATE_D)
    # 它自己是内部一致的：失败不可能来自"文件矛盾"。
    require_detonation_velocity(payload, control)

    receiving = _fresh(monkeypatch, tmp_path, "alt_d_recv", velocity=DEFAULT_D)
    before = _state(receiving)
    with pytest.raises(TrainingError) as info:
        receiving.load(control)
    message = str(info.value)
    assert "7000" in message and "6930" in message, message

    # 拒绝发生在改动任何状态之前，而且**不**为了让检查点通过而改自己。
    assert _state(receiving) == before, "拒绝之前就改了接收实例"
    assert receiving._frozen_detonation_velocity() == pytest.approx(DEFAULT_D)


def test_a_refused_resume_leaves_model_optimizer_rng_and_stage_untouched(
    monkeypatch, tmp_path
):
    """规则 7：拒绝必须发生在模型、优化器、RNG、历史、阶段状态被改之前。"""
    trainer, control = _trained(monkeypatch, tmp_path, "untouched")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.__setitem__(DETONATION_VELOCITY_FIELD, 1234.0)
    )
    receiving = _fresh(monkeypatch, tmp_path, "untouched_recv")
    before = _state(receiving)
    with pytest.raises(TrainingError):
        receiving.load(target)
    assert _state(receiving) == before


def test_the_same_instance_still_resumes_its_own_checkpoint(monkeypatch, tmp_path):
    """反面：同尺度的实例读同一份检查点必须照常恢复。"""
    trainer, control = _trained(monkeypatch, tmp_path, "same_scale")
    receiving = _fresh(monkeypatch, tmp_path, "same_scale_recv")
    receiving.load(control)
    assert receiving.step == trainer.step
    assert receiving._frozen_detonation_velocity() == pytest.approx(DEFAULT_D)


# ==========================================================================
# D. 正向控制：合法检查点照旧，且两端时间原点一致
# ==========================================================================


@pytest.mark.parametrize("artifact", ["last.pt", "best.pt"])
def test_a_legal_checkpoint_resumes_and_predicts_with_one_shared_time_origin(
    monkeypatch, tmp_path, artifact
):
    """规则 5：同一个检查点在**恢复**与**预测**里的冻结参数与时间原点必须一致，
    并且都等于独立算出的 ``T_h = R_0/D + epsilon``。``last.pt`` 与 ``best.pt``
    都是产品支持的产物，两个都查。"""
    trainer, last = _trained(monkeypatch, tmp_path, f"origin_{artifact.split('.')[0]}")
    checkpoint = trainer.output_dir / artifact
    assert checkpoint.exists(), f"run() 没有产出 {artifact}"
    radius = _radius(trainer)
    expected = _expected_time_origin(radius, DEFAULT_D)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(DEFAULT_D)

    receiving = _fresh(monkeypatch, tmp_path, f"origin_{artifact}_recv")
    receiving.load(checkpoint)
    resume_origin = receiving.scales_by_radius[radius].start_time()

    model, _ = predict_module.load_model_from_checkpoint(checkpoint, charge_radius=radius)
    predict_origin = model.scales.start_time()

    assert resume_origin == pytest.approx(expected, rel=0, abs=1e-18)
    assert predict_origin == pytest.approx(expected, rel=0, abs=1e-18)
    assert resume_origin == predict_origin, "同一个检查点在两个入口里被解释成两条时间轴"


def test_predict_main_writes_a_result_at_the_training_radius_and_another_one(
    monkeypatch, tmp_path
):
    """真实 ``predict.main(argv)``：训练半径与另一个目标半径都产出文件，时间原点
    都由检查点里的冻结 ``D`` 重建（不是当前配置、不是模块默认值）。"""
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "predict_main")
    training_radius = _radius(trainer)
    config = _predict_config(tmp_path)

    for target_radius in (training_radius, 0.08):
        output = tmp_path / f"prediction_{int(round(target_radius * 1000))}"
        code = predict_module.main([
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--radius", str(target_radius),
            "--output", str(output / "out.npz"),
            "--n-times", "5",
            "--n-radii", "20",
        ])
        assert code == 0
        written = list(output.glob("*.npz"))
        assert written, f"半径 {target_radius} 上没有产出预测文件"
        with np.load(written[0], allow_pickle=True) as handle:
            result = {key: handle[key] for key in handle.files}
        assert float(result["charge_radius"]) == pytest.approx(target_radius)
        assert float(result["start_time"]) == pytest.approx(
            _expected_time_origin(target_radius, DEFAULT_D), rel=0, abs=1e-18
        )


def test_predict_main_refuses_and_writes_nothing_on_a_broken_checkpoint(
    monkeypatch, tmp_path
):
    """规则 7：预测校验失败不产生新结果、不覆盖已有结果。"""
    trainer, control = _trained(monkeypatch, tmp_path, "predict_refuse")
    target, _ = _mutated(
        control, tmp_path, lambda payload: payload.pop(DETONATION_VELOCITY_FIELD)
    )
    config = _predict_config(tmp_path)
    output_dir = tmp_path / "prediction"
    output_dir.mkdir()
    existing = output_dir / "80mm.npz"
    sentinel = b"a previous committed prediction"
    existing.write_bytes(sentinel)

    with pytest.raises(SetupError):
        predict_module.main([
            "--config", str(config),
            "--checkpoint", str(target),
            "--radius", str(0.08),
            "--output", str(existing),
            "--n-times", "5",
            "--n-radii", "20",
        ])
    assert existing.read_bytes() == sentinel, "被拒绝的预测覆盖了已有结果"
    assert sorted(p.name for p in output_dir.iterdir()) == ["80mm.npz"]


# ==========================================================================
# 测试专用的另一个爆速：证明读取端没有把 6930 写死
# ==========================================================================


def test_a_legally_different_detonation_velocity_is_not_replaced_by_the_default(
    monkeypatch, tmp_path
):
    """整条链路用一个**非默认**的 ``D`` 建立：算例元数据、参考场、尺度集合一致。

    读取端必须用它、而不是 6930。这个夹具只证明"没有硬编码"，
    **不**用来宣称物理模型有效。
    """
    trainer, checkpoint = _trained(monkeypatch, tmp_path, "alt_chain", velocity=ALTERNATE_D)
    radius = _radius(trainer)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(ALTERNATE_D)
    assert require_detonation_velocity(payload, checkpoint) == pytest.approx(ALTERNATE_D)
    # 逐算例记录也是按 7000 冻结的，不是 6930 那条。
    record = next(iter(payload["scales"].values()))
    assert float(record["start_time"]) == pytest.approx(
        _expected_time_origin(radius, ALTERNATE_D), rel=0, abs=1e-18
    )
    assert float(record["start_time"]) != pytest.approx(
        _expected_time_origin(radius, DEFAULT_D)
    )

    receiving = _fresh(monkeypatch, tmp_path, "alt_chain_recv", velocity=ALTERNATE_D)
    receiving.load(checkpoint)
    assert receiving.scales_by_radius[radius].start_time() == pytest.approx(
        _expected_time_origin(radius, ALTERNATE_D), rel=0, abs=1e-18
    )

    config = _predict_config(tmp_path)
    for target_radius in (radius, 0.08):
        output = tmp_path / f"alt_prediction_{int(round(target_radius * 1000))}"
        assert predict_module.main([
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--radius", str(target_radius),
            "--output", str(output / "out.npz"),
            "--n-times", "5",
            "--n-radii", "20",
        ]) == 0
        with np.load(next(output.glob("*.npz")), allow_pickle=True) as handle:
            result = {key: handle[key] for key in handle.files}
        assert float(result["start_time"]) == pytest.approx(
            _expected_time_origin(target_radius, ALTERNATE_D), rel=0, abs=1e-18
        ), f"半径 {target_radius} 上没有用检查点里的 7000"


def test_the_alternate_fixture_is_the_only_thing_that_differs(monkeypatch, tmp_path):
    """夹具自证：除了 ``D`` 之外，交替夹具与默认夹具在结构上必须一模一样。"""
    default_trainer, default_ckpt = _trained(monkeypatch, tmp_path, "shape_default")
    alt_trainer, alt_ckpt = _trained(
        monkeypatch, tmp_path, "shape_alt", velocity=ALTERNATE_D
    )
    default_payload = torch.load(default_ckpt, map_location="cpu", weights_only=False)
    alt_payload = torch.load(alt_ckpt, map_location="cpu", weights_only=False)
    assert sorted(default_payload) == sorted(alt_payload)
    assert sorted(next(iter(default_payload["scales"].values()))) == sorted(
        next(iter(alt_payload["scales"].values()))
    )
    assert float(alt_payload[DETONATION_VELOCITY_FIELD]) != float(
        default_payload[DETONATION_VELOCITY_FIELD]
    )
