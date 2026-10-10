"""§16 验证隔离的定向回归。

在这之前，"500 mm 只做最终独立验证"只靠配置里的一个字符串
（``split: extrapolation_test``）和一句 ``splits.strict_blind_test: true``——而后者
**没有任何代码读它**。把 500 mm 那一行的 ``split`` 改成 ``train``，什么都不会
报错，所有报表仍然写着"盲测半径已隔离"。

这些测试逐个钉住四种破坏方式。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from spherical_blast_pinn.config import Config, load_config
from spherical_blast_pinn.data.isolation import (
    DEFAULT_FINAL_VALIDATION_RADII_MM,
    IsolationError,
    evaluate_validation_isolation,
    final_validation_radii_m,
)


def _config(cases: list[dict], *, strict_blind: bool = True, reference_m: float = 0.05) -> Config:
    return Config.from_mapping({
        "reference_field": {"charge_radius_m": reference_m},
        "data": {"cases": cases},
        "splits": {"strict_blind_test": strict_blind},
    })


def _case(case_id: str, radius_m: float, split: str) -> dict:
    return {"case_id": case_id, "charge_radius_m": radius_m,
            "directory": f"/nonexistent/{case_id}", "split": split}


def _stub_case(case_id: str, radius_m: float):
    """``evaluate_validation_isolation`` 只用到这两个属性。"""
    return SimpleNamespace(case_id=case_id, charge_radius=radius_m)


def test_the_protected_radius_is_500mm_by_default():
    assert DEFAULT_FINAL_VALIDATION_RADII_MM == (500.0,)
    config = _config([_case("tnt_500mm", 0.5, "extrapolation_test")])
    assert final_validation_radii_m(config) == (0.5,)


def test_a_protected_radius_declared_as_training_breaks_isolation():
    """把 500 mm 的 split 改成 train 必须被拦住。"""
    config = _config([
        _case("tnt_50mm", 0.05, "train"),
        _case("tnt_500mm", 0.5, "train"),          # 破坏了隔离
    ])
    verdict = evaluate_validation_isolation(
        config, [_stub_case("tnt_50mm", 0.05)]
    )
    assert not verdict.ok
    assert any("tnt_500mm" in reason for reason in verdict.reasons)
    assert any("final-validation radius" in reason for reason in verdict.reasons)
    with pytest.raises(IsolationError, match="section 16"):
        verdict.raise_if_broken()


def test_a_training_case_at_a_protected_radius_breaks_isolation_even_if_the_split_says_otherwise():
    """第二层检查**不看** ``split`` 字符串，只看半径。

    过滤器写错时（比如 split 拼成了 ``Train``、或者以后有人加了新的过滤逻辑），
    只看字符串就一起错了。半径是独立的一份信息。
    """
    config = _config([_case("weird", 0.5, "validation")])
    verdict = evaluate_validation_isolation(config, [_stub_case("sneaky", 0.5)])
    assert not verdict.ok
    assert any("sneaky" in reason for reason in verdict.reasons)


def test_calibrating_the_initial_field_on_the_protected_radius_breaks_isolation():
    """初值标定也是一次"用过了"（§16 明列）。"""
    config = _config([_case("tnt_50mm", 0.05, "train"),
                      _case("tnt_500mm", 0.5, "extrapolation_test")])
    verdict = evaluate_validation_isolation(
        config, [_stub_case("tnt_50mm", 0.05)], reference_radius_m=0.5
    )
    assert not verdict.ok
    assert any("initial-field calibration" in reason for reason in verdict.reasons)

    ok = evaluate_validation_isolation(
        config, [_stub_case("tnt_50mm", 0.05)], reference_radius_m=0.05
    )
    assert ok.ok, ok.reasons


def test_strict_blind_test_needs_a_real_blind_counterpart():
    """声明了盲测却没有盲测对象，是一句空话。"""
    config = _config([_case("tnt_50mm", 0.05, "train")], strict_blind=True,
                     reference_m=0.05)
    config = Config.from_mapping({
        **config.values,
        "splits": {"strict_blind_test": True, "final_validation_radii_mm": [500.0]},
    })
    verdict = evaluate_validation_isolation(config, [_stub_case("tnt_50mm", 0.05)])
    assert not verdict.ok
    assert any("no case at a final-validation radius" in r for r in verdict.reasons)

    # 关掉声明就不再有这条；破坏与否由调用方决定，不靠这一条冒充
    relaxed = Config.from_mapping({
        **config.values,
        "splits": {"strict_blind_test": False, "final_validation_radii_mm": [500.0]},
    })
    assert evaluate_validation_isolation(
        relaxed, [_stub_case("tnt_50mm", 0.05)]
    ).ok


def test_the_declared_radii_can_be_overridden_in_the_config():
    config = Config.from_mapping({
        "reference_field": {"charge_radius_m": 0.05},
        "data": {"cases": []},
        "splits": {"strict_blind_test": False, "final_validation_radii_mm": [250.0, 500.0]},
    })
    assert final_validation_radii_m(config) == (0.25, 0.5)


def test_the_repository_base_config_declares_the_isolation_it_claims():
    """仓库里那份真实配置必须真的自洽，不能只是写着一句话。"""
    base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = load_config(base)
    specs = config.get("data.cases", [])
    assert specs, "base.yaml 里没有算例，这条测试失去意义"

    training = [
        _stub_case(spec["case_id"], float(spec["charge_radius_m"]))
        for spec in specs if str(spec.get("split")) == "train"
    ]
    reference_m = float(config.get("reference_field.charge_radius_m"))
    verdict = evaluate_validation_isolation(
        config, training, reference_radius_m=reference_m
    )
    assert verdict.ok, verdict.reasons
    assert 0.5 not in verdict.detail["training_radii_mm"], (
        "500 mm 进了训练半径"
    )
    assert 500.0 in verdict.detail["final_validation_radii_mm"]
    assert verdict.detail["strict_blind_test"] is True


def _write_config(directory: Path, cases: list[tuple[str, float, str]]) -> Path:
    """写一份继承 ``configs/base.yaml`` 的最小配置，只覆盖算例清单。"""
    base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"inherits: {base.as_posix()}", "", "data:", "  cases:"]
    for case_id, radius_m, split in cases:
        lines += [
            f"    - case_id: {case_id}",
            f"      charge_radius_m: {radius_m}",
            f"      directory: /nonexistent/{case_id}",
            f"      split: {split}",
        ]
    path = directory / "config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_training_entry_refuses_when_the_final_validation_radius_is_not_isolated(
    tmp_path, monkeypatch
):
    """§15：最终训练入口必须真的检查验证隔离，而不是把结论写在报表里。"""
    from spherical_blast_pinn.cli import train as train_module

    broken = _write_config(tmp_path / "broken", [
        ("tnt_50mm", 0.050, "train"),
        ("tnt_500mm", 0.500, "train"),          # 破坏了隔离
    ])

    def fake_build_context(config, **kwargs):
        return SimpleNamespace(
            model=object(),
            data_contract=SimpleNamespace(ready=True, reasons=[]),
            validation_isolation=evaluate_validation_isolation(config, []),
            training_cases=[],
            reference=None,
            scales={},
            time_scale=1.0,
            sampling=None,
            compatibility={},
        )

    monkeypatch.setattr(train_module, "build_context", fake_build_context)
    code = train_module.main(
        ["--config", str(broken), "--output-dir", str(tmp_path / "out")]
    )
    assert code == 5, "500 mm 被标成训练算例时，训练入口必须拒绝开始"

    # 对照组：把 500 mm 恢复成非训练 split，隔离闸门就放行。
    good = _write_config(tmp_path / "good", [
        ("tnt_50mm", 0.050, "train"),
        ("tnt_500mm", 0.500, "extrapolation_test"),
    ])
    verdict = fake_build_context(load_config(good)).validation_isolation
    assert verdict.ok, verdict.reasons


def test_predict_never_reaches_for_the_target_case_data():
    """§16.2：predict 只依赖模型、材料、参考 B 初场、目标 R0、查询时间与位置。

    这条用**命名空间**检查，不用行为检查：predict 的模块里根本不该出现算例加载
    的名字。行为检查在这里没有用——它必然通过，因为没有任何代码路径会去读数据；
    而真正会出问题的改动（有人为了画对比图顺手把目标算例接进 predict）是**引入
    一个名字**，那正是这个检查看得见的。
    """
    from spherical_blast_pinn.cli import predict as predict_module
    import inspect

    forbidden = {
        "load_cases", "load_extracted_case", "ExtractedCase", "build_context",
        "evaluate_data_contract", "shell_average", "raw_reader",
    }
    present = sorted(forbidden & set(vars(predict_module)))
    assert not present, (
        f"predict 里出现了算例数据入口 {present}；§16.2 不允许它读目标 DYNA 场、"
        "前沿、初值或归一化极值"
    )

    # 而且接口本身也没有接收算例数据的地方
    signature = inspect.signature(predict_module.predict_grid)
    assert "case" not in " ".join(signature.parameters)
    assert set(signature.parameters) == {
        "model", "physical_times", "radii", "scaled_time_coverage"
    }
