"""有效性记录**被消费**了吗（提示词 §11.1、§10）。

这一组测试针对的是一类很具体的假完成：把新字段读进来了、契约检查也变绿了，
但**取数的地方**仍然用旧的那张联合掩码。字段存在 ≠ 字段生效，而两者在报表上
长得一模一样。

每一条都先构造出"两种判据给出不同答案"的输入，再断言取数用的是哪一种。没有这种
输入，任何实现都会通过。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.extracted_case import ExtractedCase, DetectorSettings
from spherical_blast_pinn.physics import material as mat


def _case(
    *,
    material: np.ndarray,
    valid: np.ndarray,
    field_validity: dict[str, np.ndarray] | None,
    density: np.ndarray,
    velocity: np.ndarray,
    energy: np.ndarray,
    contact_radius: np.ndarray,
    shock_radius: np.ndarray,
    contact_valid: np.ndarray | None = None,
    shock_valid: np.ndarray | None = None,
) -> ExtractedCase:
    n_t, n_x = material.shape
    radius = np.linspace(0.004, 0.30, n_x)
    # 时间必须落在 PINN 原点上之后（``T_h = R_0/D + epsilon`` ≈ 8.2 µs），否则
    # ``supervisable_frames`` 一帧都不剩，采样器会先拒绝整份数据。
    times = np.linspace(2.0e-5, 2.0e-4, n_t)
    return ExtractedCase(
        case_id="crafted",
        charge_radius=0.05,
        times=times,
        radius=radius,
        density=density,
        velocity=velocity,
        pressure=np.full_like(density, 1.0e5),
        specific_energy=energy,
        contact_radius=contact_radius,
        shock_radius=shock_radius,
        contact_valid=(
            np.ones(n_t, dtype=bool) if contact_valid is None else contact_valid
        ),
        shock_valid=np.ones(n_t, dtype=bool) if shock_valid is None else shock_valid,
        material=material,
        valid=valid,
        source={"detonation_velocity_m_s": 6930.0, "symmetry_factor": 8.0},
        detector=DetectorSettings(),
        field_validity=field_validity,
    )


def _two_medium_case(**overrides) -> ExtractedCase:
    """一帧两介质的最小算例：内侧产物、外侧空气。"""
    n_t, n_x = 1, 40
    radius = np.linspace(0.004, 0.30, n_x)
    material = np.where(radius < 0.05, 0, 1).reshape(1, -1).repeat(n_t, axis=0)
    density = np.where(material == 0, 1600.0, 6.0 * mat.AIR_DENSITY)
    velocity = np.full((n_t, n_x), 10.0)
    energy = np.where(material == 0, 1.0e7, 2.0e5)
    base = dict(
        material=material,
        valid=np.ones((n_t, n_x), dtype=bool),
        field_validity=None,
        density=density,
        velocity=velocity,
        energy=energy,
        contact_radius=np.full(n_t, 0.05),
        shock_radius=np.full(n_t, 0.10),
    )
    base.update(overrides)
    return _case(**base)


# ==========================================================================
# §11.1 逐字段有效性必须真的控制取数
# ==========================================================================


















# ==========================================================================
# §11.1 两条前沿独立；§10 监督资格与检测状态分开
# ==========================================================================




def test_the_eligibility_arrays_win_over_the_detection_code(tmp_path):
    """资格数组与三态码冲突时，以**资格**为准（§10：两者分开）。

    三态码说的是"看见了什么"，资格说的是"够不够格当标签"。只看三态码的话，
    一个被提取器明确标为不具资格的前沿会以"已检测"的身份进入监督。
    """
    import json

    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    n_t, n_x = 3, 60
    radius = np.linspace(0.004, 0.30, n_x)
    directory = Path(tmp_path) / "elig"
    directory.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 2.0e-4, n_t)
    density = np.where(radius < 0.05, 1600.0, 6.0 * mat.AIR_DENSITY)
    density = density[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "raw_line.npz",
        t=times, x=radius, rho=density, u=np.zeros_like(density), P=np.full_like(density, 1.0e5),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"R_0": 0.05, "detonation_velocity_m_s": 6930.0}), encoding="utf-8"
    )
    material = np.where(radius < 0.05, 0, 1)[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "material.npz",
        material=material,
        valid=material != 2,
        contact_radius=np.full(n_t, 0.05),
        shock_radius=np.full(n_t, 0.10),
        # 三态码说"分辨得出"，资格数组说"不够格"
        contact_state=np.full(n_t, 2, dtype=np.int64),
        shock_state=np.full(n_t, 2, dtype=np.int64),
        contact_has_supervision=np.array([True, False, False]),
        shock_has_supervision=np.array([True, False, False]),
    )
    case = load_extracted_case(directory, case_id="elig")
    assert case.contact_valid.tolist() == [True, False, False], (
        "三态码压过了资格数组；§10 要求以资格为准"
    )
    assert case.shock_valid.tolist() == [True, False, False]
    sources = case.source["front_eligibility_source"]
    assert set(sources) == {"contact", "shock"}
    assert all("eligibility array" in value for value in sources.values())


def test_a_missing_eligibility_array_on_one_side_does_not_overwrite_the_other(tmp_path):
    """一侧明确写着 False、另一侧**没有**资格数组时，前者不得被回退覆盖。

    以前的实现是"两侧数组都在才用数组，否则两侧一起退回三态码"。于是接触侧被
    显式否决的记录，会因为**激波侧字段的缺失**而被丢掉——A 的缺失改变了 B 的
    判据，既违反"资格优先"，也违反"两条前沿独立"。
    """
    import json

    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    n_t, n_x = 3, 60
    radius = np.linspace(0.004, 0.30, n_x)
    directory = Path(tmp_path) / "onesided"
    directory.mkdir(parents=True, exist_ok=True)
    times = np.linspace(2.0e-5, 2.0e-4, n_t)
    density = np.where(radius < 0.05, 1600.0, 6.0 * mat.AIR_DENSITY)
    density = density[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "raw_line.npz",
        t=times, x=radius, rho=density, u=np.zeros_like(density), P=np.full_like(density, 1.0e5),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"R_0": 0.05, "detonation_velocity_m_s": 6930.0}), encoding="utf-8"
    )
    material = np.where(radius < 0.05, 0, 1)[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "material.npz",
        material=material,
        valid=material != 2,
        contact_radius=np.full(n_t, 0.05),
        shock_radius=np.full(n_t, 0.10),
        # 旧三态码说"两个都分辨得出"
        contact_state=np.full(n_t, 2, dtype=np.int64),
        shock_state=np.full(n_t, 2, dtype=np.int64),
        # 接触侧**明确否决**；激波侧没有资格数组
        contact_has_supervision=np.zeros(n_t, dtype=bool),
    )
    case = load_extracted_case(directory, case_id="onesided")
    assert not case.contact_valid.any(), (
        "接触侧被明确记录的 False 被另一侧字段的缺失触发的回退覆盖了"
    )
    assert case.shock_valid.all(), "激波侧应当按它自己那一档（三态码）判定"
    sources = case.source["front_eligibility_source"]
    assert "eligibility array" in sources["contact"]
    assert "three-state" in sources["shock"]


def test_an_all_nan_eligibility_array_is_refused(tmp_path):
    """全是 NaN 的资格表不得被读成"全具资格"。"""
    import json

    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    n_t, n_x = 3, 60
    radius = np.linspace(0.004, 0.30, n_x)
    directory = Path(tmp_path) / "nanelig"
    directory.mkdir(parents=True, exist_ok=True)
    times = np.linspace(2.0e-5, 2.0e-4, n_t)
    density = np.where(radius < 0.05, 1600.0, 6.0 * mat.AIR_DENSITY)
    density = density[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "raw_line.npz",
        t=times, x=radius, rho=density, u=np.zeros_like(density), P=np.full_like(density, 1.0e5),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"R_0": 0.05, "detonation_velocity_m_s": 6930.0}), encoding="utf-8"
    )
    material = np.where(radius < 0.05, 0, 1)[None, :].repeat(n_t, axis=0)
    np.savez(
        directory / "material.npz",
        material=material,
        valid=material != 2,
        contact_radius=np.full(n_t, 0.05),
        shock_radius=np.full(n_t, 0.10),
        contact_has_supervision=np.full(n_t, np.nan),
        shock_has_supervision=np.full(n_t, np.nan),
    )
    with pytest.raises(Exception, match="dtype"):
        load_extracted_case(directory, case_id="nanelig")
