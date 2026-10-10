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


class _StubReference:
    """只提供采样器真正用到的那两个方法。

    这里关心的是**数据监督**怎么取点，所以初值那一侧给一个解析的、处处有定义的
    替身即可——用一个真的参考场会把这条测试变成"参考场求解"的测试。
    """

    def contact_and_shock_at(self, time):
        return np.asarray([0.05]), np.asarray([0.10])

    def sample(self, radius, time):
        radius = np.asarray(radius, dtype=np.float64)
        return (
            np.where(radius < 0.05, 1600.0, 6.0 * mat.AIR_DENSITY),
            np.zeros_like(radius),
            np.where(radius < 0.05, 1.0e7, 2.0e5),
        )


def _sampler(case, count: int = 40):
    from spherical_blast_pinn.physics.similitude import SimilitudeScales
    from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

    scales = SimilitudeScales.for_radius(
        0.05, time_scale=2.0, detonation_velocity=case.detonation_velocity
    )
    return RadiusSampler(
        case=case, scales=scales, reference=_StubReference(),
        config=SamplingConfig(
            n_pde_product=2, n_pde_air=2, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=count, n_data_air=count,
            n_data_front=2,
        ),
        seed=0,
    )


def _finite_fraction(values) -> float:
    array = np.asarray(values.reshape(-1))
    return float(np.mean(np.isfinite(array)))


# ==========================================================================
# §11.1 逐字段有效性必须真的控制取数
# ==========================================================================


def test_a_field_valid_where_the_joint_mask_says_invalid_is_kept():
    """联合掩码说不可用、该字段自己的掩码说可用 ⇒ **保留**。

    这是"逐字段"与"联合"给出不同答案的第一个方向。旧实现取的是联合掩码，所以
    这里的密度标签会被整片丢掉。
    """
    case = _two_medium_case()
    validity = {name: np.ones((1, 40), dtype=bool) for name in
                ("density", "velocity", "specific_energy", "pressure")}
    # 位置 10..19 上联合掩码为假，但密度自己的掩码为真
    case.valid = case.valid.copy()
    case.valid[0, 10:20] = False
    validity["density"][0, 10:20] = True
    validity["velocity"][0, 10:20] = False
    validity["specific_energy"][0, 10:20] = False
    case.field_validity = validity

    batch = _sampler(case, count=200).batch()
    # 密度标签必须还在（联合掩码为假的那些点上也有）
    assert _finite_fraction(batch.rho_data_product) > 0.0, (
        "密度在这个位置上自己的掩码是有效的，标签却被联合掩码丢掉了"
    )
    assert batch.diagnostics["data_validity_source"].startswith("per-field"), (
        "取数用的是联合掩码，逐字段记录没有生效"
    )


def test_a_field_invalid_where_the_joint_mask_says_valid_is_dropped():
    """联合掩码说可用、该字段自己的掩码说不可用 ⇒ **丢掉**。

    反方向。旧实现同样取联合掩码，所以这里的标签会被当成可用——而它明明被标为
    不可用，且数值恰好有限（有限不等于有效）。
    """
    case = _two_medium_case()
    validity = {name: np.ones((1, 40), dtype=bool) for name in
                ("density", "velocity", "specific_energy", "pressure")}
    validity["density"][0, 5:15] = False
    case.field_validity = validity

    batch = _sampler(case, count=400).batch()
    assert _finite_fraction(batch.rho_data_product) < 1.0, (
        "密度被标为不可用的位置上，标签仍然全是有限值——逐字段掩码被忽略了"
    )
    # 其他字段不受影响（§11.1：一个字段缺失不丢掉其他字段）
    assert _finite_fraction(batch.u_data_product) == pytest.approx(1.0)
    assert _finite_fraction(batch.e_data_product) == pytest.approx(1.0)


def test_the_sampler_records_which_validity_mask_it_used():
    """用的是哪一档要能读出来——退化和等价在报表上不能长得一样。"""
    without = _two_medium_case()
    assert without.field_validity is None
    batch = _sampler(without, count=20).batch()
    assert "joint" in batch.diagnostics["data_validity_source"]

    with_validity = _two_medium_case(
        field_validity={name: np.ones((1, 40), dtype=bool) for name in
                        ("density", "velocity", "specific_energy", "pressure")}
    )
    batch = _sampler(with_validity, count=20).batch()
    assert "per-field" in batch.diagnostics["data_validity_source"]


def test_a_partial_field_validity_mapping_is_refused_by_the_sampler():
    """只给一半的逐字段记录时必须抛错，不能悄悄退回联合掩码。

    退回会让"缺失的那一半"变成"都有效"，而那正是这份记录要防的事。
    """
    from spherical_blast_pinn.sampling.sampler import SamplingError

    case = _two_medium_case(
        field_validity={"density": np.ones((1, 40), dtype=bool)}
    )
    with pytest.raises(SamplingError, match="per-field validity"):
        _sampler(case, count=20).batch()


# ==========================================================================
# §11.1 两条前沿独立；§10 监督资格与检测状态分开
# ==========================================================================


def test_a_missing_shock_does_not_strike_out_a_usable_contact():
    """激波缺失不得连累接触面。"""
    case = _two_medium_case(
        shock_radius=np.full(1, np.nan),
        # 半径与有效性必须一致：真实读取器里 ``shock_valid`` 是由
        # ``isfinite(shock_radius) & 资格`` 得来的。只把半径置 NaN 而不动有效性，
        # 造出的是一个**真实读取器不可能产生**的对象，测的就不是读取器的行为了。
        shock_valid=np.zeros(1, dtype=bool),
    )
    assert case.contact_valid.all()
    sampler = _sampler(case, count=20)
    contact_frames = sampler.case.contact_supervisable_frames()
    assert contact_frames.any(), (
        "激波没识别出来时接触面被一起撤标了；§11.1 要求两条前沿独立"
    )
    assert not sampler.case.air_supervisable_frames().any(), (
        "激波缺失时空气组不该有资格——空气层要两条前沿都在"
    )


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
    assert "eligibility arrays" in case.source["front_eligibility_source"]
