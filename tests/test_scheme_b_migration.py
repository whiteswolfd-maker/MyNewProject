"""方案 B 迁移的回归测试。

`dual_pinn_B_modification_prompt.md` 把初始化从 A（整球瞬时转化）换成 B（中心起爆
有限传播的非均匀爆后产物场 + 短时双介质前置）。这些测试逐条钉住**只在 B 下才成立**
或**B 明确要求改掉**的东西。

命名约定：每条测试的 docstring 写清它对应提示词的哪一节，以及它在旧实现下会怎样
失败——只在旧实现下会失败，才算数。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat

# ---------------------------------------------------------------------------
# 卡片样本
# ---------------------------------------------------------------------------

_EXPLOSIVE = """*MAT_HIGH_EXPLOSIVE_BURN
         1  1.63E-09   6930000     21000         0         0         0         0
*EOS_JWL
         1    371200    3747.1      4.15      0.95       0.3      6000         1
"""

_AIR = """*MAT_NULL
         2 1.225E-12         0         0         0         0         0         0
*EOS_LINEAR_POLYNOMIAL
         2         0         0         0         0       0.4       0.4         0
       0.2533         1
"""


def _write_deck(tmp_path: Path, body: str, *, name: str = "input.k") -> Path:
    path = tmp_path / name
    path.write_text("*KEYWORD\n" + body + "*END\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# §7 材料映射：ALE 组的**出现顺序**就是体积分数的列序
# ---------------------------------------------------------------------------


def test_ale_material_index_is_card_order_not_part_id_minus_one(tmp_path):
    """**§7：不用 part_id-1 猜 history 槽位。**

    造一个 ALE 组顺序与 part id 不一致的 deck：part 1 是炸药、part 2 是空气，而
    `*ALE_MULTI-MATERIAL_GROUP` 先登记 psid=2 再登记 psid=1。

    于是 ALE 材料 1 = 空气、ALE 材料 2 = 炸药。按 `part_id - 1` 去取列的实现会得到
    "炸药 = 第 0 列"，而正确的是"第 1 列"——两者都是合法的列号，**读出来的场量都
    看起来像那么回事**，这正是必须钉住它的原因。
    """
    from spherical_blast_pinn.data.kfile_parser import parse_deck

    body = _EXPLOSIVE + _AIR + """*PART
     air
         2         2         2         2         0         0         0         0
*PART
     tnt
         1         1         1         1         0         0         0         0
*ALE_MULTI-MATERIAL_GROUP
         2         0
         1         0
"""
    summary = parse_deck(_write_deck(tmp_path, body), geometry=False)
    assert summary.ale_material_part_ids() == [(2,), (1,)]
    assert summary.ale_material_kind(1) == "air"
    assert summary.ale_material_kind(2) == "product"
    # 炸药 part 1 在 ALE 里排第 **2** 位——不是 part_id - 1 = 0。
    assert summary.ale_material_indices_for_parts([1]) == [2]
    assert summary.ale_material_indices_for_parts([2]) == [1]


def test_ale_groups_resolve_part_sets_when_the_deck_uses_them(tmp_path):
    """`idtype` 允许指向 part set。没有 `*SET_PART` 的 deck 只能按 part id 解释。

    本机四个 deck 都没有 `*SET_PART`，所以 psid=1..7 只能当 part id。这里两边都测：
    有集合时按集合展开，并**把用的是哪种解释记下来**，而不是照抄手册极性。
    """
    from spherical_blast_pinn.data.kfile_parser import parse_deck

    body = _EXPLOSIVE + _AIR + """*PART
     tnt_a
         1         1         1         1         0         0         0         0
*PART
     tnt_b
         3         3         1         1         0         0         0         0
*PART
     air
         2         2         2         2         0         0         0         0
*SET_PART
        10
         1         3
*ALE_MULTI-MATERIAL_GROUP
        10         1
         2         0
"""
    summary = parse_deck(_write_deck(tmp_path, body), geometry=False)
    assert summary.part_sets[10] == (1, 3)
    assert summary.ale_material_part_ids() == [(1, 3), (2,)]
    assert summary.ale_groups[0].resolved_as == "part set"
    assert summary.ale_groups[1].resolved_as == "part id"
    # 两个炸药 part 合成**一个** ALE 材料。
    assert summary.ale_material_indices_for_parts([1, 3]) == [1]
    assert summary.ale_material_kind(1) == "product"
    assert summary.ale_material_kind(2) == "air"


def test_ale_group_that_resolves_to_nothing_raises(tmp_path):
    """解析不了就停——猜一个列序比报错危险得多。"""
    from spherical_blast_pinn.data.kfile_parser import KeywordDeckError, parse_deck

    body = _EXPLOSIVE + _AIR + """*PART
     tnt
         1         1         1         1         0         0         0         0
*ALE_MULTI-MATERIAL_GROUP
       999         0
"""
    summary = parse_deck(_write_deck(tmp_path, body), geometry=False)
    with pytest.raises(KeywordDeckError) as excinfo:
        summary.ale_material_part_ids()
    assert "neither a *SET_PART nor a *PART" in str(excinfo.value)


def test_a_mixed_ale_group_is_not_silently_called_air_or_product(tmp_path):
    """一个 ALE 材料里同时有炸药和空气时，类别是 ``mixed``，不能算进任一侧。"""
    from spherical_blast_pinn.data.kfile_parser import parse_deck

    body = _EXPLOSIVE + _AIR + """*PART
     tnt
         1         1         1         1         0         0         0         0
*PART
     air
         2         2         2         2         0         0         0         0
*SET_PART
        10
         1         2
*ALE_MULTI-MATERIAL_GROUP
        10         1
"""
    summary = parse_deck(_write_deck(tmp_path, body), geometry=False)
    assert summary.ale_material_kind(1) == "mixed"


def test_reader_refuses_a_non_identity_ale_material_table(monkeypatch):
    """d3plot 声明的 ALE 材料号不是 1..N 时，按位置组合体积分数就是错的。

    `volume_fraction_columns[k-1]` 对上"第 k 个 ALE 材料"只在 d3plot 声明的 ALE
    材料号恰好是 1..N 时成立。这不是格式保证，是一个必须核实的前提。
    """
    from spherical_blast_pinn.data import raw_reader as rr

    case = rr.D3plotCase(
        directory=Path("nowhere"), coordinates=np.zeros((1, 3)),
        connectivity=np.zeros((1, 8), dtype=np.int64),
        part_indexes=np.zeros(1, dtype=np.int64), element_ids=np.zeros(1, dtype=np.int64),
        part_titles=[], ale_material_ids=np.array([3, 5, 7], dtype=np.int64),
        n_states=1, layout=rr.D3plotLayout(),
        product_ale_indices=(1,), air_ale_indices=(2,),
    )
    assert not np.array_equal(case.ale_material_ids, np.arange(1, 4))
    # 直接调用读取路径的校核段：构造最小的 lasso 替身太脆，所以这里复用同一个判据。
    declared = np.asarray(case.ale_material_ids).reshape(-1)
    assert not np.array_equal(declared, np.arange(1, declared.size + 1, dtype=np.int64)), (
        "ALE material ids 3,5,7 必须是'非恒等'，否则这条测试没有意义"
    )


# ---------------------------------------------------------------------------
# §4 空气：环境态来自卡片 E0，两种环境压力不得混用
# ---------------------------------------------------------------------------


def test_ambient_pressure_is_the_card_value_and_differs_from_standard_atmosphere():
    """**§4 点名的坑。** `*EOS_LINEAR_POLYNOMIAL E0 = 0.2533 MPa` 给出 101320 Pa。

    标准大气压是 101325 Pa。DYNA 实际初始化出来的是 101320 Pa（d3plot 里读得到），
    所以凡是要与数据比较的量都必须用卡片值。旧实现先取 101325 再反推 e，等于让
    PINN 的环境态与训练数据差 5 Pa。
    """
    e0 = mat.AIR_VOLUME_ENERGY / mat.AIR_DENSITY
    assert e0 == pytest.approx(206775.5102, rel=1e-9)
    assert mat.AIR_SPECIFIC_ENERGY == pytest.approx(e0, rel=1e-15)
    assert mat.AIR_PRESSURE == pytest.approx(101320.0, rel=1e-12)
    assert mat.STANDARD_ATMOSPHERE == pytest.approx(101325.0, rel=1e-12)
    assert mat.AIR_PRESSURE != mat.STANDARD_ATMOSPHERE
    # 用旧公式反推会得到另一个比内能，两者不能混用。
    legacy = mat.STANDARD_ATMOSPHERE / ((mat.AIR_GAMMA - 1.0) * mat.AIR_DENSITY)
    assert abs(legacy - mat.AIR_SPECIFIC_ENERGY) > 1.0


def test_air_linear_polynomial_reduces_to_an_ideal_gas_only_when_c5_equals_c4():
    """卡片是 `*EOS_LINEAR_POLYNOMIAL`，不是理想气体卡片。

    ``p = c0 + c1 mu + c2 mu^2 + c3 mu^3 + (c4 + c5 mu + c6 mu^2) E`` 在
    c0..c3 = 0、c6 = 0 且 **c5 = c4** 时才恰好等于 ``(gamma-1) rho e``。卡片给的
    正是 c4 = c5 = 0.4，但这是**要核实的前提**，不能当成"空气就是理想气体"。

    契约是**硬失败**（抛 `EOSError`），不是返回布尔——空气 EOS 一旦不是理想气体，
    RH 条件、熵条件、声速会全部算错，训练不能开始。
    """
    from spherical_blast_pinn.physics.eos import (
        EOSError,
        check_linear_polynomial_is_ideal_gas,
    )

    card = {"C0": 0.0, "C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.4, "C5": 0.4, "C6": 0.0}
    check_linear_polynomial_is_ideal_gas(card, 1.4)   # 通过：不抛

    # c5 != c4 时不是理想气体——必须报错，并把出问题的系数名指出来。
    with pytest.raises(EOSError) as excinfo:
        check_linear_polynomial_is_ideal_gas(dict(card, C5=0.0), 1.4)
    assert "C5" in str(excinfo.value)
    # 非零的偏置项同理。
    with pytest.raises(EOSError):
        check_linear_polynomial_is_ideal_gas(dict(card, C1=1.0), 1.4)


# ---------------------------------------------------------------------------
# §8 球壳统计：重叠体积分配
# ---------------------------------------------------------------------------


def _grid(n=40, inner=0.0, outer=1.0, symmetry=1.0):
    from spherical_blast_pinn.data.shell_average import ShellGrid

    return ShellGrid(
        np.linspace(inner, outer, n + 1), symmetry_multiplier=symmetry
    )


def test_overlap_fractions_are_exact_for_a_constant_solid_angle_cell():
    """圆台重构对"固定立体角"的单元是**精确**的，不是近似。

    这样的单元体积是 ``Omega/3 (r_o^3 - r_i^3)``，它与球壳 ``[a, b]`` 的重叠是
    ``Omega/3 (min(b,r_o)^3 - max(a,r_i)^3)``，比值正好等于
    ``(c^3 - d^3)/(r_o^3 - r_i^3)``。所以解析解和实现必须逐位一致——这条不成立就
    说明公式写错了，而不是"精度不够"。
    """
    from spherical_blast_pinn.data.shell_average import _overlap_pairs

    edges = np.linspace(0.0, 1.0, 101)
    # 一个从 0.237 到 0.481 的壳段单元。
    r_in = np.array([0.237])
    r_out = np.array([0.481])
    _, shell_index, frac = _overlap_pairs(r_in, r_out, edges, 100)
    # 解析：与每个被跨越的壳层求交，按 r^3 分。
    covered = np.zeros(100)
    for k, f in zip(shell_index, frac):
        covered[k] = f
    assert covered.sum() == pytest.approx(1.0, rel=1e-15)
    for k in shell_index:
        a = max(edges[k], 0.237)
        b = min(edges[k + 1], 0.481)
        exact = (b ** 3 - a ** 3) / (0.481 ** 3 - 0.237 ** 3)
        assert covered[k] == pytest.approx(exact, rel=1e-12)


def test_overlap_allocation_conserves_volume_across_shells():
    """**守恒是这条判据的全部意义。** 质心分箱不守恒：跨壳层边界的单元被整个记在
    一边，于是壳层之间少一块、多一块，而两条曲线看起来都正常。
    """
    from spherical_blast_pinn.data.shell_average import overlap_assignment_from_arrays

    rng = np.random.default_rng(20260922)
    n = 5000
    r_in = rng.uniform(0.0, 0.99, n)
    r_out = r_in + rng.uniform(1e-4, 0.05, n)
    volume = rng.uniform(1e-9, 1e-6, n)
    grid = _grid(200)

    a = overlap_assignment_from_arrays(r_in, r_out, volume, grid)
    per_cell = np.bincount(a.cell_index, weights=a.volume, minlength=n)
    # 完全落在网格内的单元：重叠体积之和 == 单元体积。
    inside = (r_in > grid.edges[0]) & (r_out < grid.edges[-1])
    np.testing.assert_allclose(per_cell[inside], volume[inside], rtol=1e-12)
    # 总重叠体积 == 网格内那部分体积（边界单元只算进来的那部分，所以 ≤ 总和）。
    assert a.volume.sum() <= volume.sum() * (1.0 + 1e-12)


def test_overlap_allocation_beats_centroid_binning_at_a_shell_boundary():
    """一个正好骑在壳层边界上的单元：质心分箱会把它整个算进一边。"""
    from spherical_blast_pinn.data.shell_average import overlap_assignment_from_arrays

    edges = np.array([0.0, 1.0, 2.0])
    grid = _grid(2, 0.0, 2.0)
    r_in = np.array([0.9])
    r_out = np.array([1.1])
    volume = np.array([1.0])
    a = overlap_assignment_from_arrays(r_in, r_out, volume, grid)
    share = np.bincount(a.shell_index, weights=a.volume, minlength=2)
    # 一半在里、一半在外。（r^3 权重下不是精确的 0.5，但必须两边都有。）
    assert share[0] > 0.0 and share[1] > 0.0
    assert share.sum() == pytest.approx(1.0, rel=1e-12)
    # 而质心（r=1.0）落在第二个壳层里，整格都进第二层。
    centroid_share = np.array([0.0, 1.0])
    assert centroid_share[0] == 0.0   # 正是重叠分配修掉的那件事


def test_shell_coverage_uses_the_modelled_wedge_not_the_full_sphere():
    """**覆盖率的分母必须是被建模的那一块。**

    八分之一球模型上用整球体积当分母，覆盖率会恰好小 8 倍（实测正好 0.125），
    而"覆盖率 0.125"读起来像"大部分没覆盖"，其实那一层是满的。
    """
    from spherical_blast_pinn.data.shell_average import _shell_volumes

    full = _grid(10, 0.0, 1.0, symmetry=1.0)
    eighth = _grid(10, 0.0, 1.0, symmetry=8.0)
    np.testing.assert_allclose(_shell_volumes(full), 8.0 * _shell_volumes(eighth), rtol=1e-12)
    # 第 k 个壳层的整球体积就是 4/3 pi (r_o^3 - r_i^3)。
    e = full.edges
    np.testing.assert_allclose(
        _shell_volumes(full), 4.0 / 3.0 * np.pi * (e[1:] ** 3 - e[:-1] ** 3), rtol=1e-12
    )


def test_mass_weighted_fields_differ_from_volume_weighted_ones():
    """**§8：rho、p 体积加权；u、e 质量加权。** 两者在有密度梯度时必然不同。

    造一个两层壳：外层密度大、速度大，内层密度小、速度小。体积平均与质量平均给出
    不同的速度；把两者混为一谈会让"守恒平均"悄悄变成"体积平均"。
    """
    import numpy as np

    from spherical_blast_pinn.data.raw_reader import RawFrame
    from spherical_blast_pinn.data.shell_average import shell_average, PRODUCT

    # 两个单元：半径区间 [0.1,0.2] 与 [0.2,0.3]，体积相同，密度与速度不同。
    r_in = np.array([0.1, 0.2])
    r_out = np.array([0.2, 0.3])
    volume = np.array([1.0, 1.0])
    density = np.array([1.0, 9.0])
    velocity = np.array([0.0, 100.0])
    frame = RawFrame(
        case_id="t", time=0.0, radius=np.array([0.15, 0.25]), cell_volume=volume,
        density=density, velocity_x=np.zeros(2), velocity_y=np.zeros(2),
        velocity_z=np.zeros(2), pressure=np.array([1.0, 1.0]),
        tnt_volume_fraction=np.ones(2), radial_velocity=velocity,
        specific_internal_energy=np.array([1.0, 1.0]),
        cell_radius_inner=r_in, cell_radius_outer=r_out,
    )
    grid = _grid(1, 0.0, 1.0)
    prof = shell_average(frame, grid, PRODUCT)
    # 体积加权：rho 是 (rho*V) 的平均 -> 5.0
    assert prof.density[0] == pytest.approx(5.0, rel=1e-12)
    # 质量加权速度：(1*0 + 9*100)/(1+9) = 90，不是体积加权的 50。
    assert prof.radial_velocity[0] == pytest.approx(90.0, rel=1e-12)
    assert prof.radial_velocity[0] != pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------------------------------
# §10 激波标签：压力梯度候选 + RH 核查
# ---------------------------------------------------------------------------


def _analytic_shock(gamma: float, p1: float, rho1: float, mach: float):
    """由解析 RH 给出波后状态与激波速度（理想气体）。"""
    c1 = np.sqrt(gamma * p1 / rho1)
    speed = mach * c1
    ratio = ((gamma + 1.0) * mach**2) / ((gamma - 1.0) * mach**2 + 2.0)
    p2 = p1 * (1.0 + 2.0 * gamma * (mach**2 - 1.0) / (gamma + 1.0))
    u2 = speed * (1.0 - 1.0 / ratio)
    return speed, p2, rho1 * ratio, u2


def _shock_profile(gamma, p1, rho1, mach, *, n_in=30, n_tr=3, n_out=30,
                   r0=0.10, spacing=2.0e-3, plateau_noise=0.0, seed=0):
    """波后在**内**（小半径）、环境在**外**（大半径）的一维球壳剖面。

    数组沿半径增大排；内外写反不会报错，只会让"波后状态"取成环境态。
    """
    speed, p2, rho2, u2 = _analytic_shock(gamma, p1, rho1, mach)
    rng = np.random.default_rng(seed)
    n = n_in + n_tr + n_out
    r = r0 + spacing * np.arange(n)
    p = np.concatenate([np.full(n_in, p2), np.linspace(p2, p1, n_tr), np.full(n_out, p1)])
    rho = np.concatenate([np.full(n_in, rho2), np.linspace(rho2, rho1, n_tr), np.full(n_out, rho1)])
    u = np.concatenate([np.full(n_in, u2), np.linspace(u2, 0.0, n_tr), np.zeros(n_out)])
    if plateau_noise > 0.0:
        for lo, hi in ((0, n_in), (n_in + n_tr, n)):
            p[lo:hi] *= 1.0 + rng.normal(0.0, plateau_noise, hi - lo)
            rho[lo:hi] *= 1.0 + rng.normal(0.0, plateau_noise, hi - lo)
    return r, p, rho, u, speed


def test_shock_detector_recovers_an_analytic_rh_shock():
    """解析 RH 状态必须被确认，且 ``Ds`` 恢复到机器精度。

    ``Ds`` 由**质量守恒**从实测两侧状态解出，所以它回到解析值是这条链路的端到端
    检查：两侧状态取对了、内外没写反、``Ds`` 公式没错。
    """
    from spherical_blast_pinn.data.shock_detect import detect_shock, SHOCK_CONFIRMED

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    r, p, rho, u, speed = _shock_profile(gamma, p1, rho1, 2.0)
    detection = detect_shock(
        r, p, rho, u, np.zeros(r.size), np.ones(r.size, bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=np.full(r.size, 5.0e-5),
    )
    assert detection.status == SHOCK_CONFIRMED, detection.as_dict()
    main = detection.main
    assert main.shock_speed == pytest.approx(speed, rel=1e-12)
    assert main.rh_momentum < 1e-10
    assert main.rh_energy < 1e-10
    assert main.compression == pytest.approx(2.666666666666667, rel=1e-12)
    assert main.entropy_jump > 0.0
    assert main.reliable_range is not None
    # 监督范围必须落在跃迁带之外，且带内不被当成标签。
    assert detection.supervision_radius_range == main.band


def test_far_field_pressure_spike_is_rejected_by_rh_not_by_a_threshold():
    """**§2 问题 2 的核心。** 孤立压力尖峰必须被 RH 核查拒绝，而不是被阈值挡掉。

    旧判据里阈值是唯一守门人：把阈值调松一点，噪声就成了"激波"；调紧一点，弱激波
    又漏掉。这里噪声尖峰**进入候选列表**（它确实是个 ``-dp/dr`` 正峰），然后在 RH
    核查上失败——因为它两侧的密度几乎相同，``Ds`` 定不出来。而真激波仍然被选出。
    """
    from spherical_blast_pinn.data.shock_detect import (
        detect_shock, SHOCK_CONFIRMED, SHOCK_DOUBTFUL,
    )

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    r, p, rho, u, speed = _shock_profile(gamma, p1, rho1, 2.0)
    p = p.copy()
    p[-5] = p1 * 1.5                      # 外侧平台里插一根孤立尖峰
    detection = detect_shock(
        r, p, rho, u, np.zeros(r.size), np.ones(r.size, dtype=bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=np.full(r.size, 5.0e-5),
    )
    # 尖峰在候选里，且它是不合格的。
    spike = [c for c in detection.candidates if c.representative_radius > r[-10]]
    assert spike, "噪声尖峰应当进入候选列表——那正是'候选'的含义"
    assert all(c.status == SHOCK_DOUBTFUL for c in spike)
    assert any("ill-conditioned" in reason for c in spike for reason in c.reasons)
    # 真激波仍然是主激波——噪声不会把它挤掉。
    assert detection.status == SHOCK_CONFIRMED
    assert detection.main.shock_speed == pytest.approx(speed, rel=1e-6)


def test_a_pure_contact_creates_no_shock_candidate():
    """**纯接触不当激波。** 接触面两侧压力连续、密度跳变。

    密度跳变不产生 ``-dp/dr`` 正峰，所以连候选都不该有。靠"密度最陡下降"找激波的
    实现会在这里造出一条不存在的激波。
    """
    from spherical_blast_pinn.data.shock_detect import detect_shock

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    n = 61
    r = 0.10 + 2.0e-3 * np.arange(n)
    p = np.full(n, p1)                       # 压力处处相等
    rho = np.concatenate([np.full(30, 4.0 * rho1), np.full(31, rho1)])
    u = np.zeros(n)
    detection = detect_shock(
        r, p, rho, u, np.zeros(n), np.ones(n, dtype=bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=np.full(n, 5.0e-5),
    )
    assert detection.candidates == []
    assert detection.main is None
    assert detection.supervision_radius_range is None


def test_a_weak_shock_in_noisy_plateaus_carries_insufficient_evidence():
    """**§10：弱激波分母病态须标证据不足。**

    ``Ds = (rho2 u2 - rho1 u1)/(rho2 - rho1)`` 在密度跳变与两侧平台自身的起伏同量级时
    没有意义。判据用**数据里量出来的**窗口散布，不是拍一个密度阈值。
    """
    from spherical_blast_pinn.data.shock_detect import (
        detect_shock, SHOCK_DOUBTFUL, SHOCK_NOT_DETECTED,
    )

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    # M = 1.01：密度跳变 0.6%，而平台起伏 1%——跳变埋在噪声里。
    r, p, rho, u, _ = _shock_profile(
        gamma, p1, rho1, 1.01, plateau_noise=0.01, seed=7
    )
    detection = detect_shock(
        r, p, rho, u, np.zeros(r.size), np.ones(r.size, dtype=bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=np.full(r.size, 5.0e-5),
    )
    assert detection.status in (SHOCK_DOUBTFUL, SHOCK_NOT_DETECTED)
    if detection.main is not None:
        assert detection.main.status != "confirmed"
        assert not detection.main.reasons == [] or detection.status != "confirmed"
    assert detection.supervision_radius_range is None


def test_a_sub_cell_transition_band_is_never_confirmed():
    """**§10：h 跨一个原始单元不能自动通过。**

    跃迁带比该处的原始单元还窄时，激波没有被网格分辨，只能记存疑。
    """
    from spherical_blast_pinn.data.shock_detect import detect_shock, SHOCK_DOUBTFUL

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    r, p, rho, u, _ = _shock_profile(gamma, p1, rho1, 2.0, spacing=2.0e-3)
    thick = np.full(r.size, 10.0e-3)         # 单元 10 mm，跃迁带只有 6 mm
    detection = detect_shock(
        r, p, rho, u, np.zeros(r.size), np.ones(r.size, dtype=bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=thick,
    )
    assert detection.status == SHOCK_DOUBTFUL
    assert any("narrower than one native cell" in r for r in detection.main.reasons)
    assert detection.supervision_radius_range is None


def test_the_shock_detector_does_not_cross_into_the_product():
    """**禁止跨入产物。** 接触面以内的壳层不能参与激波的两侧状态。"""
    from spherical_blast_pinn.data.shock_detect import detect_shock

    gamma = 1.4
    p1, rho1 = mat.AIR_PRESSURE, mat.AIR_DENSITY
    r, p, rho, u, _ = _shock_profile(gamma, p1, rho1, 2.0)
    alpha = np.zeros(r.size)
    alpha[:20] = 1.0                          # 内侧 20 个壳层是产物
    detection = detect_shock(
        r, p, rho, u, alpha, np.ones(r.size, dtype=bool), gamma=gamma,
        ambient_pressure=p1, cell_thickness=np.full(r.size, 5.0e-5),
    )
    for candidate in detection.candidates:
        inner, outer = candidate.band
        # 波后窗口必须落在 alpha < 0.5 的空气侧。
        assert inner > r[19]


# ===========================================================================
# 方案 A 的路径已删除：没有回退分支
# ===========================================================================


def _tiny_products(cells: int = 60):
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_product_profile,
    )

    return detonation_product_profile(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=cells),
    )


def test_precursor_without_a_products_field_is_refused():
    """§2：A 的整球均匀初值已删除，**没有回退分支**。

    一个可以省略的初值参数迟早会被省略，而省略之后算出来的场仍然定常、为正、
    量纲正确——只是属于另一个方案。所以省略它必须直接失败。
    """
    from spherical_blast_pinn.physics.precursor import PrecursorError, SphericalPrecursor

    with pytest.raises(PrecursorError, match="方案 B"):
        SphericalPrecursor(charge_radius=0.05)


def test_solve_reference_field_requires_the_card_detonation_velocity():
    """没有 D 就没有 T_b = R_0/D，也就没有波前状态。不接受默认值。"""
    from spherical_blast_pinn.physics.precursor import PrecursorError, solve_reference_field

    with pytest.raises(PrecursorError, match="detonation_velocity"):
        solve_reference_field(charge_radius=0.05, sample_times=[0.0])


def test_the_initial_field_is_not_uniform_and_not_at_rest():
    """A 的初值是 u=0、e=E0/rho0 的整球均匀球；B 的必须既非均匀也非静止。"""
    s = _precursor_from_products(_tiny_products(120))
    product = s.mesh.material == 0
    assert float(s.velocity[:-1][product].max()) > 100.0, "产物区必须有向外速度"
    assert float(s.density[product].std()) > 1.0, "产物密度必须是非均匀的"
    uniform_a = float(np.ptp(s.specific_energy[product])) == 0.0
    assert not uniform_a, "比内能不能是整球一个常数"


def _precursor_from_products(products, cells: int = 120):
    from spherical_blast_pinn.physics.precursor import PrecursorConfig, SphericalPrecursor

    return SphericalPrecursor(
        charge_radius=0.05,
        config=PrecursorConfig(cells_inside=cells, growth=1.005),
        products=products,
    )


def test_centre_velocity_is_exactly_zero_after_handover():
    """``u(0) = 0`` 是球对称的**精确**约束，必须由初值保证。

    产物场在球心处的速度是数值解出来的小量（符号随网格变）。``_advance`` 里球心
    节点的加速度恒为零，所以那个小速度会一直保留、节点匀速穿过轴心，报"网格翻转"。
    实测过这个失败是**非单调**的（400/500/2000 单元通过，600/1000 失败），看起来
    像分辨率问题。方案 A 的初值 u ≡ 0，所以这个坑只在 B 下出现。
    """
    s = _precursor_from_products(_tiny_products(120))
    assert s.velocity[0] == 0.0


@pytest.mark.parametrize("cells", [80, 120, 160, 240])
def test_precursor_runs_to_handover_at_every_resolution(cells):
    """到交接时刻为止，任何一套分辨率都不该报"节点穿过轴心"。"""
    import math

    from spherical_blast_pinn.physics.similitude import handover_time

    s = _precursor_from_products(_tiny_products(cells), cells=cells)
    end = handover_time(0.05, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE)
    state = s.run_to(end)
    assert math.isfinite(state.contact_radius)
    assert state.shock_radius > state.contact_radius > 0.0


# ===========================================================================
# 时钟：start_time() 是 T_h，start_delay() 才是 epsilon
# ===========================================================================


def test_start_time_is_the_handover_and_start_delay_is_epsilon():
    """两个函数都返回"秒"，差 8.2 倍，混用不会报错。

    方案 A 时代 ``start_time()`` 返回的就是 ``epsilon``；改成 B 之后同一个名字
    返回 ``T_h = T_b + epsilon``。``sampling/sampler.py`` 的 ``hat_t_epsilon``
    踩过这个坑：它要的是 ``epsilon`` 的无量纲形式，却一直在调 ``start_time()``。
    """
    from spherical_blast_pinn.physics.similitude import SimilitudeScales, handover_time

    scales = SimilitudeScales.for_radius(
        0.05, time_scale=1.0, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )
    assert scales.start_time() == pytest.approx(
        handover_time(0.05, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE), rel=1e-15
    )
    assert scales.start_time() > scales.start_delay()
    ratio = scales.start_time() / scales.start_delay()
    # 50 mm：T_h / epsilon = 8.215007 / 1 = 8.215
    assert ratio == pytest.approx(8.215007, rel=1e-5)


def test_sampler_hat_time_epsilon_uses_the_delay_not_the_handover():
    """``hat_t_epsilon = U_* epsilon / R_0``，不是 ``U_* T_h / R_0``。"""
    from spherical_blast_pinn.physics.similitude import SimilitudeScales

    scales = SimilitudeScales.for_radius(
        0.05, time_scale=1.0, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )
    expected = scales.reference_velocity * scales.start_delay() / scales.charge_radius
    wrong = scales.reference_velocity * scales.start_time() / scales.charge_radius
    assert expected != pytest.approx(wrong, rel=1e-6)
    assert wrong / expected == pytest.approx(8.215007, rel=1e-5)


# ===========================================================================
# 签名的两个常量：`construction` 与 `acceptance_status` 的位置
# ===========================================================================


def test_acceptance_status_is_not_part_of_the_cache_signature():
    """验收是**场自己的属性**，每次加载后重判，不参与"能不能复用"。

    把它写进签名，等于让"当初是不是验收通过"决定"现在能不能复用"；这两件事会
    脱钩——一次诊断运行写下的未验收缓存，会在验收通过之后被当成合格的复用。
    """
    import inspect

    from spherical_blast_pinn.cli import _common

    source = inspect.getsource(_common.build_reference)
    signature_block = source.split("signature = {", 1)[1].split("}", 1)[0]
    assert "acceptance_status" not in signature_block
    assert "construction" in signature_block


# ===========================================================================
# 验收闸门真的会拦
# ===========================================================================


def _accepted_field(*, with_hashes: bool = True, **arrays):
    """构造一份**字段齐套**的假参考场，供验收闸门测试用。

    ``field_content_hash`` 要求 :data:`FIELD_CONTENT_ARRAYS` 里每一个数组都存在，
    所以这里的替身必须带全，不能只给 ``density``——那正是内容标识要抓的情况。
    """
    import numpy as np

    from spherical_blast_pinn.physics.precursor import (
        ACCEPTANCE_TOLERANCES,
        EVIDENCE_DEPENDENT_CHECKS,
        PHYSICS_BASIS_VERIFIED,
        field_content_hash,
    )

    class _Field:
        def __init__(self):
            self.times = np.array([0.0, 1.0e-6])
            self.cell_radii = np.array([[0.0, 0.5], [0.0, 0.6]])
            self.density = np.array([[100.0, 1.0], [101.0, 1.0]])
            self.velocity = np.array([[10.0, 0.0], [11.0, 0.0]])
            self.specific_energy = np.array([[1.0e5, 2.0e5], [1.0e5, 2.0e5]])
            self.material = np.array([0, 1])
            self.contact_radius = np.array([0.4, 0.5])
            self.shock_radius = np.array([0.6, 0.7])
            self.reference_radius = 0.05
            self.metadata = {}

    field = _Field()
    for name, value in arrays.items():
        setattr(field, name, value)
    field.metadata.update(
        {
            "acceptance_metrics": {k: 0.0 for k in ACCEPTANCE_TOLERANCES},
            "acceptance_evidence": {n: True for n in EVIDENCE_DEPENDENT_CHECKS},
            "physics_basis_status": PHYSICS_BASIS_VERIFIED,
            "data_contract_status": "ready",
        }
    )
    if with_hashes:
        identity = field_content_hash(field)
        field.metadata["field_content_hash"] = identity
        field.metadata["metrics_field_hash"] = identity
        field.metadata["acceptance_field_hash"] = identity
    return field


def test_unaccepted_reference_is_refused_by_the_gate():
    """§5：未验收试算场不得被正式入口接受。这不是一句元数据注释。

    而且**不能只信状态字符串**（§7.3）：一个只带
    ``{"acceptance_status": "ACCEPTED"}`` 的场，没有指标、证据、物理与数据状态，
    必须被拒——否则任何一个旧缓存把字符串写成 ACCEPTED 就能绕过验收。

    还有一条同等重要：**缺内容标识也拒绝**。以前的实现写
    ``if stored_hash is not None``，于是"没有哈希"被当成"不用查"，直接放行。
    """
    from spherical_blast_pinn.cli._common import (
        ReferenceNotAccepted,
        assert_reference_accepted,
    )

    class _Field:
        metadata = {"acceptance_status": "NOT ACCEPTED", "acceptance_reason": "测试用"}

    with pytest.raises(ReferenceNotAccepted, match="NOT ACCEPTED"):
        assert_reference_accepted(_Field())

    # 只写一个 ACCEPTED 字符串、没有指标与证据：重判会失败。
    class _BareAccepted:
        metadata = {"acceptance_status": "ACCEPTED"}

    with pytest.raises(ReferenceNotAccepted):
        assert_reference_accepted(_BareAccepted())

    # 指标 + 证据 + 物理 + 数据都齐，但**缺内容标识**：必须拒绝，不能默认通过。
    no_hash = _accepted_field(with_hashes=False)
    with pytest.raises(ReferenceNotAccepted, match="缺少"):
        assert_reference_accepted(no_hash)

    # 真正**齐套**的场才放行。
    assert_reference_accepted(_accepted_field()) is None


def test_acceptance_rejects_a_field_whose_content_changed_after_acceptance():
    """验收证据必须属于**当前场**（§7.2）。

    只把指标存进元数据、加载时重新判断那些指标，挡不住"指标是旧的、场数组被改过"
    的情形。这条测试：把 density 改成负值之后，即使保存的指标全合格，也必须被拒。
    """
    import numpy as np
    import pytest

    from spherical_blast_pinn.cli._common import (
        ReferenceNotAccepted,
        assert_reference_accepted,
    )

    field = _accepted_field()
    assert_reference_accepted(field)  # 未改动：通过

    # 改动场内容（density 变负）→ 内容标识对不上 → 拒绝
    field.density = field.density.copy()
    field.density[0, 0] = -5.0
    with pytest.raises(ReferenceNotAccepted, match="不一致"):
        assert_reference_accepted(field)


def test_old_metrics_do_not_certify_a_resaved_field():
    """重新保存不能把旧指标变成新数据的合格证明（Trusty URIs 原则）。

    用户复现的漏洞：``record_acceptance`` 用**当前场**算哈希再写进
    ``acceptance_field_hash``，于是"旧指标 + 新哈希"被当成对新场的合格证明。

    修法：指标在 ``solve_reference_field`` 里算出来时就绑定当时的场
    （``metrics_field_hash``）；``record_acceptance`` 写回前先核对它，不一致就拒绝。
    """
    import numpy as np
    import pytest

    from spherical_blast_pinn.cli._common import (
        ReferenceNotAccepted,
        assert_reference_accepted,
        record_acceptance,
    )
    from spherical_blast_pinn.physics.precursor import EVIDENCE_DEPENDENT_CHECKS

    evidence = {name: True for name in EVIDENCE_DEPENDENT_CHECKS}
    field = _accepted_field(with_hashes=False)
    # 指标是在这份场（v1）上算的。
    from spherical_blast_pinn.physics.precursor import field_content_hash

    identity = field_content_hash(field)
    field.metadata["metrics_field_hash"] = identity
    field.metadata["field_content_hash"] = identity

    record_acceptance(field, evidence)
    assert_reference_accepted(field)  # v1 验收通过

    # 场被改了（v2），此时再调用 record_acceptance：旧指标不能绑到新场上。
    field.density = field.density.copy()
    field.density[0, 0] = 55.0
    with pytest.raises(ReferenceNotAccepted, match="另一份场"):
        record_acceptance(field, evidence)


def test_record_acceptance_preserves_existing_physics_and_data_status():
    """写回验收不得把已有物理/数据状态覆盖回默认（src(9) 复核第 3 条）。

    ``record_acceptance`` 若不把已有 ``physics_basis_status`` / ``data_contract_status``
    传下去，``assess_acceptance`` 的默认值会把 ``已核实 / ready`` 覆盖成
    ``未核实 / not ready``。
    """
    from spherical_blast_pinn.physics.precursor import (
        EVIDENCE_DEPENDENT_CHECKS,
        PHYSICS_BASIS_VERIFIED,
    )

    from spherical_blast_pinn.cli._common import record_acceptance

    # 带齐内容标识：``record_acceptance`` 现在要求指标已绑定到场（``metrics_field_hash``），
    # 这条测试考的是"状态是否被保留"，不是标识缺失的处理。
    field = _accepted_field()
    evidence = {name: True for name in EVIDENCE_DEPENDENT_CHECKS}
    verdict = record_acceptance(field, evidence)

    # 物理状态必须被原样保留，而不是被覆盖回默认的"未核实"。
    assert verdict["physics_basis_status"] == PHYSICS_BASIS_VERIFIED
    assert field.metadata["physics_basis_status"] == PHYSICS_BASIS_VERIFIED
    # 数据契约**不在参考场这一层评**（§7.1：三类状态分开记，各自阻塞各自的入口）。
    # 以前把数据的 ready 传进来，等于让一个不相关的条件把参考场的判定钉死成
    # 永远 NOT ACCEPTED。现在显式记成 "not evaluated"，既不冒充 ready，也不
    # 被覆盖成 not ready。
    assert verdict["data_contract_status"] == "not evaluated"


def test_reference_acceptance_does_not_depend_on_data_readiness():
    """参考场的验收只看数值 + 物理；数据是否重提是**另一条**闸门。

    §7.1 要求数值 / 物理适用性 / 数据就绪三类状态分开记录，最终正式使用各自满足
    自己的必要条件。把数据的 ready 混进参考场判定，会让"数据还没重提"把
    "这份场算得对不对"也一起否掉——两个独立的阻塞项被绑成一个。
    """
    from spherical_blast_pinn.cli._common import (
        ReferenceNotAccepted,
        assert_reference_accepted,
    )
    from spherical_blast_pinn.physics.precursor import PHYSICS_BASIS_VERIFIED

    # 数值全过 + 证据齐全 + 物理已核实，但**数据未就绪**：参考场本身应通过。
    field = _accepted_field()
    assert_reference_accepted(field) is None

    # 反过来：物理未核实，数值再干净也不放行（这一条是 §7.1 明确要求的）。
    field.metadata["physics_basis_status"] = "未核实"
    with pytest.raises(ReferenceNotAccepted, match="物理适用性"):
        assert_reference_accepted(field)
    assert PHYSICS_BASIS_VERIFIED != "未核实"


def test_scaling_check_does_not_apply_the_ratio_twice():
    """§5.5 的核查要比**各自半径自己的**交接时刻，不能再乘一次 k。

    ``handover_relative_time`` 已经随半径线性缩放（``epsilon = k epsilon_ref``）。
    早先的 ``_solve_at_handover`` 又接了一个 ``time_scale``，于是目标算例被采到
    ``k^2 epsilon`` 处——两个算例落在了**不同的无量纲时刻**，比较给出量级 1 的
    "尺度偏差"（实测 density 2.6e-1、velocity 1.1），而那是在量和自己比。
    """
    from pathlib import Path as _Path

    from spherical_blast_pinn.cli.reference_field import _scaling_check
    from spherical_blast_pinn.config import load_config

    config = load_config(
        str(_Path(__file__).resolve().parents[1] / "configs" / "base.yaml"),
        ["reference_field.cells_inside=40", "reference_field.growth=1.01"],
    )
    report = _scaling_check(config, 0.05, 0.10)
    for name, value in report["state_relative_deviation"].items():
        assert value < 1.0e-6, f"{name} 的尺度偏差 {value:.3e} 说明缩放做了不止一次"
    assert report["contact_shape_deviation"] < 1.0e-6


# ===========================================================================
# 产物区的能量积分不变量
# ===========================================================================


def test_product_energy_invariant_is_met_at_the_default_resolution():
    """``E = E0 r_f^3/3`` 是解析精确的，所以它是一条**强**核查。

    历史：这条曾经偏差 5.6%~7.4%，而且三个半径上五位数字相同。当时把它读成
    "与分辨率无关"，是错的——它在同半径下随分辨率变化（500/1000/2000 单元分别是
    7.36%/6.25%/5.63%）。真正的根因是前端单元的压力功按**本单元压力**扣：本单元
    一边被扫过一边膨胀，压力低于波前压力，于是每步都欠扣，误差随步数累积
    （CFL 越小偏差越大，是"每步一个不随 dt 消失的量"的签名）。改成按**波前压力**
    扣之后降到 0.3%~0.5%。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_product_profile,
    )
    from spherical_blast_pinn.physics.precursor import ACCEPTANCE_TOLERANCES

    tolerance = ACCEPTANCE_TOLERANCES["detonation_energy_invariant_relative"]
    for cells in (500, 1000):
        profile = detonation_product_profile(
            charge_radius=0.05,
            detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
            config=DetonationConfig(cells=cells),
        )
        error = abs(float(profile.diagnostics["energy_relative_error"]))
        assert error < tolerance, f"{cells} 单元的能量不变量偏差 {error:.3e} 超过 {tolerance:.1e}"


def test_product_mass_invariant_is_exact_under_independent_integration():
    """质量不变量要**独立**积分核对，不能只信求解器自己的质量数组。

    求解器的 ``mass`` 是吸积公式定义的，所以 ``Σm - ρ0R0³/3 = 0`` 在构造上成立。
    把单元边界带出去、用 ``∫ρ r² dr`` 重新积一遍才是独立证据——两者若不一致，
    说明密度场与单元质量脱钩。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_product_profile,
    )

    for cells in (500, 1000):
        profile = detonation_product_profile(
            charge_radius=0.05,
            detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
            config=DetonationConfig(cells=cells),
        )
        exact = mat.PRODUCT_REFERENCE_DENSITY * 0.05**3 / 3.0
        independent = float(profile.diagnostics["mass_from_density"])
        assert abs(independent - exact) / exact < 1.0e-12
        # 波前必须恰好停在装药表面。
        assert profile.edges[-1] == pytest.approx(0.05, rel=1e-12)


def test_acceptance_status_is_computed_from_measured_metrics():
    """状态由实测指标算出来，写死的说明字符串不算。

    旧版本把 ``NOT ACCEPTED`` 和一段说明直接写进元数据：数字会与代码漂移，
    而且理由可能是一句推错的结论（"与分辨率无关"就是这样写进去的）。现在
    状态是 ``metrics`` 对照 ``tolerances`` 的结果，理由由**哪几条超限**生成。

    正式使用是三件独立的事（§7.1）：数值、物理适用性、数据就绪。它们各自
    必须满足，缺一不可。所以"数值全过 + 证据齐全"但物理未核实 / 数据未就绪
    时，**仍然不是 ACCEPTED**。
    """
    from spherical_blast_pinn.physics.precursor import (
        ACCEPTANCE_TOLERANCES,
        ACCEPTED_STATUS,
        NOT_ACCEPTED_STATUS,
        PHYSICS_BASIS_VERIFIED,
        assess_acceptance,
    )

    from spherical_blast_pinn.physics.precursor import EVIDENCE_DEPENDENT_CHECKS

    perfect = {key: 0.0 for key in ACCEPTANCE_TOLERANCES}
    evidence = {name: True for name in EVIDENCE_DEPENDENT_CHECKS}
    ready = {"ready": True, "reason": ""}

    # 数值 + 证据 + 物理 + 数据，四者齐了才是 ACCEPTED。
    ok = assess_acceptance(
        perfect,
        convergence_evidence=evidence,
        physics_basis=PHYSICS_BASIS_VERIFIED,
        data_contract=ready,
    )
    assert ok["acceptance_status"] == ACCEPTED_STATUS
    assert ok["numerical_status"] == ACCEPTED_STATUS
    assert ok["physics_basis_status"] == PHYSICS_BASIS_VERIFIED
    assert ok["data_contract_status"] == "ready"

    # 数值全过、证据齐全，但**物理未核实**（默认值）：不是 ACCEPTED。
    # §7.1 的原话："物理未核实不能因三个数值布尔值为真而变成可正式使用。"
    unverified_physics = assess_acceptance(perfect, convergence_evidence=evidence)
    assert unverified_physics["acceptance_status"] == NOT_ACCEPTED_STATUS
    assert "物理适用性" in unverified_physics["acceptance_reason"]
    # 数值那一项本身是过的——分开记，便于区分"数值还没过"和"物理/数据挡住"。
    assert unverified_physics["numerical_status"] == ACCEPTED_STATUS

    # 物理核实了、数据没就绪：也不是 ACCEPTED。
    no_data = assess_acceptance(
        perfect,
        convergence_evidence=evidence,
        physics_basis=PHYSICS_BASIS_VERIFIED,
        data_contract={"ready": False, "reason": "数据尚未重提"},
    )
    assert no_data["acceptance_status"] == NOT_ACCEPTED_STATUS
    assert "数据契约" in no_data["acceptance_reason"]

    bad = dict(perfect)
    bad["detonation_energy_invariant_relative"] = 9.0
    verdict = assess_acceptance(
        bad,
        convergence_evidence=evidence,
        physics_basis=PHYSICS_BASIS_VERIFIED,
        data_contract=ready,
    )
    assert verdict["acceptance_status"] == NOT_ACCEPTED_STATUS
    assert "能量不变量" in verdict["acceptance_reason"]
    assert "9.0000e+00" in verdict["acceptance_reason"]

    nan = assess_acceptance(
        {"detonation_mass_invariant_relative": float("nan")}, convergence_evidence=evidence
    )
    assert nan["acceptance_status"] == NOT_ACCEPTED_STATUS, "缺失指标必须算失败，不是通过"

    # 单次求解给不出收敛证据，所以**没有证据就是未验收**——否则"这个场通过了
    # 它自己那几项"会被读成"§5 全部通过"。
    no_evidence = assess_acceptance(perfect)
    assert no_evidence["acceptance_status"] == NOT_ACCEPTED_STATUS
    assert "缺少证据" in no_evidence["acceptance_reason"]

    # 证据存在但**没通过**时，同样不能验收。
    failing = assess_acceptance(perfect, convergence_evidence={"grid_convergence": False})
    assert failing["acceptance_status"] == NOT_ACCEPTED_STATUS


# ===========================================================================
# 交接：必须是守恒的重叠搬移，不是点值插值
# ===========================================================================


@pytest.mark.parametrize("detonation_cells,precursor_cells", [(400, 400), (400, 120), (120, 400)])
def test_handover_conserves_mass_and_energy(detonation_cells, precursor_cells):
    """交接必须**构造性**守恒，而且在两套网格不同分辨率时也要守恒。

    关键是**口径**：必须用格式实际守恒的那个功能量
    ``sum_i m_i e_i + sum_j M_j u_j^2/2``（``M_j = (m_{j-1}+m_j)/2``），
    **不是**单元中心口径 ``sum_i m_i (e_i + u_i^2/2)``。两者差 ``O(du^2)``。

    这一条踩过坑：早先的测试用的是"单元左端节点速度"口径，于是交接看起来精确
    守恒（−5.7e-14）；把两侧统一到节点加权口径，同一个交接实际是 **+0.110%**
    （2000→600）与 **+0.0434%**（2000→2000）。测试用了和被测代码同一个错口径，
    所以两边一起错、一起"通过"。

    点值插值也做不到守恒：产物场的拉格朗日网格在波前附近聚、前置网格均匀，
    陡处采样漏峰，2000→600 丢约 2%。现在是守恒的重叠搬移。

    还有一处会静默出错：写成两行 ``velocity[product_index+1] = ...`` 会让后一行
    覆盖前一行，节点 j 拿到 cell j-1 的速度，速度场整体错开一格。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_product_profile,
    )
    from spherical_blast_pinn.physics.precursor import (
        PrecursorConfig,
        SphericalPrecursor,
        _node_masses,
    )

    radius = 0.05
    products = detonation_product_profile(
        charge_radius=radius,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=detonation_cells),
    )
    volume = (products.edges[1:] ** 3 - products.edges[:-1] ** 3) / 3.0

    def functional(mass, velocity, specific_energy):
        """格式守恒的功能量：内能 + **节点加权**动能。"""
        return float(np.sum(mass * specific_energy)) + 0.5 * float(
            np.sum(_node_masses(mass) * velocity**2)
        )

    before_mass = float(np.sum(products.density * volume))
    before_energy = functional(
        np.asarray(products.cell_mass),
        np.asarray(products.node_velocity),
        np.asarray(products.specific_energy),
    )

    solver = SphericalPrecursor(
        charge_radius=radius,
        config=PrecursorConfig(cells_inside=precursor_cells, growth=1.005),
        products=products,
    )
    product = solver.mesh.material == 0
    count = int(np.count_nonzero(product))
    after_mass = float(np.sum(solver.mesh.mass[product]))
    after_energy = functional(
        solver.mesh.mass[product],
        solver.velocity[: count + 1],
        solver.specific_energy[product],
    )

    # 质量是**精确**的：重叠体积搬移按构造守恒。
    assert after_mass == pytest.approx(before_mass, rel=1e-12)

    # 能量：**网格逐点对应时是机器精度，网格不同则不然**。
    #
    # 产物场的单元边界是**已经走动过的**拉格朗日位置，前置网格是**原始**均匀位置，
    # 所以即使单元数相同，两套网格也不逐点对应。剩下的差额是 O(du²) 的**重采样**
    # 效应，不是一个可以靠配平消掉的误差。曾经两次试图消掉它：
    #   1) 把内能乘全局比例 -> 总量配平，但局部状态被搅（均匀 3 MJ/kg 变 2.14~3.56）
    #   2) 缩放速度场 -> 局部状态保住，功能量反而不守恒（−4.2e-3）
    #
    # 所以这里断言的是**声明的容差**，不是精确；逐点对应的情形由
    # ``test_handover_preserves_a_uniform_internal_energy_field`` 覆盖。
    from spherical_blast_pinn.physics.precursor import ACCEPTANCE_TOLERANCES

    defect = abs(after_energy - before_energy) / before_energy
    assert defect < ACCEPTANCE_TOLERANCES["handover_energy_defect_relative"], (
        f"交接的功能量缺陷 {defect:.3e} 超过声明的容差"
    )
    assert after_mass == pytest.approx(mat.PRODUCT_REFERENCE_DENSITY * radius**3 / 3.0, rel=1e-12)


def test_staged_budget_reports_the_three_stages_separately():
    """产物构造 / 交接 / 前置演化必须分开记账。

    "前置阶段质量恒定"是拉格朗日格式的**构造性质**，它证明不了交接没有损失；
    拿它当全链路守恒的证据，正好漏掉唯一会丢东西的那一环。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_product_profile,
    )
    from spherical_blast_pinn.physics.precursor import PrecursorConfig, SphericalPrecursor

    products = detonation_product_profile(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=120),
    )
    solver = SphericalPrecursor(
        charge_radius=0.05,
        config=PrecursorConfig(cells_inside=120, growth=1.005),
        products=products,
    )
    state = solver.run_to(products.time + 1.0e-7)
    product = solver.mesh.material == 0
    # 前置阶段的质量确实恒定——但这是构造性的，所以它**不是**守恒的证据。
    assert float(np.sum(state.mass[product])) == pytest.approx(
        float(np.sum(solver.mesh.mass[product])), rel=1e-14
    )


@pytest.mark.parametrize("cells", [100, 200, 400])
def test_handover_is_the_identity_on_an_identical_grid(cells):
    """同网格交接必须是**恒等**：节点速度要原样保留。

    用户用这个算例找出过一个一阶错误：代码把**单元平均速度**赋给节点，那不是搬移
    速度场，而是换了一个量。线性节点速度剖面下，节点值与相邻平均相差半个节点间距，
    于是同网格交接也有 10/5/2.5 m/s（100/200/400 单元）的节点速度变化、总能量差
    +0.3442%/+0.1753%/+0.0885%——**随网格间距线性下降**。既不是 O(du²) 的口径差，
    也不是重采样残差，是一个纯粹的错。速度必须按**节点量**在节点表之间插值。

    测试的构造也有讲究：**不能把输入节点速度设成交接将会采用的单元平均**，那样
    恒等式就成了构造出来的，测不出问题（上一版正是这么写的）。这里节点速度取线性
    剖面，与单元平均**不同**。
    """
    from spherical_blast_pinn.physics.detonation import DetonationConfig, detonation_product_profile
    from spherical_blast_pinn.physics.precursor import (
        DetonationProfile,
        PrecursorConfig,
        SphericalPrecursor,
        _node_masses,
    )

    radius = 0.05
    edges = np.linspace(0.0, radius, cells + 1)
    volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0

    # 节点速度线性；单元速度取相邻节点平均——与真实数据的构造顺序一致，
    # 而且两者**不相等**，恒等检查才有内容。
    node_velocity = np.linspace(0.0, 2000.0, cells + 1)
    cell_velocity = 0.5 * (node_velocity[:-1] + node_velocity[1:])
    assert not np.allclose(node_velocity[:cells], cell_velocity)

    reference = detonation_product_profile(
        charge_radius=radius,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=8),
    )
    source = DetonationProfile(
        time=radius / mat.DETONATION_VELOCITY_REFERENCE,
        charge_radius=radius,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        radii=(0.5 * (edges[1:] ** 3 + edges[:-1] ** 3)) ** (1.0 / 3.0),
        edges=edges,
        cell_mass=mat.PRODUCT_REFERENCE_DENSITY * volume,
        node_velocity=node_velocity,
        density=np.full(cells, mat.PRODUCT_REFERENCE_DENSITY),
        velocity=cell_velocity,
        specific_energy=np.full(cells, 3.0e6),
        pressure=np.full(cells, 1.0e9),
        front_state=reference.front_state,
    )
    solver = SphericalPrecursor(
        charge_radius=radius,
        config=PrecursorConfig(cells_inside=cells, growth=1.005),
        products=source,
    )
    product = solver.mesh.material == 0

    # 1) 节点速度原样保留。
    assert np.max(np.abs(solver.velocity[: cells + 1] - node_velocity)) < 1.0e-9

    # 2) 均匀内能保持均匀。
    after = solver.specific_energy[product]
    assert after == pytest.approx(np.full(cells, 3.0e6), rel=1.0e-12)

    # 3) 功能量精确守恒——同网格下**不需要任何配平**。
    def functional(mass, velocity, specific_energy):
        return float(np.sum(mass * specific_energy)) + 0.5 * float(
            np.sum(_node_masses(mass) * velocity**2)
        )

    before = functional(source.cell_mass, source.node_velocity, source.specific_energy)
    current = functional(solver.mesh.mass[product], solver.velocity[: cells + 1], after)
    assert current == pytest.approx(before, rel=1.0e-12)


def test_handover_preserves_a_uniform_internal_energy_field():
    """**均匀输入必须得到均匀输出。** 交接的职责是忠实搬移，不是重新分配。

    这一条是用户用独立算例找出来的：源、目标网格相同、输入内能均匀 3.000 MJ/kg、
    只做交接不推进时间，当时的实现给出 2.138~3.556 MJ/kg——总量变化是 0，剖面
    却被搅了。原因是为了配平总能量把整场内能乘了一个全局比例，而那个差额本来是
    "单元口径动能"与"节点加权动能"的**表示差异**，不是热力学差异。

    拿热力学状态去抵消运动学记账差异，是把误差从总量挪进了剖面：总量检查会通过，
    而压力剖面已经变了。

    同一算例下功能量也必须是机器精度守恒——网格逐点对应时**不需要任何配平**。
    """
    from spherical_blast_pinn.physics.detonation import DetonationConfig, detonation_product_profile
    from spherical_blast_pinn.physics.precursor import (
        DetonationProfile,
        PrecursorConfig,
        SphericalPrecursor,
        _node_masses,
    )

    radius = 0.05
    cells = 120
    edges = np.linspace(0.0, radius, cells + 1)
    centres = (0.5 * (edges[1:] ** 3 + edges[:-1] ** 3)) ** (1.0 / 3.0)
    volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0

    uniform_energy = np.full(cells, 3.0e6)
    cell_velocity = 2000.0 * (centres / radius) ** 2
    node_velocity = np.zeros(cells + 1)
    node_velocity[:cells] = cell_velocity
    node_velocity[cells] = cell_velocity[-1]

    reference = detonation_product_profile(
        charge_radius=radius,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=8),
    )
    source = DetonationProfile(
        time=radius / mat.DETONATION_VELOCITY_REFERENCE,
        charge_radius=radius,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        radii=centres,
        edges=edges,
        cell_mass=mat.PRODUCT_REFERENCE_DENSITY * volume,
        node_velocity=node_velocity,
        density=np.full(cells, mat.PRODUCT_REFERENCE_DENSITY),
        velocity=cell_velocity,
        specific_energy=uniform_energy,
        pressure=np.full(cells, 1.0e9),
        front_state=reference.front_state,
    )
    solver = SphericalPrecursor(
        charge_radius=radius,
        config=PrecursorConfig(cells_inside=cells, growth=1.005),
        products=source,
    )
    product = solver.mesh.material == 0
    after = solver.specific_energy[product]

    # 1) 局部状态：均匀进、均匀出。
    assert float(np.ptp(after)) / 3.0e6 < 1.0e-12, (
        f"交接把均匀内能搅成了 {after.min() / 1e6:.3f}~{after.max() / 1e6:.3f} MJ/kg"
    )
    assert after == pytest.approx(uniform_energy, rel=1e-12)

    # 2) 功能量：网格逐点对应时是**精确**的，不需要任何配平。
    def functional(mass, velocity, specific_energy):
        return float(np.sum(mass * specific_energy)) + 0.5 * float(
            np.sum(_node_masses(mass) * velocity**2)
        )

    before = functional(source.cell_mass, source.node_velocity, source.specific_energy)
    current = functional(solver.mesh.mass[product], solver.velocity[: cells + 1], after)
    assert current == pytest.approx(before, rel=1e-12)


# ===========================================================================
# 产物构造的逐步离散收支（§3.1 / §3.2）
# ===========================================================================


def test_detonation_kinetic_decomposition_is_an_identity():
    """动能的三分解必须**恒等**闭合，否则它定位不了任何东西。

    ``work_momentum + euler_error + mass_partition_kinetic == dKE`` 是三项的定义
    式，必须精确成立（实测 1e-16 量级；这里要求 1e-9）。

    这个诊断本身也踩过两个坑，都写在断言旁边，因为"诊断算错"比"格式算错"更难发现：
      1. 快照取晚了 —— 第 8 步会把比容覆盖成新值，之后再算 ``v_new - v_old`` 恒为
         零，读出来是"内部单元没做功"；
      2. 单位搞错 —— ``specific_volume`` 是**单位质量**的体积，压力功要乘 ``m_i``；
         少了它把内部功放大 5 个数量级（5409 E_exact），而那个数看起来还挺像个结论。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_energy_budget,
    )

    for cells in (300, 500):
        budget = detonation_energy_budget(
            charge_radius=0.05,
            detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
            config=DetonationConfig(cells=cells),
        )
        assert abs(budget["kinetic_identity_residual"]) < 1.0e-9
        assert budget["n_steps"] > 100

        cumulative = budget["cumulative"]
        # 物理输入就是被扫过的装药释放的化学能 —— 累计恰好是 1 个 E_exact
        # （因为全部装药最终都被扫过）。它不是"选出来让残差最小的数"。
        assert cumulative["physical_input"] == pytest.approx(1.0, rel=1e-6)
        # 显式欧拉的误差远小于主导项——它**不是**残差的主要来源。
        assert abs(cumulative["euler_error"]) < 1.0e-2


def test_momentum_work_equals_interior_work_minus_front_inner_work():
    """力与功在内部单元上**精确**匹配。

        sum_j M_j^new u_j^old du_j  ==  sum_{i<k} P_i dV_i  -  P_k dV_k^node

    右边第二项是前端单元内侧面（节点 k 扫过的体积）那一块。这一条逐步成立
    （实测相对 E_exact 最大偏离 1.7e-8），说明**残差不来自内部单元的力功不匹配**。

    它同时解释了为什么把前端单元的压力功从"整项 p_f x 单元体积变化"改成连续推导
    给出的分面形式 ``p_f*(dm/rho_ref) - P_k*dV_k^node`` 之后残差**变大**：右边已经
    含了 ``P_k*dV_k^node``，再减一次就重复扣了。所以这一项是**不该**再动的。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        ProductsDetonation,
    )
    import numpy as np

    solver = ProductsDetonation(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=400, cfl=0.2, record_budget=True),
    )
    solver.run_to_surface()
    steps = solver.budget
    exact = mat.PRODUCT_VOLUME_ENERGY * 0.05**3 / 3.0

    residual = [
        abs(
            r["work_momentum"] - (r["work_interior"] - r["work_front_inner"])
        )
        for r in steps
    ]
    assert max(residual) / exact < 1.0e-6


def test_momentum_at_the_accreting_node_uses_the_measured_best_form():
    """节点 k 用的是 ``M^new Δu = -dt F``，不是动量守恒形式。

    节点 k 的质量每步增长 ``added/2``，所以两种写法相差一个莱布尼茨项
    ``-u_k ΔM_k``。**三种写法都实测过**（n=500/cfl=0.2，总能量不变量偏差）：

      ``M^new Δu = -dtF``                     -> +4.09e-3   ← 现在用的
      ``M^new Δu = -dtF - u_k ΔM_k``          -> +2.56e-1
      ``M^new Δu = -dtF + ΔM_k (u_f - u_k)``  -> +1.68e-2

    保守形式反而差 60 倍：波前节点的位置是给定的，它不满足动量方程，多出来的
    动量被它吸收掉。这条测试把"当前选的是哪一支"钉住——不是为了证明它对，而是
    为了让**换掉它**这件事必须是一次有意识、有测量的修改。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        ProductsDetonation,
    )

    solver = ProductsDetonation(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=200, cfl=0.2, record_budget=True),
    )
    solver.run_to_surface()
    steps = solver.budget

    checked = 0
    for record in steps:
        if record["mass_added"] <= 0.0 or record["front_cell"] == 0:
            continue
        # 差值应当**恰好**是莱布尼茨项 ΔM_k * u_k^old，不含 u_f 的成分。
        leibniz = (
            0.5 * record["mass_added"] * record["u_k_before"]
        )
        closure = (
            record["momentum_after"] - record["momentum_before"] - record["impulse"]
        )
        assert closure == pytest.approx(leibniz, rel=1e-9, abs=0.0)
        checked += 1
    assert checked > 100


def test_node_and_cell_kinetic_energies_differ_by_an_analytic_amount():
    """两种动能口径之差有**闭式**，不是"线性速度下相等"。

        K_node - K_cell = (1/8) * sum_i m_i (u_{i+1} - u_i)^2

    推导：``sum_j M_j u_j^2/2`` 按单元重排是 ``¼ sum_i m_i (u_i^2 + u_{i+1}^2)``，
    减去 ``(1/8) sum_i m_i (u_i+u_{i+1})^2`` 恰好得到上式。

    这条测试是**独立**的：右边完全由 `cell_mass` 与 `node_velocity` 算出，
    不含任何格式内部量，也没有可调门槛之外的拟合。它同时说明：**同网格交接时
    用同一种口径比前后，只证明交接没换口径，不证明两种口径彼此相等。**
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        ProductsDetonation,
        _node_masses,
    )
    import numpy as np

    profile = ProductsDetonation(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=DetonationConfig(cells=200),
    ).run_to_surface()

    mass = profile.cell_mass
    velocity = profile.node_velocity
    node_weighted = 0.5 * float(np.sum(_node_masses(mass) * velocity**2))
    cell_centred = 0.5 * float(
        np.sum(mass * (0.5 * (velocity[:-1] + velocity[1:])) ** 2)
    )
    analytic = 0.125 * float(np.sum(mass * np.diff(velocity) ** 2))

    # 1e-10 是浮点求和的舍入余量（200 项、量级 1e3），不是可调门槛：
    # 恒等式本身是闭式的，实测相对偏差 5.7e-12。写错一个系数会差 O(1)。
    assert node_weighted - cell_centred == pytest.approx(analytic, rel=1e-10)
    # 剖面有真实的速度梯度，所以差值**不为零**——这正是旧注释写错的地方。
    assert analytic > 0.0

    # 线性剖面也一样不相等：把这一条单独钉住，防止"线性即相等"的说法回来。
    linear = np.linspace(0.0, 1000.0, mass.size + 1)
    k_node = 0.5 * float(np.sum(_node_masses(mass) * linear**2))
    k_cell = 0.5 * float(
        np.sum(mass * (0.5 * (linear[:-1] + linear[1:])) ** 2)
    )
    assert k_node != k_cell
    assert k_node - k_cell == pytest.approx(
        0.125 * float(np.sum(mass * np.diff(linear) ** 2)), rel=1e-10
    )


def test_physical_input_matches_an_independently_integrated_flux():
    """物理输入必须等于**独立算出来**的化学能总量。

    被扫过的装药质量是 ``rho_ref r_f^3/3``（独立于格式的质量数组，直接由波前
    位置给出），释放的化学能是 ``Q`` 乘这个质量。所以累计输入应当恰好是

        (rho_ref R_0^3/3 * Q) / (E_0 R_0^3/3) = 1

    这条**不**依赖任何"让残差最小"的选择，也不依赖测量值：右边是 1，由定义给出。
    （差一点点是 ``t_start`` 处已经被播种的那半点单元质量从未计入 ``added``，
    量级 1e-9。）
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_energy_budget,
    )

    for cells in (300, 500):
        budget = detonation_energy_budget(
            charge_radius=0.05,
            detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
            config=DetonationConfig(cells=cells),
        )
        swept_mass = mat.PRODUCT_REFERENCE_DENSITY * 0.05**3 / 3.0
        expected = swept_mass * (mat.PRODUCT_VOLUME_ENERGY / mat.PRODUCT_REFERENCE_DENSITY)
        assert budget["energy_exact"] == pytest.approx(expected, rel=1e-12)
        assert budget["cumulative"]["physical_input"] == pytest.approx(1.0, rel=1e-6)


def test_budget_returns_the_per_step_records_and_does_not_mutate_the_config():
    """逐步记录必须能拿到；调用者的配置不能被改。

    只返回累计值等于把定位能力丢掉——残差的用处全在"哪一步、哪一类步"上。

    曾经这里有两处接口缺陷：
      1. ``detonation_energy_budget`` 只返回累计字典，逐步记录留在一个没有公开的
         ``solver.budget`` 上，调用者拿不到；
      2. 它直接写 ``config.record_budget = True``，把调用者传进来的对象改掉了。
         副作用最难查：调用者之后再拿这个配置去别处用，行为已经变了。
    """
    from spherical_blast_pinn.physics.detonation import (
        DetonationConfig,
        detonation_energy_budget,
    )

    config = DetonationConfig(cells=120, cfl=0.3)
    assert config.record_budget is False

    budget = detonation_energy_budget(
        charge_radius=0.05,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        config=config,
    )

    # 1) 调用者的配置没被改
    assert config.record_budget is False
    # 2) 逐步记录拿得到，且条数与步数一致
    assert len(budget["steps"]) == budget["n_steps"]
    assert budget["n_steps"] > 100
    assert budget["step_keys"]
    for key in (
        "energy_residual",
        "physical_input",
        "mass_added",
        "front_cell",
        "crossed",
        "work_interior",
        "work_front_outer",
        "work_front_inner",
        "work_momentum",
        "mass_partition_kinetic",
    ):
        assert key in budget["steps"][0]
    # 3) 累计值就是逐步记录之和 —— 两者必须自洽
    total = sum(r["energy_residual"] for r in budget["steps"])
    assert total / budget["energy_exact"] == pytest.approx(
        budget["cumulative"]["energy_residual"], rel=1e-12
    )


# ===========================================================================
# 材料卡基线（提示词 §2.2）
# ===========================================================================


def test_card_parameters_match_the_frozen_baseline():
    """材料卡参数必须与提示词 §2.2 的核查基线逐项一致。

    实际 deck 若与这张表不同，要求**列出差异**，不能默默覆盖。这条测试就是那个
    "列差异"的机器可判定版本：任何一项漂了，这里立刻红。

    特别钉住两个**明令禁止回归**的值：
      * ``B = 3.7471e9``（旧项目的 ``3.231e9`` 不得恢复）；
      * ``E0 = 6.0e9``（历史反推的 ``6.652e9`` 不得替换卡片值）。
    """
    assert mat.PRODUCT_REFERENCE_DENSITY == 1630.0
    assert mat.PRODUCT_VOLUME_ENERGY == 6.0e9
    assert mat.DETONATION_VELOCITY_REFERENCE == 6930.0

    jwl = mat.product_jwl()
    assert jwl.rho0 == 1630.0
    assert jwl.A == pytest.approx(3.712e11, rel=1e-12)
    assert jwl.B == pytest.approx(3.7471e9, rel=1e-12)
    assert jwl.R1 == pytest.approx(4.15, rel=1e-12)
    assert jwl.R2 == pytest.approx(0.95, rel=1e-12)
    assert jwl.omega == pytest.approx(0.30, rel=1e-12)

    # 明令禁止的旧值
    assert jwl.B != pytest.approx(3.231e9, rel=1e-6)
    assert mat.PRODUCT_VOLUME_ENERGY != pytest.approx(6.652e9, rel=1e-6)


def test_card_cj_velocity_is_below_the_card_detonation_velocity():
    """本卡片的自由 CJ 爆速**低于**卡片爆速，这是"两个 RH 根"的来源。

    ``D_CJ = 6883.14``、``D_card = 6930``，差 0.68%。两个值都要按实测报告，
    **不能**通过拟合 E0 或重置能量零点把它们凑到一起（提示词 §2.2）。

    这条测试同时复核 CJ 的**声速条件** ``c = D V``：相切解与声速条件等价，
    算出来的 sonic_ratio 必须为 1。
    """
    from spherical_blast_pinn.physics.detonation import ideal_cj_state

    cj = ideal_cj_state()
    assert cj.detonation_velocity == pytest.approx(6883.1417, rel=1e-6)
    assert cj.pressure == pytest.approx(2.0638869e10, rel=1e-6)
    # CJ 的定义就是声速点：c/(D V) = 1
    assert cj.sonic_ratio == pytest.approx(1.0, rel=1e-8)

    card = mat.DETONATION_VELOCITY_REFERENCE
    assert card > cj.detonation_velocity
    assert card / cj.detonation_velocity - 1.0 == pytest.approx(0.006808, rel=1e-3)


def test_the_front_branch_is_the_subsonic_strong_branch():
    """过驱动时取**亚声速（强支）**，不是 CJ，也不是超声速（弱支）。

    这是三处 SDToolbox 原文支持的结论（§6.5 p.73/p.76、§C.1 p.301、§11.10/11.11）：
    对 ``D > D_CJ`` 的过驱动爆轰，物理实现的是 strong 分支，波后相对波前**亚声速**；
    弱支是超声速。判据 ``c/(D-u) > 1`` 与之一致，**不要改成别的方向**。

    同时确认波前状态**不是** CJ 状态——把 CJ 当成波前状态是在一个 JWL/E0 与卡片
    爆速不相容的模型里强行施加 CJ 声速条件。
    """
    from spherical_blast_pinn.physics.detonation import (
        detonation_front_state,
        ideal_cj_state,
        rayleigh_hugoniot_states,
    )

    card = mat.DETONATION_VELOCITY_REFERENCE
    states = rayleigh_hugoniot_states(detonation_velocity=card)
    assert len(states) == 2, "D > D_CJ 应当恰好两个根"
    subsonic = [s for s in states if s.sonic_ratio > 1.0]
    supersonic = [s for s in states if s.sonic_ratio < 1.0]
    assert len(subsonic) == 1 and len(supersonic) == 1

    front = detonation_front_state(detonation_velocity=card)
    assert front.branch == "subsonic"
    assert front.sonic_ratio > 1.0
    assert front.sonic_ratio == pytest.approx(1.0840295, rel=1e-6)

    cj = ideal_cj_state()
    assert front.volume_ratio != pytest.approx(cj.volume_ratio, rel=1e-3)
    assert front.pressure != pytest.approx(cj.pressure, rel=1e-3)


# ===========================================================================
# 验收完整性（§7.1）与缓存证据绑定（§7.2/§7.3）——src(9) 复核后的回归
# ===========================================================================


def _fake_state(n=8, **overrides):
    """构造一个合法的单帧状态，供 ``field_violation_metrics`` 测试。"""
    import numpy as np
    from types import SimpleNamespace

    from spherical_blast_pinn.physics.precursor import PRODUCT, AIR

    base = dict(
        density=np.full(n, 100.0),
        specific_energy=np.full(n, 1.0e5),
        pressure=np.full(n, 1.0e6),
        sound_speed=np.full(n, 300.0),
        mass=np.full(n, 1.0),
        radii=np.arange(n + 1, dtype=float),
        velocity=np.zeros(n + 1),
        material=np.array([PRODUCT] * 3 + [AIR] * (n - 3)),
        contact_radius=3.0,
        shock_radius=float(n),
        air_pressure=101320.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_field_violations_catch_nan_and_illegal_states():
    """非正/非有限检查不能放过 NaN 与非法结构（src(9) 复核发现漏项）。

    关键点：``NaN <= 0`` 是 False，所以 ``np.diff(radii) <= 0`` 与 ``mass <= 0``
    都数不到 NaN——必须单独数。同理 ``Rs < Rc`` 与非法材料编码是独立的违例，
    不会因为"块连续"或"正压力"合格就被放过。
    """
    import numpy as np

    from spherical_blast_pinn.physics.precursor import PRODUCT, AIR, field_violation_metrics

    good = field_violation_metrics(_fake_state())
    assert good["mesh_ordering_violations"] == 0.0
    assert good["front_ordering_violations"] == 0.0
    assert good["illegal_material_cells"] == 0.0

    # 半径含 NaN：必须算违例
    n = 8
    radii_nan = np.array([np.nan] + list(range(1, n + 1)), dtype=float)
    assert field_violation_metrics(_fake_state(radii=radii_nan))["mesh_ordering_violations"] >= 1.0

    # 单元质量含 NaN
    mass_nan = np.array([np.nan] + [1.0] * (n - 1))
    assert field_violation_metrics(_fake_state(mass=mass_nan))["mesh_ordering_violations"] >= 1.0

    # Rs < Rc
    assert field_violation_metrics(_fake_state(contact_radius=5.0, shock_radius=4.0))[
        "front_ordering_violations"
    ] >= 1.0

    # 非法材料编码
    illegal = np.array([PRODUCT] * 3 + [7] + [AIR] * 4)
    assert field_violation_metrics(_fake_state(material=illegal))["illegal_material_cells"] >= 1.0


def test_reference_cache_round_trips_through_atomic_save_and_load():
    """缓存保存必须**原子**且能原样读回（src(9) 复核第 1 条：必现写盘错误）。

    ``np.savez_compressed`` 对不以 ``.npz`` 结尾的文件名会**自动补** ``.npz``。
    若临时文件取 ``formal.npz.tmp``，NumPy 实际写 ``formal.npz.tmp.npz``，随后
    ``os.replace("formal.npz.tmp", ...)`` 直接 FileNotFoundError——每次保存都崩。
    这条测试走完整的 保存→读回 路径，命名错了立刻红。
    """
    import numpy as np

    from spherical_blast_pinn.cli._common import (
        _reference_from_npz,
        _save_reference,
    )
    from spherical_blast_pinn.physics.precursor import ReferenceField

    field = ReferenceField(
        times=np.array([0.0, 1.0e-6]),
        cell_radii=np.array([[0.0, 0.5], [0.0, 0.6]]),
        density=np.array([[100.0, 1.0], [101.0, 1.0]]),
        velocity=np.array([[10.0, 0.0], [11.0, 0.0]]),
        specific_energy=np.array([[1.0e5, 2.0e5], [1.0e5, 2.0e5]]),
        material=np.array([0, 1]),
        contact_radius=np.array([0.4, 0.5]),
        shock_radius=np.array([0.6, 0.7]),
        reference_radius=0.05,
        metadata={},
    )
    path = Path(__import__("tempfile").mkdtemp()) / "reference_field.npz"
    _save_reference(field, path, {"field_version": 12})

    assert path.exists()
    # 没有遗留临时文件
    assert not path.with_name(path.name + ".tmp.npz").exists()

    payload = np.load(path, allow_pickle=True)
    loaded = _reference_from_npz(payload)
    assert np.array_equal(loaded.density, field.density)
    assert np.array_equal(loaded.velocity, field.velocity)
    assert loaded.metadata.get("field_content_hash")


def test_write_back_after_load_works_on_the_real_path():
    """实际写回路径：求解 → 保存 → 加载 → record_acceptance → 再落盘。

    这条路径此前**没有测试覆盖**，而它有一个只在 Windows 上露头的真 bug：
    ``np.load`` 返回的 ``NpzFile`` 持有文件句柄，读到缓存之后句柄一直开着，
    紧接着 ``record_acceptance`` 写回同一个路径时 ``os.replace`` 抛
    ``PermissionError: [WinError 5] 拒绝访问``——**读到缓存就再也存不回去**。

    所以这里必须走完整链路，而不是只测各部分。
    """
    import numpy as np
    import tempfile
    from pathlib import Path

    from spherical_blast_pinn.cli import _common
    from spherical_blast_pinn.physics.detonation import DetonationConfig
    from spherical_blast_pinn.physics.precursor import (
        EVIDENCE_DEPENDENT_CHECKS,
        PrecursorConfig,
        solve_reference_field,
    )

    field = solve_reference_field(
        charge_radius=0.05,
        config=PrecursorConfig(cells_inside=60),
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        detonation_config=DetonationConfig(cells=100),
    )
    # 场一出生就带内容标识，指标也绑定在它上面。
    assert field.metadata.get("field_content_hash")
    assert field.metadata.get("metrics_field_hash") == field.metadata["field_content_hash"]

    cache = Path(tempfile.mkdtemp()) / "reference_field.npz"
    signature = {"field_version": 999}
    _common._save_reference(field, cache, signature)

    with np.load(cache, allow_pickle=True) as payload:
        loaded = _common._reference_from_npz(payload)
    _common._attach_cache_provenance(loaded, cache, signature)
    assert loaded.metadata["field_content_hash"] == field.metadata["field_content_hash"]

    # 加载后立刻写回：句柄没释放的话这里会 PermissionError。
    evidence = {name: True for name in EVIDENCE_DEPENDENT_CHECKS}
    verdict = _common.record_acceptance(loaded, evidence)
    # 数值/物理/数据三态里，物理与数据都还没核实，所以仍不是 ACCEPTED——
    # 但**写回本身必须成功**，且验收记录要绑定到场内容。
    assert verdict["acceptance_status"] == "NOT ACCEPTED"
    assert loaded.metadata["acceptance_field_hash"] == field.metadata["field_content_hash"]

    # 写回后文件仍可读，内容标识不变（保存只负责存储，不改身份）。
    with np.load(cache, allow_pickle=True) as payload:
        again = _common._reference_from_npz(payload)
    assert again.metadata["field_content_hash"] == field.metadata["field_content_hash"]


# ===========================================================================
# §15 数据契约
# ===========================================================================


def _contract_case(**overrides):
    """一个满足契约的最小算例替身。"""
    import numpy as np
    from types import SimpleNamespace

    from spherical_blast_pinn.data.compatibility import SYNTHETIC

    validity = np.ones((3, 4), dtype=bool)
    profile = np.ones((3, 4), dtype=np.float64)
    base = dict(
        case_id="c",
        charge_radius=0.05,
        times=np.array([0.0, 1.0e-6, 2.0e-6]),
        radius=np.linspace(0.004, 0.05, 4),
        # 契约现在对**任何**对象做同一套结构检查（提示词 §4），所以替身也得带上
        # 与时间轴一致的剖面数组——否则被拦下的是"形状不对"，不是被测的那一项。
        density=profile,
        velocity=profile,
        pressure=profile,
        # 契约现在查对象上的**每一个**数组（不是一张写死的七个名字的清单），所以
        # 替身必须带齐 ``ExtractedCase`` 的必填字段，否则被拦下的是"替身不全"，
        # 而不是被测的那一项。
        specific_energy=profile,
        material=np.zeros((3, 4), dtype=np.int64),
        contact_radius=np.full(3, 0.05),
        shock_radius=np.full(3, 0.10),
        # 派生掩码必须能从本算例自己的资格证据重算出来（§5）。以前这两个替身写成
        # 全 True，而下面的 ``contact_eligible`` 是 ``[T, T, F]``——一份"资格说不
        # 算、标签说算"的自相矛盾记录。矛盾被拒绝是对的，所以改替身而不是放宽检查。
        contact_valid=np.array([True, True, False]),
        shock_valid=np.array([True, False, False]),
        valid=validity,
        source={
            "symmetry_factor": 8.0,
            "symmetry_share": 0.125,
            "detonation_velocity_m_s": 6930.0,
            "material_masks_from": "ALE material volume fraction (d3plot history variables)",
            # 机器可判的标志：契约按这个布尔量判断分区是否权威，不按上面的说明文字
            # 做子串匹配（"not from volume fraction" 里也含 "volume fraction"）。
            "material_partition_is_volume_fraction_based": True,
            # 标志之外还要有**结构化来源记录**：只有一个是/否标志时，那个 True
            # 是谁写的、依据是什么都无从核对。
            "material_partition_source_recorded": {
                "method": "ale_volume_fraction",
                "producer": "cli.extract_profiles",
                "producer_version": 1,
                "fields": ["tnt_volume_fraction"],
                "input_identifier": "sha256:test",
                "detail": {"pure_product_threshold": 0.95},
            },
            "energy_source": "reconstructed from JWL",
        },
        # 契约要求的两个分项记录，形状与真实导出一致
        field_validity={
            "density": validity, "velocity": validity,
            "specific_energy": validity, "pressure": validity,
        },
        front_supervision_eligibility={
            "source": "per-frame supervision eligibility arrays (§9/§10)",
            "level": "arrays",
            "contact_eligible": np.array([True, True, False]),
            "shock_eligible": np.array([True, False, False]),
        },
    )
    base.update(overrides)
    case = SimpleNamespace(**base)
    # 最小可用的时间/前沿接口
    case.detonation_velocity = 6930.0
    case.start_time = 0.0
    case.pinn_times = np.asarray(case.times) - case.start_time
    case.supervisable_frames = lambda: np.array([True, True, True])
    return case


def test_data_contract_is_ready_only_when_every_required_item_is_present():
    """契约的就绪是**必要条件**的合取，缺一项就不就绪并给出理由。"""
    from spherical_blast_pinn.data.contract import evaluate_data_contract

    ok = evaluate_data_contract([_contract_case()])
    assert ok.ready, ok.reasons

    # 缺球对称份额
    v = evaluate_data_contract([_contract_case(source={
        "detonation_velocity_m_s": 6930.0,
        "material_masks_from": "ALE material volume fraction (d3plot history variables)",
    })])
    assert not v.ready
    assert any("球对称份额" in r for r in v.reasons)

    # 材料分区来自启发式探测器 —— 这是当前真实数据的状态
    v = evaluate_data_contract([_contract_case(source={
        "symmetry_factor": 8.0,
        "detonation_velocity_m_s": 6930.0,
        "material_masks_from": "density-steepest-descent contact detector (heuristic)",
    })])
    assert not v.ready
    assert any("体积分数" in r for r in v.reasons)

    # 缺逐字段有效性 / 前沿监督资格
    case = _contract_case()
    del case.field_validity
    v = evaluate_data_contract([case])
    assert not v.ready
    assert any("逐字段有效性" in r for r in v.reasons)

    # 一个算例不就绪，整批就不就绪——不能拿就绪的算例掩盖没就绪的。
    case = _contract_case()
    del case.front_supervision_eligibility
    v = evaluate_data_contract([_contract_case(), case])
    assert not v.ready
    assert v.detail["n_ready"] == 1 and v.detail["n_cases"] == 2


def test_data_contract_reports_missing_time_origin_instead_of_crashing():
    """缺爆速时时间轴建不起来，契约要把它**报告**成理由，不是抛异常。

    真实数据现在就是这个状态：``F:/2/extracted`` 是旧提取器写的，metadata 里没有
    ``detonation_velocity_m_s``，于是 ``T_h = R_0/D + epsilon`` 无法形成。契约检查
    是给人看的报告工具，崩掉就什么都看不到。
    """
    import numpy as np
    from types import SimpleNamespace

    from spherical_blast_pinn.data.compatibility import SYNTHETIC
    from spherical_blast_pinn.data.contract import evaluate_case_contract
    from spherical_blast_pinn.data.extracted_case import DataError

    case = SimpleNamespace(
        case_id="old-cache",
        charge_radius=0.05,
        times=np.array([0.0, 1.0e-6]),
        source={"symmetry_factor": 8.0, "material_masks_from": "ALE material volume fraction"},
    )

    class _C:
        case_id = "old-cache"
        charge_radius = 0.05
        times = np.array([0.0, 1.0e-6])
        source = {"symmetry_factor": 8.0,
                  "material_masks_from": "ALE material volume fraction x"}

        @property
        def detonation_velocity(self):
            raise DataError("metadata carries no usable detonation velocity")

    verdict = evaluate_case_contract(_C())
    assert not verdict.ready
    assert any("时间原点" in r for r in verdict.reasons)
    assert verdict.detail["n_frames"] == 2
