"""第二轮审查（2026-09-22）逐条的回归测试。

这一轮的四条都指向同一类错误：**把"看起来对"当成"已经验证过"**。

* 比内能反解被说成"压力转负后没有实数解"——它其实是一次除法，永远有解；
* 激波位置测到的是远场数值噪声，不是激波；
* 2500 / 5000 层一致被当成"网格够用"，而它只说明提取对分箱不敏感。

所以下面每一条都用会让错误实现在同一个位置上给出**另一个合法数字**的构造。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.compatibility import (
    COMPATIBLE,
    INCOMPATIBLE,
    material_partition_verdict,
)
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.physics import material as mat


# ---------------------------------------------------------------------------
# 1. 比内能反解：除法，不是求根
# ---------------------------------------------------------------------------


def test_jwl_energy_inverse_is_a_division_not_a_root_solve():
    """``e = (p - p_cold)/(omega rho)`` 对有限的 ``p``、``rho > 0`` **永远**有实数值。

    实测 50 mm 算例 40 个状态 × 1220736 个单元，反解出非有限值的**一个都没有**。
    真正发生的是反解出**负值**，被下游的 ``value > 0`` 判据过滤成了 ``nan``。把那个
    ``nan`` 归因于"没有实数解"会让人去查求解器，而问题在过滤器的语义上。
    """
    from spherical_blast_pinn.data.raw_reader import reconstruct_product_energy
    from spherical_blast_pinn.physics import material as material_mod
    from spherical_blast_pinn.physics.eos import jwl_pressure_numpy

    jwl = material_mod.product_jwl()
    rho = np.array([1630.0, 1630.0, 700.0, 200.0, 20.0])
    p_cold = jwl_pressure_numpy(rho, np.zeros_like(rho), jwl)
    # 全部取在冷压曲线**之下**：这正是"还没释放化学能"的那一档。
    pressure = p_cold - 1.0e6

    energy = reconstruct_product_energy(rho, pressure, jwl)
    assert np.all(np.isfinite(energy)), "反解是除法，不可能给不出有限值"
    assert np.all(energy < 0.0), "冷压曲线之下必然反解出负比内能"
    np.testing.assert_allclose(energy * (jwl.omega * rho), pressure - p_cold, rtol=1e-12)


def test_energy_nan_downstream_is_the_positivity_filter_not_a_missing_solution():
    """下游那个 ``nan`` 必须能追到"负值被过滤"，而不是"解不出来"。"""
    from spherical_blast_pinn.data.extracted_case import specific_energy_from_pressure
    from spherical_blast_pinn.data.raw_reader import reconstruct_product_energy
    from spherical_blast_pinn.physics import material as material_mod
    from spherical_blast_pinn.physics.eos import jwl_pressure_numpy

    jwl = material_mod.product_jwl()
    rho = np.array([1630.0, 1630.0])
    p_cold = jwl_pressure_numpy(rho, np.zeros_like(rho), jwl)
    pressure = p_cold - 1.0e6
    material = np.array([0, 0], dtype=np.int64)

    raw = reconstruct_product_energy(rho, pressure, jwl)
    assert np.all(np.isfinite(raw)) and np.all(raw < 0.0)
    loader = specific_energy_from_pressure(rho, pressure, material, jwl=jwl)
    assert np.all(np.isnan(loader)), "非正比内能必须是 nan，不能是负数标签"
    # 把压力提到冷压曲线之上，同一个函数立刻给出有限正值——差别只在符号，
    # 不在"有没有解"。
    ok = specific_energy_from_pressure(rho, p_cold + 1.0e6, material, jwl=jwl)
    assert np.all(np.isfinite(ok)) and np.all(ok > 0.0)


def test_a_cell_below_the_cold_curve_gets_no_energy_label(monkeypatch):
    """没有产物状态的单元：密度/速度/压力照留，比内能置空，并逐单元标出来。"""
    from spherical_blast_pinn.data import raw_reader as rr
    from spherical_blast_pinn.physics.eos import jwl_pressure_numpy

    jwl = mat.product_jwl()
    density = np.array([1630.0, 1630.0, 1.225])
    p_cold = jwl_pressure_numpy(density[:2], np.zeros(2), jwl)
    pressure = np.array([p_cold[0] - 1.0e6, p_cold[1] + 1.0e6, mat.AIR_PRESSURE])

    product = rr.reconstruct_product_energy(density[:2], pressure[:2], jwl)
    assert np.all(np.isfinite(product))
    assert product[0] < 0.0 < product[1]
    # 有效性的判据就是"反解是否为正"，不是"反解是否存在"。
    valid = np.isfinite(product) & (product > 0.0)
    assert valid.tolist() == [False, True]
    air = rr.reconstruct_air_energy(density[2:], pressure[2:], mat.AIR_GAMMA)
    assert np.all(air > 0.0)


# ---------------------------------------------------------------------------
# 2. 激波定位：连通段的外缘，不是最外侧的超压壳层
# ---------------------------------------------------------------------------


def test_shock_radius_is_the_contiguous_run_not_the_outermost_threshold_cell():
    """**实测复现的那一条。**

    50 mm 算例第 10 状态：真实超压区外缘在 58.2 mm 附近，而远场噪声在 94–101 mm 处
    也偶尔超过 1% 阈值。旧的"最外侧超压壳层"判据因此测到 95.2 mm（2500 层）/
    101.0 mm（5000 层）——两套网格各测噪声的不同尖峰，看起来像"网格分辨率不够"。
    """
    from spherical_blast_pinn.cli.extract_profiles import _shock_radius

    centres = np.linspace(0.040, 0.20, 3201)
    pressure = np.full_like(centres, mat.AIR_PRESSURE)
    alpha = np.zeros_like(centres)
    run = (centres >= 0.052) & (centres <= 0.0582)
    pressure[run] = mat.AIR_PRESSURE * 50.0
    product = centres <= 0.052
    pressure[product] = 5.0e9
    alpha[product] = 1.0
    # 远场噪声：三根孤立尖峰，都超过 1% 阈值，都比真实激波靠外 40 mm 以上。
    for r in (0.0946, 0.0952, 0.1010):
        pressure[np.argmin(np.abs(centres - r))] = mat.AIR_PRESSURE * 1.014

    radius, diagnostics = _shock_radius(
        centres, pressure, alpha, ambient_pressure=mat.AIR_PRESSURE
    )
    assert abs(radius - 0.0582) < 5.0e-4, (
        f"shock radius {radius * 1e3:.3f} mm locked onto the far-field noise instead of "
        "the contiguous over-pressure run at 58.2 mm"
    )
    assert diagnostics["stopped_at_drop"] == 1.0


def test_shock_radius_says_why_it_found_nothing():
    """测不到和分辨不出是两件事，返回的诊断必须能区分。"""
    from spherical_blast_pinn.cli.extract_profiles import _shock_radius

    centres = np.linspace(0.040, 0.20, 3201)
    alpha = np.zeros_like(centres)
    alpha[centres <= 0.052] = 1.0
    pressure = np.where(centres <= 0.052, 5.0e9, mat.AIR_PRESSURE)

    radius, diagnostics = _shock_radius(
        centres, pressure, alpha, ambient_pressure=mat.AIR_PRESSURE
    )
    assert not np.isfinite(radius)
    assert diagnostics["peak_overpressure"] == pytest.approx(1.0)
    assert diagnostics["air_side_shells"] > 0


def test_shock_radius_ignores_isolated_holes_inside_the_run():
    """连接段内部的孤立低值不能被当成激波外缘。"""
    from spherical_blast_pinn.cli.extract_profiles import _shock_radius

    centres = np.linspace(0.040, 0.20, 3201)
    alpha = np.zeros_like(centres)
    alpha[centres <= 0.052] = 1.0
    pressure = np.where(centres <= 0.052, 5.0e9, mat.AIR_PRESSURE)
    run = (centres > 0.052) & (centres <= 0.0620)
    pressure[run] = mat.AIR_PRESSURE * 30.0
    hole = np.flatnonzero((centres > 0.056) & (centres < 0.0575))[:2]
    pressure[hole] = mat.AIR_PRESSURE * 0.999

    radius, _ = _shock_radius(centres, pressure, alpha, ambient_pressure=mat.AIR_PRESSURE)
    assert abs(radius - 0.0620) < 5.0e-4


# ---------------------------------------------------------------------------
# 3. 状态数：探针 + 边界复核，绝不退回文件数
# ---------------------------------------------------------------------------


def test_state_count_probe_never_falls_back_to_the_file_count(monkeypatch):
    """解析失败必须**停下**，不能退回"文件数＝状态数"。"""
    from spherical_blast_pinn.cli import extract_profiles as ep
    from spherical_blast_pinn.data import raw_reader as rr

    class _ArrayType:
        global_timesteps = "global_timesteps"

    class _D3plot:
        def __init__(self, path, state_filter=None, buffered_reading=True,
                     state_array_filter=None):
            raise ValueError("requested state is out of range")

    monkeypatch.setattr(rr, "_import_lasso", lambda: (_D3plot, _ArrayType))
    monkeypatch.setattr(rr, "lasso_version", lambda: "0.0.0-test")

    with pytest.raises(ep.ExtractionError) as excinfo:
        ep._resolve_state_count(Path("nowhere"), file_count=780)
    assert "file count" in str(excinfo.value)


def test_state_count_probe_verifies_its_own_answer(monkeypatch):
    """探针的结论要独立复核：报的数必须真的在边界上，差一位都不行。"""
    from spherical_blast_pinn.cli import extract_profiles as ep
    from spherical_blast_pinn.data import raw_reader as rr

    class _ArrayType:
        global_timesteps = "global_timesteps"

    def _make(reported: int, true_max: int):
        """``true_max`` 是**可读的最大下标 + 1**，与 LASSO 的语义一致。"""

        class _D3plot:
            def __init__(self, path, state_filter=None, buffered_reading=True,
                         state_array_filter=None):
                requested = sorted(int(s) for s in (state_filter or {0}))
                if any(s >= true_max for s in requested):
                    raise ValueError(f"maximum number of states of '{reported}'")
                self.arrays = {"global_timesteps": np.zeros(len(requested))}

        return _D3plot

    monkeypatch.setattr(rr, "lasso_version", lambda: "0.0.0-test")

    monkeypatch.setattr(rr, "_import_lasso", lambda: (_make(5, 5), _ArrayType))
    n, note = ep._resolve_state_count(Path("nowhere"), file_count=6)
    assert n == 5
    assert "boundary verification" in note
    assert "file count 6 differs by 1" in note

    monkeypatch.setattr(rr, "_import_lasso", lambda: (_make(5, 6), _ArrayType))
    with pytest.raises(ep.ExtractionError) as excinfo:
        ep._resolve_state_count(Path("nowhere"), file_count=7)
    assert "off by at least one" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. 网格自己的分辨率
# ---------------------------------------------------------------------------


def test_cell_thickness_table_reads_the_mesh_resolution():
    """网格自己的径向分辨率必须能从几何里读出来。"""
    from spherical_blast_pinn.cli.extract_profiles import (
        _cell_thickness_table,
        _mesh_cell_thickness,
    )

    radii = np.array([0.01, 0.02, 0.03, 0.2, 0.4, 0.6])
    thickness = np.array([0.8e-3, 0.8e-3, 0.8e-3, 40e-3, 50e-3, 60e-3])
    table = _cell_thickness_table(radii, thickness)
    assert table is not None
    assert _mesh_cell_thickness(table, 0.02) == pytest.approx(0.8e-3, rel=0.05)
    assert _mesh_cell_thickness(table, 0.5) == pytest.approx(55e-3, rel=0.15)
    # 没有几何时宁可说"不知道"。
    assert not np.isfinite(_mesh_cell_thickness(None, 0.02))


def test_fixed_mesh_geometry_reports_the_element_radial_thickness():
    """``radial_thickness`` 是节点半径的极差，来自真实几何而不是体积推算。"""
    from spherical_blast_pinn.data import raw_reader as rr

    # 一个位于半径 0.1 m 处的六面体单元，节点半径从 0.099 到 0.101。
    coords = np.array([
        [0.099, 0.0, 0.0], [0.101, 0.0, 0.0],
        [0.0, 0.099, 0.0], [0.0, 0.101, 0.0],
        [0.0, 0.0, 0.099], [0.0, 0.0, 0.101],
        [0.099, 0.099, 0.0], [0.101, 0.101, 0.0],
    ])

    class _Case:
        coordinates = coords
        connectivity = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        n_elements = 1
        position_convention = rr.POSITION_IS_ABSOLUTE
        position_note = "synthetic"

    geometry = rr.fixed_mesh_geometry(_Case())
    assert geometry.radial_thickness is not None
    expected = np.linalg.norm(coords, axis=1).max() - np.linalg.norm(coords, axis=1).min()
    assert geometry.radial_thickness[0] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. 端到端：把一个算例的关键帧跑过真正的 extract_case
# ---------------------------------------------------------------------------


def _deck_context(charge_radius: float = 0.05):
    from spherical_blast_pinn.cli.extract_profiles import DeckContext

    return DeckContext(
        deck_path=Path("input.k"), unit_system="si",
        length_to_m=1.0, time_to_s=1.0, density_to_si=1.0, velocity_to_si=1.0,
        pressure_to_si=1.0, product_part_ids=(2,), air_part_ids=(1,),
        product_ale_indices=(2,), air_ale_indices=(1,),
        detonation_velocity_m_s=6930.0, initial_detonation=(0.0, 0.0, 0.0),
        termination_time=1.0e-3, source={"synthetic": True},
    )


def _run_extract_case(
    tmp_path: Path,
    *,
    times,
    contact,
    shock,
    air_overpressure: bool,
    invalid_product_radius=None,
    element_thickness: float = 0.0008,
    monkeypatch=None,
):
    """跑真正的 ``extract_case``，只把"从哪里拿帧"换成合成帧。

    分类、污染判定、无激波撤标、元数据组装全部走真实代码路径。
    """
    from spherical_blast_pinn.cli import extract_profiles as ep
    from spherical_blast_pinn.data import raw_reader as rr
    from spherical_blast_pinn.data.shell_average import ShellGrid

    charge_radius = 0.05
    n_cells = 6000
    cell_radius = np.linspace(0.004, 0.30, n_cells)
    edges = np.concatenate([
        [0.0],
        0.5 * (cell_radius[1:] + cell_radius[:-1]),
        [cell_radius[-1] + 0.01],
    ])
    volume = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ambient = mat.AIR_PRESSURE

    frames = []
    for i, t in enumerate(times):
        alpha = (cell_radius <= contact[i]).astype(np.float64)
        pressure = np.full(n_cells, ambient)
        density = np.full(n_cells, 1.225)
        velocity = np.zeros(n_cells)
        product = alpha > 0.5
        pressure[product] = 5.0e9
        density[product] = 1630.0
        if air_overpressure:
            layer = (cell_radius > contact[i]) & (cell_radius <= shock[i])
            # **用解析 RH 的波后状态**，不是随手编的三个数。
            #
            # §10 之后空气监督要有资格，激波必须通过 RH 核查；随便给一个密度跳变和
            # 速度跳变是过不去的，于是"有空气标签"这件事本身就不该成立。测试夹具
            # 因此必须给出物理自洽的激波，否则它测的是一条根本不该存在的数据路径。
            gamma_air = mat.AIR_GAMMA
            mach = 3.0
            c1 = math.sqrt(gamma_air * ambient / mat.AIR_DENSITY)
            speed = mach * c1
            ratio = ((gamma_air + 1.0) * mach**2) / ((gamma_air - 1.0) * mach**2 + 2.0)
            pressure[layer] = ambient * (
                1.0 + 2.0 * gamma_air * (mach**2 - 1.0) / (gamma_air + 1.0)
            )
            density[layer] = mat.AIR_DENSITY * ratio
            velocity[layer] = speed * (1.0 - 1.0 / ratio)
        energy = pressure / ((mat.AIR_GAMMA - 1.0) * density)
        energy[product] = mat.PRODUCT_SPECIFIC_ENERGY
        state_valid = np.ones(n_cells, dtype=bool)
        if invalid_product_radius is not None:
            bad = product & (cell_radius <= invalid_product_radius)
            state_valid[bad] = False
            energy[bad] = np.nan
        frames.append(
            rr.RawFrame(
                case_id="synth", time=float(t), radius=cell_radius, cell_volume=volume,
                density=density, velocity_x=velocity, velocity_y=np.zeros(n_cells),
                velocity_z=np.zeros(n_cells), pressure=pressure,
                tnt_volume_fraction=alpha, radial_velocity=velocity,
                specific_internal_energy=energy, state_valid=state_valid,
                cell_centroid=np.stack(
                    [cell_radius, np.zeros(n_cells), np.zeros(n_cells)], axis=1
                ),
                provenance={"state_index": i},
            )
        )

    class _Summary:
        parts = {1: "AIR", 2: "TNT"}

    class _Case:
        part_indexes = np.array([0, 1], dtype=np.int64)
        n_elements = 2
        position_convention = "POSITION_IS_ABSOLUTE"
        position_note = "synthetic"

    geometry = rr.FixedMeshGeometry(
        centroid=np.stack([cell_radius, np.zeros(n_cells), np.zeros(n_cells)], axis=1),
        radius=cell_radius.copy(), volume=volume.copy(),
        radial_thickness=np.full(n_cells, element_thickness),
    )

    monkeypatch.setattr(ep, "parse_deck", lambda *a, **k: _Summary())
    monkeypatch.setattr(ep, "list_d3plot_family", lambda *a, **k: [Path("d3plot")])
    monkeypatch.setattr(ep, "open_d3plot_case", lambda *a, **k: _Case())
    monkeypatch.setattr(ep, "fixed_mesh_geometry", lambda *a, **k: geometry)
    monkeypatch.setattr(ep, "_iter_frames", lambda *a, **k: iter(frames))

    grid = ShellGrid.graded(inner=charge_radius / 500.0, outer=0.28, n_shells=900,
                            anchor=charge_radius)
    return ep.extract_case(
        case_id="synth", d3plot_dir=Path("nowhere"), deck=_deck_context(charge_radius),
        output_dir=Path(tmp_path) / "synth", charge_radius_m=charge_radius, grid=grid,
        states=list(range(len(times))), n_states=len(times), batch_states=1,
    )


def test_a_frame_without_a_resolvable_shock_carries_no_air_label(tmp_path, monkeypatch):
    """没有激波就没有空气域：环境态按规范是**无标签**的。

    装药还没把激波推出来时 ``R_s`` 定不出来，被监督的那条带子 ``R_c < r < R_s``
    就没有边界。旧实现会把两米以内的均匀大气整片标成空气送进训练——那些壳层场量
    齐全、看起来完全正常，只是它们全都是环境态。
    """
    times = [0.0, 1.0e-5, 2.0e-5]
    contact = [0.052, 0.056, 0.062]

    result = _run_extract_case(
        tmp_path, times=times, contact=contact,
        shock=[float("nan")] * 3, air_overpressure=False, monkeypatch=monkeypatch,
    )
    z = np.load(Path(result.output_dir) / "material.npz")
    # §10 之后状态是字符串：这些帧里空气侧压力处处等于环境，没有 -dp/dr 候选。
    assert set(np.asarray(z["shock_status"]).tolist()) == {"not_detected"}
    assert not z["shock_has_supervision"].any(), "没有候选就没有监督资格"
    assert int((z["material"] == 1).sum()) == 0, (
        "环境态被标成了空气；规范要求激波之外的区域无标签"
    )
    assert int((z["material"] == 0).sum()) > 0, "产物侧的标签不该被一起撤掉"


def test_the_air_group_survives_every_resolved_frame(tmp_path, monkeypatch):
    """第一轮审查那条"标成空气、状态为空"的直接对应。"""
    times = [0.0, 1.0e-5, 2.0e-5, 4.0e-5]
    contact = [0.052, 0.056, 0.062, 0.075]
    # 波后空气层要有**几个原始单元**厚。夹具的合成单元厚 0.8 mm，原来的 2 mm 层
    # 宽只有 ~2 个单元，会被 §10 的亚单元规则正确地判成"分辨不出"——那时"空气有
    # 标签"这件事本身就不该成立。这里的层宽是 38–125 mm。
    shock = [0.090, 0.110, 0.140, 0.200]

    result = _run_extract_case(
        tmp_path, times=times, contact=contact, shock=shock, air_overpressure=True,
        monkeypatch=monkeypatch,
    )
    z = np.load(Path(result.output_dir) / "material.npz")
    raw = np.load(Path(result.output_dir) / "raw_line.npz")
    finite = (
        np.isfinite(raw["rho"]) & np.isfinite(raw["u"]) & np.isfinite(raw["P"])
        & np.isfinite(raw["e"])
    )
    assert int((z["valid"] & ~finite).sum()) == 0
    # **监督资格严格跟着检测状态。** 只有 confirmed 才有可靠范围、才允许空气标签；
    # 跃迁带跨不过一个原始单元的帧（这里是最早那帧，带宽 0.62 mm < 合成单元 0.8 mm）
    # 必须是 doubtful **且**没有空气监督——这正是 §10 要的分开记。
    status = np.asarray(z["shock_status"]).tolist()
    supervision = z["shock_has_supervision"]
    assert not supervision[0], f"第一帧的跃迁带比单元还窄，不该有监督：{status}"
    assert supervision.any(), f"后面的帧应当被确认：{status}"
    for i in range(len(times)):
        assert bool(supervision[i]) == (status[i] == "confirmed"), (
            f"帧 {i}: 状态 {status[i]} 与监督资格 {bool(supervision[i])} 不一致"
        )
        if not supervision[i]:
            assert int((z["material"][i] == 1).sum()) == 0, "没有监督资格就不该有空气标签"
        # 被确认的帧上 RH 残差必须真的小——合成状态是解析 RH，残差应当接近机器精度。
        if status[i] == "confirmed":
            assert z["shock_rh_momentum"][i] < 1e-10
            assert z["shock_rh_energy"][i] < 1e-10
            assert z["shock_compression"][i] > 1.0
        assert int(((z["material"][i] == 0) & finite[i]).sum()) > 0
    assert int(((z["material"] == 1) & finite).sum()) > 0


def test_contaminated_shells_are_invalid_and_the_hole_is_recorded(tmp_path, monkeypatch):
    """没有产物状态的壳层不能算有效，而且**缺口必须被记录在案**。

    装药里尚未释放化学能的单元套产物 JWL 会反解出负比内能。那些壳层没有能量标签是
    对的；错的是让它们保持 ``valid=True``，那样采样器会把它们当成监督，数据损失里的
    产物组被一批空标签稀释。
    """
    times = [0.0, 1.0e-5, 2.0e-5, 4.0e-5]
    contact = [0.052, 0.056, 0.062, 0.075]
    shock = [0.054, 0.070, 0.095, 0.150]

    result = _run_extract_case(
        tmp_path, times=times, contact=contact, shock=shock, air_overpressure=True,
        invalid_product_radius=0.03, monkeypatch=monkeypatch,
    )
    z = np.load(Path(result.output_dir) / "material.npz")
    raw = np.load(Path(result.output_dir) / "raw_line.npz")
    finite = (
        np.isfinite(raw["rho"]) & np.isfinite(raw["u"]) & np.isfinite(raw["P"])
        & np.isfinite(raw["e"])
    )
    contaminated = z["contaminated_shell_fraction"] > 0.0
    assert contaminated.any(), "构造里有没状态的单元，污染必须被记录"
    assert int((z["valid"] & ~finite).sum()) == 0, "有效必须蕴含有限"
    unexplained = (z["material"] != 2) & ~finite & ~contaminated
    assert int(unexplained.sum()) == 0, "缺口必须全部落在被记录的污染里"
    assert int(((z["material"] != 2) & ~finite).sum()) > 0, "构造应当产生缺口"
    # 污染的定量记录也要在。
    assert int(z["state_invalid_cells"].sum()) > 0


def test_a_shock_layer_thinner_than_one_cell_is_flagged_subcell(tmp_path, monkeypatch):
    """``h = R_s - R_c`` 小于一个 ALE 单元厚度时，必须标成"分辨不出"。

    网格加密到 0.12 mm 的壳层不会让一个 10 mm 厚的 ALE 单元里的梯度变得可见。
    """
    times = [1.0e-5, 2.0e-5]
    contact = [0.052, 0.056]
    # 空气层 1 mm 厚，而单元径向厚度 8 mm。
    shock = [0.053, 0.057]

    result = _run_extract_case(
        tmp_path, times=times, contact=contact, shock=shock, air_overpressure=True,
        element_thickness=0.008, monkeypatch=monkeypatch,
    )
    z = np.load(Path(result.output_dir) / "material.npz")
    meta = json.loads((Path(result.output_dir) / "metadata.json").read_text("utf-8"))
    # 跃迁带跨不过一个原始单元 -> 不给监督资格（状态是 doubtful 或 not_detected）。
    assert not z["shock_has_supervision"].any(), (
        "h 比一个原始单元还窄时不能给出监督资格"
    )
    # 没有激波半径时 h/单元 无从谈起，只能是 nan——不能拿一个数冒充。
    assert np.all(~np.isfinite(z["shock_h_over_cell"]))
    assert np.all(~np.isfinite(z["shock_radius"]))
    assert meta["front_detection"]["shock_frames_with_supervision"] == 0
    # 没有前沿就没有"该处的单元厚度"；这里只能是 nan，不能拿一个数冒充。
    assert np.all(~np.isfinite(z["shock_cell_thickness_m"]))


# ---------------------------------------------------------------------------
# 6. 闸门：缺口有据可查，而不是"比例很低"
# ---------------------------------------------------------------------------


def _write_missing_state_case(directory: Path, *, recorded: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    n_t, n_x = 2, 60
    times = np.linspace(0.0, 1.0e-4, n_t)
    centres = np.linspace(1.0e-4, 0.2, n_x)
    rho = np.full((n_t, n_x), 1.225)
    vel = np.zeros((n_t, n_x))
    energy = np.full((n_t, n_x), 2.0e5)
    pressure = np.full((n_t, n_x), mat.AIR_PRESSURE)
    material = np.full((n_t, n_x), 1, dtype=np.int64)
    alpha = np.zeros((n_t, n_x))
    # 前 5 个壳层标成产物但**没有状态**（未起爆装药那一档），第 6 个有状态：
    # 这一档在真实的 50 mm 算例里是局部的，不是整个产物组都空。
    material[:, :6] = 0
    alpha[:, :6] = 1.0
    rho[:, :6] = 1630.0
    pressure[:, :6] = 5.0e9
    energy[:, :5] = np.nan
    energy[:, 5] = mat.PRODUCT_SPECIFIC_ENERGY
    # 有效性由**场量**收口，不由标签收口——这正是提取器现在的语义。
    valid = (material != 2) & np.isfinite(energy)
    extra = {}
    if recorded:
        contam = np.zeros((n_t, n_x))
        contam[:, :5] = 0.4
        extra["contaminated_shell_fraction"] = contam
    np.savez(directory / "raw_line.npz", t=times, x=centres, rho=rho, u=vel,
             P=pressure, e=energy)
    np.savez(directory / "material.npz", alpha_product=alpha, material=material,
             valid=valid, shell_count=np.ones((n_t, n_x), dtype=np.int64),
             contact_radius=np.full(n_t, 0.05), shock_radius=np.full(n_t, 0.15),
             mixed_cell_fraction=np.zeros(n_t), state_index=np.arange(n_t), **extra)
    (directory / "metadata.json").write_text(
        json.dumps({"case_id": directory.name, "R_0": 0.05, "n_t": n_t, "n_x": n_x, "detonation_velocity_m_s": 6930.0}),
        encoding="utf-8",
    )


def test_the_verdict_demands_an_explanation_for_a_labelled_shell_with_no_state(tmp_path):
    """闸门要的是"缺口有据可查"，不是"缺口比例很低"。"""
    unexplained_dir = Path(tmp_path) / "unexplained"
    _write_missing_state_case(unexplained_dir, recorded=False)
    status, evidence = material_partition_verdict(
        load_extracted_case(unexplained_dir, case_id="x")
    )
    assert status == INCOMPATIBLE
    assert evidence["labelled_shells_without_state_unexplained"] > 0
    assert evidence["contamination_is_recorded"] is False

    explained_dir = Path(tmp_path) / "explained"
    _write_missing_state_case(explained_dir, recorded=True)
    status, evidence = material_partition_verdict(
        load_extracted_case(explained_dir, case_id="x")
    )
    assert evidence["labelled_shells_without_state_unexplained"] == 0
    assert evidence["labelled_shells_without_state_explained_by_contamination"] > 0
    assert status == COMPATIBLE, evidence.get("problems")


def test_the_verdict_still_refuses_a_silent_air_hole(tmp_path):
    """第一轮那条不能因为第二轮的放宽而失效：空气整片为空仍是硬失败。"""
    directory = Path(tmp_path) / "airhole"
    directory.mkdir(parents=True, exist_ok=True)
    n_t, n_x = 3, 400
    times = np.linspace(0.0, 1.0e-4, n_t)
    centres = np.linspace(1.0e-4, 1.0, n_x)
    contact = np.array([0.05, 0.08, 0.12])
    shock = contact * 1.6
    rho = np.full((n_t, n_x), np.nan)
    vel = np.full((n_t, n_x), np.nan)
    energy = np.full((n_t, n_x), np.nan)
    pressure = np.full((n_t, n_x), np.nan)
    material = np.full((n_t, n_x), 2, dtype=np.int64)
    alpha = np.zeros((n_t, n_x))
    for i in range(n_t):
        product = centres < contact[i]
        air = (centres >= contact[i]) & (centres < shock[i])
        material[i, product] = 0
        material[i, air] = 1
        alpha[i, product] = 1.0
        rho[i, product] = 1630.0
        vel[i, product] = 100.0
        energy[i, product] = mat.PRODUCT_SPECIFIC_ENERGY
        pressure[i, product] = 5.0e9
        # 空气那一段**故意**留空。
    np.savez(directory / "raw_line.npz", t=times, x=centres, rho=rho, u=vel,
             P=pressure, e=energy)
    np.savez(directory / "material.npz", alpha_product=alpha, material=material,
             valid=material != 2, shell_count=np.ones((n_t, n_x), dtype=np.int64),
             contact_radius=contact, shock_radius=shock,
             mixed_cell_fraction=np.zeros(n_t), state_index=np.arange(n_t))
    (directory / "metadata.json").write_text(
        json.dumps({"case_id": "airhole", "R_0": 0.05, "n_t": n_t, "n_x": n_x, "detonation_velocity_m_s": 6930.0}),
        encoding="utf-8",
    )
    status, evidence = material_partition_verdict(
        load_extracted_case(directory, case_id="airhole")
    )
    assert status == INCOMPATIBLE
    assert evidence["air_shells"] > 0
    assert evidence["air_shells_with_finite_fields"] == 0


# ---------------------------------------------------------------------------
# 7. 第三轮：接触面的插值跨距，以及 LASSO 的 history 分配
# ---------------------------------------------------------------------------


def test_crossing_radius_reports_how_far_it_interpolated():
    """接触面的插值**跨距**必须一起报出来。

    80 mm 算例装药外第一个空气单元的径向厚度是 208–361 mm，体积分数从 1 直接跳到 0。
    0.5 穿越被线性插值到 131.97 mm，而装药半径是 80 mm——那个数看起来完全正常。
    只有把跨距一起报出来，才可能在数据里发现"那一带根本没有单元"。
    """
    from spherical_blast_pinn.cli.extract_profiles import _crossing_radius_and_gap

    # 相邻壳层：跨距就是一个壳层宽度，中间没有空壳层。
    centres = np.linspace(0.001, 0.30, 3000)
    values = np.where(centres < 0.05, 1.0, 0.0)
    radius, gap, holes = _crossing_radius_and_gap(centres, values, 0.5)
    assert abs(radius - 0.05) < 1e-3
    assert gap < 2e-4
    assert holes == 0

    # 中间有一大段没有数据（nan）：跨距与空壳层个数都必须反映出来。
    centres = np.array([0.01, 0.02, 0.05, 0.20, 0.40])
    values = np.array([1.0, 1.0, np.nan, 0.0, 0.0])
    radius, gap, holes = _crossing_radius_and_gap(centres, values, 0.5)
    # 有数据的最后一个是 0.02（值 1），第一个低于阈值的是 0.20（值 0），
    # 于是在这两者之间插值——而 0.05 那个壳层是空的，中间什么都没有。
    assert radius == pytest.approx(0.11)
    assert gap == pytest.approx(0.18)
    assert holes == 1

    # 完全没有穿越。
    radius, gap, holes = _crossing_radius_and_gap(centres, np.ones_like(centres), 0.5)
    assert not np.isfinite(radius)

    # 判据是**空壳层个数**，不是跨距与单元厚度之比：单个巨型单元内部处处是空的，
    # 跨距反而小于单元厚度。这一构造就是 80 mm 算例的形状。
    centres = np.array([0.02, 0.04, 0.06] + list(np.linspace(0.07, 0.45, 500))
                       + [0.50, 0.60])
    values = np.array([1.0, 1.0, 1.0] + [np.nan] * 500 + [0.0, 0.0])
    radius, gap, holes = _crossing_radius_and_gap(centres, values, 0.5)
    assert holes == 500
    assert gap == pytest.approx(0.44)
    assert radius == pytest.approx(0.28)   # "两个壳层中间某处"


def test_solid_history_allocation_fix_grows_a_short_master(monkeypatch):
    """LASSO 把 16 列的数据往 10 列的容器里塞；补丁只把容器加宽，不动数值。

    真正的判据不在这条测试里，而在 ``_charge_radius_from_volume``：取错列不可能反推出
    正确的装药半径。这里钉住的是补丁本身的语义——**只增不减、已有状态原样保留**。
    """
    import numpy as np
    from spherical_blast_pinn.data import raw_reader as rr

    calls = {}

    class _ArrayType:
        element_solid_history_variables = "element_solid_history_variables"

    class _Fake:
        @staticmethod
        def _read_states_transfer_memory(i_state, buffer_array_dict, master_array_dict):
            # 原始实现：直接赋值，形状不符就抛（这正是 LASSO 的行为）。
            for name, buf in buffer_array_dict.items():
                master_array_dict[name][i_state:i_state + buf.shape[0]] = buf
            calls["done"] = True

    monkeypatch.setattr(rr, "_SOLID_HISTORY_FIX_APPLIED", False)
    monkeypatch.setattr(rr, "_SOLID_HISTORY_FIX_APPROVED_VERSION", "0.0.0-test")
    monkeypatch.setattr(rr, "lasso_version", lambda: "0.0.0-test")
    monkeypatch.setattr(rr, "LASSO_FIX_LOG", [])
    rr._install_solid_history_allocation_fix(_Fake, _ArrayType)

    field = _ArrayType.element_solid_history_variables
    # 容器宽 10，数据宽 16——没有补丁就会抛 ValueError。
    # 注意补丁是**替换字典里的那一项**，所以必须从字典里取回来看结果。
    master_array_dict = {field: np.full((2, 4, 10), np.nan)}
    master_array_dict[field][0] = 7.0
    buffers = {field: np.ones((1, 4, 16))}
    _Fake._read_states_transfer_memory(1, buffers, master_array_dict)
    assert calls.get("done")
    grown = master_array_dict[field]
    # 已有的状态原样保留在重叠的列里，新状态写在正确的位置，宽度变成 16。
    assert grown.shape == (2, 4, 16)
    assert np.all(grown[0, :, :10] == 7.0)
    assert np.all(grown[1] == 1.0)
    # 加宽过什么必须被记下来——静默生效的补丁和没有补丁从外部看不出区别。
    assert len(rr.LASSO_FIX_LOG) == 1
    assert rr.LASSO_FIX_LOG[0]["widths_before_after"] == [10, 16]

    # 形状已经一致时不得重新分配（否则会白拷一遍，还会丢掉已经读入的状态）。
    same = np.zeros((2, 4, 16))
    before = id(same)
    _Fake._read_states_transfer_memory(1, {field: np.ones((1, 4, 16))}, {field: same})
    assert id(same) == before

    monkeypatch.setattr(rr, "_SOLID_HISTORY_FIX_APPLIED", False)


def test_solid_history_fix_touches_nothing_outside_the_reproduced_case(monkeypatch):
    """**§7：补丁范围必须收窄。** 不是"任何形状不一致都顺手抹平"。

    只有 ``(字段=history, 宽度=(10, 16), 版本=已批准)`` 这一组合才加宽。其余一律不
    干预，让 LASSO 原来的异常原样抛出——否则真正的字段错位会被当成这个已知 bug 静默
    吞掉，而它本来应该让提取停下来。
    """
    import numpy as np
    from spherical_blast_pinn.data import raw_reader as rr

    class _ArrayType:
        element_solid_history_variables = "element_solid_history_variables"

    field = _ArrayType.element_solid_history_variables

    def run(*, name, master_width, buffer_width, version, approved, expected_grow):
        # **每次都用全新的类。** 补丁是给类属性赋值的，复用同一个类会让上一条的
        # 包装器留在链上，于是"版本没批准所以不加宽"那条会被旧的包装器抢先加宽。
        class _Fake:
            @staticmethod
            def _read_states_transfer_memory(i_state, buffer_array_dict, master_array_dict):
                for key, buf in buffer_array_dict.items():
                    master_array_dict[key][i_state:i_state + buf.shape[0]] = buf

        monkeypatch.setattr(rr, "_SOLID_HISTORY_FIX_APPLIED", False)
        monkeypatch.setattr(rr, "_SOLID_HISTORY_FIX_APPROVED_VERSION", approved)
        monkeypatch.setattr(rr, "lasso_version", lambda: version)
        monkeypatch.setattr(rr, "LASSO_FIX_LOG", [])
        rr._install_solid_history_allocation_fix(_Fake, _ArrayType)
        master = {name: np.zeros((2, 4, master_width))}
        buffer = {name: np.ones((1, 4, buffer_width))}
        try:
            _Fake._read_states_transfer_memory(1, buffer, master)
            raised = False
        except ValueError:
            raised = True
        grew = bool(rr.LASSO_FIX_LOG)
        assert grew is expected_grow, (name, master_width, buffer_width, version, approved)
        assert raised is not expected_grow, "grow 与抛异常必须互补"
        return master[name].shape

    # 已复现的那一组：加宽。
    assert run(name=field, master_width=10, buffer_width=16,
               version="2.0.3", approved="2.0.3", expected_grow=True)[-1] == 16
    # 另一个字段：不碰。
    run(name="element_solid_stress", master_width=10, buffer_width=16,
        version="2.0.3", approved="2.0.3", expected_grow=False)
    # 另一对宽度：不碰。
    run(name=field, master_width=10, buffer_width=12,
        version="2.0.3", approved="2.0.3", expected_grow=False)
    # 版本没验证过：不碰。
    run(name=field, master_width=10, buffer_width=16,
        version="9.9.9", approved="2.0.3", expected_grow=False)
    # 缩窄（容器比数据宽）绝不允许。
    run(name=field, master_width=16, buffer_width=10,
        version="2.0.3", approved="2.0.3", expected_grow=False)


def test_front_labels_require_a_resolved_front(tmp_path):
    """**"分辨得出吗"必须在读取端生效，不能只是元数据里的一行字。**

    前沿位置是监督标签。一个跨着空壳层插出来的半径不是被测量出来的量——80 mm 算例的
    接触面因此被插到 131.97 mm，而装药半径是 80 mm。把这种数当标签用，等于让网络去
    拟合一个网格根本没分辨的界面。
    """
    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    def write(directory: Path, contact_state, shock_state):
        directory.mkdir(parents=True, exist_ok=True)
        n_t, n_x = len(contact_state), 60
        times = np.linspace(0.0, 1.0e-4, n_t)
        centres = np.linspace(1.0e-4, 0.2, n_x)
        rho = np.full((n_t, n_x), 1.225)
        energy = np.full((n_t, n_x), 2.0e5)
        pressure = np.full((n_t, n_x), mat.AIR_PRESSURE)
        material = np.full((n_t, n_x), 1, dtype=np.int64)
        alpha = np.zeros((n_t, n_x))
        contact = np.full(n_t, 0.05)
        shock = np.full(n_t, 0.12)
        np.savez(directory / "raw_line.npz", t=times, x=centres, rho=rho,
                 u=np.zeros((n_t, n_x)), P=pressure, e=energy)
        np.savez(directory / "material.npz", alpha_product=alpha, material=material,
                 valid=material != 2, shell_count=np.ones((n_t, n_x), dtype=np.int64),
                 contact_radius=contact, shock_radius=shock,
                 contact_state=np.asarray(contact_state, dtype=np.int64),
                 shock_state=np.asarray(shock_state, dtype=np.int64),
                 mixed_cell_fraction=np.zeros(n_t), state_index=np.arange(n_t))
        (directory / "metadata.json").write_text(
            json.dumps({"case_id": directory.name, "R_0": 0.05, "n_t": n_t, "n_x": n_x, "detonation_velocity_m_s": 6930.0}),
            encoding="utf-8",
        )

    resolved = Path(tmp_path) / "resolved"
    write(resolved, [2, 2, 0], [2, 2, 1])
    case = load_extracted_case(resolved, case_id="resolved")
    # 第 3 帧接触面没找到；第 1、2 帧两个前沿都分辨得出。
    assert case.contact_valid.tolist() == [True, True, False]
    # 激波在第 3 帧是亚单元（1）——一个跨着空壳层插出来的激波同样没有标签。
    assert case.shock_valid.tolist() == [True, True, False]

    unresolved = Path(tmp_path) / "unresolved"
    write(unresolved, [1, 1, 1], [0, 0, 0])
    case = load_extracted_case(unresolved, case_id="unresolved")
    assert not case.contact_valid.any(), "跨着空壳层插出来的接触面不能当标签"
    assert not case.shock_valid.any()
    # 这一份导出只有三态码、没有资格数组，走的是中间那一档来源
    sources = case.source["front_eligibility_source"]
    assert all("three-state detection code" in value for value in sources.values())
    # 每一侧各自说明用了哪一档证据
    assert "never changes the other side" in case.source["front_resolution"]
