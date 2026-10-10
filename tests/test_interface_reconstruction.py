"""§9 接触面逐帧几何重构的定向回归。

§9 给的验证清单是：球形已知界面、公共面极限、代表性真实帧、重构加密；并且
"不得把重构的子单元位置当作原始数据获得了额外分辨率"。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spherical_blast_pinn.data.interface_reconstruction import (
    STATUS_ON_SHARED_FACE,
    STATUS_RECONSTRUCTED,
    STATUS_UNRESOLVED,
    ReconstructionError,
    fraction_from_radius,
    plic_radius,
    reconstruct_interface,
    shell_crossing_radius,
    volume_equivalent_radius,
)


def _edges(outer: float = 0.5, n: int = 200, inner: float = 0.0) -> np.ndarray:
    return np.linspace(inner, outer, n + 1)


def _alpha_for_sphere(edges: np.ndarray, radius: float) -> np.ndarray:
    """把界面放在 ``radius``，逐单元算产物体积分数（产物在内侧）。"""
    inner, outer = edges[:-1], edges[1:]
    alpha = np.zeros(edges.size - 1, dtype=np.float64)
    full = outer <= radius
    alpha[full] = 1.0
    partial = (~full) & (inner < radius)
    alpha[partial] = (
        (radius**3 - inner[partial] ** 3) / (outer[partial] ** 3 - inner[partial] ** 3)
    )
    return alpha


# --------------------------------------------------------------------------
# 球形已知界面
# --------------------------------------------------------------------------


def test_a_known_spherical_interface_is_recovered_exactly():
    edges = _edges()
    truth = 0.137
    alpha = _alpha_for_sphere(edges, truth)

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_RECONSTRUCTED
    assert result.radius == pytest.approx(truth, rel=1.0e-12)
    # 闭式解代回去必须再现该单元的体积分数（浮点尾数）
    assert result.max_volume_fraction_error < 1.0e-12
    assert result.neighbourhood_complete


def test_the_shared_face_limit_is_an_exact_location_not_an_interpolation():
    """界面恰在公共面上：相邻两个纯相单元。

    这是**数据里的一个面**，不是插出来的亚单元位置，所以
    ``sub_cell_position`` 必须是假，位置不确定性是 0。
    """
    edges = _edges(n=100)
    face = float(edges[40])
    alpha = np.where(edges[1:] <= face, 1.0, 0.0)

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_ON_SHARED_FACE
    assert result.radius == pytest.approx(face, rel=0.0, abs=1.0e-15)
    assert result.interval == pytest.approx((face, face))
    assert result.sub_cell_position is False
    assert result.position_uncertainty == pytest.approx(0.0)
    assert result.supervision_eligible is True


def test_a_sub_cell_position_is_flagged_and_claims_no_extra_resolution():
    """§9：不得把重构的子单元位置当作原始数据获得了额外分辨率。"""
    edges = _edges(n=50)
    alpha = _alpha_for_sphere(edges, 0.2113)  # 落在某个单元内部

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_RECONSTRUCTED
    assert result.sub_cell_position is True
    assert any("resolution" in note for note in result.notes)
    # 不确定性不小于混合带的一半——位置只保证在带内
    band = result.mixing_band
    assert result.position_uncertainty == pytest.approx(0.5 * (band[1] - band[0]))
    # 原始分辨率照实报告，且不因重构而变小
    assert result.native_resolution == pytest.approx(float(np.median(np.diff(edges))))


def test_refining_the_grid_refines_the_native_resolution_not_the_claim():
    """加密的是**原始网格**：分辨率随之变小，这是真的；重构本身没有凭空造分辨率。"""
    truth = 0.137
    resolutions = []
    for n in (40, 160, 640):
        edges = _edges(n=n)
        alpha = _alpha_for_sphere(edges, truth)
        result = reconstruct_interface(edges, alpha)
        assert result.radius == pytest.approx(truth, rel=1.0e-10)
        resolutions.append(result.native_resolution)
        # 不管网格多细，位置不确定性始终由混合带宽度给出，不因"解得更准"而消失
        assert result.position_uncertainty > 0.0
    assert resolutions == sorted(resolutions, reverse=True)
    assert resolutions[0] > resolutions[-1]


# --------------------------------------------------------------------------
# 定位不了的情形：给区间与原因，不给半径
# --------------------------------------------------------------------------


def test_an_inconsistent_topology_is_refused_outright():
    edges = _edges(n=20)
    with pytest.raises(ReconstructionError, match="volume fractions"):
        reconstruct_interface(edges, np.full(19, 0.5))
    with pytest.raises(ReconstructionError, match="strictly increasing"):
        reconstruct_interface(np.array([0.0, 0.2, 0.2, 0.3]), np.full(3, 0.5))
    with pytest.raises(ReconstructionError, match=r"\[0, 1\]"):
        reconstruct_interface(edges, np.full(20, 1.5))


def test_a_mixing_band_without_an_inner_neighbour_is_unresolved():
    """缺邻域：混合带贴着网格内边界，无法确认内侧是产物。

    §9：这时给区间与原因，**不能**强制返回精确半径。
    """
    edges = _edges(n=30)
    alpha = np.zeros(30, dtype=np.float64)
    alpha[0] = 0.4          # 混合带就是第 0 个单元，内侧没有邻居
    alpha[1:] = 0.0

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_UNRESOLVED
    assert math.isnan(result.radius)
    assert result.neighbourhood_complete is False
    assert "inner side" in result.reason
    lo, hi = result.interval
    assert hi > lo, "未定位时也必须给出一个有效的外括号"


def test_a_mixing_band_whose_outer_neighbour_is_not_air_is_unresolved():
    edges = _edges(n=30)
    alpha = np.full(30, 1.0)
    alpha[10] = 0.6
    alpha[11] = 0.6
    alpha[12:] = 1.0        # 外侧还是产物，不构成产物→空气的过渡
    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_UNRESOLVED
    assert "outer side" in result.reason or "not pure air" in result.reason


def test_two_separated_bands_are_unresolved_because_the_band_is_not_contiguous():
    edges = _edges(n=40)
    alpha = np.zeros(40, dtype=np.float64)
    alpha[8:26] = 1.0
    alpha[10] = 0.5         # 带里插一个中间值
    alpha[20] = 0.5         # 另一个，与上一个不连续
    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_UNRESOLVED
    assert "contiguous" in result.reason


def test_a_frame_with_no_product_and_no_pair_is_unresolved():
    edges = _edges(n=20)
    result = reconstruct_interface(edges, np.zeros(20))
    assert result.status == STATUS_UNRESOLVED
    assert math.isnan(result.radius)
    assert result.interval[0] == pytest.approx(float(edges[0]))


def test_the_cross_check_quantities_are_still_reported_when_unresolved():
    """定位不了不等于什么都报不了：等效半径与壳平均穿越仍然给出来作对照。"""
    edges = _edges(n=20)
    alpha = np.zeros(20, dtype=np.float64)
    alpha[0] = 0.4
    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_UNRESOLVED
    assert result.volume_equivalent_radius > 0.0
    assert result.shell_crossing_radius is not None


# --------------------------------------------------------------------------
# 辅助量
# --------------------------------------------------------------------------


def test_the_symmetry_multiplier_moves_the_equivalent_radius_by_the_cube_root():
    """八分之一模型必须乘 8，否则等效半径小 2 倍——而那个数看起来仍像半径。"""
    edges = _edges(outer=0.05, n=100)
    alpha = np.ones(100)
    full = volume_equivalent_radius(edges, alpha, symmetry_multiplier=1.0)
    octant = volume_equivalent_radius(edges, alpha, symmetry_multiplier=8.0)
    assert octant == pytest.approx(full * 8.0 ** (1.0 / 3.0), rel=1.0e-12)
    assert full == pytest.approx(0.05, rel=1.0e-12)


def test_the_shell_crossing_is_an_auxiliary_and_differs_from_the_geometry():
    """``alpha = 0.5`` 的壳平均穿越只作对照，不做最终几何重构（§9）。"""
    edges = _edges(n=40)
    truth = 0.137
    alpha = _alpha_for_sphere(edges, truth)
    geometric = reconstruct_interface(edges, alpha)
    crossing = shell_crossing_radius(edges, alpha)

    assert geometric.radius == pytest.approx(truth, rel=1.0e-12)
    assert math.isfinite(crossing)
    # 两者都要报出来，且差异是可见的——正因为如此才不能拿穿越当几何位置
    assert geometric.shell_crossing_radius == pytest.approx(crossing)
    assert geometric.cross_check_relative_difference >= 0.0


def test_the_plic_radius_and_its_inverse_round_trip():
    for alpha in (0.0, 0.25, 0.5, 0.9, 1.0):
        radius = plic_radius(0.1, 0.3, alpha)
        assert fraction_from_radius(0.1, 0.3, radius) == pytest.approx(alpha, rel=1.0e-14)


def test_direction_spread_is_not_fabricated_when_there_is_no_directional_data():
    """径向平均过的数据没有方向信息。这里必须报 ``nan`` 加原因，不能填 0。

    填 0 会在报表里读成"方向离散为零"，也就是"界面完美球对称"——一个从未被测过的
    结论。
    """
    edges = _edges(n=40)
    alpha = _alpha_for_sphere(edges, 0.137)
    result = reconstruct_interface(edges, alpha)
    assert math.isnan(result.direction_spread)
    assert "no directional information" in result.direction_note

    with_directions = reconstruct_interface(
        edges, alpha, direction_radii=[0.136, 0.138, 0.1375]
    )
    assert with_directions.direction_spread == pytest.approx(
        (0.138 - 0.136) / ((0.136 + 0.138 + 0.1375) / 3), rel=1.0e-12
    )


def test_detection_uncertainty_and_eligibility_are_reported_separately():
    """§9：接触面检测、位置不确定性、监督资格分别输出，且与激波是否可用无关。"""
    edges = _edges(n=60)
    alpha = _alpha_for_sphere(edges, 0.137)
    payload = reconstruct_interface(edges, alpha).as_dict()
    for key in ("detection", "position_uncertainty_m", "supervision_eligible",
                "supervision_reason", "native_resolution_m", "mixing_band_m"):
        assert key in payload
    # 检测与资格是两个字段，不是一个字段的两种说法
    assert payload["detection"] == STATUS_RECONSTRUCTED
    assert payload["supervision_eligible"] is True
    # 没有激波字段参与：这个模块的输入里根本没有激波
    assert not any("shock" in key.lower() for key in payload)


def test_empty_shells_inside_the_transition_make_the_frame_unresolved():
    """空壳层的体积分数是 ``nan``——那一层**根本没有相信息**，不是空气。

    真实提取里这很常见：壳层细于 ALE 单元间距、或者没有单元质心落在里面时，
    ``mixture_fraction_profile`` 就返回 ``nan``。把它当 0 会凭空造出一个空气侧，
    界面被"定位"到空洞的边上；正确做法是承认定位不了。
    """
    edges = _edges(n=40)
    alpha = _alpha_for_sphere(edges, 0.137)
    alpha[18] = np.nan          # 空洞正好落在产物到空气的过渡里

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_UNRESOLVED
    assert math.isnan(result.radius)
    assert "empty shells" in result.reason
    lo, hi = result.interval
    assert hi > lo


def test_empty_shells_outside_the_transition_do_not_block_the_reconstruction():
    """空洞在过渡**之外**时不影响定位：外面那一段本来就不参与。"""
    edges = _edges(n=40)
    alpha = _alpha_for_sphere(edges, 0.137)
    alpha[0:3] = np.nan         # 轴心附近的空壳层，在产物区内部
    alpha[35:] = np.nan         # 远场的空壳层，在空气区内部

    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_RECONSTRUCTED
    assert result.radius == pytest.approx(0.137, rel=1.0e-12)
    # 混合带的下标要能对回**原网格**
    assert all(3 <= index < 35 for index in result.mixing_cells)


def test_a_frame_with_no_finite_fraction_at_all_is_unresolved_not_an_error():
    edges = _edges(n=12)
    result = reconstruct_interface(edges, np.full(12, np.nan))
    assert result.status == STATUS_UNRESOLVED
    assert math.isnan(result.radius)
    assert "finite product volume fraction" in result.reason


def test_a_real_sized_frame_from_the_raw_grid_shape():
    """代表性真实帧的量级：50 mm 装药、单元宽度约 0.1 mm。"""
    edges = _edges(outer=0.02, n=200)
    truth = 0.00637
    alpha = _alpha_for_sphere(edges, truth)
    result = reconstruct_interface(edges, alpha)
    assert result.status == STATUS_RECONSTRUCTED
    assert result.radius == pytest.approx(truth, rel=1.0e-9)
    assert result.native_resolution == pytest.approx(1.0e-4, rel=1.0e-9)
    # 位置不确定性远小于半径，但仍然如实报告，不写成 0
    assert 0.0 < result.position_uncertainty < 0.1 * truth
