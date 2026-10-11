"""场量布局：**声明出来的，不是从形状猜出来的**（提示词 §4）。

这一组盯的是一句看起来无害的 ``np.atleast_2d``。它在读取器里出现十次，作用是把
任意一维数组静默提升成 ``(1, n_x)``。三件事因此同时发生：

* **没有声明**——"这份文件是什么布局"在文件里没有答案，只有读取器的一次猜测；
* **没有记录**——读完之后 ``density.shape`` 是 ``(1, n)``，与"一份只存了一帧的
  二维导出"逐字相同，下游分不出这两种来源；
* **报错转移了责任**——校验拿到的形状是读取器自己造的，于是"一维场配两帧时间轴"
  这种真正的格式错误被报成 ``density has shape (1, 4)``，把读取器的猜测说成文件里
  的内容。而校验函数里那句 ``ndim == 1 and ... and times.size == 1`` 的顺手展平，
  在两条真实读取路径上从来没有被走到过——它是一句死代码，掩盖了真正的机制。

修法是提示词要求的**版本化规则**：布局在 ``metadata.json`` 里声明，读取器核对
声明与文件是否一致，并把读到的布局记在 ``source`` 上。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.extracted_case import (
    LAYOUT_FRAMES_BY_RADIUS,
    LAYOUT_SINGLE_FRAME_1D,
    PROFILE_LAYOUT_VERSION,
    DataError,
    load_extracted_case,
)

GRID = np.linspace(0.002, 0.30, 6)


def _profiles_one_dimensional() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside = GRID < 0.05
    density = np.where(inside, 1600.0, 1.225)
    pressure = np.where(inside, 1.0e10, 1.0e5)
    velocity = np.zeros_like(GRID)
    return density, velocity, pressure


def _profiles_two_dimensional(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density, velocity, pressure = _profiles_one_dimensional()
    return (
        np.tile(density, (n_frames, 1)),
        np.tile(velocity, (n_frames, 1)),
        np.tile(pressure, (n_frames, 1)),
    )


def _export(
    directory: Path,
    *,
    times,
    one_dimensional: bool,
    declaration=...,
    companion: dict | None = None,
) -> Path:
    """写一份最小导出。``declaration=...`` 表示**不写** ``profile_layout`` 这一项。"""
    directory.mkdir(parents=True, exist_ok=True)
    n_frames = len(times)
    if one_dimensional:
        density, velocity, pressure = _profiles_one_dimensional()
    else:
        density, velocity, pressure = _profiles_two_dimensional(n_frames)
    np.savez(
        directory / "raw_line.npz",
        t=np.asarray(times, dtype=float), x=GRID,
        rho=density, u=velocity, P=pressure,
    )
    metadata = {
        "R_0": 0.05,
        "detonation_velocity_m_s": 6930.0,
        "material_partition_is_volume_fraction_based": True,
        "material_partition_source": {
            "method": "ale_volume_fraction",
            "producer": "cli.extract_profiles",
            "producer_version": 1,
            "fields": ["tnt_volume_fraction"],
            "input_identifier": "sha256:test",
            "detail": {},
        },
        "symmetry_factor": 8.0,
        "symmetry_share": 0.125,
    }
    if declaration is not ...:
        metadata["profile_layout"] = declaration
    (directory / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    if companion is not None:
        material = np.where(GRID < 0.05, 0, 1)
        payload = {
            "material": material,
            "valid": material != 2,
            "contact_radius": np.full(n_frames, 0.05),
            "shock_radius": np.full(n_frames, 0.10),
            "contact_state": np.full(n_frames, 2, dtype=np.int64),
            "shock_state": np.full(n_frames, 2, dtype=np.int64),
            "contact_has_supervision": np.ones(n_frames, dtype=bool),
            "shock_has_supervision": np.ones(n_frames, dtype=bool),
            "field_valid_density": material != 2,
            "field_valid_velocity": material != 2,
            "field_valid_energy": material != 2,
            "field_valid_pressure": material != 2,
        }
        payload.update(companion)
        np.savez(directory / "material.npz", **payload)
    return directory


# --------------------------------------------------------------------------
# 一维单帧必须声明才读得进来
# --------------------------------------------------------------------------


def test_a_one_dimensional_profile_without_a_declaration_is_refused(tmp_path):
    """修前：读得进去，而且什么记录都不留。现在必须拒绝。

    "读得进去"本身不是错，错在**没有声明**——读取器是从形状上猜的，而形状是它
    自己刚造出来的。
    """
    directory = _export(
        tmp_path / "undeclared", times=[1.0e-5], one_dimensional=True,
        declaration=...,
    )
    with pytest.raises(DataError, match="single_frame_1d"):
        load_extracted_case(directory, case_id="undeclared")


def test_a_declared_single_frame_one_dimensional_export_reads_and_says_so(tmp_path):
    """声明了就读得进来，而且对象上看得出来：布局、版本、以及它确实是声明过的。"""
    directory = _export(
        tmp_path / "declared", times=[1.0e-5], one_dimensional=True,
        declaration={"version": PROFILE_LAYOUT_VERSION, "layout": LAYOUT_SINGLE_FRAME_1D},
    )
    case = load_extracted_case(directory, case_id="declared")

    assert case.density.shape == (1, GRID.size)
    assert case.velocity.shape == case.pressure.shape == case.density.shape
    recorded = case.source["profile_layout"]
    assert recorded == {
        "version": PROFILE_LAYOUT_VERSION,
        "layout": LAYOUT_SINGLE_FRAME_1D,
        "declared": True,
    }


def test_the_default_layout_is_recorded_as_inferred_not_declared(tmp_path):
    """旧导出没有这一项。仍然读得出来（缺省即逐帧二维，与既有数据一致），
    但 ``declared`` 必须是 ``False``——"缺省"是推断，不是文件里写过的声明。"""
    directory = _export(
        tmp_path / "legacy", times=[1.0e-5, 2.0e-5], one_dimensional=False,
        declaration=...,
    )
    case = load_extracted_case(directory, case_id="legacy")
    assert case.source["profile_layout"] == {
        "version": PROFILE_LAYOUT_VERSION,
        "layout": LAYOUT_FRAMES_BY_RADIUS,
        "declared": False,
    }


def test_a_declaration_that_contradicts_the_file_is_refused(tmp_path):
    """声明的是单帧一维，文件里却是二维逐帧——声明必须与内容一致，否则"声明"
    只是一个没人核对的字。"""
    directory = _export(
        tmp_path / "contradiction", times=[1.0e-5], one_dimensional=False,
        declaration={"version": PROFILE_LAYOUT_VERSION, "layout": LAYOUT_SINGLE_FRAME_1D},
    )
    with pytest.raises(DataError, match="one value per radial point"):
        load_extracted_case(directory, case_id="contradiction")


def test_the_single_frame_layout_says_nothing_about_several_frames(tmp_path):
    """一维单帧是"只有一帧，所以省掉那一维"的写法，对多帧什么也没说。

    多帧配这一档是**文件自相矛盾**，不是"大概是一帧"。这条要在看任何场量之前
    就报出来。
    """
    directory = _export(
        tmp_path / "multiframe", times=[1.0e-5, 2.0e-5], one_dimensional=False,
        declaration={"version": PROFILE_LAYOUT_VERSION, "layout": LAYOUT_SINGLE_FRAME_1D},
    )
    with pytest.raises(DataError, match="has 2 frames"):
        load_extracted_case(directory, case_id="multiframe")


@pytest.mark.parametrize(
    "declaration, needle",
    [
        ("frames_by_radius", "must be a mapping"),
        ({"version": PROFILE_LAYOUT_VERSION, "layout": "row_major"}, "does not\ndefine|does not define"),
        ({"version": PROFILE_LAYOUT_VERSION, "layout": LAYOUT_FRAMES_BY_RADIUS, "extra": 1}, "unknown keys"),
        ({"version": 2, "layout": LAYOUT_FRAMES_BY_RADIUS}, "supported"),
        ({"version": True, "layout": LAYOUT_FRAMES_BY_RADIUS}, "not an integer"),
        ({"version": "1", "layout": LAYOUT_FRAMES_BY_RADIUS}, "not an integer"),
        ({"layout": LAYOUT_FRAMES_BY_RADIUS}, "not an integer"),
    ],
)
def test_an_unrecognised_declaration_is_refused(tmp_path, declaration, needle):
    """不认识的布局、不支持的版本、类型不对的版本——都是**格式对不上**，
    不是"那就退回默认值"。``True`` 是 ``int`` 的子类（``True == 1``），
    所以版本必须显式排除布尔量，否则 ``"version": true`` 会被当成版本 1 读进来。
    """
    directory = _export(
        tmp_path / "bad", times=[1.0e-5, 2.0e-5], one_dimensional=False,
        declaration=declaration,
    )
    with pytest.raises(DataError, match=needle):
        load_extracted_case(directory, case_id="bad")


# --------------------------------------------------------------------------
# 报错不得把读取器造出来的形状说成文件里的内容
# --------------------------------------------------------------------------


def test_the_error_does_not_blame_a_shape_the_reader_invented(tmp_path):
    """**修前的原反例。** 一维场配两帧时间轴，实际报的是
    ``density has shape (1, 4), expected (2, 4)``——但文件里的 ``rho`` 是一维的，
    ``(1, 4)`` 是读取器自己 ``atleast_2d`` 出来的。

    报错里出现一个文件里根本没有的形状，读者会去找一个不存在的错误。
    """
    directory = _export(
        tmp_path / "misattributed", times=[1.0e-5, 2.0e-5], one_dimensional=True,
        declaration=...,
    )
    with pytest.raises(DataError) as caught:
        load_extracted_case(directory, case_id="misattributed")
    message = str(caught.value)
    assert f"(1, {GRID.size})" not in message, message   # 读取器造出来的那个形状
    assert f"({GRID.size},)" in message, message         # 文件里真实的形状
    assert LAYOUT_FRAMES_BY_RADIUS in message            # 声明的那一档
    assert LAYOUT_SINGLE_FRAME_1D in message             # 以及要哪种布局才读得进来


# --------------------------------------------------------------------------
# 配套掩码走同一条规则
# --------------------------------------------------------------------------


def test_the_companion_masks_follow_the_same_layout_rule(tmp_path, monkeypatch):
    """``material`` / ``valid`` / 逐字段掩码是**场量**，不是逐帧的前沿记录。

    修前它们各自 ``atleast_2d`` 之后与 ``density.shape`` 相比——比得上是因为读取器
    刚把两边改成同一个形状，而不是因为文件里本来就是那个形状。
    """
    one_dimensional_masks = {
        "material": np.where(GRID < 0.05, 0, 1),
        "valid": np.ones(GRID.size, dtype=bool),
        "field_valid_density": np.ones(GRID.size, dtype=bool),
        "field_valid_velocity": np.ones(GRID.size, dtype=bool),
        "field_valid_energy": np.ones(GRID.size, dtype=bool),
        "field_valid_pressure": np.ones(GRID.size, dtype=bool),
    }

    # 多帧 + 一维掩码：拒绝。
    multi = _export(
        tmp_path / "multi", times=[1.0e-5, 2.0e-5], one_dimensional=False,
        declaration=..., companion=dict(one_dimensional_masks),
    )
    with pytest.raises(DataError, match="'material'"):
        load_extracted_case(multi, case_id="multi")

    # 声明了单帧一维 + 一维掩码：读得进来，且掩码跟着变成规范形状。
    single = _export(
        tmp_path / "single", times=[1.0e-5], one_dimensional=True,
        declaration={"version": PROFILE_LAYOUT_VERSION, "layout": LAYOUT_SINGLE_FRAME_1D},
        companion=dict(one_dimensional_masks),
    )
    case = load_extracted_case(single, case_id="single")
    assert case.material.shape == (1, GRID.size)
    assert case.valid.shape == (1, GRID.size)
    assert case.field_validity["density"].shape == (1, GRID.size)
