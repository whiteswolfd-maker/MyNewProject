"""数组与两条坐标轴的关系必须**一次核完**（提示词 §15）。

这一组针对的是一类反复出现的漏法：**每个数组各查各的**。分散的后果不是"少查了
一项"，而是**两份数据用两套规则**——同一份导出换条读取路径读，一个报错、一个放行。

复核给出的具体漏法（这里逐条钉住）：

* 帧数错配的字段：``density (2, 3)`` 配 ``velocity (1, 3)``、时间两帧 —— 读取成功，
  然后查第二帧抛 ``IndexError``。**被判就绪的数据在正常消费时会崩**。
* 时间轴倒序或重复 —— 直接读取成功。
* 资格数组形状 ``(1, 2)`` —— 被 ``reshape(-1)`` 展平成 ``(2,)`` 通过。
* 状态码 ``999`` —— 当成普通无效帧放行。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.extracted_case import (
    DataError,
    load_extracted_case,
    validate_case_arrays,
)

GRID = np.linspace(0.002, 0.30, 3)


def _export(directory: Path, *, times, rho, u=None, P=None, mat_extra=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    n_t = len(times)
    np.savez(
        directory / "raw_line.npz",
        t=np.asarray(times, dtype=float), x=GRID,
        rho=np.asarray(rho, dtype=float),
        u=np.ones((n_t, 3)) if u is None else np.asarray(u, dtype=float),
        P=np.full((n_t, 3), 1.0e5) if P is None else np.asarray(P, dtype=float),
    )
    (directory / "metadata.json").write_text(json.dumps({
        "R_0": 0.05, "detonation_velocity_m_s": 6930.0,
        "material_partition_is_volume_fraction_based": True,
        # 结构化来源记录（提示词 §5）：自由文本与数字都不能当机器证据。
        "material_partition_source": {
            "method": "ale_volume_fraction",
            "producer": "cli.extract_profiles",
            "producer_version": 1,
            "fields": ["tnt_volume_fraction"],
            "input_identifier": "sha256:test",
            "detail": {"pure_product_threshold": 0.95},
        },
        "symmetry_factor": 8.0, "symmetry_share": 0.125,
    }), encoding="utf-8")
    material = np.where(GRID < 0.05, 0, 1)[None, :].repeat(n_t, axis=0)
    payload = {
        "material": material, "valid": material != 2,
        "contact_radius": np.full(n_t, 0.05), "shock_radius": np.full(n_t, 0.10),
        "contact_state": np.full(n_t, 2, dtype=np.int64),
        "shock_state": np.full(n_t, 2, dtype=np.int64),
        "contact_has_supervision": np.ones(n_t, dtype=bool),
        "shock_has_supervision": np.ones(n_t, dtype=bool),
        "field_valid_density": material != 2, "field_valid_velocity": material != 2,
        "field_valid_energy": material != 2, "field_valid_pressure": material != 2,
    }
    payload.update(mat_extra or {})
    np.savez(directory / "material.npz", **payload)
    return directory


# --------------------------------------------------------------------------
# 直接调校验函数
# --------------------------------------------------------------------------


def test_a_field_whose_frame_count_differs_from_the_time_axis_is_refused():
    """**这条是复核给的原反例。** 读取成功、然后查询崩掉，比直接拒绝危险得多。

    报错要说清是**布局**对不上（声明的是逐帧二维，来了一个 ``(1, 3)``），而不是
    只丢一句"expected (2, 3)"——后者把"这份文件是什么布局"这个问题绕开了。
    """
    with pytest.raises(DataError, match="declared layout"):
        validate_case_arrays(
            times=np.array([1.0e-5, 2.0e-5]),
            radius=GRID,
            profiles={
                "density": np.ones((2, 3)),
                "velocity": np.ones((1, 3)),      # 少了一帧
                "pressure": np.ones((2, 3)),
            },
            path="crafted",
        )


@pytest.mark.parametrize(
    "times, needle",
    [
        ([2.0e-5, 1.0e-5], "strictly increasing"),
        ([1.0e-5, 1.0e-5], "strictly increasing"),
        ([1.0e-5, np.nan], "non-finite"),
    ],
)
def test_the_time_axis_must_be_finite_and_strictly_increasing(times, needle):
    """排序、``searchsorted``、分层取帧全都建立在"时间严格递增"上，而它们都不报错。"""
    with pytest.raises(DataError, match=needle):
        validate_case_arrays(
            times=np.asarray(times, dtype=float), radius=GRID,
            profiles={"density": np.ones((2, 3))}, path="crafted",
        )


def test_the_radial_axis_must_be_strictly_increasing():
    with pytest.raises(DataError, match="strictly increasing"):
        validate_case_arrays(
            times=np.array([1.0e-5]), radius=np.array([0.1, 0.05, 0.2]),
            profiles={"density": np.ones((1, 3))}, path="crafted",
        )


def test_a_two_dimensional_eligibility_array_is_refused_not_flattened():
    """``(1, 2)`` 被 ``reshape(-1)`` 展平成 ``(2,)`` 之后长度检查会通过。

    **长度相同不代表格式正确**：一个二维的资格表说明它本来不是逐帧的量。
    """
    with pytest.raises(DataError, match="one boolean per frame"):
        validate_case_arrays(
            times=np.array([1.0e-5, 2.0e-5]), radius=GRID,
            profiles={"density": np.ones((2, 3))},
            eligibility={"contact_has_supervision": np.array([[True, False]])},
            path="crafted",
        )


def test_a_non_boolean_eligibility_array_is_refused():
    with pytest.raises(DataError, match="dtype"):
        validate_case_arrays(
            times=np.array([1.0e-5, 2.0e-5]), radius=GRID,
            profiles={"density": np.ones((2, 3))},
            eligibility={"contact_has_supervision": np.array([1.0, 0.0])},
            path="crafted",
        )


def test_an_unknown_state_code_is_a_format_error_not_an_unusable_frame():
    """不认识的码说明这份导出与当前读取器不是同一套约定。

    当成普通无效帧处理，会把"我们对不上这份数据"说成"这一帧没测到"——后者听起来
    像一条正常的观测结果。
    """
    with pytest.raises(DataError, match="does not define"):
        validate_case_arrays(
            times=np.array([1.0e-5, 2.0e-5]), radius=GRID,
            profiles={"density": np.ones((2, 3))},
            states={"contact_state": np.array([2, 999])},
            path="crafted",
        )


# --------------------------------------------------------------------------
# 走完整条读取路径
# --------------------------------------------------------------------------


def test_the_reader_refuses_a_frame_count_mismatch_end_to_end(tmp_path):
    """复核的原反例：读取成功 → 查询第二帧 ``IndexError``。现在在读取处就拒绝。"""
    directory = _export(
        tmp_path / "mismatch",
        times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)), u=np.ones((1, 3)),
    )
    with pytest.raises(DataError, match="declared layout"):
        load_extracted_case(directory, case_id="mismatch")


def test_the_reader_refuses_a_descending_time_axis_end_to_end(tmp_path):
    directory = _export(tmp_path / "descending", times=[2.0e-5, 1.0e-5], rho=np.ones((2, 3)))
    with pytest.raises(DataError, match="strictly increasing"):
        load_extracted_case(directory, case_id="descending")


def test_the_reader_refuses_an_unknown_state_code_end_to_end(tmp_path):
    directory = _export(
        tmp_path / "state999", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)),
        mat_extra={"contact_state": np.array([2, 999])},
    )
    with pytest.raises(DataError, match="does not define"):
        load_extracted_case(directory, case_id="state999")


def test_both_reading_paths_apply_the_same_rule(tmp_path):
    """**同一份错数据，两条路径必须都拒绝。**

    以前启发式路径核了密度形状与轴的单调性，体积分数路径两条都没核；同一份导出
    换个目录（有 / 没有 ``material.npz``）就读出两个结果，一个报错一个放行。
    """
    with_companion = _export(
        tmp_path / "with", times=[2.0e-5, 1.0e-5], rho=np.ones((2, 3)),
    )
    without_companion = _export(
        tmp_path / "without", times=[2.0e-5, 1.0e-5], rho=np.ones((2, 3)),
    )
    (without_companion / "material.npz").unlink()

    for directory in (with_companion, without_companion):
        with pytest.raises(DataError, match="strictly increasing"):
            load_extracted_case(directory, case_id=directory.name)


# --------------------------------------------------------------------------
# §4 原始结构必须在任何规范化之前检查
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "which, needle",
    [("times", "must be 1-D"), ("radius", "must be 1-D")],
)
def test_a_two_dimensional_axis_is_refused_not_flattened(tmp_path, which, needle):
    """**复核点名的原反例。** ``times=[[1,2]]``、``radius=[[1,2,3]]`` 以前读取成功。

    读取器先 ``reshape(-1)``、再把结果交给一个检查一维的函数——检查收到的已经是
    展平后的数组，原始的维度错误早就不在了。所以规范化必须排在检查**之后**。
    """
    directory = _export(
        tmp_path / which, times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)),
    )
    raw = dict(np.load(directory / "raw_line.npz", allow_pickle=True))
    if which == "times":
        raw["t"] = np.asarray(raw["t"]).reshape(1, -1)
    else:
        raw["x"] = np.asarray(raw["x"]).reshape(1, -1)
    np.savez(directory / "raw_line.npz", **raw)

    with pytest.raises(DataError, match=needle):
        load_extracted_case(directory, case_id=which)


def test_an_unknown_material_code_is_refused(tmp_path):
    """``material`` 里出现 999 时不能当成一个"别的区域"放行。"""
    directory = _export(tmp_path / "mat999", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)))
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["material"] = np.full(payload["material"].shape, 999)
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(DataError, match="does not define"):
        load_extracted_case(directory, case_id="mat999")


def test_an_all_nan_joint_valid_mask_is_refused(tmp_path):
    """联合 ``valid`` 全是 NaN 时不得被读成"整帧都有效"。"""
    directory = _export(tmp_path / "nanvalid", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)))
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["valid"] = np.full(payload["valid"].shape, np.nan)
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(DataError, match="dtype"):
        load_extracted_case(directory, case_id="nanvalid")


def test_legacy_front_state_codes_are_not_supervision_eligibility(tmp_path):
    """只有三态码、没有资格数组的旧导出不能进入新监督链。

    三态码说明"定位器找到了一个位置"，不说明"这个位置够格当标签"。新提取器
    显式写 ``contact_has_supervision`` / ``shock_has_supervision``；旧导出必须
    重提，不能在转换时把三态码直接升级成资格。
    """
    from spherical_blast_pinn.data.dyna_supervision import (
        DynaSupervisionError,
        supervision_from_extracted_case,
    )
    from spherical_blast_pinn.data.startup_supervision import TimeOrigin

    directory = _export(
        tmp_path / "legacy", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3))
    )
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload.pop("contact_has_supervision")
    payload.pop("shock_has_supervision")
    np.savez(directory / "material.npz", **payload)

    case = load_extracted_case(directory, case_id="legacy")
    assert case.front_supervision_eligibility["level"] == "legacy_state_code"
    origin = TimeOrigin(
        status="confirmed", origin_time_s=0.0,
        method="test", evidence="legacy eligibility check",
    )
    with pytest.raises(DynaSupervisionError, match="legacy front state codes"):
        supervision_from_extracted_case(
            case, time_origin=origin, epsilon_s=1.0e-7, synthetic=False
        )


def test_the_supervision_conversion_does_not_trust_that_the_case_was_read(tmp_path):
    """监督转换对**已经构造好的对象**重新核形状，不信任"据说读取过"。

    读取时的校验只保护"从文件读进来的那一份"；把 velocity 改成错误形状后再转换，
    必须失败，而不是产出一个按帧号错位的 ``SupervisionDataset``。
    """
    from spherical_blast_pinn.data.dyna_supervision import (
        DynaSupervisionError,
        supervision_from_extracted_case,
    )
    from spherical_blast_pinn.data.startup_supervision import TimeOrigin

    directory = _export(tmp_path / "mutated", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)))
    case = load_extracted_case(directory, case_id="mutated")
    origin = TimeOrigin(
        status="confirmed", origin_time_s=0.0,
        method="test", evidence="synthetic structural check",
    )
    baseline = supervision_from_extracted_case(
        case, time_origin=origin, epsilon_s=1.0e-7, synthetic=True
    )
    assert baseline.n_frames == 2

    case.velocity = np.ones((1, 3))          # 帧数与时间轴对不上
    with pytest.raises(DynaSupervisionError, match="velocity has shape"):
        supervision_from_extracted_case(
            case, time_origin=origin, epsilon_s=1.0e-7, synthetic=True
        )


def test_a_two_dimensional_state_code_is_refused_end_to_end(tmp_path):
    """**复核点名的那个：二维状态码数组。** 长度对得上不代表格式对——``(1, 2)``
    说明它本来就不是逐帧的量，放进 ``states`` 的维检查里拦下。
    """
    directory = _export(
        tmp_path / "state2d", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)),
        mat_extra={"contact_state": np.array([[2, 2]])},
    )
    with pytest.raises(DataError, match="one code per frame"):
        load_extracted_case(directory, case_id="state2d")


def test_a_two_dimensional_material_table_is_refused_end_to_end(tmp_path):
    """材料表也一样：它是 ``(n_t, n_x)`` 的场量，二维的二维要按布局核，不是展平。"""
    directory = _export(tmp_path / "mat2d", times=[1.0e-5, 2.0e-5], rho=np.ones((2, 3)))
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["material"] = payload["material"].reshape(1, -1)
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(DataError, match="'material'"):
        load_extracted_case(directory, case_id="mat2d")
