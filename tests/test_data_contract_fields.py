"""数据契约必须是一道**闸门**，不是一张盖章（提示词 §15）。

这里用的都是**构造出来的**导出目录，条件逐项可控。测的是契约的分支行为：

* 缺球对称份额 ⇒ 未就绪；
* 缺逐字段有效性 ⇒ 未就绪（而且**不能**因为属性存在就算通过）；
* 缺前沿监督资格 ⇒ 未就绪；
* 材料分区不是体积分数 ⇒ 未就绪；
* 全都补齐 ⇒ 就绪。

真实数据上的表现由 ``tests/test_real_data_chain.py`` 负责。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.contract import evaluate_case_contract, evaluate_data_contract
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.physics import material as mat


def _base_arrays(n_frames: int = 6, n_cells: int = 200):
    grid = np.linspace(0.002, 0.30, n_cells)
    times = np.linspace(0.0, 3.0e-4, n_frames)
    density = np.empty((n_frames, n_cells))
    pressure = np.empty_like(density)
    velocity = np.zeros_like(density)
    for i in range(n_frames):
        contact = 0.05 * (1.0 + 0.05 * i)
        shock = contact * 2.0
        inside = grid < contact
        density[i] = np.where(inside, 1600.0, np.where(grid < shock, 6.0 * mat.AIR_DENSITY, mat.AIR_DENSITY))
        pressure[i] = np.where(inside, 1.0e10, np.where(grid < shock, 25.0 * mat.AIR_PRESSURE, mat.AIR_PRESSURE))
    return grid, times, density, velocity, pressure


def _write_export(
    directory: Path,
    *,
    with_companion: bool = True,
    with_field_validity: bool = True,
    with_symmetry: bool = True,
    with_eligibility: bool = True,
    with_states: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    grid, times, density, velocity, pressure = _base_arrays()
    np.savez(
        directory / "raw_line.npz",
        t=times, x=grid, rho=density, u=velocity, P=pressure,
    )
    metadata = {
        "R_0": 0.05,
        "x_end": float(grid[-1]),
        "detonation_velocity_m_s": 6930.0,
    }
    if with_companion:
        # 提取器把"分区是不是从体积分数读的"写在元数据里。读取器**照读**，
        # 不自己盖章——否则判据只是从"信任说明文字"换成"信任读取器"。
        metadata["material_partition_is_volume_fraction_based"] = True
        metadata["material_partition_source"] = "ALE material volume fraction (d3plot)"
    if with_symmetry:
        metadata["symmetry_factor"] = 8.0
        metadata["symmetry_share"] = 0.125
    if with_eligibility:
        metadata["front_supervision_eligibility"] = {
            "contact": {"status_counts": {"reconstructed": 6}, "n_frames_eligible": 6},
            "shock": {"status_counts": {"confirmed": 4, "doubtful": 2}, "n_frames_eligible": 4},
        }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    if with_companion:
        n_frames, n_cells = density.shape
        contact = np.array([0.05 * (1.0 + 0.05 * i) for i in range(n_frames)])
        shock = contact * 2.0
        material = np.where(
            grid[None, :] < contact[:, None], 0,
            np.where(grid[None, :] < shock[:, None], 1, 2),
        )
        valid = material != 2
        payload = {
            "material": material,
            "valid": valid,
            "contact_radius": contact,
            "shock_radius": shock,
        }
        if with_states:
            payload["contact_state"] = np.full(n_frames, 2, dtype=np.int64)
            payload["shock_state"] = np.full(n_frames, 2, dtype=np.int64)
            # 提取器写出的逐帧资格数组（最强的一档来源）。最后一帧不具资格，
            # 好让"资格"与"半径有限"真的能区分开。
            contact_ok = np.ones(n_frames, dtype=bool)
            contact_ok[-1] = False
            payload["contact_has_supervision"] = contact_ok
            payload["shock_has_supervision"] = contact_ok.copy()
        if with_field_validity:
            payload["field_valid_density"] = valid
            payload["field_valid_velocity"] = valid
            payload["field_valid_energy"] = valid
            payload["field_valid_pressure"] = valid
        np.savez(directory / "material.npz", **payload)
    return directory


def _load(directory: Path):
    return load_extracted_case(directory, case_id="crafted")


def test_a_complete_export_satisfies_the_contract(tmp_path):
    case = _load(_write_export(tmp_path / "complete"))
    verdict = evaluate_case_contract(case)
    assert verdict.ready, verdict.reasons


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"with_symmetry": False}, "对称份额"),
        ({"with_field_validity": False}, "逐字段有效性"),
        # 既没有资格数组、也没有三态码 —— 只剩"半径有限即可"这一档
        ({"with_states": False, "with_eligibility": False}, "未核实"),
        ({"with_companion": False}, "体积分数"),
    ],
)
def test_each_missing_piece_blocks_the_contract(tmp_path, kwargs, needle):
    case = _load(_write_export(tmp_path / "partial", **kwargs))
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize(
    "symmetry, needle",
    [
        ({"symmetry_factor": -8.0}, "有限正数"),
        ({"symmetry_factor": float("nan")}, "有限正数"),
        ({"symmetry_factor": 8.0, "symmetry_share": 8.0}, "互相矛盾"),
    ],
)
def test_an_impossible_symmetry_factor_blocks_the_contract(tmp_path, symmetry, needle):
    """球对称份额必须是**有限正数**，两个键同时给出时还必须自洽。

    以前这里只判"真值"：``-8`` 与 ``nan`` 都是真值，于是都能通过。份额是体积等效
    半径与覆盖率的分母，一个负的或 NaN 的分母会把折算出来的每一个半径都毁掉，
    而报表上只会看到一个数字。
    """
    directory = _write_export(tmp_path / "sym", with_symmetry=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(symmetry)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    verdict = evaluate_case_contract(_load(directory))
    assert not verdict.ready
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_a_denial_in_the_source_text_is_not_read_as_a_confirmation(tmp_path):
    """材料来源按**布尔标志**判断，不按字符串匹配。

    ``"not from volume fraction"`` 里也含 ``"volume fraction"``。用子串匹配的话，
    一句明确的**否认**会被读成**肯定**——而且读起来完全正常。
    """
    case = _load(_write_export(tmp_path / "denial"))

    # 说明文字里写着 volume fraction、但**没有**机器可判的标志 ⇒ 不能放行。
    # 旧实现正是只看这句话，所以它能通过。
    case.source.pop("material_partition_is_volume_fraction_based", None)
    case.source["material_masks_from"] = "ALE material volume fraction (looks right)"
    verdict = evaluate_case_contract(case)
    assert not verdict.ready, (
        "没有机器可判的标志时放行了；闸门不能靠说明文字的子串"
    )

    # 反过来：标志为假时，一句自称权威的说明文字也翻不了案。
    case.source["material_partition_is_volume_fraction_based"] = False
    assert not evaluate_case_contract(case).ready

    # 标志是真才算数——说明文字怎么写都不影响结论。
    case.source["material_partition_is_volume_fraction_based"] = True
    case.source["material_masks_from"] = "NOT from volume fraction, a heuristic detector"
    assert evaluate_case_contract(case).ready, (
        "闸门看的应当是布尔标志，而不是说明文字"
    )


def test_the_reader_does_not_stamp_the_partition_flag_itself(tmp_path):
    """来源标志必须**照读**提取器的元数据，不能由读取器自己盖。

    判据从"信任说明文字"换成"信任读取器"并没有解决来源证据的问题：读取器只是
    照着"有这个文件"推了一句。元数据明确写着 False 时，读出来必须还是 False。
    """
    directory = _write_export(tmp_path / "flagfalse")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata["material_partition_is_volume_fraction_based"] = False
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    case = _load(directory)
    assert case.source["material_partition_is_volume_fraction_based"] is False, (
        "读取器把提取器明确写下的 False 覆盖成了 True"
    )
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any("体积分数" in reason for reason in verdict.reasons)

    # 元数据里根本没有这个标志 ⇒ 未核实，同样拒绝
    metadata.pop("material_partition_is_volume_fraction_based")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    case = _load(directory)
    assert case.source["material_partition_is_volume_fraction_based"] is None
    assert not evaluate_case_contract(case).ready


@pytest.mark.parametrize(
    "symmetry, needle",
    [
        ({"symmetry_factor": 8.0, "full_sphere_factor": 4.0}, "互相矛盾"),
        ({"symmetry_factor": "eight"}, "不是数字"),
        ({"symmetry_share": "1/8"}, "不是数字"),
    ],
)
def test_alias_conflicts_and_non_numeric_symmetry_are_reported(
    tmp_path, symmetry, needle
):
    """别名必须互相核对；非数字要被**报告**，不能抛裸异常。

    以前是"找到第一个别名就 break"：同时写着 8 与 4 时，读第一个的人算出 8、
    读第二个的人算出 4，而契约只说"有"。字符串形式的份额则会让 ``float()`` 抛出
    一个裸 ``ValueError``——契约是报告工具，崩掉就什么都看不到。
    """
    directory = _write_export(tmp_path / "alias", with_symmetry=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(symmetry)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    verdict = evaluate_case_contract(_load(directory))
    assert not verdict.ready
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_a_non_finite_time_axis_blocks_the_contract(tmp_path):
    """时间轴含非有限值时不能放行。

    ``nan`` 会让排序、``searchsorted`` 与时间窗口判断全部给出无意义的结果，
    而它们一个都不报错。
    """
    case = _load(_write_export(tmp_path / "time"))
    case.times = np.array([0.0, np.nan, 2.0e-4])
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any("非有限" in reason for reason in verdict.reasons), verdict.reasons


def test_an_all_nan_validity_mask_cannot_read_back_as_all_true(tmp_path):
    """全是 NaN 的掩码不得被读成全 True。

    ``np.asarray(x, dtype=bool)`` 会把 ``nan`` 变成 ``True``（``bool(nan)`` 为真），
    于是一张什么都没说的掩码读进来是"全有效"，而形状检查对此毫无办法。
    """
    directory = _write_export(tmp_path / "nanmask")
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["field_valid_density"] = np.full(payload["valid"].shape, np.nan)
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(Exception, match="dtype"):
        _load(directory)


def test_a_length_mismatch_in_the_companion_is_refused_not_downgraded(tmp_path):
    """配套文件的数组长度对不上时**抛错**，不能"忽略它、退回更弱的判据"。

    退回等于把一份**对不上**的导出当成一份"证据更少"的导出，而后者仍然会被当成
    同一批数据用。
    """
    directory = _write_export(tmp_path / "length")
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["contact_has_supervision"] = np.array([True, True])
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(Exception, match="do not belong together"):
        _load(directory)


def test_the_contract_does_not_accept_a_missing_record_as_a_valid_one(tmp_path):
    """**缺失必须显式。** 属性存在但为 ``None`` 时，契约必须判未就绪。

    这条是针对一种很具体的假修复：把 ``hasattr`` 当判据。``field_validity`` 与
    ``front_supervision_eligibility`` 是带默认值的字段，``hasattr`` 恒为真——用
    它做判据的话，"这份导出没记录"和"这份导出都有效"在契约上变成同一件事，那句
    检查永远通过。
    """
    case = _load(_write_export(tmp_path / "nofv", with_field_validity=False,
                               with_states=False, with_eligibility=False))
    assert case.field_validity is None, "缺字段时必须是 None，不能补一张全真掩码"
    # 资格那一项不是 None，而是"最弱的一档 + 明确写着未核实"
    assert case.front_supervision_eligibility["contact_eligible"] is None
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any("逐字段有效性" in r for r in verdict.reasons)
    assert any("未核实" in r for r in verdict.reasons), verdict.reasons


def test_a_malformed_field_validity_companion_is_refused(tmp_path):
    """形状对不上就拒绝，不能静默切一刀让它"能用"。"""
    directory = _write_export(tmp_path / "badshape")
    payload = dict(np.load(directory / "material.npz", allow_pickle=True))
    payload["field_valid_density"] = payload["field_valid_density"][:, :5]
    np.savez(directory / "material.npz", **payload)
    with pytest.raises(Exception, match="per-field validity"):
        _load(directory)


def test_a_batch_is_ready_only_when_every_case_is(tmp_path):
    """一个算例没有权威分区，就不能拿整批做正式训练（§15）。"""
    good = _load(_write_export(tmp_path / "good"))
    bad = _load(_write_export(tmp_path / "bad", with_companion=False))

    assert evaluate_data_contract([good]).ready
    mixed = evaluate_data_contract([good, bad])
    assert not mixed.ready
    assert mixed.detail["n_ready"] == 1
    assert len(mixed.detail["cases"]) == 2


def test_the_verdict_records_what_it_looked_at(tmp_path):
    """就绪与否都要留下可核对的字段，否则"就绪"只是一个字。"""
    case = _load(_write_export(tmp_path / "detail"))
    verdict = evaluate_case_contract(case)
    assert verdict.ready, verdict.reasons
    detail = verdict.detail
    assert detail["symmetry_factor"] == 8.0
    assert detail["detonation_velocity_m_s"] == pytest.approx(6930.0)
    assert detail["material_partition_is_volume_fraction_based"] is True
    assert detail["field_validity_fields"] == [
        "density", "pressure", "specific_energy", "velocity"
    ]
    sources = detail["front_eligibility_source"]
    assert set(sources) == {"contact", "shock"}
    assert all("eligibility array" in value for value in sources.values())
    # 最后一帧不具资格，所以资格计数比帧数少一 —— 这正是"资格"与"半径有限"分开了
    assert detail["front_eligibility_counts"] == {"contact": 5, "shock": 5}
    assert detail["n_frames"] == 6
