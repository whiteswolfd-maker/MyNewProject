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
        ({"with_eligibility": False}, "前沿监督资格"),
        ({"with_companion": False}, "体积分数"),
    ],
)
def test_each_missing_piece_blocks_the_contract(tmp_path, kwargs, needle):
    case = _load(_write_export(tmp_path / "partial", **kwargs))
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_the_contract_does_not_accept_a_missing_record_as_a_valid_one(tmp_path):
    """**缺失必须显式。** 属性存在但为 ``None`` 时，契约必须判未就绪。

    这条是针对一种很具体的假修复：把 ``hasattr`` 当判据。``field_validity`` 与
    ``front_supervision_eligibility`` 是带默认值的字段，``hasattr`` 恒为真——用
    它做判据的话，"这份导出没记录"和"这份导出都有效"在契约上变成同一件事，那句
    检查永远通过。
    """
    case = _load(_write_export(tmp_path / "nofv", with_field_validity=False,
                               with_eligibility=False))
    assert case.field_validity is None, "缺字段时必须是 None，不能补一张全真掩码"
    assert case.front_supervision_eligibility is None
    verdict = evaluate_case_contract(case)
    assert not verdict.ready
    assert any("逐字段有效性" in r for r in verdict.reasons)
    assert any("前沿监督资格" in r for r in verdict.reasons)


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
    detail = verdict.detail
    assert detail["symmetry_factor"] == 8.0
    assert detail["detonation_velocity_m_s"] == pytest.approx(6930.0)
    assert "volume fraction" in detail["material_masks_from"].lower()
    assert detail["field_validity_fields"] == [
        "density", "pressure", "specific_energy", "velocity"
    ]
    assert set(detail["front_supervision_eligibility"]) == {"contact", "shock"}
