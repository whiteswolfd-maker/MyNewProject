"""The frame-index sampler must reproduce the original draw sequence exactly."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.data import AirObservationStore
from spherical_blast_pinn.contact_air.models import AirNet, ContactNet
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    seed_everything,
)


def _reference_sample(store, count, generator, *, t_min=None, t_max=None):
    """The pre-index implementation, verbatim, used as the oracle."""
    empty = torch.zeros(0, dtype=torch.float64)
    empty_bool = empty.bool()
    if store.size == 0 or count <= 0:
        return {
            "t": empty, "radius": empty, "rho": empty, "u": empty, "p": empty,
            "valid_rho": empty_bool, "valid_u": empty_bool,
            "valid_p": empty_bool, "count": 0,
        }
    allowed = np.ones(store.size, dtype=bool)
    if t_min is not None:
        allowed &= store.time >= float(t_min)
    if t_max is not None:
        allowed &= store.time <= float(t_max)
    frames = np.unique(store.frame_index[allowed])
    if frames.size == 0:
        return {
            "t": empty, "radius": empty, "rho": empty, "u": empty, "p": empty,
            "valid_rho": empty_bool, "valid_u": empty_bool,
            "valid_p": empty_bool, "count": 0,
        }
    by_frame = {
        int(frame): np.flatnonzero(allowed & (store.frame_index == frame))
        for frame in frames
    }
    chosen_frames = torch.randint(
        0, int(frames.size), (int(count),), generator=generator
    )
    frame_counts = torch.bincount(
        chosen_frames, minlength=int(frames.size)
    ).cpu().numpy()
    selected_parts = []
    for slot, frame in enumerate(frames):
        n_take = int(frame_counts[slot])
        if n_take <= 0:
            continue
        rows = by_frame[int(frame)]
        picks = torch.randint(
            0, int(rows.size), (n_take,), generator=generator
        ).cpu().numpy()
        selected_parts.append(rows[picks])
    selected = (
        np.concatenate(selected_parts)
        if selected_parts else np.zeros(0, dtype=np.int64)
    )
    return {
        "t": torch.as_tensor(store.time[selected]),
        "radius": torch.as_tensor(store.radius[selected]),
        "rho": torch.as_tensor(store.density[selected]),
        "u": torch.as_tensor(store.velocity[selected]),
        "p": torch.as_tensor(store.pressure[selected]),
        "valid_rho": torch.as_tensor(store.valid_density[selected]).bool(),
        "valid_u": torch.as_tensor(store.valid_velocity[selected]).bool(),
        "valid_p": torch.as_tensor(store.valid_pressure[selected]).bool(),
        "count": int(selected.size),
    }


def _assert_same(a, b):
    assert a["count"] == b["count"]
    for key in ("t", "radius", "rho", "u", "p"):
        assert torch.equal(a[key], b[key]), key
    for key in ("valid_rho", "valid_u", "valid_p"):
        assert torch.equal(a[key], b[key]), key


def _store(frame_counts=(3, 1, 7, 2), *, interleaved=False):
    counts = np.asarray(frame_counts, dtype=np.int64)
    frames = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    if interleaved:
        frames = np.array([0, 1, 0, 2, 2, 2, 3, 0, 2, 2, 2, 2, 1], dtype=np.int64)
    size = frames.size
    time = 1.0e-3 * frames.astype(np.float64)
    radius = np.linspace(0.01, 0.2, size)
    density = np.arange(size, dtype=np.float64) + 1.0
    velocity = -density
    pressure = 3.0 * density
    valid_density = np.ones(size, bool)
    valid_density[::5] = False
    valid_velocity = np.ones(size, bool)
    valid_velocity[1::7] = False
    valid_pressure = np.ones(size, bool)
    valid_pressure[2::9] = False
    return AirObservationStore(
        time=time, radius=radius, density=density, velocity=velocity,
        pressure=pressure, valid_density=valid_density,
        valid_velocity=valid_velocity, valid_pressure=valid_pressure,
        frame_index=frames,
    )


@pytest.mark.parametrize("window", [(None, None), (0.5e-3, 2.0e-3), (0.0, 1.0e-3)])
def test_index_sampler_matches_reference_for_100_draws(window):
    store = _store()
    t_min, t_max = window
    gen_new = torch.Generator().manual_seed(12345)
    gen_ref = torch.Generator().manual_seed(12345)
    for _ in range(100):
        new = store.sample(37, gen_new, t_min=t_min, t_max=t_max)
        ref = _reference_sample(store, 37, gen_ref, t_min=t_min, t_max=t_max)
        _assert_same(new, ref)


def test_index_sampler_matches_reference_after_subset():
    store = _store()
    mask = np.ones(store.size, bool)
    mask[::4] = False          # gaps inside frames
    subset = store.subset(mask)
    gen_new = torch.Generator().manual_seed(777)
    gen_ref = torch.Generator().manual_seed(777)
    for _ in range(100):
        new = subset.sample(23, gen_new, t_min=0.0, t_max=2.0e-3)
        ref = _reference_sample(subset, 23, gen_ref, t_min=0.0, t_max=2.0e-3)
        _assert_same(new, ref)


def test_interleaved_frame_order_uses_the_reference_scan():
    store = _store(interleaved=True)
    assert store._frames_sorted is False
    gen_new = torch.Generator().manual_seed(9)
    gen_ref = torch.Generator().manual_seed(9)
    for _ in range(20):
        _assert_same(
            store.sample(17, gen_new),
            _reference_sample(store, 17, gen_ref),
        )


def test_empty_window_returns_empty_draw():
    store = _store()
    draw = store.sample(10, torch.Generator().manual_seed(1),
                        t_min=10.0, t_max=20.0)
    assert draw["count"] == 0
    assert draw["t"].numel() == 0


def _trainer(cases, validation, output_dir, seed=11):
    seed_everything(seed)
    contact = ContactNet(width=8, depth=2, time_reference_s=0.01)
    air = AirNet(width=8, depth=2, time_reference_s=0.01)
    config = TrainConfig(
        n_pde=16, n_ic=8, n_right=8, n_data=8, n_contact=8,
        loops=2, n_schedule=10, seed=seed,
        validation_holdout_fraction=0.0,
        validation_data_points=8, validation_contact_points=8,
        contact_data_weight=3.0,
    )
    return AlternatingTrainer(
        cases=cases, contact_net=contact, air_net=air, config=config,
        output_dir=output_dir, external_validation_cases=validation,
    )


def test_two_short_steps_match_with_the_scan_sampler(tmp_path, monkeypatch):
    train = [
        synthetic_case_for(0.05, case_id="s50", role="train",
                           n_frames=21, n_radial=128),
        synthetic_case_for(0.05, case_id="s100", role="train",
                           n_frames=21, n_radial=128),
    ]
    validation = [
        synthetic_case_for(0.05, case_id="s150", role="validation",
                           n_frames=21, n_radial=128),
    ]

    fast = _trainer(train, validation, tmp_path / "fast")
    fast_records = [fast.step(), fast.step()]

    monkeypatch.setattr(
        AirObservationStore, "sample", AirObservationStore._sample_scan
    )
    scan = _trainer(train, validation, tmp_path / "scan")
    scan_records = [scan.step(), scan.step()]

    for first, second in zip(fast_records, scan_records):
        for key in ("air_loss", "contact_loss", "lambda_pde", "lambda_data"):
            assert first[key] == pytest.approx(second[key], rel=0.0, abs=0.0)
        for key, value in first["contact_parts"].items():
            assert value == pytest.approx(second["contact_parts"][key],
                                          rel=0.0, abs=0.0)
        assert first["counts"] == second["counts"]

    for pair in (
        (fast.contact_net, scan.contact_net),
        (fast.air_net, scan.air_net),
    ):
        for left, right in zip(pair[0].parameters(), pair[1].parameters()):
            assert torch.equal(left, right)
