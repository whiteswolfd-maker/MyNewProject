"""Fixed-point PDE and interface residuals for two checkpoints.

Both models are evaluated on the same physical coordinates (t, r), restricted
to points inside both predicted domains.  The normalized mass, momentum and
energy residuals are computed directly from the mapped Euler equations, not
inferred from a training loss.  Interface compatibility is evaluated on each
model's own predicted contact surface.

Usage:
    python scripts/physical_residuals.py --config <cfg> \
        --checkpoint-a <2000.pt> --checkpoint-b <best.pt> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import (
    _case_specs,
    _load_checkpoint,
    _make_case,
    _networks,
    _section,
)
from spherical_blast_pinn.contact_air.losses import residual_scales
from spherical_blast_pinn.contact_air.mapping import assemble_residuals, build_geometry
from spherical_blast_pinn.contact_air.models import (
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)


def _subsample(case, limit: int):
    obs = case.observations
    mask = (
        np.asarray(obs.valid_density, dtype=bool)
        & np.asarray(obs.valid_velocity, dtype=bool)
        & np.asarray(obs.valid_pressure, dtype=bool)
        & np.isfinite(obs.density)
        & np.isfinite(obs.velocity)
        & np.isfinite(obs.pressure)
        # The shock trajectory and the model domain exist only on the aligned
        # segment [t0, t_end]; earlier cache rows are outside it.
        & (obs.time >= float(case.t0))
        & (obs.time <= float(case.t_end))
    )
    index = np.flatnonzero(mask)
    if index.size > limit:
        pick = np.linspace(0, index.size - 1, limit).astype(np.int64)
        index = index[np.unique(pick)]
    t = torch.as_tensor(obs.time[index], dtype=torch.float64)
    r = torch.as_tensor(obs.radius[index], dtype=torch.float64)
    return t, r


def _domain(case, contact, t: torch.Tensor, r: torch.Tensor):
    r0 = torch.full_like(t, float(case.charge_radius_m))
    rs, ds = case.shock_tensors(t)
    closure = closure_from_precursor(case.precursor, like=t)
    scales = scales_from_precursor(case.precursor, like=t)
    with torch.enable_grad():
        fields = contact(t, r0, rs, ds, closure, create_graph=False)
    rc = fields.radius.detach().reshape(-1)
    rc_dot = fields.speed.detach().reshape(-1)
    inside = (r >= rc) & (r <= rs) & torch.isfinite(rc) & torch.isfinite(rc_dot)
    return inside, rc, rc_dot, rs, ds, closure, scales


def _residual_rms(case, air, t, r, rc, rc_dot, rs, ds, scales, keep):
    if int(keep.sum()) == 0:
        return {"mass": None, "momentum": None, "energy": None, "n": 0}
    length = (rs - rc)[keep]
    if not bool(torch.all(length > 0.0)):
        raise RuntimeError(f"{case.case_id}: non-positive layer thickness on kept points")
    zeta = ((r - rc)[keep] / length).reshape(-1, 1)
    fields, derivatives = air_fields_with_derivatives(
        air, zeta, t[keep], torch.full_like(t[keep], float(case.charge_radius_m)),
        scales, create_graph=False,
    )
    geometry = build_geometry(
        rc[keep].reshape(-1, 1), rc_dot[keep].reshape(-1, 1),
        rs[keep].reshape(-1, 1), ds[keep].reshape(-1, 1), zeta,
    )
    residuals = assemble_residuals(
        fields, derivatives, geometry,
        gamma=torch.full_like(fields.density, float(case.gamma_air)),
    )
    scale = residual_scales(case)
    out = {"n": int(keep.sum())}
    for name, values, key in (
        ("mass", residuals.mass, "mass_residual"),
        ("momentum", residuals.momentum, "momentum_residual"),
        ("energy", residuals.energy, "energy_residual"),
    ):
        normalized = values.reshape(-1) / float(scale[key])
        out[name] = float(torch.sqrt(torch.mean(normalized * normalized)).detach())
    return out


def _interface_errors(case, contact, air, *, points: int):
    t = torch.linspace(
        float(case.t0), float(case.t_end), int(points), dtype=torch.float64
    )
    r0 = torch.full_like(t, float(case.charge_radius_m))
    rs, ds = case.shock_tensors(t)
    closure = closure_from_precursor(case.precursor, like=t)
    scales = scales_from_precursor(case.precursor, like=t)
    with torch.enable_grad():
        c_fields = contact(t, r0, rs, ds, closure, create_graph=False)
        air_fields = air(
            torch.zeros_like(t).reshape(-1, 1), t.reshape(-1, 1),
            r0.reshape(-1, 1), scales,
        )
    velocity = (air_fields.velocity - c_fields.speed) / float(case.u_scale)
    pressure = (air_fields.pressure - c_fields.pressure) / float(case.p_scale)
    return {
        "n": int(t.numel()),
        "velocity_rms": float(torch.sqrt(torch.mean(velocity * velocity)).detach()),
        "pressure_rms": float(torch.sqrt(torch.mean(pressure * pressure)).detach()),
    }


def _pool(items, key: str) -> float | None:
    total = sum(item["n"] for item in items)
    if total == 0:
        return None
    return float(math.sqrt(sum(
        item[key] ** 2 * item["n"] for item in items if item[key] is not None
    ) / total))


def _aggregate(rows: list[dict], block: str) -> dict:
    return {
        "mass": _pool([row[block] for row in rows], "mass"),
        "momentum": _pool([row[block] for row in rows], "momentum"),
        "energy": _pool([row[block] for row in rows], "energy"),
        "n": sum(row[block]["n"] for row in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-a", default="checkpoint_a")
    parser.add_argument("--label-b", default="checkpoint_b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--points-per-case", type=int, default=20000)
    parser.add_argument("--interface-points", type=int, default=128)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)

    def load(path):
        contact, air = _networks(section)
        payload = _load_checkpoint(Path(path), contact, air)
        return contact, air, int(payload.get("loop", -1))

    contact_a, air_a, loop_a = load(args.checkpoint_a)
    contact_b, air_b, loop_b = load(args.checkpoint_b)

    cases: dict[str, dict] = {}
    groups: dict[str, list[dict]] = {"train": [], "validation": []}
    interface: dict[str, list[dict]] = {"train": [], "validation": []}

    for spec in _case_specs(section):
        case = _make_case(section, spec, config, allow_synthetic=False)
        t, r = _subsample(case, int(args.points_per_case))
        inside_a, rc_a, rcd_a, rs_a, ds_a, _, scales_a = _domain(
            case, contact_a, t, r
        )
        inside_b, rc_b, rcd_b, rs_b, ds_b, _, scales_b = _domain(
            case, contact_b, t, r
        )
        common = inside_a & inside_b
        res_a = _residual_rms(case, air_a, t, r, rc_a, rcd_a, rs_a, ds_a, scales_a, common)
        res_b = _residual_rms(case, air_b, t, r, rc_b, rcd_b, rs_b, ds_b, scales_b, common)
        inter_a = _interface_errors(
            case, contact_a, air_a, points=int(args.interface_points)
        )
        inter_b = _interface_errors(
            case, contact_b, air_b, points=int(args.interface_points)
        )
        row = {
            "case_id": case.case_id,
            "role": case.role,
            "candidates": int(t.numel()),
            "common": int(common.sum()),
            "excluded_fraction": float(1.0 - int(common.sum()) / max(t.numel(), 1)),
            "a": res_a, "b": res_b,
        }
        cases[case.case_id] = {
            "role": case.role,
            "residuals": row,
            "interface": {"a": inter_a, "b": inter_b},
        }
        groups[case.role].append(row)
        interface[case.role].append({"a": inter_a, "b": inter_b})

    def interface_pool(rows, tag, key):
        total = sum(row[tag]["n"] for row in rows)
        if total == 0:
            return None
        return float(math.sqrt(sum(
            row[tag][key] ** 2 * row[tag]["n"] for row in rows
        ) / total))

    summary = {}
    for role in ("train", "validation"):
        group_rows = groups[role]
        summary[role] = {
            args.label_a: _aggregate(group_rows, "a"),
            args.label_b: _aggregate(group_rows, "b"),
            "candidates": sum(row["candidates"] for row in group_rows),
            "common": sum(row["common"] for row in group_rows),
            "interface": {
                label: {
                    "velocity_rms": interface_pool(interface[role], tag, "velocity_rms"),
                    "pressure_rms": interface_pool(interface[role], tag, "pressure_rms"),
                    "n": sum(row[tag]["n"] for row in interface[role]),
                }
                for label, tag in ((args.label_a, "a"), (args.label_b, "b"))
            },
        }
        summary[role]["excluded_fraction"] = 1.0 - (
            summary[role]["common"] / max(summary[role]["candidates"], 1)
        )

    payload = {
        "config": str(args.config),
        "checkpoints": {
            args.label_a: {"path": str(args.checkpoint_a), "loop": loop_a},
            args.label_b: {"path": str(args.checkpoint_b), "loop": loop_b},
        },
        "protocol": {
            "points": "deterministic subsample of valid observations per case",
            "common": "inside both predicted [R_c, R_s] domains",
            "residuals": "mapped Euler residuals divided by residual_scales; RMS",
            "interface": "each model on its own R_c(t), zeta=0",
        },
        "groups": summary,
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(value):
        return "   n/a" if value is None else f"{value:.4e}"

    print(f"{'group':10s} {'model':12s} {'mass':>10s} {'momentum':>10s} {'energy':>10s} {'n':>8s}")
    for role, entry in summary.items():
        for tag in (args.label_a, args.label_b):
            row = entry[tag]
            print(
                f"{role:10s} {tag:12s} {fmt(row['mass']):>10s} "
                f"{fmt(row['momentum']):>10s} {fmt(row['energy']):>10s} {row['n']:8d}"
            )
        print(f"  excluded fraction {entry['excluded_fraction']:.3f} "
              f"({entry['common']}/{entry['candidates']})")
        for tag in (args.label_a, args.label_b):
            row = entry["interface"][tag]
            print(f"  interface {tag:12s} u {fmt(row['velocity_rms'])}  p {fmt(row['pressure_rms'])}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
