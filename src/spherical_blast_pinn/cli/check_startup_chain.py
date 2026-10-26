"""检查新启动场能否直接驱动完整物理批次、总损失与一次反向传播。

这是一次**装配检查**，不是训练：不建优化器、不更新参数、不写检查点，也不改变
``training_qualified``。输入是从磁盘重新加载的 ``handoff.json`` / ``handoff.npz``。

用法::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.check_startup_chain \
        --handoff-json outputs/planar_startup/<model>/<key>/handoff.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..losses.objective import contact_pressure_scale, total_loss
from ..models.air_net import AirNet
from ..models.coupled_model import CoupledModel
from ..models.front_net import FrontNet
from ..models.product_net import ProductNet
from ..physics.similitude import PlanarStartupScales
from ..physics.two_medium_startup_handoff import load_planar_startup_handoff
from ..sampling.sampler import RadiusSampler, SamplingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=1.0e-6,
                        help="duration from the network start, seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--n-pde", type=int, default=256)
    parser.add_argument("--n-ic", type=int, default=64)
    parser.add_argument("--n-interface", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


#: 每个组都必须真的参与计算图：这是“连通性”检查，不是“每个参数都非零”。
REQUIRED_GRADIENT_GROUPS = ("product", "air", "front")


def _grad_summary(model: CoupledModel) -> dict[str, dict[str, float | bool]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, parameters in model.parameter_groups().items():
        grads = [p.grad for p in parameters if p.grad is not None]
        summary[name] = {
            "n_parameters": len(parameters),
            "n_with_gradient": len(grads),
            "all_finite": bool(all(torch.isfinite(g).all() for g in grads)),
            "max_abs": float(max((g.abs().max() for g in grads), default=torch.zeros(1))),
            "any_nonzero": bool(any(float(g.abs().max()) > 0.0 for g in grads)),
        }
    return summary


def _failure_reasons(report: dict[str, Any]) -> list[str]:
    """成功判据：损失有限 **且** 每个组都有有限、非全零的梯度。

    ``all([])`` 为真是这里最容易踩的坑：没有梯度时“所有梯度都有限”自动成立。
    所以先要求 ``n_with_gradient > 0``，再谈有限性与非零。不要求每个参数都
    非零——整组有一个非零梯度就说明该网络确实连在损失上。
    """
    reasons: list[str] = []
    loss = report.get("loss") if isinstance(report.get("loss"), dict) else {}
    if loss.get("all_finite") is not True:
        reasons.append("loss: total or a group value is not finite")

    gradients = report.get("gradients") if isinstance(report.get("gradients"), dict) else {}
    for name in REQUIRED_GRADIENT_GROUPS:
        entry = gradients.get(name)
        if not isinstance(entry, dict):
            reasons.append(f"{name}: no gradient record in the report")
            continue
        n_parameters = int(entry.get("n_parameters", 0))
        n_with_gradient = int(entry.get("n_with_gradient", 0))
        if n_with_gradient == 0:
            reasons.append(f"{name}: no parameter received a gradient")
        elif n_with_gradient != n_parameters:
            reasons.append(
                f"{name}: only {n_with_gradient}/{n_parameters} parameters received a gradient"
            )
        if entry.get("all_finite") is not True:
            reasons.append(f"{name}: a produced gradient is not finite")
        if entry.get("any_nonzero") is not True:
            reasons.append(f"{name}: every produced gradient is exactly zero")
    return reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_default_dtype(torch.float64)

    field = load_planar_startup_handoff(args.handoff_json)
    if field.checks.get("passed") is not True:
        raise SystemExit("the loaded startup field did not pass its consistency checks")
    scales = PlanarStartupScales.for_startup_field(field, time_scale=args.time_scale)
    sampler = RadiusSampler(
        field=field,
        scales=scales,
        tau_end=args.tau_end,
        config=SamplingConfig(
            n_pde_product=args.n_pde,
            n_pde_air=args.n_pde,
            n_ic_product=args.n_ic,
            n_ic_air=args.n_ic,
            n_contact=args.n_interface,
            n_shock=args.n_interface,
        ),
        seed=args.seed,
    )
    batch = sampler.batch()
    model = CoupledModel(
        product=ProductNet(scales=scales, width=args.width, depth=args.depth, seed=args.seed),
        air=AirNet(scales=scales, width=args.width, depth=args.depth, seed=args.seed + 1),
        front=FrontNet(
            scales=scales,
            width=args.width,
            depth=args.depth,
            initial_contact_hat=float(field.contact_radius / field.charge_radius),
            initial_gap_hat=float(
                (field.shock_radius - field.contact_radius) / field.charge_radius
            ),
            seed=args.seed + 2,
        ),
        scales=scales,
    )
    pressure = contact_pressure_scale(field)
    loss = total_loss([batch], model, contact_pressure=pressure)
    loss.total.backward()

    loss_all_finite = bool(
        torch.isfinite(loss.total)
        and all(torch.isfinite(v).all() for v in loss.group_tensors.values())
    )
    gradients = _grad_summary(model)
    failures = _failure_reasons(
        {"loss": {"all_finite": loss_all_finite}, "gradients": gradients}
    )
    report = {
        "handoff_json": str(args.handoff_json),
        "case_id": batch.case_id,
        "field_epsilon_s": float(field.epsilon),
        "s0_physical_time_s": float(scales.physical_time_from_network(0.0)),
        "tau_end_s": float(args.tau_end),
        "batch_tau_max_s": float(batch.pinn_time_max),
        "time_convention": batch.scales.time_convention,
        "detonation_velocity": batch.detonation_velocity,
        "point_counts": {
            "pde_product": int(batch.s_product.shape[0]),
            "pde_air": int(batch.s_air.shape[0]),
            "ic_product": int(batch.xi_ic_product.shape[0]),
            "ic_air": int(batch.eta_ic_air.shape[0]),
            "contact": int(batch.s_contact.shape[0]),
            "shock": int(batch.s_shock.shape[0]),
            "data_product": 0 if batch.s_data_product is None else int(batch.s_data_product.shape[0]),
            "data_air": 0 if batch.s_data_air is None else int(batch.s_data_air.shape[0]),
        },
        "contact_pressure_pa": float(pressure),
        "loss": {
            "total": float(loss.total.detach()),
            "terms": {k: float(v) for k, v in loss.terms.items()},
            "groups": {k: float(v.detach()) for k, v in loss.group_tensors.items()},
            "all_finite": loss_all_finite,
        },
        "gradients": gradients,
        "training_qualified": bool(getattr(field, "training_qualified", False)),
        "ok": not failures,
        "failures": failures,
        "note": "one assembly check and one backward pass; no optimizer, no training",
    }
    print(json.dumps(report, indent=2, default=float))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, default=float), encoding="utf-8"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
