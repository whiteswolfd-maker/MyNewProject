"""Alternating ContactNet/AirNet training.

One loop has two independent updates:

1. AirNet update with ``L_a``: ContactNet geometry is evaluated once and
   detached; AirNet fields and their ``t``/``zeta`` derivatives stay live.
2. ContactNet update with ``L_c``: AirNet fields and derivatives are evaluated
   once, detached, and the ContactNet geometry is recomputed with gradients.

Two Adam optimizers exist for the whole run.  After each backward pass the
trainer verifies that the non-target network has **no** gradient at all and
that its parameters did not move; a zero gradient hidden inside a larger
optimizer is not accepted as isolation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import CaseData
from .losses import (
    TermResult,
    air_data_loss,
    air_ic_loss,
    combine_air,
    combine_contact,
    contact_data_loss,
    contact_ic_loss,
    contact_violation_rmse,
    interface_loss,
    masked_rmse,
    pde_loss,
    right_loss,
    schedule_weights,
)
from .mapping import build_geometry
from .models import (
    AirFields,
    AirNet,
    ContactFields,
    ContactNet,
    JWLClosure,
    Scales,
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)
from .sampling import CaseBatch, sample_case_batch, split_counts


class TrainingError(RuntimeError):
    """Training cannot start or an isolation check failed."""


def _detach_fields(fields: AirFields) -> AirFields:
    return AirFields(
        density=fields.density.detach(), velocity=fields.velocity.detach(),
        pressure=fields.pressure.detach(),
        specific_energy=fields.specific_energy.detach(),
    )


def _detach_contact(fields: ContactFields) -> ContactFields:
    return ContactFields(
        radius=fields.radius.detach(), speed=fields.speed.detach(),
        sigmoid=fields.sigmoid.detach(), raw=fields.raw.detach(),
        pressure=fields.pressure.detach(),
        relative_volume=fields.relative_volume.detach(),
    )


def _detach_derivatives(derivatives: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in derivatives.items()}


def _full(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.full_like(like, float(value))


def _summarise(results: Sequence[TermResult], like: torch.Tensor) -> TermResult:
    if not results:
        return TermResult(total=torch.zeros((), dtype=like.dtype, device=like.device))
    total = sum(result.total for result in results) / float(len(results))
    counts: dict[str, int] = {}
    parts: dict[str, float] = {}
    for result in results:
        for name, value in result.counts.items():
            counts[name] = counts.get(name, 0) + int(value)
        for name, value in result.parts.items():
            parts[name] = parts.get(name, 0.0) + float(value)
    for name in parts:
        parts[name] /= float(len(results))
    return TermResult(total=total, counts=counts, parts=parts)


@dataclass
class TrainConfig:
    n_pde: int = 512
    n_ic_shock: int = 128
    n_ic_ambient: int = 128
    n_right: int = 128
    n_data: int = 128
    n_contact: int = 128
    loops: int = 2000
    learning_rate: float = 1.0e-4
    n_schedule: int = 2000
    log_every: int = 50
    checkpoint_every: int = 100
    validate_every: int = 100
    seed: int = 0
    gradient_clip: float = 0.0
    validation_holdout_fraction: float = 0.3
    validation_data_points: int = 256
    validation_contact_points: int = 256

    def __post_init__(self) -> None:
        for name in ("n_pde", "n_ic_shock", "n_ic_ambient", "n_right", "n_data", "n_contact"):
            if int(getattr(self, name)) < 0:
                raise TrainingError(f"{name} must be non-negative")
        if self.loops < 0:
            raise TrainingError("loops must be non-negative")
        if self.learning_rate <= 0.0:
            raise TrainingError("learning_rate must be positive")
        if self.n_schedule <= 0:
            raise TrainingError("n_schedule must be positive")
        if not math.isfinite(float(self.validation_holdout_fraction)) or not (
            0.0 <= float(self.validation_holdout_fraction) < 1.0
        ):
            raise TrainingError("validation_holdout_fraction must lie in [0, 1)")
        for name in ("validation_data_points", "validation_contact_points"):
            if int(getattr(self, name)) < 0:
                raise TrainingError(f"{name} must be non-negative")


@dataclass
class ValidationSample:
    case_id: str
    batch: CaseBatch


@dataclass
class TrainingHistory:
    loops: list[dict[str, Any]] = field(default_factory=list)
    best_score: float = float("inf")
    best_loop: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "loops": self.loops,
            "best_score": None if not math.isfinite(self.best_score) else self.best_score,
            "best_loop": self.best_loop,
        }


class AlternatingTrainer:
    def __init__(
        self,
        *,
        cases: Sequence[CaseData],
        contact_net: ContactNet,
        air_net: AirNet,
        config: TrainConfig,
        output_dir: Path | str,
    ) -> None:
        if not cases:
            raise TrainingError("at least one training case is required")
        for case in cases:
            if case.role == "test":
                raise TrainingError(
                    f"case {case.case_id!r} is a test case; it must never enter training"
                )
            if abs(float(case.charge_radius_m) * 1000.0 - 500.0) <= 1.0e-9:
                raise TrainingError(
                    f"case {case.case_id!r}: R0=500 mm is the final-validation radius; "
                    "it must not enter training even if the role label is wrong"
                )
        ids = [case.case_id for case in cases]
        if len(set(ids)) != len(ids):
            raise TrainingError(f"duplicate case_id in training cases: {ids}")
        train_cases: list[CaseData] = []
        validation_cases: list[CaseData] = []
        fraction = float(config.validation_holdout_fraction)
        if fraction == 0.0:
            # No independent validation frames by explicit choice.  The trainer
            # reports "not evaluable" rather than pretending the training frames
            # are a validation set.
            train_cases = list(cases)
        else:
            for case in cases:
                # Split only frames inside the actual solve window [t0, t_end]
                # and inside a continuous R_b segment.  A contact label after a
                # gap must not drag the validation interval across that gap.
                frames = np.asarray([
                    index for index in case.contact.indices()
                    if case.t0 <= float(case.contact.times[index]) <= case.t_end
                    and case.boundary.has_slope(float(case.contact.times[index]))
                ], dtype=np.int64)
                if frames.size < 2:
                    raise TrainingError(
                        f"case {case.case_id!r} has {frames.size} usable contact "
                        "frame(s); an independent validation split needs at least "
                        "two. Set validation.holdout_fraction: 0 explicitly to "
                        "train without validation, or improve the extraction."
                    )
                n_validation = max(1, int(round(fraction * frames.size)))
                n_validation = min(n_validation, frames.size - 1)
                validation_frames = frames[-n_validation:]
                training_frames = frames[:-n_validation]
                if training_frames.size == 0:
                    raise TrainingError(
                        f"case {case.case_id!r}: the frame split left no training frames"
                    )
                train_cases.append(case.subset_frames(training_frames, role="train"))
                validation_cases.append(
                    case.subset_frames(validation_frames, role="validation")
                )
        if not train_cases:
            raise TrainingError("the frame split left no training frames")
        self.cases = train_cases
        self.validation_cases = validation_cases
        self.contact_net = contact_net
        self.air_net = air_net
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opt_contact = torch.optim.Adam(
            self.contact_net.parameters(), lr=float(config.learning_rate)
        )
        self.opt_air = torch.optim.Adam(
            self.air_net.parameters(), lr=float(config.learning_rate)
        )
        self.generator = torch.Generator().manual_seed(int(config.seed))
        torch.manual_seed(int(config.seed))
        np.random.seed(int(config.seed) % 2**32)
        random.seed(int(config.seed))
        self.loop = 0
        self.history = TrainingHistory()
        self._validation = self._build_validation()
        self._closures = {
            case.case_id: closure_from_precursor(case.precursor, like=torch.zeros(1, 1, dtype=torch.float64))
            for case in self.cases
        }
        self._scales = {
            case.case_id: scales_from_precursor(case.precursor, like=torch.zeros(1, 1, dtype=torch.float64))
            for case in self.cases
        }

    # -- batches ---------------------------------------------------------
    def _batches(self, generator=None) -> list[tuple[CaseData, CaseBatch]]:
        cfg = self.config
        n = len(self.cases)
        counts = {
            "pde": split_counts(cfg.n_pde, n),
            "ic_shock": split_counts(cfg.n_ic_shock, n),
            "ic_amb": split_counts(cfg.n_ic_ambient, n),
            "right": split_counts(cfg.n_right, n),
            "data": split_counts(cfg.n_data, n),
            "contact": split_counts(cfg.n_contact, n),
        }
        result = []
        for index, case in enumerate(self.cases):
            result.append((
                case,
                sample_case_batch(
                    case,
                    n_pde=counts["pde"][index],
                    n_ic_shock=counts["ic_shock"][index],
                    n_ic_ambient=counts["ic_amb"][index],
                    n_right=counts["right"][index],
                    n_data=counts["data"][index],
                    n_contact=counts["contact"][index],
                    generator=self.generator if generator is None else generator,
                ),
            ))
        return result

    def _build_validation(self) -> list[ValidationSample]:
        generator = torch.Generator().manual_seed(int(self.config.seed) + 20261104)
        samples: list[ValidationSample] = []
        for case in self.validation_cases:
            batch = sample_case_batch(
                case, n_pde=0, n_ic_shock=0, n_ic_ambient=0, n_right=0,
                n_data=int(self.config.validation_data_points),
                n_contact=int(self.config.validation_contact_points),
                generator=generator,
            )
            samples.append(ValidationSample(case_id=case.case_id, batch=batch))
        return samples

    def _contact_inputs(self, case: CaseData, t: torch.Tensor):
        t_flat = t.reshape(-1)
        r0 = _full(case.charge_radius_m, t_flat).reshape(-1, 1)
        rb, rb_dot = case.boundary_tensors(t_flat)
        return r0, rb.reshape(-1, 1), rb_dot.reshape(-1, 1)

    # -- gradient isolation ---------------------------------------------
    def _parameter_snapshot(self, net: torch.nn.Module) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in net.parameters()]

    def _restore_if_changed(self, net: torch.nn.Module, snapshot: list[torch.Tensor], name: str) -> None:
        for parameter, before in zip(net.parameters(), snapshot):
            if not torch.equal(parameter.detach(), before):
                raise TrainingError(
                    f"gradient isolation failed: {name} parameters changed in the wrong stage"
                )

    def _check_target_gradients(
        self, target: torch.nn.Module, other: torch.nn.Module, *, stage: str
    ) -> None:
        target_grads = [p.grad for p in target.parameters()]
        other_grads = [p.grad for p in other.parameters()]
        generated = [g for g in target_grads if g is not None]
        if not generated:
            raise TrainingError(f"{stage}: target network has no gradient at all")
        non_finite = [name for (name, _), g in zip(target.named_parameters(), target_grads)
                      if g is not None and not bool(torch.isfinite(g).all())]
        if non_finite:
            raise TrainingError(
                f"{stage}: non-finite gradients in the target network at {non_finite[:5]}"
            )
        if not any(bool(g.abs().sum() > 0) for g in generated):
            raise TrainingError(f"{stage}: target network has no usable gradient")
        leaked = [name for (name, _), g in zip(other.named_parameters(), other_grads)
                  if g is not None]
        if leaked:
            raise TrainingError(
                f"{stage}: gradients leaked into the non-target network at {leaked[:3]}"
            )

    def _clip(self) -> None:
        if self.config.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.contact_net.parameters(), float(self.config.gradient_clip)
            )
            torch.nn.utils.clip_grad_norm_(
                self.air_net.parameters(), float(self.config.gradient_clip)
            )

    # -- one alternating loop -------------------------------------------
    def step(self) -> dict[str, Any]:
        lambda_pde, lambda_data = schedule_weights(self.loop, self.config.n_schedule)
        batches = self._batches()
        self.opt_contact.zero_grad(set_to_none=True)
        self.opt_air.zero_grad(set_to_none=True)
        contact_before = self._parameter_snapshot(self.contact_net)

        # ---- AirNet update ------------------------------------------------
        air_results: list[TermResult] = []
        for case, batch in batches:
            closure = self._closures[case.case_id]
            scales = self._scales[case.case_id]
            r0, rb, rb_dot = self._contact_inputs(case, batch.t_pde)
            c_fields = self.contact_net(
                batch.t_pde.reshape(-1, 1), r0, rb, rb_dot, closure, create_graph=False
            )
            c_fields = _detach_contact(c_fields)
            geometry = build_geometry(
                c_fields.radius, c_fields.speed, rb, rb_dot, batch.zeta_pde.reshape(-1, 1)
            )
            fields, derivatives = air_fields_with_derivatives(
                self.air_net, batch.zeta_pde, batch.t_pde, r0, scales,
                create_graph=True,
            )
            pde = pde_loss(
                fields=fields, derivatives=derivatives, geometry=geometry, case=case
            )
            ic = air_ic_loss(self.air_net, batch, case, scales)
            if batch.t_pde.numel():
                air0 = self.air_net(
                    torch.zeros_like(batch.t_pde).reshape(-1, 1),
                    batch.t_pde.reshape(-1, 1), r0.reshape(-1, 1), scales,
                )
                interface = interface_loss(
                    air_at_contact=air0, contact=c_fields, case=case
                )
            else:
                interface = TermResult(total=torch.zeros((), dtype=torch.float64))
            if batch.t_right.numel():
                tr0, trb, trbd = self._contact_inputs(case, batch.t_right)
                air1 = self.air_net(
                    torch.ones_like(batch.t_right).reshape(-1, 1),
                    batch.t_right.reshape(-1, 1), tr0.reshape(-1, 1), scales,
                )
                right = right_loss(air_at_right=air1, case=case)
            else:
                right = TermResult(total=torch.zeros((), dtype=torch.float64))
            data = air_data_loss(self.air_net, batch, case, scales)
            air_results.append(combine_air(
                pde=pde, ic=ic, interface=interface, right=right, data=data,
                lambda_pde=lambda_pde, lambda_data=lambda_data,
            ))
        air_total = _summarise(air_results, batches[0][1].t_ic)
        if not bool(torch.isfinite(air_total.total)):
            raise TrainingError("AirNet loss is not finite")
        air_total.total.backward()
        self._check_target_gradients(self.air_net, self.contact_net, stage="air")
        self._clip()
        self.opt_air.step()
        self._restore_if_changed(self.contact_net, contact_before, "ContactNet")

        # ---- ContactNet update -------------------------------------------
        self.opt_contact.zero_grad(set_to_none=True)
        self.opt_air.zero_grad(set_to_none=True)
        air_before = self._parameter_snapshot(self.air_net)
        contact_results: list[TermResult] = []
        for case, batch in batches:
            closure = self._closures[case.case_id]
            scales = self._scales[case.case_id]
            r0, rb, rb_dot = self._contact_inputs(case, batch.t_pde)
            c_fields = self.contact_net(
                batch.t_pde.reshape(-1, 1), r0, rb, rb_dot, closure, create_graph=True
            )
            geometry = build_geometry(
                c_fields.radius, c_fields.speed, rb, rb_dot, batch.zeta_pde.reshape(-1, 1)
            )
            a_fields, a_derivs = air_fields_with_derivatives(
                self.air_net, batch.zeta_pde, batch.t_pde, r0, scales,
                create_graph=False,
            )
            a_fields = _detach_fields(a_fields)
            a_derivs = _detach_derivatives(a_derivs)
            pde = pde_loss(
                fields=a_fields, derivatives=a_derivs, geometry=geometry, case=case
            )
            t0 = torch.full_like(batch.t_pde, float(case.t0))
            r00, rb0, rb0d = self._contact_inputs(case, t0)
            c0 = self.contact_net(t0.reshape(-1, 1), r00, rb0, rb0d, closure, create_graph=True)
            ic = contact_ic_loss(c0, case)
            if batch.t_pde.numel():
                air0 = _detach_fields(self.air_net(
                    torch.zeros_like(batch.t_pde).reshape(-1, 1),
                    batch.t_pde.reshape(-1, 1), r0.reshape(-1, 1), scales,
                ))
                interface = interface_loss(
                    air_at_contact=air0, contact=c_fields, case=case
                )
            else:
                interface = TermResult(total=torch.zeros((), dtype=torch.float64))
            if batch.t_contact.numel():
                # The interval labels belong to batch.t_contact.  Re-evaluate
                # R_b, dR_b/dt and ContactNet at those exact times; reusing the
                # PDE-time prediction would compare different times.
                r0c, rbc, rbdc = self._contact_inputs(case, batch.t_contact)
                c_contact = self.contact_net(
                    batch.t_contact.reshape(-1, 1), r0c, rbc, rbdc, closure,
                    create_graph=True,
                )
                data = contact_data_loss(c_contact, batch, case)
            else:
                data = TermResult(
                    total=torch.zeros((), dtype=torch.float64),
                    counts={"contact_data": 0},
                )
            contact_results.append(combine_contact(
                pde=pde, ic=ic, interface=interface, data=data,
                lambda_pde=lambda_pde, lambda_data=lambda_data,
            ))
        contact_total = _summarise(contact_results, batches[0][1].t_ic)
        if not bool(torch.isfinite(contact_total.total)):
            raise TrainingError("ContactNet loss is not finite")
        contact_total.total.backward()
        self._check_target_gradients(self.contact_net, self.air_net, stage="contact")
        self._clip()
        self.opt_contact.step()
        self._restore_if_changed(self.air_net, air_before, "AirNet")

        self.loop += 1
        record = {
            "loop": self.loop,
            "lambda_pde": lambda_pde,
            "lambda_data": lambda_data,
            "air_loss": float(air_total.total.detach()),
            "contact_loss": float(contact_total.total.detach()),
            "air_parts": dict(air_total.parts),
            "contact_parts": dict(contact_total.parts),
            "counts": {**{f"air.{k}": v for k, v in air_total.counts.items()},
                       **{f"contact.{k}": v for k, v in contact_total.counts.items()}},
            "scales_by_case": {
                case.case_id: case.scale_dict() for case, _ in batches
            },
            "clock_by_case": {
                case.case_id: dict(case.precursor.get("clock", {}))
                for case, _ in batches
            },
        }
        self.history.loops.append(record)
        return record

    # -- validation ------------------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        if not self._validation:
            return {
                "score": None,
                "evaluable": False,
                "reason": "no independent validation frames were configured "
                          "(validation.holdout_fraction is 0 or no split exists)",
                "cases": [],
            }
        per_case: list[dict[str, Any]] = []
        with torch.no_grad():
            for sample in self._validation:
                case = next(c for c in self.validation_cases if c.case_id == sample.case_id)
                batch = sample.batch
                scales = self._scales[case.case_id]
                rho_err = u_err = p_err = None
                n_rho = n_u = n_p = 0
                if batch.t_data.numel():
                    r0 = _full(case.charge_radius_m, batch.t_data)
                    fields = self.air_net(
                        batch.zeta_data.reshape(-1, 1), batch.t_data.reshape(-1, 1),
                        r0.reshape(-1, 1), scales,
                    )
                    rho_err, n_rho = masked_rmse(
                        fields.density, batch.rho_data, batch.valid_rho, case.rho_scale
                    )
                    u_err, n_u = masked_rmse(
                        fields.velocity, batch.u_data, batch.valid_u, case.u_scale
                    )
                    p_err, n_p = masked_rmse(
                        fields.pressure, batch.p_data, batch.valid_p, case.p_scale
                    )
                components = [value for value in (rho_err, u_err, p_err) if value is not None]
                air_error = sum(components) / len(components) if components else None
                violation = None
                n_contact = 0
                if batch.t_contact.numel():
                    r0, rb, rb_dot = self._contact_inputs(case, batch.t_contact)
                    with torch.enable_grad():
                        c_fields = self.contact_net(
                            batch.t_contact.reshape(-1, 1), r0, rb, rb_dot,
                            self._closures[case.case_id], create_graph=False,
                        )
                    c_fields = _detach_contact(c_fields)
                    violation, n_contact = contact_violation_rmse(c_fields, batch, case)
                available = [value for value in (air_error, violation) if value is not None]
                # A case with no evaluable component must not contribute a
                # free zero to the score.
                score = sum(available) if available else None
                per_case.append({
                    "case_id": case.case_id,
                    "evaluable": bool(available),
                    "air_rmse": air_error, "rho_rmse": rho_err,
                    "u_rmse": u_err, "p_rmse": p_err,
                    "n_rho": n_rho, "n_u": n_u, "n_p": n_p,
                    "contact_violation": violation, "n_contact": n_contact,
                    "score": score,
                })
        scores = [item["score"] for item in per_case if item["score"] is not None]
        return {
            "score": sum(scores) / len(scores) if scores else None,
            "evaluable": bool(scores),
            "n_cases": len(per_case),
            "n_evaluable": len(scores),
            "reason": (
                None if scores else
                "no validation case has any valid data or contact label; the score "
                "is not evaluable and must not be used for best-model selection"
            ),
            "cases": per_case,
        }

    def train(self, *, loops: int | None = None, resume_loop: int | None = None) -> dict[str, Any]:
        target = self.config.loops if loops is None else int(loops)
        if target < self.loop:
            raise TrainingError(f"target loops {target} is below current loop {self.loop}")
        records = []
        while self.loop < target:
            record = self.step()
            records.append(record)
            if self.config.log_every > 0 and self.loop % self.config.log_every == 0:
                print(
                    f"loop {self.loop:5d}  air={record['air_loss']:.6e}  "
                    f"contact={record['contact_loss']:.6e}  "
                    f"lambda_data={record['lambda_data']:.3f}"
                )
            if self.config.validate_every > 0 and self.loop % self.config.validate_every == 0:
                verdict = self.evaluate()
                if verdict["evaluable"] and verdict["score"] is not None:
                    if verdict["score"] < self.history.best_score:
                        self.history.best_score = verdict["score"]
                        self.history.best_loop = self.loop
                        self.save(self.output_dir / "best.pt")
                    print(f"  validation score = {verdict['score']:.6e}")
                else:
                    print(
                        "  validation not evaluable: "
                        f"{verdict.get('reason') or 'no usable validation samples'}"
                    )
            if self.config.checkpoint_every > 0 and self.loop % self.config.checkpoint_every == 0:
                self.save(self.output_dir / "last.pt")
        self.save(self.output_dir / "last.pt")
        return {
            "loops": self.loop,
            "history": self.history.as_dict(),
            "last_validation": self.evaluate(),
        }

    # -- checkpoint ------------------------------------------------------
    @staticmethod
    def _case_signature(case: CaseData) -> str:
        digest = hashlib.sha256()

        def add(name: str, value) -> None:
            digest.update(name.encode("utf-8"))
            if isinstance(value, str):
                digest.update(value.encode("utf-8"))
                return
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())

        add("case_id", case.case_id)
        add("role", case.role)
        add("charge_radius", np.asarray([case.charge_radius_m]))
        add("precursor", json.dumps(case.precursor, sort_keys=True, allow_nan=True))
        add("t0", np.asarray([case.t0]))
        add("t_end", np.asarray([case.t_end]))
        add("initial", np.asarray([
            case.initial.Rc0, case.initial.Rs0, case.initial.Rb0,
            case.initial.zeta_s0,
        ]))
        add("boundary_times", case.boundary.times)
        add("boundary_radii", case.boundary.radii)
        add("boundary_valid", case.boundary.valid)
        add("contact_times", case.contact.times)
        add("contact_low", case.contact.low)
        add("contact_up", case.contact.up)
        add("contact_raw", case.contact.raw)
        add("contact_valid", case.contact.valid)
        add("obs_time", case.observations.time)
        add("obs_radius", case.observations.radius)
        add("obs_zeta", case.observations.zeta)
        add("obs_density", case.observations.density)
        add("obs_velocity", case.observations.velocity)
        add("obs_pressure", case.observations.pressure)
        add("obs_valid_density", case.observations.valid_density)
        add("obs_valid_velocity", case.observations.valid_velocity)
        add("obs_valid_pressure", case.observations.valid_pressure)
        add("obs_frame", case.observations.frame_index)
        add("notes", "\n".join(case.notes))
        add("tolerances", json.dumps(case.tolerances, sort_keys=True))
        return digest.hexdigest()

    @staticmethod
    def _batch_payload(batch: CaseBatch) -> dict[str, Any]:
        names = (
            "t_pde", "zeta_pde", "t_ic", "zeta_ic_shock", "zeta_ic_ambient",
            "t_right", "t_data", "zeta_data", "rho_data", "u_data", "p_data",
            "valid_rho", "valid_u", "valid_p", "t_contact", "low_contact",
            "up_contact",
        )
        payload: dict[str, Any] = {"case_id": batch.case_id}
        for name in names:
            value = getattr(batch, name)
            payload[name] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return payload

    @staticmethod
    def _batch_from_payload(payload: Mapping[str, Any]) -> CaseBatch:
        def array(name: str, dtype: str):
            value = np.asarray(payload[name])
            return torch.as_tensor(value, dtype=getattr(torch, dtype))

        return CaseBatch(
            case_id=str(payload["case_id"]),
            t_pde=array("t_pde", "float64"), zeta_pde=array("zeta_pde", "float64"),
            t_ic=array("t_ic", "float64"),
            zeta_ic_shock=array("zeta_ic_shock", "float64"),
            zeta_ic_ambient=array("zeta_ic_ambient", "float64"),
            t_right=array("t_right", "float64"),
            t_data=array("t_data", "float64"), zeta_data=array("zeta_data", "float64"),
            rho_data=array("rho_data", "float64"), u_data=array("u_data", "float64"),
            p_data=array("p_data", "float64"),
            valid_rho=array("valid_rho", "bool"), valid_u=array("valid_u", "bool"),
            valid_p=array("valid_p", "bool"),
            t_contact=array("t_contact", "float64"),
            low_contact=array("low_contact", "float64"),
            up_contact=array("up_contact", "float64"),
        )

    @staticmethod
    def _check_state_dict(module: torch.nn.Module, state: Mapping[str, Any], name: str) -> None:
        current = module.state_dict()
        if set(state) != set(current):
            missing = sorted(set(current) - set(state))
            extra = sorted(set(state) - set(current))
            raise TrainingError(
                f"checkpoint {name} keys do not match the current architecture; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for key, value in current.items():
            saved = state[key]
            if tuple(saved.shape) != tuple(value.shape) or saved.dtype != value.dtype:
                raise TrainingError(
                    f"checkpoint {name} architecture mismatch at {key!r}: saved "
                    f"shape {tuple(saved.shape)} / dtype {saved.dtype}, current is "
                    f"{tuple(value.shape)} / {value.dtype}"
                )

    @staticmethod
    def _check_optimizer(
        optimizer: torch.optim.Optimizer, state: Mapping[str, Any],
        module: torch.nn.Module, name: str,
    ) -> None:
        parameters = list(module.parameters())
        groups = state.get("param_groups", [])
        if len(groups) != 1 or len(groups[0].get("params", [])) != len(parameters):
            raise TrainingError(
                f"checkpoint {name} optimizer parameter groups do not match the model"
            )
        for key, entry in state.get("state", {}).items():
            if int(key) >= len(parameters):
                raise TrainingError(f"checkpoint {name} optimizer references unknown parameter {key}")
            parameter = parameters[int(key)]
            for field_name in ("exp_avg", "exp_avg_sq"):
                value = entry.get(field_name)
                if value is not None and tuple(value.shape) != tuple(parameter.shape):
                    raise TrainingError(
                        f"checkpoint {name} optimizer state {field_name}[{key}] has "
                        f"shape {tuple(value.shape)}, parameter has {tuple(parameter.shape)}"
                    )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "spherical_blast_pinn.contact_air_checkpoint",
            "version": 2,
            "loop": self.loop,
            "contact_net": self.contact_net.state_dict(),
            "air_net": self.air_net.state_dict(),
            "opt_contact": self.opt_contact.state_dict(),
            "opt_air": self.opt_air.state_dict(),
            "contact_net_describe": self.contact_net.describe(),
            "air_net_describe": self.air_net.describe(),
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
            "case_ids": [case.case_id for case in self.cases],
            "case_signatures": [self._case_signature(case) for case in self.cases],
            "case_metadata": [case.as_dict() for case in self.cases],
            "validation_case_ids": [case.case_id for case in self.validation_cases],
            "validation_case_signatures": [
                self._case_signature(case) for case in self.validation_cases
            ],
            "validation": [
                {"case_id": sample.case_id, "batch": self._batch_payload(sample.batch)}
                for sample in self._validation
            ],
            "history": self.history.as_dict(),
            "torch_rng": torch.get_rng_state(),
            "generator": self.generator.get_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        torch.save(payload, path)
        return path

    def load(self, path: Path | str) -> dict[str, Any]:
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "spherical_blast_pinn.contact_air_checkpoint":
            raise TrainingError(f"{path}: not a contact-air checkpoint")
        if payload.get("version") != 2:
            raise TrainingError(
                f"{path}: checkpoint version {payload.get('version')!r} is not "
                "supported. Rebuild it: version 1 predates the contact-label time, "
                "broadcast, R_b-gap, energy-scale and validation fixes"
            )
        expected = [case.case_id for case in self.cases]
        if list(payload.get("case_ids", [])) != expected:
            raise TrainingError(
                f"{path}: checkpoint case ids {payload.get('case_ids')!r} do not match "
                f"the current training cases {expected!r}"
            )
        signatures = [self._case_signature(case) for case in self.cases]
        if list(payload.get("case_signatures", [])) != signatures:
            raise TrainingError(
                f"{path}: training data / precursor / boundary / split signature "
                "does not match the current cases; refusing to resume on different data"
            )
        validation_ids = [case.case_id for case in self.validation_cases]
        if list(payload.get("validation_case_ids", [])) != validation_ids:
            raise TrainingError(
                f"{path}: checkpoint validation case ids do not match the current split"
            )
        validation_signatures = [
            self._case_signature(case) for case in self.validation_cases
        ]
        if list(payload.get("validation_case_signatures", [])) != validation_signatures:
            raise TrainingError(
                f"{path}: checkpoint validation split/data signature differs from the "
                "current one; the fixed validation frames would change"
            )
        saved_config = payload.get("config", {})
        for key in (
            "validation_holdout_fraction", "validation_data_points",
            "validation_contact_points", "n_schedule", "n_pde", "n_ic_shock",
            "n_ic_ambient", "n_right", "n_data", "n_contact", "seed",
        ):
            saved = saved_config.get(key)
            current = getattr(self.config, key)
            if saved is None or float(saved) != float(current):
                raise TrainingError(
                    f"{path}: checkpoint {key}={saved!r} does not match the current "
                    f"{current!r}; the training/validation split or sampling changed"
                )
        # Validate every state dict before mutating anything: a partially
        # loaded checkpoint is worse than a refused one.
        self._check_state_dict(self.contact_net, payload["contact_net"], "contact_net")
        self._check_state_dict(self.air_net, payload["air_net"], "air_net")
        self._check_optimizer(
            self.opt_contact, payload["opt_contact"], self.contact_net, "opt_contact"
        )
        self._check_optimizer(self.opt_air, payload["opt_air"], self.air_net, "opt_air")
        saved_validation = payload.get("validation", [])
        if [entry.get("case_id") for entry in saved_validation] != validation_ids:
            raise TrainingError(
                f"{path}: saved validation samples do not match the current validation cases"
            )
        restored_validation = [
            ValidationSample(
                case_id=str(entry["case_id"]),
                batch=self._batch_from_payload(entry["batch"]),
            )
            for entry in saved_validation
        ]
        self.contact_net.load_state_dict(payload["contact_net"])
        self.air_net.load_state_dict(payload["air_net"])
        self.opt_contact.load_state_dict(payload["opt_contact"])
        self.opt_air.load_state_dict(payload["opt_air"])
        self.loop = int(payload["loop"])
        history = payload.get("history", {})
        self.history = TrainingHistory(
            loops=list(history.get("loops", [])),
            best_score=float(history.get("best_score", float("inf")) or float("inf")),
            best_loop=history.get("best_loop"),
        )
        self._validation = restored_validation
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
        if "generator" in payload:
            self.generator.set_state(payload["generator"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])
        if "python_rng" in payload:
            random.setstate(payload["python_rng"])
        return {"loop": self.loop, "case_ids": expected}


__all__ = [
    "AlternatingTrainer",
    "TrainConfig",
    "TrainingError",
    "TrainingHistory",
]
