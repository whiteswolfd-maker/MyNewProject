"""Residual-based adaptive refinement (RAR) for PDE collocation points.

RAR here only changes **which interior PDE points** a case samples.  It does not
sample data labels, does not touch the initial/contact/shock conditions, and does
not update model parameters.  Each case and each material has its own bounded
point pool and uses that case's own scales, moving coordinates and time window.

Candidate scores are the same scaled residuals as :func:`medium_pde_residual`:
``(R_mass^2 + R_momentum^2 + R_energy^2) / 3`` after each equation has been
divided by the scales in :mod:`physics.pde`.  Raw dimensional residuals are never
mixed.  A finite check runs before any point is selected; a non-finite residual is
an error carrying the case, medium and offending position, never zeroed or
clipped.

Scoring uses autograd derivatives, but never calls ``backward`` or an optimizer.
It therefore does not write ``.grad`` and does not change the model's train/eval
mode.  The base stratified random sampling remains in place: RAR replaces only a
configured fraction of each PDE group, always keeps at least one base point when
the group is nonempty, and the pool itself is capped and deduped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..losses.terms import medium_pde_residual
from ..physics.moving_frame import AIR, PRODUCT
from .batch import SampleBatch

#: Coordinates are kept strictly inside the open material domains.
_COORDINATE_FLOOR = 1.0e-9


class RARError(ValueError):
    """RAR configuration, point state or residual is invalid."""


@dataclass(frozen=True)
class RARConfig:
    """RAR controls.  All values are validated even when ``enabled`` is false."""

    enabled: bool = False
    interval: int = 500
    n_candidates_product: int = 512
    n_candidates_air: int = 512
    n_select_product: int = 128
    n_select_air: int = 128
    pool_capacity_product: int = 512
    pool_capacity_air: int = 512
    #: Fraction of the *existing* PDE point count replaced by pool points.
    #: Must lie strictly between 0 and 1; nonempty groups keep a base point.
    pool_fraction: float = 0.25

    @staticmethod
    def _integer(name: str, value: Any) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise RARError(f"RAR {name} must be an integer, got {value!r}")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise RARError(f"RAR {name} must be an integer, got {value!r}") from exc
        if number != value:
            raise RARError(f"RAR {name} must be an integer, got {value!r}")
        return number

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise RARError(f"RAR enabled must be a boolean, got {self.enabled!r}")
        interval = self._integer("interval", self.interval)
        if interval < 1:
            raise RARError(f"RAR interval must be >= 1, got {self.interval!r}")
        for medium in (PRODUCT, AIR):
            candidates = self._integer(
                f"n_candidates_{medium}", getattr(self, f"n_candidates_{medium}")
            )
            selected = self._integer(
                f"n_select_{medium}", getattr(self, f"n_select_{medium}")
            )
            capacity = self._integer(
                f"pool_capacity_{medium}", getattr(self, f"pool_capacity_{medium}")
            )
            if candidates < 1:
                raise RARError(
                    f"RAR n_candidates_{medium} must be >= 1, got {candidates!r}"
                )
            if selected < 1 or selected > candidates:
                raise RARError(
                    f"RAR n_select_{medium} must be in [1, n_candidates_{medium}], "
                    f"got {selected!r}"
                )
            if capacity < selected:
                raise RARError(
                    f"RAR pool_capacity_{medium} must be >= n_select_{medium}, "
                    f"got {capacity!r} < {selected!r}"
                )
        raw_fraction = self.pool_fraction
        if isinstance(raw_fraction, (bool, np.bool_)) or not isinstance(
            raw_fraction, (int, float, np.integer, np.floating)
        ):
            raise RARError(
                f"RAR pool_fraction must be a finite number in (0, 1), got "
                f"{raw_fraction!r}"
            )
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise RARError(
                f"RAR pool_fraction must be finite and lie strictly in (0, 1), got "
                f"{raw_fraction!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "interval": int(self.interval),
            "n_candidates_product": int(self.n_candidates_product),
            "n_candidates_air": int(self.n_candidates_air),
            "n_select_product": int(self.n_select_product),
            "n_select_air": int(self.n_select_air),
            "pool_capacity_product": int(self.pool_capacity_product),
            "pool_capacity_air": int(self.pool_capacity_air),
            "pool_fraction": float(self.pool_fraction),
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "RARConfig":
        if payload is None or payload is False:
            return cls(enabled=False)
        if payload is True:
            config = cls(enabled=True)
            config.validate()
            return config
        if not isinstance(payload, Mapping):
            raise RARError(f"training.rar must be a bool or a mapping, got {payload!r}")
        known = {
            "enabled",
            "interval",
            "n_candidates_product",
            "n_candidates_air",
            "n_select_product",
            "n_select_air",
            "pool_capacity_product",
            "pool_capacity_air",
            "pool_fraction",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise RARError(f"training.rar has unknown key(s) {unknown}")
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RARError(f"training.rar.enabled must be a boolean, got {enabled!r}")
        config = cls(
            enabled=enabled,
            interval=payload.get("interval", 500),
            n_candidates_product=payload.get("n_candidates_product", 512),
            n_candidates_air=payload.get("n_candidates_air", 512),
            n_select_product=payload.get("n_select_product", 128),
            n_select_air=payload.get("n_select_air", 128),
            pool_capacity_product=payload.get("pool_capacity_product", 512),
            pool_capacity_air=payload.get("pool_capacity_air", 512),
            pool_fraction=payload.get("pool_fraction", 0.25),
        )
        config.validate()
        return config

    @classmethod
    def from_config(cls, config: Any) -> "RARConfig":
        raw = config.get("training.rar", None)
        if config.get("losses.rar", None) not in (None, False):
            raise RARError(
                "losses.rar is not a supported location; put RAR controls under "
                "training.rar"
            )
        return cls.from_mapping(raw)


def _checked_point_arrays(
    s: Any, coordinate: Any, score: Any, *, medium: str, where: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_array = np.asarray(s, dtype=np.float64).reshape(-1)
    c_array = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    v_array = np.asarray(score, dtype=np.float64).reshape(-1)
    if not (s_array.shape == c_array.shape == v_array.shape):
        raise RARError(
            f"{where}: point arrays have mismatched shapes "
            f"s={s_array.shape}, coordinate={c_array.shape}, score={v_array.shape}"
        )
    if not (
        np.all(np.isfinite(s_array))
        and np.all(np.isfinite(c_array))
        and np.all(np.isfinite(v_array))
    ):
        raise RARError(f"{where}: RAR point arrays contain non-finite values")
    if np.any(s_array < 0.0):
        raise RARError(f"{where}: RAR candidate time is negative")
    lower = _COORDINATE_FLOOR
    upper = 1.0 - _COORDINATE_FLOOR
    if np.any(c_array <= lower) or np.any(c_array >= upper):
        raise RARError(
            f"{where}: RAR coordinate is outside the open {medium} domain"
        )
    return s_array, c_array, v_array


def _dedupe(
    s: np.ndarray, coordinate: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the highest-scoring representative of each rounded point."""
    if s.size == 0:
        return s, coordinate, score
    order = np.argsort(-score, kind="stable")
    seen: set[tuple[float, float]] = set()
    keep: list[int] = []
    for index in order:
        key = (round(float(s[index]), 12), round(float(coordinate[index]), 12))
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    return s[keep_array], coordinate[keep_array], score[keep_array]


@dataclass
class RARPointPool:
    """Bounded, deduplicated pool for one case and one material."""

    medium: str
    capacity: int
    s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    coordinate: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if self.medium not in (PRODUCT, AIR):
            raise RARError(f"unknown RAR medium {self.medium!r}")
        if int(self.capacity) < 1:
            raise RARError("RAR pool capacity must be >= 1")
        s, coordinate, score = _checked_point_arrays(
            self.s,
            self.coordinate,
            self.score,
            medium=self.medium,
            where=f"RAR pool {self.medium}",
        )
        self.s, self.coordinate, self.score = s, coordinate, score
        if s.size > int(self.capacity):
            raise RARError(
                f"RAR pool {self.medium} holds {s.size} points, above capacity "
                f"{self.capacity}"
            )

    def __len__(self) -> int:
        return int(self.s.size)

    def replace(
        self, s: Any, coordinate: Any, score: Any
    ) -> None:
        s_array, c_array, v_array = _checked_point_arrays(
            s,
            coordinate,
            score,
            medium=self.medium,
            where=f"RAR pool {self.medium}",
        )
        s_array, c_array, v_array = _dedupe(s_array, c_array, v_array)
        if s_array.size > int(self.capacity):
            s_array = s_array[: int(self.capacity)]
            c_array = c_array[: int(self.capacity)]
            v_array = v_array[: int(self.capacity)]
        self.s, self.coordinate, self.score = s_array, c_array, v_array

    def sample(
        self, count: int, rng: np.random.Generator, *, window_s_max: float
    ) -> tuple[np.ndarray, np.ndarray]:
        count = int(count)
        if count <= 0 or self.s.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        eligible = np.flatnonzero(self.s <= float(window_s_max))
        if eligible.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        take = min(count, int(eligible.size))
        chosen = rng.choice(eligible, size=take, replace=False)
        return self.s[chosen], self.coordinate[chosen]

    def state_dict(self) -> dict[str, Any]:
        return {
            "medium": self.medium,
            "capacity": int(self.capacity),
            "s": self.s.copy(),
            "coordinate": self.coordinate.copy(),
            "score": self.score.copy(),
        }

    @classmethod
    def from_state(cls, payload: Any, *, medium: str, capacity: int) -> "RARPointPool":
        if not isinstance(payload, Mapping):
            raise RARError(f"RAR pool state for {medium} is not a mapping")
        if payload.get("medium") != medium:
            raise RARError(
                f"RAR pool state medium {payload.get('medium')!r} is not {medium!r}"
            )
        if int(payload.get("capacity", -1)) != int(capacity):
            raise RARError(
                f"RAR pool capacity for {medium} is {payload.get('capacity')!r}, "
                f"expected {capacity!r}"
            )
        try:
            s = payload["s"]
            coordinate = payload["coordinate"]
            score = payload["score"]
        except KeyError as exc:
            raise RARError(f"RAR pool state for {medium} is missing {exc}") from exc
        return cls(medium=medium, capacity=capacity, s=s, coordinate=coordinate, score=score)


def _rar_state(
    *,
    config: RARConfig,
    last_update_step: int | None,
    pools: Mapping[str, RARPointPool],
) -> dict[str, Any]:
    if last_update_step is not None and int(last_update_step) < 0:
        raise RARError("RAR last_update_step must be non-negative when present")
    return {
        "enabled": bool(config.enabled),
        "config": config.as_dict(),
        "last_update_step": (
            None if last_update_step is None else int(last_update_step)
        ),
        "pools": {
            medium: pools[medium].state_dict() for medium in (PRODUCT, AIR)
        },
    }


def _score_candidates(
    sampler: Any,
    *,
    medium: str,
    s: np.ndarray,
    coordinate: np.ndarray,
    model: Any,
) -> np.ndarray:
    """Score candidate points with the production PDE residual.

    No ``backward`` and no optimizer call.  ``torch.enable_grad`` only permits the
    derivatives needed by the residual; the caller's mode is restored by the
    context manager.
    """
    if torch is None:  # pragma: no cover
        raise RARError("PyTorch is required")
    with torch.enable_grad():
        s_column = sampler._column(s)
        c_column = sampler._column(coordinate)
        if medium == PRODUCT:
            batch = SampleBatch(
                case_id=sampler.case_id,
                charge_radius=float(sampler.scales.charge_radius),
                detonation_velocity=None,
                scales=sampler.scales,
                pinn_time_min=0.0,
                pinn_time_max=float(sampler.window_pinn_time_max),
                s_product=s_column,
                xi_product=c_column,
            )
        else:
            batch = SampleBatch(
                case_id=sampler.case_id,
                charge_radius=float(sampler.scales.charge_radius),
                detonation_velocity=None,
                scales=sampler.scales,
                pinn_time_min=0.0,
                pinn_time_max=float(sampler.window_pinn_time_max),
                s_air=s_column,
                eta_air=c_column,
            )
        try:
            residual = medium_pde_residual(batch, model, medium)
        except Exception as exc:  # noqa: BLE001 - add case/medium/position context
            raise RARError(
                f"case {sampler.case_id!r}, medium {medium!r}: residual evaluation "
                f"failed at candidate s={s[0]!r}, coordinate={coordinate[0]!r}: {exc}"
            ) from exc
        if residual is None:
            raise RARError(
                f"case {sampler.case_id!r}, medium {medium!r}: no residual points"
            )
        mass, momentum, energy = residual.scaled()
        values = ((mass**2 + momentum**2 + energy**2) / 3.0).detach()
        array = values.reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        index = int(np.flatnonzero(~np.isfinite(array))[0])
        raise RARError(
            f"case {sampler.case_id!r}, medium {medium!r}: non-finite PDE residual "
            f"at s={float(s[index])!r}, coordinate={float(coordinate[index])!r}"
        )
    return array


def maybe_update_rar(sampler: Any, model: Any, step: int) -> dict[str, Any] | None:
    """Update the sampler's RAR pools at a configured step.

    ``step`` is the number of completed optimizer updates.  Step 0 is never
    updated; a non-multiple is ignored.  ``last_update_step`` is saved in the
    sampler's RAR state, so restoring at the same step cannot repeat an update.
    """
    config = sampler.rar_config
    if not config.enabled:
        return None
    step = int(step)
    if step <= 0 or step % int(config.interval) != 0:
        return None
    if sampler._rar_last_update_step == step:
        return None

    summary: dict[str, Any] = {"step": step, "media": {}}
    for medium in (PRODUCT, AIR):
        pool = sampler.rar_pools[medium]
        n_candidates = int(getattr(config, f"n_candidates_{medium}"))
        n_select = int(getattr(config, f"n_select_{medium}"))
        capacity = int(getattr(config, f"pool_capacity_{medium}"))

        s_new = sampler.sample_network_time(n_candidates)
        coordinate_new = sampler._rar_coordinates(medium, n_candidates)
        s_existing = pool.s
        coordinate_existing = pool.coordinate
        s_all = np.concatenate([s_existing, s_new])
        coordinate_all = np.concatenate([coordinate_existing, coordinate_new])
        scores_all = _score_candidates(
            sampler,
            medium=medium,
            s=s_all,
            coordinate=coordinate_all,
            model=model,
        )
        if scores_all.size != s_all.size:
            raise RARError(
                f"case {sampler.case_id!r}, medium {medium!r}: residual returned "
                f"{scores_all.size} score(s) for {s_all.size} candidate(s)"
            )
        if not np.all(np.isfinite(scores_all)):
            index = int(np.flatnonzero(~np.isfinite(scores_all))[0])
            raise RARError(
                f"case {sampler.case_id!r}, medium {medium!r}: non-finite PDE "
                f"residual at s={float(s_all[index])!r}, "
                f"coordinate={float(coordinate_all[index])!r}"
            )
        new_scores = scores_all[s_existing.size:]
        order = np.argsort(-new_scores, kind="stable")[:n_select]
        selected_s = s_new[order]
        selected_coordinate = coordinate_new[order]
        selected_score = new_scores[order]
        merged_s = np.concatenate([s_existing, selected_s])
        merged_coordinate = np.concatenate([coordinate_existing, selected_coordinate])
        merged_score = np.concatenate([scores_all[: s_existing.size], selected_score])
        pool.replace(merged_s, merged_coordinate, merged_score)
        if len(pool) > capacity:  # defensive; replace already caps
            raise RARError(f"RAR pool {medium} exceeded its capacity")
        summary["media"][medium] = {
            "candidates": int(n_candidates),
            "selected": int(selected_s.size),
            "pool_size": int(len(pool)),
            "pool_capacity": int(capacity),
            "pool_fraction": float(config.pool_fraction),
            "score_min": float(selected_score.min()) if selected_score.size else None,
            "score_max": float(selected_score.max()) if selected_score.size else None,
        }
    sampler._rar_last_update_step = step
    sampler._rar_last_summary = summary
    return summary


def validate_rar_state(
    state: Any,
    *,
    config: RARConfig,
    where: str = "RAR state",
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise RARError(f"{where}: state is {type(state).__name__}, not a mapping")
    if bool(state.get("enabled")) != bool(config.enabled):
        raise RARError(
            f"{where}: enabled={state.get('enabled')!r}, current RAR config says "
            f"{config.enabled!r}"
        )
    recorded_config = state.get("config")
    if recorded_config != config.as_dict():
        raise RARError(
            f"{where}: RAR config differs from the current configuration"
        )
    last = state.get("last_update_step")
    if last is not None and (
        isinstance(last, bool) or not isinstance(last, (int, np.integer)) or int(last) < 0
    ):
        raise RARError(f"{where}: last_update_step is {last!r}")
    pools = state.get("pools")
    if not isinstance(pools, Mapping):
        raise RARError(f"{where}: no pools mapping")
    result: dict[str, Any] = {}
    for medium in (PRODUCT, AIR):
        capacity = int(getattr(config, f"pool_capacity_{medium}"))
        pool = RARPointPool.from_state(
            pools.get(medium), medium=medium, capacity=capacity
        )
        result[medium] = pool
    if not config.enabled and any(len(pool) for pool in result.values()):
        raise RARError(f"{where}: disabled RAR carries a non-empty pool")
    return {
        "enabled": bool(config.enabled),
        "last_update_step": None if last is None else int(last),
        "pools": result,
    }


__all__ = [
    "RARConfig",
    "RARError",
    "RARPointPool",
    "maybe_update_rar",
    "validate_rar_state",
]
