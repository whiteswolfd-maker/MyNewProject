"""把配置装配成"多个半径共用一个共享模型"。

新链路没有 ``ReferenceField``，也没有 ``T_h``。每个算例有自己的：

* ``PlanarStartupField``（磁盘 handoff 或现场构造并写盘）；
* ``PlanarStartupScales``（``t_phys = epsilon + tau``）；
* ``RadiusSampler``（自己的 ``tau_end``、采样配置与 RNG）；
* 可选 ``SupervisionDataset``（显式时间原点与逐字段有效性）。

**可训练参数在半径之间共享**：一个 ProductNet、一个 AirNet、一套前沿参数。
网络本身无量纲，物理输出换算逐次使用 ``batch.scales`` 的 ``rho_*/U_*/e_*``、
``R_0`` 与时间尺度。支持范围是同材料、同环境、同 ``delta``、同启动方法与
时间尺度，只有 ``R_0`` 与 ``epsilon`` 不同；超出范围在装配时拒绝。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..config import Config
from ..data.startup_supervision import SupervisionDataset, SupervisionError
from ..losses.objective import contact_pressure_scale
from ..models.coupled_model import CoupledModel, build_model
from ..physics.material_state import (
    MaterialConditionsError,
    condition_mismatches,
    config_condition_mismatches,
    configured_air_state,
    configured_jwl_and_energy,
    field_material_conditions,
    model_material_conditions,
)
from ..physics.similitude import PlanarStartupScales
from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup_handoff import (
    load_planar_startup_handoff,
    write_planar_startup_handoff,
)
from ..physics.two_medium_startup_planar import (
    PLANAR_STARTUP_MODEL_ID,
    PlanarStartupConfig,
    PlanarStartupField,
    build_planar_startup_field,
)
from ..sampling.rar import RARConfig, RARError
from ..sampling.sampler import RadiusSampler, SamplingConfig


class CaseSetupError(RuntimeError):
    """训练算例装配失败。"""


@dataclass
class StartupCase:
    """一个半径的**算例上下文**；不含任何可训练参数。

    场、尺度、时间原点、采样器、RNG 与监督数据逐算例独立；ProductNet /
    AirNet / 前沿网络在所有算例之间共享同一组参数（由
    :class:`StartupTrainingContext` 持有）。
    """

    case_id: str
    split: str
    field: PlanarStartupField
    scales: PlanarStartupScales
    sampler: RadiusSampler
    contact_pressure: float
    handoff_json: Path | None = None
    data: SupervisionDataset | None = None
    geometry: dict[str, Any] = field(default_factory=dict)

    @property
    def charge_radius(self) -> float:
        return float(self.scales.charge_radius)

    @property
    def tau_end(self) -> float:
        return float(self.sampler.tau_end)

    def data_identity(self) -> dict[str, Any] | None:
        return None if self.data is None else self.data.identity()

    def describe(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "charge_radius_m": self.charge_radius,
            "epsilon_s": float(self.field.epsilon),
            "initial_contact_hat": float(
                self.field.contact_radius / self.field.charge_radius
            ),
            "initial_gap_hat": float(
                (self.field.shock_radius - self.field.contact_radius)
                / self.field.charge_radius
            ),
            "tau_end_s": self.tau_end,
            "time_convention": self.scales.time_convention,
            "contact_pressure_pa": float(self.contact_pressure),
            "startup": {
                "model_id": self.field.model_id,
                "method_version": self.field.method_version,
                "cache_key": self.field.cache_key(),
                "delta": float(self.field.delta),
                "training_qualified": bool(
                    getattr(self.field, "training_qualified", False)
                ),
            },
            "handoff_json": None if self.handoff_json is None else str(self.handoff_json),
            "data": None if self.data is None else {
                "identity": self.data.identity(),
                "synthetic": self.data.synthetic,
                "frames": self.data.n_frames,
                "time_origin": self.data.time_origin.as_dict(),
                "notes": list(self.data.notes or []),
            },
            "sampling": self.sampler.sampling_config_state(),
        }


@dataclass
class StartupTrainingContext:
    """共享模型 + 多个半径的算例上下文。"""

    config: Config
    cases: list[StartupCase]
    model: CoupledModel
    sampling: SamplingConfig
    time_scale: float
    seed: int
    diagnostic: bool
    shared_compatibility: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]

    @property
    def shared_scales(self) -> PlanarStartupScales:
        return self.cases[0].scales

    def describe(self) -> dict[str, Any]:
        return {
            "time_scale": self.time_scale,
            "seed": self.seed,
            "diagnostic": self.diagnostic,
            "shared_compatibility": dict(self.shared_compatibility),
            "model": self.model.describe(),
            "sampling": {
                name: getattr(self.sampling, name)
                for name in SamplingConfig.__dataclass_fields__
            },
            "cases": [case.describe() for case in self.cases],
            "notes": list(self.notes),
        }


#: 新链路认识的配置键。未知键一律报错：拼错的键被静默忽略，会让"我配置了
#: 这个功能"和"它根本没生效"同时成立。
_STARTUP_KEYS = {"time_scale", "tau_end_s", "handoff_dir", "build", "cases"}
_STARTUP_BUILD_KEYS = {
    "delta", "outer_radius_factor", "inner_cells", "fan_cells",
    "plateau_cells", "air_cells", "ambient_cells", "dense_samples",
    "integration_step_factor", "root_relative_tolerance",
}
_CASE_KEYS = {"case_id", "charge_radius_m", "split", "handoff_json", "data"}
_DATA_KEYS = {"path", "synthetic"}


def _reject_unknown(section: Mapping[str, Any], known: set[str], *, where: str) -> None:
    unknown = sorted(set(section) - known)
    if unknown:
        raise CaseSetupError(
            f"{where}: unknown key(s) {unknown}; refusing to ignore configuration that "
            "has no effect"
        )


def _startup_cases_spec(config: Config) -> list[dict[str, Any]]:
    startup_section = config.section("startup")
    _reject_unknown(startup_section, _STARTUP_KEYS, where="startup")
    build_section = dict(startup_section.get("build") or {})
    if build_section:
        _reject_unknown(build_section, _STARTUP_BUILD_KEYS, where="startup.build")
    specs = config.list_of("startup.cases", [])
    if not specs:
        raise CaseSetupError(
            "no startup cases are configured. Add a 'startup:' section with "
            "'cases:' entries (case_id, charge_radius_m, split, optional handoff_json "
            "and data). The legacy data.cases section describes the old extraction "
            "time axis and is not used by the new chain."
        )
    if not all(isinstance(spec, Mapping) for spec in specs):
        raise CaseSetupError("every startup.cases entry must be a mapping")
    result = []
    for spec in specs:
        _reject_unknown(spec, _CASE_KEYS, where=f"startup case {spec.get('case_id')!r}")
        data = spec.get("data")
        if isinstance(data, Mapping):
            _reject_unknown(
                data, _DATA_KEYS,
                where=f"startup case {spec.get('case_id')!r} data",
            )
        result.append(dict(spec))
    return result


def _require_field_matches_config(
    field: PlanarStartupField, config: Config, *, case_id: str
) -> None:
    """The handoff is authoritative; explicit YAML physical values must agree."""
    try:
        problems = config_condition_mismatches(
            field, config, where=f"case {case_id!r} startup field"
        )
    except MaterialConditionsError as exc:
        raise CaseSetupError(f"case {case_id!r}: {exc}") from exc
    if problems:
        raise CaseSetupError(
            f"case {case_id!r}: configuration and startup field disagree on physical "
            "conditions: " + "; ".join(problems)
        )


def _build_or_load_field(
    spec: Mapping[str, Any], config: Config, handoff_root: Path
) -> tuple[PlanarStartupField, Path]:
    radius = _positive("charge_radius_m", spec.get("charge_radius_m"))
    case_id = str(spec.get("case_id", "") or "")
    explicit = spec.get("handoff_json")
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = config.root / path
        if not path.exists():
            raise CaseSetupError(
                f"handoff_json {path} does not exist; refusing to build a different "
                "startup field silently"
            )
        field = load_planar_startup_handoff(path)
        if abs(float(field.charge_radius) - radius) > 1.0e-12 * max(radius, 1.0):
            raise CaseSetupError(
                f"case {case_id!r}: handoff has R_0 = "
                f"{field.charge_radius!r} m, configuration says {radius!r} m"
            )
        _require_field_matches_config(field, config, case_id=case_id)
        return field, path

    build = dict(config.section("startup.build") or {})
    outer_factor = float(build.get("outer_radius_factor", 40.0))
    if not math.isfinite(outer_factor) or outer_factor <= 1.0:
        raise CaseSetupError(
            f"startup.build.outer_radius_factor must be > 1, got {outer_factor!r}"
        )
    try:
        jwl, e0_volume_energy = configured_jwl_and_energy(config)
        air_gamma, air_density, air_pressure = configured_air_state(config)
    except MaterialConditionsError as exc:
        raise CaseSetupError(f"case {case_id!r}: {exc}") from exc
    initial = build_two_medium_initial_field(
        charge_radius=radius,
        outer_radius=outer_factor * radius,
        jwl=jwl,
        product_specific_energy=jwl.initial_specific_energy(e0_volume_energy),
        air_density=air_density,
        air_pressure=air_pressure,
        air_gamma=air_gamma,
    )
    startup_config = PlanarStartupConfig(
        delta=float(build.get("delta", 1.0 / 200.0)),
        inner_cells=int(build.get("inner_cells", 64)),
        fan_cells=int(build.get("fan_cells", 96)),
        plateau_cells=int(build.get("plateau_cells", 8)),
        air_cells=int(build.get("air_cells", 16)),
        ambient_cells=int(build.get("ambient_cells", 64)),
        dense_samples=int(build.get("dense_samples", 8192)),
        integration_step_factor=float(build.get("integration_step_factor", 0.005)),
        root_relative_tolerance=float(build.get("root_relative_tolerance", 1.0e-11)),
    )
    field = build_planar_startup_field(initial, config=startup_config)
    if field.checks.get("passed") is not True:
        raise CaseSetupError(
            f"case {case_id!r}: the freshly built startup field did not "
            "pass its consistency checks"
        )
    # The same check as for a handoff: proves that the configured JWL, E0,
    # air gamma/density/pressure and the resulting initial state were the ones
    # actually used, not merely accepted and then ignored.
    _require_field_matches_config(field, config, case_id=case_id)
    directory = Path(handoff_root) / PLANAR_STARTUP_MODEL_ID / field.cache_key()[:16]
    json_path, _ = write_planar_startup_handoff(field, directory)
    return field, json_path


def _positive(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise CaseSetupError(f"{name} must be finite and positive, got {value!r}")
    return number


def _development_fraction(config: Config) -> float:
    raw = config.get("data.development_holdout_fraction", 0.0)
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise CaseSetupError(
            f"data.development_holdout_fraction must be a number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or not (0.0 <= value < 1.0):
        raise CaseSetupError(
            "data.development_holdout_fraction must be finite and lie in [0, 1), "
            f"got {raw!r}"
        )
    return value


def _apply_development_split(
    dataset: SupervisionDataset, config: Config, *, case_id: str
) -> None:
    fraction = _development_fraction(config)
    if fraction == 0.0:
        return
    raw_group = config.get("data.development_group_frames", None)
    if raw_group is None:
        raise CaseSetupError(
            f"case {case_id!r}: data.development_holdout_fraction={fraction!r} "
            "requests a development split but data.development_group_frames is not set; "
            "the consecutive-frame group size must be explicit"
        )
    if isinstance(raw_group, bool):
        raise CaseSetupError(
            f"case {case_id!r}: data.development_group_frames must be an integer, got "
            f"{raw_group!r}"
        )
    try:
        group_frames = int(raw_group)
    except (TypeError, ValueError) as exc:
        raise CaseSetupError(
            f"case {case_id!r}: data.development_group_frames must be an integer, got "
            f"{raw_group!r}"
        ) from exc
    if group_frames != raw_group or group_frames < 1:
        raise CaseSetupError(
            f"case {case_id!r}: data.development_group_frames must be an integer >= 1, "
            f"got {raw_group!r}"
        )
    raw_seed = config.get("data.split_seed", 0)
    if isinstance(raw_seed, bool):
        raise CaseSetupError(
            f"case {case_id!r}: data.split_seed must be an integer, got {raw_seed!r}"
        )
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise CaseSetupError(
            f"case {case_id!r}: data.split_seed must be an integer, got {raw_seed!r}"
        ) from exc
    if seed != raw_seed:
        raise CaseSetupError(
            f"case {case_id!r}: data.split_seed must be an integer, got {raw_seed!r}"
        )
    try:
        dataset.assign_frame_split(
            group_frames=group_frames,
            holdout_fraction=fraction,
            seed=seed,
        )
    except SupervisionError as exc:
        raise CaseSetupError(f"case {case_id!r}: {exc}") from exc


def _load_supervision(
    spec: Mapping[str, Any], config: Config, *, case_id: str, radius: float
) -> SupervisionDataset | None:
    entry = spec.get("data")
    if entry is None:
        fraction = _development_fraction(config)
        if fraction > 0.0:
            raise CaseSetupError(
                f"case {case_id!r}: data.development_holdout_fraction={fraction!r} "
                "requests a development split, but this case has no supervision data. "
                "Disable the split or provide labelled data with explicit time origin"
            )
        return None
    if isinstance(entry, Mapping):
        path_value = entry.get("path")
        declared_synthetic = entry.get("synthetic")
    else:
        path_value = entry
        declared_synthetic = None
    if not path_value:
        raise CaseSetupError(
            f"case {case_id!r}: data entry has no 'path'"
        )
    path = Path(str(path_value))
    if not path.is_absolute():
        path = config.root / path
    try:
        dataset = SupervisionDataset.load(path)
    except SupervisionError as exc:
        raise CaseSetupError(f"case {case_id!r}: {exc}") from exc
    if dataset.case_id != case_id:
        raise CaseSetupError(
            f"case {case_id!r}: the supervision file declares case_id "
            f"{dataset.case_id!r}; refusing to relabel data"
        )
    if abs(dataset.charge_radius_m - radius) > 1.0e-12 * max(radius, 1.0):
        raise CaseSetupError(
            f"case {case_id!r}: supervision R_0 = {dataset.charge_radius_m!r} m, "
            f"configuration says {radius!r} m"
        )
    if declared_synthetic is not None and bool(declared_synthetic) != dataset.synthetic:
        raise CaseSetupError(
            f"case {case_id!r}: configuration says synthetic={declared_synthetic!r} "
            f"but the file says synthetic={dataset.synthetic!r}"
        )
    try:
        dataset.require_supervision_ready()
    except SupervisionError as exc:
        raise CaseSetupError(
            f"case {case_id!r}: supervision cannot be enabled: {exc}"
        ) from exc
    _apply_development_split(dataset, config, case_id=case_id)
    return dataset


def build_startup_training_context(
    config: Config,
    *,
    seed: int = 0,
    diagnostic: bool = False,
    device: str | None = None,
    handoff_root: Path | str | None = None,
    sampling: SamplingConfig | None = None,
) -> StartupTrainingContext:
    """装配所有训练算例；任何缺失/错配都在这里硬失败。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    try:
        rar_config = RARConfig.from_config(config)
    except RARError as exc:
        raise CaseSetupError(str(exc)) from exc
    for key in ("startup.development_holdout_fraction", "training.development_holdout_fraction"):
        value = config.get(key, None)
        if value not in (None, 0, 0.0):
            raise CaseSetupError(
                f"configuration {key!r} = {value!r} is ignored by this chain; put the "
                "development split under data.development_holdout_fraction"
            )
    specs = _startup_cases_spec(config)
    time_scale = float(config.get("startup.time_scale", 2.0))
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        raise CaseSetupError(
            f"startup.time_scale must be finite and positive, got {time_scale!r}"
        )
    tau_end = _positive("startup.tau_end_s", config.get("startup.tau_end_s", 1.0e-6))
    sampling = sampling or SamplingConfig.from_config(config)
    if handoff_root is None:
        handoff_root = config.path_of(
            "startup.handoff_dir", default=Path("outputs/planar_startup")
        )
    handoff_root = Path(handoff_root)

    final_validation = {
        float(value) for value in config.list_of(
            "splits.final_validation_radii_mm", []
        )
    }
    cases: list[StartupCase] = []
    seen: set[str] = set()
    notes: list[str] = []
    for index, spec in enumerate(specs):
        case_id = str(spec.get("case_id", "") or "").strip()
        if not case_id:
            raise CaseSetupError("every startup case needs a non-empty case_id")
        if case_id in seen:
            raise CaseSetupError(f"duplicate case_id {case_id!r} in startup.cases")
        seen.add(case_id)
        split = str(spec.get("split", "train"))
        if split != "train":
            raise CaseSetupError(
                f"case {case_id!r}: split={split!r} is not a training split. Final "
                "validation cases must stay isolated (§16)."
            )
        radius = _positive("charge_radius_m", spec.get("charge_radius_m"))
        if radius * 1000.0 in final_validation:
            raise CaseSetupError(
                f"case {case_id!r}: R_0 = {radius * 1000.0:g} mm is listed under "
                "splits.final_validation_radii_mm and must not be used for training"
            )
        if "handoff_json" not in spec:
            notes.append(
                f"case {case_id!r}: no handoff_json; building a startup field and "
                "writing the handoff under startup.handoff_dir"
            )
        field, handoff_json = _build_or_load_field(spec, config, handoff_root)
        if field.checks.get("passed") is not True:
            raise CaseSetupError(
                f"case {case_id!r}: the startup field did not pass its consistency "
                "report"
            )
        scales = PlanarStartupScales.for_startup_field(field, time_scale=time_scale)
        data = _load_supervision(spec, config, case_id=case_id, radius=radius)
        sampler = RadiusSampler(
            field=field,
            scales=scales,
            tau_end=tau_end,
            case=data,
            case_id=case_id,
            config=sampling,
            seed=int(seed) + index,
            device=device,
            r_min_fraction=float(config.get("training.r_min_fraction", 1.0e-2)),
            rar_config=rar_config,
        )
        cases.append(StartupCase(
            case_id=case_id,
            split=split,
            field=field,
            scales=scales,
            sampler=sampler,
            contact_pressure=contact_pressure_scale(field),
            handoff_json=handoff_json,
            data=data,
            geometry={
                "charge_radius_m": radius,
                "epsilon_s": float(field.epsilon),
                "delta": float(field.delta),
            },
        ))

    compatibility = check_shared_compatibility(cases)
    first = cases[0]
    try:
        conditions = field_material_conditions(first.field)
    except MaterialConditionsError as exc:
        raise CaseSetupError(f"case {first.case_id!r}: {exc}") from exc
    model = build_model(
        config,
        scales=first.scales,
        conditions=conditions,
        initial_contact_hat=float(
            first.field.contact_radius / first.field.charge_radius
        ),
        initial_gap_hat=float(
            (first.field.shock_radius - first.field.contact_radius)
            / first.field.charge_radius
        ),
        seed=int(seed) + 1000,
        device=device,
    )
    # The first case is not exempt: the model was built from its conditions,
    # and this checks the object that will actually be trained.
    first_problems = condition_mismatches(
        conditions,
        model_material_conditions(model),
        reference_label=f"case {first.case_id!r} startup field",
        actual_label="shared model",
    )
    if first_problems:
        raise CaseSetupError(
            "the shared model does not match the first training case's material/"
            "environment: " + "; ".join(first_problems)
        )
    compatibility["model_material"] = {
        "checked_case_id": first.case_id,
        "mismatches": [],
    }
    return StartupTrainingContext(
        config=config,
        cases=cases,
        model=model,
        sampling=sampling,
        time_scale=time_scale,
        seed=int(seed),
        diagnostic=bool(diagnostic),
        shared_compatibility=compatibility,
        notes=notes,
    )


#: 共享网络允许变化的尺度量：只有装药半径与启动时刻。材料/参考尺度必须一致，
#: 否则同一条无量纲权重在不同半径下对应不同的物理量，不能称为共享模型。
_SHARED_SCALE_INVARIANTS = (
    "reference_velocity",
    "reference_specific_energy",
    "product_density",
    "air_density",
    "time_scale",
)


def check_shared_compatibility(cases: Sequence[StartupCase]) -> dict[str, Any]:
    """核对所有半径能否共用同一套无量纲网络参数。

    支持范围：相同材料、相同环境、相同 ``delta``、相同时间尺度与启动方法，
    只有 ``charge_radius`` 与 ``epsilon`` 不同。超出范围明确拒绝。
    """
    if not cases:
        raise CaseSetupError("shared training needs at least one case")
    first = cases[0]
    reference = {
        key: float(getattr(first.scales, key)) for key in _SHARED_SCALE_INVARIANTS
    }
    reference.update({
        "delta": float(first.field.delta),
        "model_id": str(first.field.model_id),
        "method_version": str(first.field.method_version),
        "initial_contact_hat": float(
            first.field.contact_radius / first.field.charge_radius
        ),
        "initial_gap_hat": float(
            (first.field.shock_radius - first.field.contact_radius)
            / first.field.charge_radius
        ),
    })
    try:
        reference_conditions = field_material_conditions(first.field)
    except MaterialConditionsError as exc:
        raise CaseSetupError(f"case {first.case_id!r}: {exc}") from exc
    problems: list[str] = []
    for case in cases[1:]:
        try:
            actual_conditions = field_material_conditions(case.field)
        except MaterialConditionsError as exc:
            raise CaseSetupError(f"case {case.case_id!r}: {exc}") from exc
        for problem in condition_mismatches(
            reference_conditions,
            actual_conditions,
            reference_label=f"first case {first.case_id!r}",
            actual_label=f"case {case.case_id!r}",
        ):
            problems.append(problem)
        for key in _SHARED_SCALE_INVARIANTS:
            value = float(getattr(case.scales, key))
            expected = reference[key]
            if abs(value - expected) > 1.0e-12 * max(
                abs(value), abs(expected), 1.0e-30
            ):
                problems.append(
                    f"case {case.case_id!r}: {key}={value!r}, first case has "
                    f"{expected!r}"
                )
        for key, actual in (
            ("delta", float(case.field.delta)),
            ("initial_contact_hat", float(
                case.field.contact_radius / case.field.charge_radius
            )),
            ("initial_gap_hat", float(
                (case.field.shock_radius - case.field.contact_radius)
                / case.field.charge_radius
            )),
        ):
            expected = reference[key]
            if abs(actual - expected) > 1.0e-12 * max(
                abs(actual), abs(expected), 1.0e-30
            ):
                problems.append(
                    f"case {case.case_id!r}: {key}={actual!r}, first case has "
                    f"{expected!r}"
                )
        for key, actual in (
            ("model_id", str(case.field.model_id)),
            ("method_version", str(case.field.method_version)),
        ):
            if actual != reference[key]:
                problems.append(
                    f"case {case.case_id!r}: {key}={actual!r}, first case has "
                    f"{reference[key]!r}"
                )
    if problems:
        raise CaseSetupError(
            "these radii cannot share one dimensionless network: "
            + "; ".join(problems[:6])
            + ". The supported range is identical materials/environment, delta, startup "
            "method and time scale; only R_0 and epsilon may differ."
        )
    return {
        "invariants": reference,
        "material": reference_conditions,
        "radii_m": [float(case.charge_radius) for case in cases],
        "epsilon_s": [float(case.field.epsilon) for case in cases],
        "note": (
            "shared ProductNet/AirNet/front parameters; per-case charge_radius, "
            "epsilon, sampler and supervision; identical JWL/air/initial state"
        ),
    }


__all__ = [
    "CaseSetupError",
    "StartupCase",
    "StartupTrainingContext",
    "build_startup_training_context",
]
