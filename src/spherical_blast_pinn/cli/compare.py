"""预测与外部对照数据对比（相同物理时间与位置）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.compare \
        --prediction outputs/predict/run.npz \
        --reference synthetic_case.npz --output outputs/compare/run.json

两边都必须带**显式时间原点**：预测从 ``epsilon`` 起算，参照数据用
``t_phys = T_DYNA - T_origin``。参照数据的原点未知时，本命令**拒绝**同时间
对比，而不是默认 ``T_origin=0``。

缺失或无效的量一律输出 ``null``，不填 0：没有观测不等于观测为零。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.dyna_supervision import pressure_gauge_series
from ..data.startup_supervision import SupervisionDataset, SupervisionError
from ..evaluation.field_compare import compare_field_blocks
from ._common import SetupError, configure_stdout


PRODUCT_CODE = 0
AIR_CODE = 1


def _load_prediction(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata_raw = archive["metadata"].item()
            text = (
                metadata_raw.decode("utf-8")
                if isinstance(metadata_raw, (bytes, bytearray))
                else str(metadata_raw)
            )
            metadata = json.loads(text)
            arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    except SetupError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SetupError(f"{path}: cannot read prediction file: {exc}") from exc
    if metadata.get("format") != "spherical_blast_pinn.startup_prediction":
        raise SetupError(
            f"{path}: format {metadata.get('format')!r} is not a startup prediction"
        )
    required = (
        "time_index", "region", "radius", "density", "velocity",
        "specific_energy", "pressure", "overpressure", "times_phys",
        "contact_radius", "shock_radius",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise SetupError(f"{path}: prediction is missing arrays {missing}")
    return arrays, metadata


def _relative_error(predicted: np.ndarray, reference: np.ndarray) -> tuple[float | None, float]:
    difference = np.asarray(predicted) - np.asarray(reference)
    rms = float(np.sqrt(np.mean(difference**2)))
    norm = float(np.sqrt(np.mean(np.asarray(reference) ** 2)))
    relative = None if not math.isfinite(norm) or norm <= 0.0 else rms / norm
    return relative, rms


def _frame_blocks(arrays: dict[str, np.ndarray], index: int):
    mask = arrays["time_index"] == index
    return {
        "radius": arrays["radius"][mask],
        "region": arrays["region"][mask],
        "density": arrays["density"][mask],
        "velocity": arrays["velocity"][mask],
        "specific_energy": arrays["specific_energy"][mask],
        "pressure": arrays["pressure"][mask],
        "overpressure": arrays["overpressure"][mask],
    }


def compare_prediction(
    prediction: Path,
    reference: Path,
    *,
    time_tolerance: float = 1.0e-12,
    gauge_radii_m: Sequence[float] = (),
) -> dict[str, Any]:
    if not reference.exists():
        raise SetupError(f"reference file {reference} does not exist")
    try:
        dataset = SupervisionDataset.load(reference)
    except SupervisionError as exc:
        raise SetupError(str(exc)) from exc
    if not dataset.time_origin.ready:
        raise SetupError(
            f"reference data cannot be compared at the same physical time: the "
            f"DYNA-to-model time origin is unknown ({dataset.time_origin.evidence!r}); "
            "state which DYNA time is t_phys=0 instead of assuming T_origin=0"
        )
    arrays, metadata = _load_prediction(prediction)
    if abs(
        dataset.charge_radius_m - float(metadata["charge_radius_m"])
    ) > 1.0e-12 * max(dataset.charge_radius_m, 1.0):
        raise SetupError(
            f"reference R_0 = {dataset.charge_radius_m!r} m, prediction R_0 = "
            f"{metadata['charge_radius_m']!r} m"
        )
    prediction_times = np.asarray(arrays["times_phys"], dtype=np.float64)
    reference_times = dataset.absolute_physical_time()
    matched: list[tuple[int, int]] = []
    for predicted_index, time in enumerate(prediction_times):
        distances = np.abs(reference_times - time)
        if distances.size and float(distances.min()) <= (
            time_tolerance * max(abs(time), 1.0e-30)
        ):
            matched.append((predicted_index, int(np.argmin(distances))))
    if not matched:
        raise SetupError(
            "no reference frame has the same physical time as a prediction frame; "
            "the two time origins or time axes disagree"
        )
    radius = float(metadata["charge_radius_m"])
    frames: list[dict[str, Any]] = []
    for predicted_index, reference_index in matched:
        blocks = _frame_blocks(arrays, predicted_index)
        product_mask = blocks["region"] == PRODUCT_CODE
        air_mask = blocks["region"] == AIR_CODE
        predicted_contact = float(arrays["contact_radius"][predicted_index])
        predicted_shock = float(arrays["shock_radius"][predicted_index])
        reference_contact = float(dataset.contact_radius[reference_index])
        reference_shock = float(dataset.shock_radius[reference_index])
        contact_valid = bool(dataset.contact_valid[reference_index])
        shock_valid = bool(dataset.shock_valid[reference_index])
        product = {
            "density": dataset.valid_density[reference_index],
            "velocity": dataset.valid_velocity[reference_index],
            "specific_energy": dataset.valid_specific_energy[reference_index],
            "pressure": dataset.valid_pressure[reference_index],
        }
        reference_values = {
            "density": dataset.density[reference_index],
            "velocity": dataset.velocity[reference_index],
            "specific_energy": dataset.specific_energy[reference_index],
            "pressure": dataset.pressure[reference_index],
        }
        air_region = dataset.medium[reference_index] == AIR_CODE
        product_region = dataset.medium[reference_index] == PRODUCT_CODE
        fields: dict[str, Any] = {}
        for name in ("density", "velocity", "specific_energy", "pressure"):
            product_entry = compare_field_blocks(
                reference_radius=dataset.radius[reference_index],
                reference_values=reference_values[name],
                valid=product[name] & product_region,
                predicted_radius=blocks["radius"][product_mask],
                predicted_values=blocks[name][product_mask],
                low=0.0,
                high=predicted_contact,
            )
            air_entry = compare_field_blocks(
                reference_radius=dataset.radius[reference_index],
                reference_values=reference_values[name],
                valid=product[name] & air_region,
                predicted_radius=blocks["radius"][air_mask],
                predicted_values=blocks[name][air_mask],
                low=predicted_contact,
                high=predicted_shock,
            )
            fields[f"product_{name}"] = product_entry
            fields[f"air_{name}"] = air_entry
        # 超压曲线只在空气侧、且参照压力有效时比较。
        ambient_pressure = float(metadata.get("ambient_pressure", float("nan")))
        reference_overpressure = (
            reference_values["pressure"] - ambient_pressure
        )
        overpressure = compare_field_blocks(
            reference_radius=dataset.radius[reference_index],
            reference_values=reference_overpressure,
            valid=dataset.valid_pressure[reference_index] & air_region,
            predicted_radius=blocks["radius"][air_mask],
            predicted_values=blocks["overpressure"][air_mask],
            low=predicted_contact,
            high=predicted_shock,
        )
        curve_mask = (
            dataset.valid_pressure[reference_index]
            & air_region
            & np.isfinite(reference_overpressure)
            & (dataset.radius[reference_index] >= predicted_contact)
            & (dataset.radius[reference_index] <= predicted_shock)
        )
        if bool(curve_mask.any()) and blocks["radius"][air_mask].size >= 2:
            curve_radius = dataset.radius[reference_index][curve_mask]
            curve_predicted = np.interp(
                curve_radius,
                blocks["radius"][air_mask],
                blocks["overpressure"][air_mask],
            )
        else:
            curve_radius = np.zeros(0)
            curve_predicted = np.zeros(0)
        overpressure["curve"] = {
            "radius_m": [float(value) for value in curve_radius],
            "predicted_overpressure_pa": [float(value) for value in curve_predicted],
            "reference_overpressure_pa": [
                float(value) for value in reference_overpressure[curve_mask]
            ],
        }
        frames.append({
            "prediction_index": int(predicted_index),
            "reference_index": int(reference_index),
            "physical_time_s": float(prediction_times[predicted_index]),
            "tau_s": float(prediction_times[predicted_index]) - float(metadata["epsilon_s"]),
            "prediction_extrapolated": bool(
                metadata.get("extrapolated", [False] * len(prediction_times))[predicted_index]
            ),
            "fronts": {
                "contact_radius_predicted": predicted_contact,
                "contact_radius_reference": reference_contact if contact_valid else None,
                "contact_relative_error": (
                    abs(predicted_contact - reference_contact) / radius
                    if contact_valid else None
                ),
                "contact_available": contact_valid,
                "shock_radius_predicted": predicted_shock,
                "shock_radius_reference": reference_shock if shock_valid else None,
                "shock_relative_error": (
                    abs(predicted_shock - reference_shock) / radius
                    if shock_valid else None
                ),
                "shock_available": shock_valid,
            },
            "fields": fields,
            "overpressure": overpressure,
        })

    gauge_ambient = float(metadata.get("ambient_pressure", float("nan")))
    gauges: list[dict[str, Any]] = []
    if gauge_radii_m:
        if not math.isfinite(gauge_ambient):
            raise SetupError(
                "prediction metadata has no finite ambient_pressure; fixed-gauge "
                "overpressure cannot be formed"
            )
        if "epsilon_s" not in metadata:
            raise SetupError(
                "prediction metadata has no epsilon_s; fixed-gauge tau cannot be formed"
            )
        for gauge_radius in gauge_radii_m:
            radius_value = float(gauge_radius)
            series = pressure_gauge_series(
                dataset,
                radius_m=radius_value,
                ambient_pressure_pa=gauge_ambient,
                epsilon_s=float(metadata["epsilon_s"]),
            )
            entries = []
            for frame in frames:
                reference_index = int(frame["reference_index"])
                predicted_index = int(frame["prediction_index"])
                blocks = _frame_blocks(arrays, predicted_index)
                air_mask = blocks["region"] == AIR_CODE
                air_radius = blocks["radius"][air_mask]
                air_overpressure = blocks["overpressure"][air_mask]
                predicted_value = None
                predicted_reason = None
                if air_radius.size < 2:
                    predicted_reason = "prediction has fewer than two air points"
                elif not (
                    float(frame["fronts"]["contact_radius_predicted"])
                    <= radius_value
                    <= float(frame["fronts"]["shock_radius_predicted"])
                ):
                    predicted_reason = "outside predicted air block"
                elif radius_value < float(air_radius.min()) or radius_value > float(air_radius.max()):
                    predicted_reason = "outside predicted air samples"
                else:
                    predicted_value = float(
                        np.interp(radius_value, air_radius, air_overpressure)
                    )
                reference_value = (
                    series["overpressure_pa"][reference_index]
                    if series["valid"][reference_index] else None
                )
                entries.append({
                    "reference_index": reference_index,
                    "prediction_index": predicted_index,
                    "physical_time_s": float(frame["physical_time_s"]),
                    "reference_overpressure_pa": reference_value,
                    "predicted_overpressure_pa": predicted_value,
                    "reference_reason": series["reason"][reference_index],
                    "predicted_reason": predicted_reason,
                })
            gauges.append({
                "radius_m": radius_value,
                "tracking": series["tracking"],
                "interpolation": series["interpolation"],
                "incident_or_reflected": series["incident_or_reflected"],
                "frames": entries,
            })

    def aggregate(key: str) -> float | None:
        values = []
        for frame in frames:
            if key == "overpressure":
                value = frame["overpressure"]["relative_l2"]
            elif key.startswith("field:"):
                value = frame["fields"][key[len("field:"):]]["relative_l2"]
            elif key in ("contact", "shock"):
                value = frame["fronts"][f"{key}_relative_error"]
            else:  # pragma: no cover - programming error
                value = None
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return float(np.mean(values)) if values else None

    return {
        "prediction": str(prediction),
        "reference": str(reference),
        "case_id": dataset.case_id,
        "reference_synthetic": bool(dataset.synthetic),
        "reference_source_kind": dataset.source_kind,
        "reference_time_origin": dataset.time_origin.as_dict(),
        "charge_radius_m": radius,
        "epsilon_s": float(metadata["epsilon_s"]),
        "time_convention": metadata.get("time_convention"),
        "matched_frames": len(frames),
        "frames": frames,
        "gauges": gauges,
        "summary": {
            "contact_relative_error_mean": aggregate("contact"),
            "shock_relative_error_mean": aggregate("shock"),
            "overpressure_relative_l2_mean": aggregate("overpressure"),
            "fields": {
                key: aggregate(f"field:{key}")
                for key in (
                    "product_density", "product_velocity", "product_specific_energy",
                    "product_pressure", "air_density", "air_velocity",
                    "air_specific_energy", "air_pressure",
                )
            },
        },
        "missing_policy": (
            "null means the quantity is absent or invalid at that time; missing data is "
            "never replaced by zero"
        ),
        "synthetic_note": (
            "this comparison is against a SYNTHETIC dataset for interface verification; "
            "it is not a validation against LS-DYNA"
            if dataset.synthetic else
            "reference data is marked real; its time origin was explicitly confirmed by "
            "the caller"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (SetupError, SupervisionError) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--gauge-radius-m",
        action="append",
        type=float,
        default=[],
        help="fixed physical radius for a reference/prediction pressure history; repeatable",
    )
    args = parser.parse_args(argv)
    configure_stdout()
    result = compare_prediction(
        Path(args.prediction),
        Path(args.reference),
        time_tolerance=args.time_tolerance,
        gauge_radii_m=args.gauge_radius_m,
    )
    output = Path(args.output) if args.output else Path(args.prediction).with_suffix(".compare.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"matched frames: {result['matched_frames']}")
    for key, value in result["summary"].items():
        if key == "fields":
            continue
        print(f"  {key}: {value}")
    for key, value in result["summary"]["fields"].items():
        print(f"  field {key}: {value}")
    for gauge in result["gauges"]:
        reference_count = sum(
            entry["reference_overpressure_pa"] is not None
            for entry in gauge["frames"]
        )
        predicted_count = sum(
            entry["predicted_overpressure_pa"] is not None
            for entry in gauge["frames"]
        )
        print(
            f"  gauge r={gauge['radius_m']:.6g} m: "
            f"reference {reference_count}/{len(gauge['frames'])}, "
            f"prediction {predicted_count}/{len(gauge['frames'])}"
        )
    print(f"written: {output}")
    if not result["reference_synthetic"]:
        print("reference is marked REAL; verify its time origin and extraction provenance.")
    else:
        print("reference is SYNTHETIC: interface verification only, not DYNA validation.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
