#!/usr/bin/env python3
"""Import every raw profile frame and build the full-frame caches.

This is an orchestrator, not a second data path.  It calls the existing
``import_lsdyna_profiles_1d.convert`` and the existing
``contact_air.cli.command_build_cache``; it only supplies the all-frames
output directories from ``reference_audit_fullframes.yaml`` and records
per-case raw/imported/cache frame counts.

Old strided standards, caches and checkpoints are never touched: the output
paths are under ``outputs/fullframes/``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from import_lsdyna_profiles_1d import (  # noqa: E402
    _input_fingerprint,
    _profile_frames,
    convert,
)
from spherical_blast_pinn.contact_air.cli import command_build_cache  # noqa: E402
from spherical_blast_pinn.contact_air.one_d import (  # noqa: E402
    standard_source_digest,
)

_PROFILE_PATTERNS = {
    "velocity": "profile1_x-velocity_x_time*.xy",
    "pressure": "profile2_pressure_x_time*.xy",
    "density": "profile3_density_x_time*.xy",
    "alpha_product": "profile4_volume_fraction_x_time*.xy",
}


def _raw_paths(raw_dir: Path) -> dict[str, dict[str, Path]]:
    paths = {
        name: _profile_frames(raw_dir, pattern, name)
        for name, pattern in _PROFILE_PATTERNS.items()
    }
    pattern = _alpha_air_pattern(raw_dir)
    if pattern is not None:
        paths["alpha_air"] = _profile_frames(raw_dir, pattern, "alpha_air")
    return paths


def _standard_is_current(
    raw_dir: Path, standard: Path,
) -> tuple[bool, str]:
    """Check that an existing standard really covers the current raw frames."""
    if not standard.is_file():
        return False, "standard is missing"
    with np.load(standard, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    source = metadata.get("source", {})
    if source.get("frame_selection_policy") != "all_frames":
        return False, "frame_selection_policy is not all_frames"
    paths = _raw_paths(raw_dir)
    raw_keys = sorted(paths["pressure"], key=float)
    selected = list(source.get("selected_raw_frame_keys", []))
    raw_meta = list(source.get("raw_frame_keys", []))
    times = list(metadata.get("frame_times", []))
    if not (len(raw_keys) == len(selected) == len(raw_meta) == len(times)):
        return False, (
            f"frame counts differ: disk={len(raw_keys)} selected={len(selected)} "
            f"metadata={len(raw_meta)} frame_times={len(times)}"
        )
    if set(selected) != set(raw_keys) or set(raw_meta) != set(raw_keys):
        return False, "raw frame key sets differ from disk"
    for key, value in zip(selected, times):
        if abs(float(key) - float(value)) > 1.0e-12 * max(1.0, abs(float(value))):
            return False, f"frame_time {value!r} does not match raw key {key!r}"
    fingerprint = _input_fingerprint(paths)
    if source.get("raw_input_fingerprint") != fingerprint:
        return False, "raw input fingerprint changed (stale standard)"
    if source.get("raw_frame_count") != len(raw_keys):
        return False, "metadata raw_frame_count differs from disk"
    if source.get("imported_frame_count") != len(raw_keys):
        return False, "metadata imported_frame_count differs from disk"
    return True, "current"


def _cache_is_current(standard: Path, cache: Path) -> tuple[bool, str]:
    """Check that a cache was built from this exact standard file.

    Frame counts and times are not sufficient: a standard rebuilt from changed
    raw data can keep the same frames and times while carrying different
    pressure/density arrays.  The cache therefore records the standard's
    content digest at build time, and that digest must still match.
    """
    if not cache.is_file():
        return False, "cache is missing"
    if not standard.is_file():
        return False, "standard is missing"
    with np.load(standard, allow_pickle=False) as archive:
        standard_meta = json.loads(str(archive["metadata"]))
    frame_times = np.asarray(standard_meta["frame_times"], dtype=np.float64)
    standard_digest = standard_source_digest(standard)
    with np.load(cache, allow_pickle=False) as archive:
        if "raw_times" not in archive.files:
            return False, "cache has no raw_times"
        raw_times = np.asarray(archive["raw_times"], dtype=np.float64)
        contact_times = np.asarray(archive["contact_times"], dtype=np.float64)
        shock_times = np.asarray(archive["shock_times"], dtype=np.float64)
        cache_meta = json.loads(str(archive["metadata"]))
    recorded = (
        ((cache_meta.get("case") or {}).get("source") or {})
        .get("standard_data_digest")
    )
    if not isinstance(recorded, dict) or "sha256" not in recorded:
        return False, (
            "cache has no standard data fingerprint; it predates fingerprinting "
            "and cannot be proven current"
        )
    if recorded.get("algorithm") != standard_digest["algorithm"]:
        return False, (
            f"cache fingerprint algorithm {recorded.get('algorithm')!r} != "
            f"{standard_digest['algorithm']!r}"
        )
    if recorded.get("sha256") != standard_digest["sha256"]:
        return False, (
            "cache standard fingerprint differs from the current standard: "
            f"cache={recorded.get('sha256')} standard={standard_digest['sha256']}"
        )
    if raw_times.size != frame_times.size:
        return False, (
            f"cache raw_times {raw_times.size} != standard frames {frame_times.size}"
        )
    if not np.allclose(raw_times, frame_times, rtol=0.0, atol=1.0e-15):
        return False, "cache raw_times differ from standard frame_times"
    if contact_times.size != frame_times.size or shock_times.size != frame_times.size:
        return False, (
            f"cache frame arrays {contact_times.size}/{shock_times.size} != "
            f"standard frames {frame_times.size}"
        )
    return True, "current"


def _audit(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["audit"]


def _case_config(audit: dict, case_id: str) -> dict:
    for case in audit["cases"]:
        if str(case["case_id"]) == case_id:
            return case
    raise SystemExit(f"case {case_id!r} is not in the audit config")


def _alpha_air_pattern(raw_dir: Path) -> str | None:
    if list(raw_dir.glob("profile5_volume_fraction_x_time*.xy")):
        return "profile5_volume_fraction_x_time*.xy"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(
            _REPO_ROOT / "configs/contact_air/reference_audit_fullframes.yaml"
        ),
    )
    parser.add_argument("--cases", default=None,
                        help="comma-separated case ids; default all")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild an existing full-frame standard/cache")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--summary", default="outputs/fullframes/summary.json",
        help="summary JSON path (use distinct paths for parallel case groups)",
    )
    args = parser.parse_args(argv)
    audit = _audit(Path(args.config))
    root = (_REPO_ROOT / audit.get("repo_root", ".")).resolve()
    cases = (
        [item.strip() for item in args.cases.split(",") if item.strip()]
        if args.cases else [str(case["case_id"]) for case in audit["cases"]]
    )
    training_config = audit.get(
        "training_config",
        "configs/contact_air/split_train_r50_500_fullframes.yaml",
    )
    summaries = []
    for case_id in cases:
        case = _case_config(audit, case_id)
        raw_dir = Path(case["raw_dir"])
        standard = root / case["standard"]
        cache = root / case["cache"]
        started = time.time()
        entry: dict[str, object] = {
            "case_id": case_id,
            "raw_dir": str(raw_dir),
            "standard": str(standard),
            "cache": str(cache),
        }
        standard_rebuilt = False
        if args.cache_only:
            ok, reason = _standard_is_current(raw_dir, standard)
            if not ok:
                raise SystemExit(
                    f"{case_id}: --cache-only refuses a stale/missing standard: "
                    f"{reason}. Re-run without --cache-only to rebuild it."
                )
            entry["standard_validation"] = reason
        if not args.cache_only:
            reusable = standard.is_file() and not args.overwrite
            reason = "not present"
            if reusable:
                reusable, reason = _standard_is_current(raw_dir, standard)
                if reusable:
                    entry["import"] = "skipped_existing"
                    entry["standard_validation"] = reason
                else:
                    entry["standard_validation"] = f"rebuild: {reason}"
            if not reusable:
                pattern = _alpha_air_pattern(raw_dir)
                summary = convert(
                    source_dir=raw_dir, case_id=case_id,
                    radius_m=float(case["radius_m"]), out=standard,
                    velocity_pattern="profile1_x-velocity_x_time*.xy",
                    pressure_pattern="profile2_pressure_x_time*.xy",
                    density_pattern="profile3_density_x_time*.xy",
                    alpha_product_pattern="profile4_volume_fraction_x_time*.xy",
                    alpha_air_pattern=pattern,
                )
                entry["import"] = summary
                entry["standard_bytes"] = int(standard.stat().st_size)
                standard_rebuilt = True
        entry["standard_rebuilt"] = standard_rebuilt
        if not args.import_only:
            cache_existed = cache.is_file() and not args.overwrite
            reusable = cache_existed
            reason = "not present"
            if reusable and standard_rebuilt:
                # A cache is derived from the standard.  When this run rewrote
                # the standard, the old cache must not survive even if the
                # frame times did not change.
                reusable, reason = False, "standard was rebuilt in this run"
            if reusable:
                if not standard.is_file():
                    reusable, reason = False, "standard is missing"
                else:
                    reusable, reason = _cache_is_current(standard, cache)
                if reusable:
                    entry["cache"] = "skipped_existing"
                    entry["cache_validation"] = reason
                else:
                    entry["cache_validation"] = f"rebuild: {reason}"
            elif cache_existed:
                entry["cache_validation"] = f"rebuild: {reason}"
            if not reusable:
                if not standard.is_file():
                    raise SystemExit(
                        f"{case_id}: standard source {standard} is missing"
                    )
                cache.parent.mkdir(parents=True, exist_ok=True)
                build_args = argparse.Namespace(
                    config=str(_REPO_ROOT / training_config),
                    source=str(standard), case=case_id,
                    radius=float(case["radius_m"]), role=None,
                    out=str(cache),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    command_build_cache(build_args)
                entry["cache"] = "built"
                entry["cache_bytes"] = int(cache.stat().st_size)
        entry["elapsed_s"] = round(time.time() - started, 2)
        summaries.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    out = root / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
