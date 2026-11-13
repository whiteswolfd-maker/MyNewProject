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

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from import_lsdyna_profiles_1d import convert  # noqa: E402
from spherical_blast_pinn.contact_air.cli import command_build_cache  # noqa: E402


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
        if not args.cache_only:
            if standard.is_file() and not args.overwrite:
                entry["import"] = "skipped_existing"
            else:
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
        if not args.import_only:
            if cache.is_file() and not args.overwrite:
                entry["cache"] = "skipped_existing"
            else:
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
