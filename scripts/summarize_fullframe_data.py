#!/usr/bin/env python3
"""Summarise raw, imported and cache frame counts for the full-frame dataset.

Read-only: opens the standards and caches, reads metadata and validity masks,
and writes one CSV/Markdown table.  It never rebuilds or edits data.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_audit(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["audit"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(
            _REPO_ROOT / "configs/contact_air/reference_audit_fullframes.yaml"
        ),
    )
    parser.add_argument(
        "--output", default=str(
            _REPO_ROOT / "outputs/fullframes/frame_inventory.csv"
        ),
    )
    args = parser.parse_args(argv)
    audit = _load_audit(Path(args.config))
    root = (_REPO_ROOT / audit.get("repo_root", ".")).resolve()
    rows = []
    for case in audit["cases"]:
        case_id = str(case["case_id"])
        standard = root / case["standard"]
        cache = root / case["cache"]
        row = {"case_id": case_id}
        if standard.is_file():
            with np.load(standard, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            source = metadata.get("source", {})
            row.update({
                "standard_exists": True,
                "standard_bytes": standard.stat().st_size,
                "raw_frames": source.get("raw_frame_count"),
                "imported_frames": source.get("imported_frame_count",
                                              len(metadata["frame_times"])),
                "dropped_frames": source.get("dropped_frame_count"),
                "frame_selection_policy": source.get("frame_selection_policy"),
                "first_time_s": metadata["frame_times"][0],
                "last_time_s": metadata["frame_times"][-1],
                "cells_per_frame": metadata["cells_per_frame"][0],
                "fingerprint": (source.get("raw_input_fingerprint") or {}).get("sha256"),
            })
        else:
            row["standard_exists"] = False
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as archive:
                cache_meta = json.loads(str(archive["metadata"]))
                contact_frames = int(np.asarray(archive["contact_times"]).size)
                contact_valid = int(np.count_nonzero(archive["contact_valid"]))
                shock_valid = int(np.count_nonzero(archive["shock_valid"]))
                observations = int(np.asarray(archive["obs_time"]).size)
            case_meta = cache_meta["case"]
            status = (case_meta.get("separation", {}) or {}).get(
                "status_counts", {})
            row.update({
                "cache_exists": True,
                "cache_bytes": cache.stat().st_size,
                "cache_frames": contact_frames,
                "contact_valid_frames": contact_valid,
                "shock_valid_frames": shock_valid,
                "observations": observations,
                "separated_frames": status.get("separated"),
                "unresolvable_frames": status.get("unresolvable"),
                "t0_s": case_meta.get("t0"),
                "t_end_s": case_meta.get("t_end"),
            })
        else:
            row["cache_exists"] = False
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"inventory: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
