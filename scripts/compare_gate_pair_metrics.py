#!/usr/bin/env python3
"""Compare old and corrected C/D metrics without rewriting either file."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compare(old, new, path, result):
    if isinstance(old, dict):
        if not isinstance(new, dict):
            result["missing_or_changed"].append({
                "path": path, "reason": "new is not a dict",
            })
            return
        for key, value in old.items():
            child = f"{path}/{key}" if path else key
            if key not in new:
                result["missing_or_changed"].append({
                    "path": child, "reason": "missing in new",
                })
                continue
            _compare(value, new[key], child, result)
        return
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            result["missing_or_changed"].append({
                "path": path,
                "reason": f"list length old={len(old)} "
                          f"new={len(new) if isinstance(new, list) else 'n/a'}",
            })
            return
        for index, (left, right) in enumerate(zip(old, new)):
            _compare(left, right, f"{path}/{index}", result)
        return
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        if not isinstance(new, (int, float)) or isinstance(new, bool):
            result["missing_or_changed"].append({
                "path": path, "reason": "new is not numeric",
            })
            return
        if not (math.isfinite(float(old)) and math.isfinite(float(new))):
            if float(old) != float(new):
                result["missing_or_changed"].append({
                    "path": path, "reason": "nonfinite difference",
                    "old": old, "new": new,
                })
            return
        difference = abs(float(old) - float(new))
        scale = max(abs(float(old)), abs(float(new)), 1.0)
        result["max_abs_difference"] = max(
            result["max_abs_difference"], difference
        )
        result["max_relative_difference"] = max(
            result["max_relative_difference"], difference / scale
        )
        if difference > 1.0e-9 * scale:
            result["missing_or_changed"].append({
                "path": path, "reason": "numeric difference",
                "old": old, "new": new, "abs_difference": difference,
            })
        return
    if old != new:
        result["missing_or_changed"].append({
            "path": path, "reason": "value difference", "old": old, "new": new,
        })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--old",
        default=(
            "outputs/fullframes/time_input_experiment/gate_pair/"
            "evaluation/metrics.json"
        ),
    )
    parser.add_argument(
        "--new",
        default=(
            "outputs/fullframes/time_input_experiment/gate_pair/"
            "evaluation_completeness_fix/metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/fullframes/time_input_experiment/gate_pair/"
            "failure_semantics/metrics_comparison.json"
        ),
    )
    args = parser.parse_args(argv)
    old_path = Path(args.old)
    new_path = Path(args.new)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    result = {
        "old_path": str(old_path),
        "old_sha256": _sha256(old_path),
        "new_path": str(new_path),
        "new_sha256": _sha256(new_path),
        "max_abs_difference": 0.0,
        "max_relative_difference": 0.0,
        "missing_or_changed": [],
        "old_values_preserved": False,
    }
    _compare(old, new, "", result)
    result["old_values_preserved"] = not result["missing_or_changed"]
    result["exit_code"] = 0 if result["old_values_preserved"] else 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "old_values_preserved": result["old_values_preserved"],
        "exit_code": result["exit_code"],
        "n_differences": len(result["missing_or_changed"]),
        "max_abs_difference": result["max_abs_difference"],
        "max_relative_difference": result["max_relative_difference"],
    }, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
