"""Compare two prediction sets on the observations valid for both.

Two runs can score different observation sets because the predicted contact
radius changes the zeta mapping.  This script intersects the per-field valid
masks of both runs and reports pooled RMSE on the shared points, so an
accuracy change cannot be produced by the scored set changing.

Usage:
    python scripts/compare_common_points.py --config <cfg> \
        --pred-dir-a <dir> --pred-dir-b <dir> --label-a baseline \
        --label-b tref --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _case_specs, _make_case, _section

from scripts.diagnose_radius_split import _evaluate_case, _load_prediction

FIELDS = ("rho", "u", "p")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir-a", required=True)
    parser.add_argument("--pred-dir-b", required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--tag", default="best")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)
    rows: dict[str, list[dict]] = {"train": [], "validation": []}

    for spec in _case_specs(section):
        case = _make_case(section, spec, config, allow_synthetic=False)
        case_id = case.case_id
        pred_a = _load_prediction(
            Path(args.pred_dir_a) / f"pred_{args.tag}_{case_id}.npz"
        )
        pred_b = _load_prediction(
            Path(args.pred_dir_b) / f"pred_{args.tag}_{case_id}.npz"
        )
        eval_a = _evaluate_case(case, pred_a)
        eval_b = _evaluate_case(case, pred_b)
        row = {"case_id": case_id, "role": case.role, "fields": {}}
        for name in FIELDS:
            fa = eval_a["fields"][name]
            fb = eval_b["fields"][name]
            common = fa["use"] & fb["use"]
            entry = {"n_common": int(np.count_nonzero(common))}
            for label, field in ((args.label_a, fa), (args.label_b, fb)):
                if common.any():
                    error = (field["predicted"][common] - field["target"][common]) / field["scale"]
                    entry[f"{label}_rmse"] = float(np.sqrt(np.mean(error * error)))
                else:
                    entry[f"{label}_rmse"] = None
            row["fields"][name] = entry
        rows[case.role].append(row)

    summary = {}
    for role, group in rows.items():
        entry = {}
        for name in FIELDS:
            n = sum(row["fields"][name]["n_common"] for row in group)
            item = {"n_common": n}
            for label in (args.label_a, args.label_b):
                if n:
                    item[f"{label}_rmse"] = float(np.sqrt(sum(
                        row["fields"][name][f"{label}_rmse"] ** 2
                        * row["fields"][name]["n_common"]
                        for row in group if row["fields"][name]["n_common"]
                    ) / n))
                else:
                    item[f"{label}_rmse"] = None
            entry[name] = item
        summary[role] = entry

    payload = {
        "config": str(args.config),
        "pred_dir_a": str(args.pred_dir_a), "label_a": args.label_a,
        "pred_dir_b": str(args.pred_dir_b), "label_b": args.label_b,
        "tag": args.tag,
        "summary": summary,
        "cases": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(value):
        return "  n/a" if value is None else f"{value:.3f}"

    print(f"common points ({args.tag}):")
    for role, entry in summary.items():
        print(
            f"{role:10s} n={entry['rho']['n_common']:>8d}  "
            f"rho {fmt(entry['rho'][f'{args.label_a}_rmse'])}/{fmt(entry['rho'][f'{args.label_b}_rmse'])}  "
            f"u {fmt(entry['u'][f'{args.label_a}_rmse'])}/{fmt(entry['u'][f'{args.label_b}_rmse'])}  "
            f"p {fmt(entry['p'][f'{args.label_a}_rmse'])}/{fmt(entry['p'][f'{args.label_b}_rmse'])}"
        )
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
