#!/usr/bin/env python3
"""Record an immutable description of a training baseline.

Read-only: hashes the source files, config, standards, caches and checkpoints,
and records the interpreter/GPU environment.  It never writes to a cache,
checkpoint or training output; the only file it writes is the requested JSON.

Example:

    PYTHONPATH=src python scripts/record_baseline.py \
        --config configs/contact_air/split_train_r50_500_fullframes.yaml \
        --checkpoint-dir outputs/contact_air_split_r50_500_fullframes_cw3_4000 \
        --output outputs/fullframes/baseline_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from spherical_blast_pinn.config import load_config  # noqa: E402
from spherical_blast_pinn.contact_air.cli import _section  # noqa: E402
from spherical_blast_pinn.contact_air.one_d import (  # noqa: E402
    npz_content_digest,
    standard_source_digest,
)

_SOURCE_PATHS = (
    "src/spherical_blast_pinn/contact_air",
    "scripts",
    "configs",
    "tests",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        path = _ROOT / relative
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    hashes[str(child.relative_to(_ROOT)).replace("\\", "/")] = (
                        _sha256(child)
                    )
        elif path.is_file():
            hashes[relative] = _sha256(path)
    return hashes


def _git(path: Path) -> dict:
    if not path.is_dir():
        return {"path": str(path), "available": False}
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                capture_output=True, text=True, check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        "path": str(path),
        "available": True,
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--upload-repo", default=str(Path("F:/MyNewProject/MyNewProject")),
        help="local clone whose commit/status is recorded",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    section = _section(config)
    checkpoint_dir = Path(args.checkpoint_dir)
    cases = []
    for spec in list(section.get("cases", []) or []):
        cache = Path(str(spec["cache"]))
        standard = None
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            source = metadata.get("case", {}).get("source", {})
            standard = source.get("source_path")
        entry = {
            "case_id": str(spec["case_id"]),
            "role": str(spec.get("role", "train")),
            "charge_radius_m": float(spec["charge_radius_m"]),
            "cache": str(cache),
            "cache_exists": cache.is_file(),
            "cache_bytes": cache.stat().st_size if cache.is_file() else None,
            "cache_digest": npz_content_digest(cache) if cache.is_file() else None,
            "standard": standard,
        }
        if standard and Path(standard).is_file():
            entry["standard_digest"] = standard_source_digest(standard)
        cases.append(entry)

    checkpoints = {}
    for tag in ("best", "last"):
        path = checkpoint_dir / f"{tag}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            checkpoints[tag] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "loop": payload.get("loop"),
                "config": payload.get("config"),
                "contact_net": payload.get("contact_net_describe"),
                "air_net": payload.get("air_net_describe"),
                "case_ids": payload.get("case_ids"),
                "validation_case_ids": payload.get("validation_case_ids"),
            }

    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "numpy": np.__version__,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "pid": os.getpid(),
    }

    payload = {
        "format": "spherical_blast_pinn.baseline_manifest",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_name": "fullframes_cw3_4000",
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "resolved": config.values,
            "overrides": dict(config.overrides),
            "raw": yaml.safe_load(config_path.read_text(encoding="utf-8")),
        },
        "training_command": (
            "PYTHONPATH=src python -m spherical_blast_pinn.contact_air.cli train "
            f"--config {config_path} --cases all --max-loops 4000 --no-synthetic"
        ),
        "working_tree": str(_ROOT),
        "source_sha256": _source_hashes(),
        "upload_repo": _git(Path(args.upload_repo)),
        "cases": cases,
        "checkpoints": checkpoints,
        "environment": environment,
        "note": (
            "Read-only baseline record.  This run is the fixed comparison point; "
            "new experiments must use a separate output directory and must not "
            "overwrite these caches or checkpoints."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"baseline manifest: {out}")
    print(
        f"cases={len(cases)} checkpoints={sorted(checkpoints)} "
        f"source_files={len(payload['source_sha256'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
