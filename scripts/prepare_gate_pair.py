#!/usr/bin/env python3
"""Preflight for the paired C (none) / D (similarity_time) gate experiment.

Read-only: builds fresh trainers, records code/config/data identity,
initial-state equality, first sampling batch, validation reference batches and
the runtime environment.  It does not train or write a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402


C_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_none_pair.yaml"
)
D_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_gate_pair.yaml"
)
C_OUTPUT = "outputs/contact_air_time_input_pair_c_none_4000"
D_OUTPUT = "outputs/contact_air_time_input_pair_d_gate_4000"
OUTPUT = "outputs/fullframes/time_input_experiment/gate_pair/preflight.json"
CLONE = Path("F:/MyNewProject/MyNewProject")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable_config(value):
    if isinstance(value, dict):
        return {str(key): _jsonable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _hash_state_dict(module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_batches(batches) -> str:
    digest = hashlib.sha256()
    for case, batch in batches:
        digest.update(case.case_id.encode("utf-8"))
        for name in (
            "t_pde", "zeta_pde", "t_ic", "zeta_ic", "t_right", "t_data",
            "radius_data", "rho_data", "u_data", "p_data", "valid_rho",
            "valid_u", "valid_p", "t_contact", "low_contact", "up_contact",
        ):
            value = getattr(batch, name)
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            digest.update(name.encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_validation(trainer) -> str:
    digest = hashlib.sha256()
    for sample in trainer._validation:
        digest.update(sample.case_id.encode("utf-8"))
        digest.update(_hash_batches([(next(
            c for c in trainer.validation_cases if c.case_id == sample.case_id
        ), sample.batch)]).encode("ascii"))
    return digest.hexdigest()


def _signature(trainer) -> dict:
    return {
        "train": {
            case.case_id: trainer._case_signature(case)
            for case in trainer.cases
        },
        "validation": {
            case.case_id: trainer._case_signature(case)
            for case in trainer.validation_cases
        },
    }


def _strip_pair_differences(section: dict) -> dict:
    import copy

    clean = copy.deepcopy(section)
    clean.pop("output_dir", None)
    networks = clean.get("networks", {})
    networks.pop("r0_input_gate", None)
    return clean


def _git_head() -> dict:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(CLONE), "rev-parse", "HEAD"], text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(CLONE), "branch", "--show-current"], text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(CLONE), "status", "--short"], text=True,
        ).strip()
        return {"head": head, "branch": branch, "dirty": bool(status)}
    except Exception as exc:  # pragma: no cover - local environment
        return {"error": str(exc)}


def main() -> int:
    output = _REPO_ROOT / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    for directory in (C_OUTPUT, D_OUTPUT):
        path = _REPO_ROOT / directory
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"refusing to reuse non-empty output directory {directory}"
            )

    c_trainer, c_meta = diag._build_trainer(C_CONFIG)
    d_trainer, d_meta = diag._build_trainer(D_CONFIG)
    c_section = c_meta["section"]
    d_section = d_meta["section"]
    if _strip_pair_differences(c_section) != _strip_pair_differences(d_section):
        raise RuntimeError("C/D resolved configurations differ beyond output_dir/gate")

    c_state = _hash_state_dict(c_trainer.air_net)
    d_state = _hash_state_dict(d_trainer.air_net)
    if c_state != d_state:
        raise RuntimeError("C/D AirNet initial parameters differ")
    if _hash_state_dict(c_trainer.contact_net) != _hash_state_dict(d_trainer.contact_net):
        raise RuntimeError("C/D ContactNet initial parameters differ")
    if c_trainer.opt_air.state or d_trainer.opt_air.state:
        raise RuntimeError("C/D Adam was not fresh")

    c_first = c_trainer._batches()
    d_first = d_trainer._batches()
    c_first_hash = _hash_batches(c_first)
    d_first_hash = _hash_batches(d_first)
    if c_first_hash != d_first_hash:
        raise RuntimeError("C/D first sampling batch differs")

    c_sig = _signature(c_trainer)
    d_sig = _signature(d_trainer)
    if c_sig != d_sig:
        raise RuntimeError("C/D case signatures differ")
    c_val = _hash_validation(c_trainer)
    d_val = _hash_validation(d_trainer)
    if c_val != d_val:
        raise RuntimeError("C/D validation reference batches differ")

    payload = {
        "experiment": "r0_gate_pair_C_none_D_similarity_time",
        "code": _git_head(),
        "configs": {
            "C": {
                "path": C_CONFIG,
                "sha256": _sha256_file(_REPO_ROOT / C_CONFIG),
                "resolved": _jsonable_config(c_section),
                "output_dir": C_OUTPUT,
            },
            "D": {
                "path": D_CONFIG,
                "sha256": _sha256_file(_REPO_ROOT / D_CONFIG),
                "resolved": _jsonable_config(d_section),
                "output_dir": D_OUTPUT,
            },
            "only_differences": (
                "output_dir and contact_air.networks.r0_input_gate; "
                "all training factors compared equal"
            ),
        },
        "data_identity": {
            "case_signatures_equal": c_sig == d_sig,
            "case_signatures": c_sig,
            "validation_batches_equal": c_val == d_val,
            "validation_batch_hash": c_val,
        },
        "initialisation": {
            "air_state_equal": c_state == d_state,
            "air_state_sha256": c_state,
            "contact_state_equal": True,
            "contact_state_sha256": _hash_state_dict(c_trainer.contact_net),
            "fresh_adam": True,
        },
        "first_sampling_batch": {
            "equal": c_first_hash == d_first_hash,
            "C_sha256": c_first_hash,
            "D_sha256": d_first_hash,
        },
        "training_protocol": {
            "from_scratch": True,
            "resume": None,
            "loops": int(c_meta["section"]["training"]["loops"]),
            "seed": int(c_meta["section"]["training"]["seed"]),
            "learning_rate": float(c_meta["section"]["training"]["learning_rate"]),
            "contact_data_weight": float(
                c_meta["section"]["training"]["contact_data_weight"]
            ),
            "n_schedule": int(c_meta["section"]["training"]["n_schedule"]),
            "sampling": dict(c_meta["section"]["sampling"]),
            "time_input": "similarity",
            "only_changed_factor": "AirNet r0_input_gate",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cwd": str(_REPO_ROOT),
        },
        "no_training_performed": True,
        "notes": [
            "C and D are fresh from-scratch runs; existing B is not resumed.",
            "The first sampling batch and validation reference batches are equal.",
            "Only the AirNet gate differs; output_dir is the only config-path difference.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256_file(output),
        "code": payload["code"],
        "initialisation": payload["initialisation"],
        "first_batch_equal": payload["first_sampling_batch"]["equal"],
        "validation_batches_equal": payload["data_identity"]["validation_batches_equal"],
        "loops": payload["training_protocol"]["loops"],
        "seed": payload["training_protocol"]["seed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
