"""Legacy extracted-profile audit helpers.

These helpers read *already extracted* ``raw_line.npz`` directories through
:class:`~spherical_blast_pinn.data.extracted_case.ExtractedCase`.  They belong
to the legacy audit commands ``audit_data`` / ``validate_eos`` only.

They are deliberately **not** imported by the training entry.  Training reads
startup handoffs and :class:`~spherical_blast_pinn.data.startup_supervision.SupervisionDataset`
files instead; a ``data.cases`` block in a training configuration has no effect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..data.extracted_case import (
    DetectorSettings,
    ExtractedCase,
    FieldMapping,
    load_extracted_case
)

from ._common import SetupError


def case_specs(config: Config, *, split: str | None = None) -> list[dict[str, Any]]:
    section = config.section("data")
    specs = list(section.get("cases", []) or [])
    if split is not None:
        specs = [s for s in specs if str(s.get("split")) == split]
    return specs


def load_extracted_cases(
    config: Config,
    *,
    split: str | None = None,
    strict: bool | None = None,
) -> list[ExtractedCase]:
    """Read configured legacy extracted profiles.

    ``strict`` defaults to ``split == "train"``.  Missing training data is a
    hard failure; a missing validation/audit case is reported and skipped.
    """
    if strict is None:
        strict = split == "train"
    section = config.section("data")
    mapping = FieldMapping.from_mapping(section.get("field_map"))
    settings = DetectorSettings(
        **{
            key: value
            for key, value in (section.get("detector") or {}).items()
            if key in DetectorSettings.__dataclass_fields__
        }
    )
    cases: list[ExtractedCase] = []
    missing: list[str] = []
    for specification in case_specs(config, split=split):
        directory_value = specification.get("directory")
        if not directory_value:
            missing.append(f"{specification.get('case_id')} -> <no directory>")
            continue
        directory = Path(str(directory_value))
        if not (directory / "raw_line.npz").exists():
            missing.append(f"{specification.get('case_id')} -> {directory}")
            continue
        case = load_extracted_case(
            directory,
            case_id=specification.get("case_id"),
            field_mapping=mapping,
            settings=settings,
            gamma_air=float(config.get("air.gamma", 1.4)),
        )
        case.source["split"] = specification.get("split")
        expected = float(specification.get("charge_radius_m", case.charge_radius))
        if abs(expected - case.charge_radius) > 1.0e-9:
            raise SetupError(
                f"case {case.case_id!r}: configuration says R_0 = {expected} m but the "
                f"data declares {case.charge_radius} m"
            )
        cases.append(case)
    if missing and strict:
        raise SetupError(
            "these configured extracted profiles have no readable dataset:\n  "
            + "\n  ".join(missing)
            + "\nMissing data is reported, never silently substituted.  These "
            "commands audit legacy extracted profiles only; they are not the "
            "training entry."
        )
    if not cases:
        raise SetupError(
            f"no legacy extracted profile matched split={split!r}.  audit_data and "
            "validate_eos require an explicit data.cases block with extracted "
            "profile directories; training uses startup.cases instead."
        )
    if missing:
        setattr(cases[0], "_missing_cases", list(missing))
    return cases


__all__ = ["case_specs", "load_extracted_cases"]
