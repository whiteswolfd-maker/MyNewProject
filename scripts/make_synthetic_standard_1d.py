"""Write a synthetic standard one-dimensional source.

This is an engineering fixture for the file -> build-cache -> train -> predict
chain.  It is clearly labelled ``synthetic`` and is never TNT validation data.
The generated file uses the documented intermediate format, not a guessed
LS-DYNA binary layout.

Usage (from the repository root)::

    python scripts/make_synthetic_standard_1d.py \
        --out outputs/contact_air_smoke/standard_1d.npz \
        --case tnt_50mm --radius 0.05 --frames 21 --cells 2400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_blast_pinn.contact_air.data import synthetic_radial_case  # noqa: E402
from spherical_blast_pinn.contact_air.one_d import write_standard_1d  # noqa: E402
from spherical_blast_pinn.contact_air.testing import default_precursor  # noqa: E402

_FIELDS = (
    "cell_id", "r_left", "r_right", "radius", "alpha_product", "alpha_air",
    "material", "density", "velocity", "pressure", "valid_density",
    "valid_velocity", "valid_pressure",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", default="tnt_50mm")
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--cells", type=int, default=2400)
    parser.add_argument("--duration", type=float, default=2.0e-5)
    args = parser.parse_args(argv)
    precursor = default_precursor(args.radius)
    radial = synthetic_radial_case(
        case_id=args.case, charge_radius_m=args.radius, precursor=precursor,
        n_frames=int(args.frames), n_radial=int(args.cells),
        duration_s=float(args.duration),
    )
    frames = [
        {field: getattr(frame, field) for field in _FIELDS}
        for frame in radial.frames
    ]
    write_standard_1d(
        args.out, case_id=args.case, charge_radius_m=args.radius,
        frames=frames, frame_times=[frame.time_raw_s for frame in radial.frames],
        position_type=radial.position_type,
        source={
            "kind": "synthetic_engineering_fixture",
            "synthetic": True,
            "note": "not TNT data; not a validated material card",
        },
    )
    print(f"written synthetic standard 1-D source: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
