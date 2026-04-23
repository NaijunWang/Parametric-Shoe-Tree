from __future__ import annotations

from pathlib import Path

import numpy as np

from last_generator.align import align_to_canonical
from last_generator.io import load_scan


def test_align_to_canonical_normalizes_origin_and_axes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scan_path = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    aligned = align_to_canonical(load_scan(scan_path))
    assert np.isclose(aligned.bounds[0][1], 0.0)
    assert np.isclose(aligned.bounds[0][2], 0.0)
    assert np.isclose((aligned.bounds[0][0] + aligned.bounds[1][0]) / 2.0, 0.0)
    assert int(aligned.extents.argmax()) == 1
