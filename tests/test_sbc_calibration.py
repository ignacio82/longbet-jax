import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.encouragement.sbc_calibration import (
    evaluate_sbc,
    generate_sbc_panel,
)


def test_generate_sbc_panel():
    y, d, z = generate_sbc_panel(n=40, t_len=4, instrument_effect=0.5, seed=123)
    assert y.shape == (40, 4)
    assert d.shape == (40, 4)
    assert z.shape == (40, 4)
    # Check absorbing property
    for i in range(40):
        assert np.all(np.diff(d[i]) >= 0)
        assert np.all(np.diff(z[i]) >= 0)


def test_sbc_calibration_evaluation():
    """Fast calibration check verifying Randomization AR maintains high coverage."""
    summaries = evaluate_sbc(replications=5, n=50, t_len=4, seed=42)
    assert len(summaries) == 3

    for s in summaries:
        assert s.ar_coverage >= 0.8  # Randomization AR maintains high coverage
        assert s.balke_pearl_coverage >= 0.8
        assert s.replications == 5
