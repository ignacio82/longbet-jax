"""Shared fixtures. The panel factory itself lives in ``tests/_panels.py``."""

import pytest

from _panels import make_staggered_panel


@pytest.fixture
def staggered_panel():
    return make_staggered_panel()


@pytest.fixture
def unbalanced_panel():
    return make_staggered_panel(seed=123, missing_rate=0.12)


@pytest.fixture
def binary_panel():
    return make_staggered_panel(seed=7, binary=True)
