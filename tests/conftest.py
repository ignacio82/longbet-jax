# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
