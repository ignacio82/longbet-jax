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

"""Independent finite-tree oracles for the whole-tree prior proposal.

The oracle uses ordinary Python recursion over categorical ranges, without
calling bartz rule helpers or the sampler's prior-density implementation.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from longbet._refresh_trees import draw_prior_tree, tree_support
from longbet._shared_forest import enable_x64


@pytest.fixture(autouse=True)
def _matrix_precision_context():
    with enable_x64(True):
        yield


def _enumerate_prior(max_split, pnt, *, blocked=(), variable_weights=None):
    """Enumerate the generative prior before any design-count restriction."""
    half = len(pnt) // 2
    weights = (np.ones(len(max_split)) if variable_weights is None
               else np.asarray(variable_weights, float))
    blocked = set(blocked)

    def descend(node, ranges):
        eligible = [v for v, (lo, hi) in enumerate(ranges)
                    if lo < hi and v not in blocked]
        split_probability = float(pnt[node]) if eligible and node < half else 0.
        answers = [({}, 1. - split_probability)]
        if not split_probability:
            return answers
        total_variable_weight = sum(weights[v] for v in eligible)
        for variable in eligible:
            lo, hi = ranges[variable]
            rule_probability = (split_probability * weights[variable]
                                / total_variable_weight / (hi - lo))
            for cut in range(lo, hi):
                left, right = list(ranges), list(ranges)
                left[variable] = (lo, cut)
                right[variable] = (cut + 1, hi)
                for (lv, lp), (rv, rp) in itertools.product(
                        descend(2 * node, tuple(left)),
                        descend(2 * node + 1, tuple(right))):
                    answers.append(({node: (variable, cut)} | lv | rv,
                                    rule_probability * lp * rp))
        return answers

    output = []
    for rules, probability in descend(1, tuple((1, m + 1) for m in max_split)):
        var, split = np.zeros(half, np.uint8), np.zeros(half, np.uint8)
        for node, (variable, cut) in rules.items():
            var[node], split[node] = variable, cut
        output.append((var, split, probability))
    np.testing.assert_allclose(sum(row[2] for row in output), 1., atol=2e-15)
    return output


def _probabilities(depth=4):
    nodes = np.arange(2**depth)
    pnt = np.zeros(nodes.size, np.float32)
    for node in nodes[1:2**(depth - 1)]:
        pnt[node] = .65 / (1 + int(node).bit_length() - 1)
    return pnt


def _draws(max_split, pnt, *, blocked=(), weights=None, count=30000, seed=71):
    template = jnp.zeros(len(pnt) // 2, jnp.uint8)
    args = (template, template, jnp.asarray(max_split, jnp.uint8),
            None if not blocked else jnp.asarray(blocked, jnp.uint8),
            None if weights is None else jnp.log(jnp.asarray(weights, jnp.float32)),
            jnp.asarray(pnt))
    result = jax.jit(jax.vmap(lambda key: draw_prior_tree(key, *args)))(
        jax.random.split(jax.random.key(seed), count))
    return tuple(np.asarray(value) for value in result)


@pytest.mark.parametrize("max_split,blocked,weights", [
    ([1, 1], (), None),
    ([1, 1, 1], (2,), [.8, .2, 40.]),
    ([3], (), None),
    ([1], (0,), None),
])
def test_prior_proposal_matches_independent_enumeration(max_split, blocked, weights):
    """Ancestor ranges, variable renormalization and forced terminals are exact.

    Depth four leaves room below all feasible geometric partitions, so a
    positive nominal p_nonterminal must become zero when rules are exhausted.
    """
    pnt = _probabilities()
    oracle = _enumerate_prior(max_split, pnt, blocked=blocked,
                              variable_weights=weights)
    vs, ss = _draws(max_split, pnt, blocked=blocked, weights=weights)
    frequencies = []
    classified = np.zeros(len(vs), bool)
    for v, s, probability in oracle:
        match = np.all(ss == s, axis=1) & np.all((vs == v) | (ss == 0), axis=1)
        classified |= match
        frequencies.append(match.mean())
        # These are independent proposals, so the binomial MCSE is appropriate.
        mcse = np.sqrt(probability * (1 - probability) / len(vs))
        assert abs(match.mean() - probability) <= 5 * mcse + 1 / len(vs)
    assert classified.all()
    assert len(frequencies) == (1 if len(blocked) == len(max_split)
                                else 15 if max_split == [3] else 9)
    if len(frequencies) > 1:
        # The proposal genuinely changes shape, including stump and full trees.
        assert set(np.count_nonzero(ss, axis=1)) == {0, 1, 2, 3}


def _independent_support(var, split, X, min_leaf, min_decision):
    """Traverse every row and count every ancestor; no sampler helper reuse."""
    counts = np.zeros(2 * len(split), int)
    ids = []
    for row in X.T:
        node = 1
        while True:
            counts[node] += 1
            if node >= len(split) or split[node] == 0:
                ids.append(node)
                break
            node = 2 * node + int(row[var[node]] >= split[node])
    actual_leaves = set(ids)
    # Include empty leaves, if any, by walking the rule graph independently.
    pending = [1]
    leaves = []
    while pending:
        node = pending.pop()
        if node >= len(split) or split[node] == 0:
            leaves.append(node)
        else:
            pending.extend((2 * node, 2 * node + 1))
    assert actual_leaves <= set(leaves)
    valid = all(counts[node] >= min_leaf for node in leaves)
    valid &= all(counts[node] >= min_decision
                 for node in range(len(split)) if split[node])
    leaf_counts = np.bincount(ids, minlength=2 * len(split))
    return valid, np.asarray(ids), leaf_counts


@pytest.mark.parametrize("min_leaf,min_decision,expected_valid", [
    (2, 0, 5),
    (0, 6, 3),
    (2, 6, 3),
])
def test_count_restrictions_reject_whole_candidates_without_prior_renormalization(
        min_leaf, min_decision, expected_valid):
    # Cell (0,0) has only one design row. Thus the left child cannot split
    # under min_leaf=2; min_decision=6 also forbids splitting the right child.
    X = np.repeat(np.array([[0, 0, 1, 1], [0, 1, 0, 1]], np.uint8),
                  [1, 2, 2, 3], axis=1)
    pnt = _probabilities()
    oracle = _enumerate_prior([1, 1], pnt, variable_weights=[.8, .2])
    support = []
    for v, s, _ in oracle:
        expected = _independent_support(v, s, X, min_leaf, min_decision)
        actual = tree_support(jnp.asarray(v), jnp.asarray(s), jnp.asarray(X),
                              min_leaf, min_decision)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        support.append(expected[0])
    assert sum(support) == expected_valid
    expected_valid_mass = sum(row[2] for row, valid in zip(oracle, support) if valid)
    assert .5 < expected_valid_mass < .9

    vs, ss = _draws([1, 1], pnt, weights=[.8, .2], seed=921)
    valid = np.asarray(jax.jit(jax.vmap(lambda v, s: tree_support(
        v, s, jnp.asarray(X), min_leaf, min_decision)[0]))(vs, ss))
    mcse = np.sqrt(expected_valid_mass * (1 - expected_valid_mass) / len(valid))
    assert abs(valid.mean() - expected_valid_mass) <= 5 * mcse
    assert (~valid).any()  # Invalid candidates are returned for whole-move rejection.
    for (v, s, probability), supported in zip(oracle, support):
        if not supported:
            continue
        match = np.all(ss == s, axis=1) & np.all((vs == v) | (ss == 0), axis=1)
        mcse = np.sqrt(probability * (1 - probability) / len(vs))
        assert abs(np.mean(match & valid) - probability) <= 5 * mcse + 1 / len(vs)
        # The unnormalized proposal mass remains p(tree), not p(tree|valid).
        if not np.any(s):
            assert abs(match.mean() - probability / expected_valid_mass) > .05
