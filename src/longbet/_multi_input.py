"""Input normalization, validation, ordering, and missingness checks for LongBetMulti."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as stats

from longbet._config import LongBetConfig


@dataclass(frozen=True)
class NormalizedMultiInput:
    """Normalized and validated inputs for a multi-outcome fit."""

    # Raw arrays in user order
    y_user: list[np.ndarray]
    outcome_names: tuple[str, ...]
    user_outcomes: tuple[str, ...]

    # Permutations between user order and internal sampling order
    order: tuple[int, ...]
    inverse_order: tuple[int, ...]

    # Internal order quantities (binary first, then continuous)
    internal_names: tuple[str, ...]
    internal_outcomes: tuple[str, ...]
    obs_masks: list[np.ndarray]
    y_prepared: list[np.ndarray]
    meany: tuple[float, ...]
    sdy: tuple[float, ...]
    offset_: tuple[float, ...]

    M: int
    N: int
    T: int


def normalize_multi_inputs(
    y: Any,
    *,
    outcome: str | Sequence[str] | Mapping[str, str] | None,
    outcome_names: Sequence[str] | None,
    config: LongBetConfig,
) -> NormalizedMultiInput:
    """Validate and normalize multi-outcome responses.

    Parameters
    ----------
    y
        Mapping of outcome names to (N, T) arrays, sequence of (N, T) arrays,
        or 3-D numeric array of shape (N, T, M).
    outcome
        Type per outcome: 'continuous' or 'binary'. Broadcasts from config when
        None.
    outcome_names
        Optional explicit names for the outcomes.
    config
        LongBet configuration.

    Returns
    -------
    NormalizedMultiInput
    """
    # ------------------------------------------------------------------
    # 1. Parse container into user-ordered list of (N, T) arrays and names
    # ------------------------------------------------------------------
    container_names: list[str] | None = None
    y_list: list[np.ndarray]

    if isinstance(y, Mapping):
        container_names = list(y.keys())
        for name in container_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Outcome names in mapping must be non-empty strings, got {name!r}"
                )
        if len(container_names) != len(set(container_names)):
            raise ValueError("Duplicate outcome names in mapping.")
        y_list = [np.asarray(y[k], dtype=np.float64) for k in container_names]
    elif isinstance(y, (list, tuple)):
        y_list = [np.asarray(elem, dtype=np.float64) for elem in y]
    elif isinstance(y, np.ndarray):
        if y.ndim != 3:
            raise ValueError(
                f"Array input y for LongBetMulti must be 3-D with shape (N, T, M), "
                f"got ndim={y.ndim} with shape {y.shape}. Bare 2-D panels are invalid; "
                f"use LongBet for single-outcome models."
            )
        # Extract slices with drop=False semantics
        M_dim = y.shape[2]
        y_list = [np.asarray(y[:, :, m], dtype=np.float64) for m in range(M_dim)]
    else:
        raise TypeError(
            f"Unsupported type for y: {type(y)}. Expected mapping, sequence of "
            f"arrays, or 3-D numeric array (N, T, M)."
        )

    M = len(y_list)
    if M < 2:
        raise ValueError(
            f"LongBetMulti requires at least 2 outcomes, got {M}."
        )

    # Validate individual arrays
    for idx, arr in enumerate(y_list):
        if arr.ndim != 2:
            raise ValueError(
                f"Each outcome must be a 2-D matrix of shape (N, T), but outcome "
                f"at index {idx} has shape {arr.shape}."
            )

    N, T = y_list[0].shape
    if N < 1 or T < 1:
        raise ValueError(f"Dimensions N and T must be >= 1, got N={N}, T={T}.")

    for idx, arr in enumerate(y_list):
        if arr.shape != (N, T):
            raise ValueError(
                f"All outcomes must have matching panel dimensions (N, T)=({N}, {T}), "
                f"but outcome at index {idx} has shape {arr.shape}."
            )

    # ------------------------------------------------------------------
    # 2. Resolve outcome names
    # ------------------------------------------------------------------
    final_names: list[str]
    if outcome_names is not None:
        if isinstance(outcome_names, str):
            raise ValueError("outcome_names must be a sequence of non-empty strings, not one string.")
        if len(outcome_names) != M:
            raise ValueError(
                f"Length of outcome_names ({len(outcome_names)}) does not match "
                f"number of outcomes ({M})."
            )
        final_names = list(outcome_names)
        for name in final_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Outcome names must be non-empty strings.")
        if len(final_names) != len(set(final_names)):
            raise ValueError("Outcome names must be unique.")
        if container_names is not None and list(final_names) != list(container_names):
            raise ValueError(
                f"Explicit outcome_names {final_names} conflict with mapping keys "
                f"{container_names}."
            )
    elif container_names is not None:
        final_names = list(container_names)
    else:
        final_names = [f"outcome_{i + 1}" for i in range(M)]

    outcome_names_tuple = tuple(final_names)

    # ------------------------------------------------------------------
    # 3. Resolve outcome types in user order
    # ------------------------------------------------------------------
    user_outcomes: list[str]
    if outcome is None:
        user_outcomes = [config.outcome] * M
    elif isinstance(outcome, str):
        if outcome not in ("continuous", "binary"):
            raise ValueError(
                f"outcome must be 'continuous' or 'binary', got {outcome!r}"
            )
        user_outcomes = [outcome] * M
    elif isinstance(outcome, Mapping):
        expected_keys = set(final_names)
        actual_keys = set(outcome.keys())
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            msg_parts = []
            if missing:
                msg_parts.append(f"missing {sorted(missing)}")
            if extra:
                msg_parts.append(f"unexpected {sorted(extra)}")
            raise ValueError(
                f"Outcome type mapping keys must exactly match outcome names: {'; '.join(msg_parts)}"
            )
        user_outcomes = [outcome[name] for name in final_names]
        for t, name in zip(user_outcomes, final_names):
            if t not in ("continuous", "binary"):
                raise ValueError(
                    f"outcome for {name!r} must be 'continuous' or 'binary', got {t!r}"
                )
    elif isinstance(outcome, (Sequence, np.ndarray)):
        if len(outcome) != M:
            raise ValueError(
                f"Length of outcome sequence ({len(outcome)}) does not match "
                f"number of outcomes ({M})."
            )
        user_outcomes = [str(t) for t in outcome]
        for t, name in zip(user_outcomes, final_names):
            if t not in ("continuous", "binary"):
                raise ValueError(
                    f"outcome for {name!r} must be 'continuous' or 'binary', got {t!r}"
                )
    else:
        raise TypeError(f"Unsupported type for outcome: {type(outcome)}")

    user_outcomes_tuple = tuple(user_outcomes)

    # ------------------------------------------------------------------
    # 4. Binary-first stable partition
    # ------------------------------------------------------------------
    binary_indices = [i for i, t in enumerate(user_outcomes) if t == "binary"]
    continuous_indices = [i for i, t in enumerate(user_outcomes) if t == "continuous"]
    order = tuple(binary_indices + continuous_indices)
    inverse_order = tuple(int(idx) for idx in np.argsort(order))

    internal_names = tuple(final_names[i] for i in order)
    internal_outcomes = tuple(user_outcomes[i] for i in order)

    # ------------------------------------------------------------------
    # 5. Outcome preprocessing in internal order
    # ------------------------------------------------------------------
    obs_masks: list[np.ndarray] = []
    y_prepared: list[np.ndarray] = []
    meany_list: list[float] = []
    sdy_list: list[float] = []
    offset_list: list[float] = []

    for m in range(M):
        u_idx = order[m]
        name = final_names[u_idx]
        otype = user_outcomes[u_idx]
        arr = y_list[u_idx]

        if np.isneginf(arr).any() or np.isposinf(arr).any():
            raise ValueError(f"Outcome {name!r} contains infinite values.")

        mask = np.isfinite(arr)
        if not np.any(mask):
            raise ValueError(f"Outcome {name!r} has no observed values.")

        obs_vals = arr[mask]

        if otype == "continuous":
            if config.standardize:
                sdy = float(np.std(obs_vals, ddof=0))
                if sdy == 0.0 or not np.isfinite(sdy):
                    raise ValueError(
                        f"Outcome {name!r} has zero or non-finite standard deviation."
                    )
                meany = float(np.mean(obs_vals))
                offset_val = 0.0
                y_std = np.where(mask, (arr - meany) / sdy, 0.0).astype(np.float32)
            else:
                meany = 0.0
                sdy = 1.0
                offset_val = 0.0
                y_std = np.where(mask, arr, 0.0).astype(np.float32)
        else:  # binary
            # Observed labels must be 0 or 1
            unique_vals = np.unique(obs_vals)
            if not np.all(np.isin(unique_vals, [0, 1])):
                raise ValueError(
                    f"Binary outcome {name!r} contains invalid labels: "
                    f"{unique_vals.tolist()}. Only 0 and 1 are accepted."
                )
            rate = float(np.mean(obs_vals))
            rate_clipped = float(np.clip(rate, 1e-4, 1.0 - 1e-4))
            offset_val = float(stats.norm.ppf(rate_clipped))
            meany = 0.0
            sdy = 1.0
            y_std = np.where(mask, arr, 0.0).astype(np.float32)

        obs_masks.append(mask)
        y_prepared.append(y_std)
        meany_list.append(meany)
        sdy_list.append(sdy)
        offset_list.append(offset_val)

    # ------------------------------------------------------------------
    # 6. Missingness policy checks (Section 4.4)
    # ------------------------------------------------------------------
    if config.sur_active:
        for m in range(M):
            if internal_outcomes[m] == "continuous" and m > 0:
                required_mask = np.logical_and.reduce(obs_masks[:m], axis=0)
                missing_preds = obs_masks[m] & (~required_mask)
                if np.any(missing_preds):
                    raise ValueError(
                        f"Observed continuous outcome {internal_names[m]!r} has missing "
                        f"predecessor in cells where it is observed. Triangular SUR requires "
                        f"all predecessor outcomes to be observed in cells where a downstream "
                        f"continuous outcome is observed."
                    )

    return NormalizedMultiInput(
        y_user=y_list,
        outcome_names=outcome_names_tuple,
        user_outcomes=user_outcomes_tuple,
        order=order,
        inverse_order=inverse_order,
        internal_names=internal_names,
        internal_outcomes=internal_outcomes,
        obs_masks=obs_masks,
        y_prepared=y_prepared,
        meany=tuple(meany_list),
        sdy=tuple(sdy_list),
        offset_=tuple(offset_list),
        M=M,
        N=N,
        T=T,
    )
