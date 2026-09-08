"""Model serialization to a self-contained ``.npz`` archive.

The archive holds plain arrays plus two JSON blobs (the config and the
metadata, which carries the design spec).  It contains no pickled objects, so it
loads with ``allow_pickle=False`` and can be embedded in an R object as a raw
vector and round-tripped through ``saveRDS`` without a live Python handle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from bartz.mcmcloop._trace import MainTrace

from longbet._config import LongBetConfig
from longbet._loop import LongBetTrace

_PARAM_KEYS = ("beta", "gamma", "b0", "b1", "alpha", "sigma2", "sigma_gamma2")
_FOREST_KEYS = ("leaf_tree", "split_tree", "var_tree", "offset", "leaf_unit", "error_cov_inv")

# Both scalar and coupled fits used stale bartz leaf precision sums before this
# contract was introduced. Loading trees cannot repair their posterior draws.
PRECISION_CACHE_VERSION = 1


def _require_current_precision_cache(meta: dict[str, Any]) -> None:
    if meta.get("precision_cache_version") != PRECISION_CACHE_VERSION:
        raise ValueError(
            "Archive predates the dynamic leaf-precision cache repair or uses "
            "an unsupported sampler version. Its posterior draws cannot be "
            "upgraded by loading them. Refit from the original data."
        )


def save_npz(
    path: str | Path,
    trace: LongBetTrace,
    config: LongBetConfig,
    meany: float,
    sdy: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write posterior traces, configuration and design spec to ``path``.

    Parameters
    ----------
    path
        Output path; the ``.npz`` suffix is added by NumPy if absent.
    trace
        Posterior trace from ``run_longbet_mcmc``.
    config
        The configuration the model was fitted with.
    meany, sdy
        Outcome centring and scaling.
    metadata
        Extra JSON-serializable metadata; the design spec travels here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    meta["precision_cache_version"] = PRECISION_CACHE_VERSION
    meta["meany"] = float(meany)
    meta["sdy"] = float(sdy)
    meta["has_chains"] = bool(np.asarray(trace.beta).ndim > 2)

    arrays: dict[str, Any] = {k: np.asarray(getattr(trace, k)) for k in _PARAM_KEYS}
    for prefix, sub in (("mu", trace.mu_trace), ("nu", trace.nu_trace)):
        for key in _FOREST_KEYS:
            arrays[f"{prefix}_{key}"] = np.asarray(getattr(sub, key))

    arrays["_config_json"] = json.dumps(config.to_dict())
    arrays["_meta_json"] = json.dumps(meta)

    np.savez_compressed(path, **arrays)


def _rebuild_trace(data: Any, prefix: str, has_chains: bool) -> MainTrace:
    """Rebuild one forest's ``MainTrace`` from the archive."""
    leaf_tree = jnp.asarray(data[f"{prefix}_leaf_tree"])
    shape = (leaf_tree.shape[0], leaf_tree.shape[1]) if has_chains else (leaf_tree.shape[0],)
    zeros = jnp.zeros(shape, dtype=jnp.int32)
    return MainTrace(
        has_chains=has_chains,
        mesh=None,
        leaf_tree=leaf_tree,
        split_tree=jnp.asarray(data[f"{prefix}_split_tree"]),
        var_tree=jnp.asarray(data[f"{prefix}_var_tree"]),
        offset=jnp.asarray(data[f"{prefix}_offset"]),
        leaf_unit=jnp.asarray(data[f"{prefix}_leaf_unit"]),
        error_cov_inv=jnp.asarray(data[f"{prefix}_error_cov_inv"]),
        theta=None,
        grow_prop_count=zeros,
        grow_acc_count=zeros,
        prune_prop_count=zeros,
        prune_acc_count=zeros,
        log_likelihood=None,
        log_trans_prior=None,
        varprob=None,
    )


def load_npz(
    path: str | Path,
) -> tuple[LongBetTrace, LongBetConfig, float, float, dict[str, Any]]:
    """Read an archive written by :func:`save_npz`.

    Returns
    -------
    trace, config, meany, sdy, metadata
    """
    data = np.load(Path(path), allow_pickle=False)

    config = LongBetConfig.from_dict(json.loads(str(data["_config_json"])))
    meta = json.loads(str(data["_meta_json"]))
    if meta.get("model_kind") == "multi":
        raise ValueError(
            f"Archive at {path} contains a multi-outcome LongBet model (model_kind='multi'). "
            "Use LongBetMulti.load() or load_multi_npz() to load this model."
        )
    _require_current_precision_cache(meta)
    origin = meta.get("multi_origin")
    if origin is not None:
        # Extracting one marginal must not bypass validation of its joint target.
        # Older child archives omit the parent's outcome types, so require a
        # proper prior unless independence is explicitly recorded in config.
        config.validate_multi_variance_prior(
            tuple(origin.get("outcomes", ("continuous", "continuous"))))
    meany = float(meta.pop("meany", 0.0))
    sdy = float(meta.pop("sdy", 1.0))
    has_chains = bool(meta.pop("has_chains", np.asarray(data["beta"]).ndim > 2))

    trace = LongBetTrace(
        mu_trace=_rebuild_trace(data, "mu", has_chains),
        nu_trace=_rebuild_trace(data, "nu", has_chains),
        **{k: jnp.asarray(data[k]) for k in _PARAM_KEYS},
    )
    return trace, config, meany, sdy, meta
