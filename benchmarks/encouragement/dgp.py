"""Finite populations for longitudinal randomized-encouragement calibration.

Potential paths are generated before, and independently of, random assignment.
The returned truth is a finite-population assignment contrast, including in the
deliberately invalid-IV scenarios; it is never silently called a CACE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Scenario:
    name: str = "strong"
    complier_share: float = .4
    always_share: float = .15
    accelerated_share: float = 0.
    defier_share: float = 0.
    adoption_delay: int = 0
    effect: float = 1.
    effect_heterogeneity: float = .4
    unit_correlation: float = .5
    serial_correlation: float = .3
    heteroskedasticity: float = .2
    direct_encouragement_effect: float = 0.
    binary: bool = False

    def __post_init__(self):
        shares = (self.complier_share, self.always_share,
                  self.accelerated_share, self.defier_share)
        if any(not np.isfinite(p) or p < 0 for p in shares) or sum(shares) > 1:
            raise ValueError("Compliance shares must be nonnegative and sum to at most one.")
        if (not np.isfinite((self.unit_correlation, self.serial_correlation)).all()
                or abs(self.unit_correlation) >= 1 or abs(self.serial_correlation) >= 1):
            raise ValueError("Correlations must be strictly between -1 and one.")
        if not isinstance(self.adoption_delay, int) or self.adoption_delay < 0:
            raise ValueError("adoption_delay must be a nonnegative integer.")
        if not np.isfinite(self.heteroskedasticity) or self.heteroskedasticity < 0:
            raise ValueError("heteroskedasticity must be finite and nonnegative.")
        if not np.isfinite((self.effect, self.effect_heterogeneity,
                            self.direct_encouragement_effect)).all():
            raise ValueError("Effect parameters must be finite.")

    @property
    def assumption_failures(self) -> list[str]:
        return (["exclusion" ] if self.direct_encouragement_effect else []) + (
            ["monotonicity"] if self.defier_share else [])


SCENARIOS = {
    s.name: s for s in (
        Scenario(),
        Scenario(name="weak", complier_share=.05),
        Scenario(name="near_zero", complier_share=.005),
        Scenario(name="zero", complier_share=0.),
        Scenario(name="one_sided", always_share=0.),
        Scenario(name="delayed", adoption_delay=2),
        Scenario(name="acceleration", complier_share=.2, accelerated_share=.25),
        Scenario(name="negative_heterogeneous", effect=-1., effect_heterogeneity=.7),
        Scenario(name="correlated_serial_heteroskedastic", unit_correlation=.8,
                 serial_correlation=.8, heteroskedasticity=.8),
        Scenario(name="independent_units", unit_correlation=0.),
        Scenario(name="binary", binary=True),
        Scenario(name="binary_weak", binary=True, complier_share=.05),
        Scenario(name="exclusion_violation", direct_encouragement_effect=.3),
        Scenario(name="defiers", complier_share=.15, defier_share=.2),
    )
}


@dataclass
class Population:
    x: np.ndarray
    t: np.ndarray
    start: int
    d0: np.ndarray
    d1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    mean_y0: np.ndarray
    mean_y1: np.ndarray
    unit_effects: np.ndarray
    compliance_type: np.ndarray
    adoption_time0: np.ndarray
    adoption_time1: np.ndarray
    scenario: Scenario
    data_seed: int

    @property
    def truth(self) -> dict[str, np.ndarray]:
        dy = (self.y1 - self.y0).mean(axis=0)[self.start:]
        dd = (self.d1 - self.d0).mean(axis=0)[self.start:]
        ratio = np.divide(dy, dd, out=np.full_like(dy, np.nan), where=dd != 0)
        return {"itt_y": dy, "itt_d": dd, "wald": ratio}

    @property
    def conditional_mean_truth(self) -> dict[str, np.ndarray]:
        """Average over outcome innovations, holding unit effects/paths fixed."""
        dy = (self.mean_y1 - self.mean_y0).mean(axis=0)[self.start:]
        dd = self.truth["itt_d"]
        ratio = np.divide(dy, dd, out=np.full_like(dy, np.nan), where=dd != 0)
        return {"itt_y": dy, "itt_d": dd, "wald": ratio}

    def assign(self, assignment_seed: int) -> dict[str, np.ndarray]:
        """Complete randomization with floor(N/2) encouraged independent units."""
        n = len(self.x)
        assigned = np.zeros(n, dtype=bool)
        assigned[np.random.default_rng(assignment_seed).permutation(n)[:n // 2]] = True
        z = np.zeros_like(self.d0)
        z[assigned, self.start:] = 1
        return dict(x=self.x, t=self.t, assignment=assigned, z=z,
                    d=np.where(assigned[:, None], self.d1, self.d0),
                    y=np.where(assigned[:, None], self.y1, self.y0))

    def archive(self, assignment_seed: int) -> dict[str, np.ndarray]:
        """Portable NPZ data; metadata are separately serialized by runners."""
        return {**self.assign(assignment_seed), "d0": self.d0, "d1": self.d1,
                "y0": self.y0, "y1": self.y1,
                "mean_y0": self.mean_y0, "mean_y1": self.mean_y1,
                "unit_effects": self.unit_effects,
                "compliance_type": self.compliance_type,
                "adoption_time0": self.adoption_time0,
                "adoption_time1": self.adoption_time1,
                **{f"truth_{k}": v for k, v in self.truth.items()}}

    def covariance_truth(self) -> dict[str, np.ndarray]:
        """Exact randomization covariance and expected conservative estimate.

        Variables are ordered Y(h=1..H), then D(h=1..H). The finite-population
        covariance of assignment effects divided by N is the Neyman gap.
        """
        n = len(self.x)
        a = np.concatenate([self.y1[:, self.start:], self.d1[:, self.start:]], axis=1)
        b = np.concatenate([self.y0[:, self.start:], self.d0[:, self.start:]], axis=1)
        expected = np.cov(a, rowvar=False, ddof=1) / (n // 2)
        expected += np.cov(b, rowvar=False, ddof=1) / (n - n // 2)
        gap = np.cov(a - b, rowvar=False, ddof=1) / n
        return {"randomization": expected - gap, "expected_neyman": expected,
                "unidentified_neyman_gap": gap}


def make_population(n: int = 400, scenario: str | Scenario = "strong", *,
                    seed: int = 20260909, periods: int = 6, start: int = 2,
                    x: np.ndarray | None = None) -> Population:
    """Generate baseline-confounded adoption and duration-dependent outcomes.

    Types have fixed rounded shares; ranking a latent uptake propensity allocates
    types, correlating uptake with baseline X and unobserved outcome intercepts.
    Both potential outcomes reuse the same serial innovations. Binary potential
    outcomes threshold Gaussian latent values, giving risk-difference truth.
    """
    if not isinstance(n, int) or n < 4:
        raise ValueError("n must be an integer of at least four.")
    if not isinstance(periods, int) or not isinstance(start, int) or not 0 <= start < periods:
        raise ValueError("Require integer periods/start with 0 <= start < periods.")
    s = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    if not isinstance(s, Scenario):
        raise TypeError("scenario must name a scenario or be a Scenario.")
    if (s.adoption_delay or s.accelerated_share) and periods - start < 3:
        raise ValueError("Delayed and acceleration scenarios require three post periods.")
    rng = np.random.default_rng(seed)
    if x is None:
        x = np.column_stack([rng.normal(size=n), rng.integers(0, 2, size=n)])
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != (n, 2) or not np.isfinite(x).all():
            raise ValueError("x must be finite with shape (n, 2).")
    uptake_effect = rng.normal(size=n)
    outcome_effect = (s.unit_correlation * uptake_effect
                      + np.sqrt(1 - s.unit_correlation**2) * rng.normal(size=n))
    order = np.argsort(.8 * x[:, 0] + .5 * x[:, 1] + uptake_effect + rng.normal(size=n))
    kind = np.full(n, "never", dtype="U12")
    cursor = 0
    for name, share in (("always", s.always_share), ("complier", s.complier_share),
                        ("accelerated", s.accelerated_share), ("defier", s.defier_share)):
        count = int(np.floor(n * share))
        kind[order[cursor:cursor + count]] = name
        cursor += count
    never = periods + 1
    time0, time1 = np.full(n, never), np.full(n, never)
    always = kind == "always"
    time0[always] = time1[always] = max(0, start - 1)
    compliers = kind == "complier"
    time1[compliers] = start + rng.integers(0, s.adoption_delay + 1, compliers.sum())
    accelerated = kind == "accelerated"
    time0[accelerated], time1[accelerated] = start + 2, start
    time0[kind == "defier"] = start
    columns = np.arange(periods)
    d0 = (columns[None, :] >= time0[:, None]).astype(float)
    d1 = (columns[None, :] >= time1[:, None]).astype(float)
    effect = s.effect + s.effect_heterogeneity * x[:, 0]
    exposure0 = np.maximum(columns[None, :] - time0[:, None] + 1, 0)
    exposure1 = np.maximum(columns[None, :] - time1[:, None] + 1, 0)
    base = (.7 * x[:, 0, None] + .3 * x[:, 1, None] + outcome_effect[:, None]
            + .1 * columns[None, :] + .1 * x[:, 0, None] * columns[None, :])
    mu0 = base + effect[:, None] * np.sqrt(exposure0)
    mu1 = base + effect[:, None] * np.sqrt(exposure1)
    mu1[:, start:] += s.direct_encouragement_effect
    scale = .6 + s.heteroskedasticity * np.abs(x[:, 0])
    errors = rng.normal(size=(n, periods))
    for j in range(1, periods):
        errors[:, j] = (s.serial_correlation * errors[:, j - 1]
                        + np.sqrt(1 - s.serial_correlation**2) * errors[:, j])
    errors *= scale[:, None]
    if s.binary:
        mean0, mean1 = ndtr(mu0 / scale[:, None]), ndtr(mu1 / scale[:, None])
        y0, y1 = (mu0 + errors > 0).astype(float), (mu1 + errors > 0).astype(float)
    else:
        mean0, mean1 = mu0, mu1
        y0, y1 = mu0 + errors, mu1 + errors
    return Population(x, columns + 1, start, d0, d1, y0, y1, mean0, mean1,
                      np.column_stack([outcome_effect, uptake_effect]), kind,
                      time0, time1, s, seed)


def population_standardized_truth(population: Population, *, draws: int = 256,
                                  seed: int = 712733) -> dict:
    """Integrate new latent units at the empirical baseline-X distribution.

    This averages all latent adoption types, intercepts, and outcome innovations,
    unlike conditional finite-unit standardization. Monte Carlo uncertainty in
    this target is reported independently of posterior MCSE. Ratios divide the
    averaged ITTs, rather than averaging simulation-specific ratios.
    """
    if draws < 2:
        raise ValueError("Use at least two population integration draws.")
    values = []
    for child in np.random.SeedSequence(seed).spawn(draws):
        p = make_population(len(population.x), population.scenario,
                            seed=int(child.generate_state(1)[0]), x=population.x,
                            periods=len(population.t), start=population.start)
        values.append([p.conditional_mean_truth[k] for k in ("itt_y", "itt_d")])
    values = np.asarray(values)
    dy, dd = values.mean(axis=0)
    ratio = np.divide(dy, dd, out=np.full_like(dy, np.nan), where=dd != 0)
    return {"target": "new_latent_units_at_empirical_baseline_covariates",
            "draws": draws, "seed": seed, "itt_y": dy, "itt_d": dd, "wald": ratio,
            "mcse_itt_y": values[:, 0].std(axis=0, ddof=1) / np.sqrt(draws),
            "mcse_itt_d": values[:, 1].std(axis=0, ddof=1) / np.sqrt(draws)}
