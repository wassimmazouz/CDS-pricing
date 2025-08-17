import numpy as np
import matplotlib.pyplot as plt

"""
CDS Pricing Models (Structural vs Reduced-Form)
-----------------------------------------------
This module implements four toy models and plotting helpers:

1) Merton classic (terminal default) via Monte Carlo
2) Extended Merton (time-varying mu, sigma, and mean-reverting debt)
3) Constant-intensity (closed-form) reduced-form CDS
4) Stochastic-intensity (CIR/Cox) via Monte Carlo

All functions are self-contained and return numpy arrays.
"""

# -------------------------------
# Utilities
# -------------------------------

def _annuity_flat_discount(t, r, delta=0.25):
    """
    Flat discount annuity used as a quick proxy in structural MC (no survival).
    For a maturity t, sum_{i=1}^{n} delta * exp(-r * i * delta), where n=floor(t/delta).
    """
    if t <= 0:
        return 0.0
    n = int(np.floor(t / delta))
    times = np.arange(1, n + 1) * delta
    return float(np.sum(delta * np.exp(-r * times)))


def _annuity_with_survival(t, r, surv_probs, delta=0.25):
    """
    Annuity with survival adjustment: sum delta * exp(-r * t_i) * S(t_i),
    where S(t_i)=surv_probs[i-1] for t_i=i*delta up to t.
    surv_probs is a 1D array of survival probabilities at grid delta, 2*delta, ..., T_max.
    """
    if t <= 0:
        return 0.0
    n = int(np.floor(t / delta))
    if n == 0:
        return 0.0
    t_i = np.arange(1, n + 1) * delta
    S = surv_probs[:n]
    return float(np.sum(delta * np.exp(-r * t_i) * S))


def _ensure_rng(seed):
    return np.random.default_rng(seed)


# -------------------------------
# 1) Merton classic (terminal default only)
# -------------------------------

def merton_classic_mc(
    V0=100.0, D=50.0, r=0.012, mu=0.122, sigma=0.5, R=0.5,
    T_max=10.0, N=100, Nsim=10000, delta=0.25, seed=42
):
    """
    Monte Carlo CDS spreads under the classic Merton setup where default can occur
    only at maturity t (i.e., V_t < D at that t). This is a didactic, non-realistic
    proxy for CDS (structural first-passage models are more appropriate).

    Returns:
        t_grid: shape (N+1,)
        cds: shape (N+1,)
        pd: shape (N+1,)
    """
    rng = _ensure_rng(seed)

    t_grid = np.linspace(0.0, T_max, N + 1)
    cds = np.zeros(N + 1)
    pd = np.zeros(N + 1)

    for j in range(1, N + 1):
        t = t_grid[j]
        Z = rng.standard_normal(Nsim)
        V_t = V0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)

        default = (V_t < D)
        pd[j] = np.mean(default)

        # Protection leg (toy): one-shot payment at t if default at t
        prot_leg = (1.0 - R) * np.exp(-r * t) * pd[j]

        # Premium leg (toy): discount-only annuity (or use survival-adjusted proxy below)
        annuity = _annuity_flat_discount(t, r, delta=delta)

        cds[j] = prot_leg / annuity if annuity > 0 else np.nan

    return t_grid, cds, pd


# --------------------------------------
# 2) Extended Merton (time-varying params)
# --------------------------------------

def extended_merton_mc(
    V0=100.0, D0=60.0, r=0.012, mu=0.122, sigma0=0.5, R=0.5,
    a=0.0, b=0.25, alpha=0.12, k=0.2, eta=0.18, D_target=50.0, kD=0.10, sigmaD=0.15,
    T_max=10.0, N=100, Nsim=10000, delta=0.25, seed=123
):
    """
    A simple 'extended Merton' toy where mu, sigma, and D evolve in time (one-step).
    At each maturity t, draw epsilons and evaluate V_t and D_t in an ad-hoc way.
    This is purely illustrative and not a standard structural calibration approach.

    Returns:
        t_grid: (N+1,)
        yield_spread: (N+1,)
        cds: (N+1,)
        pd: (N+1,)
    """
    rng = _ensure_rng(seed)

    t_grid = np.linspace(0.0, T_max, N + 1)
    yield_spread = np.zeros(N + 1)
    cds = np.zeros(N + 1)
    pd = np.zeros(N + 1)

    for j in range(1, N + 1):
        t = t_grid[j]
        ert = np.exp(-r * t)

        eps = rng.standard_normal(size=(Nsim, 4))  # epsV, epsmu, epssigma, epsD
        epsV, epsmu, epssigma, epsD = eps[:, 0], eps[:, 1], eps[:, 2], eps[:, 3]

        mu_t = mu + mu * a * t + mu * b * np.sqrt(t) * epsmu
        sigma_t = sigma0 + (alpha - k * sigma0) * t + eta * np.sqrt(max(sigma0, 1e-8) * t) * epssigma

        V_t = V0 + mu_t * V0 * t + V0 * sigma_t * np.sqrt(t) * epsV
        D_t = D0 + kD * (D_target - D0) * t + sigmaD * D0 * np.sqrt(t) * epsD

        # Bond price approximation (min(V_t, D_t)) discounted
        B = ert * np.mean(np.minimum(V_t, D_t))

        # Yield spread over risk-free: y = -log(B/D0)/t - r
        yield_spread[j] = -np.log(max(B / D0, 1e-12)) / max(t, 1e-12) - r

        # Default indicator at t
        default = (V_t < D_t)
        pd[j] = np.mean(default)

        prot_leg = (1.0 - R) * ert * pd[j]

        # Premium annuity proxy with survival S(t_i) ~ 1 - PD(t_i) at discrete times
        # We approximate S(t_i) by assuming PD grows linearly with t near j.
        # For a more accurate approach, one would compute pd on each smaller grid point.
        n = int(np.floor(t / delta))
        if n > 0:
            t_i = np.arange(1, n + 1) * delta
            # crude linear interpolation from pd[j-1] to pd[j] (avoid negative/ >1)
            pd_prev = pd[j - 1]
            pd_line = pd_prev + (pd[j] - pd_prev) * (t_i / max(t, 1e-12))
            S_i = np.clip(1.0 - pd_line, 0.0, 1.0)
            annuity = float(np.sum(delta * np.exp(-r * t_i) * S_i))
        else:
            annuity = 0.0

        cds[j] = prot_leg / annuity if annuity > 1e-16 else np.nan

    return t_grid, yield_spread, cds, pd
