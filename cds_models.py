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


# --------------------------------------
# 3) Constant-intensity CDS (closed form)
# --------------------------------------

def constant_intensity_cds_spread(
    lam=0.12, r=0.05, R=0.4, delta=0.25, T_grid=None
):
    """
    Closed-form CDS fair spreads under flat hazard lam and flat discount rate r.
    Protection leg: (1-R) * \int_0^T e^{-(r+lam)t} lam dt
    Annuity (discrete): sum_i delta * e^{-(r+lam) t_i}, t_i=i*delta <= T

    Args:
        lam: constant hazard rate
        r: risk-free rate
        R: recovery rate
        delta: premium interval (years)
        T_grid: 1D array of maturities (years)

    Returns:
        spreads: 1D array same shape as T_grid
    """
    if T_grid is None:
        T_grid = np.linspace(0.25, 10.0, 40)
    T_grid = np.asarray(T_grid, dtype=float)

    spreads = np.zeros_like(T_grid)
    rl = r + lam
    for k, T in enumerate(T_grid):
        prot = (1.0 - R) * (lam / rl) * (1.0 - np.exp(-rl * T))
        n = int(np.floor(T / delta))
        times = np.arange(1, n + 1) * delta
        ann = np.sum(delta * np.exp(-rl * times))
        spreads[k] = prot / ann if ann > 0 else np.nan
    return spreads


# --------------------------------------
# 4) Stochastic-intensity CDS (CIR/Cox) via MC
# --------------------------------------

def cox_cir_cds_spread(
    kappa=0.8, theta=0.12, sigma=0.2, lambda0=0.12,
    r=0.05, R=0.4,
    T_grid=None, M=100000, dt=1/52, delta=0.25, seed=7
):
    """
    Monte Carlo CDS spreads with a Cox process where lambda_t follows CIR:
        d lambda_t = kappa (theta - lambda_t) dt + sigma sqrt(lambda_t) dW_t
    We simulate lambda_t (Euler with full truncation), compute the integrated hazard,
    sample default times by inversion with Exp(1) variables, and price CDS.

    Args:
        T_grid: 1D array of maturities (years).
    Returns:
        spreads: 1D array of fair spreads.
    """
    rng = _ensure_rng(seed)
    if T_grid is None:
        T_grid = np.linspace(0.25, 10.0, 40)
    T_grid = np.asarray(T_grid, dtype=float)

    T_max = float(np.max(T_grid))
    Nt = int(np.ceil(T_max / dt))
    t_axis = np.linspace(0.0, Nt * dt, Nt + 1)

    lam = np.zeros((M, Nt + 1), dtype=float)
    lam[:, 0] = lambda0

    for t in range(1, Nt + 1):
        dW = rng.standard_normal(M) * np.sqrt(dt)
        # full truncation Euler to preserve non-negativity
        lam_prev = np.maximum(lam[:, t - 1], 0.0)
        lam[:, t] = lam_prev + kappa * (theta - lam_prev) * dt + sigma * np.sqrt(lam_prev) * dW
        lam[:, t] = np.maximum(lam[:, t], 0.0)

    # cumulative hazard \int_0^t lambda_s ds via trapezoid or simple Riemann
    # use left Riemann: sum lam[:, :-1] * dt
    cum_hazard = np.cumsum(lam[:, 1:] * dt, axis=1)  # shape (M, Nt)

    # Default time by inversion with E~Exp(1): tau = inf{ t : H_t >= E }
    E = rng.exponential(size=M)
    crossed = cum_hazard >= E[:, None]
    # argmax returns 0 where all False; we mask later
    first_cross = np.argmax(crossed, axis=1)
    has_default = crossed.any(axis=1)
    tau = np.full(M, np.inf)
    tau[has_default] = t_axis[1:][first_cross[has_default]]

    # Price CDS for each T
    spreads = np.zeros_like(T_grid)
    for k, T in enumerate(T_grid):
        prot = np.mean((1.0 - R) * np.exp(-r * np.minimum(tau, T)) * (tau <= T))
        n = int(np.floor(T / delta))
        times = np.arange(1, n + 1) * delta
        # survival at payment times
        surv = np.mean(tau[:, None] > times, axis=0)
        annuity = np.sum(delta * np.exp(-r * times) * surv)
        spreads[k] = prot / annuity if annuity > 0 else np.nan

    return spreads


# -------------------------------
# Plotting helpers
# -------------------------------

def plot_series(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demo_all():
    # 1) Merton classic
    t, cds_mc, pd_mc = merton_classic_mc()
    plot_series(t, cds_mc, "CDS spread vs maturity — Merton classic (toy)", "Maturity (years)", "Spread")

    # 2) Extended Merton
    t, yld, cds_ext, pd_ext = extended_merton_mc()
    plot_series(t, yld, "Yield spread vs maturity — Extended Merton (toy)", "Maturity (years)", "Yield spread")
    plot_series(t, cds_ext, "CDS spread vs maturity — Extended Merton (toy)", "Maturity (years)", "Spread")
    plot_series(t, pd_ext, "Default probability vs maturity — Extended Merton (toy)", "Maturity (years)", "PD")

    # 3) Constant intensity
    T_grid = np.linspace(0.25, 10.0, 40)
    spreads_ci = constant_intensity_cds_spread(T_grid=T_grid)
    plot_series(T_grid, spreads_ci, "CDS spread vs maturity — Constant intensity", "Maturity (years)", "Spread")

    # 4) CIR/Cox intensity
    spreads_cox = cox_cir_cds_spread(T_grid=T_grid, M=20000)  # lower M for speed in demo
    plot_series(T_grid, spreads_cox, "CDS spread vs maturity — CIR/Cox intensity (MC)", "Maturity (years)", "Spread")


if __name__ == "__main__":
    demo_all()
