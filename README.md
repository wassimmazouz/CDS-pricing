# CDS Pricing — Structural vs Reduced‑Form

This repository contains a compact implementation of **CDS pricing** under both **structural** (Merton‑type) and **reduced‑form** (intensity‑based) models, plus a clear theoretical walkthrough. Code is intentionally lightweight and self‑contained.

> ⚠️ Structural "Merton classic" only allows **terminal default at maturity**. That is **not** realistic for CDS, which can default at any time. It is included here for pedagogy. For CDS, **intensity models** (constant or stochastic) are the standard baseline.

---

## 📦 Contents

```
.
├─ cds_models.py        # all models + plotting helpers
├─ requirements.txt     # numpy, matplotlib
└─ README.md            # this file
```

---

## 🚀 Quickstart

```bash
# 1) create & activate a venv (optional)
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) install
pip install -r requirements.txt

# 3) run the demo (opens several plots)
python cds_models.py
```

---

## 🔧 What the code does

- **Structural**
  - **Merton classic (toy)**: Monte Carlo of $V_t$ as GBM; "default" if $V_t<D$ at maturity $t$.
  - **Extended Merton (toy)**: ad‑hoc time‑variation in $\mu$, $\sigma$, and debt $D_t$, priced by Monte Carlo.
- **Reduced‑form**
  - **Constant intensity**: closed‑form CDS spread with flat hazard $\lambda$ and flat $r$.
  - **Stochastic intensity (CIR/Cox)**: Monte Carlo with $\lambda_t$ following a CIR process; default time via exponential inversion.

Each routine returns arrays ready to plot. See `demo_all()` in `cds_models.py`.

---

## 📚 Theory (concise but complete)

### 1) Credit default swaps (CDS)

A payer CDS pays the **protection leg** upon default before maturity $T$, in exchange for a periodic **premium leg** at rate $S$ (the **CDS spread**). Under risk‑neutral measure:
- **Protection leg PV** up to $T$:
$$\text{PL}(T)=(1-R)\,\mathbb{E}\big[e^{-\int_0^\tau r(u)du}\mathbf{1}_{\{\tau\le T\}}\big]$$
- **Premium leg PV** with coupon interval $\delta$ at payment dates $0< t_1<\dots<t_n\le T$:
$$\text{A}(T)=\sum_{i=1}^{n}\delta\,\mathbb{E}\big[e^{-\int_0^{t_i} r(u)du}\mathbf{1}_{\{\tau>t_i\}}\big]$$

The **fair spread** solves $S(T)=\text{PL}(T)/\text{A}(T)$. With **flat** rate $r$ and survival $Q(t)=\mathbb{Q}(\tau>t)$:
$$\text{PL}(T)=(1-R)\int_0^{T}e^{-rt}\big(-dQ(t)\big)$$
$$\text{A}(T)=\sum_{i=1}^{n}\delta\,e^{-rt_i}Q(t_i)$$

If $Q(t)=e^{-\Lambda(t)}$ with cumulative hazard $\Lambda(t)=\int_0^t\lambda(u)du$, then $-dQ(t)=\lambda(t)e^{-\Lambda(t)}dt$.

---