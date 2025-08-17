# CDS Pricing — Structural vs Reduced‑Form

This repository contains a compact implementation of **CDS pricing** under both **structural** (Merton‑type) and **reduced‑form** (intensity‑based) models, plus a clear theoretical walkthrough. Code is intentionally lightweight and self‑contained.

> ⚠️ Structural "Merton classic" only allows **terminal default at maturity**. That is **not** realistic for CDS, which can default at any time. It is included here for pedagogy. For CDS, **intensity models** (constant or stochastic) are the standard baseline.

---

##  Quickstart

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

## 📚 Theory

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

### 2) Structural model (Merton classic)

**Firm value** $V_t$ follows a GBM:
$$dV_t=\mu V_t\,dt+\sigma V_t\,dW_t,$$
and **default** occurs **only at maturity $T$** if $V_T<D$ for a fixed debt face value $D$. Under risk‑neutral pricing, set $\mu=r$. Then
$$\mathbb{Q}(V_T<D)=\Phi(-d_2),\quad d_2=\frac{\ln(V_0/D)+(r-\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}.$$
This gives a terminal **PD** and is useful for equity‑credit links, but not for true CDS timing. In the code we simulate $V_t$ and estimate $\mathbb{Q}(V_t<D)$ by Monte Carlo at each $t$ to form a **toy CDS** spread:
- Protection leg proxy: $(1-R)e^{-rt}\,\mathbb{Q}(V_t<D)$.
- Premium leg proxy: discount‑only annuity $\sum_i\delta e^{-rt_i}$ (or a crude survival proxy).

**Limitations:** no early default; recovery is ad‑hoc; premium survival is approximate. Use reduced‑form models for CDS curves.

---

### 3) Extended Merton (toy)

We allow **time variation** in $\mu_t$, $\sigma_t$, and a **mean‑reverting debt** $D_t$ (one‑step evaluation at each maturity $t$):
- $\mu_t\approx \mu+\mu\,a\,t+\mu\,b\sqrt{t}\,\varepsilon_\mu$
- $\sigma_t\approx \sigma_0+( \alpha-k\sigma_0)t+\eta\sqrt{\sigma_0 t}\,\varepsilon_\sigma$
- $D_t\approx D_0+k_D(D_{\text{target}}-D_0)t+\sigma_D D_0\sqrt{t}\,\varepsilon_D$

Then we simulate $V_t$ and $D_t$, compute an approximate risky bond price $B(t)\approx e^{-rt}\mathbb{E}[\min(V_t,D_t)]$ and define the **yield spread** $y(t)=-\frac{1}{t}\log(B(t)/D_0)-r$. A toy CDS spread is obtained as in the classic case with survival‑adjusted annuity proxy.

**Note:** This is illustrative, not a calibrated structural model (e.g., no first‑passage barrier like Black‑Cox).

---

### 4) Reduced‑form: constant intensity

Assume **flat hazard** $\lambda$ and flat $r$ so $Q(t)=e^{-\lambda t}$ and $Z(t)=e^{-rt}$. Then
$$\text{PL}(T)=(1-R)\int_0^{T}e^{-(r+\lambda)t}\lambda\,dt=(1-R)\frac{\lambda}{r+\lambda}\big(1-e^{-(r+\lambda)T}\big),$$
and with discrete coupons at $t_i=i\delta$,
$$\text{A}(T)=\sum_{i=1}^{\lfloor T/\delta\rfloor}\delta\,e^{-(r+\lambda)t_i}.$$
Thus the **fair spread** is
$$S(T)=\frac{(1-R)\frac{\lambda}{r+\lambda}\big(1-e^{-(r+\lambda)T}\big)}{\sum_{i=1}^{\lfloor T/\delta\rfloor}\delta\,e^{-(r+\lambda)t_i}}.$$
As $\delta\to 0$ and $r\to 0$, $S(T)\to (1-R)\lambda$.

---

### 5) Reduced‑form: stochastic intensity (CIR/Cox)

Let the **intensity** follow a **CIR** process
$$d\lambda_t=\kappa(\theta-\lambda_t)\,dt+\sigma\sqrt{\lambda_t}\,dW_t,$$
ensuring mean reversion and non‑negativity (Feller condition $2\kappa\theta\ge\sigma^2$ helps). Default time $\tau$ is defined by
$$\tau=\inf\{t:\Lambda_t\ge E\},\quad \Lambda_t=\int_0^t\lambda_u\,du,\ E\sim\text{Exp}(1).$$
In Monte Carlo:
1. Simulate $\lambda_t$ on a fine grid with **full‑truncation Euler**.
2. Build cumulative hazard $\Lambda_t$ by summation.
3. Draw $E\sim\text{Exp}(1)$ and locate the first time $\Lambda_t\ge E$ to get $\tau$.
4. Price CDS via the generic PL/A formulas with simulated survival probabilities.

---

## 🧪 Reproducibility & parameters

- We set RNG seeds in each function for reproducibility.
- Premium interval $\delta$ defaults to quarterly $0.25$.
- For the CIR/Cox demo, reduce $M$ if plots are slow; increase for smoother curves.

---

## 🛠️ Calibration notes (high level)

- In practice, one **bootstraps** $\lambda(t)$ from market CDS quotes $\{S(T_k)\}$ using the discrete PL/A equations and an assumed $R$.
- Structural calibration typically infers $V_0$, $\sigma$, and $D$ (or barrier) from equity and balance‑sheet data; for CDS curves, **first‑passage** models (e.g., Black‑Cox) are preferred to terminal‑default Merton.
- Recovery $R$ is a major driver; market convention often fixes $R$ (e.g., $40\%$) for bootstraps.

---

## 📈 Plot gallery (what you should see)

- Merton classic (toy): increasing PD and spreads with maturity; noisy due to MC.
- Extended Merton (toy): yield spread curve plus toy CDS spreads.
- Constant intensity: smooth, closed‑form term structure.
- CIR/Cox: smooth but slightly noisy spreads; shape driven by $\kappa,\theta,\sigma$.

---

## 📎 References (classic sources)

- Merton, R. C. (1974): "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates."
- Cox, Ingersoll, and Ross (1985): "A Theory of the Term Structure of Interest Rates."

---

## ✅ License

MIT (or adapt to your needs).
