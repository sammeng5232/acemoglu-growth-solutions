"""Numerical part of the Chapter 6 solutions.

Exercise 6.10: a plan that satisfies every Euler equation but violates the transversality
condition, in the log/Cobb-Douglas optimal growth model with full depreciation (Example 6.4),

    max sum_t beta^t log c(t)   s.t.  k(t+1) = k(t)^alpha - c(t),   k(0) given.

With the consumption share u(t) = c(t)/k(t)^alpha (so the saving rate is s(t) = 1 - u(t)), the
Euler equation is equivalent to the first-order difference equation

    u(t+1) = alpha*beta*u(t)/(1 - u(t)),

whose fixed points are u* = 1 - alpha*beta (the optimum) and u = 0 (save everything).  The
optimum is an *unstable* fixed point of this forward map: every u(0) != u* satisfies all the
Euler equations, yet drifts away.  Paths with u(0) < u* stay feasible forever, have u(t) -> 0
geometrically, and violate the transversality condition; paths with u(0) > u* run capital
negative in finite time.

Everything is computed in logs, since u(t) and k(t) both vanish geometrically.

Run with `python code/ch6.py`.
"""
import numpy as np

from acemoglulib import plt, save

ALPHA, BETA, K0 = 1 / 3, 0.9, 1.0
USTAR = 1 - ALPHA * BETA                  # optimal consumption share, = 1 - alpha*beta


def path(u0, T, constant=False):
    """Consumption shares, log capital, log consumption and the transversality term.

    Returns (u, logk, logc, tv, T_feasible): if the path runs capital negative, the arrays
    stop there and T_feasible < T records the last feasible date.
    """
    u = np.empty(T + 1)
    u[0] = u0
    T_feas = T
    for t in range(T):
        if u[t] >= 1:                     # c(t) >= k(t)^alpha: capital would go negative
            T_feas = t - 1
            u = u[:t]
            break
        u[t + 1] = u0 if constant else ALPHA * BETA * u[t] / (1 - u[t])
    n = len(u)
    logk = np.empty(n + 1)
    logk[0] = np.log(K0)
    for t in range(n):
        logk[t + 1] = np.log1p(-u[t]) + ALPHA * logk[t]
    logc = np.log(u) + ALPHA * logk[:-1]
    # transversality term beta^t f'(k(t)) u'(c(t)) k(t) = beta^t * alpha / u(t)
    tv = BETA ** np.arange(n) * ALPHA / u
    return u, logk, logc, tv, T_feas


def value(logc):
    return float(np.sum(BETA ** np.arange(len(logc)) * logc))


def main():
    T = 400
    print(f"alpha={ALPHA:.4f}, beta={BETA}, k(0)={K0}")
    print(f"optimal consumption share u* = 1 - alpha*beta = {USTAR:.4f} "
          f"(saving rate s* = alpha*beta = {ALPHA*BETA:.4f})")
    print("\nEuler map u(t+1) = alpha*beta*u(t)/(1-u(t)):")
    print(f"  fixed points u = {USTAR:.4f}, slope alpha*beta/(1-u*)^2 = "
          f"{ALPHA*BETA/(1-USTAR)**2:.3f} > 1  (unstable)")
    print(f"               u = 0,      slope alpha*beta = {ALPHA*BETA:.3f} < 1  (stable)")

    # the optimum is imposed exactly: iterating the map from u* drifts away numerically,
    # which is itself the content of the exercise (the forward Euler map is unstable).
    u_opt, _, logc_opt, tv_opt, _ = path(USTAR, T, constant=True)
    v_opt = value(logc_opt)
    print(f"\noptimal plan: value = {v_opt:.6f}; transversality term beta^t*alpha/u* = "
          f"{tv_opt[100]:.3e} at t=100 and {tv_opt[-1]:.3e} at t={T}, so it vanishes")

    print("\nplans satisfying every Euler equation but starting away from u*:")
    for u0 in (0.60, 0.50, 0.40):
        u, _, logc, tv, _ = path(u0, T)
        print(f"  u(0)={u0:.2f} (over-saving): u(20)={u[20]:.3e}, u(100)={u[100]:.3e} -> 0;  "
              f"value = {value(logc):.6f}  (loss {v_opt - value(logc):.6f});  "
              f"transversality term at t=100: {tv[100]:.3e}, at t={T}: {tv[-1]:.3e}")
    for u0 in (0.75, 0.80):
        u, _, _, _, tf = path(u0, T)
        print(f"  u(0)={u0:.2f} (under-saving): capital goes negative at t = {tf + 2} "
              f"(infeasible)")

    u, _, _, tv, _ = path(0.50, T)
    tt = np.arange(0, 61)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax1.semilogy(tt, u[:61], color="#003399", label=r"$u(0)=0.5$: Euler path")
    ax1.axhline(USTAR, color="#993300", ls="--", lw=1.0, label=r"optimum $u^*=1-\alpha\beta$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"consumption share $c(t)/k(t)^\alpha$")
    ax1.legend(frameon=False, fontsize=7, loc="lower left")
    ax2.semilogy(tt, tv[:61], color="#003399", label="Euler path")
    ax2.semilogy(tt, tv_opt[:61], color="#993300", ls="--", lw=1.0, label="optimal path")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\beta^tf'(k(t))u'(c(t))k(t)$")
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "ch6_transversality")


if __name__ == "__main__":
    main()
