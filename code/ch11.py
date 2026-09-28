"""Numerical parts of the Chapter 11 solutions (first-generation endogenous growth).

  * Exercise 11.3   Y = AK + BL: the equilibrium has a constant consumption growth rate but
    nontrivial transitional dynamics in capital and output, and a labour share falling to zero;
  * Exercise 11.6   the neoclassical model with y = A k^alpha and log preferences as alpha -> 1:
    the steady state runs off to infinity and convergence becomes infinitely slow;
  * Exercise 11.7   the AK model's calibration: two countries differing only in the capital
    income tax, a century apart;
  * Exercise 11.13  the two-sector model of Section 11.3: consumption growth always rises with
    alpha, capital growth only when theta < 1;
  * Exercise 11.19  the Pigouvian subsidy that decentralizes the optimum in the Romer model,
    and the fact that its cost is exactly the wage bill;
  * Exercise 11.20  the discrete-time version: the BGP growth rate and the admissible (beta,
    theta) region.

Run with `python code/ch11.py`.
"""
import numpy as np

from acemoglulib import plt, save

# ----------------------------------------------------------------------------------------
# Exercise 11.3: Y = A K + B L
# ----------------------------------------------------------------------------------------
def ex113(A=0.15, B=0.30, DELTA=0.05, RHO=0.02, N=0.01, THETA=3.0, k0=1.0):
    lam = A - DELTA - N                      # effective return on assets
    g = (A - DELTA - RHO) / THETA            # consumption growth
    c0 = (lam - g) * (k0 + B / lam)          # from the transversality condition
    print("Exercise 11.3: Y = A K + B L")
    print(f"  A={A}, B={B}, delta={DELTA}, rho={RHO}, n={N}, theta={THETA}, k(0)={k0}")
    print(f"  r = A - delta = {A-DELTA:.4f}, w = B = {B}, so consumption grows at "
          f"g = (A-delta-rho)/theta = {g:.6f}")
    print(f"  lambda = A - delta - n = {lam:.4f} > g, and human wealth B/lambda = "
          f"{B/lam:.6f}")
    print(f"  c(0) = (lambda - g)(k(0) + B/lambda) = {c0:.6f}")
    print(f"  {'t':>4} {'k(t)':>12} {'y(t)':>12} {'kdot/k':>10} {'ydot/y':>10} "
          f"{'labour share':>13}")
    ts = np.array([0, 10, 25, 50, 100, 200, 400])
    for t in ts:
        k = (k0 + B / lam) * np.exp(g * t) - B / lam
        y = A * k + B
        gk = g * (k + B / lam) / k
        gy = A * g * (k + B / lam) / y
        print(f"  {t:>4} {k:>12.6f} {y:>12.6f} {gk:>10.6f} {gy:>10.6f} {B/y:>13.6f}")
    print(f"  k, y and c all converge to the growth rate {g:.6f}, but from above: the capital")
    print(f"  stock grows faster than consumption while labour income is still large relative")
    print(f"  to capital.  The labour share B/(Ak+B) falls monotonically to zero.")
    return A, B, lam, g, k0, c0


# ----------------------------------------------------------------------------------------
# Exercise 11.6: the neoclassical model as alpha -> 1
# ----------------------------------------------------------------------------------------
def ex116(A=0.15, DELTA=0.05, RHO=0.02, N=0.0):
    """Log preferences (theta = 1): cdot/c = alpha A k^{alpha-1} - delta - rho."""
    print("\nExercise 11.6: y = A k^alpha with log preferences, as alpha -> 1")
    print(f"  A={A}, delta={DELTA}, rho={RHO}, n={N}; the AK limit grows at "
          f"A - delta - rho = {A-DELTA-RHO:.6f}")
    # det J = c* f''(k*) = (alpha-1)(rho+delta)[(rho+delta)/alpha - (n+delta)], free of k*
    print(f"  {'alpha':>7} {'log10 k*':>10} {'xi1':>10} {'half-life':>11} "
          f"{'growth at k=1':>14}")
    for al in (1 / 3, 0.6, 0.8, 0.9, 0.99, 0.999, 0.9999):
        log10k = np.log10(al * A / (RHO + DELTA)) / (1 - al)
        det = (al - 1) * (RHO + DELTA) * ((RHO + DELTA) / al - (N + DELTA))
        tr = RHO - N                                    # theta = 1
        xi1 = 0.5 * (tr - np.sqrt(tr ** 2 - 4 * det))
        print(f"  {al:>7.4f} {log10k:>10.2f} {xi1:>10.6f} {np.log(2)/abs(xi1):>11.2f} "
              f"{al*A-DELTA-RHO:>14.6f}")
    print("  k* diverges, the stable eigenvalue tends to zero and the half-life to infinity:")
    print("  the neoclassical model's transitional dynamics vanish in the AK limit.")


# ----------------------------------------------------------------------------------------
# Exercise 11.7: the calibration
# ----------------------------------------------------------------------------------------
def ex117(A=0.15, DELTA=0.05, RHO=0.02, THETA=3.0):
    g = lambda tau: ((1 - tau) * (A - DELTA) - RHO) / THETA
    s = lambda tau, n=0.0: ((1 - tau) * A - RHO + THETA * n
                            - (1 - tau - THETA) * DELTA) / (THETA * A)
    t1, t2 = 0.2, 0.4
    print(f"\nExercise 11.7: A={A}, delta={DELTA}, rho={RHO}, theta={THETA}")
    print(f"  g(tau)   = [(1-tau)(A-delta) - rho]/theta")
    print(f"  tau = {t1}: g = {g(t1):.6f} ({100*g(t1):.3f}% a year), saving rate {s(t1):.4f}")
    print(f"  tau = {t2}: g = {g(t2):.6f} ({100*g(t2):.3f}% a year), saving rate {s(t2):.4f}")
    print(f"  growth gap = {g(t1)-g(t2):.6f} a year")
    for T in (50, 100, 200, 500):
        print(f"    after {T:>3} years the income ratio is "
              f"{np.exp((g(t1)-g(t2))*T):>8.3f}")
    # the neoclassical comparison: a permanent level effect only
    for alpha in (1 / 3, 0.5):
        lev = ((1 - t1) / (1 - t2)) ** (alpha / (1 - alpha))
        print(f"  by contrast, the neoclassical model with alpha = {alpha:.4f} gives a "
              f"permanent level ratio of {lev:.4f}")
    print("  the AK model turns a bounded level difference into an unbounded growth difference")
    return g(t1), g(t2)


# ----------------------------------------------------------------------------------------
# Exercise 11.13: the two-sector model
# ----------------------------------------------------------------------------------------
def ex1113(A=0.15, DELTA=0.05, RHO=0.02):
    print(f"\nExercise 11.13: two-sector model, g_K = (A-delta-rho)/[1-alpha(1-theta)], "
          f"g_C = alpha g_K")
    num = A - DELTA - RHO
    print(f"  A-delta-rho = {num:.4f}")
    print(f"  {'theta':>6} | " + " ".join(f"{'a='+format(a,'.2f'):>16}"
                                          for a in (0.3, 0.5, 0.7)))
    for th in (0.5, 1.0, 2.0, 3.0):
        row = []
        for al in (0.3, 0.5, 0.7):
            gk = num / (1 - al * (1 - th))
            row.append(f"{gk:.5f}/{al*gk:.5f}")
        print(f"  {th:>6.2f} | " + " ".join(f"{r:>16}" for r in row)
              + ("   (g_K, g_C)" if th == 0.5 else ""))
    print("  reading down each column: g_C rises with alpha at every theta, while g_K rises")
    print("  with alpha only when theta < 1 and falls when theta > 1")
    print("  d g_K/d alpha has the sign of (1-theta); d g_C/d alpha = "
          "(A-delta-rho)/[1-alpha(1-theta)]^2 > 0 always")


# ----------------------------------------------------------------------------------------
# Exercise 11.19: the Pigouvian subsidy in the Romer model
# ----------------------------------------------------------------------------------------
def ex1119(alpha=1 / 3, ft=0.30, DELTA=0.05, RHO=0.02, THETA=2.0):
    """F(K, AL) = K^alpha (AL)^{1-alpha} with A = BK, so ftilde(L) = (BL)^{1-alpha} = Y/K
    and L ftilde'(L) = (1-alpha) ftilde(L), the labour share of the output-capital ratio."""
    Lftp = (1 - alpha) * ft                           # L ftilde'(L)
    R = ft - Lftp                                     # = alpha ftilde(L), the private rental
    print(f"\nExercise 11.19: Romer model with F = K^alpha (AL)^(1-alpha), A = BK "
          f"(alpha={alpha:.4f})")
    print(f"  calibrate the output-capital ratio ftilde(L) = {ft:.4f} "
          f"(a capital-output ratio of {1/ft:.2f})")
    print(f"  L ftilde'(L) = {Lftp:.6f}, private return R = ftilde - L ftilde' = {R:.6f}")
    ftp = Lftp
    ge = (R - DELTA - RHO) / THETA
    gs = (ft - DELTA - RHO) / THETA
    print(f"  with delta={DELTA}, rho={RHO}, theta={THETA}: equilibrium growth {ge:.6f} "
          f"vs social optimum {gs:.6f}")
    print(f"  finite utility needs (1-theta)(R-delta) < rho "
          f"[{(1-THETA)*(R-DELTA):+.4f} < {RHO}] and, for the planner, "
          f"(1-theta)(ftilde-delta) < rho [{(1-THETA)*(ft-DELTA):+.4f} < {RHO}]")
    zeta = ftp / R
    print(f"  the capital income subsidy that closes the gap is zeta = L ftilde'/R = "
          f"{zeta:.6f},")
    print(f"  that is {100*zeta:.1f}% of capital income --- the labour share {1-alpha:.4f} "
          f"divided by the capital share {alpha:.4f}")
    print(f"  its cost per unit of capital is L ftilde'(L) = {ftp:.6f}, which is exactly the")
    print(f"  wage bill per unit of capital: a 100% tax on labour income balances the budget")
    print(f"  at every date")


# ----------------------------------------------------------------------------------------
# Exercise 11.20: the discrete-time version
# ----------------------------------------------------------------------------------------
def ex1120(alpha=1 / 3, ft=4.0):
    """Full depreciation makes a period long, so the gross returns are large."""
    R = alpha * ft                    # F_K(K, KL) = alpha ftilde(L) for Cobb-Douglas
    print(f"\nExercise 11.20: discrete time, full depreciation, A(t) = K(t)")
    print(f"  with F = K^alpha (AL)^(1-alpha) and alpha={alpha:.4f}: ftilde(L) = {ft:.4f}, "
          f"private gross return R = alpha ftilde(L) = {R:.6f}")
    print(f"  1+g = (beta R)^(1/theta); positive growth needs beta R > 1, finite utility "
          f"needs beta R^(1-theta) < 1,")
    print(f"  so beta must lie in (1/R, R^(theta-1)), nonempty iff R > 1")
    b = 0.80
    print(f"  {'theta':>6} {'admissible beta':>22} {'beta=0.80?':>11} {'1+g':>10} "
          f"{'planner 1+g':>13}")
    for th in (0.5, 1.0, 2.0, 3.0):
        lo, hi = 1 / R, R ** (th - 1)
        ok = lo < b < min(hi, 1.0)
        grs = (b * ft) ** (1 / th)
        ok_s = b * ft ** (1 - th) < 1
        print(f"  {th:>6.2f} ({lo:>8.4f}, {hi:>8.4f}) {'yes' if ok else 'no':>11} "
              f"{(b*R)**(1/th):>10.6f} "
              f"{grs:>13.6f}{'' if ok_s else '  (planner unbounded)'}")
    print(f"  the planner internalizes A = K and faces the return ftilde(L) = {ft:.4f} rather")
    print(f"  than R = {R:.4f}, so she always grows faster --- when her problem is well posed:")
    print(f"  for theta < 1 the spillover can make her objective infinite")


def figure(A, B, lam, g, k0, c0, g1, g2):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    kk = np.linspace(0.05, 12.0, 500)
    Ak, DELTA, RHO = 0.15, 0.05, 0.02
    for al, col, ls in ((1 / 3, "#666666", ":"), (0.6, "#993300", "-."),
                        (0.9, "#006633", "--"), (0.99, "#003399", "-")):
        ax.plot(kk, al * Ak * kk ** (al - 1) - DELTA - RHO, color=col, ls=ls, lw=1.0,
                label=rf"$\alpha={al:.2f}$")
    ax.axhline(Ak - DELTA - RHO, color="black", lw=0.8)
    ax.annotate(r"$AK$ limit", (2.5, Ak - DELTA - RHO), textcoords="offset points",
                xytext=(0, -11), fontsize=7)
    ax.axhline(0.0, color="#cccccc", lw=0.5)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\dot c/c$")
    ax.set_ylim(-0.04, 0.12)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = axes[1]
    ts = np.linspace(0, 300, 600)
    k = (k0 + B / lam) * np.exp(g * ts) - B / lam
    y = A * k + B
    ln1, = ax.plot(ts, B / y, color="#003399", lw=1.2, label="labour share (left)")
    ax.set_xlabel("$t$")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 0.75)
    ax2 = ax.twinx()
    ln2, = ax2.plot(ts, A * g * (k + B / lam) / y, color="#993300", ls="--", lw=1.0,
                    label=r"$\dot y/y$ (right)")
    ax2.axhline(g, color="#666666", lw=0.7, ls=":")
    ax2.annotate(rf"$g={g:.3f}$", (300, g), textcoords="offset points", xytext=(-36, 4),
                 fontsize=7)
    ax2.set_ylim(0.024, 0.042)
    ax2.tick_params(labelsize=7)
    ax.legend(handles=[ln1, ln2], frameon=False, fontsize=7, loc="upper right")
    save(fig, "ch11_endogenous_growth")


def main():
    A, B, lam, g, k0, c0 = ex113()
    ex116()
    g1, g2 = ex117()
    ex1113()
    ex1119()
    ex1120()
    figure(A, B, lam, g, k0, c0, g1, g2)


if __name__ == "__main__":
    main()
