"""Numerical parts of the Chapter 10 solutions (human capital and economic growth).

  * Exercise 10.3   the closed form (10.9) checked against numerical integration of (10.8);
  * Exercise 10.4   heterogeneous discount rates: what a Mincerian wage regression recovers
    when the only source of variation in schooling is impatience;
  * Exercises 10.9 and 10.10  the locus h = xi(k) implied by (10.25), its slope, the steady
    state of Proposition 10.1, and the equivalence of the two forms of the Euler equation;
  * Exercise 10.11  the elasticity of steady-state output to investment distortions, with and
    without human capital;
  * Exercise 10.15  the identity F_KH = -(K/H) F_KK for a constant-returns technology;
  * Exercise 10.19  the closed economy of Section 10.6: raising one group's ability raises the
    other group's human capital, but only from the next period on.

Run with `python code/ch10.py`.
"""
import numpy as np
from scipy.integrate import quad

from acemoglulib import bisect, plt, save

# ----------------------------------------------------------------------------------------
# Exercises 10.3 and 10.4: schooling
# ----------------------------------------------------------------------------------------
NU, GW, GH = 0.02, 0.02, 0.0            # death rate, wage growth, on-the-job human capital growth
A_ETA, B_ETA = 0.20, 0.012              # eta(S) = exp(a S - b S^2 / 2), so eta'/eta = a - b S


def ex103():
    """max_S int_S^inf e^{-(r+nu)t} w(t) h(t) dt  reduces to (10.9)."""
    r, w0, S = 0.10, 1.0, 8.0
    eta = np.exp(A_ETA * S - 0.5 * B_ETA * S ** 2)
    integrand = lambda t: np.exp(-(r + NU) * t) * w0 * np.exp(GW * t) * eta * np.exp(GH * (t - S))
    num, _ = quad(integrand, S, np.inf, limit=400)
    closed = eta * w0 * np.exp(-(r + NU - GW) * S) / (r + NU - GH - GW)
    print("Exercise 10.3: the objective (10.8) and its closed form (10.9)")
    print(f"  r={r}, nu={NU}, g_w={GW}, g_h={GH}, S={S}, eta(S)={eta:.6f}")
    print(f"  numerical integral of (10.8) = {num:.9f}")
    print(f"  closed form (10.9)           = {closed:.9f}   (difference {abs(num-closed):.2e})")


def schooling(r):
    """S* solves eta'(S)/eta(S) = r + nu - g_w, i.e. a - b S = r + nu - g_w."""
    return (A_ETA - (r + NU - GW)) / B_ETA


def ex104(rs):
    """Heterogeneous discount rates and the Mincer regression."""
    S = schooling(rs)
    marginal = A_ETA - B_ETA * S                 # = r + nu - g_w, the individual marginal return
    logW = A_ETA * S - 0.5 * B_ETA * S ** 2      # log eta(S), up to a constant
    beta = np.cov(logW, S, bias=True)[0, 1] / S.var()
    print("\nExercise 10.4: schooling when discount rates differ")
    print(f"  eta(S) = exp({A_ETA} S - {B_ETA}/2 S^2), nu={NU}, g_w={GW}")
    print(f"  S*(r) = (a - r - nu + g_w)/b, so dS*/dr = -1/b = {-1/B_ETA:.2f} years per unit of r")
    print(f"  {'r':>6} {'S* (years)':>11} {'marginal return':>16}")
    for r in (0.04, 0.07, 0.10, 0.13, 0.16):
        s = schooling(r)
        print(f"  {r:>6.3f} {s:>11.3f} {A_ETA - B_ETA*s:>16.4f}")
    print(f"  with r spread uniformly over [{rs.min():.2f}, {rs.max():.2f}]:")
    print(f"    schooling ranges over [{S.min():.2f}, {S.max():.2f}] years, mean {S.mean():.3f}")
    print(f"    individual marginal returns range over "
          f"[{marginal.min():.4f}, {marginal.max():.4f}], mean {marginal.mean():.4f}")
    print(f"    the OLS Mincer coefficient is {beta:.4f}: the average marginal return, not any")
    print(f"    individual's, and not a technological parameter")
    return S, logW, marginal, beta


# ----------------------------------------------------------------------------------------
# Exercises 10.9, 10.10 and 10.11: neoclassical growth with two capitals
# ----------------------------------------------------------------------------------------
AL, BE, DK, DH, RHO = 1 / 3, 1 / 3, 0.08, 0.04, 0.02    # f = k^al h^be, depreciation, discount

f = lambda k, h: k ** AL * h ** BE
fk = lambda k, h: AL * k ** (AL - 1) * h ** BE
fh = lambda k, h: BE * k ** AL * h ** (BE - 1)
fkk = lambda k, h: AL * (AL - 1) * k ** (AL - 2) * h ** BE
fhh = lambda k, h: BE * (BE - 1) * k ** AL * h ** (BE - 2)
fkh = lambda k, h: AL * BE * k ** (AL - 1) * h ** (BE - 1)


def bracket(g, lo, hi):
    """Widen [lo, hi] geometrically until g changes sign, then bisect."""
    while g(lo) * g(hi) > 0:
        lo, hi = lo / 10, hi * 10
        if hi > 1e300:
            raise ValueError("no sign change found")
    return bisect(g, lo, hi)


def xi(k):
    """h = xi(k) solves f_k(k,h) - f_h(k,h) = delta_k - delta_h  (equation (10.25))."""
    g = lambda h: fk(k, h) - fh(k, h) - (DK - DH)
    return bracket(g, 1e-8 * k, 1e8 * k)


def xi_prime(k, h):
    """Implicit differentiation of (10.25)."""
    return (fkh(k, h) - fkk(k, h)) / (fkh(k, h) - fhh(k, h))


def ex109_110():
    print("\nExercises 10.9 and 10.10: f(k,h) = k^alpha h^beta with "
          f"alpha={AL:.4f}, beta={BE:.4f}, delta_k={DK}, delta_h={DH}, rho={RHO}")
    print(f"  {'k':>8} {'xi(k)':>10} {'xi-prime':>10} {'numerical':>11} "
          f"{'f_k - delta_k':>14} {'f_h - delta_h':>14}")
    for k in (0.5, 1.0, 2.0, 5.0):
        h = xi(k)
        num = (xi(k * 1.000001) - xi(k * 0.999999)) / (k * 2e-6)
        print(f"  {k:>8.3f} {h:>10.6f} {xi_prime(k, h):>10.6f} {num:>11.6f} "
              f"{fk(k, h)-DK:>14.6f} {fh(k, h)-DH:>14.6f}")
    print("  the last two columns agree at every k: (10.25) says the two net returns are equal,")
    print("  which is exactly Exercise 10.10(b)")

    kstar = bracket(lambda k: fk(k, xi(k)) - DK - RHO, 1e-4, 1e4)
    hstar = xi(kstar)
    print(f"\n  steady state: f_k(k*, xi(k*)) = delta_k + rho = {DK+RHO:.4f}")
    print(f"    k* = {kstar:.6f}, h* = {hstar:.6f}, h*/k* = {hstar/kstar:.6f}")
    print(f"    check f_h(k*,h*) = delta_h + rho: {fh(kstar,hstar):.6f} vs {DH+RHO:.6f}")
    cstar = f(kstar, hstar) - DH * hstar - DK * kstar
    print(f"    c* = f(k*,h*) - delta_h h* - delta_k k* = {cstar:.6f}, "
          f"xi'(k*) = {xi_prime(kstar,hstar):.6f}")
    print(f"    with delta_k = delta_h the locus would be exactly h = (beta/alpha) k = "
          f"{BE/AL:.4f} k; here delta_k > delta_h tilts it toward human capital")
    return kstar, hstar


def ex1011():
    print("\nExercise 10.11: steady-state output and investment distortions")
    print("  f_k = (1+tau)(rho+delta_k) and f_h = (1+tau)(rho+delta_h) give")
    print("  Y(tau)/Y(tau') = [(1+tau')/(1+tau)]^{(alpha+beta)/(1-alpha-beta)}")
    print(f"  {'alpha':>6} {'beta':>6} {'exponent':>9} {'8-fold distortion':>18}")
    for a, b in ((1 / 3, 0.0), (1 / 3, 1 / 6), (1 / 3, 1 / 3), (0.30, 0.40)):
        e = (a + b) / (1 - a - b)
        print(f"  {a:>6.4f} {b:>6.4f} {e:>9.4f} {8**e:>18.2f}")
    print("  with beta = 0 this is Chapter 8's one-capital elasticity alpha/(1-alpha) = 1/2;")
    print("  adding human capital with beta = alpha quadruples the exponent, to 2")


def ex1015():
    """F_KH = -(K/H) F_KK >= 0 for any constant-returns F."""
    print("\nExercise 10.15: constant returns force capital and human capital to be complements")
    rho_ces, A = -0.5, 1.0            # F = A (K^rho + H^rho)^{1/rho}, sigma = 1/(1-rho)
    F = lambda K, H: A * (K ** rho_ces + H ** rho_ces) ** (1 / rho_ces)
    d = 1e-5
    print(f"  CES F = (K^rho + H^rho)^(1/rho) with rho = {rho_ces} "
          f"(elasticity of substitution {1/(1-rho_ces):.4f})")
    print(f"  {'K':>6} {'H':>6} {'F_KK':>12} {'F_KH':>12} {'-(K/H) F_KK':>14}")
    for K, H in ((1.0, 1.0), (1.0, 3.0), (4.0, 1.0)):
        FKK = (F(K + d, H) - 2 * F(K, H) + F(K - d, H)) / d ** 2
        FKH = (F(K + d, H + d) - F(K + d, H - d)
               - F(K - d, H + d) + F(K - d, H - d)) / (4 * d ** 2)
        print(f"  {K:>6.2f} {H:>6.2f} {FKK:>12.6f} {FKH:>12.6f} {-K/H*FKK:>14.6f}")
    print("  F_KH = -(K/H) F_KK exactly, so concavity in K alone (F_KK <= 0) already implies")
    print("  F_KH >= 0: with two factors and constant returns they cannot be substitutes")


# ----------------------------------------------------------------------------------------
# Exercise 10.19: the closed economy with imperfect labour markets
# ----------------------------------------------------------------------------------------
LAM, ETA_B, PI = 0.5, 0.4, 0.5      # worker's share, consumption share, measure of group 1
AK = 1 / 3                          # F(k,h) = k^AK h^(1-AK); gamma(e) = e^3/3


def h_choice(k, a):
    """lambda a F_h(k,h) = gamma'(h/a) = (h/a)^2  =>  h^{7/3} = (1-AK) lambda a^3 k^{AK}."""
    return ((1 - AK) * LAM * a ** 3 * k ** AK) ** (3 / 7)


def output(k, a1, a2):
    return k ** AK * (PI * h_choice(k, a1) ** (1 - AK)
                      + (1 - PI) * h_choice(k, a2) ** (1 - AK))


def ex1019():
    a1, a2 = 1.0, 1.0
    C = lambda x, y: PI * ((1 - AK) * LAM * x ** 3) ** ((1 - AK) * 3 / 7) \
        + (1 - PI) * ((1 - AK) * LAM * y ** 3) ** ((1 - AK) * 3 / 7)
    # k(t+1) = (1-eta) Phi(k(t)) with Phi(k) = C k^{AK + AK(1-AK)3/7}
    expo = AK + AK * (1 - AK) * 3 / 7
    kss = lambda x, y: ((1 - ETA_B) * C(x, y)) ** (1 / (1 - expo))
    k0 = kss(a1, a2)
    print(f"\nExercise 10.19: the closed economy of Section 10.6 "
          f"(F = k^{AK:.4f} h^{1-AK:.4f}, gamma(e)=e^3/3, lambda={LAM}, eta={ETA_B})")
    print(f"  k(t+1) = (1-eta) C k(t)^{expo:.6f}, so the map is a contraction and k* is unique")
    print(f"  initial steady state with a1 = a2 = 1: k* = {k0:.6f}, "
          f"h1 = h2 = {h_choice(k0, a1):.6f}, Y = {output(k0, a1, a2):.6f}")

    a1n = 1.10                                       # a 10% rise in group 1's ability at t=0
    print(f"\n  raise a1 to {a1n} at t = 0, with k(0) still at the old steady state:")
    print(f"    {'t':>3} {'k(t)':>10} {'h1(t)':>10} {'h2(t)':>10} {'Y(t)':>10} "
          f"{'h2 gain':>10}")
    k, h2_0 = k0, h_choice(k0, a2)
    for t in range(7):
        h1, h2, Y = h_choice(k, a1n), h_choice(k, a2), output(k, a1n, a2)
        print(f"    {t:>3} {k:>10.6f} {h1:>10.6f} {h2:>10.6f} {Y:>10.6f} "
              f"{100*(h2/h2_0-1):>9.3f}%")
        k = (1 - ETA_B) * output(k, a1n, a2)
    k1 = kss(a1n, a2)
    print(f"    ...  {k1:>10.6f} {h_choice(k1,a1n):>10.6f} {h_choice(k1,a2):>10.6f} "
          f"{output(k1,a1n,a2):>10.6f} {100*(h_choice(k1,a2)/h2_0-1):>9.3f}%  (new steady state)")
    print("  group 2's human capital is unchanged at t = 0, because k(0) is predetermined, and")
    print("  rises from t = 1 on: the externality is dynamic, working through bequests.")
    # h_i propto a_i^{3/(2+AK)} k^{AK/(2+AK)}, so C carries a_1 with exponent 3(1-AK)/(2+AK)
    s1 = PI * ((1 - AK) * LAM) ** ((1 - AK) * 3 / 7) / C(1.0, 1.0)
    el_k = s1 * 3 * (1 - AK) / (2 + AK) / (1 - expo)
    el_h2 = AK / (2 + AK) * el_k
    print(f"  closed form: d log k*/d log a1 = {el_k:.6f} and d log h2*/d log a1 = "
          f"{el_h2:.6f}")
    print(f"  numerical:                      {np.log(k1/k0)/np.log(a1n):.6f}          "
          f"          {np.log(h_choice(k1,a2)/h2_0)/np.log(a1n):.6f}")
    return k0, k1, a1, a1n, a2, expo, C


def figure(S, logW, beta, k0, k1, a1, a1n, a2, expo, C):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    ax.plot(S, logW - logW.mean(), color="#003399", lw=1.4, label=r"$\log\eta(S^*)$, the truth")
    fit = beta * (S - S.mean())
    ax.plot(S, fit, color="#993300", ls="--", lw=1.0,
            label=rf"OLS slope $\gamma_s={beta:.3f}$")
    ax.set_xlabel("years of schooling $S^*$")
    ax.set_ylabel(r"$\log W$ (demeaned)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    a1f = 2.0                      # a larger rise, so that the staircase is visible
    kf = ((1 - ETA_B) * C(a1f, a2)) ** (1 / (1 - expo))
    kk = np.linspace(1e-6, 1.35 * kf, 400)
    ax.plot(kk, kk, color="#999999", lw=0.7)
    for a, col, ls, lab in ((a1, "#666666", ":", r"$a_1=1$"),
                            (a1f, "#003399", "-", rf"$a_1={a1f:.0f}$")):
        ax.plot(kk, (1 - ETA_B) * C(a, a2) * kk ** expo, color=col, ls=ls, lw=1.1, label=lab)
    k, xs, ys = k0, [], []
    for _ in range(7):
        kp = (1 - ETA_B) * C(a1f, a2) * k ** expo
        xs += [k, k]
        ys += [k, kp]
        k = kp
    ax.plot(xs, ys, color="#993300", lw=0.8)
    ax.set_xlabel("$k(t)$")
    ax.set_ylabel("$k(t+1)$")
    ax.set_xlim(0, 1.35 * kf)
    ax.set_ylim(0, 1.35 * kf)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "ch10_human_capital")


def main():
    ex103()
    rs = np.linspace(0.04, 0.16, 2001)
    S, logW, marginal, beta = ex104(rs)
    ex109_110()
    ex1011()
    ex1015()
    k0, k1, a1, a1n, a2, expo, C = ex1019()
    figure(S, logW, beta, k0, k1, a1, a1n, a2, expo, C)


if __name__ == "__main__":
    main()
