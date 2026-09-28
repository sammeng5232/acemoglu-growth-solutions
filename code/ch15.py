"""Numerical parts of the Chapter 15 solutions (directed technological change).

  * Exercise 15.5   the ambiguous effect of gamma on the BGP growth rate, and the exact
    condition s_H > gamma_H under which it is positive;
  * Exercise 15.10  how the BGP allocation of scientists responds to eta_H;
  * Exercise 15.25  stability of the labor-augmenting BGP;
  * Exercise 15.30  the variance of a Pareto distribution.

Run with `python code/ch15.py`.
"""
import numpy as np


def sigma_of(eps, beta):
    return 1 + (eps - 1) * beta


def R(gam_H, eps, beta, etaH, H, etaL, L):
    """r* / beta = [gamma_H^eps (eta_H H)^{sigma-1} + gamma_L^eps (eta_L L)^{sigma-1}]^{1/(sigma-1)}."""
    s = sigma_of(eps, beta)
    gL = 1 - gam_H
    return (gam_H ** eps * (etaH * H) ** (s - 1) + gL ** eps * (etaL * L) ** (s - 1)) ** (1 / (s - 1))


def ex155(beta=0.5, etaH=1.0, etaL=1.0, H=1.0, L=1.0):
    print("Exercise 15.5: the effect of gamma on the BGP growth rate")
    print("  d g*/d gamma_H has the sign of (sigma-1) x [s_H - gamma_H], where")
    print("  s_H is the H sector's share of gamma_H^eps (eta_H H)^(sigma-1) + ...")
    print(f"  {'eps':>6} {'sigma':>7} {'gamma_H':>8} {'s_H':>8} {'s_H>g_H?':>9} "
          f"{'d r*/d gamma_H':>15}")
    for eps in (0.5, 2.0):
        s = sigma_of(eps, beta)
        for gH in (0.2, 0.5, 0.8):
            gL = 1 - gH
            nH = gH ** eps * (etaH * H) ** (s - 1)
            nL = gL ** eps * (etaL * L) ** (s - 1)
            sH = nH / (nH + nL)
            d = (R(gH + 1e-6, eps, beta, etaH, H, etaL, L)
                 - R(gH - 1e-6, eps, beta, etaH, H, etaL, L)) / 2e-6
            print(f"  {eps:>6.2f} {s:>7.4f} {gH:>8.2f} {sH:>8.4f} "
                  f"{'yes' if sH > gH else 'no':>9} {beta*d:>15.6f}")
    print("  the sign of d r*/d gamma_H always matches sign(sigma-1) x sign(s_H - gamma_H)")
    print("  with eta_H H = eta_L L the two sectors are symmetric, s_H > gamma_H iff")
    print("  gamma_H < 1/2 when eps < 1 and iff gamma_H > 1/2 when eps > 1")


def ex1510(beta=0.5, delta=0.3, gam=1.0, H=1.0, L=2.0, S=1.0):
    print(f"\nExercise 15.10: the BGP allocation of scientists (delta={delta}, H/L={H/L})")
    print(f"  S_L*/(S - S_L*) = eta^((1-sigma)/(1-delta sigma)) x ... so S_H* rises with eta_H")
    print(f"  iff (sigma-1)/(1-delta sigma) > 0, that is iff sigma > 1 when sigma < 1/delta")
    print(f"  {'eps':>6} {'sigma':>7} {'1/delta':>8} {'eta':>6} {'S_H*':>9} {'d S_H*/d eta':>14}")
    for eps in (0.4, 2.0):
        s = sigma_of(eps, beta)
        for eta in (0.8, 1.0, 1.25):
            def SH(e):
                ratio = (e ** ((1 - s) / (1 - delta * s))
                         * ((1 - gam / (1 + gam)) / (gam / (1 + gam)))
                         ** (-eps * (1 - delta) / (1 - delta * s))
                         * (H / L) ** (-(s - 1) * (1 - delta) / (1 - delta * s)))
                SL = S * ratio / (1 + ratio)
                return S - SL
            d = (SH(eta * 1.000001) - SH(eta * 0.999999)) / (2e-6 * eta)
            print(f"  {eps:>6.2f} {s:>7.4f} {1/delta:>8.4f} {eta:>6.2f} {SH(eta):>9.5f} "
                  f"{d:>14.6f}")
    print("  sigma > 1 (eps > 1): more productive H research attracts MORE scientists;")
    print("  sigma < 1: it attracts fewer, because expanding N_H depresses p_H sharply")


def ex1525(beta=0.5, etaL=1.0, etaK=1.0, S=1.0, SK=0.02):
    print(f"\nExercise 15.25: stability of the labor-augmenting BGP (delta = 1)")
    print(f"  with delta = 1 the technology ratio obeys d log(N_K/N_L)/dt = eta_K S_K - eta_L S_L")
    print(f"  and the scientist allocation responds to N_K/N_L with elasticity governed by sigma")
    print(f"  {'eps':>6} {'sigma':>7} {'stable?':>9}")
    for eps in (0.4, 0.8, 1.0, 1.5, 2.5):
        s = sigma_of(eps, beta)
        print(f"  {eps:>6.2f} {s:>7.4f} {'yes' if s < 1 else 'no':>9}")
    print(f"  stability requires sigma < 1, which with delta = 1 is exactly the condition")
    print(f"  sigma < 1/delta of Proposition 15.6; it also needs S_K < eta_L S = {etaL*S}")


def ex1530():
    print("\nExercise 15.30: the variance of a Pareto distribution G(y) = 1 - B y^-alpha")
    print("  the support starts at y0 = B^(1/alpha), and E[y^k] = alpha y0^k/(alpha-k) for k<alpha")
    print(f"  {'alpha':>7} {'E[y]':>12} {'E[y^2]':>12} {'Var[y]':>14}")
    B = 1.0
    for a in (0.8, 1.5, 2.0, 2.5, 4.0):
        y0 = B ** (1 / a)
        m1 = a * y0 / (a - 1) if a > 1 else np.inf
        m2 = a * y0 ** 2 / (a - 2) if a > 2 else np.inf
        v = (m2 - m1 ** 2) if (a > 2) else np.inf
        print(f"  {a:>7.2f} {m1:>12.4f} {m2:>12.4f} {v:>14.4f}")
    print("  the variance is finite only when alpha > 2, and equals")
    print("  Var[y] = alpha y0^2 / [(alpha-1)^2 (alpha-2)]; the mean itself is infinite for")
    print("  alpha <= 1.  Empirical estimates of alpha for firm sizes and incomes are close to 1,")
    print("  so the second moment is the one that fails first")


def main():
    ex155()
    ex1510()
    ex1525()
    ex1530()


if __name__ == "__main__":
    main()
