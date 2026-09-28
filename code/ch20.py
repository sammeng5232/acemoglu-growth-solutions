"""Numerical parts of the Chapter 20 solutions (structural change and economic growth).

  * Exercise 20.1   Engel's Law from the Stone-Geary aggregator, and why (20.18) is
    exactly what makes the manufacturing budget share constant;
  * Exercise 20.13  the comparative statics (20.45) and (20.46), against numerical
    derivatives of the static equilibrium;
  * Exercises 20.14 and 20.15  the asymptotic growth rates (20.62)-(20.65) in all four
    sign configurations, and the asymptotic interest rate;
  * Exercise 20.20  the open-economy Matsuyama model: high agricultural productivity
    promotes industrialization in a closed economy and prevents it in an open one.

Run with `python code/ch20.py`.
"""
import numpy as np
from scipy.optimize import brentq, fsolve

from acemoglulib import FIG, plt, save  # noqa: F401


# ---------------------------------------------------------------------------
# Section 20.1: Engel's Law
# ---------------------------------------------------------------------------


def shares(e, pA, pS, gA, gS, eA, eM, eS):
    """Budget shares of the three goods at expenditure e under (20.2)."""
    sup = e - pA * gA + pS * gS                      # supernumerary expenditure
    return (np.array([(pA * gA + eA * sup) / e,
                      eM * sup / e,
                      (eS * sup - pS * gS) / e]), sup)


def ex201(BA=1.0, BM=1.2, BS=0.8, eA=0.3, eM=0.4, eS=0.3, gA=0.4):
    print("Exercise 20.1: Engel's Law")
    pA, pS = BM / BA, BM / BS                        # (20.15)
    print("  Stone-Geary demands give p^A c^A = p^A gamma^A + eta^A * (supernumerary),")
    print("  so the budget shares are")
    print("    s^A = eta^A + [(1-eta^A) p^A gamma^A + eta^A p^S gamma^S]/e   (falling),")
    print("    s^S = eta^S - [eta^S p^A gamma^A + (1-eta^S) p^S gamma^S]/e   (rising),")
    print("    s^M = eta^M [1 - (p^A gamma^A - p^S gamma^S)/e],")
    print("  and the income elasticities are eta^A/s^A < 1 and eta^S/s^S > 1.")
    out = {}
    for name, gS in (("gamma^S from (20.18)", gA * BS / BA), ("gamma^S = 0", 0.0)):
        print(f"\n  {name}: gamma^S = {gS:.4f}, "
              f"p^A gamma^A = {pA * gA:.4f}, p^S gamma^S = {pS * gS:.4f}")
        ebar = (pA * gA * eS + pS * gS * (1 - eS)) / eS       # where c^S turns positive
        print(f"  services are demanded only above e = {ebar:.5f}")
        print(f"  {'e':>8} {'s^A':>9} {'s^M':>9} {'s^S':>9} {'elast A':>9} {'elast S':>9}")
        es, ss = [], []
        for e in (2.0, 4.0, 8.0, 32.0, 128.0):
            s, sup = shares(e, pA, pS, gA, gS, eA, eM, eS)
            print(f"  {e:>8.1f} {s[0]:>9.5f} {s[1]:>9.5f} {s[2]:>9.5f} "
                  f"{eA / s[0]:>9.5f} {eS / s[2]:>9.5f}")
        for e in np.exp(np.linspace(np.log(max(ebar * 1.02, 1.0)), np.log(200), 200)):
            s, _ = shares(e, pA, pS, gA, gS, eA, eM, eS)
            es.append(e); ss.append(s)
        out[name] = (np.asarray(es), np.asarray(ss))
    print("\n  under (20.18), gamma^A/B^A = gamma^S/B^S, the two constants are equal,")
    print("  p^A gamma^A = p^S gamma^S, and s^M = eta^M EXACTLY at every income.  That is")
    print("  precisely why (20.18) delivers a constant manufacturing share alongside")
    print("  a falling agricultural and a rising services share.")
    return out


# ---------------------------------------------------------------------------
# Section 20.2: the static equilibrium and its comparative statics
# ---------------------------------------------------------------------------


def static(K, L, A1, A2, a1e, a2e, gam, eps):
    """Solve (20.43)-(20.44) for (kappa, lambda) and return the static equilibrium.

    Solved in the logit variable v = log[kappa/(1-kappa)], which keeps the root
    finder well behaved as kappa approaches one along a constant growth path."""
    def resid(v):
        k = 1.0 / (1 + np.exp(-v))
        lam = 1.0 / (1 + (a1e / a2e) * ((1 - a2e) / (1 - a1e)) * np.exp(-v))
        lY1 = np.log(A1) + a1e * np.log(k * K) + (1 - a1e) * np.log(lam * L)
        lY2 = np.log(A2) + a2e * np.log((1 - k) * K) + (1 - a2e) * np.log((1 - lam) * L)
        # log[kappa/(1-kappa)] implied by (20.43)
        return v - (np.log(gam / a1e) - np.log((1 - gam) / a2e)
                    - (1 - eps) / eps * (lY1 - lY2))
    lo, hi = -1.0, 1.0
    while resid(lo) > 0:
        lo *= 2
        if lo < -1e6:
            raise ValueError("no bracket")
    while resid(hi) < 0:
        hi *= 2
        if hi > 1e6:
            raise ValueError("no bracket")
    v = brentq(resid, lo, hi, xtol=1e-14, rtol=8.9e-16)
    static.v = v                     # the logit, for callers that need 1 - kappa exactly
    kap = 1.0 / (1 + np.exp(-v))
    lam = 1.0 / (1 + (a1e / a2e) * ((1 - a2e) / (1 - a1e)) * np.exp(-v))
    Y1 = A1 * (kap * K) ** a1e * (lam * L) ** (1 - a1e)
    Y2 = A2 * ((1 - kap) * K) ** a2e * ((1 - lam) * L) ** (1 - a2e)
    Y = (gam * Y1 ** ((eps - 1) / eps)
         + (1 - gam) * Y2 ** ((eps - 1) / eps)) ** (eps / (eps - 1))
    r = gam * a1e * (Y / Y1) ** (1 / eps) * Y1 / (kap * K)
    w = gam * (1 - a1e) * (Y / Y1) ** (1 / eps) * Y1 / (lam * L)
    return kap, lam, Y1, Y2, Y, r, w


def ex2013(K=3.0, L=1.0, A1=1.0, A2=1.0, a1e=0.25, a2e=0.55, gam=0.5):
    print("\nExercise 20.13: the comparative statics of Proposition 20.6")
    print("  differentiating (20.43) and using d log lambda = [(1-lambda)/(1-kappa)]")
    print("  d log kappa from (20.44), the coefficient on d log kappa collects to")
    print("    [1 + (alpha_2 - alpha_1)(kappa - lambda)]/(1 - kappa),")
    print("  and solving the resulting linear equation gives the denominator")
    print("    eps + (1-eps)[1 + (alpha_2-alpha_1)(kappa-lambda)]")
    print("     = 1 + (1-eps)(alpha_2-alpha_1)(kappa-lambda),")
    print("  which is exactly the denominator of (20.45) and (20.46).")
    print(f"  {'eps':>6} {'kappa':>9} {'lambda':>9} {'dlogk/dlogK':>13} {'(20.45)':>11} "
          f"{'dlogk/dlogA2':>13} {'(20.46)':>11}")
    for eps in (0.3, 0.7, 1.5, 3.0):
        kap, lam, *_ = static(K, L, A1, A2, a1e, a2e, gam, eps)
        den = 1 + (1 - eps) * (a2e - a1e) * (kap - lam)
        f45 = (1 - eps) * (a2e - a1e) * (1 - kap) / den
        f46 = (1 - eps) * (1 - kap) / den
        h = 1e-6
        dK = (np.log(static(K * (1 + h), L, A1, A2, a1e, a2e, gam, eps)[0])
              - np.log(static(K * (1 - h), L, A1, A2, a1e, a2e, gam, eps)[0])) / (2 * h)
        dA = (np.log(static(K, L, A1, A2 * (1 + h), a1e, a2e, gam, eps)[0])
              - np.log(static(K, L, A1, A2 * (1 - h), a1e, a2e, gam, eps)[0])) / (2 * h)
        print(f"  {eps:>6.2f} {kap:>9.6f} {lam:>9.6f} {dK:>13.8f} {f45:>11.8f} "
              f"{dA:>13.8f} {f46:>11.8f}")
    print("  the denominator is always positive: for eps > 1 both factors of")
    print("  (1-eps)(alpha_2-alpha_1)(kappa-lambda) are negative, and for eps < 1 its")
    print("  magnitude is below one.  So (20.45) has the sign of (1-eps)(alpha_2-alpha_1)")
    print("  and (20.46) the sign of (1-eps), and kappa < lambda whenever alpha_1 < alpha_2")


# ---------------------------------------------------------------------------
# Exercises 20.14 and 20.15
# ---------------------------------------------------------------------------


def cgp(a1, a2, a1e, a2e, eps, n):
    """Asymptotic rates (20.62)-(20.65) with sector 1 asymptotically dominant."""
    g = n + a1 / (1 - a1e)
    x = eps * ((1 - a2e) * a1 / (1 - a1e) - a2)      # g* - g*_2
    d = (eps - 1) / eps * x
    return dict(g=g, z1=g, n1=n, g2=g - x, z2=g - d, n2=n - d)


def ex2014(a1e=0.25, a2e=0.55, n=0.01, rho=0.03, theta=2.0):
    print("\nExercises 20.14 and 20.15: the asymptotic growth rates")
    print("  with kappa* = lambda* = 1 the dominant sector absorbs all factors, so")
    print("  z*_1 = g* and n*_1 = n; the sector-1 production function then gives")
    print("  g* = n + a_1/(1-alpha_1), and (20.57)-(20.58) give the rest.  Writing")
    print("  x = g* - g*_2 = eps[(1-alpha_2) a_1/(1-alpha_1) - a_2], the same closed forms")
    print("  hold in BOTH branches of (20.61); only the sign of x, and hence the")
    print("  inequality in (20.64), flips.")
    print(f"  {'eps':>6} {'a1':>6} {'a2':>6} {'branch':>9} {'dom':>4} {'g*':>9} "
          f"{'g* other':>9} {'z* other':>9} {'n* other':>9} {'checks':>8} {'r*':>8}")
    for eps, a1, a2 in ((0.4, 0.010, 0.030), (0.4, 0.030, 0.010),
                        (2.5, 0.030, 0.010), (2.5, 0.010, 0.030)):
        A1e, A2e = a1 / (1 - a1e), a2 / (1 - a2e)
        dom1 = (eps < 1 and A1e < A2e) or (eps > 1 and A1e > A2e)
        if dom1:                       # sector 1 dominant: (20.62)-(20.65) as printed
            s, branch, dom = cgp(a1, a2, a1e, a2e, eps, n), "(20.61)", 1
            aD, aDe, aO, aOe = a1, a1e, a2, a2e
        else:                          # mirror image: swap the roles of the two sectors
            s, branch, dom = cgp(a2, a1, a2e, a1e, eps, n), "converse", 2
            aD, aDe, aO, aOe = a2, a2e, a1, a1e
        gD, zD, nD = s["g"], s["z1"], s["n1"]
        gO, zO, nO = s["g2"], s["z2"], s["n2"]
        ok = [abs(gD - (aD + aDe * zD + (1 - aDe) * nD)) < 1e-12,
              abs(gO - (aO + aOe * zO + (1 - aOe) * nO)) < 1e-12,
              abs((nD - nO) - (eps - 1) / eps * (gD - gO)) < 1e-12,
              abs((zD - zO) - (eps - 1) / eps * (gD - gO)) < 1e-12,
              abs(s["g"] - (min(gD, gO) if eps < 1 else max(gD, gO))) < 1e-12]
        rstar = rho + theta * (s["g"] - n)
        print(f"  {eps:>6.1f} {a1:>6.3f} {a2:>6.3f} {branch:>9} {dom:>4d} {s['g']:>9.6f} "
              f"{gO:>9.6f} {zO:>9.6f} {nO:>9.6f} "
              f"{sum(ok):>4d}/{len(ok)} {rstar:>8.6f}")
    print("  the five checks are the two production functions, (20.57), (20.58) and")
    print("  Proposition 20.9.  The asymptotic interest rate is r* = rho + theta g*_c")
    print("  = rho + theta a_1/(1-alpha_1) in the dominant-sector-1 case, constant because")
    print("  in (20.42) Y/Y_1 -> gamma^{eps/(eps-1)}, kappa -> 1 and Y_1/K_1 is constant")


def ex2014b(a1e=0.25, a2e=0.55, n=0.01, a1=0.010, a2=0.030, eps=0.4,
            gam=0.5, T=4000.0, dt=0.25):
    print("\n  simulating the static equilibrium along K(t) = K(0) exp(g* t):")
    print("  (kappa saturates numerically, so the sector-2 quantities are tracked in")
    print("   logs through the logit v = log[kappa/(1-kappa)]; beyond t of about 1200")
    print("   even that runs out of double precision, as 1 - kappa falls below 1e-16)")
    s = cgp(a1, a2, a1e, a2e, eps, n)
    lK = lL = lA1 = lA2 = 0.0
    rows = []
    for t in np.arange(0.0, T + dt, dt):
        kap, lam, Y1, Y2, Y, r, w = static(np.exp(lK), np.exp(lL), np.exp(lA1),
                                           np.exp(lA2), a1e, a2e, gam, eps)
        v = static.v
        l1mk = -v - np.log1p(np.exp(-v))                        # log(1 - kappa)
        c = (a1e / a2e) * ((1 - a2e) / (1 - a1e))
        l1ml = np.log(c) - v - np.log1p(c * np.exp(-v))         # log(1 - lambda)
        lK2, lL2 = l1mk + lK, l1ml + lL
        lY2 = lA2 + a2e * lK2 + (1 - a2e) * lL2
        lY1 = lA1 + a1e * (np.log(kap) + lK) + (1 - a1e) * (np.log(lam) + lL)
        rows.append((t, kap, lY1, lY2, lK2, lL2))
        lK += dt * s["g"]; lL += dt * n; lA1 += dt * a1; lA2 += dt * a2
    for i in (int(200 / dt), int(600 / dt), int(1000 / dt)):
        t, kap, lY1, lY2, lK2, lL2 = rows[i]
        nx = rows[i + 1]
        print(f"    t={t:>6.0f}: 1-kappa={1 - kap:.3e}   "
              f"g_1={(nx[2] - lY1) / dt:.6f} [{s['g']:.6f}]   "
              f"g_2={(nx[3] - lY2) / dt:.6f} [{s['g2']:.6f}]")
        print(f"              z_2={(nx[4] - lK2) / dt:.6f} [{s['z2']:.6f}]   "
              f"n_2={(nx[5] - lL2) / dt:.6f} [{s['n2']:.6f}]")


# ---------------------------------------------------------------------------
# Section 20.3 and Exercise 20.20: agricultural productivity and industrialization
# ---------------------------------------------------------------------------


def phi(nn, beta, eta):
    """phi(n) = G(1-n) - eta G'(1-n) F(n) / [(1-eta) F'(n)] with F = G = x^beta."""
    return (1 - nn) ** beta - eta / (1 - eta) * nn * (1 - nn) ** (beta - 1)


def nclosed(BA, gA, beta, eta):
    return brentq(lambda nn: phi(nn, beta, eta) - gA / BA, 1e-12, 1 - 1e-9)


def nopen(X, XF, BA, BF, gA, beta, eta, guess):
    """Free-trade allocation: equal world price and world market clearing."""
    th = ((BF * X) / (BA * XF)) ** (1 / (1 - beta))

    def eqs(v):
        nn, nF = v
        nn = min(max(nn, 1e-9), 1 - 1e-9)
        nF = min(max(nF, 1e-9), 1 - 1e-9)
        p = BA / X * (nn / (1 - nn)) ** (1 - beta)
        e1 = nn / (1 - nn) - th * nF / (1 - nF)
        e2 = (BA * (1 - nn) ** beta + BF * (1 - nF) ** beta - 2 * gA
              - eta / (1 - eta) * p * (X * nn ** beta + XF * nF ** beta))
        return [e1, e2]

    sol = fsolve(eqs, guess, full_output=False, xtol=1e-13)
    return float(np.clip(sol[0], 1e-12, 1 - 1e-12)), float(np.clip(sol[1], 1e-12, 1 - 1e-12))


def ex2020(beta=0.6, eta=0.4, gA=0.25, kap=0.02, T=300.0, dt=0.25):
    print("\nExercise 20.20: agricultural productivity in a closed and an open economy")
    print("  with F = G = x^beta the autarky price is p = (B^A/X)(n/(1-n))^{1-beta}, so")
    print("  comparative advantage in manufacturing goes to the country with the higher")
    print("  X/B^A -- the RATIO, not the level, exactly as in a Ricardian model.")
    print(f"  {'B^A':>6} {'n* closed':>11} {'growth kF(n*)':>14}")
    for BA in (0.8, 1.0, 1.3, 1.8):
        nn = nclosed(BA, gA, beta, eta)
        print(f"  {BA:>6.2f} {nn:>11.6f} {kap * nn ** beta:>14.6f}")
    print("  n* and the growth rate rise with B^A: Proposition 20.11.")

    BA, BF = 1.6, 1.0                      # home is the better farmer
    X0, XF0 = 1.0, 1.0
    print(f"\n  open economy, B^A = {BA}, B^F = {BF}, X(0) = X^F(0) = {X0}:")
    print(f"  X(0)/B^A = {X0 / BA:.4f} < X^F(0)/B^F = {XF0 / BF:.4f}, so the home economy")
    print("  has a comparative advantage in AGRICULTURE precisely because it is the")
    print("  better farmer -- the exact reverse of the closed-economy result.")
    nA = nclosed(BA, gA, beta, eta)
    X, XF = X0, XF0
    guess = [nA * 0.8, nA * 1.2]
    path = []
    for t in np.arange(0.0, T, dt):
        nn, nF = nopen(X, XF, BA, BF, gA, beta, eta, guess)
        guess = [nn, nF]
        path.append((t, nn, nF, X, XF))
        X *= np.exp(dt * kap * nn ** beta)
        XF *= np.exp(dt * kap * nF ** beta)
    path = np.asarray(path)
    print(f"  autarky n* at home = {nA:.6f}")
    print(f"  {'t':>6} {'n(t)':>10} {'n^F(t)':>10} {'X/B^A':>10} {'X^F/B^F':>10}")
    for i in (0, 40, 200, 600, len(path) - 1):
        t, nn, nF, X, XF = path[i]
        print(f"  {t:>6.0f} {nn:>10.6f} {nF:>10.6f} {X / BA:>10.4f} {XF / BF:>10.4f}")
    print("  n(0) < n* as (20.78) requires, and n(t) falls monotonically toward zero")
    print("  while n^F(t) rises to a constant strictly below one -- the world still needs")
    print("  food, so once the home economy is fully specialized in agriculture the")
    print("  foreign economy supplies the remainder.  Learning-by-doing amplifies the initial")
    print("  comparative disadvantage until the home economy de-industrializes")
    print("  completely.  This is Section 19.7's mechanism with agricultural")
    print("  productivity, rather than a small initial gap, supplying the asymmetry.")
    return path, nA


# ---------------------------------------------------------------------------


def figure(engel, path, nA):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    for (name, (es, ss)), style in zip(engel.items(), ("-", "--")):
        for j, lab in enumerate(("A", "M", "S")):
            ax1.plot(es, ss[:, j], style, color=f"{0.15 * j:.2f}",
                     label=f"$s^{lab}$" if style == "-" else None)
    ax1.set_xscale("log")
    ax1.set_xlabel("expenditure $e$")
    ax1.set_ylabel("budget share")
    ax1.legend(frameon=False, fontsize=7, loc="center right")
    ax1.set_title("Engel curves; dashed: (20.18) violated", fontsize=7.5)

    ax2.plot(path[:, 0], path[:, 1], "-", color="black", label="home $n(t)$")
    ax2.plot(path[:, 0], path[:, 2], "--", color="black", label="foreign $n^F(t)$")
    ax2.axhline(nA, color="0.6", lw=0.5)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("manufacturing employment")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=False, fontsize=7, loc="center right")
    ax2.set_title("open economy; grey: home autarky $n^{*}$", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch20_structural_change")


def main():
    engel = ex201()
    ex2013()
    ex2014()
    ex2014b()
    path, nA = ex2020()
    figure(engel, path, nA)


if __name__ == "__main__":
    main()
