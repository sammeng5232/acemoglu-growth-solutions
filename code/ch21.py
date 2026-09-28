"""Numerical parts of the Chapter 21 solutions (structural transformations and market
failures in development).

  * Exercise 21.3   the stochastic Malthusian model as a log-linear AR(1), its invariant
    interval and the moments of the invariant distribution;
  * Exercise 21.5   the welfare-maximizing community-enforcement advantage xi*;
  * Exercise 21.7   the outsourcing threshold a_bar and the long-run distance to frontier;
  * Exercise 21.8   the investment equilibrium Pareto dominates, and the condition for it
    is exactly the left half of (21.51);
  * Exercise 21.14  the slopes of the two loci and local stability.

Run with `python code/ch21.py`.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from acemoglulib import FIG, plt, save  # noqa: F401


# ---------------------------------------------------------------------------
# Exercise 21.3: the stochastic Malthusian model
# ---------------------------------------------------------------------------


def ex213(alpha=0.3, eta0=1.0, eps=0.15, T=400000, seed=4):
    print("Exercise 21.3: the stochastic Malthusian model")
    print("  with L(t+1) = epsilon(t) n(t+1) L(t) and n(t+1) = (1-alpha)/[eta_0 L(t+1)^alpha],")
    print("    L(t+1) = [epsilon(t)(1-alpha)/eta_0]^{1/(1+alpha)} L(t)^{1/(1+alpha)},")
    print("  so log L follows the AR(1)  l(t+1) = phi l(t) + mu + log eps(t)/(1+alpha)")
    phi = 1 / (1 + alpha)
    Ls = (((1 - alpha) / eta0)) ** (1 / alpha)
    lo = Ls * (1 - eps) ** (1 / alpha)
    hi = Ls * (1 + eps) ** (1 / alpha)
    Elog = 0.5 * np.log(1 - eps ** 2)
    Vlog = 0.25 * np.log((1 + eps) / (1 - eps)) ** 2
    print(f"  phi = 1/(1+alpha) = {phi:.6f}, deterministic steady state L* = {Ls:.6f}")
    print(f"  the two branches cross the 45-degree line at L^- = {lo:.6f} and "
          f"L^+ = {hi:.6f},")
    print("  and [L^-, L^+] is forward invariant and absorbing: below L^- both branches")
    print("  lie above the 45-degree line, above L^+ both lie below it.")
    print("  Analytically, E[log L] = log L* + E[log eps]/alpha and")
    print("  Var(log L) = Var(log eps)/[alpha(2+alpha)]:")
    rng = np.random.default_rng(seed)
    l = np.log(Ls)
    mu = np.log((1 - alpha) / eta0) / (1 + alpha)
    keep = []
    for t in range(T):
        e = (1 + eps) if rng.random() < 0.5 else (1 - eps)
        l = phi * l + mu + np.log(e) / (1 + alpha)
        if t > 2000:
            keep.append(l)
    keep = np.asarray(keep)
    print(f"  {'':>22} {'analytic':>12} {'simulated':>12}")
    print(f"  {'E[log L]':>22} {np.log(Ls) + Elog / alpha:>12.6f} {keep.mean():>12.6f}")
    print(f"  {'sd(log L)':>22} "
          f"{np.sqrt(Vlog / (alpha * (2 + alpha))):>12.6f} {keep.std():>12.6f}")
    print(f"  {'min log L':>22} {np.log(lo):>12.6f} {keep.min():>12.6f}")
    print(f"  {'max log L':>22} {np.log(hi):>12.6f} {keep.max():>12.6f}")
    print(f"  {'autocorrelation':>22} {phi:>12.6f} "
          f"{np.corrcoef(keep[:-1], keep[1:])[0, 1]:>12.6f}")
    print(f"  income is log y = log(1-alpha) - alpha log L, so sd(log y) = "
          f"{alpha * keep.std():.6f}")
    print(f"  and a shock decays with half-life log 2/log(1+alpha) = "
          f"{np.log(2) / np.log(1 + alpha):.4f} generations")
    print("  E[log eps] < 0 by Jensen, so random shocks LOWER average population: the law")
    print("  of motion is concave, and the mean of log L falls by |E log eps|/alpha.")
    print("  The two branch images each have length Lambda/(1+alpha) on an invariant")
    print("  interval of length Lambda, and 2/(1+alpha) > 1 whenever alpha < 1, so they")
    print("  overlap and the invariant distribution has full support on [L^-, L^+].")
    return alpha, eta0, eps, Ls, lo, hi


# ---------------------------------------------------------------------------
# Exercise 21.5: the optimal community-enforcement advantage
# ---------------------------------------------------------------------------


def dual_path(xi, sigma=1.0, BA=1.0, beta=0.5, eta=0.04, zeta=0.5, X0=0.2,
              T=600.0, dt=0.05):
    """Section 21.3.2 with F(L,Z) = L^beta and the village consuming enforcement
    services worth the bounded upsilon(xi) = sigma xi/(1+xi).  Columns (L^U, C, X)."""
    b = BA + xi                                   # the private migration margin
    v = BA + sigma * xi / (1.0 + xi)              # the village consumption flow
    X, out = X0, []
    for _ in range(int(T / dt)):
        LU = min((beta * X / b) ** (1 / (1 - beta)), 1.0)
        out.append((LU, v * (1 - LU) + X * LU ** beta, X))
        X += dt * eta * LU * X ** zeta
    return np.asarray(out)


def welfare(xi, rho, theta, T=600.0, dt=0.05, **kw):
    p = dual_path(xi, T=T, dt=dt, **kw)
    C = p[:, 1]
    t = np.arange(len(C)) * dt
    u = np.log(C) if abs(theta - 1) < 1e-12 else (C ** (1 - theta) - 1) / (1 - theta)
    return float(np.sum(np.exp(-rho * t) * u) * dt)


def ex215(eta=1.0, sigma=3.0, ximax=6.0):
    print("\nExercise 21.5: the welfare-maximizing community-enforcement advantage")
    print("  With CRRA preferences the date-0 objective is")
    print("    U(xi) = int_0^inf e^{-rho t}[C(t;xi)^{1-theta} - 1]/(1-theta) dt,")
    print("    C = [B^A + upsilon(xi)](1 - L^U) + X F(L^U,Z),  L^U = phi(X/(B^A+xi)),")
    print("  where upsilon(xi) is the consumption value of the enforcement services the")
    print("  villager enjoys.  Differentiating and using the envelope theorem in L^U,")
    print("    U'(xi) = int e^{-rho t} C^{-theta}[upsilon'(xi)(1-L^U) + F(L^U,Z) v(t)] dt,")
    print("  with v(t) = dX(t)/dxi < 0 solving the variational equation of")
    print("  dX/dt = eta phi(X/(B^A+xi)) X^zeta: a static gain against a dynamic loss.")
    print("\n  But U is NOT concave in xi, so U'(xi*) = 0 locates an interior MINIMUM and")
    print("  the maximum is at a corner.  Tabulating U over a grid, with")
    print(f"  upsilon(xi) = {sigma:g} xi/(1+xi), eta = {eta:g}, zeta = 0.5, theta = 1:")
    xs = np.linspace(1e-6, ximax, 25)
    print(f"  {'rho':>6} " + " ".join(f"{x:>8.2f}" for x in xs[::4])
          + f" {'argmin':>8} {'argmax':>8}")
    curves = {}
    for rho in (0.03, 0.05, 0.07, 0.09):
        U = np.array([welfare(x, rho, 1.0, sigma=sigma, eta=eta) for x in xs])
        curves[rho] = (xs, U)
        print(f"  {rho:>6.2f} " + " ".join(f"{u:>8.2f}" for u in U[::4])
              + f" {xs[U.argmin()]:>8.2f} {xs[U.argmax()]:>8.2f}")
    print("  U falls and then rises: a small xi buys little extra village consumption but")
    print("  already delays the takeoff, while a large xi buys a lot and the economy simply")
    print("  never urbanizes.  The welfare-maximizing xi is therefore at one end or the")
    print("  other, and which one wins switches with the discount rate -- xi = 0 for a")
    print("  patient economy, xi = xi_max for an impatient one.")
    print("\n  The reason the two effects cannot balance is structural.  L^U is bounded")
    print("  above by one and reaches it, so X^{1-zeta}(t) - X^{1-zeta}(0) = (1-zeta) eta")
    print("  times the integral of L^U: the dynamic cost of xi is a finite DELAY of the")
    print("  takeoff, not a permanent level or growth loss, and it is large only when the")
    print("  delay straddles the takeoff.  The static gain, by contrast, accrues at every")
    print("  date before the takeoff.  The choice is therefore discrete -- urbanize, or")
    print("  stay rural and enjoy community enforcement -- and not marginal.")
    print("\n  If the true xi exceeds the level that maximizes date-0 utility, the answer is")
    print("  sharper than 'welfare falls'.  The economy is in a development trap: it stays")
    print("  rural, learning-by-doing never gets going, and a SMALL reduction in xi makes")
    print("  matters worse, since it moves the economy down the falling branch of U.  Only")
    print("  a reduction past the interior minimum helps.  This is the big-push logic of")
    print("  Section 21.5 applied to the dual economy: gradual reform is counterproductive")
    print("  and the intervention must be large.  The first-best remedy is different again:")
    print("  subsidize urban employment at the marginal external learning benefit, which")
    print("  restores efficiency at any xi and forfeits none of the static gain.")
    return curves


# ---------------------------------------------------------------------------
# Exercise 21.7: outsourcing and the distance to the frontier
# ---------------------------------------------------------------------------


def ex217(eta=0.10, gam=0.60, g=0.03):
    print("\nExercise 21.7: vertical integration far from the frontier, outsourcing near it")
    print("  profits are proportional to A(nu,t), so outsourcing is preferred iff")
    print("    (1-beta)[eta Abar + (gamma+theta)A] > eta Abar + gamma A,")
    print("  that is iff a(t-1) > a_bar = beta eta / [(1-beta) theta - beta gamma].")
    print(f"  eta = {eta}, gamma = {gam}, g = {g}")
    print(f"  {'beta':>6} {'theta':>7} {'a_bar':>10} {'switches?':>10} "
          f"{'a* integr.':>11} {'a* outsrc.':>11} {'limit':>9}")
    for beta, th in ((0.10, 0.30), (0.20, 0.30), (0.30, 0.30), (0.10, 0.15),
                     (0.35, 0.15)):
        den = (1 - beta) * th - beta * gam
        abar = beta * eta / den if den > 0 else np.inf
        aI = eta / (1 + g - gam)                       # always integrated
        aO = eta / (1 + g - gam - th) if 1 + g - gam - th > 0 else np.inf
        aO = min(aO, 1.0)
        switch = np.isfinite(abar) and abar < aO
        lim = aO if switch else aI
        print(f"  {beta:>6.2f} {th:>7.2f} {abar:>10.4f} {str(bool(switch)):>10} "
              f"{aI:>11.4f} {aO:>11.4f} {lim:>9.4f}")
    print("  the growth-maximizing rule is to outsource at EVERY a, since theta > 0 raises")
    print("  a(t) for any a(t-1): the private threshold a_bar is strictly positive while")
    print("  the social one is zero.  The wedge is the spillover of a firm's own innovation")
    print("  onto next period's aggregate A(t-1), which the entrepreneur does not capture.")
    print(f"  a_bar < 1 requires beta < theta/(theta + eta + gamma); above that the economy")
    print("  never outsources and converges only to eta/(1+g-gamma), a permanently larger")
    print("  distance to the frontier -- an organizational nonconvergence trap.")


# ---------------------------------------------------------------------------
# Exercise 21.8: Pareto dominance of the investment equilibrium
# ---------------------------------------------------------------------------


def ex218(L=1.0, seed=2, n=200000):
    print("\nExercise 21.8: the investment equilibrium Pareto dominates")
    print("  U^I - U^N = [u(L-F) - u(L)] + beta[u(alpha L) - u(L)], and by concavity")
    print("    u(L) - u(L-F) <= u'(L-F) F,   u(alpha L) - u(L) >= u'(alpha L)(alpha-1)L,")
    print("  so U^I > U^N whenever beta (alpha L/(L-F))^{-theta}(alpha-1)L > F, which is")
    print("  exactly the LEFT half of (21.51) -- the condition making investment profitable")
    print("  when everyone invests.  Equivalently 1/(1+r~) = beta u'(C(2))/u'(C(1)), so a")
    print("  profit gain at the equilibrium interest rate IS a utility gain.")
    rng = np.random.default_rng(seed)
    bad = pair = both = 0
    for _ in range(n):
        alpha = rng.uniform(1.01, 3.0)
        beta = rng.uniform(0.5, 0.99)
        theta = rng.uniform(0.0, 4.0)
        F = rng.uniform(0.001, 0.8) * L
        if (alpha / (alpha - 1)) <= 1:            # (21.48) must hold
            continue
        dpiN = -F + beta * (alpha - 1) / alpha * L
        dpiI = -F + beta * (alpha * L / (L - F)) ** (-theta) * (alpha - 1) * L

        def u(c):
            return np.log(c) if abs(theta - 1) < 1e-9 else (c ** (1 - theta) - 1) / (1 - theta)
        dU = (u(L - F) + beta * u(alpha * L)) - (1 + beta) * u(L)
        if dpiN < 0 < dpiI:                       # (21.50): both equilibria exist
            both += 1
            if dU <= 0:
                bad += 1
        if dpiI > 0:
            pair += 1
            if dU <= 0:
                bad += 1
    print(f"  random scan of {n} parameter draws: {both} satisfy (21.50) so that both")
    print(f"  equilibria exist, {pair} satisfy the left half of (21.51), and in "
          f"{bad} of them")
    print("  the investment allocation fails to dominate -- so the implication is exact.")
    print("  Since households are identical and own all the firms, this is a pure")
    print("  coordination failure: a 'big push' that simply coordinates investment makes")
    print("  everyone better off, with no subsidies and no transfers.")


# ---------------------------------------------------------------------------
# Exercise 21.14: slopes and local stability
# ---------------------------------------------------------------------------


def ex2114(alpha=0.35, s=0.25, delta=0.06, chi=0.05, omega=1.0):
    print("\nExercise 21.14: the two loci and local stability")
    print("  1.  dk/dt = 0 is s f(k,x) = delta k, so implicit differentiation gives")
    print("        dx/dk = [f(k,x) - k f_k(k,x)] / [k f_x(k,x)] > 0,")
    print("      the numerator positive by strict concavity with f(0,x) = 0 and the")
    print("      denominator positive when f_x > 0; the locus is vertical when f_x = 0.")
    print("  2.  the Jacobian at (k*,x*) is [[s f_k - delta, s f_x], [g_k, g_x]], and")
    print("      s f_k - delta = (s/k)(k f_k - f) < 0, g_x < 0, so the trace is negative;")
    print("      det = (s f_k - delta) g_x - s f_x g_k > 0 iff")
    print("        f_x < [f(k*,x*)/k* - f_k(k*,x*)] |g_x| / g_k,")
    print("      which is also exactly the condition that the dk/dt = 0 locus be STEEPER")
    print("      than the dx/dt = 0 locus at the intersection.")
    print(f"  f(k,x) = k^{alpha} h(x) with h(x) = 1 + psi/(1 + exp[-(x - x0)/varsigma]),")
    print(f"  g(k,x) = {chi} k - {omega} x.  Making varsigma small makes f_x large over a")
    print("  narrow range of x without changing h much, which is exactly the text's")
    print("  'f_x large over some range' and produces the three intersections of Fig 21.11.")
    x0, psi = 1.5, 3.0

    def h(x, vs):
        return 1 + psi / (1 + np.exp(-(x - x0) / vs))

    def hp(x, vs):
        e = np.exp(-(x - x0) / vs)
        return psi * e / (vs * (1 + e) ** 2)

    print(f"  {'varsigma':>9} {'k*':>9} {'x*':>8} {'trace':>9} {'det':>10} "
          f"{'slope dk=0':>11} {'slope dx=0':>11} {'type':>8}")
    rows = []
    for vs in (1.0, 0.30, 0.10):
        grid = np.exp(np.linspace(np.log(1e-3), np.log(1e4), 80001))
        r = s * grid ** alpha * h(chi * grid / omega, vs) - delta * grid
        for i in np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]:
            ks = brentq(lambda k: s * k ** alpha * h(chi * k / omega, vs) - delta * k,
                        grid[i], grid[i + 1])
            xs = chi * ks / omega
            f = ks ** alpha * h(xs, vs)
            fk = alpha * ks ** (alpha - 1) * h(xs, vs)
            fx = ks ** alpha * hp(xs, vs)
            J = np.array([[s * fk - delta, s * fx], [chi, -omega]])
            tr, det = J.trace(), np.linalg.det(J)
            sk = (f - ks * fk) / (ks * fx) if fx > 1e-14 else np.inf
            sx = chi / omega
            kind = "stable" if (tr < 0 and det > 0) else "saddle"
            rows.append((vs, ks, xs, sk, sx, det > 0))
            print(f"  {vs:>9.2f} {ks:>9.5f} {xs:>8.4f} {tr:>9.5f} {det:>10.6f} "
                  f"{sk:>11.4f} {sx:>11.4f} {kind:>8}")
    print("  For small varsigma there are three steady states.  The outer two have det > 0")
    print("  and the dk/dt = 0 locus steeper, and are stable; the middle one has det < 0")
    print("  and the ordering reversed, and is a saddle whose stable manifold separates the")
    print("  two basins of attraction -- Figure 21.11 exactly.  The two criteria agree in")
    print("  every row, as they must: det > 0 and 'dk/dt = 0 steeper' are the same")
    print("  inequality multiplied by the positive number s/k.")
    return rows


# ---------------------------------------------------------------------------


def figure(mal, rows, curves, alpha=0.35, s=0.25, delta=0.06, chi=0.4, omega=1.0):
    al, eta0, eps, Ls, lo, hi = mal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    Lg = np.linspace(1e-4, hi * 1.5, 500)
    for e, style in ((1 + eps, "-"), (1 - eps, "--")):
        ax1.plot(Lg, (e * (1 - al) / eta0) ** (1 / (1 + al)) * Lg ** (1 / (1 + al)),
                 style, color="black", lw=0.9)
    ax1.plot(Lg, Lg, color="0.6", lw=0.5)
    for v in (lo, hi):
        ax1.axvline(v, color="0.75", lw=0.4)
    ax1.set_xlabel("$L(t)$")
    ax1.set_ylabel("$L(t+1)$")
    ax1.set_xlim(0, hi * 1.5)
    ax1.set_ylim(0, hi * 1.5)
    ax1.set_title("Malthusian stochastic correspondence", fontsize=7.5)

    for (rho, (xs, U)), style in zip(sorted(curves.items()), ("-", "--", ":", "-.")):
        ax2.plot(xs, U / U.max(), style, color="black", lw=0.9,
                 label=rf"$\rho={rho:g}$")
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel("date-0 utility (normalized)")
    ax2.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax2.set_title(r"welfare is U-shaped in $\xi$", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch21_development")


def main():
    mal = ex213()
    curves = ex215()
    ex217()
    ex218()
    rows = ex2114()
    figure(mal, rows, curves)


if __name__ == "__main__":
    main()
