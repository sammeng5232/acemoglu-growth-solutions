"""Numerical parts of the Chapter 14 solutions (Schumpeterian growth).

  * Exercise 14.4   the BGP of Section 14.1: uniqueness, positivity, and the equivalence of the
    transversality condition with rho > (1-theta)(lambda-1) eta beta L;
  * Exercises 14.9 and 14.10  the drastic threshold, the limit price when it fails, and the
    growth rate lost to competition from the previous vintage;
  * Exercise 14.11  Schumpeterian growth without scale effects: g* = n/(phi-1);
  * Exercise 14.17  the one-sector Aghion-Howitt model, and why its growth rate carries log
    lambda rather than lambda - 1;
  * Exercise 14.25  the planner's problem of Section 14.3: entry is always excessive, but growth
    may be higher or lower than in equilibrium;
  * Exercise 14.28  the stationary distribution of relative profits is geometric in log lambda,
    that is, Pareto in levels.

Run with `python code/ch14.py`.
"""
import numpy as np

from acemoglulib import plt, save

BETA, ETA, L, RHO, THETA = 0.5, 0.12, 1.0, 0.02, 2.0


def lam_drastic(beta=BETA):
    """(14.5): lambda >= (1/(1-beta))^{(1-beta)/beta}."""
    return (1 / (1 - beta)) ** ((1 - beta) / beta)


def bgp_141(lam, eta=ETA, beta=BETA, L=L, rho=RHO, theta=THETA):
    """(14.23) and the associated r*, z*."""
    g = (lam * eta * beta * L - rho) / (theta + 1 / (lam - 1))
    return g, theta * g + rho, g / (lam - 1)


def gS_141(lam, eta=ETA, beta=BETA, L=L, rho=RHO, theta=THETA):
    """(14.27): the Pareto optimal growth rate."""
    return (eta * (lam - 1) * (1 - beta) ** (-1 / beta) * beta * L - rho) / theta


def ex144():
    lam = 2.0
    g, r, z = bgp_141(lam)
    print(f"Exercise 14.4: the BGP of Section 14.1 (beta={BETA}, eta={ETA}, L={L}, "
          f"rho={RHO}, theta={THETA}, lambda={lam})")
    print(f"  drastic threshold (14.5): lambda >= {lam_drastic():.6f}  -> "
          f"{'satisfied' if lam >= lam_drastic() else 'VIOLATED'}")
    print(f"  g* = {g:.6f}, r* = theta g* + rho = {r:.6f}, z* = g*/(lambda-1) = {z:.6f}")
    print(f"  check r* + z* = lambda eta beta L: {r+z:.6f} vs {lam*ETA*BETA*L:.6f}")
    print(f"  positive growth needs lambda eta beta L > rho: "
          f"{lam*ETA*BETA*L:.6f} > {RHO}")
    print(f"  transversality needs g* < r*, equivalently (1-theta)(lambda-1) eta beta L < rho:")
    print(f"    g* = {g:.6f} < r* = {r:.6f}, and "
          f"{(1-THETA)*(lam-1)*ETA*BETA*L:+.6f} < {RHO}")
    A = BETA * (2 - BETA) * L / (1 - BETA)
    print(f"  consumption C/Q = beta(2-beta)L/(1-beta) - z*/eta = {A:.6f} - "
          f"{z/ETA:.6f} = {A - z/ETA:.6f} > 0")


def ex1410():
    lam_star = lam_drastic()
    print(f"\nExercises 14.9 and 14.10: the drastic threshold is lambda* = "
          f"(1/(1-beta))^((1-beta)/beta) = {lam_star:.6f}")
    print(f"  a final good producer ranks machines by q/p^(1-beta); comparing quality q at the")
    print(f"  monopoly price psi q/(1-beta) with quality q/lambda at marginal cost psi q/lambda")
    print(f"  gives exactly lambda >= (1-beta)^(-(1-beta)/beta)")
    print(f"  when it fails the limit price is p = lambda^(beta/(1-beta)) psi q, so")
    print(f"  {'lambda':>8} {'regime':>9} {'profit/q':>10} {'g*':>10} {'g* if no rival':>16}")
    for lam in (1.2, 1.5, lam_star, 2.5, 4.0):
        if lam < lam_star - 1e-12:
            Lam = lam ** (BETA / (1 - BETA))
            pi = (Lam - 1) * Lam ** (-1 / BETA) * (1 - BETA) ** (-(1 - BETA) / BETA) * L
            reg = "limit"
        else:
            pi = BETA * L
            reg = "monopoly"
        g = (lam * ETA * pi - RHO) / (THETA + 1 / (lam - 1))
        g_unc = (lam * ETA * BETA * L - RHO) / (THETA + 1 / (lam - 1))
        print(f"  {lam:>8.4f} {reg:>9} {pi:>10.6f} {g:>10.6f} {g_unc:>16.6f}")
    print("  the last column is part (d): removing the previous vintage lets the monopolist")
    print("  charge its ideal markup, which raises profits, the value of a patent and growth")


def ex1411(phi=2.0, n=0.01):
    print(f"\nExercise 14.11: no scale effects with flow rate eta Z/q^phi, phi = {phi} > 1, "
          f"n = {n}")
    print(f"  free entry requires eta lambda V(q) = q^phi with V(q) proportional to q L(t),")
    print(f"  so L(t) must grow at the same rate as Q(t)^(phi-1): n = (phi-1) g_Q")
    print(f"  g* = n/(phi-1) = {n/(phi-1):.6f}, independent of L, eta, beta and lambda")
    print(f"  {'phi':>6} {'g* = n/(phi-1)':>16}")
    for p in (1.25, 1.5, 2.0, 3.0):
        print(f"  {p:>6.2f} {n/(p-1):>16.6f}")
    print("  as in Section 13.3, the scale effect moves from the growth rate to the level")


def ex1417(lam=2.0, eta=0.15, beta=BETA, L=1.0, rho=0.05):
    """One-sector Aghion-Howitt: risk-neutral households, r = rho."""
    LR = (lam * (1 - beta) * eta * L - rho) / (eta * (1 + lam * (1 - beta)))
    g = eta * LR * np.log(lam)
    print(f"\nExercise 14.17: the one-sector model (lambda={lam}, eta={eta}, beta={beta}, "
          f"L={L}, rho={rho})")
    print(f"  free entry eta V(lambda q) = w(q) gives L_R* = "
          f"[lambda(1-beta) eta L - rho] / [eta(1 + lambda(1-beta))] = {LR:.6f}")
    print(f"  average growth g* = eta L_R* log lambda = {g:.6f}")
    print(f"  positive research needs lambda(1-beta) eta L > rho: "
          f"{lam*(1-beta)*eta*L:.6f} > {rho}")
    print(f"  finite utility needs rho > g*: {rho} > {g:.6f}, that is")
    print(f"    lambda(1-beta) eta L < rho [1 + (1+lambda(1-beta))/log lambda] = "
          f"{rho*(1+(1+lam*(1-beta))/np.log(lam)):.6f}")
    print(f"  the expected interval between innovations is 1/(eta L_R*) = {1/(eta*LR):.4f}")
    print(f"    (Exercise 14.16: output is constant between Poisson arrivals)")
    print(f"  log lambda = {np.log(lam):.6f} against lambda - 1 = {lam-1:.6f}: the one-sector")
    print(f"  model's quality jumps multiplicatively at Poisson dates, so it is E[log q] that")
    print(f"  grows, at rate z log lambda; the continuum model averages across sectors, so it")
    print(f"  is E[q] that grows, at rate z(lambda-1) > z log lambda by Jensen's inequality")


# ----------------------------------------------------------------------------------------
# Exercise 14.25: incumbents and entrants, equilibrium against the planner
# ----------------------------------------------------------------------------------------
def sec143(phi, lam, kap, eta0, gam, beta=BETA, L=1.0, rho=RHO, theta=THETA):
    """eta(z) = eta0 z^{-gam}, so z eta(z) = eta0 z^{1-gam} is increasing and concave."""
    chi = (1 - beta) ** (-1 / beta)
    # equilibrium: eta(zhat*) = phi(lambda-1)/kappa
    zhat_e = (eta0 * kap / (phi * (lam - 1))) ** (1 / gam)
    flow_e = eta0 * zhat_e ** (1 - gam)                      # zhat eta(zhat)
    g_e = (phi * (lam - 1) * beta * L - flow_e - rho) / theta
    z_e = (g_e - (kap - 1) * flow_e) / ((lam - 1) * phi)     # incumbent R&D, from (14.52)
    # planner: (lambda-1) phi = (kappa-1) d[z eta(z)]/dz = (kappa-1)(1-gam) eta(z)
    zhat_s = (eta0 * (kap - 1) * (1 - gam) / (phi * (lam - 1))) ** (1 / gam)
    flow_s = eta0 * zhat_s ** (1 - gam)
    g_s = ((lam - 1) * phi * (chi * beta * L - zhat_s) + (kap - 1) * flow_s - rho) / theta
    return zhat_e, g_e, z_e, zhat_s, g_s


def ex1425():
    print(f"\nExercise 14.25: the planner's problem of Section 14.3 "
          f"(eta(z) = eta0 z^-gamma)")
    print(f"  the planner sets (lambda-1) phi = (kappa-1)(1-gamma) eta(zhat), the equilibrium")
    print(f"  sets (lambda-1) phi = kappa eta(zhat): since (kappa-1)(1-gamma) < kappa always,")
    print(f"  the planner ALWAYS chooses less entry than the market")
    print(f"  {'phi':>6} {'lam':>6} {'kap':>6} {'eta0':>6} {'gam':>5} "
          f"{'zhat*':>8} {'zhat^S':>8} {'g*':>9} {'gS':>9}")
    for phi, lam, kap, eta0, gam in ((0.6, 1.2, 2.0, 0.0219, 0.5),
                                     (0.6, 1.2, 3.0, 0.0150, 0.5),
                                     (0.9, 1.3, 2.0, 0.0300, 0.5),
                                     (0.6, 1.2, 2.0, 0.0219, 0.7)):
        zh_e, g_e, z_e, zh_s, g_s = sec143(phi, lam, kap, eta0, gam)
        print(f"  {phi:>6.2f} {lam:>6.2f} {kap:>6.2f} {eta0:>6.2f} {gam:>5.2f} "
              f"{zh_e:>8.4f} {zh_s:>8.4f} {g_e:>9.5f} {g_s:>9.5f}"
              + ("   gS > g*" if g_s > g_e else "   gS < g*"))
    print("  gS > g* in every case, and in fact always: writing f(z) = z eta(z),")
    print("    theta(gS - g*) = (lam-1) phi beta L (chi-1) + (kap-1)[f(zS) - zS f'(zS)] + f(z*)")
    print("  the first term is positive because chi > 1, the second because f is concave with")
    print("  f(0) = 0, and the third trivially.  The planner's lower entry does not outweigh")
    print("  her freedom from the markup.")


def ex1428(phi=0.6, lam=1.2, kap=2.0, eta0=0.0219, gam=0.5):
    zh, g, z, _, _ = sec143(phi, lam, kap, eta0, gam)
    inc, ent = phi * z, zh * (eta0 * zh ** (-gam))
    p = inc / (inc + ent)
    print(f"\nExercise 14.28: firm dynamics with the Section 14.1 production function")
    print(f"  with cost psi q, x = L for every firm: sales are identical whatever the quality,")
    print(f"  so there are NO firm-size dynamics, while profits beta q L still scale with q")
    print(f"  incumbent innovation rate phi z* = {inc:.6f}, replacement rate "
          f"zhat eta(zhat) = {ent:.6f}")
    print(f"  the number of incremental steps a surviving firm has taken is geometric with")
    print(f"  parameter p = {p:.6f}, so relative profits are lambda^k with")
    print(f"  {'k':>3} {'Pr[n=k]':>10} {'profit ratio':>13}")
    for k in range(6):
        print(f"  {k:>3} {(1-p)*p**k:>10.6f} {lam**k:>13.4f}")
    zeta = np.log(1 / p) / np.log(lam)
    print(f"  the tail is Pareto with index zeta = log(1/p)/log lambda = {zeta:.4f}:")
    print(f"    Pr[profit >= lambda^k] = p^k = (lambda^k)^(-zeta)")


def ex_prop143():
    """Proposition 14.3 asserts equilibrium growth may exceed the optimum.  It cannot, in the
    model of Section 14.1 under condition (14.5)."""
    th, be, lam, et, Ll, rh = 1.0, 0.9, 1.3, 1.0, 1.0, 0.38
    g = (lam * et * be * Ll - rh) / (th + 1 / (lam - 1))
    gS = (et * (lam - 1) * (1 - be) ** (-1 / be) * be * Ll - rh) / th
    gS_alt = (et * (lam - 1) * (1 - be) ** (-(1 - be) / be) * be * Ll - rh) / th
    print(f"\nProposition 14.3: the text's example (theta={th}, beta={be}, lambda={lam}, "
          f"eta={et}, L={Ll}, rho={rh})")
    print(f"  drastic threshold {(1/(1-be))**((1-be)/be):.6f} <= lambda = {lam}: satisfied")
    print(f"  g*  = {g:.6f}")
    print(f"  gS  = {gS:.6f} using (14.27) as printed, with (1-beta)^(-1/beta) = "
          f"{(1-be)**(-1/be):.4f}")
    print(f"  gS  = {gS_alt:.6f} if (1-beta)^(-(1-beta)/beta) = "
          f"{(1-be)**(-(1-be)/be):.4f} is used instead")
    print(f"  the text reports gS approximately 0, which is the second number; but (14.25)")
    print(f"  gives net output beta (1-beta)^(-1/beta) Q L, so (14.27) as printed is correct")
    print(f"  and the example does not deliver excessive growth")
    print(f"  a grid search over the whole admissible parameter space:")
    found, worst = 0, -np.inf
    for be2 in np.linspace(0.02, 0.98, 97):
        c = (1 - be2) ** (-1 / be2)
        ls = (1 / (1 - be2)) ** ((1 - be2) / be2)
        for lam2 in np.linspace(ls, 20 * ls, 300):
            for th2 in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
                for rho2 in np.linspace(1e-4, 0.999 * lam2, 200):
                    gg = (lam2 - rho2) / (th2 + 1 / (lam2 - 1))
                    gg_s = ((lam2 - 1) * c - rho2) / th2
                    if gg <= 0 or rho2 <= (1 - th2) * (lam2 - 1):
                        continue
                    if rho2 <= (1 - th2) * (lam2 - 1) * c:
                        continue
                    if gg > gg_s:
                        found += 1
                    worst = max(worst, gg / gg_s)
    print(f"    (eta beta L normalised to 1; lambda >= lambda*, g* > 0, both transversality")
    print(f"     conditions imposed)  cases with g* > gS: {found};  max g*/gS = {worst:.4f}")
    print(f"  so under (14.5) the equilibrium ALWAYS grows more slowly than the optimum: the")
    print(f"  appropriability effect, worth a factor chi = (1-beta)^(-1/beta) >= e, dominates")
    print(f"  the business stealing effect at every admissible parameter value")


def figure():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    lams = np.linspace(1.05, 6.0, 400)
    ge = np.array([bgp_141(l)[0] for l in lams])
    gs = np.array([gS_141(l) for l in lams])
    ax.plot(lams, ge, color="#003399", lw=1.2, label="equilibrium $g^*$")
    ax.plot(lams, gs, color="#993300", ls="--", lw=1.2, label="optimum $g^S$")
    ax.axvline(lam_drastic(), color="#cccccc", lw=0.8)
    ax.annotate(r"$\lambda^*$", (lam_drastic(), 0.0), textcoords="offset points",
                xytext=(3, 4), fontsize=7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("growth rate")
    ax.set_xlim(1.0, 6.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    eta0s = np.linspace(0.002, 0.030, 300)
    ge2, gs2 = [], []
    for e0 in eta0s:
        _, g_e, _, _, g_s = sec143(0.6, 1.2, 2.0, e0, 0.5)
        ge2.append(g_e)
        gs2.append(g_s)
    ax.plot(eta0s, ge2, color="#003399", lw=1.2, label="equilibrium $g^*$")
    ax.plot(eta0s, gs2, color="#993300", ls="--", lw=1.2, label="optimum $g^S$")
    ax.axhline(0.0, color="#cccccc", lw=0.5)
    ax.set_xlabel(r"entry productivity $\eta_0$")
    ax.set_ylabel("growth rate")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "ch14_schumpeterian")


def main():
    ex144()
    ex1410()
    ex1411()
    ex1417()
    ex1425()
    ex1428()
    ex_prop143()
    figure()


if __name__ == "__main__":
    main()
