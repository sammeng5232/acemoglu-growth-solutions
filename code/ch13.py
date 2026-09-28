"""Numerical parts of the Chapter 13 solutions (expanding variety models).

  * Exercise 13.9   parameters for which the equilibrium satisfies the transversality condition
    while the social planner's problem is unbounded;
  * Exercise 13.11  research subsidies versus machine subsidies: the subsidy that matches the
    Pareto optimal growth rate, the welfare-maximizing (second-best) one, and the machine
    subsidy s_x = beta that decentralizes the optimum exactly;
  * Exercise 13.12  a corporate profit tax and the exponential divergence of relative income;
  * Exercise 13.14  competition policy: the growth-maximizing gamma is the largest one, but the
    welfare-maximizing gamma is interior and falls with the discount rate;
  * Exercise 13.17  the knowledge-spillover model: equilibrium against Pareto optimal growth;
  * Exercises 13.20 and 13.21  growth without scale effects: the permanent level scale effect,
    and the saddle-path dynamics from an arbitrary N(0);
  * Exercise 13.27  in the expanding product variety model the optimal growth rate is exactly
    eps times the equilibrium one.

Run with `python code/ch13.py`.
"""
import numpy as np
from scipy.optimize import brentq

from acemoglulib import plt, save

BETA, ETA, L, RHO, THETA = 0.5, 0.12, 1.0, 0.02, 2.0
MS = (1 - BETA) ** (-1 / BETA)          # the planner's machine-intensity factor


def welfare(C0, g, rho=RHO, theta=THETA):
    """int_0^inf e^{-rho t} (C0 e^{gt})^{1-theta}-1)/(1-theta) dt, finite iff rho > (1-theta)g."""
    if C0 <= 0 or rho <= (1 - theta) * g:
        return np.nan                       # infeasible, or the objective diverges
    if abs(theta - 1.0) < 1e-12:
        return np.log(C0) / rho + g / rho ** 2
    return C0 ** (1 - theta) / ((1 - theta) * (rho - (1 - theta) * g)) - 1 / ((1 - theta) * rho)


def ex139():
    print("Exercise 13.9: the equilibrium can be well posed when the optimum is not")
    th, rho = 0.5, 0.05
    eq = (1 - th) * ETA * BETA * L
    op = (1 - th) * ETA * MS * BETA * L
    print(f"  beta={BETA}, eta={ETA}, L={L}, theta={th}, rho={rho}")
    print(f"  equilibrium needs (1-theta) eta beta L < rho:  {eq:.4f} < {rho}  -> "
          f"{'holds' if eq < rho else 'fails'}")
    print(f"  optimum     needs (1-theta) eta (1-beta)^(-1/beta) beta L < rho: "
          f"{op:.4f} < {rho}  -> {'holds' if op < rho else 'FAILS'}")
    print(f"  so for rho in ({eq:.4f}, {op:.4f}] the market allocation has finite utility while")
    print(f"  the planner's objective diverges: with theta < 1 she can always do better by")
    print(f"  postponing consumption, and no optimum exists")


def ex1311():
    """Research subsidies, machine subsidies, and the second best."""
    A_mkt = BETA * (2 - BETA) * L / (1 - BETA)        # net output per unit of N, with markups
    A_opt = MS * BETA * L                             # net output per unit of N, at x = x^S
    g_eq = (ETA * BETA * L - RHO) / THETA
    g_opt = (ETA * MS * BETA * L - RHO) / THETA
    g_sb = (ETA * A_mkt - RHO) / THETA                # second best given the markup
    print(f"\nExercise 13.11: policy in the lab-equipment model "
          f"(beta={BETA}, eta={ETA}, L={L}, rho={RHO}, theta={THETA})")
    print(f"  equilibrium growth g* = {g_eq:.6f}")
    print(f"  Pareto optimal  gS   = {g_opt:.6f}   [(1-beta)^(-1/beta) = {MS:.4f}]")
    print(f"  second best     gSB  = {g_sb:.6f}   (research subsidy only, markup left in place)")
    sR_match = 1 - (1 - BETA) ** (1 / BETA)
    sR_sb = 1 / (2 - BETA)
    print(f"  the subsidy matching gS is sR = 1 - (1-beta)^(1/beta) = {sR_match:.6f}")
    print(f"  the welfare-maximizing subsidy is sR = 1/(2-beta) = {sR_sb:.6f} < {sR_match:.6f}:")
    print(f"    matching the first-best growth rate OVERSHOOTS when the markup is still there")
    print(f"  {'sR':>7} {'growth':>10} {'C(0)/N(0)':>11} {'welfare':>12}")
    best = (-np.inf, None)
    for s in (0.0, 0.25, 0.5, sR_sb, sR_match, 0.85):
        g = (ETA * BETA * L / (1 - s) - RHO) / THETA
        C0 = A_mkt - g / ETA
        W = welfare(C0, g)
        if not np.isnan(W):
            best = max(best, (W, s))
        print(f"  {s:>7.4f} {g:>10.6f} {C0:>11.6f} "
              + (f"{W:>12.6f}" if not np.isnan(W) else f"{'infeasible':>12}"))
    print(f"  the best of these is sR = {best[1]:.4f}, confirming the closed form")
    print(f"  a machine subsidy of s_x = beta = {BETA} instead restores x = x^S exactly:")
    print(f"    the monopolist still charges psi/(1-beta), but final good firms pay")
    print(f"    (1-beta) psi/(1-beta) = psi, so x = (1-beta)^(-1/beta) L = {MS*L:.4f} L")
    print(f"    profits become beta x^S, free entry gives r = eta beta (1-beta)^(-1/beta) L = "
          f"{ETA*BETA*MS*L:.6f},")
    print(f"    and the growth rate is exactly gS = {g_opt:.6f}: one instrument suffices")
    return g_eq, g_opt, g_sb, A_mkt, sR_sb, sR_match


def ex1312():
    print(f"\nExercise 13.12: a corporate profit tax")
    print(f"  r* = eta (1-tau) beta L, so g(tau) = [eta(1-tau) beta L - rho]/theta")
    print(f"  {'tau':>6} {'r*':>9} {'growth':>10} {'ratio vs tau=0 after 100 yrs':>30}")
    g0 = (ETA * BETA * L - RHO) / THETA
    for tau in (0.0, 0.1, 0.2, 0.3):
        g = (ETA * (1 - tau) * BETA * L - RHO) / THETA
        print(f"  {tau:>6.2f} {ETA*(1-tau)*BETA*L:>9.6f} {g:>10.6f} "
              f"{np.exp((g0-g)*100):>30.4f}")
    print(f"  Y(tau,t)/Y(tau',t) = exp[eta beta L (tau'-tau) t / theta]: the ratio diverges")
    print(f"  exponentially, so any tax difference produces unbounded income differences")


# ----------------------------------------------------------------------------------------
# Exercise 13.14: competition policy
# ----------------------------------------------------------------------------------------
def comp_policy(gamma, rho=RHO, theta=THETA):
    """Limit price gamma*psi with psi = 1-beta.  Returns (profit flow, net output per N, g, W)."""
    px = gamma * (1 - BETA)
    x = px ** (-1 / BETA) * L
    pi = (px - (1 - BETA)) * x
    net = px ** (-1 / BETA) * (gamma - 1 + BETA) * L      # net output per unit of N
    g = (ETA * pi - rho) / theta
    C0 = net - g / ETA
    return pi, net, g, (welfare(C0, g, rho, theta) if C0 > 0 else np.nan)


def ex1314():
    gmax = 1 / (1 - BETA)
    print(f"\nExercise 13.14: competition policy, fringe cost gamma*psi (gamma up to "
          f"1/(1-beta) = {gmax:.4f})")
    print(f"  {'gamma':>7} {'profit':>10} {'net Y/N':>10} {'growth':>10} {'welfare':>12}")
    for gam in (1.05, 1.2, 1.4, 1.6, 1.8, gmax):
        pi, net, g, W = comp_policy(gam)
        print(f"  {gam:>7.4f} {pi:>10.6f} {net:>10.6f} {g:>10.6f} {W:>12.6f}")
    gs = np.linspace(1.001, gmax, 4000)
    Ws = np.array([comp_policy(g)[3] for g in gs])
    gopt = gs[np.nanargmax(Ws)]
    print(f"  growth is maximized at gamma = 1/(1-beta) = {gmax:.4f} (profits peak exactly at")
    print(f"    the unconstrained monopoly price), but welfare is maximized at gamma = "
          f"{gopt:.4f}")
    print(f"  the optimal gamma falls as the household becomes more impatient:")
    print(f"  {'rho':>7} {'optimal gamma':>15} {'growth there':>13}")
    for rho in (0.01, 0.02, 0.04, 0.08):
        W2 = np.array([comp_policy(g, rho=rho)[3] for g in gs])
        go = gs[np.nanargmax(W2)]
        print(f"  {rho:>7.3f} {go:>15.4f} {comp_policy(go, rho=rho)[2]:>13.6f}")
    return gs, Ws, gmax, gopt


# ----------------------------------------------------------------------------------------
# Exercise 13.17: knowledge spillovers
# ----------------------------------------------------------------------------------------
def ex1317():
    LE = (THETA * ETA * L + RHO) / ((1 - BETA) * ETA + THETA * ETA)
    g_eq = ((1 - BETA) * ETA * LE - RHO) / THETA
    g_opt = (ETA * L - RHO) / THETA
    LR_opt = g_opt / ETA
    print(f"\nExercise 13.17: the knowledge spillover model of Section 13.2")
    print(f"  equilibrium: L_E* = {LE:.6f}, L_R* = {L-LE:.6f}, r* = "
          f"{(1-BETA)*ETA*LE:.6f}, g* = {g_eq:.6f}")
    print(f"  optimum:     the planner's costate gives d mu/mu = rho - eta L, so")
    print(f"               gS = (eta L - rho)/theta = {g_opt:.6f} and L_R^S = {LR_opt:.6f}")
    print(f"  gS > g* because the social return eta L = {ETA*L:.6f} exceeds the private")
    print(f"  return (1-beta) eta L_E* = {(1-BETA)*ETA*LE:.6f} on two counts: the markup")
    print(f"  factor (1-beta) = {1-BETA} and the knowledge spillover, which replaces L_E* by L")


# ----------------------------------------------------------------------------------------
# Exercises 13.20 and 13.21: growth without scale effects
# ----------------------------------------------------------------------------------------
PHI, N_POP = 0.5, 0.01


def jones_bgp(phi=PHI, n=N_POP, rho=RHO, theta=THETA):
    gN = n / (1 - phi)
    r = theta * gN + rho
    nE = (1 - phi) * (r - n) / (r * (1 - phi) + n * (phi - BETA))
    a = n / ((1 - phi) * (1 - nE))            # a = eta z*
    return gN, r, nE, a


def jones_rhs(z, nE, phi=PHI, n=N_POP, rho=RHO, theta=THETA):
    """(zdot, nEdot) with z = N^{phi-1} L and nE = L_E/L."""
    gN = ETA * z * (1 - nE)
    r = ETA * z * ((1 - phi) + (phi - BETA) * nE)
    return (z * (n - (1 - phi) * gN),
            nE * ((r - rho) / theta - gN))


def ex1320_1321():
    gN, r, nE, a = jones_bgp()
    zstar = a / ETA
    print(f"\nExercises 13.20 and 13.21: growth without scale effects "
          f"(phi={PHI}, n={N_POP})")
    print(f"  BGP: g_N* = n/(1-phi) = {gN:.6f}, r* = theta g_N* + rho = {r:.6f},")
    print(f"       production share L_E/L = {nE:.6f}, z* = N^(phi-1) L = {zstar:.6f}")
    print(f"  Exercise 13.20: N = [eta(1-beta)L_E/(r*-n)]^(1/(1-phi)), and L_E/L is the same in")
    print(f"    both countries, so y2/y1 = N2/N1 = (L2/L1)^(1/(1-phi)) at every date:")
    for ratio in (1.5, 2.0, 5.0):
        print(f"      a population ratio of {ratio:>4.1f} gives an income ratio of "
              f"{ratio**(1/(1-PHI)):>6.3f}, forever")
    print(f"    growth rates are equal, but the level scale effect never disappears")
    # Exercise 13.21: Jacobian of the planar system at the steady state
    h = 1e-6
    J = np.zeros((2, 2))
    for j, (dz, dn) in enumerate(((h, 0.0), (0.0, h))):
        hi = np.array(jones_rhs(zstar + dz, nE + dn))
        lo = np.array(jones_rhs(zstar - dz, nE - dn))
        J[:, j] = (hi - lo) / (2 * h)
    vals = np.linalg.eigvals(J)
    print(f"  Exercise 13.21: the system in (z, n_E) has Jacobian eigenvalues "
          f"{vals[0].real:+.6f} and {vals[1].real:+.6f}")
    print(f"    det = {np.linalg.det(J):+.6f} < 0, so the steady state is a saddle: z is")
    print(f"    predetermined by N(0) and n_E jumps onto the stable arm, giving a unique path")
    return zstar, nE


def ex1327(eps=3.0, eta=0.12, rho=0.02):
    LR = (eta * L - (eps - 1) * rho) / (eps * eta)
    g_eq = (eta * L - (eps - 1) * rho) / (eps * (eps - 1))
    LR_S = L - (eps - 1) * rho / eta
    g_S = eta * LR_S / (eps - 1)
    print(f"\nExercise 13.27: expanding product varieties (eps={eps}, eta={eta}, L={L}, "
          f"rho={rho})")
    print(f"  equilibrium: L_R* = {LR:.6f}, g* = {g_eq:.6f}")
    print(f"  optimum:     L_R^S = L - (eps-1) rho/eta = {LR_S:.6f}, gS = {g_S:.6f}")
    print(f"  gS/g* = {g_S/g_eq:.6f} = eps exactly: the planner values a new variety at its full")
    print(f"  love-of-variety contribution, while the innovator captures only the fraction 1/eps")
    print(f"  of revenue that the markup eps/(eps-1) leaves as profit")
    print(f"  utility is finite with log preferences for any rho > 0, and the transversality")
    print(f"  condition holds because r* = g* + rho > g*")


def figure(gs, Ws, gmax, gopt, A_mkt, g_eq, g_opt, g_sb, sR_sb, sR_match, zstar, nEstar):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    for rho, col, ls in ((0.01, "#666666", ":"), (0.02, "#003399", "-"),
                         (0.06, "#993300", "--")):
        W = np.array([comp_policy(g, rho=rho)[3] for g in gs])
        ax.plot(gs, (W - np.nanmin(W)) / (np.nanmax(W) - np.nanmin(W)), color=col, ls=ls,
                lw=1.1, label=rf"$\rho={rho}$")
        ax.plot([gs[np.nanargmax(W)]], [1.0], "o", ms=3, color=col)
    ax.axvline(gmax, color="#cccccc", lw=0.8)
    ax.annotate(r"$\gamma$ maximizing growth", (gmax, 0.05), textcoords="offset points",
                xytext=(-96, 0), fontsize=6.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("welfare (normalized)")
    ax.set_xlim(1.0, gmax)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[1]
    ss = np.linspace(0.0, 0.92, 400)
    W = []
    for s in ss:
        g = (ETA * BETA * L / (1 - s) - RHO) / THETA
        C0 = A_mkt - g / ETA
        W.append(welfare(C0, g) if C0 > 0 else np.nan)
    W = np.array(W)
    ax.plot(ss, W, color="#003399", lw=1.2)
    for s, lab, col in ((sR_sb, "welfare", "#003399"), (sR_match, r"$g=g^S$", "#993300")):
        ax.axvline(s, color=col, lw=0.8, ls="--")
        ax.annotate(lab, (s, np.nanmin(W)), textcoords="offset points", xytext=(-24, 6),
                    fontsize=6.5, color=col)
    ax.set_xlabel("research subsidy $s_R$")
    ax.set_ylabel("welfare")
    ax.set_xlim(0, 0.92)
    save(fig, "ch13_expanding_variety")


def main():
    ex139()
    g_eq, g_opt, g_sb, A_mkt, sR_sb, sR_match = ex1311()
    ex1312()
    gs, Ws, gmax, gopt = ex1314()
    ex1317()
    zstar, nEstar = ex1320_1321()
    ex1327()
    figure(gs, Ws, gmax, gopt, A_mkt, g_eq, g_opt, g_sb, sR_sb, sR_match, zstar, nEstar)


if __name__ == "__main__":
    main()
