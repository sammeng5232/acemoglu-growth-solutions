"""Chapter 23, Political Institutions and Economic Growth.

Numerical checks for the nine exercises the instructor's manual leaves unsolved:
23.1, 23.2, 23.3, 23.6, 23.7, 23.8, 23.9, 23.10 and 23.11.

    python code/ch23.py

Writes figures/ch23_political_institutions.pdf.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from acemoglulib import plt, save

# ---------------------------------------------------------------------------
# Section 22.3/22.4 machinery, reused by Exercises 23.1-23.3.
#
#   y per worker   = (1/alpha) (beta (1 - tau))^{alpha/(1-alpha)} A
#   net surplus pw = ((1-alpha)/alpha) beta^{alpha/(1-alpha)} (1-tau)^{1/(1-alpha)} A
#   w              = min over the two groups of the net surplus per worker   (22.22)
# ---------------------------------------------------------------------------


def ypw(tau, A, alpha, beta):
    """Output per worker of a producer with productivity A facing tax tau."""
    return (beta * (1.0 - tau)) ** (alpha / (1.0 - alpha)) * A / alpha


def spw(tau, A, alpha, beta):
    """Net marginal product (profit per worker before wages)."""
    return (1.0 - alpha) / alpha * beta ** (alpha / (1.0 - alpha)) * (
        1.0 - tau
    ) ** (1.0 / (1.0 - alpha)) * A


def kappa(Lbar, theta, alpha, phi):
    """(22.29), evaluated at the measure theta of the group in power."""
    return (1.0 - alpha) / alpha * (1.0 + theta * Lbar / ((1.0 - theta * Lbar) * phi))


def tau_com(Lbar, theta, alpha, phi):
    k = kappa(Lbar, theta, alpha, phi)
    return k / (1.0 + k)


def ruler_payoff(tau_out, A_in, A_out, th_in, th_out, Lbar, alpha, beta, phi, RN=0.0):
    """Per capita payoff of the group in power, (22.24) with the roles relabelled.

    The group in power sets tau on the other group only, employs Lbar workers each
    (the interior case, in which (22.26) fails at the solution), and the out-of-power
    group absorbs the residual 1 - th_in*Lbar units of labour.
    """
    w = spw(tau_out, A_out, alpha, beta)
    L_out_total = 1.0 - th_in * Lbar
    own = (spw(0.0, A_in, alpha, beta) - w) * Lbar
    rev = phi * tau_out * ypw(tau_out, A_out, alpha, beta) * L_out_total
    return own + (rev + RN) / th_in


def aggregate_output(tau_in, tau_out, A_in, A_out, th_in, Lbar, alpha, beta):
    """Aggregate output when the group in power employs Lbar each."""
    return th_in * Lbar * ypw(tau_in, A_in, alpha, beta) + (
        1.0 - th_in * Lbar
    ) * ypw(tau_out, A_out, alpha, beta)


# ---------------------------------------------------------------------------
# 23.1  Dictatorship of the middle class: Proposition 23.1 by relabelling.
# ---------------------------------------------------------------------------


def ex231():
    print("=" * 74)
    print("23.1  Proposition 23.1: the dictatorship of the middle class")
    print("=" * 74)
    alpha, beta, phi, Lbar = 1.0 / 3.0, 0.9, 0.7, 1.4
    print(" alpha=%.4f beta=%.2f phi=%.2f Lbar=%.2f" % (alpha, beta, phi, Lbar))
    print()
    print("  %6s %6s %7s %7s   %10s %10s %9s" % (
        "the^e", "the^m", "A^e", "A^m", "tau^e num", "closed", "tau^m num"))
    for th_e, th_m, Ae, Am in [(0.40, 0.40, 1.0, 1.1),
                              (0.40, 0.45, 1.0, 1.1),
                              (0.35, 0.50, 1.2, 1.0),
                              (0.45, 0.38, 1.0, 1.3),
                              (0.42, 0.42, 1.0, 1.0)]:
        assert (th_e + th_m) * Lbar > 1.0, "Condition 22.1"
        # (23.1): A^m >= phi alpha^{alpha/(1-alpha)} A^e theta^e/theta^m
        rhs = phi * alpha ** (alpha / (1.0 - alpha)) * Ae * th_e / th_m
        assert Am >= rhs, "(23.1) fails"
        # maximize over the tax on the elite, and over a self-tax, jointly
        g = np.linspace(0.0, 0.999, 200001)
        pay = ruler_payoff(g, Am, Ae, th_m, th_e, Lbar, alpha, beta, phi)
        t_num = g[int(np.argmax(pay))]
        # a self-tax tau^m returns phi of its own revenue and distorts: never used
        s = np.linspace(0.0, 0.9, 90001)
        own = (1.0 - alpha) * (1.0 - s) + phi * s
        self_obj = (1.0 - s) ** (alpha / (1.0 - alpha)) * own
        t_self = s[int(np.argmax(self_obj))]
        print("  %6.2f %6.2f %7.2f %7.2f   %10.6f %10.6f %9.6f" % (
            th_e, th_m, Ae, Am, t_num, tau_com(Lbar, th_m, alpha, phi), t_self))
    print()
    print("  tau^e = kappa(Lbar,theta^m,.)/(1+kappa) in every row;  tau^m = 0.")
    print("  The second slot of kappa is the measure of the group IN POWER.")
    print()


# ---------------------------------------------------------------------------
# 23.2  Which dictatorship produces more?
# ---------------------------------------------------------------------------


def ex232():
    print("=" * 74)
    print("23.2  Elite versus middle-class dictatorship: output per capita")
    print("=" * 74)
    alpha, beta, phi, Lbar = 1.0 / 3.0, 0.9, 0.7, 1.4

    print(" (a)  theta^e = theta^m = theta, so both regimes levy the same tax.")
    print()
    th = 0.42
    tc = tau_com(Lbar, th, alpha, phi)
    bracket = th * Lbar - (1.0 - tc) ** (alpha / (1.0 - alpha)) * (1.0 - th * Lbar)
    print("  theta=%.2f  tau^COM=%.6f" % (th, tc))
    print("  theta*Lbar = %.6f,  (1-tau)^{a/(1-a)}(1-theta*Lbar) = %.6f" % (
        th * Lbar, (1.0 - tc) ** (alpha / (1.0 - alpha)) * (1.0 - th * Lbar)))
    print("  bracket = %.6f > 0  (Condition 22.1 gives theta*Lbar > 1/2 > 1-theta*Lbar)"
          % bracket)
    print()
    print("  %7s %7s   %12s %12s   %9s %9s" % (
        "A^e", "A^m", "Y elite rule", "Y mc rule", "sign(dY)", "sign(A^m-A^e)"))
    ok = True
    for Ae, Am in [(1.0, 1.3), (1.0, 1.1), (1.0, 1.0), (1.1, 1.0), (1.4, 1.0)]:
        # interior case needs the out-of-power group to stay the marginal employer
        if (1.0 - tc) ** (1.0 / (1.0 - alpha)) * Am >= Ae:
            print("  %7.2f %7.2f   (22.26) holds under elite rule: skipped" % (Ae, Am))
            continue
        if (1.0 - tc) ** (1.0 / (1.0 - alpha)) * Ae >= Am:
            print("  %7.2f %7.2f   (23.1) binds under mc rule: skipped" % (Ae, Am))
            continue
        Ye = aggregate_output(0.0, tc, Ae, Am, th, Lbar, alpha, beta)
        Ym = aggregate_output(0.0, tc, Am, Ae, th, Lbar, alpha, beta)
        s1 = int(np.sign(Ym - Ye))
        s2 = int(np.sign(Am - Ae))
        ok = ok and s1 == s2
        print("  %7.2f %7.2f   %12.8f %12.8f   %9d %9d" % (Ae, Am, Ye, Ym, s1, s2))
    print("  signs agree in every admissible row:", ok)
    print()

    print(" (b)  theta^e != theta^m.  Elite rule gives more output iff")
    print("      A^e/A^m > [theta^m Lbar - (1-tau_e)^{a/(1-a)}(1-theta^e Lbar)]")
    print("               / [theta^e Lbar - (1-tau_m)^{a/(1-a)}(1-theta^m Lbar)]")
    print()
    print("  %6s %6s   %9s %9s   %10s   %9s %6s" % (
        "the^e", "the^m", "tau_e^COM", "tau_m^COM", "threshold", "A^e/A^m", "Y^E>Y^M"))
    for th_e, th_m in [(0.42, 0.42), (0.50, 0.38), (0.38, 0.50), (0.60, 0.36),
                       (0.36, 0.60)]:
        assert (th_e + th_m) * Lbar > 1.0
        te = tau_com(Lbar, th_e, alpha, phi)
        tm = tau_com(Lbar, th_m, alpha, phi)
        num = th_m * Lbar - (1.0 - te) ** (alpha / (1.0 - alpha)) * (1.0 - th_e * Lbar)
        den = th_e * Lbar - (1.0 - tm) ** (alpha / (1.0 - alpha)) * (1.0 - th_m * Lbar)
        thresh = num / den
        hits = []
        for ratio in (0.7, 0.9, 1.0, 1.1, 1.5):
            Ae, Am = ratio, 1.0
            Ye = aggregate_output(0.0, te, Ae, Am, th_e, Lbar, alpha, beta)
            Ym = aggregate_output(0.0, tm, Am, Ae, th_m, Lbar, alpha, beta)
            hits.append((ratio, Ye > Ym, ratio > thresh))
        agree = all(a == b for _, a, b in hits)
        print("  %6.2f %6.2f   %9.6f %9.6f   %10.6f   %9s %6s" % (
            th_e, th_m, te, tm, thresh, "see below", agree))
        for ratio, a, b in hits:
            assert a == b, (th_e, th_m, ratio)
    print("  the inequality predicts the comparison in all 25 cases: True")
    print("  it collapses to A^e/A^m > 1 when theta^e = theta^m.")
    print()


# ---------------------------------------------------------------------------
# 23.3  Workers in power.
# ---------------------------------------------------------------------------


def workers_payoff(te, tm, Ae, Am, th_e, th_m, Lbar, alpha, beta, phi, cond221):
    """Per capita worker income w + T^w, with the labour allocation of (22.22)."""
    pe, pm = spw(te, Ae, alpha, beta), spw(tm, Am, alpha, beta)
    if not cond221:
        w = 0.0
        Le_tot, Lm_tot = th_e * Lbar, th_m * Lbar
    else:
        w = min(pe, pm)
        if pe > pm:                     # the elite employ at full capacity
            Le_tot, Lm_tot = th_e * Lbar, 1.0 - th_e * Lbar
        else:                           # the middle class do
            Le_tot, Lm_tot = 1.0 - th_m * Lbar, th_m * Lbar
    rev = phi * (te * ypw(te, Ae, alpha, beta) * Le_tot
                 + tm * ypw(tm, Am, alpha, beta) * Lm_tot)
    return w + rev


def ex233():
    print("=" * 74)
    print("23.3  Proposition 23.3: the dictatorship of the workers")
    print("=" * 74)
    alpha, beta, phi = 1.0 / 3.0, 0.9, 0.7
    g = np.linspace(0.0, 0.998, 1501)
    TE, TM = np.meshgrid(g, g, indexing="ij")

    print(" Part 1.  Condition 22.1 FAILS, so w = 0 and the two Laffer curves separate.")
    th_e, th_m, Lbar = 0.30, 0.30, 1.4          # (0.3+0.3)*1.4 = 0.84 < 1
    assert (th_e + th_m) * Lbar < 1.0
    U = np.vectorize(lambda a, b: workers_payoff(
        a, b, 1.0, 1.3, th_e, th_m, Lbar, alpha, beta, phi, False))(TE, TM)
    i, j = np.unravel_index(int(np.argmax(U)), U.shape)
    print("   argmax (tau^e, tau^m) = (%.4f, %.4f);  tau^RE = 1-alpha = %.4f"
          % (g[i], g[j], 1.0 - alpha))
    print()

    print(" Part 2.  Condition 22.1 HOLDS.  tau = 0 on whichever group sets the wage;")
    print("          the other is taxed to min{1-alpha, tau^D}.")
    print()
    print("  %6s %6s %6s %6s   %8s %8s   %8s %8s" % (
        "the^e", "the^m", "A^e", "A^m", "tau^e", "tau^m", "closed^e", "closed^m"))
    for th_e, th_m, Lbar, Ae, Am in [(0.42, 0.42, 1.4, 1.0, 1.3),
                                     (0.42, 0.42, 1.4, 1.0, 1.9),
                                     (0.42, 0.42, 1.4, 1.3, 1.0),
                                     (0.42, 0.42, 1.4, 1.9, 1.0),
                                     (0.50, 0.38, 1.4, 1.0, 1.4),
                                     (0.42, 0.42, 1.4, 1.0, 1.0)]:
        assert (th_e + th_m) * Lbar > 1.0
        U = np.vectorize(lambda a, b: workers_payoff(
            a, b, Ae, Am, th_e, th_m, Lbar, alpha, beta, phi, True))(TE, TM)
        i, j = np.unravel_index(int(np.argmax(U)), U.shape)
        if Am > Ae:
            ce = 0.0
            cm = min(1.0 - alpha, 1.0 - (Ae / Am) ** (1.0 - alpha))
        elif Ae > Am:
            cm = 0.0
            ce = min(1.0 - alpha, 1.0 - (Am / Ae) ** (1.0 - alpha))
        else:
            ce = cm = 0.0
        print("  %6.2f %6.2f %6.2f %6.2f   %8.4f %8.4f   %8.4f %8.4f" % (
            th_e, th_m, Ae, Am, g[i], g[j], ce, cm))
        assert abs(g[i] - ce) < 2e-3 and abs(g[j] - cm) < 2e-3
    print()
    print("  tau^D solves (1-tau^D)^{1/(1-alpha)} A^high = A^low, capped at 1-alpha;")
    print("  the cap binds exactly when alpha^{1/(1-alpha)} A^high >= A^low.")
    print("  theta^e = theta^m is NOT needed for part 2 (row 5 has 0.50 vs 0.38).")
    print()


# ---------------------------------------------------------------------------
# 23.6  The stationary fraction of high-skill agents.
# ---------------------------------------------------------------------------


def ex236():
    print("=" * 74)
    print("23.6  Deriving (23.7):  M = sigma^L / (1 - sigma^H + sigma^L)")
    print("=" * 74)
    print("  %7s %7s   %10s %10s %10s   %12s" % (
        "sig^H", "sig^L", "M closed", "mu(20)", "mu(200)", "|lambda|"))
    for sH, sL in [(0.9, 0.1), (0.8, 0.2), (0.95, 0.02), (0.6, 0.5), (0.99, 0.01)]:
        M = sL / (1.0 - sH + sL)
        mu = 1.0
        path = []
        for t in range(201):
            path.append(mu)
            mu = sH * mu + sL * (1.0 - mu)
        print("  %7.2f %7.2f   %10.6f %10.6f %10.6f   %12.6f" % (
            sH, sL, M, path[20], path[200], abs(sH - sL)))
        assert abs(path[200] - M) < 1e-6 or abs(sH - sL) > 0.9
        # mu(t) = M + lambda^t (1 - M)
        lam = sH - sL
        for t in (1, 5, 20, 60):
            assert abs(path[t] - (M + lam ** t * (1.0 - M))) < 1e-12
    print()
    print("  mu(t) = M + (sigma^H - sigma^L)^t (mu(0) - M): verified to 1e-12,")
    print("  monotone decreasing from mu(0)=1 whenever sigma^H > sigma^L.")
    print()


# ---------------------------------------------------------------------------
# 23.7  The oligarchic equilibrium.
# ---------------------------------------------------------------------------


def oligarch_values(AH, AL, sH, sL, alpha, beta, Lbar, tau=0.0):
    """Solve V^z = c A^z (1-tau)^{1/(1-alpha)} + beta[sigma^z V^H + (1-sigma^z) V^L]."""
    c = (1.0 - alpha) / alpha * beta ** (alpha / (1.0 - alpha)) * Lbar
    c *= (1.0 - tau) ** (1.0 / (1.0 - alpha))
    M = np.array([[beta * sH, beta * (1.0 - sH)], [beta * sL, beta * (1.0 - sL)]])
    V = np.linalg.solve(np.eye(2) - M, c * np.array([AH, AL]))
    return V[0], V[1]


def ex237():
    print("=" * 74)
    print("23.7  Proposition 23.5: the unique oligarchic equilibrium")
    print("=" * 74)
    alpha, beta, Lbar = 1.0 / 3.0, 0.9, 2.5
    AH, AL, sH, sL = 2.0, 1.0, 0.9, 0.05
    c = (1.0 - alpha) / alpha * beta ** (alpha / (1.0 - alpha))

    VH, VL = oligarch_values(AH, AL, sH, sL, alpha, beta, Lbar)
    # closed forms printed in the text
    den = 1.0 - beta * (sH - sL)
    cVH = c * Lbar / (1.0 - beta) * ((1.0 - beta * (1.0 - sL)) * AH
                                     + beta * (1.0 - sH) * AL) / den
    cVL = c * Lbar / (1.0 - beta) * ((1.0 - beta * sH) * AL
                                     + beta * sL * AH) / den
    print("  tilde V^H  linear solve %.10f   closed form %.10f" % (VH, cVH))
    print("  tilde V^L  linear solve %.10f   closed form %.10f" % (VL, cVL))
    # value-function iteration, as an independent check
    v = np.zeros(2)
    for _ in range(200000):
        w = np.array([c * Lbar * AH + beta * (sH * v[0] + (1.0 - sH) * v[1]),
                      c * Lbar * AL + beta * (sL * v[0] + (1.0 - sL) * v[1])])
        if np.max(np.abs(w - v)) < 1e-14:
            v = w
            break
        v = w
    print("  VFI                     %.10f               %.10f" % (v[0], v[1]))
    bE = c / (1.0 - beta) * ((1.0 - beta * (1.0 - sL)) * AH
                             + beta * (1.0 - sH) * AL) / den
    print("  b^E (23.21) = %.10f,   tilde V^H / Lbar = %.10f" % (bE, VH / Lbar))
    assert abs(bE - VH / Lbar) < 1e-10
    print()

    print("  dV^z/dw = 1 - Lbar = %.2f < 0: every elite member, whatever his skill,"
          % (1.0 - Lbar))
    print("  wants the wage at zero.  Unanimity on the entry barrier.")
    print()
    print("  The tax vote.  Elite member i prefers tau=0 iff a_i Lbar >= abar(t).")
    mustar = (Lbar - 1.0) * AL / (AH - AL)
    cond231 = 0.5 * AH / AL + 0.5
    print("    mu* = (Lbar-1)A^L/(A^H-A^L) = %.6f" % mustar)
    print("    Condition 23.1: Lbar >= %.6f  (here Lbar = %.2f) -> %s"
          % (cond231, Lbar, Lbar >= cond231))
    print("    Condition 23.1 <=> mu* >= 1/2:", mustar >= 0.5)
    for mu in (1.0, 0.8, 0.6, 0.5, 0.4, 1.0 / 3.0):
        abar = mu * AH + (1.0 - mu) * AL
        lo = AL * Lbar >= abar            # a low-skill elite prefers tau = 0
        hi = AH * Lbar >= abar
        maj = "high" if mu > 0.5 else "low"
        print("    mu=%.2f  abar=%.3f  low-skill wants 0: %-5s  high: %-5s"
              "  majority: %-4s  tau=%.1f"
              % (mu, abar, lo, hi, maj, 0.0))
        assert hi and (lo or mu > 0.5)
    print()

    M = sL / (1.0 - sH + sL)
    YE0 = beta ** (alpha / (1.0 - alpha)) * AH / alpha
    YEinf = beta ** (alpha / (1.0 - alpha)) * (AL + M * (AH - AL)) / alpha
    print("  Y^E(0) = %.8f   Y^E(inf) = %.8f  (M = %.6f)" % (YE0, YEinf, M))
    mu, prev = 1.0, np.inf
    for t in range(60):
        Y = beta ** (alpha / (1.0 - alpha)) * (mu * AH + (1.0 - mu) * AL) / alpha
        assert Y < prev
        prev = Y
        mu = sH * mu + sL * (1.0 - mu)
    print("  Y^E(t) strictly decreasing for t = 0..59 and -> Y^E(inf): True")
    print()


# ---------------------------------------------------------------------------
# 23.8  Leapfrogging.
# ---------------------------------------------------------------------------


def leapfrog_date(taubar, AH, AL, sH, sL, alpha):
    """Closed-form t' and the first t with Y^E(t) < Y^D, compared."""
    M = sL / (1.0 - sH + sL)
    lam, D = sH - sL, AH - AL
    X = (1.0 - taubar) ** (alpha / (1.0 - alpha)) * AH
    Theta = (X - AL - M * D) / ((1.0 - M) * D)
    if Theta <= 0.0:
        return None, None, Theta, M
    tprime = int(np.floor(np.log(Theta) / np.log(lam)))
    mu, first = 1.0, None
    for t in range(0, 20000):
        Y = mu * AH + (1.0 - mu) * AL
        if Y < X:
            first = t
            break
        mu = sH * mu + sL * (1.0 - mu)
    return tprime, first, Theta, M


def ex238():
    print("=" * 74)
    print("23.8  Proposition 23.6: when does democracy leapfrog oligarchy?")
    print("=" * 74)
    alpha = 1.0 / 3.0
    print("  Condition 23.2: (1-taubar)^{a/(1-a)} > A^L/A^H + M(1 - A^L/A^H)")
    print()
    print("  %7s %6s %6s %6s %6s   %8s %8s   %7s %7s" % (
        "taubar", "A^L/A^H", "sig^H", "sig^L", "M", "Cond23.2", "Theta",
        "t' form", "t' sim"))
    for taubar, ratio, sH, sL in [(0.20, 0.50, 0.90, 0.10),
                                  (0.30, 0.50, 0.90, 0.10),
                                  (0.20, 0.80, 0.90, 0.10),
                                  (0.20, 0.50, 0.95, 0.30),
                                  (0.60, 0.50, 0.90, 0.10),
                                  (0.10, 0.20, 0.90, 0.05)]:
        AH, AL = 1.0, ratio
        M = sL / (1.0 - sH + sL)
        c232 = (1.0 - taubar) ** (alpha / (1.0 - alpha)) > ratio + M * (1.0 - ratio)
        tp, ts, Theta, _ = leapfrog_date(taubar, AH, AL, sH, sL, alpha)
        print("  %7.2f %6.2f %6.2f %6.2f %6.3f   %8s %8.4f   %7s %7s" % (
            taubar, ratio, sH, sL, M, c232, Theta,
            "-" if tp is None else tp, "-" if ts is None else ts))
        if c232:
            assert tp is not None and ts is not None and ts == tp + 1
        else:
            assert tp is None
    print()
    print("  t' = floor(log Theta / log(sigma^H - sigma^L)); the simulation's first")
    print("  date with Y^E < Y^D is t' + 1 in every leapfrogging row, as stated.")
    print()
    print("  Comparative statics of Theta (higher Theta = earlier leapfrogging):")
    base = dict(taubar=0.2, ratio=0.5, sH=0.9, sL=0.1)
    def theta_of(taubar, ratio, sH, sL):
        return leapfrog_date(taubar, 1.0, ratio, sH, sL, alpha)[2]
    b = theta_of(base["taubar"], base["ratio"], base["sH"], base["sL"])
    print("    base Theta = %.6f" % b)
    print("    taubar 0.20 -> 0.10 : %.6f  (up, so sooner)"
          % theta_of(0.10, 0.5, 0.9, 0.1))
    print("    A^L/A^H 0.50 -> 0.30: %.6f  (up, so sooner)"
          % theta_of(0.2, 0.3, 0.9, 0.1))
    print("    M 0.500 -> 0.182    : %.6f  (up, so sooner)"
          % theta_of(0.2, 0.5, 0.9, 0.02))
    assert theta_of(0.10, 0.5, 0.9, 0.1) > b
    assert theta_of(0.2, 0.3, 0.9, 0.1) > b
    assert theta_of(0.2, 0.5, 0.9, 0.02) > b
    print()


# ---------------------------------------------------------------------------
# 23.9  Condition 23.1 fails.
# ---------------------------------------------------------------------------


def oligarchy_path(AH, AL, sH, sL, alpha, beta, Lbar, taubar, T=80):
    """Tax and output path of the oligarchy without imposing Condition 23.1.

    An elite member with skill a prefers tau = 0 iff a*Lbar >= abar(t); the elite
    vote by majority, so the tax is taubar exactly when the low-skill elite are
    both a majority (mu < 1/2) and want to tax (mu > mu*).
    """
    mustar = (Lbar - 1.0) * AL / (AH - AL)
    mu, taus, Ys = 1.0, [], []
    for t in range(T):
        tau = taubar if (mu < 0.5 and mu > mustar) else 0.0
        Ys.append((beta * (1.0 - tau)) ** (alpha / (1.0 - alpha))
                  * (mu * AH + (1.0 - mu) * AL) / alpha)
        taus.append(tau)
        mu = sH * mu + sL * (1.0 - mu)
    return np.array(taus), np.array(Ys), mustar


def ex239():
    print("=" * 74)
    print("23.9  Condition 23.1 relaxed: an oligarchy that taxes itself")
    print("=" * 74)
    alpha, beta, taubar = 1.0 / 3.0, 0.9, 0.3
    AH, AL = 2.0, 1.0
    print("  An elite member with skill a prefers tau = 0 iff a*Lbar >= abar(t);")
    print("  the low-skill elite want to tax iff mu(t) > mu* = (Lbar-1)A^L/(A^H-A^L),")
    print("  and they carry the vote iff mu(t) < 1/2.  Hence tau = taubar exactly on")
    print("  mu(t) in (mu*, 1/2).  Condition 23.1 <=> mu* >= 1/2 closes that window;")
    print("  mu* >= 1 <=> Lbar >= A^H/A^L makes taxation unattractive at every mu.")
    print()
    print("  %6s %8s %9s %8s   %-20s %10s %10s %7s" % (
        "Lbar", "mu*", "Cond23.1", "M", "taxing dates", "Y^D", "Y^E(inf)", "leapfr"))
    for Lbar, sH, sL in [(2.50, 0.90, 0.05), (1.80, 0.90, 0.05), (1.40, 0.90, 0.05),
                         (1.30, 0.90, 0.05), (1.25, 0.90, 0.05), (1.40, 0.90, 0.45)]:
        taus, Ys, mustar = oligarchy_path(AH, AL, sH, sL, alpha, beta, Lbar, taubar)
        M = sL / (1.0 - sH + sL)
        YD = (beta * (1.0 - taubar)) ** (alpha / (1.0 - alpha)) * AH / alpha
        tlong = taubar if (M < 0.5 and M > mustar) else 0.0
        YEinf = (beta * (1.0 - tlong)) ** (alpha / (1.0 - alpha)) * (
            AL + M * (AH - AL)) / alpha
        idx = np.where(taus > 0)[0]
        rng = "none" if idx.size == 0 else "t = %d .. %d" % (idx[0], idx[-1])
        print("  %6.2f %8.4f %9s %8.4f   %-20s %10.6f %10.6f %7s" % (
            Lbar, mustar, mustar >= 0.5, M, rng, YD, YEinf, YEinf < YD))
        if mustar >= 0.5:
            assert idx.size == 0, (Lbar, rng)
    print()
    print("  Row 3 (Lbar=1.40, mu*=0.40, M=0.3333 < mu*) is the interesting one: the")
    print("  oligarchy taxes itself for a FINITE spell in the middle of its decline,")
    print("  because for a while the low-skill elite are both a majority and still")
    print("  face an elite average skill above their own.  Output is non-monotone.")
    taus, Ys, mustar = oligarchy_path(AH, AL, 0.9, 0.05, alpha, beta, 1.40, taubar)
    idx = np.where(taus > 0)[0]
    d = np.diff(Ys)
    print("    taxing dates %s;  sign changes in Delta Y: %d"
          % ([int(x) for x in idx], int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-12])) != 0))))
    print("    Y just before / during / after: %.6f  %.6f  %.6f"
          % (Ys[idx[0] - 1], Ys[idx[0]], Ys[idx[-1] + 1]))
    assert idx.size > 0 and Ys[idx[0]] < Ys[idx[0] - 1] and Ys[idx[-1] + 1] > Ys[idx[-1]]
    print()
    print("  When M itself lies in (mu*, 1/2) the tax never switches off, so")
    print("  Y^E(inf)/Y^D = [A^L + M(A^H-A^L)]/A^H < 1 whatever taubar is, and")
    print("  Condition 23.2 is no longer needed: leapfrogging is certain.")
    Lbar = 1.25
    taus, Ys, mustar = oligarchy_path(AH, AL, 0.9, 0.05, alpha, beta, Lbar, taubar, 400)
    M = 0.05 / (1.0 - 0.9 + 0.05)
    YD = (beta * (1.0 - taubar)) ** (alpha / (1.0 - alpha)) * AH / alpha
    print("    Lbar=%.2f: mu*=%.4f < M=%.4f < 1/2;  Y^E(inf)=%.6f < Y^D=%.6f"
          % (Lbar, mustar, M, Ys[-1], YD))
    assert mustar < M < 0.5 and Ys[-1] < YD and taus[-1] > 0
    print("    ratio = %.6f = [A^L+M(A^H-A^L)]/A^H = %.6f"
          % (Ys[-1] / YD, (AL + M * (AH - AL)) / AH))
    print()


# ---------------------------------------------------------------------------
# 23.10  The arrival of a new technology.
# ---------------------------------------------------------------------------


def ex2310():
    print("=" * 74)
    print("23.10  A new technology, psi times as productive")
    print("=" * 74)
    alpha, beta, taubar = 1.0 / 3.0, 0.9, 0.3
    AH, AL, sH, sL = 2.0, 1.0, 0.9, 0.05
    M = sL / (1.0 - sH + sL)
    psibar = (AH / AL) ** (1.0 - alpha)
    print("  Psi = psi^{1/(1-alpha)} is the factor by which effective productivity")
    print("  rises;  an incumbent switches iff Psi*ahat_i > a_i.")
    print("  psibar = (A^H/A^L)^{1-alpha} = %.6f  (so Psi = A^H/A^L = %.4f)"
          % (psibar, AH / AL))
    print()
    print("  %7s %8s   %-34s %10s %10s %9s" % (
        "psi", "Psi", "who switches", "olig x", "dem x = Psi", "olig<dem"))
    for psi in (1.05, 1.2, 1.5, psibar, 2.0, 4.0):
        Psi = psi ** (1.0 / (1.0 - alpha))
        mu = 1.0                                  # arrival at t' with mu(t') = 1
        if Psi * AL > AH:
            who = "all incumbents"
            abar_new = Psi * (AL + M * (AH - AL))
        else:
            who = "all but (old high, new low)"
            abar_new = M * Psi * AH + (1.0 - M) * (mu * AH + (1.0 - mu) * Psi * AL)
        abar_old = mu * AH + (1.0 - mu) * AL
        x_olig = abar_new / abar_old
        print("  %7.4f %8.4f   %-34s %10.6f %10.6f %9s" % (
            psi, Psi, who, x_olig, Psi, x_olig < Psi + 1e-12))
        assert x_olig <= Psi + 1e-12
    print()
    print("  At mu(t')=1 the oligarchy's factor is 1 + M(Psi-1) for Psi < A^H/A^L")
    print("  and Psi[A^L+M(A^H-A^L)]/A^H above it; both are < Psi since M < 1.")
    for psi in (1.2, psibar, 3.0):
        Psi = psi ** (1.0 / (1.0 - alpha))
        a = 1.0 + M * (Psi - 1.0)
        b = Psi * (AL + M * (AH - AL)) / AH
        print("    psi=%.4f  Psi=%.4f   1+M(Psi-1)=%.6f   Psi[A^L+M D]/A^H=%.6f"
              % (psi, Psi, a, b))
    print("  the two agree at psi = psibar: %.10f vs %.10f"
          % (1.0 + M * (psibar ** (1.0 / (1.0 - alpha)) - 1.0),
             psibar ** (1.0 / (1.0 - alpha)) * (AL + M * (AH - AL)) / AH))
    print()
    print("  After the switch the elite's NEW-technology skills are a fresh draw, so")
    print("  mu jumps straight to M = %.6f: the oligarchy loses its whole initial" % M)
    print("  selection advantage in one period, while democracy keeps mu = 1.")
    print()


# ---------------------------------------------------------------------------
# 23.11  Entry barriers and multiple equilibrium wages.
# ---------------------------------------------------------------------------


def ex2311():
    print("=" * 74)
    print("23.11  Entry barriers and multiple equilibrium wages")
    print("=" * 74)
    alpha, beta, b = 1.0 / 3.0, 0.9, 0.25
    c = (1.0 - alpha) / alpha * beta ** (alpha / (1.0 - alpha))
    # a lognormal talent distribution, so that G is continuous and strictly increasing
    rng = np.random.default_rng(23)
    a = np.exp(0.4 * rng.standard_normal(4000000))
    amed = float(np.median(a))
    print("  c = (1-alpha)/alpha beta^{a/(1-a)} = %.6f,  entry cost b = %.2f" % (c, b))
    print("  talent: lognormal(0, 0.4^2); median a^med = %.6f (exact 1)" % amed)
    amed = 1.0
    print()

    print(" (a)  One period.  Entrepreneur iff c a - w - b >= w, i.e. a >= (2w+b)/c.")
    print("      Each hires one worker, so market clearing forces G(abar) = 1/2.")
    w1 = (c * amed - b) / 2.0
    print("      unique w = (c a^med - b)/2 = %.8f" % w1)
    f = lambda w: (1.0 - np.mean(a >= (2.0 * w + b) / c)) - np.mean(a >= (2.0 * w + b) / c)
    wnum = brentq(f, 0.0, c * amed / 2.0, xtol=1e-12)
    print("      numerical root of (workers - entrepreneurs) = %.8f" % wnum)
    print("      entrepreneurs are the upper half of the talent distribution.")
    print()

    print(" (b)  Two periods, incumbents exempt from b.  In period 2 the incumbents")
    print("      are the top half, so ANY wage in")
    lo2, hi2 = (c * amed - b) / 2.0, c * amed / 2.0
    print("        [ (c a^med - b)/2 , c a^med/2 ] = [%.8f, %.8f ]" % (lo2, hi2))
    print("      clears the market: width %.8f = b/2." % (hi2 - lo2))
    for w2 in np.linspace(lo2, hi2, 7):
        stay = c * amed - w2 >= w2                 # marginal incumbent stays
        enter = c * amed - w2 - b >= w2            # best outsider enters
        assert stay and not (enter and w2 > lo2 + 1e-12)
    print("      every incumbent stays, no outsider enters: verified at 7 points.")
    Om = lambda w2: max(c * amed - w2, w2) - max(c * amed - w2 - b, w2)
    print("      option value of incumbency Omega(w2) = c a^med - 2 w2 in [0, b]:")
    for w2 in (lo2, 0.5 * (lo2 + hi2), hi2):
        print("        w2 = %.8f  ->  Omega = %.8f   (c a^med - 2 w2 = %.8f)"
              % (w2, Om(w2), c * amed - 2.0 * w2))
        assert abs(Om(w2) - (c * amed - 2.0 * w2)) < 1e-12
    w1_of = lambda w2: (c * amed - b + beta * Om(w2)) / 2.0
    print("      hence w1 = (c a^med - b + beta Omega)/2 ranges over")
    print("        [%.8f, %.8f], width beta b/2 = %.8f"
          % (w1_of(hi2), w1_of(lo2), beta * b / 2.0))
    assert abs((w1_of(lo2) - w1_of(hi2)) - beta * b / 2.0) < 1e-12
    print("      a HIGHER second-period wage gives a LOWER first-period wage.")
    print()

    print(" (c)  A fraction eps die.  Surviving incumbents are only (1-eps)/2 < 1/2,")
    print("      so entry of measure eps/2 is forced and the marginal ENTRANT must")
    print("      break even: abar = a^med and w2 = (c a^med - b)/2, uniquely.")
    print()
    print("  %8s   %12s %12s   %12s" % ("eps", "w2", "w1", "entrants"))
    for eps in (0.5, 0.2, 0.05, 0.01, 1e-4):
        w2 = (c * amed - b) / 2.0
        w1 = (c * amed - b + beta * (1.0 - eps) * Om(w2)) / 2.0
        # entrants: new agents above the median, measure eps/2
        ent = eps * np.mean(a >= amed)
        total = (1.0 - eps) / 2.0 + ent
        print("  %8.5f   %12.8f %12.8f   %12.8f" % (eps, w2, w1, ent))
        assert abs(total - 0.5) < 2e-3
    print("      total entrepreneurs = 1/2 in every row.")
    print()

    print(" (d)  eps -> 0 selects w2 = %.8f, the LOWER end of the period-2" % lo2)
    print("      interval, and w1 = %.8f, the UPPER end of the period-1" % w1_of(lo2))
    print("      interval.  At eps = 0 labour demand is flat at 1/2 over the whole")
    print("      interval and coincides with supply, so nothing pins the wage; any")
    print("      positive turnover forces fresh entry and the entrant's break-even")
    print("      condition is a single equation.  The equilibrium set is upper but")
    print("      not lower hemicontinuous at eps = 0.")
    print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def figure():
    alpha, beta, taubar = 1.0 / 3.0, 0.9, 0.3
    AH, AL, sH, sL = 2.0, 1.0, 0.9, 0.05
    M = sL / (1.0 - sH + sL)
    T = 26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.95))

    # ---- left: output paths -------------------------------------------------
    YD = (beta * (1.0 - taubar)) ** (alpha / (1.0 - alpha)) * AH / alpha
    _, Y1, _ = oligarchy_path(AH, AL, sH, sL, alpha, beta, 2.5, taubar, T)   # Cond 23.1
    t2, Y2, ms2 = oligarchy_path(AH, AL, sH, sL, alpha, beta, 1.4, taubar, T)
    t = np.arange(T)
    ax1.axhline(YD, color="0.25", lw=1.1, ls="--", label=r"democracy $Y^D$")
    ax1.plot(t, Y1, lw=1.3, color="C0", label=r"oligarchy, Cond. 23.1 holds")
    ax1.plot(t, Y2, lw=1.3, color="C3",
             label=r"oligarchy, Cond. 23.1 fails")
    on = np.where(t2 > 0)[0]
    if on.size:
        ax1.axvspan(on[0] - 0.5, on[-1] + 0.5, color="C3", alpha=0.10, lw=0)
        ax1.text(0.5 * (on[0] + on[-1]), ax1.get_ylim()[0] + 0.05,
                 r"$\tau=\bar\tau$", color="C3", ha="center", fontsize=8)
    cross = np.where(Y1 < YD)[0]
    if cross.size:
        ax1.plot([cross[0]], [Y1[cross[0]]], "o", ms=4, color="C0")
        ax1.annotate(r"$t'$", (cross[0], Y1[cross[0]]),
                     textcoords="offset points", xytext=(4, 6), fontsize=8, color="C0")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"aggregate output")
    ax1.set_title("Leapfrogging (Exercises 23.8, 23.9)", fontsize=9)
    ax1.legend(fontsize=7, frameon=False, loc="upper right")

    # ---- right: the labour market with entry barriers -----------------------
    b, amed = 0.25, 1.0
    c = (1.0 - alpha) / alpha * beta ** (alpha / (1.0 - alpha))
    lo, hi = (c * amed - b) / 2.0, c * amed / 2.0
    wg = np.linspace(lo - 0.26, hi + 0.10, 4000)
    # period-2 labour demand with eps = 0: incumbents (top half) plus entrants
    dem0 = np.where(wg <= hi, 0.5, 0.0) + np.where(wg < lo, 0.5, 0.0)
    eps = 0.18
    demE = (np.where(wg <= hi, 0.5 * (1.0 - eps), 0.0)
            + np.where(wg < lo, 0.5, 0.0))
    ax2.plot(wg, np.minimum(dem0, 1.0), lw=1.4, color="C0",
             label=r"demand, $\varepsilon=0$")
    ax2.plot(wg, np.minimum(demE, 1.0), lw=1.1, color="C3", ls="-.",
             label=r"demand, $\varepsilon=0.18$")
    ax2.axhline(0.5, color="0.25", lw=1.1, ls="--", label=r"supply $=1/2$")
    ax2.axvspan(lo, hi, color="C0", alpha=0.12, lw=0)
    ax2.annotate("", xy=(lo, 0.62), xytext=(hi, 0.62),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="C0"))
    ax2.text(0.5 * (lo + hi), 0.645, r"width $b/2$", ha="center", fontsize=8,
             color="C0")
    ax2.plot([lo], [0.5], "o", ms=4, color="C3")
    ax2.annotate(r"$\varepsilon\!\to\!0$ selects this", (lo, 0.5),
                 textcoords="offset points", xytext=(-78, -24), fontsize=7.5,
                 color="C3")
    ax2.set_xlabel(r"wage $w$")
    ax2.set_ylabel(r"labour")
    ax2.set_ylim(0.0, 1.08)
    ax2.set_xlim(wg[0], wg[-1])
    ax2.set_title(r"Multiple wages at $\varepsilon=0$ (Exercise 23.11)", fontsize=9)
    ax2.legend(fontsize=7, frameon=False, loc="lower left")

    fig.tight_layout()
    save(fig, "ch23_political_institutions")


if __name__ == "__main__":
    ex231()
    ex232()
    ex233()
    ex236()
    ex237()
    ex238()
    ex239()
    ex2310()
    ex2311()
    figure()
