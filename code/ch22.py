"""Numerical parts of the Chapter 22 solutions (institutions, political economy, growth).

  * Exercise 22.1   the revenue-maximizing tax is interior, and equals 1 - alpha under
    Cobb-Douglas;
  * Exercise 22.4   the voluntary-contribution SPE: the two incentive constraints, the
    discount factor they require, and when the scheme is feasible at all;
  * Exercise 22.7   the closed form (22.28) for tau^COM against a numerical maximization
    of (22.24);
  * Exercise 22.12  the best stationary SPE of the holdup game when tau_bar < 1;
  * Exercise 22.29  the median entrepreneur's objective (22.46): single crossing always,
    quasi-concavity not always;
  * Exercise 22.31  the median voter's capital tax, its concavity, and its comparative
    statics in the median capital share;
  * Exercise 22.32  the four values of tau_bar and the ordering between them.

Run with `python code/ch22.py`.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from acemoglulib import FIG, plt, save  # noqa: F401


# ---------------------------------------------------------------------------
# Exercise 22.1: the revenue-maximizing tax is interior
# ---------------------------------------------------------------------------


def ex221(beta=0.9, A=1.0):
    print("Exercise 22.1: tau_hat lies strictly between 0 and 1")
    print("  The elite maximize R(tau) = tau f(k_hat(tau)) on [0,1].  R(0) = 0; as tau -> 1,")
    print("  k_hat(tau) = (f')^{-1}[(beta^{-1}+delta-1)/(1-tau)] -> (f')^{-1}(infinity) = 0")
    print("  by the Inada condition, so f(k_hat) -> 0 and R(1) = 0; and R(tau) > 0 on the")
    print("  interior.  A continuous function that is positive inside and zero at both ends")
    print("  attains its maximum inside, so tau_hat is in (0,1) and satisfies (22.16).")
    print(f"  {'alpha':>7} {'tau_hat (numerical)':>20} {'1 - alpha':>11} {'R(tau_hat)':>12}")
    for al in (0.2, 1 / 3, 0.5, 0.7):
        def R(tau):
            k = (beta * (1 - tau)) ** (1 / (1 - al)) * A
            return tau * (A ** (1 - al) * k ** al) / al
        r = minimize_scalar(lambda t: -R(t), bounds=(1e-9, 1 - 1e-9), method="bounded",
                            options={"xatol": 1e-12})
        print(f"  {al:>7.4f} {r.x:>20.10f} {1 - al:>11.4f} {R(r.x):>12.6f}")
    print("  Under Cobb-Douglas R(tau) is proportional to tau(1-tau)^{alpha/(1-alpha)}, whose")
    print("  first-order condition is 1 - tau = tau alpha/(1-alpha), giving tau_hat = 1-alpha.")


# ---------------------------------------------------------------------------
# Exercise 22.4: the voluntary-contribution SPE
# ---------------------------------------------------------------------------


def ex224(alpha=1 / 3, beta=0.9, A=1.0, Lbar=1.0):
    print("\nExercise 22.4: voluntary contributions and the SPE with zero taxes")
    print("  Trigger profile: each of the N entrepreneurs pays T_tilde/N, the elite set")
    print("  tau = 0; any shortfall or any positive tax triggers the MPE (tau = 1-alpha,")
    print("  zero contributions) for ever.  Two incentive constraints.")
    print("  ELITE: deviating means announcing tau(t+1) > 0, which yields nothing extra")
    print("  today under the one-period-commitment timing, so the constraint is simply")
    print("    T_tilde >= R^RE,   R^RE the elite's MPE revenue -- no discounting needed.")
    print("  ENTREPRENEUR i: free-riding saves T_tilde/N once and costs S(0) - S(tau_hat)")
    print("  for ever, so")
    print("    beta >= (T_tilde/N) / [S(0) - S(tau_hat)],")
    print("  and with T_tilde = R^RE the scheme works iff N[S(0)-S(tau_hat)] > R^RE.")
    th = 1 - alpha                                   # tau_hat

    def kh(tau):
        return (beta * (1 - tau)) ** (1 / (1 - alpha)) * A

    def f(k):
        return A ** (1 - alpha) * k ** alpha / alpha

    def wage(tau):
        return (1 - tau) * (f(kh(tau)) - kh(tau) * alpha * f(kh(tau)) / kh(tau))

    print(f"\n  alpha = {alpha:.4f}, beta = {beta}, per worker:")
    print(f"  {'':>26} {'tau = 0':>10} {'tau = 1-alpha':>14}")
    for name, g in (("output f(k_hat)", lambda t: f(kh(t))),
                    ("investment k_hat", kh),
                    ("wage w_hat", wage),
                    ("entrepreneur surplus S", lambda t: (1 - beta) / beta * kh(t)),
                    ("elite revenue tau f", lambda t: t * f(kh(t)))):
        print(f"  {name:>26} {g(0.0):>10.5f} {g(th):>14.5f}")
    S0, Sh = (1 - beta) / beta * kh(0.0), (1 - beta) / beta * kh(th)
    RRE = th * f(kh(th))
    print(f"  per worker: S(0) - S(tau_hat) = {S0 - Sh:.5f}, R^RE = {RRE:.5f}")
    print(f"  ratio = {RRE / (S0 - Sh):.3f} > 1, so with POSITIVE wages the entrepreneurs")
    print("  cannot buy the elite off: the incidence of an output tax falls mostly on")
    print("  wages, since w_hat(tau) = (1-tau)(1-alpha) f(k_hat(tau)).  The scheme needs the")
    print("  workers to contribute too.")
    S0z, Shz = f(kh(0.0)) - kh(0.0), (1 - th) * f(kh(th)) - kh(th)
    print(f"\n  With zero wages (Condition 22.1 failing, Section 22.4) the entrepreneur keeps")
    print(f"  (1-tau) f(k_hat) - k_hat: S(0) = {S0z:.5f}, S(tau_hat) = {Shz:.5f}, so")
    print(f"  S(0) - S(tau_hat) = {S0z - Shz:.5f} > R^RE = {RRE:.5f} and the scheme is")
    print(f"  feasible, with the discount factor threshold beta >= {RRE / (S0z - Shz):.5f}.")
    print("  Why N must be finite: with a continuum each contribution has measure zero, so")
    print("  a shortfall is undetectable, no trigger can ever fire, and the scheme unravels")
    print("  -- which is exactly why SPE and MPE coincide in the text (Proposition 22.7).")
    return RRE / (S0z - Shz)


# ---------------------------------------------------------------------------
# Exercise 22.7: the competition tax rate
# ---------------------------------------------------------------------------


def ex227(alpha=1 / 3, beta=0.9, Am=1.0, Ae=3.0):
    print("\nExercise 22.7: tau^COM against a numerical maximization of (22.24)")
    print("  Writing Psi = phi(1 - theta^e Lbar)/theta^e, the first-order condition reduces")
    print("  to (Lbar + Psi)(1 - tau) = Psi alpha tau/(1-alpha), whence")
    print("    tau = (1-alpha)(Lbar + Psi) / [(1-alpha)(Lbar + Psi) + alpha Psi],")
    print("  which is exactly kappa/(1+kappa) with kappa = (1-alpha)/alpha [1 + Lbar/Psi].")
    print(f"  {'Lbar':>6} {'theta^e':>8} {'phi':>6} {'tau^COM':>10} {'numerical':>11} "
          f"{'> 1-alpha?':>11}")
    B = beta ** (alpha / (1 - alpha)) * Am / alpha
    for Lbar, te, phi in ((0.6, 0.4, 1.0), (0.6, 0.4, 0.5), (0.9, 0.3, 0.8),
                          (0.4, 0.6, 1.0), (0.8, 0.2, 0.3)):
        kap = (1 - alpha) / alpha * (1 + te * Lbar / ((1 - te * Lbar) * phi))
        tcom = kap / (1 + kap)

        def obj(tau):
            w = (1 - alpha) * B * (1 - tau) ** (1 / (1 - alpha))
            elite = ((1 - alpha) * beta ** (alpha / (1 - alpha)) * Ae / alpha - w) * Lbar
            rev = phi * B * tau * (1 - tau) ** (alpha / (1 - alpha)) * (1 - te * Lbar) / te
            return elite + rev
        r = minimize_scalar(lambda t: -obj(t), bounds=(1e-9, 1 - 1e-9),
                            method="bounded", options={"xatol": 1e-13})
        print(f"  {Lbar:>6.2f} {te:>8.2f} {phi:>6.2f} {tcom:>10.7f} {r.x:>11.7f} "
              f"{str(bool(tcom > 1 - alpha)):>11}")
    print("  kappa exceeds (1-alpha)/alpha strictly, so tau^COM always exceeds")
    print("  tau^RE = 1-alpha: the factor price manipulation motive pushes the elite past")
    print("  the peak of the Laffer curve.  And kappa is finite, so tau^COM < 1 = tau^FPM:")
    print("  the revenue motive pulls them back from the pure-manipulation corner.")


# ---------------------------------------------------------------------------
# Exercise 22.12: the best stationary SPE of the holdup game
# ---------------------------------------------------------------------------


def ex2212(alpha=1 / 3, beta=0.9):
    print("\nExercise 22.12: the holdup game with a ceiling tau_bar < 1")
    print("  With capital sunk, deviating at an announced tau* raises the tax to tau_bar,")
    print("  and the punishment is the MPE, which now yields the elite r(tau_bar) > 0 per")
    print("  period rather than nothing.  With r(tau) = tau(1-tau)^{alpha/(1-alpha)} the")
    print("  stationary incentive constraint is")
    print("    (1-tau*)^{alpha/(1-alpha)}[tau* - (1-beta) tau_bar] >= beta r(tau_bar),")
    print("  which at tau_bar = 1 reduces to alpha^{alpha/(1-alpha)}(beta-alpha) >= 0, that")
    print("  is beta >= alpha -- Proposition 22.9.")
    e = alpha / (1 - alpha)

    def r(t):
        return t * (1 - t) ** e

    def ic(ts, tb):
        return (1 - ts) ** e * (ts - (1 - beta) * tb) - beta * r(tb)

    print(f"  alpha = {alpha:.4f}, beta = {beta}, so tau^RE = 1-alpha = {1-alpha:.4f}")
    print(f"  {'tau_bar':>8} {'MPE tax':>9} {'best stationary SPE':>20} {'gain in r':>10}")
    for tb in (0.30, 0.50, 0.6667, 0.80, 0.95, 1.00):
        if tb <= 1 - alpha + 1e-9:
            best = tb
        elif ic(1 - alpha, tb) >= 0:
            best = 1 - alpha
        else:
            lo, hi = 1 - alpha, tb
            best = brentq(lambda t: ic(t, tb), lo, hi) if ic(hi, tb) >= 0 else tb
        print(f"  {tb:>8.4f} {tb:>9.4f} {best:>20.6f} "
              f"{r(best) - r(tb):>10.6f}")
    print("  For tau_bar <= 1-alpha the ceiling already binds below the elite's own")
    print("  optimum, the MPE is the best the elite can do, and there is nothing for an")
    print("  implicit agreement to improve.  For tau_bar > 1-alpha the best stationary SPE")
    print("  brings the tax down, all the way to 1-alpha when the constraint is slack.")


# ---------------------------------------------------------------------------
# Exercise 22.29: single crossing and quasi-concavity
# ---------------------------------------------------------------------------


def ex2229(beta=0.9, Abar=1.0, alpha=1 / 3):
    print("\nExercise 22.29: V_tilde_i is single crossing but need not be quasi-concave")
    print("  Grouping (22.46) in A_i,")
    print("    V_tilde_i(tau) = G1(tau) + A_i G2(tau) + const,")
    print("    G1 = beta tau Abar f(k_hat), G2 = beta(1-tau) f(k_hat) - k_hat,")
    print("  which is the intermediate-preferences form of Exercise 22.24 with B(A_i)=A_i.")
    print("  V_i(tau) - V_i(tau') is therefore AFFINE in A_i, so the set of entrepreneurs")
    print("  preferring tau to tau' is an interval in A_i: single crossing, always.")
    print("\n  For quasi-concavity, the envelope theorem gives G2'(tau) = -beta f(k_hat),")
    print("  so V_i'(tau) = beta f(k_hat) [ (Abar - A_i) + Abar Phi(tau) ] with")
    print("    Phi(tau) = [tau/(1-tau)] (f')^2/(f'' f) = -[tau/(1-tau)] eps(k_hat)/eta(k_hat),")
    print("  eps = k f'/f the capital share and eta = -k f''/f' the curvature.  Hence")
    print("    V_i is quasi-concave for EVERY A_i  <=>  [tau/(1-tau)] eps/eta is monotone.")
    print("  Under Cobb-Douglas eps = alpha and eta = 1-alpha are constants, the bracket is")
    print("  strictly increasing, and quasi-concavity always holds -- which is why a")
    print("  counterexample needs an f whose curvature varies.")

    def build(e0, e1, m, sg):
        """f with log-curvature eta(k) = e0 + e1 exp[-(log k - m)^2/(2 s^2)]."""
        lk = np.linspace(np.log(1e-9), np.log(1e3), 600001)
        k = np.exp(lk)
        eta = e0 + e1 * np.exp(-((lk - m) ** 2) / (2 * sg * sg))
        lfp = -np.concatenate([[0.0], np.cumsum(eta[:-1] * np.diff(lk))])
        fp = np.exp(lfp)
        fp = fp / fp[np.searchsorted(lk, 0.0)]
        f = np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(k))])
        eps = k * fp / np.maximum(f, 1e-300)
        tau = 1 - 1 / (beta * fp)
        ok = (tau > 1e-3) & (tau < 1 - 1e-3) & (f > 0)
        t, kk, ff, ee, nn = tau[ok], k[ok], f[ok], eps[ok], eta[ok]
        o = np.argsort(t)
        return t[o], kk[o], ff[o], ee[o], nn[o]

    print(f"\n  {'case':>26} {'tau/(1-tau) eps/eta':>21} {'A_i':>6} {'local maxima':>13}")
    out = {}
    # Cobb-Douglas, for reference
    t = np.linspace(1e-4, 1 - 1e-4, 6000)
    kh = (beta * (1 - t)) ** (1 / (1 - alpha))
    ff = kh ** alpha / alpha
    for A_i in (0.0, 0.5, 1.5):
        V = beta * t * Abar * ff + A_i * (beta * (1 - t) * ff - kh)
        d = np.diff(V)
        n = int(np.sum((np.sign(d[:-1]) > 0) & (np.sign(d[1:]) < 0)))
        print(f"  {'Cobb-Douglas':>26} {'monotone':>21} {A_i:>6.2f} {n:>13d}")
        out[("cobb", A_i)] = (t, V)
    # a curvature dip
    t, kk, ff, ee, nn = build(0.9, -0.88, -2.0, 0.10)
    mPhi = t / (1 - t) * ee / nn
    dm = np.diff(mPhi)
    flips = int(np.sum(np.sign(dm[:-1]) * np.sign(dm[1:]) < 0))
    for A_i in (0.0, 0.3, 1.0):
        V = beta * t * Abar * ff + A_i * (beta * (1 - t) * ff - kk)
        d = np.diff(V)
        n = int(np.sum((np.sign(d[:-1]) > 0) & (np.sign(d[1:]) < 0)))
        print(f"  {'curvature dip':>26} {('%d sign flips' % flips):>21} {A_i:>6.2f} "
              f"{n:>13d}")
        out[("dip", A_i)] = (t, V)
    print("  With eta(k) dipping to 0.02 over a narrow range of k -- a stretch over which f")
    print("  is nearly linear -- the bracket is no longer monotone, and V_tilde_i acquires")
    print("  TWO local maxima for small A_i: quasi-concavity fails while single crossing,")
    print("  which is an algebraic property of the A_i-affine form, survives untouched.")
    print("  That is exactly why Proposition 22.17 appeals to Theorem 22.4 rather than to")
    print("  the ordinary Median Voter Theorem.")
    return out


# ---------------------------------------------------------------------------
# Exercise 22.31: the median voter's capital tax
# ---------------------------------------------------------------------------


def ex2231(alpha=0.4, beta=0.95, A=2.0):
    print("\nExercise 22.31: capital taxation for infrastructure and the median voter")
    print("  With Y_j = A K^{1-alpha} G^alpha L^alpha and G = tau Kbar, the rental rate is")
    print("    r = (1-alpha) A tau^alpha,   net return R(tau) = (1-alpha) A tau^alpha - tau,")
    print("  independent of Kbar, so the economy grows at 1+g = beta(1+R) whatever the")
    print("  distribution of Kbar(0).  Wages are w = alpha A tau^alpha Kbar, so human")
    print("  wealth is h Kbar with h = alpha A tau^alpha/(1-beta); all wealth grows at the")
    print("  common rate beta(1+R), hence K_i(t) = omega_i Kbar(t) with omega_i constant.")

    def R(t):
        return (1 - alpha) * A * t ** alpha - t

    tgr = (alpha * (1 - alpha) * A) ** (1 / (1 - alpha))
    print(f"  growth-maximizing tax tau^gr = [alpha(1-alpha)A]^{{1/(1-alpha)}} = {tgr:.6f}")

    def U(t, om):
        lvl = (1 + R(t)) * om + alpha * A * t ** alpha / (1 - beta)
        if lvl <= 0 or 1 + R(t) <= 0:
            return -1e18
        return (np.log(lvl) / (1 - beta)
                + beta * np.log(beta * (1 + R(t))) / (1 - beta) ** 2)

    print(f"  {'omega_M':>9} {'preferred tau':>14} {'> tau^gr?':>10} {'growth 1+g':>11} "
          f"{'concave?':>9}")
    curves = {}
    for om in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
        r = minimize_scalar(lambda t: -U(t, om), bounds=(1e-6, 0.999),
                            method="bounded", options={"xatol": 1e-11})
        tg = np.linspace(1e-4, 0.999, 3000)
        Ug = np.array([U(t, om) for t in tg])
        d2 = np.diff(Ug, 2)
        curves[om] = (tg, Ug)
        print(f"  {om:>9.2f} {r.x:>14.7f} {str(bool(r.x > tgr)):>10} "
              f"{beta * (1 + R(r.x)):>11.6f} {str(bool(np.all(d2 <= 1e-9))):>9}")
    print("  Every voter wants MORE than the growth-maximizing rate, because at tau^gr the")
    print("  growth term is flat while the wage term alpha A tau^alpha/(1-beta) is still")
    print("  rising; and the preferred rate falls with omega_i, since a capital-rich voter")
    print("  weighs the return on his own capital more heavily.  U is concave in tau -- a")
    print("  sum of logs of concave functions -- so preferences are single peaked and the")
    print("  median voter's rate is implemented.  A fall in omega_M therefore raises the")
    print("  tax above tau^gr and LOWERS growth.")
    print("\n  Part (d): this is not an MPE.  The tax is legislated once at t=0 and committed")
    print("  to, whereas an MPE lets the policy at t depend only on the date-t state.  Once")
    print("  capital is in place the median voter would want a higher tax than he wanted")
    print("  ex ante -- the holdup of Section 22.5.2.  To set the problem up as an MPE one")
    print("  posits a policy rule tau = Phi(Kbar) (the shares omega_i being constant by")
    print("  part b), solves each household's Bellman equation taking Phi as given, and")
    print("  requires the median voter's most preferred current tax, given Phi for the")
    print("  future, to equal Phi at the current state -- a fixed point in the policy rule.")
    return curves, tgr


# ---------------------------------------------------------------------------
# Exercise 22.32: the four values of tau_bar
# ---------------------------------------------------------------------------


def ex2232(alpha=0.4, beta=0.9, zeta=2.0):
    print("\nExercise 22.32: output, welfare, elite and citizen optima")
    th = alpha / (1 - alpha)
    print("  Writing theta = alpha/(1-alpha), A[tau] is proportional to")
    print("  [(1-tau)^theta tau]^{1/(zeta-1)} and Y[tau] to (1-tau)^theta A[tau], so")
    print("    log Y = const + theta zeta/(zeta-1) log(1-tau) + log(tau)/(zeta-1),")
    print("  whose first-order condition gives (1-alpha)(1-tau) = alpha zeta tau, that is")
    print("    tau* = (1-alpha)/(1-alpha+alpha zeta)   -- equation (22.60).")
    print("  The elite's flow is proportional to (1-tau)^theta tau A[tau], giving")
    print("    tau^e = 1/(1+theta) = 1 - alpha,")
    print("  and the citizen's consumption to (1-tau)^{1+theta} A[tau], giving")
    print("    tau^c = 1/[(1+theta) zeta] = (1-alpha)/zeta.")

    def Aof(t):
        return (beta ** (1 / (1 - alpha)) / (1 - alpha) * (1 - t) ** th * t) ** (1 / (zeta - 1))

    def Y(t):
        return (beta * (1 - t)) ** th * Aof(t) / alpha

    def W(t):                       # total consumption: output less investment and G
        return Y(t) * (1 - beta * t / zeta - alpha * beta * (1 - t))

    def Ce(t):
        return t * Y(t) * (1 - beta / zeta)

    def Cc(t):
        return (1 - t) * Y(t) - alpha * beta * (1 - t) * Y(t)

    print(f"\n  alpha = {alpha}, beta = {beta}, zeta = {zeta}")
    print(f"  {'objective':>12} {'numerical':>12} {'closed form':>13}")
    rows = []
    for name, g, cf in (("output", Y, (1 - alpha) / (1 - alpha + alpha * zeta)),
                        ("welfare", W, None),
                        ("elite", Ce, 1 - alpha),
                        ("citizens", Cc, (1 - alpha) / zeta)):
        r = minimize_scalar(lambda t: -g(t), bounds=(1e-9, 1 - 1e-9),
                            method="bounded", options={"xatol": 1e-12})
        rows.append((name, r.x, g))
        print(f"  {name:>12} {r.x:>12.8f} "
              f"{('%.8f' % cf) if cf is not None else '--':>13}")
    tc, ts, te = rows[3][1], rows[0][1], rows[2][1]
    tw = rows[1][1]
    print(f"  ordering: 0 < tau^c = {tc:.5f} < tau* = {ts:.5f} < tau^e = {te:.5f} < 1: "
          f"{str(bool(0 < tc < ts < te < 1))}")
    print(f"            0 < tau^c = {tc:.5f} < tau^wm = {tw:.5f} < tau^e = {te:.5f} < 1: "
          f"{str(bool(0 < tc < tw < te < 1))}")
    print("  tau^c < tau* because zeta > 1: the citizen bears the tax on his own output")
    print("  one for one and values public goods only through his own production, while")
    print("  output counts the elite's spending too.  tau* < tau^e because the elite")
    print("  ignore the investment the tax discourages and sit at the peak of their own")
    print("  Laffer curve.  A state with tau_bar below tau^c is too WEAK -- the elite")
    print("  expect too few future rents to invest in public goods -- and one above tau^e")
    print("  is too strong.")
    grid = np.linspace(1e-4, 0.999, 2000)
    return {n: (grid, np.array([g(t) for t in grid])) for n, _, g in rows}, rows


# ---------------------------------------------------------------------------


def figure(vcur, tcur):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    dips = sorted(((k[1], v) for k, v in vcur.items() if k[0] == "dip"))
    for (A_i, (t, V)), style in zip(dips, ("-", "--", ":", "-.")):
        ax1.plot(t, (V - V.min()) / (V.max() - V.min()), style, color="black", lw=0.9,
                 label=rf"$A_i={A_i:g}$")
    ax1.set_xlabel(r"$\tau'$")
    ax1.set_ylabel("normalized $\\tilde V_i$")
    ax1.legend(frameon=False, fontsize=6, loc="lower left")
    ax1.set_title(r"$\tilde V_i$ with a curvature dip (Ex. 22.29)", fontsize=7.5)

    for (name, (g, v)), style in zip(tcur.items(), ("-", "--", ":", "-.")):
        ax2.plot(g, v / v.max(), style, color="black", lw=0.9, label=name)
        ax2.axvline(g[v.argmax()], color="0.8", lw=0.4)
    ax2.set_xlabel(r"$\bar\tau$")
    ax2.set_ylabel("normalized objective")
    ax2.set_ylim(0, 1.08)
    ax2.legend(frameon=False, fontsize=6, loc="lower center", ncol=2)
    ax2.set_title(r"weak versus strong states (Ex. 22.32)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch22_political_economy")


def main():
    ex221()
    ex224()
    ex227()
    ex2212()
    vcur = ex2229()
    ex2231()
    tcur, _ = ex2232()
    figure(vcur, tcur)


if __name__ == "__main__":
    main()
