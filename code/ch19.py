"""Numerical parts of the Chapter 19 solutions (trade and growth).

  * Exercise 19.17  global stability of the Acemoglu-Ventura world equilibrium, written
    as a replicator equation on the simplex with an explicit Lyapunov function;
  * Exercise 19.18  autarky growth rates against the world growth rate with trade;
  * Exercise 19.19  how far the world income distribution (19.40) can spread;
  * Exercise 19.23  the sign of d g*/d tau and the welfare condition;
  * Exercise 19.30  the product cycle with and without trade: relative income against
    real consumption;
  * Exercise 19.36  learning-by-doing under autarky and trade, for eps > 1, eps = 1 and
    eps < 1, and with infant-industry protection until T.

Run with `python code/ch19.py`.
"""
import numpy as np
from scipy.optimize import brentq

from acemoglulib import FIG, plt, save  # noqa: F401


# ---------------------------------------------------------------------------
# Section 19.4: the Acemoglu-Ventura world equilibrium
# ---------------------------------------------------------------------------


def gstar(mu, zeta, rho, eps, tau):
    """Unique root of (19.38)."""
    def F(g):
        return np.sum(mu * (zeta * (rho + g)) ** ((1 - eps) / tau)) - 1.0
    lo = -min(rho) + 1e-12
    hi = 1.0
    while F(hi) > 0:
        hi *= 2
        if hi > 1e12:
            raise ValueError("no root")
    return brentq(F, lo, hi, xtol=1e-15, rtol=1e-15)


def shares(mu, zeta, rho, eps, tau, g):
    """Steady-state relative incomes (19.40)."""
    return mu * (zeta * (rho + g)) ** ((1 - eps) / tau)


def draw(J=6, seed=5):
    rng = np.random.default_rng(seed)
    mu = np.round(rng.uniform(0.6, 1.8, J), 4)
    zeta = np.round(rng.uniform(15.0, 25.0, J), 4)
    rho = np.round(rng.uniform(0.02, 0.06, J), 4)
    return mu, zeta, rho


def ex1917(eps=4.0, tau=0.3, T=400.0, dt=0.01, seed=5):
    print("Exercise 19.17: global stability as a replicator equation")
    mu, zeta, rho = draw(seed=seed)
    J = len(mu)
    g = gstar(mu, zeta, rho, eps, tau)
    ystar = shares(mu, zeta, rho, eps, tau, g)
    print(f"  J = {J}, eps = {eps}, tau = {tau}, N = sum(mu) = {mu.sum():.4f}")
    print(f"  g* = {g:.8f}, sum of steady-state shares = {ystar.sum():.12f}")
    print("  writing y_j = Y_j/Y, (19.36) gives r_j = (mu_j/y_j)^{1/(eps-1)} and hence")
    print("    dy_j/dt = [(eps-1)/eps] y_j [ h_j(y_j) - sum_i y_i h_i(y_i) ],")
    print("    h_j(y) = zeta_j^{-1} (mu_j/y)^{tau/(eps-1)} - rho_j,")
    print("  a replicator equation with each h_j strictly decreasing.  V(y) = sum_j y*_j")
    print("  log(y*_j/y_j) then has dV/dt = -[(eps-1)/eps] sum_j (y*_j - y_j)[h_j(y_j) - g*]")
    print("  < 0 off the rest point, since h_j is strictly decreasing and h_j(y*_j) = g*.")

    def h(y):
        return (mu / y) ** (tau / (eps - 1)) / zeta - rho

    rng = np.random.default_rng(seed + 1)
    print(f"  {'start':>32} {'max |y - y*|':>13} {'V decreasing?':>14} {'final g':>10}")
    paths = None
    for trial in range(3):
        y = rng.dirichlet(np.ones(J))
        if trial == 0:
            y = np.full(J, 1.0 / J)
        y0 = y.copy()
        Vs, ys = [], []
        for _ in range(int(T / dt)):
            hv = h(y)
            gw = float(y @ hv)
            Vs.append(float(ystar @ np.log(ystar / y)))
            ys.append(y.copy())
            y = y + dt * ((eps - 1) / eps) * y * (hv - gw)
            y = np.maximum(y, 1e-14)
            y /= y.sum()
        Vs = np.asarray(Vs)
        mono = bool(np.all(np.diff(Vs) <= 1e-14))
        print(f"  {np.array2string(np.round(y0, 3)):>32} "
              f"{np.max(np.abs(y - ystar)):>13.2e} {str(mono):>14} "
              f"{float(y @ h(y)):>10.6f}")
        if paths is None:
            paths = (np.asarray(ys), Vs, ystar, dt)
    print("  every trajectory converges to (19.40) and V falls monotonically along it")
    return paths


def ex1918(eps=4.0, tau=0.3, seed=5):
    print("\nExercise 19.18: autarky against trade")
    mu, zeta, rho = draw(seed=seed)
    g = gstar(mu, zeta, rho, eps, tau)
    gA = mu ** (tau / (eps - 1)) / zeta - rho
    ident = np.sum(((rho + gA) / (rho + g)) ** ((eps - 1) / tau))
    print("  in autarky country j uses only its own mu_j varieties, so the ideal price")
    print("  index of its intermediates is mu_j^{1/(1-eps)} r_j and the return on capital")
    print("  is mu_j^{tau/(eps-1)}/zeta_j -- a pure AK economy with")
    print("    g^A_j = mu_j^{tau/(eps-1)}/zeta_j - rho_j.")
    print(f"  {'j':>3} {'mu':>8} {'zeta':>8} {'rho':>8} {'g^A_j':>10} {'g*':>10} "
          f"{'g* > g^A?':>10}")
    for j in range(len(mu)):
        print(f"  {j:>3} {mu[j]:>8.4f} {zeta[j]:>8.4f} {rho[j]:>8.4f} {gA[j]:>10.6f} "
              f"{g:>10.6f} {str(bool(g > gA[j])):>10}")
    print(f"  substituting zeta_j = mu_j^{{tau/(eps-1)}}/(rho_j + g^A_j) into (19.38)")
    print(f"  turns it into sum_j [(rho_j + g^A_j)/(rho_j + g*)]^{{(eps-1)/tau}} = 1,")
    print(f"  numerically {ident:.12f}.  With J >= 2 every term is then below one, so")
    print("  g* exceeds EVERY country's autarky growth rate.  Since C_j = rho_j zeta_j K_j")
    print("  in both regimes and K_j(0) is given, steady-state welfare is")
    print("  log(rho_j zeta_j K_j(0))/rho_j + g/rho_j^2, so every country -- including the")
    print("  fastest-growing one -- is strictly better off with trade")
    return gA, g


def ex1919(eps=4.0, tau=0.3, alpha=1 / 3):
    print("\nExercise 19.19: how far the world income distribution can spread")
    print("  (19.40) gives d log y*_j/d log mu_j = 1 and")
    print("  d log y*_j/d log zeta_j = d log y*_j/d log(rho_j + g*) = -(eps-1)/tau,")
    print(f"  against the Solow elasticity alpha/(1-alpha) = {alpha / (1 - alpha):.4f}"
          f" on the saving rate.")
    print(f"  {'eps':>6} {'tau':>6} {'(eps-1)/tau':>12} {'2x zeta':>12} {'4x zeta':>14}")
    for e in (3.0, 4.0, 8.0):
        for t in (0.15, 0.30, 0.50):
            el = (e - 1) / t
            print(f"  {e:>6.1f} {t:>6.2f} {el:>12.4f} {2 ** el:>12.2f} "
                  f"{4 ** el:>14.4g}")
    print(f"  Solow with alpha = {alpha:.4f}: a 4-fold difference in s gives "
          f"{4 ** (alpha / (1 - alpha)):.4f}")
    print("  plausible values (tau = 0.15-0.30 for imported intermediates in GDP,")
    print("  eps = 3-8 from the trade elasticity literature) give elasticities of 7 to 47,")
    print("  so the model generates far larger income differences than the neoclassical")
    print("  one -- arguably too large, which is the discipline it needs")


def ex1923(eps=4.0, seed=5):
    print("\nExercise 19.23: the effect of openness on the world growth rate")
    print("  differentiating (19.38) at the equilibrium, where the weights mu_j z_j^{-phi}")
    print("  ARE the relative incomes y*_j and phi = (eps-1)/tau,")
    print("    d g*/d tau = [sum_j y*_j log z_j] / [tau sum_j y*_j/(rho_j + g*)],")
    print("  with z_j = zeta_j(rho_j + g*) = (r*_j)^tau.  Using y*_j = mu_j z_j^{-phi},")
    print("    sign(d g*/d tau) = sign( log N - KL(y* || mu/N) ).")
    print(f"  {'case':>22} {'tau':>6} {'g*':>10} {'log N':>9} {'KL':>9} "
          f"{'sign pred':>10} {'numerical':>11}")
    mu0, zeta0, rho0 = draw(seed=seed)
    cases = [("symmetric, N > 1", np.full(6, 1.2), np.full(6, 20.0), np.full(6, 0.04)),
             ("symmetric, N < 1", np.full(6, 0.1), np.full(6, 20.0), np.full(6, 0.04)),
             ("drawn", mu0, zeta0, rho0),
             ("N > 1 but dispersed", np.full(6, 0.2),
              np.array([18.0, 20.0, 24.0, 30.0, 40.0, 60.0]), np.full(6, 0.04))]
    for name, mu, zeta, rho in cases:
        for tau in (0.3,):
            g = gstar(mu, zeta, rho, eps, tau)
            y = shares(mu, zeta, rho, eps, tau, g)
            N = mu.sum()
            KL = float(np.sum(y * np.log(y / (mu / N))))
            pred = np.sign(np.log(N) - KL)
            num = (gstar(mu, zeta, rho, eps, tau + 1e-6)
                   - gstar(mu, zeta, rho, eps, tau - 1e-6)) / 2e-6
            print(f"  {name:>22} {tau:>6.2f} {g:>10.6f} {np.log(N):>9.4f} {KL:>9.4f} "
                  f"{pred:>10.0f} {np.sign(num):>11.0f}")
    print("  so openness raises world growth exactly when the gains from variety, log N,")
    print("  outweigh the dispersion of the world income distribution around the")
    print("  distribution of technological capabilities.")
    print("  In the AK version C_j(t) = rho_j zeta_j K_j(t), so steady-state welfare is")
    print("  monotone in g* and welfare and growth move together.  In the model with")
    print("  labour, p^C_j C_j is proportional to K_j but p^C_j is proportional to")
    print("  K_j^{(1-tau)(1-gamma)}, so consumption grows at [gamma + (1-gamma) tau] g*:")
    print(f"  {'gamma':>7} {'tau':>6} {'(1-b) = gamma+(1-gamma)tau':>28}")
    for gam in (0.3, 0.6):
        for tau in (0.15, 0.30, 0.50):
            print(f"  {gam:>7.2f} {tau:>6.2f} {gam + (1 - gam) * tau:>28.4f}")
    print("  d[(1-b) g*]/d tau = (1-gamma) g* + (1-b) d g*/d tau, so consumption growth")
    print("  -- and hence welfare -- rises with tau over a strictly larger range than g*")
    print("  itself does.  That is how (a) and (b) can both hold.")


# ---------------------------------------------------------------------------
# Section 19.5: the product cycle
# ---------------------------------------------------------------------------


def ex1930(eps=3.0):
    print("\nExercise 19.30: the product cycle with and without trade")
    print("  without trade the North consumes all N goods and the South only N_o, so")
    print("  w_n = N^{1/(eps-1)}, w_s = N_o^{1/(eps-1)} and w_n/w_s = (1+eta/iota)^{1/(eps-1)};")
    print("  with trade (19.54) gives max{(eta/iota)(L^s/L^n))^{1/eps}, 1}.")
    print(f"  {'eta/iota':>9} {'L^s/L^n':>9} {'no trade':>10} {'trade':>10} "
          f"{'trade worse?':>13} {'South real c':>13} {'North real c':>13}")
    for x in (0.5, 2.0):
        for ell in (1.0, 4.0, 12.0):
            No, Nn = 1.0, x                       # normalize N_o = 1
            N = No + Nn
            rA = (1 + x) ** (1 / (eps - 1))
            rT = max((x * ell) ** (1 / eps), 1.0)
            # real consumption per worker
            csA, cnA = No ** (1 / (eps - 1)), N ** (1 / (eps - 1))
            csT = (Nn * rT ** (1 - eps) + No) ** (1 / (eps - 1))
            cnT = (Nn + No * rT ** (eps - 1)) ** (1 / (eps - 1))
            print(f"  {x:>9.2f} {ell:>9.2f} {rA:>10.4f} {rT:>10.4f} "
                  f"{str(bool(rT > rA)):>13} {csT / csA:>13.4f} {cnT / cnA:>13.4f}")
    print("  the gap can be larger with trade -- it needs (1+eta/iota)^{eps/(eps-1)} <")
    print("  (eta/iota)(L^s/L^n), so a large South and substitutable goods -- because")
    print("  trade converts a love-of-variety disadvantage into a terms-of-trade one.")
    print("  But real consumption rises for BOTH blocks in every case:")
    print("  c_s^T/c_s^A = [1 + (N_n/N_o) omega^{1-eps}]^{1/(eps-1)} > 1 always, and")
    print("  c_n^T/c_n^A = [(N_n + N_o omega^{eps-1})/N]^{1/(eps-1)} > 1 whenever omega > 1.")
    print("  Relative income and welfare therefore move in opposite directions")


# ---------------------------------------------------------------------------
# Section 19.7: learning-by-doing
# ---------------------------------------------------------------------------


def lbd_autarky(eps, eta, A0, T, dt):
    A, out = A0, []
    for _ in range(int(T / dt)):
        L1 = A ** (eps - 1) / (1 + A ** (eps - 1)) if eps != 1 else 0.5
        out.append((A, L1))
        A *= np.exp(dt * eta * L1)
    return np.asarray(out)


def lbd_trade(eps, eta, An0, As0, T, dt):
    """North and South with free trade.  Three regimes, each pinned down by which
    country is indifferent between the two sectors:
      * South diversified  (A_s^eps > A_n): w_n/w_s = A_n/A_s;
      * complete specialization:            w_n/w_s = A_n^{(eps-1)/eps};
      * North diversified  (eps < 1):       w_n/w_s = 1.
    Returns columns (A_n, A_s, L^1_n, L^1_s, w_n/w_s)."""
    An, As, out = An0, As0, []
    for _ in range(int(T / dt)):
        if As ** eps > An:                        # South also produces good 1
            L1n, L1s = 1.0, (As ** eps - An) / (As ** eps + As)
            omega = An / As
        elif An ** (1 / eps) <= An:               # complete specialization
            L1n, L1s = 1.0, 0.0
            omega = An ** ((eps - 1) / eps)
        else:                                     # eps < 1: the North diversifies
            L1n = min(2 * An ** (eps - 1) / (1 + An ** (eps - 1)), 1.0)
            L1s, omega = 0.0, 1.0
        out.append((An, As, L1n, L1s, omega))
        An *= np.exp(dt * eta * L1n)
        As *= np.exp(dt * eta * L1s)
    return np.asarray(out)


def ex1936(eta=0.05, delta=0.02, T=160.0, dt=0.01):
    print("\nExercises 19.35 and 19.36: learning-by-doing, autarky and trade")
    print(f"  eta = {eta}, A_n(0) = 1, A_s(0) = 1 - delta = {1 - delta}")
    print(f"  {'eps':>6} {'regime':>10} {'L^1_n(T)':>10} {'L^1_s(T)':>10} "
          f"{'dA_n/A_n':>10} {'dA_s/A_s':>10} {'Y_n/Y_s':>12}")
    curves = {}
    for eps in (2.0, 1.0, 0.5):
        a = lbd_autarky(eps, eta, 1.0, T, dt)
        An, L1n = a[-1]
        b = lbd_autarky(eps, eta, 1 - delta, T, dt)
        print(f"  {eps:>6.1f} {'autarky':>10} {L1n:>10.6f} {b[-1, 1]:>10.6f} "
              f"{eta * L1n:>10.6f} {eta * b[-1, 1]:>10.6f} "
              f"{(An / b[-1, 0]) ** ((eps - 1) / eps) if eps != 1 else 1.0:>12.6f}")
        t = lbd_trade(eps, eta, 1.0, 1 - delta, T, dt)
        Ant, Ast, l1n, l1s, omega = t[-1]
        print(f"  {eps:>6.1f} {'trade':>10} {l1n:>10.6f} {l1s:>10.6f} "
              f"{eta * l1n:>10.6f} {eta * l1s:>10.6f} {omega:>12.6f}")
        curves[eps] = (a, t)
    print("  eps > 1: trade locks the South into sector 2, A_s freezes and")
    print("  Y_n/Y_s = A_n^{(eps-1)/eps} diverges; in autarky both countries reach")
    print("  dA/A -> eta and the gap stays bounded.")
    print("  eps = 1: the terms-of-trade loss exactly offsets the productivity gain, so")
    print("  relative incomes are constant even though the South never learns.")
    print("  eps < 1: complete specialization is not sustainable -- p^2/p^1 = A_n^{1/eps}")
    print("  exceeds A_n, so Northern workers move back into sector 2, wages are equalized")
    print("  across blocks and Northern learning dies out.")

    print("\n  infant-industry protection until T0, then free trade for 600 more years:")
    print("  opening with A_s^eps > A_n leaves the South a share")
    print("    L^1_s = (A_s^eps - A_n)/(A_s^eps + A_s),  w_n/w_s = A_n/A_s,")
    print("  and x = A_s^eps/A_n then obeys dlog x/dt = eta[eps L^1_s - 1], so the South")
    print("  keeps its foothold for ever if and only if, at the opening date,")
    print("    x > (eps + A_s/A_n)/(eps - 1)   -- a condition that is self-reinforcing.")
    eps = 2.0
    print(f"  {'T0':>6} {'A_n(T0)':>10} {'A_s(T0)':>10} {'x':>9} {'threshold':>10} "
          f"{'escapes?':>9} {'L^1_s(end)':>11} {'dlog(w_n/w_s)':>14}")
    for T0 in (0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 120.0):
        a_n = lbd_autarky(eps, eta, 1.0, max(T0, dt), dt)[-1, 0] if T0 > 0 else 1.0
        a_s = (lbd_autarky(eps, eta, 1 - delta, max(T0, dt), dt)[-1, 0]
               if T0 > 0 else 1 - delta)
        x, thr = a_s ** eps / a_n, (eps + a_s / a_n) / (eps - 1)
        t = lbd_trade(eps, eta, a_n, a_s, 600.0, dt)
        slope = (np.log(t[-1, 4]) - np.log(t[-2, 4])) / dt
        print(f"  {T0:>6.0f} {a_n:>10.4f} {a_s:>10.4f} {x:>9.4f} {thr:>10.4f} "
              f"{str(bool(x > thr)):>9} {t[-1, 3]:>11.6f} {slope:>14.6f}")
    print(f"  when the South is locked out the wage ratio grows at (eps-1)eta/eps = "
          f"{(eps - 1) * eta / eps:.6f};")
    print("  when it escapes, L^1_s rises to one, both blocks grow at eta and the wage")
    print("  ratio settles at a constant.  So protection of SUFFICIENT length does change")
    print("  the asymptotic outcome, contrary to the remark following Proposition 19.16;")
    print("  short protection only delays the divergence")
    return curves


# ---------------------------------------------------------------------------


def figure(paths, curves):
    ys, Vs, ystar, dt = paths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    tt = np.arange(len(ys)) * dt
    for j in range(ys.shape[1]):
        ax1.plot(tt, ys[:, j], "-", color="black", lw=0.7)
        ax1.axhline(ystar[j], color="0.7", lw=0.4)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$y_j = Y_j/Y$")
    ax1.set_xlim(0, tt[-1])
    ax1.set_title("world income shares converge", fontsize=8)

    a, t = curves[2.0]
    tt2 = np.arange(len(a)) * 0.01
    ax2.plot(tt2, a[:, 1], "-", color="black", label="autarky, $L^1$")
    ax2.plot(tt2, t[:, 2], "--", color="black", label="trade, $L^1_n$")
    ax2.plot(tt2, t[:, 3], ":", color="black", label="trade, $L^1_s$")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("labour in sector 1")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=False, fontsize=6.5, loc="center right")
    ax2.set_title(r"learning-by-doing, $\varepsilon=2$", fontsize=8)
    fig.tight_layout()
    save(fig, "ch19_trade")


def main():
    paths = ex1917()
    ex1918()
    ex1919()
    ex1923()
    ex1930()
    curves = ex1936()
    figure(paths, curves)


if __name__ == "__main__":
    main()
