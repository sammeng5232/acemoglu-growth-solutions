"""Numerical parts of the Chapter 17 solutions (stochastic growth models).

  * Exercises 17.1 and 17.4  when the consumption and capital policies are increasing
    in k and in z, and why the capital-labour ratio never settles down;
  * Exercise 17.3   the stochastic Solow rule against the optimal rule;
  * Exercise 17.8   delta < 1 destroys the closed form of Example 17.1;
  * Exercises 17.25 and 17.26  the condition Q >= (2 - gamma) q, uniqueness of n*,
    and a check of the discriminant printed in (17.57);
  * Exercise 17.28  the variance of TFP and the threshold gamma = Q/(2Q - q).

The Brock-Mirman policy functions are computed by the endogenous grid method, which
returns a smooth consumption policy and therefore lets the savings rate be read off
directly.  Run with `python code/ch17.py`.
"""
import numpy as np

from acemoglulib import FIG, plt, save  # noqa: F401

# ---------------------------------------------------------------------------
# A discretized AR(1) for log z (Tauchen), and the Brock-Mirman policy functions.
# ---------------------------------------------------------------------------


def tauchen(rho, sigma, n=5, m=3.0):
    """n-state monotone Markov chain for log z ~ AR(1); returns (z, P) with
    P[j, j'] = Pr[z' = z_j' | z = z_j]."""
    from scipy.stats import norm

    s = sigma / np.sqrt(1 - rho ** 2)
    grid = np.linspace(-m * s, m * s, n)
    step = grid[1] - grid[0]
    P = np.empty((n, n))
    for j in range(n):
        mu = rho * grid[j]
        P[j, 0] = norm.cdf((grid[0] - mu + step / 2) / sigma)
        P[j, -1] = 1 - norm.cdf((grid[-1] - mu - step / 2) / sigma)
        for k in range(1, n - 1):
            P[j, k] = (norm.cdf((grid[k] - mu + step / 2) / sigma)
                       - norm.cdf((grid[k] - mu - step / 2) / sigma))
    return np.exp(grid), P


def stationary(P):
    w, v = np.linalg.eig(P.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    return p / p.sum()


def interp_extrap(xq, xp, fp):
    """Linear interpolation on (xp, fp), extended linearly beyond both ends."""
    y = np.interp(xq, xp, fp)
    s0 = (fp[1] - fp[0]) / (xp[1] - xp[0])
    s1 = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(xq < xp[0], fp[0] + s0 * (xq - xp[0]), y)
    y = np.where(xq > xp[-1], fp[-1] + s1 * (xq - xp[-1]), y)
    return y


def kdet(z, alpha, beta, delta):
    """Deterministic steady-state capital at productivity z."""
    return (alpha * beta * z / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))


def egm(alpha, beta, delta, theta, z, P, nk=600, tol=1e-12, maxit=20000):
    """Consumption policy c(k, z) of the Brock-Mirman problem, by the endogenous
    grid method.  Returns (kgrid, c, core) where `core` is the slice of the grid
    that lies inside the ergodic range and is free of boundary error."""
    nz = len(z)
    lo, hi = kdet(z[0], alpha, beta, delta), kdet(z[-1], alpha, beta, delta)
    kgrid = np.exp(np.linspace(np.log(lo / 50), np.log(hi * 6), nk))
    x_of_k = z[None, :] * kgrid[:, None] ** alpha + (1 - delta) * kgrid[:, None]

    c = 0.3 * x_of_k.copy()
    for it in range(maxit):
        R = alpha * z[None, :] * kgrid[:, None] ** (alpha - 1) + (1 - delta)
        cstar = (beta * (R * c ** (-theta)) @ P.T) ** (-1 / theta)
        xstar = cstar + kgrid[:, None]
        cnew = np.empty_like(c)
        for j in range(nz):
            cnew[:, j] = interp_extrap(x_of_k[:, j], xstar[:, j], cstar[:, j])
        # the constraint k' >= kgrid[0] binds where the unconstrained rule wants less
        cnew = np.minimum(cnew, x_of_k - kgrid[0])
        cnew = np.maximum(cnew, 1e-14)
        gap = np.max(np.abs(np.log(cnew) - np.log(c)))
        c = cnew
        if gap < tol:
            break
    core = (kgrid > lo / 3) & (kgrid < hi * 3)
    return kgrid, c, core, it, gap


def policies(alpha, beta, delta, theta, z, P, **kw):
    """Returns (k, c, kprime, savings rate, core mask)."""
    kgrid, c, core, it, gap = egm(alpha, beta, delta, theta, z, P, **kw)
    x = z[None, :] * kgrid[:, None] ** alpha + (1 - delta) * kgrid[:, None]
    kp = x - c
    return kgrid, c, kp, kp / x, core


# ---------------------------------------------------------------------------
# Exercises 17.1 and 17.4
# ---------------------------------------------------------------------------


def slopes(k, f, core, z=None):
    """Smallest forward difference of f along k (or along z) on the core range."""
    i = np.where(core)[0]
    i = i[:-1]
    if z is None:
        return (np.diff(f, axis=0)[i] / np.diff(k)[i, None]).min()
    return (np.diff(f, axis=1)[i] / np.diff(z)[None, :]).min()


def ex171(alpha=0.36, beta=0.95, delta=1.0, rho=0.95, sigma=0.30):
    print("Exercise 17.1 / 17.4: monotonicity of the policy functions")
    print(f"  monotone chain log z' = {rho} log z + eps, sd(eps) = {sigma}, 5 states")
    z, P = tauchen(rho, sigma, 5)
    print(f"  {'theta':>7} {'min dpi/dk':>11} {'min dc/dk':>11} "
          f"{'min dpi/dz':>11} {'min dc/dz':>11} {'c up in z?':>11}")
    for theta in (0.10, 0.20, 0.35, 0.50, 1.00, 2.00, 5.00):
        k, c, kp, _, core = policies(alpha, beta, delta, theta, z, P)
        a = slopes(k, kp, core)
        b = slopes(k, c, core)
        d = slopes(k, kp, core, z)
        e = slopes(k, c, core, z)
        print(f"  {theta:>7.2f} {a:>11.5f} {b:>11.5f} {d:>11.4f} {e:>11.4f} "
              f"{'yes' if e > 0 else 'NO':>11}")

    def g(th):
        k, c, kp, _, core = policies(alpha, beta, delta, th, z, P)
        return slopes(k, c, core, z)

    lo, hi = 0.10, 0.50
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    print(f"  the cut-off is theta = {0.5 * (lo + hi):.4f}, an elasticity of "
          f"intertemporal substitution of {1 / (0.5 * (lo + hi)):.3f}")
    print("  pi is increasing in k and in z at every theta (Proposition 17.2);")
    print("  c is increasing in k at every theta, as the concavity of V alone implies.")
    print("  Monotonicity of c in z is the one that can fail, and it fails when the")
    print("  elasticity of intertemporal substitution 1/theta is large: a good shock is")
    print("  then mostly an investment opportunity, and the planner saves more than the")
    print("  whole increase in output.")


def ex174(alpha=0.36, beta=0.95, delta=1.0, theta=1.0, T=120000, seed=7):
    print("\nExercise 17.4(b): the capital-labour ratio does not converge")
    rho = 0.90
    ratio = np.sqrt((1 + alpha * rho) / ((1 - alpha ** 2) * (1 - alpha * rho)))
    print(f"  in Example 17.1, log k(t+1) = log(alpha beta) + log z(t) + alpha log k(t),")
    print(f"  so sd(log k)/sd(log z) = sqrt[(1+a rho)/((1-a^2)(1-a rho))] = {ratio:.4f}")
    print(f"  {'sd(log z)':>10} {'sd(log k)':>11} {'range log k':>12} "
          f"{'sd ratio':>10}")
    rng = np.random.default_rng(seed)
    for sigma in (0.40, 0.20, 0.05, 0.005):
        z, P = tauchen(rho, sigma, 5)
        k, c, kp, _, core = policies(alpha, beta, delta, theta, z, P)
        pz = stationary(P)
        sdz = np.sqrt(pz @ np.log(z) ** 2 - (pz @ np.log(z)) ** 2)
        cum = P.cumsum(axis=1)
        j, kk = 2, kdet(z[2], alpha, beta, delta)
        lk, lc, ly = [], [], []
        for t in range(T):
            y = z[j] * kk ** alpha
            x = y + (1 - delta) * kk
            cc = min(np.interp(kk, k, c[:, j]), x - k[0])
            if t > 2000:
                lk.append(np.log(kk)); lc.append(np.log(cc)); ly.append(np.log(y))
            kk = x - cc
            j = int(np.searchsorted(cum[j], rng.random()))
        lk, lc, ly = map(np.asarray, (lk, lc, ly))
        print(f"  {sdz:>10.4f} {lk.std():>11.5f} {lk.max() - lk.min():>12.5f} "
              f"{lk.std() / sdz:>10.4f}")
    print("  the invariant distribution of k is nondegenerate for every nondegenerate z,")
    print("  and collapses to a point only as the support of z does; the simulated ratio")
    print("  reproduces the analytical one, so the dispersion of k is proportional to")
    print("  that of z and cannot vanish while z is nondegenerate")


def ex173(alpha=0.36, beta=0.95, delta=0.08, theta=2.0, sigma=0.12, T=120000, seed=11):
    print("\nExercise 17.3: a constant saving rate against the optimal rule")
    z, P = tauchen(0.90, sigma, 5)
    k, c, kp, srate, core = policies(alpha, beta, delta, theta, z, P)
    rng = np.random.default_rng(seed)
    cum = P.cumsum(axis=1)

    def simulate(rule):
        j, kk = 2, kdet(z[2], alpha, beta, delta)
        ks, cs, ys = [], [], []
        r = np.random.default_rng(seed)          # the same shocks for both rules
        for t in range(T):
            y = z[j] * kk ** alpha
            x = y + (1 - delta) * kk
            cc = min(max(rule(kk, j, y), 1e-10), x - k[0])
            if t > 2000:
                ks.append(kk); cs.append(cc); ys.append(y)
            kk = x - cc
            j = int(np.searchsorted(cum[j], r.random()))
        return map(np.asarray, (ks, cs, ys))

    ks, cs, ys = simulate(lambda kk, j, y: np.interp(kk, k, c[:, j]))
    inv = np.diff(np.append(ks, ks[-1])) + delta * ks
    sbar = float(inv[:-1].mean() / ys[:-1].mean())
    iy = inv[:-1] / ys[:-1]
    print(f"  along the optimal path the investment rate i/y averages {sbar:.5f} and")
    print(f"  ranges over [{iy.min():.5f}, {iy.max():.5f}]: it is not constant")
    rows = [("optimal", ks, cs, ys)]
    ks2, cs2, ys2 = simulate(lambda kk, j, y: (1 - sbar) * y)
    rows.append(("Solow  ", ks2, cs2, ys2))
    for name, a, b, d in rows:
        print(f"  {name}: mean log k {np.log(a).mean():>7.4f}, "
              f"sd(log k) {np.log(a).std():.5f}, "
              f"sd(log c)/sd(log y) {np.log(b).std() / np.log(d).std():.4f}")
    print(f"  the Solow rule invests the same average fraction {sbar:.5f} of output but does")
    print("  so in every state, so consumption inherits the whole volatility of output;")
    print("  the optimal rule smooths consumption relative to output.  The two rules")
    print("  coincide exactly in Example 17.1 (log utility, Cobb-Douglas, delta = 1),")
    print(f"  where the optimal savings rate is the constant alpha*beta = {alpha * beta:.5f}")


# ---------------------------------------------------------------------------
# Exercise 17.8
# ---------------------------------------------------------------------------


def ex178(alpha=0.36, beta=0.95, sigma=0.25):
    print("\nExercise 17.8: the closed form of Example 17.1 survives only at delta = 1")
    z, P = tauchen(0.90, sigma, 5)
    print(f"  {'delta':>7} {'min s(k,z)':>12} {'max s(k,z)':>12} {'spread':>11} "
          f"{'alpha*beta':>11}")
    out = {}
    for delta in (1.0, 0.9, 0.5, 0.2):
        k, c, kp, srate, core = policies(alpha, beta, 1.0 if delta == 1.0 else delta,
                                         1.0, z, P)
        lo, hi = srate[core].min(), srate[core].max()
        out[delta] = (k, srate, core)
        print(f"  {delta:>7.2f} {lo:>12.6f} {hi:>12.6f} {hi - lo:>11.3e} "
              f"{alpha * beta:>11.6f}")
    print("  at delta = 1 the savings rate equals alpha*beta at every (k, z) to grid")
    print("  accuracy; for delta < 1 it varies with k by orders of magnitude more, so no")
    print("  policy of the form B0 + B1 z k^alpha can solve the Euler equation")
    return out


# ---------------------------------------------------------------------------
# Exercises 17.25, 17.26 and 17.28
# ---------------------------------------------------------------------------


def crossings(Q, q, gam, D, s, n=2000001):
    grid = np.linspace(gam + 1e-12, 1 - 1e-12, n)
    phi = (Q - q) * s / (Q - q * grid) - D * (grid - gam) / (1 - gam)
    idx = np.where(np.sign(phi[:-1]) * np.sign(phi[1:]) < 0)[0]
    return grid[idx]


def ex1725(D=1.0, s=0.4):
    print("\nExercise 17.25: Q >= (2 - gamma) q and the uniqueness of the intersection")
    print("  M(n) = D (n - gamma)/(1 - gamma), I*(n) = (Q - q) s/(Q - q n); at every")
    print("  crossing, M'(n) - I*'(n) has the sign of Q + q gamma - 2 q n, so a single")
    print("  crossing is guaranteed whenever (Q + q gamma)/(2q) >= 1, i.e. Q >= (2 - gamma) q.")
    print("  Phi(n) = I*(n) - M(n) has Phi'(n) = q(Q-q)s/(Q-qn)^2 - D/(1-gamma) increasing")
    print("  in n, so Phi is convex and has at most two zeros; the condition puts every")
    print("  zero on the falling branch, which leaves exactly one.")
    print(f"  {'Q':>6} {'q':>6} {'gamma':>7} {'D':>6} {'s':>6} {'(2-g)q':>8} "
          f"{'holds?':>7} {'crossings':>10}")
    cases = [(2.0, 1.0, 0.3, D, s), (2.0, 1.0, 0.0, D, s), (2.0, 1.0, 0.3, D, 1.3 * D),
             (1.4, 1.0, 0.3, D, s), (1.2, 1.0, 0.0, D, 1.2 * D)]
    for (Q, q, gam, DD, ss) in cases:
        cr = crossings(Q, q, gam, DD, ss)
        print(f"  {Q:>6.3f} {q:>6.2f} {gam:>7.3f} {DD:>6.3f} {ss:>6.3f} "
              f"{(2 - gam) * q:>8.3f} {'yes' if Q >= (2 - gam) * q else 'no':>7} "
              f"{len(cr):>10d}")
        if len(cr) > 1:
            print(f"      roots {np.round(cr, 5)}; (Q + q gamma)/(2q) = "
                  f"{(Q + q * gam) / (2 * q):.5f}, and the second root lies above it, so")
            print(f"      it is an upcrossing: the equilibria are n = {cr[0]:.5f} and n = 1")
    print("  a second crossing needs both Q < (2 - gamma) q and s > D; the condition rules")
    print("  the first out, and when s >= D it leaves no interior crossing at all, so the")
    print("  unique equilibrium is n* = 1.  The condition is sufficient, not necessary.")


def n_star(K, Q, q, gam, D, Gam, alpha, printed=False):
    """Threshold sector (17.57).  `printed` uses the discriminant (Q+q)^2 as printed
    in the book; otherwise (Q + q gamma)^2, which is what the quadratic gives."""
    C = (Q - q) * (1 - gam) * Gam * K ** alpha / D
    disc = ((Q + q) ** 2 if printed else (Q + q * gam) ** 2) - 4 * q * (gam * Q + C)
    if disc < 0:                 # K above K_bar: every sector can be funded
        return 1.0
    return min(((Q + q * gam) - np.sqrt(disc)) / (2 * q), 1.0)


def ex1726(Q=2.0, q=1.0, gam=0.3, D=0.15, alpha=0.36, beta=0.95):
    print("\nExercise 17.26: the discriminant of (17.57)")
    Gam = (1 - alpha) * beta / (1 + beta)
    print(f"  Gamma = (1-alpha)beta/(1+beta) = {Gam:.6f}, D = {D}")
    print(f"  {'K':>10} {'n* (quadratic)':>16} {'n* as printed':>15} "
          f"{'M(n*) - I*(n*)':>16}")
    for K in (1e-8, 0.01, 0.1, 0.4, 1.0):
        n1 = n_star(K, Q, q, gam, D, Gam, alpha)
        n2 = n_star(K, Q, q, gam, D, Gam, alpha, printed=True)
        s = Gam * K ** alpha
        resid = D * (n1 - gam) / (1 - gam) - (Q - q) * s / (Q - q * n1)
        print(f"  {K:>10.2e} {n1:>16.8f} {n2:>15.8f} {resid:>16.2e}")
    print(f"  as K -> 0 the quadratic gives n* -> gamma = {gam}: only the sectors with no")
    print("  minimum size requirement can open, which is right.  The printed discriminant")
    print("  (Q + q)^2 gives a negative limit, so it is a misprint for (Q + q gamma)^2;")
    print("  the residual column confirms the corrected root solves M(n) = I*(n).")
    Kbar = (D / Gam) ** (1 / alpha)
    Kss = (Gam * Q) ** (1 / (1 - alpha))
    rhs = Gam ** (1 / (1 - alpha)) * Q ** (alpha / (1 - alpha))
    print(f"  K_bar = (D/Gamma)^(1/alpha) = {Kbar:.6f}, "
          f"K_SS = (Gamma Q)^(1/(1-alpha)) = {Kss:.6f}")
    print(f"  (17.61) reads D < Gamma^(1/(1-alpha)) Q^(alpha/(1-alpha)) = {rhs:.6f}: "
          f"{'satisfied' if D < rhs else 'violated'}, and it is exactly K_SS >= K_bar")


def ex1728(Q=2.0, q=1.0):
    print("\nExercise 17.28: V_n = n(1-n)[Q(Q-q)/(Q-qn)]^2 is single-peaked")
    print("  d log V_n/dn = [Q - n(2Q - q)]/[n(1-n)(Q - q n)], so V_n peaks at")
    print(f"  n_hat = Q/(2Q - q) = {Q / (2 * Q - q):.6f}, which always lies in (1/2, 1).")
    n = np.linspace(1e-9, 1 - 1e-9, 2000001)
    V = n * (1 - n) * (Q * (Q - q) / (Q - q * n)) ** 2
    print(f"  numerical argmax over a fine grid: {n[V.argmax()]:.6f}")
    print(f"  {'gamma':>7} {'n_hat':>9} {'gamma >= n_hat?':>16} {'sign of dV_n/dK':>26}")
    nh = Q / (2 * Q - q)
    for gam in (0.2, 0.5, 0.66, 0.70, 0.9):
        print(f"  {gam:>7.2f} {nh:>9.6f} {'yes' if gam >= nh else 'no':>16} "
              f"{('<= 0 for all K' if gam >= nh else '> 0 below K~, <= 0 above'):>26}")
    print("  n*[K] is increasing in K and never below gamma, so gamma >= n_hat keeps the")
    print("  economy permanently on the decreasing branch; otherwise K~ solves n*[K~] = n_hat")
    return n, V


# ---------------------------------------------------------------------------


def figure(srates, nV, Q=2.0):
    n, V = nV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    for delta, style in ((1.0, "-"), (0.5, "--"), (0.2, ":")):
        k, s, core = srates[delta]
        ax1.plot(k[core], s[core, 2], style, color="black", label=rf"$\delta={delta:g}$")
    ax1.set_xlabel("$k$")
    ax1.set_ylabel(r"$\pi/[f+(1-\delta)k]$")
    ax1.set_xscale("log")
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("savings rate at the median $z$", fontsize=8)

    for q, style in ((1.0, "-"), (1.6, "--")):
        Vq = n * (1 - n) * (Q * (Q - q) / (Q - q * n)) ** 2
        ax2.plot(n, Vq / Vq.max(), style, color="black", label=f"$q={q:g}$")
        ax2.axvline(Q / (2 * Q - q), color="0.6", lw=0.5)
    ax2.set_xlabel("$n^{*}$")
    ax2.set_ylabel("$V_n$ (normalized)")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_title(r"TFP variance, peak at $Q/(2Q-q)$", fontsize=8)
    fig.tight_layout()
    save(fig, "ch17_stochastic_growth")


def main():
    ex171()
    ex174()
    ex173()
    srates = ex178()
    ex1725()
    ex1726()
    nV = ex1728()
    figure(srates, nV)


if __name__ == "__main__":
    main()
