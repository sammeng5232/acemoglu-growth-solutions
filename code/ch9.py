"""Numerical parts of the Chapter 9 solutions (overlapping generations).

  * Exercise 9.9   the canonical OLG model: the exact condition for dynamic inefficiency,
    whether it can hold at plausible parameters, and an unfunded social security scheme whose
    whole transition path makes every generation strictly better off;
  * Exercise 9.10  the same model when the old also work: overaccumulation survives only if
    old-age labour income is below an explicit threshold;
  * Exercise 9.22  warm glow bequests: overaccumulation requires only alpha < beta/(1+beta),
    a much weaker condition than in the baseline model;
  * Exercises 9.28 and 9.29  the continuous-time perpetual youth model: the eigenvalues of
    the linearised system and the comparative statics of the steady state in n and nu;
  * Exercise 9.30  declining labour income: the threshold zeta above which k* exceeds the
    golden rule;
  * Exercise 9.33  capital income taxation reduces output with infinitely lived households
    and leaves it untouched in the OLG economy.

Run with `python code/ch9.py`.
"""
import numpy as np

from acemoglulib import FIG, bisect, plt, save

# ----------------------------------------------------------------------------------------
# Exercises 9.9 and 9.10: the canonical OLG model
# ----------------------------------------------------------------------------------------
ALPHA, BETA, N = 1 / 3, 0.545, 0.348      # one period = 30 years; see below


def canonical_steady_state(alpha=ALPHA, beta=BETA, n=N):
    """k* = [beta(1-alpha)/((1+n)(1+beta))]^{1/(1-alpha)} and R* = alpha k*^{alpha-1}."""
    kstar = (beta * (1 - alpha) / ((1 + n) * (1 + beta))) ** (1 / (1 - alpha))
    return kstar, alpha * kstar ** (alpha - 1)


def golden_rule(alpha=ALPHA, n=N):
    """f'(k_gold) = 1 + n."""
    return (alpha / (1 + n)) ** (1 / (1 - alpha))


def annual(x, years=30):
    """Turn a 30-year gross factor into an annual net rate."""
    return x ** (1 / years) - 1


def ex99():
    print("Exercise 9.9: dynamic inefficiency in the canonical OLG model")
    print("  r* < n  <=>  alpha(1+beta) < beta(1-alpha)  <=>  beta > alpha/(1-2alpha)")
    print(f"  {'alpha':>6} {'threshold beta':>15} {'annual rho needed':>19}")
    for a in (0.20, 0.25, 0.30, 1 / 3, 0.40):
        if 1 - 2 * a > 0:
            bbar = a / (1 - 2 * a)
            rho = (1 / bbar) ** (1 / 30) - 1 if bbar < 1e6 else np.inf
            note = f"{rho:+.4f}" if bbar <= 1 else f"{rho:+.4f}  (beta>1: impossible)"
            print(f"  {a:>6.3f} {bbar:>15.4f} {note:>19}")
        else:
            print(f"  {a:>6.3f} {'never':>15} {'--':>19}")

    kstar, Rstar = canonical_steady_state()
    kgold = golden_rule()
    print(f"\n  baseline calibration: alpha={ALPHA:.4f}, beta={BETA} "
          f"(= 0.98^30, annual rho = {annual(1/BETA):.4f}), "
          f"n={N} (= 1.01^30-1, annual {annual(1+N):.4f})")
    print(f"    k* = {kstar:.6f}, k_gold = {kgold:.6f}, k*/k_gold = {kstar/kgold:.4f}")
    print(f"    R* = {Rstar:.6f} vs 1+n = {1+N:.6f}: "
          f"r* {'<' if Rstar < 1+N else '>'} n, so the economy is dynamically "
          f"{'INEFFICIENT' if Rstar < 1+N else 'efficient'}")
    print(f"    annualised: r* = {annual(Rstar):.4f} per year vs n = {annual(1+N):.4f}")
    return kstar, kgold


# --- the unfunded social security experiment of Exercise 9.9(d) ---------------------------
def olg_path(alpha, beta, n, d, k0, T=60):
    """Transition of the canonical OLG model with a constant unfunded transfer d.

    Saving solves  s = [beta w - (beta + (1+n)/R') d] / (1+beta)  with R' = alpha k'^{a-1}
    and (1+n)k' = s, so k' is the root of a scalar equation at every date.
    """
    ks = [k0]
    for _ in range(T):
        k = ks[-1]
        w = (1 - alpha) * k ** alpha

        def g(kp):
            R = alpha * kp ** (alpha - 1)
            s = (beta * w - (beta + (1 + n) / R) * d) / (1 + beta)
            return (1 + n) * kp - s

        ks.append(bisect(g, 1e-12, 10 * max(k, 1.0)))
    return np.array(ks)


def generation_utility(alpha, beta, n, d, k, kp):
    """log c1 + beta log c2 for the generation born with capital k, next period kp."""
    w, R = (1 - alpha) * k ** alpha, alpha * kp ** (alpha - 1)
    s = (beta * w - (beta + (1 + n) / R) * d) / (1 + beta)
    c1, c2 = w - s - d, R * s + (1 + n) * d
    return np.log(c1) + beta * np.log(c2), c1, c2


def ex99d():
    """A small constant transfer d makes every generation, and the initial old, better off."""
    alpha, beta, n = 0.25, 0.9, 0.0          # a dynamically inefficient calibration
    kstar, Rstar = canonical_steady_state(alpha, beta, n)
    kgold = golden_rule(alpha, n)
    print(f"\nExercise 9.9(d): a dynamically inefficient calibration "
          f"alpha={alpha}, beta={beta}, n={n}")
    print(f"  beta = {beta} > alpha/(1-2alpha) = {alpha/(1-2*alpha):.4f}, so r* < n")
    print(f"  k* = {kstar:.6f} > k_gold = {kgold:.6f};  R* = {Rstar:.6f} < 1+n = {1+n:.4f}")

    d = 0.02 * (1 - alpha) * kstar ** alpha        # 2% of the wage
    ks = olg_path(alpha, beta, n, d, kstar, T=40)
    U0, _, _ = generation_utility(alpha, beta, n, 0.0, kstar, kstar)
    print(f"  introduce a constant unfunded transfer d = {d:.6f} (2% of the wage) at t = 0")
    print(f"    capital falls from {ks[0]:.6f} to {ks[-1]:.6f} "
          f"(new steady state; k_gold = {kgold:.6f})")
    print(f"    the initial old gain (1+n)d = {(1+n)*d:.6f} outright")
    print(f"    {'t':>3} {'k(t)':>10} {'U(t)':>12} {'U(t)-U*':>12}")
    worst = np.inf
    for t in (0, 1, 2, 3, 5, 10, 20, 39):
        U, _, _ = generation_utility(alpha, beta, n, d, ks[t], ks[t + 1])
        worst = min(worst, U - U0)
        print(f"    {t:>3} {ks[t]:>10.6f} {U:>12.6f} {U-U0:>+12.6f}")
    allU = [generation_utility(alpha, beta, n, d, ks[t], ks[t + 1])[0] - U0
            for t in range(len(ks) - 1)]
    print(f"    smallest gain over the whole transition: {min(allU):+.6f} > 0, "
          f"so every generation is strictly better off")

    # the analytical steady-state result: dU/dk < 0 whenever f'(k) < 1+n
    f, fp, fpp = (lambda k: k ** alpha,
                  lambda k: alpha * k ** (alpha - 1),
                  lambda k: alpha * (alpha - 1) * k ** (alpha - 2))
    C = lambda k: f(k) - (1 + n) * k
    dU = ((1 + beta) * (fp(kstar) - (1 + n)) / C(kstar)
          + beta * fpp(kstar) * ((1 + n) - fp(kstar))
          / (fp(kstar) * ((1 + n) + beta * fp(kstar))))
    print(f"    check the closed form dU/dk at k* : {dU:+.6f} < 0 "
          f"(reducing k raises steady-state utility)")


def ex910():
    """Working when old: overaccumulation needs z below an explicit threshold."""
    alpha, beta, n = 0.25, 0.9, 0.0
    A = lambda z: (beta * (1 - alpha) * alpha
                   / (alpha * (1 + beta) * (1 + n + z) + z * (1 - alpha)))
    khat = lambda z: A(z) ** (1 / (1 - alpha))
    kgold = golden_rule(alpha, n)
    zbar = (1 + n) * (beta * (1 - 2 * alpha) - alpha) / (1 + alpha * beta)
    print(f"\nExercise 9.10: the old supply z units of labour "
          f"(alpha={alpha}, beta={beta}, n={n})")
    print(f"  k(t+1) = A(z) k(t)^alpha with A(z) = alpha beta(1-alpha) / "
          f"[alpha(1+beta)(1+n+z) + z(1-alpha)]")
    print(f"  overaccumulation (k* > k_gold = {kgold:.6f}) iff z < zbar = {zbar:.6f}")
    print(f"  {'z':>6} {'k*':>10} {'k*/k_gold':>11} {'f-prime(k*)':>12} {'1+n':>7}")
    for z in sorted((0.0, 0.1, zbar, 0.2, 0.4, 1.0)):
        k = khat(z)
        print(f"  {z:>6.4f} {k:>10.6f} {k/kgold:>11.4f} "
              f"{alpha*k**(alpha-1):>12.6f} {1+n:>7.4f}")
    print("  the old earning even a fifth of a young worker's labour supply is nearly enough")
    print("  to remove overaccumulation: saving for retirement is the whole mechanism.")


def ex922():
    """Warm glow: overaccumulation iff alpha < beta/(1+beta)."""
    print("\nExercise 9.22: warm glow bequests, k* = [beta/(1+beta)] f(k*)")
    print("  k* > k_gold  <=>  f'(k*) < 1  <=>  alpha(1+beta)/beta < 1  "
          "<=>  alpha < beta/(1+beta)")
    print("  compare the baseline OLG model, where it needs alpha(1+2beta) < beta")
    print(f"  {'beta':>6} {'warm glow: alpha <':>19} {'baseline: alpha <':>19}")
    for b in (0.4, 0.5, 0.6, 0.8, 1.0):
        print(f"  {b:>6.2f} {b/(1+b):>19.4f} {b/(1+2*b):>19.4f}")
    alpha, beta, A = 1 / 3, 0.6, 1.0
    kstar = (beta * A / (1 + beta)) ** (1 / (1 - alpha))
    kgold = (alpha * A) ** (1 / (1 - alpha))
    print(f"  at alpha={alpha:.4f}, beta={beta}, A={A}: k* = {kstar:.6f} > "
          f"k_gold = {kgold:.6f}, f'(k*) = {alpha*A*kstar**(alpha-1):.6f} < 1")
    print("  the warm glow household saves out of TOTAL income, the baseline one only out of")
    print("  wages, which is why overaccumulation is so much easier here.")


# ----------------------------------------------------------------------------------------
# Exercises 9.28-9.30: the continuous-time perpetual youth model
# ----------------------------------------------------------------------------------------
PY = dict(alpha=1 / 3, A=1.0, delta=0.06, rho=0.02, n=0.02, nu=0.02, zeta=0.0)


def py_f(k, p):
    return p["A"] * k ** p["alpha"]


def py_fp(k, p):
    return p["alpha"] * p["A"] * k ** (p["alpha"] - 1)


def py_fpp(k, p):
    return p["alpha"] * (p["alpha"] - 1) * p["A"] * k ** (p["alpha"] - 2)


def py_steady(p):
    """Root of (9.50), generalised to zeta: f/k - (n-nu+delta) - (rho+nu)(n+z)/(f'-delta-rho+z)."""
    z, n, nu, rho, d = p["zeta"], p["n"], p["nu"], p["rho"], p["delta"]
    Phi = lambda k: (py_f(k, p) / k - (n - nu + d)
                     - (rho + nu) * (n + z) / (py_fp(k, p) - d - rho + z))
    # f' - delta - rho + zeta > 0 must hold: k below k_bar
    kbar = (p["alpha"] * p["A"] / (d + rho - z)) ** (1 / (1 - p["alpha"])) if d + rho > z else 1e9
    k = bisect(Phi, 1e-9, 0.999999 * kbar)
    return k, py_f(k, p) - (n - nu + d) * k


def py_jacobian(k, c, p):
    z, n, nu, rho, d = p["zeta"], p["n"], p["nu"], p["rho"], p["delta"]
    return np.array([[py_fp(k, p) - (n - nu + d), -1.0],
                     [c * py_fpp(k, p) - (rho + nu) * (n + z),
                      py_fp(k, p) - d - rho + z]])


def ex928_929():
    p = dict(PY)
    k, c = py_steady(p)
    kmgr = (p["alpha"] * p["A"] / (p["rho"] + p["delta"])) ** (1 / (1 - p["alpha"]))
    kgold = (p["alpha"] * p["A"] / (p["n"] - p["nu"] + p["delta"])) ** (1 / (1 - p["alpha"]))
    J = py_jacobian(k, c, p)
    vals = np.linalg.eigvals(J)
    print(f"\nExercises 9.28 and 9.29: continuous-time perpetual youth, "
          f"alpha={p['alpha']:.4f}, delta={p['delta']}, rho={p['rho']}, "
          f"n={p['n']}, nu={p['nu']}")
    print(f"  (k*, c*) = ({k:.6f}, {c:.6f});  k_mgr = {kmgr:.6f}, k_gold = {kgold:.6f}")
    print(f"  k* < k_mgr < k_gold: underaccumulation, as Proposition 9.10 states")
    print(f"  trace = {np.trace(J):+.6f}, det = {np.linalg.det(J):+.6f} < 0")
    print(f"  eigenvalues {vals[0].real:+.6f} and {vals[1].real:+.6f}: a saddle (Exercise 9.28)")
    print(f"  closed form det = [f'(k*) - f(k*)/k*](f'(k*)-delta-rho) + c* f''(k*) = "
          f"{(py_fp(k,p)-py_f(k,p)/k)*(py_fp(k,p)-p['delta']-p['rho']) + c*py_fpp(k,p):+.6f}")

    print("\n  Exercise 9.29, comparative statics (numerical derivatives):")
    h = 1e-6
    for key in ("n", "nu"):
        hi, lo = dict(p), dict(p)
        hi[key], lo[key] = p[key] + h, p[key] - h
        (k1, c1), (k0, c0) = py_steady(hi), py_steady(lo)
        print(f"    d k*/d {key:<3} = {(k1-k0)/(2*h):+10.4f},   "
              f"d c*/d {key:<3} = {(c1-c0)/(2*h):+10.4f}")
    # the analytical sign of dPhi/dnu
    omega = c / (p["rho"] + p["nu"]) - k
    print(f"    check: dPhi/dnu = -omega*/k* = {-omega/k:+.6f} < 0, so dk*/dnu < 0")
    print(f"    (omega* = c*/(rho+nu) - k* = {omega:.6f} is steady-state human wealth)")
    return p, k, c, kmgr, kgold


def ex930(p):
    """Declining labour income: the threshold zeta at which k* reaches k_gold."""
    kgold = (p["alpha"] * p["A"] / (p["n"] - p["nu"] + p["delta"])) ** (1 / (1 - p["alpha"]))
    n, nu, rho, d, a, A = p["n"], p["nu"], p["rho"], p["delta"], p["alpha"], p["A"]
    print(f"\nExercise 9.30: labour income declining at the rate zeta "
          f"(k_gold = {kgold:.6f})")
    print(f"  {'zeta':>7} {'k*':>10} {'k*/k_mgr':>10} {'k*/k_gold':>11}")
    kmgr = (a * A / (rho + d)) ** (1 / (1 - a))
    for z in (0.0, 0.02, 0.05, 0.10, 0.25, 1.0, 10.0):
        q = dict(p, zeta=z)
        k, _ = py_steady(q)
        print(f"  {z:>7.3f} {k:>10.6f} {k/kmgr:>10.4f} {k/kgold:>11.4f}")
    g = lambda z: py_steady(dict(p, zeta=z))[0] - kgold
    zbar = bisect(g, 1e-6, 50.0)
    print(f"  k* = k_gold at zeta = {zbar:.6f}; above it the economy overaccumulates")
    # the zeta -> infinity limit
    kinf = (A / (n + d + rho)) ** (1 / (1 - a))
    print(f"  as zeta -> infinity, f(k)/k -> n+delta+rho, so k* -> {kinf:.6f} "
          f"and f'(k*) -> {a*(n+d+rho):.6f}")
    print(f"    this exceeds k_gold iff alpha(n+delta+rho) = {a*(n+d+rho):.6f} < "
          f"n-nu+delta = {n-nu+d:.6f}: {'yes' if a*(n+d+rho) < n-nu+d else 'no'}")
    return zbar


def ex933():
    """Capital income taxation: Ramsey versus OLG."""
    alpha, beta, A = 2 / 3, 0.98 ** 30, 1.0     # Y = A K^{1-alpha} L^alpha, capital share 1-alpha
    print(f"\nExercise 9.33: Y = A K^(1-alpha) L^alpha with alpha = {alpha:.4f} "
          f"(capital share {1-alpha:.4f}), beta = {beta:.4f}")
    K_ramsey = lambda tau: (beta * (1 - tau) * (1 - alpha) * A) ** (1 / alpha)
    K_olg = lambda tau: (beta * alpha * A / (1 + beta)) ** (1 / alpha)
    Y = lambda K: A * K ** (1 - alpha)
    print(f"  {'tau':>6} {'K* (Ramsey)':>13} {'Y* (Ramsey)':>13} "
          f"{'K* (OLG)':>11} {'Y* (OLG)':>11}")
    for tau in (0.0, 0.1, 0.25, 0.5):
        print(f"  {tau:>6.2f} {K_ramsey(tau):>13.6f} {Y(K_ramsey(tau)):>13.6f} "
              f"{K_olg(tau):>11.6f} {Y(K_olg(tau)):>11.6f}")
    print(f"  Ramsey: d log Y*/d log(1-tau) = (1-alpha)/alpha = {(1-alpha)/alpha:.4f}; "
          f"OLG: 0 exactly")
    print(f"  going from tau=0 to tau=0.5 costs the Ramsey economy "
          f"{100*(Y(K_ramsey(0.5))/Y(K_ramsey(0))-1):+.2f}% of output and the OLG economy "
          f"nothing")


def figure(p, kstar, cstar, kmgr, kgold, zbar):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    kk = np.linspace(1e-4, 1.35 * kgold, 500)
    ax.plot(kk, py_f(kk, p) - (p["n"] - p["nu"] + p["delta"]) * kk,
            color="#993300", ls="--", lw=1.0, label=r"$\dot k=0$")
    den = py_fp(kk, p) - p["delta"] - p["rho"]
    ok = den > 1e-9
    ax.plot(kk[ok], (p["rho"] + p["nu"]) * p["n"] * kk[ok] / den[ok],
            color="#003399", lw=1.0, label=r"$\dot c=0$")
    ax.plot([kstar], [cstar], "o", ms=3.5, color="black")
    for x, lab, dx in ((kstar, r"$k^*$", -14), (kmgr, r"$k_{\rm mgr}$", 2),
                       (kgold, r"$k_{\rm gold}$", 2)):
        ax.axvline(x, color="#999999", lw=0.5, ls=":")
        ax.annotate(lab, (x, 0), textcoords="offset points", xytext=(dx, 3), fontsize=7)
    ax.set_xlim(0, 1.35 * kgold)
    ax.set_ylim(0, 1.25 * (py_f(kgold, p) - (p["n"] - p["nu"] + p["delta"]) * kgold))
    ax.set_xlabel("$k$")
    ax.set_ylabel("$c$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    zs = np.linspace(0.0, 0.30, 300)
    ks = np.array([py_steady(dict(p, zeta=z))[0] for z in zs])
    ax.plot(zs, ks, color="#003399", lw=1.2)
    ax.axhline(kgold, color="#993300", ls="--", lw=0.9)
    ax.axhline(kmgr, color="#666666", ls=":", lw=0.9)
    ax.annotate(r"$k_{\rm gold}$", (0.30, kgold), textcoords="offset points",
                xytext=(-30, 3), fontsize=7, color="#993300")
    ax.annotate(r"$k_{\rm mgr}$", (0.30, kmgr), textcoords="offset points",
                xytext=(-28, 3), fontsize=7, color="#666666")
    ax.plot([zbar], [kgold], "o", ms=3.5, color="black")
    ax.annotate(rf"$\zeta={zbar:.2f}$", (zbar, kgold), textcoords="offset points",
                xytext=(4, -11), fontsize=7)
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel("$k^*$")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 1.15 * max(ks.max(), kgold))
    save(fig, "ch9_perpetual_youth")


def main():
    ex99()
    ex99d()
    ex910()
    ex922()
    p, k, c, kmgr, kgold = ex928_929()
    zbar = ex930(p)
    ex933()
    figure(p, k, c, kmgr, kgold, zbar)


if __name__ == "__main__":
    main()
