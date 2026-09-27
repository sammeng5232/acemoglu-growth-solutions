"""Numerical parts of the Chapter 2 solutions.

  * Exercise 2.4   the quartic per capita production function f(k) = k^4 - 6k^3 + 11k^2 - 6k:
                   the three interior steady states, the threshold on (n + delta)/s, and the
                   stability pattern along the phase line.
  * Exercise 2.13  saving out of labour income only: an explicit technology satisfying
                   Assumptions 1 and 2 with three interior steady states.

Run with `python code/ch2.py`; it prints the numbers quoted in the solutions and writes the
figures used there.
"""
import numpy as np

from acemoglulib import bisect, plt, save

# ---------------------------------------------------------------------------
# Exercise 2.4
# ---------------------------------------------------------------------------


def f24(k):
    return k ** 4 - 6 * k ** 3 + 11 * k ** 2 - 6 * k


def f24p(k):
    return 4 * k ** 3 - 18 * k ** 2 + 22 * k - 6


def f24pp(k):
    return 12 * k ** 2 - 36 * k + 22


def avg24(k):          # f(k)/k = (k-1)(k-2)(k-3)
    return (k - 1) * (k - 2) * (k - 3)


def wage24(k):         # w = f(k) - k f'(k) = -k^2 (3k^2 - 12k + 11)
    return f24(k) - k * f24p(k)


def exercise_24():
    print("Exercise 2.4")
    # (a) the violations
    print(f"  f(0.5) = {f24(0.5):+.4f}   f(2.5) = {f24(2.5):+.4f}   (output negative)")
    print(f"  f'(0)  = {f24p(0):+.4f}   f''(0) = {f24pp(0):+.4f}")
    print(f"  w(0.5) = {wage24(0.5):+.4f}   w(2)   = {wage24(2.0):+.4f}   w(3) = {wage24(3.0):+.4f}")
    print(f"  limits: f'(k) -> {f24p(1e-9):+.4f} as k->0,  f'(10) = {f24p(10):.1f} (grows without bound)")

    # (b) the cubic f(k)/k = (k-1)(k-2)(k-3): local max and min
    kmax = 2 - 1 / np.sqrt(3)
    kmin = 2 + 1 / np.sqrt(3)
    hmax = avg24(kmax)
    print(f"  f(k)/k has a local max at k = 2 - 1/sqrt(3) = {kmax:.6f}, value {hmax:.6f}"
          f"  (= 2*sqrt(3)/9 = {2 * np.sqrt(3) / 9:.6f})")
    print(f"                a local min at k = 2 + 1/sqrt(3) = {kmin:.6f}, value {avg24(kmin):.6f}")

    # three interior steady states for 0 < h < 2 sqrt(3)/9, e.g. delta/s = 1/4 with n = 0
    h = 0.25
    roots = [bisect(lambda k: avg24(k) - h, a, b) for a, b in [(1.0, kmax), (kmax, 2.0), (3.0, 6.0)]]
    print(f"  with (n+delta)/s = {h}: k* = " + ", ".join(f"{r:.6f}" for r in roots))
    for r in roots:                       # d(kdot)/dk at a steady state has the sign of (f/k)'
        slope = (avg24(r + 1e-6) - avg24(r - 1e-6)) / 2e-6
        print(f"    k* = {r:.6f}: d(f/k)/dk = {slope:+.4f}  ->  "
              f"{'stable' if slope < 0 else 'unstable'}")
    print(f"  k = 0 is also a rest point, and it is locally stable: f(k)/k - h = {avg24(0.5) - h:+.4f} < 0 at k = 0.5")

    # figure: the phase line
    kk = np.linspace(0.01, 4.0, 900)
    fig, ax = plt.subplots()
    ax.plot(kk, avg24(kk), color="#003399", label=r"$f(k)/k=(k-1)(k-2)(k-3)$")
    ax.axhline(h, color="#993300", lw=1.0, ls="--", label=r"$(n+\delta)/s$")
    ax.axhline(0, color="black", lw=0.5)
    for j, r in enumerate(roots):
        ax.plot([r], [h], "o", ms=4, color="#993300")
        off = [(-42, 6), (4, -14), (-14, 10)][j]     # keep the three labels apart
        ax.annotate(f"$k^*_{j+1}={r:.3f}$", (r, h), textcoords="offset points",
                    xytext=off, fontsize=7)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$f(k)/k$")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "ch2_quartic_steady_states")


# ---------------------------------------------------------------------------
# Exercise 2.13   saving out of labour income only
#
#   f(k) = A [ gamma k^rho + (1-gamma) ]^{1/rho} + B k^alpha ,  rho = (sigma-1)/sigma < 0,
#
# a CES with a low elasticity of substitution plus a small Cobb-Douglas term.  The CES part
# alone satisfies Assumption 1 but fails the Inada condition at k = 0; the Cobb-Douglas term
# restores it, so f satisfies Assumptions 1 and 2.
# ---------------------------------------------------------------------------

PAR = dict(A=1.0, gamma=0.5, sigma=0.25, B=0.02, alpha=0.5)


def ces_parts(k, A, gamma, sigma, B, alpha):
    rho = (sigma - 1) / sigma
    br = gamma * k ** rho + (1 - gamma)
    f_ces = A * br ** (1 / rho)
    fp_ces = A * gamma * k ** (rho - 1) * br ** (1 / rho - 1)
    w_ces = f_ces - k * fp_ces                    # = A(1-gamma) br^{(1-rho)/rho}
    return f_ces, fp_ces, w_ces


def f13(k, **p):
    f_ces, _, _ = ces_parts(k, **p)
    return f_ces + p["B"] * k ** p["alpha"]


def fp13(k, **p):
    _, fp_ces, _ = ces_parts(k, **p)
    return fp_ces + p["B"] * p["alpha"] * k ** (p["alpha"] - 1)


def w13(k, **p):
    _, _, w_ces = ces_parts(k, **p)
    return w_ces + (1 - p["alpha"]) * p["B"] * k ** p["alpha"]


def exercise_13():
    print("\nExercise 2.13")
    p = PAR
    print("  f(k) = A[gamma k^rho + (1-gamma)]^(1/rho) + B k^alpha with "
          f"A={p['A']}, gamma={p['gamma']}, sigma={p['sigma']} (rho={(p['sigma']-1)/p['sigma']:.0f}), "
          f"B={p['B']}, alpha={p['alpha']}")

    # Assumption 1 and 2 checks
    ks = np.array([1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e6])
    print("     k        f(k)        f'(k)       f''(k)      w(k)")
    for k in ks:
        h = 1e-6 * max(k, 1e-6)
        fpp = (fp13(k + h, **p) - fp13(max(k - h, 1e-12), **p)) / (2 * h)
        print(f"  {k:9.1e} {f13(k, **p):11.5f} {fp13(k, **p):11.4f} {fpp:11.4f} {w13(k, **p):10.5f}")

    # steady states of  s w(k)/k = n + delta
    phi = lambda k: w13(k, **p) / k
    grid = np.exp(np.linspace(np.log(1e-6), np.log(1e4), 4000))
    vals = phi(grid)
    # interior local extrema of phi
    d = np.diff(vals)
    turns = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0] + 1
    print("  phi(k) = w(k)/k turning points: " +
          ", ".join(f"k={grid[i]:.4f} (phi={vals[i]:.5f})" for i in turns))

    h = 0.045                       # = (n + delta)/s
    roots = []
    for i in range(len(grid) - 1):
        if (vals[i] - h) * (vals[i + 1] - h) < 0:
            roots.append(bisect(lambda k: phi(k) - h, grid[i], grid[i + 1]))
    print(f"  with (n+delta)/s = {h}:  k* = " + ", ".join(f"{r:.5f}" for r in roots))
    for r in roots:
        slope = (phi(r * 1.000001) - phi(r * 0.999999)) / (r * 2e-6)
        print(f"    k* = {r:9.5f}: phi'(k*) = {slope:+.5f}  ->  "
              f"{'stable' if slope < 0 else 'unstable'}")

    # the characterisation  phi'(k) > 0  <=>  sigma(k) < capital share
    print("  check of phi'(k) > 0  <=>  sigma(k) < alpha_K(k):")
    for k in [0.05, 0.3, 1.0, 3.0]:
        hh = 1e-6 * k
        fpp = (fp13(k + hh, **p) - fp13(k - hh, **p)) / (2 * hh)
        sig = fp13(k, **p) * w13(k, **p) / (-k * f13(k, **p) * fpp)
        aK = k * fp13(k, **p) / f13(k, **p)
        slope = (phi(k * 1.000001) - phi(k * 0.999999)) / (k * 2e-6)
        print(f"    k={k:5.2f}: sigma(k)={sig:6.4f}  alpha_K={aK:6.4f}  phi'={slope:+9.5f}"
              f"   ({'sigma < alpha_K' if sig < aK else 'sigma > alpha_K'})")

    kk = np.exp(np.linspace(np.log(1e-3), np.log(50), 900))
    fig, ax = plt.subplots()
    ax.plot(kk, phi(kk), color="#003399", label=r"$w(k)/k$")
    ax.axhline(h, color="#993300", lw=1.0, ls="--", label=r"$(n+\delta)/s$")
    for j, r in enumerate(roots):
        ax.plot([r], [h], "o", ms=4, color="#993300")
        off = [(-30, -12), (4, -14), (-34, 8)][j]
        ax.annotate(f"$k^*_{j+1}={r:.3f}$", (r, h), textcoords="offset points",
                    xytext=off, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k$ (log scale)")
    ax.set_ylabel("$w(k)/k$ (log scale)")
    ax.set_ylim(0.012, 1.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "ch2_wage_savings_multiplicity")


if __name__ == "__main__":
    exercise_24()
    exercise_13()
