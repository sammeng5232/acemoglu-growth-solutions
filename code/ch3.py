"""Numerical parts of the Chapter 3 solutions.

  * Exercise 3.3  the exact convergence equation for the Cobb-Douglas Solow model and the
                  way the convergence coefficient varies with the distance from the steady state;
  * Exercise 3.4  how long the income gap between two identical economies takes to fall from
                  100% to 10%, under the log-linear approximation and under the exact dynamics.

Run with `python code/ch3.py`.
"""
import numpy as np

from acemoglulib import plt, save

# Example 3.1 parameters
ALPHA, G, N, DELTA = 1 / 3, 0.02, 0.01, 0.05
RHO = DELTA + G + N                  # delta + g + n
BETA_LIN = (1 - ALPHA) * RHO         # log-linear convergence coefficient, (3.11)
C = (1 - ALPHA) / ALPHA


def beta_exact(z):
    """Convergence coefficient at a log gap z = log y - log y*, from the exact equation."""
    z = np.asarray(z, dtype=float)
    out = np.where(np.abs(z) < 1e-12, BETA_LIN,
                   ALPHA * RHO * (1 - np.exp(-C * np.where(z == 0, 1e-12, z)))
                   / np.where(z == 0, 1e-12, z))
    return out


def exercise_33():
    print("Exercise 3.3")
    print(f"  parameters: alpha={ALPHA:.4f}, delta+g+n={RHO:.2f}, "
          f"log-linear coefficient (1-alpha)(delta+g+n) = {BETA_LIN:.4f}")
    print("   y/y*      z=log(y/y*)   exact coefficient   ratio to log-linear")
    for ratio in [0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0]:
        z = np.log(ratio)
        b = float(beta_exact(z))
        print(f"  {ratio:6.2f} {z:13.4f} {b:18.4f} {b / BETA_LIN:19.2f}")
    print("  the coefficient is strictly decreasing in z: poor economies converge faster,")
    print("  rich ones slower, than the constant rate of the log-linearised equation.")

    zz = np.linspace(-2.3, 2.3, 400)
    fig, ax = plt.subplots()
    ax.plot(zz, beta_exact(zz), color="#003399", label="exact")
    ax.axhline(BETA_LIN, color="#993300", ls="--", lw=1.0,
               label=r"log-linear, $(1-\alpha)(\delta+g+n)$")
    ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel(r"$\log y(t)-\log y^*(t)$")
    ax.set_ylabel("convergence coefficient")
    ax.set_ylim(0, 0.32)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "ch3_convergence_coefficient")


def gap_time_loglinear(d0=np.log(2.0), d1=np.log(1.1)):
    return np.log(d0 / d1) / BETA_LIN


def exercise_34():
    print("\nExercise 3.4")
    T = gap_time_loglinear()
    print(f"  log-linear: the log gap decays at {BETA_LIN:.4f} per year, so")
    print(f"    T = ln(ln2 / ln1.1)/{BETA_LIN:.4f} = {T:.1f} years")
    print(f"    (half-life of the gap: ln2/{BETA_LIN:.4f} = {np.log(2)/BETA_LIN:.1f} years)")

    # exact dynamics: with x = k^(1-alpha), xdot = (1-alpha)[s - (delta+g+n)x], so
    # x(t) = x* + (x(0)-x*) exp(-BETA_LIN t) and y = A k^alpha = A x^(alpha/(1-alpha)).
    def gap_exact(y1_0_over_ystar, T):
        """log gap after T years when country 1 starts at y1/y* = r1 and country 2 at r1/2."""
        r1, r2 = y1_0_over_ystar, y1_0_over_ystar / 2
        x1 = r1 ** ((1 - ALPHA) / ALPHA)          # x(0)/x* for each country
        x2 = r2 ** ((1 - ALPHA) / ALPHA)
        d = np.exp(-BETA_LIN * T)
        x1t, x2t = 1 + (x1 - 1) * d, 1 + (x2 - 1) * d
        return (ALPHA / (1 - ALPHA)) * np.log(x1t / x2t)

    target = np.log(1.1)
    print("  exact dynamics (the gap is no longer exactly exponential):")
    for r1 in [1.0, 2.0, 0.8]:
        lo, hi = 0.0, 400.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if gap_exact(r1, mid) > target:
                lo = mid
            else:
                hi = mid
        print(f"    country 1 starts at y1/y* = {r1:4.2f} (country 2 at {r1/2:4.2f}): "
              f"T = {0.5*(lo+hi):5.1f} years")


if __name__ == "__main__":
    exercise_33()
    exercise_34()
