"""Numerical parts of the Chapter 8 solutions (the neoclassical growth model).

  * Exercises 8.12 and 8.22  the phase diagram: the cdot = 0 locus always lies to the left of
    the golden rule, the steady state is a saddle, and the stable arm is the unique equilibrium;
  * Exercise 8.24  the log-preference case of Example 8.2: steady state, eigenvalues and the
    slope of the stable arm in closed form, checked against a numerical solution;
  * Exercise 8.26  the discrete-time model with log utility, Cobb-Douglas technology and full
    depreciation, where the transition is exactly log-linear;
  * Exercise 8.32  how much cross-country income dispersion differences in discount rates can
    generate.

Run with `python code/ch8.py`.
"""
import numpy as np
from scipy.integrate import solve_ivp

from acemoglulib import plt, save

ALPHA, DELTA, RHO, N, G, THETA = 1 / 3, 0.05, 0.02, 0.01, 0.02, 1.0

f = lambda k: k ** ALPHA
fp = lambda k: ALPHA * k ** (ALPHA - 1)
fpp = lambda k: ALPHA * (ALPHA - 1) * k ** (ALPHA - 2)


def steady_state(theta=THETA, g=G, tau=0.0):
    """f'(k*) = (rho + delta + theta g)/(1-tau); ctilde* = f(k*) - (n+g+delta)k*."""
    kstar = (ALPHA * (1 - tau) / (RHO + DELTA + theta * g)) ** (1 / (1 - ALPHA))
    return kstar, f(kstar) - (N + g + DELTA) * kstar


def golden_rule(g=G):
    """f'(k_gold) = n + g + delta maximises steady-state normalised consumption."""
    return (ALPHA / (N + g + DELTA)) ** (1 / (1 - ALPHA))


def rhs(t, y, theta=THETA, g=G):
    k, c = y
    return [f(k) - c - (N + g + DELTA) * k,
            c * (fp(k) - DELTA - RHO - theta * g) / theta]


def eigen(kstar, cstar, theta=THETA, g=G):
    J = np.array([[fp(kstar) - (N + g + DELTA), -1.0],
                  [cstar * fpp(kstar) / theta, 0.0]])
    return np.linalg.eigvals(J), J


def stable_arm(kstar, cstar, span=220.0, n=1500, theta=THETA, g=G):
    vals, J = eigen(kstar, cstar, theta, g)
    k = int(np.argmin(vals.real))
    v = np.linalg.eig(J)[1][:, k].real
    v = v / np.linalg.norm(v)
    arms = []
    for s in (+1, -1):
        y0 = np.array([kstar, cstar]) + s * 1e-7 * v
        sol = solve_ivp(lambda t, y: rhs(t, y, theta, g), [0, -span], y0,
                        t_eval=np.linspace(0, -span, n), rtol=1e-11, atol=1e-13)
        arms.append(sol.y)
    return vals, arms


def main():
    kstar, cstar = steady_state()
    kgold = golden_rule()
    print(f"parameters: alpha={ALPHA:.4f}, delta={DELTA}, rho={RHO}, n={N}, g={G}, theta={THETA}")
    print(f"\nExercise 8.12: f'(k*) = rho+delta+theta*g = {RHO+DELTA+THETA*G:.4f} while "
          f"f'(k_gold) = n+g+delta = {N+G+DELTA:.4f}")
    print(f"  k*      = {kstar:.6f}   (modified golden rule)")
    print(f"  k_gold  = {kgold:.6f}   (golden rule)")
    print(f"  k* < k_gold since rho - n - (1-theta)g = {RHO-N-(1-THETA)*G:+.4f} > 0 "
          f"(Assumption 4)")

    vals, arms = stable_arm(kstar, cstar)
    beta_trace = fp(kstar) - (N + G + DELTA)
    xi1 = float(min(vals.real))
    print(f"\nExercises 8.22 and 8.24: linearised system at (k*, ctilde*) = "
          f"({kstar:.6f}, {cstar:.6f})")
    print(f"  trace = rho - n - (1-theta)g = {beta_trace:+.6f},  "
          f"det = ctilde* f''(k*)/theta = {cstar*fpp(kstar)/THETA:+.6f} < 0")
    print(f"  eigenvalues {vals[0].real:+.6f} and {vals[1].real:+.6f}: a saddle")
    print(f"  stable arm slope  dctilde/dk = trace - xi1 = {beta_trace - xi1:.6f}")
    print(f"  half-life of the transition = ln2/|xi1| = {np.log(2)/abs(xi1):.2f} periods")

    # check the closed-form slope against the numerically traced arm
    lower = arms[0] if arms[0][0, -1] < kstar else arms[1]
    kk, cc = lower[0], lower[1]
    near = np.argmin(np.abs(kk - 0.98 * kstar))
    slope_num = (cstar - cc[near]) / (kstar - kk[near])
    print(f"  numerical slope near the steady state = {slope_num:.6f}  (closed form "
          f"{beta_trace - xi1:.6f})")

    print("\nExercise 8.26 (discrete time, log utility, Cobb-Douglas, full depreciation):")
    beta_d = 0.96
    kd = (ALPHA * beta_d) ** (1 / (1 - ALPHA))
    print(f"  with beta={beta_d}: k* = (alpha*beta)^(1/(1-alpha)) = {kd:.6f}, "
          f"saving rate = alpha*beta = {ALPHA*beta_d:.4f}")
    k0, path = 0.2 * kd, []
    k = k0
    for t in range(6):
        path.append(k)
        k = ALPHA * beta_d * k ** ALPHA
    exact = [np.exp(np.log(kd) + ALPHA ** t * (np.log(k0) - np.log(kd))) for t in range(6)]
    print("  k(t) iterated : " + " ".join(f"{v:.6f}" for v in path))
    print("  closed form   : " + " ".join(f"{v:.6f}" for v in exact)
          + "   (log k(t) - log k* = alpha^t [log k(0) - log k*])")

    print("\nExercise 8.32: elasticity of steady-state income to the effective discount rate")
    el = -ALPHA / (1 - ALPHA)
    print(f"  y* = [alpha/(rho+delta+theta g)]^(alpha/(1-alpha)), so d log y*/d log(rho+delta+theta g)"
          f" = {el:.4f}")
    for dd, gg in ((DELTA, G), (0.0, 0.0)):
        base, high = RHO + dd + THETA * gg, 0.022 + dd + THETA * gg
        ratio = (base / high) ** (-el * -1)
        ratio = (ALPHA / high) ** (ALPHA / (1 - ALPHA)) / (ALPHA / base) ** (ALPHA / (1 - ALPHA))
        print(f"  delta={dd}, g={gg}: rho 0.02 -> 0.022 changes y* by {100*(ratio-1):+.2f}% "
              f"(income ratio {1/ratio:.4f})")

    # figure
    fig, ax = plt.subplots()
    kk = np.linspace(0.02, 1.6 * kgold, 400)
    ax.plot(kk, f(kk) - (N + G + DELTA) * kk, color="#993300", ls="--", lw=1.0,
            label=r"$\dot k=0$:  $\tilde c=f(k)-(n+g+\delta)k$")
    ax.axvline(kstar, color="#666666", lw=1.0, label=r"$\dot{\tilde c}=0$:  $f'(k)=\rho+\delta+\theta g$")
    for arm in arms:
        ax.plot(arm[0], arm[1], color="#003399", lw=1.4)
    ax.plot([], [], color="#003399", lw=1.4, label="stable arm")
    ax.plot([kstar], [cstar], "o", ms=4, color="black")
    ax.annotate(r"$(k^*,\tilde c^*)$", (kstar, cstar), textcoords="offset points",
                xytext=(6, -12), fontsize=8)
    ax.plot([kgold], [f(kgold) - (N + G + DELTA) * kgold], "s", ms=4, color="#993300")
    ax.annotate(r"$k_{\rm gold}$", (kgold, f(kgold) - (N + G + DELTA) * kgold),
                textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\tilde c$")
    ax.set_xlim(0, 1.6 * kgold)
    ax.set_ylim(0, 1.35 * (f(kgold) - (N + G + DELTA) * kgold))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "ch8_saddle_path")


if __name__ == "__main__":
    main()
