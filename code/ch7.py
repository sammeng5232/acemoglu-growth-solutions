"""Numerical part of the Chapter 7 solutions: the q-theory of investment (Section 7.8).

The firm solves

    max int_0^infty exp(-rt)[f(K) - I - phi(I)] dt   s.t.   Kdot = I - delta K,

with f(K) = A K^alpha and phi(I) = gamma I^2 / 2, so that the necessary conditions (7.85) give

    Kdot = I - delta K,
    Idot = [(r+delta)(1 + phi'(I)) - f'(K)] / phi''(I).

This script computes the steady state, verifies the saddle-path structure used in Exercises
7.27 and 7.29 (one positive and one negative eigenvalue), traces the stable arm, and checks
that the optimal path never approaches zero capital -- which is what licenses restricting the
state to a compact interval [eps, kbar] in Exercise 7.27.

Run with `python code/ch7.py`.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from acemoglulib import plt, save

A, ALPHA, DELTA, R, GAMMA = 1.0, 1 / 3, 0.05, 0.05, 2.0

f = lambda K: A * K ** ALPHA
fp = lambda K: A * ALPHA * K ** (ALPHA - 1)
fpp = lambda K: A * ALPHA * (ALPHA - 1) * K ** (ALPHA - 2)
phip = lambda I: GAMMA * I
phipp = lambda I: GAMMA


def steady_state():
    """K* solves f'(K) = (r+delta)(1 + phi'(delta K)); then I* = delta K*."""
    g = lambda K: fp(K) - (R + DELTA) * (1 + phip(DELTA * K))
    Kstar = brentq(g, 1e-8, 1e6)
    return Kstar, DELTA * Kstar


def jacobian(Kstar, Istar):
    """Jacobian of (Kdot, Idot) at the steady state."""
    return np.array([[-DELTA, 1.0],
                     [-fpp(Kstar) / phipp(Istar),
                      (R + DELTA) * phipp(Istar) / phipp(Istar)]])


def rhs(t, z):
    K, I = z
    return [I - DELTA * K,
            ((R + DELTA) * (1 + phip(I)) - fp(K)) / phipp(I)]


def stable_arm(Kstar, Istar, span=160.0, n=1200):
    """Trace the stable arm by integrating backwards from just off the steady state."""
    J = jacobian(Kstar, Istar)
    vals, vecs = np.linalg.eig(J)
    k = int(np.argmin(vals.real))            # the negative eigenvalue
    v = vecs[:, k].real
    v = v / np.linalg.norm(v)
    out = []
    for sign in (+1, -1):
        z0 = np.array([Kstar, Istar]) + sign * 1e-6 * v
        sol = solve_ivp(rhs, [0, -span], z0, t_eval=np.linspace(0, -span, n),
                        rtol=1e-10, atol=1e-12)
        out.append(sol.y)
    return vals, out


def main():
    Kstar, Istar = steady_state()
    print(f"parameters: A={A}, alpha={ALPHA:.4f}, delta={DELTA}, r={R}, phi(I)=gamma I^2/2 with gamma={GAMMA}")
    print(f"steady state: K* = {Kstar:.6f}, I* = delta K* = {Istar:.6f}, "
          f"q* = 1 + phi'(I*) = {1 + phip(Istar):.6f}")
    print(f"  check: f'(K*) = {fp(Kstar):.6f} = (r+delta)(1+phi'(I*)) = "
          f"{(R+DELTA)*(1+phip(Istar)):.6f}")
    print(f"  without adjustment costs (gamma=0) the steady state would solve f'(K)=r+delta: "
          f"K = {(A*ALPHA/(R+DELTA))**(1/(1-ALPHA)):.6f}")

    vals, arms = stable_arm(Kstar, Istar)
    print(f"\neigenvalues of the linearised system: {vals[0].real:+.6f}, {vals[1].real:+.6f}"
          f"   -> saddle path (one of each sign), as Exercise 7.28(c) asks one to show")

    lower = arms[1] if arms[1][0, -1] < Kstar else arms[0]
    Kmin = float(np.min(lower[0]))
    print(f"\nalong the stable arm approaching K* from below, capital stays bounded away from zero:")
    print(f"  over the traced segment, min K = {Kmin:.6f} > 0, and I > 0 throughout "
          f"(min I = {float(np.min(lower[1])):.6f})")
    print("  so for any K(0) > 0 the optimal path has inf_t K(t) = min{K(0), K*} > 0,")
    print("  which is Exercise 7.27(c): choose eps below that number.")
    Ktilde = (A * ALPHA) ** (1 / (1 - ALPHA))          # f'(K) > 1 exactly when K < Ktilde
    print(f"\nExercise 7.29: f'(K) > 1 exactly when K < (A*alpha)^(1/(1-alpha)) = {Ktilde:.6f}.")
    print("  If investment stopped at t', capital would decay as K(t)=K(t')exp(-delta(t-t')):")
    for K in (Kstar, 0.1 * Kstar, 0.01 * Kstar):
        T = np.log(K / Ktilde) / DELTA if K > Ktilde else 0.0
        print(f"  starting from K={K:10.6f} (f'={fp(K):8.4f}), capital falls below {Ktilde:.4f} "
              f"after {T:6.1f} periods; from then on f'(K(t))>1 and the deviation raises profits")

    fig, ax = plt.subplots()
    KK = np.linspace(0.05 * Kstar, 2.2 * Kstar, 400)
    ax.plot(KK, DELTA * KK, color="#666666", lw=1.0, label=r"$\dot K=0$:  $I=\delta K$")
    # Idot = 0 locus: (r+delta)(1+gamma I) = f'(K)  =>  I = [f'(K)/(r+delta) - 1]/gamma
    Ilocus = (fp(KK) / (R + DELTA) - 1) / GAMMA
    ok = Ilocus > 0
    ax.plot(KK[ok], Ilocus[ok], color="#993300", lw=1.0, ls="--",
            label=r"$\dot I=0$:  $f'(K)=(r+\delta)(1+\phi'(I))$")
    for arm in arms:
        ax.plot(arm[0], arm[1], color="#003399", lw=1.4)
    ax.plot([Kstar], [Istar], "o", ms=4, color="black")
    ax.annotate(r"$(K^*,I^*)$", (Kstar, Istar), textcoords="offset points", xytext=(6, -10),
                fontsize=8)
    ax.plot([], [], color="#003399", lw=1.4, label="stable arm (saddle path)")
    ax.set_xlabel("$K$")
    ax.set_ylabel("$I$")
    ax.set_xlim(0, 2.2 * Kstar)
    ax.set_ylim(0, 2.2 * Istar)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", borderaxespad=0.6)
    save(fig, "ch7_qtheory_saddle")


if __name__ == "__main__":
    main()
