"""Numerical parts of the Chapter 18 solutions (diffusion of technology).

  * Exercise 18.6   why g = 0 removes the steady-state technology gaps;
  * Exercises 18.14 and 18.15  the world growth equation (18.13), the necessity and
    sufficiency of (18.14), and the elasticity of g* with respect to each country;
  * Exercise 18.20  the Basu-Weil two-country model and the exponent
    (1 - alpha + gamma)/(alpha - gamma);
  * Exercise 18.25  IPR enforcement in the South and the direction of technology;
  * Exercises 18.28 and 18.29  a brute-force check of the closed form for the
    incomplete-contracts technology choice, and its comparative statics.

Run with `python code/ch18.py`.
"""
import numpy as np
from scipy.optimize import brentq, minimize

from acemoglulib import FIG, plt, save  # noqa: F401


# ---------------------------------------------------------------------------
# Section 18.2: the reduced-form diffusion model
# ---------------------------------------------------------------------------


def ex186(g=0.02, sig=(0.02, 0.05, 0.20), lam=(0.0, 0.005, 0.01)):
    print("Exercise 18.6: g = 0 wipes out the steady-state technology gaps")
    print("  a*_j = sigma_j/(sigma_j + g - lambda_j); the speed of convergence is")
    print("  mu_j = sigma_j + g - lambda_j, so sigma raises a* AND speeds convergence,")
    print("  while lambda raises a* and SLOWS it.")
    print(f"  {'sigma':>7} {'lambda':>8} {'a* (g=2%)':>11} {'a* (g=0)':>10} "
          f"{'half-life':>11}")
    for s in sig:
        for l in lam:
            a1 = s / (s + g - l)
            a0 = 1.0                      # with g = 0 the restriction forces lambda = 0
            hl = np.log(2) / (s + g - l)
            print(f"  {s:>7.3f} {l:>8.3f} {a1:>11.5f} {a0:>10.5f} {hl:>11.2f}")
    print("  with g > 0 a country must stay a definite proportional distance behind so")
    print("  that sigma_j (A - A_j) is large enough to keep up; with g = 0 there is")
    print("  nothing to keep up with, absorption closes the gap completely, and sigma")
    print("  governs only the speed of convergence, not its destination")


# ---------------------------------------------------------------------------
# Section 18.3.2: the endogenous world growth rate
# ---------------------------------------------------------------------------


def Psi(g, eta, L, zeta, beta, phi, rho, theta):
    return np.mean((eta * beta * L / (zeta * (rho + theta * g))) ** (1 / phi))


def ex1814(beta=0.5, phi=2.0, rho=0.05, theta=2.0, J=5, seed=3):
    print("\nExercises 18.14 and 18.15: the world growth rate")
    rng = np.random.default_rng(seed)
    eta = np.round(rng.uniform(0.4, 1.6, J), 4)
    L = np.round(rng.uniform(0.5, 2.5, J), 4)
    zeta = np.round(rng.uniform(3.0, 9.0, J), 4)
    print(f"  J = {J}, beta = {beta}, phi = {phi}, rho = {rho}, theta = {theta}")
    print(f"  eta  = {eta}")
    print(f"  L    = {L}")
    print(f"  zeta = {zeta}")
    P0 = Psi(0.0, eta, L, zeta, beta, phi, rho, theta)
    print(f"  Psi(0) = {P0:.6f}: (18.14) is {'satisfied' if P0 > 1 else 'violated'}")
    if P0 <= 1:
        print("  the unique equilibrium has g* = 0")
        return None
    gstar = brentq(lambda g: Psi(g, eta, L, zeta, beta, phi, rho, theta) - 1, 0, 1e4)
    mu = (eta * beta * L / (zeta * (rho + theta * gstar))) ** (1 / phi)
    print(f"  g* = {gstar:.6f}, and mean(mu*) = {mu.mean():.12f} (must be 1)")
    print(f"  mu* = {np.round(mu, 5)}")
    print("  Psi is strictly decreasing in g, so the root is unique; (18.14) is exactly")
    print("  Psi(0) > 1 and is therefore necessary and sufficient for g* > 0.")
    print("  The implicit function theorem gives d g*/d log eta_j = mu*_j (rho + theta g*)"
          "/(J theta):")
    print(f"  {'j':>3} {'mu*_j':>10} {'predicted':>12} {'numerical':>12}")
    for j in range(J):
        pred = mu[j] * (rho + theta * gstar) / (J * theta)
        e2 = eta.copy(); e2[j] *= 1.000001
        e1 = eta.copy(); e1[j] *= 0.999999
        g2 = brentq(lambda g: Psi(g, e2, L, zeta, beta, phi, rho, theta) - 1, 0, 1e4)
        g1 = brentq(lambda g: Psi(g, e1, L, zeta, beta, phi, rho, theta) - 1, 0, 1e4)
        num = (g2 - g1) / 2e-6
        print(f"  {j:>3} {mu[j]:>10.5f} {pred:>12.8f} {num:>12.8f}")
    print("  the effect of any one country on world growth is proportional to its own")
    print("  relative technology mu*_j: advanced and large countries matter more")
    return gstar


# ---------------------------------------------------------------------------
# Section 18.4.2: Basu-Weil
# ---------------------------------------------------------------------------


def ex1820(alpha=2 / 3, s1=0.24, s2=0.06, A=1.0, delta=0.05, T=4000, dt=0.02):
    print("\nExercise 18.20: capital-labour ratios and inappropriate technologies")
    print("  y_1*/y_2* = (s_1/s_2)^[(1 - alpha + gamma)/(alpha - gamma)], and the")
    print("  derivative of that exponent in gamma is 1/(alpha - gamma)^2 > 0.")
    print(f"  alpha = {alpha:.4f}, s_1/s_2 = {s1 / s2:.2f}")
    print(f"  {'gamma':>7} {'exponent':>10} {'y1*/y2*':>12} {'k1*/k2*':>12} "
          f"{'conv. rate':>11}")
    out = []
    for gam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65):
        if gam >= alpha:
            print(f"  {gam:>7.2f} {'--':>10} {'no steady state':>12}")
            continue
        e = (1 - alpha + gam) / (alpha - gam)
        ratio = (s1 / s2) ** e
        kratio = (s1 / s2) ** (1 / (alpha - gam))
        out.append((gam, ratio))
        print(f"  {gam:>7.2f} {e:>10.4f} {ratio:>12.4f} {kratio:>12.4f} "
              f"{(alpha - gam) * delta:>11.5f}")
    print("  the steady state exists only for gamma < alpha; the text's illustrative")
    print("  gamma = alpha = 2/3 is exactly the borderline at which the mechanism")
    print("  becomes explosive, so it can be used for a level comparison at a given")
    print("  capital gap but not for a steady-state comparison.")

    # transition: country 1 is always at the frontier, k' = max over history
    print("\n  simulated transition (both countries start at the same k):")
    for gam in (0.0, 0.3, 0.5):
        k1 = k2 = 0.05
        kp = k1
        for _ in range(int(T / dt)):
            kp = max(kp, k1, k2)
            y1 = A * k1 ** (1 - alpha + gam) * kp ** (-gam)
            y2 = A * k2 ** (1 - alpha + gam) * kp ** (-gam)
            k1 += dt * (s1 * y1 - delta * k1)
            k2 += dt * (s2 * y2 - delta * k2)
        y1 = A * k1 ** (1 - alpha + gam) * kp ** (-gam)
        y2 = A * k2 ** (1 - alpha + gam) * kp ** (-gam)
        pred = (s1 / s2) ** ((1 - alpha + gam) / (alpha - gam))
        print(f"   gamma={gam:.2f}: y1/y2 = {y1 / y2:>9.4f}  (closed form {pred:>9.4f})")
    return out


# ---------------------------------------------------------------------------
# Section 18.4.3: appropriate technology and IPR
# ---------------------------------------------------------------------------


def yeff(NH, NL, L, H, om):
    """Income per effective unit of labour, up to the common constant."""
    E = L + om * H
    return ((NL * L) ** 0.5 + (NH * om * H) ** 0.5) ** 2 / E


def ex1825(om=2.0, Hn=0.4, Ln=0.6, Hs=0.1, Ls=0.9, Jn=1.0, Js=1.0, Nbar=2.0):
    print("\nExercise 18.25: IPR enforcement in the South")
    print(f"  omega = {om}, North H/L = {Hn / Ln:.4f}, South H/L = {Hs / Ls:.4f}")
    print("  without Southern IPR the market is the North alone, so (N_H/N_L)* ="
          " omega H^n/L^n;")
    print("  with IPR it is the world, so (N_H/N_L)^IPR = omega H^w/L^w.")
    Hw, Lw = Jn * Hn + Js * Hs, Jn * Ln + Js * Ls
    rows = []
    for name, R in (("no IPR", om * Hn / Ln), ("IPR   ", om * Hw / Lw)):
        NH = Nbar * R / (1 + R)
        NL = Nbar / (1 + R)
        yn = yeff(NH, NL, Ln, Hn, om)
        ys = yeff(NH, NL, Ls, Hs, om)
        rows.append((name, R, NH / NL, yn, ys, yn / ys))
        print(f"  {name}: N_H/N_L = {NH / NL:>8.5f}, y^eff_n = {yn:>8.5f}, "
              f"y^eff_s = {ys:>8.5f}, ratio = {yn / ys:>7.5f}")
    print(f"  IPR moves N_H/N_L from {rows[0][2]:.5f} down to {rows[1][2]:.5f}, so the")
    print(f"  North's advantage in income per effective worker falls from "
          f"{rows[0][5]:.5f} to {rows[1][5]:.5f}")
    print("  the Northern y^eff falls and the Southern one rises: the direction of")
    print("  technology is a pure composition effect at given N_H + N_L")
    # the maximizer check of Proposition 18.8
    lam_n = om * Hn / (Ln + om * Hn)
    print(f"  check of Proposition 18.8: y^eff_j peaks at lambda = N_H/(N_L+N_H); at the")
    print(f"  no-IPR ratio this is {rows[0][2] / (1 + rows[0][2]):.6f}, and the Northern")
    print(f"  effective skill share is {lam_n:.6f} -- they coincide")
    return rows


# ---------------------------------------------------------------------------
# Section 18.5: contracting institutions
# ---------------------------------------------------------------------------


def Lambda_mine(alpha, beta, mu):
    a = alpha * beta / (alpha + beta)        # alpha (1 - gamma)
    m = 1 - mu
    rho0 = a * (1 - beta * m) / (beta * (1 - a * m))
    return ((a / beta) * rho0 ** (-(1 - beta * m))) ** (1 / (1 - beta))


def Lambda_printed(alpha, beta, mu):
    a = alpha * beta / (alpha + beta)
    m = 1 - mu
    return (((1 - a * m) / (1 - beta * m)) ** (-(1 - beta * m) / (1 - beta))
            * (a / beta) ** (beta * m / (1 - beta)))


def solveN(Lam, A, kap, beta, psi, Gp, w0, nu, c):
    """Root of Gamma'(N) + w0 = A kappa beta^(1/(1-beta)) psi^(-beta/(1-beta)) Lam N^th."""
    th = (beta * (kap + 1) - 1) / (1 - beta)
    K = A * kap * beta ** (1 / (1 - beta)) * psi ** (-beta / (1 - beta)) * Lam
    return brentq(lambda N: c * N ** nu + w0 - K * N ** th, 1e-10, 1e12)


def profit(N, xc, alpha, beta, mu, kap, A, psi, w0, nu, c):
    a = alpha * beta / (alpha + beta)
    xn = (a / psi * xc ** (beta * mu) * A ** (1 - beta)
          * N ** (beta * (kap + 1) - 1)) ** (1 / (1 - beta * (1 - mu)))
    rev = A ** (1 - beta) * (xc ** mu * xn ** (1 - mu)) ** beta * N ** (beta * (kap + 1))
    return (rev - psi * N * mu * xc - psi * N * (1 - mu) * xn
            - c / (1 + nu) * N ** (1 + nu) - w0 * N), xn


def ex1828(alpha=0.6, beta=0.5, kap=0.8, A=3.0, w0=0.05, nu=3.0, c=1.0):
    psi = 1 - beta
    print("\nExercises 18.28 and 18.29: technology choice under incomplete contracts")
    print(f"  alpha = {alpha}, beta = {beta}, kappa = {kap}, A = {A}, "
          f"Gamma(N) = N^{1 + nu:g}/{1 + nu:g}, w0 = {w0}")
    th = (beta * (kap + 1) - 1) / (1 - beta)
    print(f"  theta_0 = [beta(kappa+1)-1]/(1-beta) = {th:.4f} < 0, so restriction 2,")
    print("  N Gamma''/(Gamma' + w0) > theta_0, holds for any convex Gamma and any w0")
    print(f"  {'mu':>6} {'N (brute force)':>17} {'N (derived)':>13} {'N (printed)':>13} "
          f"{'xn/xc':>9} {'closed form':>12} {'kap psi xc':>11} {'G(N)+w0':>10}")
    for mu in (0.05, 0.25, 0.5, 0.75, 0.999999):
        Nd = solveN(Lambda_mine(alpha, beta, mu), A, kap, beta, psi, None, w0, nu, c)
        Np = solveN(Lambda_printed(alpha, beta, mu), A, kap, beta, psi, None, w0, nu, c)
        res = minimize(lambda v: -profit(np.exp(v[0]), np.exp(v[1]), alpha, beta, mu,
                                         kap, A, psi, w0, nu, c)[0],
                       x0=[np.log(Nd), np.log(1.0)], method="Nelder-Mead",
                       options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000,
                                "maxfev": 40000})
        Nb, xcb = np.exp(res.x)
        _, xnb = profit(Nb, xcb, alpha, beta, mu, kap, A, psi, w0, nu, c)
        aa, mm = alpha * beta / (alpha + beta), 1 - mu
        cf = aa * (1 - beta * mm) / (beta * (1 - aa * mm))
        print(f"  {mu:>6.2f} {Nb:>17.6f} {Nd:>13.6f} {Np:>13.6f} {xnb / xcb:>9.5f} "
              f"{cf:>12.5f} {kap * psi * xcb:>11.6f} {c * Nb ** nu + w0:>10.6f}")
    a = alpha * beta / (alpha + beta)
    print("  the brute-force optimum matches the derived closed form at every mu and")
    print("  differs from the one implied by the exponent printed in (18.48).")
    print(f"  xn/xc = a(1-beta m)/[beta(1-a m)] with a = alpha beta/(alpha+beta) = "
          f"{a:.5f} < beta,")
    print("  and kappa psi xc = Gamma'(N) + w0 exactly, as under complete contracts.")

    Nstar = solveN(1.0, A, kap, beta, psi, None, w0, nu, c)
    print(f"\n  complete contracts: N* = {Nstar:.6f}, x* = "
          f"{(c * Nstar ** nu + w0) / (kap * psi):.6f}")
    print(f"  {'alpha':>7} {'mu':>6} {'Lambda':>10} {'N~/N*':>9} {'x~c/x*':>9} "
          f"{'x~n/x~c':>9}")
    curves = {}
    for al in (0.3, 0.6, 0.9):
        pts = []
        for mu in np.linspace(0.0, 1.0, 101):
            Lam = Lambda_mine(al, beta, mu)
            Nt = solveN(Lam, A, kap, beta, psi, None, w0, nu, c)
            pts.append((mu, Nt / Nstar))
            aa = al * beta / (al + beta)
            m = 1 - mu
            if mu in (0.0, 0.5, 1.0):
                xc = (c * Nt ** nu + w0) / (kap * psi)
                xs = (c * Nstar ** nu + w0) / (kap * psi)
                print(f"  {al:>7.2f} {mu:>6.2f} {Lam:>10.6f} {Nt / Nstar:>9.6f} "
                      f"{xc / xs:>9.6f} "
                      f"{aa * (1 - beta * m) / (beta * (1 - aa * m)):>9.6f}")
        curves[al] = np.array(pts)
    print("  Lambda < 1 for every mu < 1 and rises to 1 as mu -> 1, so N~ < N* always")
    print("  (Proposition 18.12), and N~ is increasing in mu and in alpha")
    return curves, Nstar


# ---------------------------------------------------------------------------


def figure(bw, curves):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    g, r = np.array(bw).T
    ax1.plot(g, r, "-", color="black")
    ax1.plot(g, r, "o", color="black", ms=2.5)
    ax1.set_xlabel(r"$\gamma$")
    ax1.set_ylabel(r"$y_1^{*}/y_2^{*}$")
    ax1.set_yscale("log")
    ax1.set_title(r"Basu--Weil, $s_1/s_2=4$, $\alpha=2/3$", fontsize=8)

    for al, style in zip(sorted(curves), ("-", "--", ":")):
        p = curves[al]
        ax2.plot(p[:, 0], p[:, 1], style, color="black", label=rf"$\alpha={al:g}$")
    ax2.set_xlabel(r"$\mu$")
    ax2.set_ylabel(r"$\tilde N/N^{*}$")
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False, fontsize=7, loc="lower right")
    ax2.set_title("contracting institutions and technology", fontsize=8)
    fig.tight_layout()
    save(fig, "ch18_diffusion")


def main():
    ex186()
    ex1814()
    bw = ex1820()
    ex1825()
    curves, _ = ex1828()
    figure(bw, curves)


if __name__ == "__main__":
    main()
