"""Numerical parts of the Chapter 12 solutions (modeling technological change).

  * Exercises 12.1, 12.3 and 12.4  the private and social values of a process innovation as
    functions of its size, the drastic threshold, and the inequalities of Proposition 12.2;
  * Exercise 12.6   Arrow's replacement effect: the incumbent monopolist's value of the same
    innovation, below the competitive firm's value in both pricing regimes;
  * Exercise 12.7   the cost handicap chibar at which entrant and incumbent incentives cross,
    and how it varies with the elasticity of demand;
  * Exercise 12.8   excessive innovation is impossible for a process innovation under Bertrand
    competition, but arises once the innovation replaces the incumbent's product;
  * Exercise 12.12  when a larger number of varieties raises Dixit-Stiglitz profits.

Run with `python code/ch12.py`.
"""
import numpy as np
from scipy.integrate import quad

from acemoglulib import bisect, plt, save

PSI = 1.0                      # pre-innovation marginal cost


class Isoelastic:
    """D(p) = p^{-eps}; monopoly price is a constant markup eps/(eps-1) over marginal cost."""
    name = "isoelastic"

    def __init__(self, eps):
        self.eps = eps

    def D(self, p):
        return p ** (-self.eps)

    def pM(self, c):
        return self.eps / (self.eps - 1) * c

    def PiM(self, c):
        p = self.pM(c)
        return self.D(p) * (p - c)

    def CS(self, p):
        return p ** (1 - self.eps) / (self.eps - 1)


class Linear:
    """D(p) = a - p on [0, a]."""
    name = "linear"

    def __init__(self, a):
        self.a = a

    def D(self, p):
        return np.maximum(self.a - p, 0.0)

    def pM(self, c):
        return 0.5 * (self.a + c)

    def PiM(self, c):
        return 0.25 * (self.a - c) ** 2

    def CS(self, p):
        return 0.5 * max(self.a - p, 0.0) ** 2


def lam_star(d):
    """Smallest lambda for which the innovation is drastic: pM(psi/lambda) <= psi."""
    return bisect(lambda lam: d.pM(PSI / lam) - PSI, 1.0 + 1e-12, 1e6)


def values(d, lam, mu=0.0):
    """Private profit, equilibrium surplus and full social value of an innovation of size lam,
    all measured against the pre-innovation competitive outcome (price psi), as in (12.1)
    and (12.5).  Returns (price, private, equilibrium surplus, full social value)."""
    c = PSI / lam
    p = min(d.pM(c), PSI)                               # limit pricing when not drastic
    private = d.D(p) * (p - c) - mu
    surplus = d.D(p) * (p - c) + quad(d.D, p, PSI)[0] - mu
    full = quad(d.D, c, PSI)[0] - mu
    return p, private, surplus, full


def ex121_123_124():
    print("Exercises 12.1, 12.3 and 12.4: the value of a process innovation")
    for d in (Isoelastic(3.0), Linear(1.5)):
        ls = lam_star(d)
        print(f"\n  {d.name} demand, psi={PSI}: drastic iff lambda >= lambda* = {ls:.6f}")
        print(f"  {'lambda':>8} {'price':>9} {'regime':>7} {'private':>10} {'surplus':>10} "
              f"{'full social':>12}")
        for lam in sorted((1.1, 1.3, ls, 1.8, 2.5, 4.0)):
            p, pr, su, fu = values(d, lam)
            reg = "limit" if lam < ls - 1e-9 else "monop"
            print(f"  {lam:>8.4f} {p:>9.6f} {reg:>7} {pr:>10.6f} {su:>10.6f} {fu:>12.6f}")
        print("  private < surplus < full social at every lambda, and all three are strictly")
        print("  increasing in lambda: this is Proposition 12.2 (Exercise 12.3)")
        p, pr, su, fu = values(d, 1.8)
        print(f"  Exercise 12.4 at lambda=1.8: surplus - private = {su-pr:.6f}, the consumer")
        print(f"    surplus gain, which is >= 0, so any mu that makes the innovation privately")
        print(f"    profitable leaves the equilibrium surplus positive")


def ex126():
    print("\nExercise 12.6: Arrow's replacement effect")
    for d in (Isoelastic(3.0), Linear(1.5)):
        ls = lam_star(d)
        print(f"\n  {d.name} demand, pre-innovation monopoly profit PiM(psi) = "
              f"{d.PiM(PSI):.6f}")
        print(f"  {'lambda':>8} {'competitive':>12} {'incumbent':>11} {'difference':>11} "
              f"{'regime':>7}")
        for lam in sorted((1.1, 1.3, ls, 1.8, 2.5, 4.0)):
            _, comp, _, _ = values(d, lam)
            inc = d.PiM(PSI / lam) - d.PiM(PSI)
            reg = "limit" if lam < ls - 1e-9 else "monop"
            print(f"  {lam:>8.4f} {comp:>12.6f} {inc:>11.6f} {comp-inc:>11.6f} {reg:>7}")
        print("  the incumbent's value is below the competitive firm's at every lambda;")
        print("  in the drastic region the gap is exactly PiM(psi), the profit it replaces")


def ex127(lam=1.8, mu=0.05):
    print(f"\nExercise 12.7: entrant pays chi*mu, incumbent pays mu (lambda={lam}, mu={mu})")
    print(f"  {'eps':>6} {'lambda*':>9} {'regime':>7} {'Pi_E':>9} {'incumbent gain':>15} "
          f"{'chibar':>9} {'ratio':>8}")
    for eps in (1.5, 2.0, 3.0, 5.0, 10.0):
        d = Isoelastic(eps)
        ls = lam_star(d)
        _, PiE, _, _ = values(d, lam)                 # entrant's gross profit (mu = 0)
        gain = d.PiM(PSI / lam) - d.PiM(PSI)          # incumbent's gross gain
        reg = "limit" if lam < ls - 1e-9 else "monop"
        print(f"  {eps:>6.2f} {ls:>9.4f} {reg:>7} {PiE:>9.6f} {gain:>15.6f} "
              f"{1+(PiE-gain)/mu:>9.4f} {PiE/gain:>8.4f}")
    print("  chibar > 1 always, so a slightly handicapped entrant still has the stronger")
    print("  incentive; but chibar and the ratio Pi_E/gain both fall as demand becomes more")
    print("  elastic, because the rent the incumbent replaces shrinks with the markup")


def ex128():
    print("\nExercise 12.8(a): can the entrant's innovation reduce total surplus?")
    print("  (i) process innovation, Bertrand against the incumbent's psi technology:")
    print(f"  {'demand':>12} {'lambda':>8} {'cons. gain':>12} {'incumbent rent':>15} "
          f"{'net':>10}")
    for d in (Isoelastic(3.0), Linear(1.5)):
        for lam in (1.05, 1.5, 3.0):
            c = PSI / lam
            p = min(d.pM(c), PSI)
            dcs = quad(d.D, p, d.pM(PSI))[0]          # price falls from the monopoly price
            rent = d.PiM(PSI)
            print(f"  {d.name:>12} {lam:>8.3f} {dcs:>12.6f} {rent:>15.6f} {dcs-rent:>10.6f}")
    print("  the consumer gain always exceeds the rent destroyed, because the price falls all")
    print("  the way from the incumbent's monopoly price to at most psi: no excessive innovation")

    print("\n  (ii) a product innovation that replaces the incumbent, demand scaled by lambda:")
    eps, lam, mu = 3.0, 1.2, 0.12
    d = Isoelastic(eps)
    Pi, CS = d.PiM(PSI), d.CS(d.pM(PSI))
    social, private = (lam - 1) * (Pi + CS), lam * Pi
    print(f"    eps={eps}, lambda={lam}: PiM(psi)={Pi:.6f}, CS={CS:.6f}")
    print(f"    social gain  = (lambda-1)(Pi + CS) = {social:.6f}")
    print(f"    private gain = lambda * Pi         = {private:.6f}")
    print(f"    so for any mu in ({social:.6f}, {private:.6f}) the entrant innovates and total")
    print(f"    surplus falls; at mu = {mu} the private value is {private-mu:+.6f} and the")
    print(f"    social value {social-mu:+.6f}")
    print(f"    the condition is (lambda-1) CS < Pi, that is lambda < 1 + Pi/CS = "
          f"{1+Pi/CS:.6f} = 2 - 1/eps")


def dixit_profit(N, eps, sigma, gamma, m, psi=PSI):
    """CES upper tier u(C,y) with elasticity sigma; returns (P, C, share, profit per firm)."""
    P = N ** (-1 / (eps - 1)) * eps / (eps - 1) * psi
    num = gamma ** sigma * P ** (1 - sigma)
    share = num / (num + (1 - gamma) ** sigma)        # PC/m
    return P, share * m / P, share, share * m / (eps * N)


def ex1212(eps=3.0, m=1.0, N=10.0):
    print(f"\nExercise 12.12: profits and the number of varieties "
          f"(eps={eps}, m={m}, N={N:.0f})")
    print(f"  d log pi / d log N = -1 - eta_E/(eps-1), so profits rise with N iff")
    print(f"  g_P(P,m) > (eps-1) C, that is the composite's demand elasticity exceeds eps;")
    print(f"  for a CES upper tier this is sigma > 1 + (eps-1)/(1-s_C)")
    print(f"  {'sigma':>7} {'gamma':>7} {'s_C':>10} {'threshold':>11} {'profit':>11} "
          f"{'d log pi/d log N':>18}")
    for sigma in (1.0, 2.0, 4.0, 8.0):
        for gamma in (0.05, 0.50):
            P, C, s, pi = dixit_profit(N, eps, sigma, gamma, m)
            _, _, _, pi2 = dixit_profit(N * 1.001, eps, sigma, gamma, m)
            print(f"  {sigma:>7.2f} {gamma:>7.2f} {s:>10.2e} {1+(eps-1)/(1-s):>11.4f} "
                  f"{pi:>11.3e} {np.log(pi2/pi)/np.log(1.001):>18.4f}")
    print("  with sigma = 1 (Cobb-Douglas) spending on the differentiated bundle is fixed and")
    print("  the elasticity is exactly -1, whatever gamma is.  Once sigma clears the threshold")
    print("  the aggregate demand externality dominates and profits RISE with N; the threshold")
    print("  itself rises with the expenditure share s_C, so a differentiated sector that")
    print("  already absorbs most of spending cannot gain from further entry")


def figure():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))

    ax = axes[0]
    d = Isoelastic(3.0)
    ls = lam_star(d)
    lams = np.linspace(1.02, 4.0, 300)
    priv, surp, full, inc = [], [], [], []
    for lam in lams:
        _, pr, su, fu = values(d, lam)
        priv.append(pr)
        surp.append(su)
        full.append(fu)
        inc.append(d.PiM(PSI / lam) - d.PiM(PSI))
    ax.plot(lams, full, color="#993300", lw=1.2, label=r"full social value $S^I$")
    ax.plot(lams, surp, color="#006633", ls="--", lw=1.0, label="equilibrium surplus")
    ax.plot(lams, priv, color="#003399", lw=1.2, label="competitive firm")
    ax.plot(lams, inc, color="#666666", ls=":", lw=1.2, label="incumbent monopolist")
    ax.axvline(ls, color="#cccccc", lw=0.7)
    ax.annotate(rf"$\lambda^*={ls:.2f}$", (ls, 0.0), textcoords="offset points",
                xytext=(3, 4), fontsize=7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("value of the innovation")
    ax.set_xlim(1.0, 4.0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    ax = axes[1]
    Ns = np.logspace(0.3, 3, 200)
    for sigma, col, ls2 in ((1.0, "#666666", ":"), (2.0, "#993300", "-."),
                            (4.0, "#006633", "--"), (8.0, "#003399", "-")):
        pis = np.array([dixit_profit(N, 3.0, sigma, 0.05, 1.0)[3] for N in Ns])
        ax.plot(Ns, pis / pis[0], color=col, ls=ls2, lw=1.0, label=rf"$\sigma={sigma:.0f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\pi(N)/\pi(2)$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "ch12_innovation_value")


def main():
    ex121_123_124()
    ex126()
    ex127()
    ex128()
    ex1212()
    figure()


if __name__ == "__main__":
    main()
