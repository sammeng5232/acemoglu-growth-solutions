"""Shared helpers for the numerical parts of the solutions.

Only numpy, scipy and matplotlib are needed.  Figures are written to ../figures
as PDF, at the size used by the solutions document.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.figsize": (5.4, 3.2),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    path = FIG / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote figures/{path.name}")


def bisect(g, lo, hi, tol=1e-14, maxit=200):
    """Root of g on [lo, hi], where g(lo) and g(hi) have opposite signs."""
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        raise ValueError(f"no sign change on [{lo}, {hi}]: {glo}, {ghi}")
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def sign_changes(x, y):
    """Indices i where y changes sign between x[i] and x[i+1]."""
    s = np.sign(y)
    return np.where(s[:-1] * s[1:] < 0)[0]
