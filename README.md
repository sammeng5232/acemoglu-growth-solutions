# Solutions to Daron Acemoglu, *Introduction to Modern Economic Growth* (Princeton University Press, 2009)

Authors: Zijun Meng and Claude Opus 5

## Contents

The book has 23 chapters with exercises; the instructor's solutions manual of Peters and
Simsek (2009) solves a selection of them. This document works through the exercises the
manual leaves out. It is being written chapter by chapter; the table shows where it stands.

| Chapter | exercises | in the manual | solved here |
|---|---:|---:|---|
| 2. The Solow Growth Model | 27 | 13 | **14** — 2.1–2.6, 2.8, 2.9, 2.10, 2.13, 2.15, 2.24, 2.25, 2.26 |
| 3. The Solow Model and the Data | 11 | 4 | **7** — 3.3–3.8, 3.11 |
| 4. Fundamental Determinants of Differences in Economic Performance | 3 | 1 | **2** — 4.1, 4.2 |
| 5. Foundations of Neoclassical Growth | 14 | 8 | **6** — 5.3–5.8 |
| 6. Infinite-Horizon Optimization and Dynamic Programming | 21 | 7 | **14** — 6.1, 6.4, 6.5, 6.6, 6.10, 6.11, 6.13–6.17, 6.19, 6.20, 6.21 |
| 7. An Introduction to the Theory of Optimal Control | 29 | 13 | **16** — 7.3, 7.4, 7.6–7.9, 7.11–7.16, 7.20, 7.22, 7.27, 7.29 |
| 8. The Neoclassical Growth Model | 39 | 15 | **24** — 8.1, 8.3–8.6, 8.8–8.10, 8.12, 8.14, 8.16–8.18, 8.20–8.22, 8.24, 8.26, 8.28, 8.29, 8.32, 8.35, 8.36, 8.39 |
| 9. Growth with Overlapping Generations | 33 | 12 | **21** — 9.2, 9.4, 9.5, 9.9–9.14, 9.18, 9.19, 9.22, 9.23, 9.25–9.31, 9.33 |
| 10. Human Capital and Economic Growth | 20 | 6 | **14** — 10.1, 10.3–10.5, 10.8–10.13, 10.15–10.17, 10.19 |
| 11. First-Generation Models of Endogenous Growth | 21 | 8 | **13** — 11.1–11.3, 11.5–11.7, 11.9–11.13, 11.19, 11.20 |
| 12–23 | 299 | 118 | in progress |

**The exercise statements are not reproduced.** Each solution is headed only by the
number of its exercise in the book, so read the statement there first. Notation and
the part letters (a), (b), … follow the book, and numbered results cited as
"(2.33)", "Assumption 2" or "Proposition 3.1" refer to Acemoglu (2009). Remarks
inside the solutions point out places where the book's own statements do not match
what the mathematics gives.

| File | Contents |
|---|---|
| `AcemogluSolutions.tex` / `.pdf` | the solutions in one self-contained document (solutions only, without the exercise statements) |
| `figures/` | figures produced by the code |
| `code/` | Python for the numerical parts (see below) |

## Building

Compile from this directory with

```bash
pdflatex AcemogluSolutions.tex
```

two or three times, so that the contents page numbers and cross-references settle.
The document needs no local style file; it includes the PDF figures from `figures/`.

## Code

`code/chN.py` (with the shared helpers in `code/acemoglulib.py`) needs only `numpy`,
`scipy` and `matplotlib`. Each script prints every number quoted in the corresponding
chapter and writes that chapter's figures to `figures/`. So far:

- **`ch2.py`** — the three interior steady states of the quartic economy of Exercise 2.4
  and the threshold \(2\sqrt3/9\) on \((n+\delta)/s\) below which they exist; and the
  technology of Exercise 2.13 (a CES with a low elasticity of substitution plus a small
  Cobb–Douglas term) that satisfies Assumptions 1 and 2 while producing three steady
  states when only labour income is saved.
- **`ch3.py`** — the exact convergence coefficient of the Cobb–Douglas Solow model as a
  function of the distance from the steady state (Exercise 3.3), and the time a twofold
  income gap takes to fall to 10%, under the log-linear approximation and under the exact
  dynamics (Exercise 3.4).
- **`ch6.py`** — the Euler path of Exercise 6.10 that satisfies every Euler equation, stays
  feasible forever, violates the transversality condition and is strictly suboptimal.
- **`ch7.py`** — the q-theory steady state, the saddle-path eigenvalues and the phase diagram
  behind Exercises 7.27 and 7.29.
- **`ch8.py`** — the neoclassical phase diagram, with the modified golden rule strictly to the
  left of the golden rule (Exercise 8.12), the saddle-path eigenvalues and the slope of the
  stable arm, in closed form and numerically (Exercises 8.22 and 8.24); the exactly log-linear
  transition of the discrete-time model with full depreciation (Exercise 8.26); and how little
  cross-country income dispersion differences in discount rates can generate (Exercise 8.32).
- **`ch9.py`** — the exact condition for dynamic inefficiency in the canonical OLG model and an
  unfunded social security scheme whose whole transition path makes every generation strictly
  better off (Exercise 9.9); the threshold on old-age labour income above which overaccumulation
  disappears (Exercise 9.10) and the much weaker condition for it under warm glow bequests
  (Exercise 9.22); the eigenvalues and comparative statics of the continuous-time perpetual youth
  model (Exercises 9.28–9.30); and capital income taxation in the Ramsey and OLG economies
  (Exercise 9.33).
- **`ch10.py`** — what a Mincerian wage regression recovers when schooling varies only because
  discount rates do (Exercise 10.4); the balance locus \(h=\xi(k)\), its slope and the steady
  state of Proposition 10.1 (Exercises 10.9 and 10.10); the elasticity of steady-state output to
  investment distortions with and without human capital (Exercise 10.11); the identity
  \(F_{KH}=-(K/H)F_{KK}\) (Exercise 10.15); and the dynamic externality of the closed economy with
  imperfect labour markets (Exercise 10.19).
- **`ch11.py`** — the transitional dynamics and vanishing labour share of the \(AK+BL\) economy
  (Exercise 11.3); the neoclassical model's steady state, convergence speed and growth rate as
  \(\alpha\to1\) (Exercise 11.6); the century-long income gap two capital tax rates generate
  (Exercise 11.7); the effect of \(\alpha\) in the two-sector model (Exercise 11.13); the
  Pigouvian subsidy that decentralizes the Romer optimum (Exercise 11.19); and the discrete-time
  balanced growth path and its admissible parameters (Exercise 11.20).

## Working files (not part of the document)

`ChN.tex` are the chapter sources, in which each solution is preceded by a short
restatement of the exercise; `tools/build_main.py` strips those restatements when it
assembles `AcemogluSolutions.tex`, and keeps them in the local
`AcemogluSolutions_with_exercises.tex`. Rebuild with `sh tools/mkchap.sh N`,
`sh tools/mkmain.sh` and `sh tools/mkmain.sh --full`; everything compiles outside the
project tree, so no `.aux`, `.log` or `.out` is ever left beside the sources.
