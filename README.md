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
| 7–23 | ~334 | ~126 | in progress |

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
- **`ch6.py`** — the Euler path of Exercise 6.10 that satisfies every Euler equation, stays
  feasible forever, violates the transversality condition and is strictly suboptimal.
- **`ch3.py`** — the exact convergence coefficient of the Cobb–Douglas Solow model as a
  function of the distance from the steady state (Exercise 3.3), and the time a twofold
  income gap takes to fall to 10%, under the log-linear approximation and under the exact
  dynamics (Exercise 3.4).

## Working files (not part of the document)

`ChN.tex` are the chapter sources, in which each solution is preceded by a short
restatement of the exercise; `tools/build_main.py` strips those restatements when it
assembles `AcemogluSolutions.tex`, and keeps them in the local
`AcemogluSolutions_with_exercises.tex`. Rebuild with `sh tools/mkchap.sh N`,
`sh tools/mkmain.sh` and `sh tools/mkmain.sh --full`; everything compiles outside the
project tree, so no `.aux`, `.log` or `.out` is ever left beside the sources.
