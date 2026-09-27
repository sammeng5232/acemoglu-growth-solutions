# Solutions to Daron Acemoglu, *Introduction to Modern Economic Growth* (Princeton University Press, 2009)

Authors: Zijun Meng and Claude Opus 5

## Contents

Chapter 2 (*The Solow Growth Model*) has 27 exercises. The instructor's solutions
manual of Peters and Simsek (2009) solves 13 of them — 2.7, 2.11, 2.12, 2.14,
2.16–2.23 and 2.27. This document solves the other **14**:

| | | |
|---|---|---|
| **2.1** competitive labor markets pay a strictly positive wage | **2.2** constant returns imply concavity, but never strict concavity | **2.3** firm size is indeterminate under constant returns |
| **2.4** a quartic production function with multiple steady states | **2.5** existence and uniqueness of the steady state in continuous time (Proposition 2.7) | **2.6** comparative statics in continuous time (Proposition 2.8) |
| **2.8** dropping strict concavity (Propositions 2.2 and 2.5) | **2.9** factor prices along the transition path (Proposition 2.6) | **2.10** stability of scalar differential equations (Corollary 2.2) |
| **2.13** saving out of labor income only | **2.15** the elasticity of output with respect to the wage | **2.24** the CES function and the Inada conditions |
| **2.25** comparative statics on the BGP (Proposition 2.12) | **2.26** stability of the BGP (Proposition 2.13) | |

**The exercise statements are not reproduced.** Each solution is headed only by the
number of its exercise in the book, so read the statement there first. Notation and
the part letters (a), (b), … follow the book, and numbered results cited as
"(2.33)", "Assumption 2" or "Proposition 2.5" refer to Acemoglu (2009). Remarks
inside the solutions point out two places where the book's own statements do not
match what the mathematics gives (the stability count in Exercise 2.4(c), and
whether the CES production function violates Assumption 1 or Assumption 2).

| File | Contents |
|---|---|
| `AcemogluSolutions.tex` / `.pdf` | the solutions in one self-contained document (17 pages; solutions only, without the exercise statements) |
| `figures/` | the two figures, produced by the code |
| `code/` | Python for the numerical parts (see below) |

## Building

Compile from this directory with

```bash
pdflatex AcemogluSolutions.tex
```

two or three times, so that the contents page numbers and cross-references settle.
The document needs no local style file; it includes the PDF figures from `figures/`.

## Code

`code/ch2.py` (with the shared helpers in `code/acemoglulib.py`) needs only `numpy`,
`scipy` and `matplotlib`. Run it with `python code/ch2.py`: it prints every number
quoted in the solutions and writes the two figures to `figures/`. It covers

- **Exercise 2.4** — the three interior steady states of the quartic economy, the
  threshold \(2\sqrt3/9\) on \((n+\delta)/s\) below which they exist, and the sign of
  \(\mathrm{d}(f(k)/k)/\mathrm{d}k\) at each of them, which settles the stability pattern;
- **Exercise 2.13** — the explicit technology (a CES with a low elasticity of
  substitution plus a small Cobb–Douglas term) that satisfies Assumptions 1 and 2 and
  yields three steady states when only labor income is saved, together with the check
  that \(w(k)/k\) rises exactly where the elasticity of substitution falls below the
  capital share.

## Working files (not part of the document)

`Ch2.tex` is the chapter source, in which each solution is preceded by a short
restatement of the exercise; `tools/build_main.py` strips those restatements when it
assembles `AcemogluSolutions.tex`, and keeps them in the local
`AcemogluSolutions_with_exercises.tex`. Rebuild with `sh tools/mkchap.sh 2`,
`sh tools/mkmain.sh` and `sh tools/mkmain.sh --full`; everything compiles outside the
project tree, so no `.aux`, `.log` or `.out` is ever left beside the sources.
