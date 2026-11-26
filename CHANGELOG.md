# Changelog

## 1.0.0 — 2026-10-25

First public release, accompanying the paper *An exact coupled benchmark for lattice Boltzmann
simulation of adsorption: transient fluid–solid diffusion with interfacial adsorption–desorption
kinetics*.

- `Benchmark`: exact eigenfunction-series solution (fields, mean loading, interfacial flux,
  interface jump, eigenvalues, error norms for solver verification).
- `LaplaceSolution`: Laplace-domain solution with Talbot inversion (independent check).
- `fv_reference`: second-order finite-volume solver, linear or Langmuir kinetics.
- `PhysicalBenchmark`: physical or lattice units, including desorption into a clean stream.
- `ldf_time`, `terms_for`: linear-driving-force time scale and truncation rule.
- Command line `adsorption-benchmark` (`eigen`, `table`, `profile`, `uptake`) writing CSV files.
- Reference data (`data/`), scripts reproducing Figs. 2–6 and Table 2, and a test suite.
