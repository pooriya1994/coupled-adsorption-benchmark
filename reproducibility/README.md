# Reproducing the paper

Run from the repository root after `pip install -e ".[plot]"`.

| Script | Output | Paper |
|---|---|---|
| `make_figures.py` | `output/fig_profiles.pdf`, `fig_truncation.pdf`, `fig_uptake.pdf`, `fig_regime.pdf`, `fig_langmuir.pdf`, `numbers.json` | Figs. 2–6; FV comparison, grid convergence, mass balance and Langmuir numbers of Sections 4–6 |
| `laplace_check.py` | printed table | Table 2 confirmed by Laplace inversion; LDF estimate of $\lambda_1^2$ |
| `reference_data.py` | `../data/*.csv` | Tables 1–2 and reference profiles/uptake curves |

`make_figures.py` takes about ten seconds; `laplace_check.py` a few seconds.
The lattice Boltzmann results of Section 7 come from a separate code that is not part of this repository.
