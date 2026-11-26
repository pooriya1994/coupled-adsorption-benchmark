"""Write the reference data files in data/ (reference case alpha=0.5, gamma=0.1, K=2, Da=2, clean start).

Run from the repository root:  python reproducibility/reference_data.py
"""
import os

import numpy as np

from adsorption_benchmark import Benchmark, __version__

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)
REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)
HEAD = (f"# coupled adsorption benchmark v{__version__}, reference case: alpha=0.5 gamma=0.1 K=2 Da=2 "
        f"theta0=0 phi0=0\n")
B = Benchmark(**REF, n_terms=1000)


def write(name, header, rows, note=""):
    with open(os.path.join(DATA, name), "w") as f:
        f.write(HEAD)
        if note:
            f.write(f"# {note}\n")
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print("wrote", os.path.join("data", name))


write("eigenvalues.csv", "n,lambda_n,lambda_n^2",
      [[str(k + 1), f"{v:.15g}", f"{v * v:.15g}"] for k, v in enumerate(B.lam[:50])],
      "modes decay as exp(-lambda_n^2 tau); Table 1 of the paper lists the first six")

rows = []
for t in (0.05, 0.2, 1.0, 3.0):
    vals = [B.theta(0.25, t)[0], B.theta(0.5, t)[0], B.phi(0.5, t)[0], B.phi(0.75, t)[0], B.phi(1.0, t)[0],
            B.phi_mean(t)[0], B.flux(t)[0]]
    rows.append([f"{t:g}"] + [f"{v:.12g}" for v in vals])
write("reference_values.csv", "tau,theta(0.25),theta(0.5),phi(0.5),phi(0.75),phi(1),phi_bar,J_star", rows,
      "Table 2 of the paper; theta(0.5) is the fluid-side and phi(0.5) the solid-side interface value; "
      "absolute accuracy about 1e-15 (round-off), so values below 1e-12 have few correct digits")

rows = []
xf, xs = np.linspace(0.0, 0.5, 101), np.linspace(0.5, 1.0, 101)
for t in (0.05, 0.2, 1.0, 3.0):
    rows += [[f"{t:g}", f"{x:.6g}", "fluid", f"{v:.12g}"] for x, v in zip(xf, B.theta(xf, t))]
    rows += [[f"{t:g}", f"{x:.6g}", "solid", f"{v:.12g}"] for x, v in zip(xs, B.phi(xs, t))]
write("reference_profiles.csv", "tau,xi,phase,value", rows,
      "value = theta = C/C_in in the fluid and phi = N/(K C_in) in the solid; xi = 0.5 appears on both sides; "
      "absolute accuracy about 1e-15")

taus = np.round(np.arange(1, 301) * 0.01, 10)
rows = [[f"{t:g}", f"{p:.12g}", f"{j:.12g}", f"{j / B.Da:.12g}"]
        for t, p, j in zip(taus, B.phi_mean(taus), B.flux(taus))]
write("reference_uptake.csv", "tau,phi_bar,J_star,jump", rows,
      "phi_bar = mean solid loading, J_star = J l/(D_f C_in), jump = theta(alpha-) - phi(alpha+) = J_star/Da")
