"""The files in data/ agree with the solver (regenerate with reproducibility/reference_data.py)."""
import os

import numpy as np

from adsorption_benchmark import Benchmark

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
B = Benchmark(alpha=0.5, gamma=0.1, K=2.0, Da=2.0, n_terms=1000)


def _load(name):
    """Read a data/ CSV file (comment lines start with #) into a dict of columns."""
    with open(os.path.join(DATA, name)) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    head, rows = lines[0].split(","), [l.split(",") for l in lines[1:]]
    cols = {}
    for k, h in enumerate(head):
        vals = [r[k] for r in rows]
        try:
            cols[h] = np.array([float(v) for v in vals])
        except ValueError:
            cols[h] = np.array(vals)
    return cols


def test_eigenvalues_file():
    d = _load("eigenvalues.csv")
    assert np.allclose(d["lambda_n"], B.lam[:len(d["lambda_n"])], rtol=1e-13, atol=0)


def test_reference_values_file():
    d = _load("reference_values.csv")
    for k, t in enumerate(d["tau"]):
        assert abs(d["theta(0.25)"][k] - B.theta(0.25, t)[0]) < 1e-11
        assert abs(d["phi(0.75)"][k] - B.phi(0.75, t)[0]) < 1e-11
        assert abs(d["phi_bar"][k] - B.phi_mean(t)[0]) < 1e-11
        assert abs(d["J_star"][k] - B.flux(t)[0]) < 1e-11


def test_profiles_and_uptake_files():
    d = _load("reference_profiles.csv")
    for t in np.unique(d["tau"]):
        f = (d["tau"] == t) & (d["phase"] == "fluid")
        s = (d["tau"] == t) & (d["phase"] == "solid")
        assert np.allclose(d["value"][f], B.theta(d["xi"][f], t), atol=1e-11, rtol=0)
        assert np.allclose(d["value"][s], B.phi(d["xi"][s], t), atol=1e-11, rtol=0)
    u = _load("reference_uptake.csv")
    assert np.allclose(u["phi_bar"], B.phi_mean(u["tau"]), atol=1e-11, rtol=0)
    assert np.allclose(u["J_star"], B.flux(u["tau"]), atol=1e-11, rtol=0)
