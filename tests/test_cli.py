"""Command-line interface writes the same values as the Python API."""
import numpy as np

from adsorption_benchmark import Benchmark
from adsorption_benchmark.cli import main

REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)


def _read(path):
    rows = [l.strip().split(",") for l in open(path) if l.strip() and not l.startswith("#")]
    return rows[0], rows[1:]


def test_eigen(tmp_path):
    out = tmp_path / "eig.csv"
    assert main(["eigen", "-n", "6", "--out", str(out)]) == 0
    head, rows = _read(out)
    assert head == ["n", "lambda_n", "lambda_n^2"]
    assert len(rows) == 6
    assert abs(float(rows[0][1]) - 0.718367) < 5.01e-7


def test_table_prints_reference_values(capsys):
    main(["table"])
    text = capsys.readouterr().out
    assert "tau,theta(0.25),theta(0.5),phi(0.5),phi(0.75),phi(1),phi_bar,J*" in text
    rows = [l.split(",") for l in text.splitlines() if l and l[0].isdigit()]
    assert abs(float(rows[0][1]) - 0.441899) < 5.01e-7      # Table 2: theta(0.25) at tau = 0.05
    assert abs(float(rows[1][7]) - 0.526807) < 5.01e-7      # Table 2: J* at tau = 0.2


def test_profile_and_xi_file(tmp_path):
    out = tmp_path / "p.csv"
    main(["profile", "--tau", "0.2", "1", "--nodes", "11", "--out", str(out)])
    head, rows = _read(out)
    assert head == ["tau", "xi", "phase", "value"]
    assert len(rows) == 2 * 2 * 11
    b = Benchmark(**REF, n_terms=600)
    for t, x, ph, v in rows:
        ref = b.theta(float(x), float(t))[0] if ph == "fluid" else b.phi(float(x), float(t))[0]
        assert abs(float(v) - ref) < 1e-11

    nodes = tmp_path / "nodes.txt"
    nodes.write_text("# my nodes\n0.0\n0.25\n0.5\n0.75\n1.0\n")
    out2 = tmp_path / "p2.csv"
    main(["profile", "--tau", "1", "--xi-file", str(nodes), "--out", str(out2)])
    _, rows = _read(out2)
    assert [r[2] for r in rows] == ["fluid", "fluid", "fluid", "solid", "solid", "solid"]   # xi = alpha on both sides


def test_uptake(tmp_path):
    out = tmp_path / "u.csv"
    main(["uptake", "--tau-min", "0.1", "--tau-max", "3", "--n", "30", "--out", str(out)])
    head, rows = _read(out)
    assert head == ["tau", "phi_bar", "J_star", "jump"]
    t = np.array([float(r[0]) for r in rows])
    b = Benchmark(**REF, n_terms=600)
    assert np.allclose([float(r[1]) for r in rows], b.phi_mean(t), atol=1e-11)
    assert np.allclose([float(r[2]) for r in rows], b.flux(t), atol=1e-11)
