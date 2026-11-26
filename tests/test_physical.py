"""Physical-unit wrapper: scaling, flux definition, equilibrium and the C_in = 0 case."""
import numpy as np
import pytest

from adsorption_benchmark import Benchmark, PhysicalBenchmark

P = dict(Df=2e-9, Ds=1e-10, ka=1e-6, kd=5e-7, a=1e-4, l=3e-4, C_in=5.0, c0=1.0, N0=2.0)


@pytest.fixture(scope="module")
def pb():
    return PhysicalBenchmark(**P, n_terms=400)


def test_scaling_matches_dimensionless(pb):
    K = P["ka"] / P["kd"]
    b = Benchmark(alpha=P["a"] / P["l"], gamma=P["Ds"] / P["Df"], K=K, Da=P["ka"] * P["l"] / P["Df"],
                  theta0=P["c0"] / P["C_in"], phi0=P["N0"] / (K * P["C_in"]), n_terms=400)
    t = 5.0
    tau = P["Df"] * t / P["l"] ** 2
    x = np.linspace(0, P["a"], 5)
    assert np.allclose(pb.C(x, t), P["C_in"] * b.theta(x / P["l"], tau), rtol=1e-13, atol=0)
    x = np.linspace(P["a"], P["l"], 5)
    assert np.allclose(pb.N(x, t), K * P["C_in"] * b.phi(x / P["l"], tau), rtol=1e-13, atol=0)
    assert pb.tau(t) == pytest.approx(tau) and pb.time(tau) == pytest.approx(t)
    assert np.allclose(pb.decay_rates, b.eigenvalues ** 2 * P["Df"] / P["l"] ** 2)
    assert pb.ldf_time() == pytest.approx(b.ldf_time() * P["l"] ** 2 / P["Df"], rel=1e-12)


def test_flux_is_ka_c_minus_kd_n_and_fluid_flux(pb):
    t, a, h = 3.0, P["a"], 1e-8
    J = pb.flux(t)[0]
    C_a, N_a = pb.C(a, t)[0], pb.N(a, t)[0]
    assert J == pytest.approx(P["ka"] * C_a - P["kd"] * N_a, rel=1e-10)
    C3 = pb.C([a, a - h, a - 2 * h], t)
    dCdx = (3 * C3[0] - 4 * C3[1] + C3[2]) / (2 * h)
    assert J == pytest.approx(-P["Df"] * dCdx, rel=1e-5)


def test_equilibrium(pb):
    t_long = 40 * pb.ldf_time()
    assert pb.mean_loading(t_long)[0] == pytest.approx(P["ka"] / P["kd"] * P["C_in"], rel=1e-9)
    assert pb.C(P["a"] / 2, t_long)[0] == pytest.approx(P["C_in"], rel=1e-9)


def test_desorption_into_clean_stream():
    q = dict(P, C_in=0.0, c0=0.0, N0=3.0)
    clean = PhysicalBenchmark(**q, n_terms=400)
    tiny = PhysicalBenchmark(**dict(q, C_in=1e-7), n_terms=400)
    K = P["ka"] / P["kd"]
    t = 2.0
    xs = np.linspace(P["a"], P["l"], 7)
    # the two differ only by the response to an inlet value of 1e-7
    assert np.max(np.abs(clean.N(xs, t) - tiny.N(xs, t))) <= 1.01e-7 * K
    assert clean.flux(t)[0] < 0                              # net desorption
    assert clean.mean_loading(1e-6)[0] == pytest.approx(3.0, rel=1e-6)
    assert abs(clean.mean_loading(40 * clean.ldf_time())[0]) < 1e-8


def test_lattice_units_reference_case():
    # wall layout of the paper with nF = 50: a = 49.5, l = 99.5, Df = 0.2, gamma = 0.1, K = 2, Da = 2
    l, Df = 99.5, 0.2
    ka = 2.0 * Df / l
    pbl = PhysicalBenchmark(Df=Df, Ds=0.1 * Df, ka=ka, kd=ka / 2.0, a=49.5, l=l, C_in=1.0)
    g = pbl.groups
    assert g["alpha"] == pytest.approx(49.5 / 99.5) and g["Da"] == pytest.approx(2.0) and g["K"] == pytest.approx(2.0)
    assert pbl.time(3.0) == pytest.approx(3.0 * l**2 / Df)


def test_invalid_physical_input():
    with pytest.raises(ValueError):
        PhysicalBenchmark(**dict(P, a=P["l"]))
    with pytest.raises(ValueError):
        PhysicalBenchmark(**dict(P, Ds=0.0))
