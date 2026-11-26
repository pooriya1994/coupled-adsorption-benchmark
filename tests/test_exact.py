"""Tests of the exact series solution against the paper and against its own equations."""
import numpy as np
import pytest
from scipy.optimize import brentq

from adsorption_benchmark import Benchmark, ldf_time, terms_for

REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)
GENERAL = dict(alpha=0.3, gamma=5.0, K=10.0, Da=25.0, theta0=0.4, phi0=0.1)

# Table 1 of the paper
TABLE1 = [0.718367, 2.270657, 3.607149, 4.573110, 6.257725, 8.128313]
# Table 2 of the paper: tau, theta(0.25), theta(0.5), phi(0.5), phi(0.75), phi(1), phi_bar, J*
TABLE2 = [
    (0.05, 0.441899, 0.190341, 0.059053, 1.4720e-5, 2.2039e-11, 0.004671, 0.262577),
    (0.2, 0.782485, 0.615198, 0.351795, 0.028256, 0.001150, 0.074939, 0.526807),
    (1.0, 0.916674, 0.836290, 0.680351, 0.364672, 0.249708, 0.398515, 0.311878),
    (3.0, 0.970644, 0.942233, 0.886972, 0.773869, 0.731762, 0.785830, 0.110523),
]


@pytest.fixture(scope="module")
def ref():
    return Benchmark(**REF, n_terms=600)


@pytest.fixture(scope="module")
def gen():
    return Benchmark(**GENERAL, n_terms=600)


def _close_to_printed(value, printed):
    """True if value rounds to the printed table entry (4-6 significant digits)."""
    if abs(printed) >= 1e-3:
        return abs(value - printed) <= 5.01e-7
    return abs(value - printed) <= 1e-3 * abs(printed)


def test_eigenvalues_table1(ref):
    assert np.allclose(ref.eigenvalues[:6], TABLE1, atol=5.01e-7, rtol=0)


def test_eigenvalues_are_roots(gen):
    lam = gen.eigenvalues
    assert np.all(np.diff(lam) > 0)
    eps = 1e-6 * lam
    slope = (gen.char(lam + eps) - gen.char(lam - eps)) / (2 * eps)
    assert np.max(np.abs(gen.char(lam) / slope) / lam) < 1e-12        # Newton step relative to the root


@pytest.mark.parametrize("row", TABLE2)
def test_reference_values_table2(ref, row):
    tau = row[0]
    got = [ref.theta(0.25, tau)[0], ref.theta(0.5, tau)[0], ref.phi(0.5, tau)[0], ref.phi(0.75, tau)[0],
           ref.phi(1.0, tau)[0], ref.phi_mean(tau)[0], ref.flux(tau)[0]]
    for g, p in zip(got, row[1:]):
        assert _close_to_printed(g, p), (tau, g, p)


def test_interface_conditions(gen):
    """-theta_xi = -gamma K phi_xi = Da (theta - phi) at xi = alpha."""
    a, h, tau = gen.alpha, 1e-4, 0.3
    th = gen.theta([a, a - h, a - 2 * h], tau)
    ph = gen.phi([a, a + h, a + 2 * h], tau)
    dth = (3 * th[0] - 4 * th[1] + th[2]) / (2 * h)
    dph = (-3 * ph[0] + 4 * ph[1] - ph[2]) / (2 * h)
    J = gen.Da * (th[0] - ph[0])
    assert abs(-dth - J) < 1e-6 * abs(J)
    assert abs(-gen.gamma * gen.K * dph - J) < 1e-6 * abs(J)
    assert abs(gen.flux(tau)[0] - J) < 1e-12


def test_outer_boundaries_and_pde(gen):
    tau, h, dt = 0.2, 1e-3, 1e-5
    assert abs(gen.theta(0.0, tau)[0] - 1.0) < 1e-14
    ph = gen.phi([1.0 - 2 * h, 1.0 - h, 1.0], tau)
    assert abs((ph[0] - 4 * ph[1] + 3 * ph[2]) / (2 * h)) < 1e-5
    # theta_tau = theta_xixi and phi_tau = gamma phi_xixi at interior points
    x = 0.15
    tt = (gen.theta(x, tau + dt)[0] - gen.theta(x, tau - dt)[0]) / (2 * dt)
    t3 = gen.theta([x - h, x, x + h], tau)
    assert abs(tt - (t3[0] - 2 * t3[1] + t3[2]) / h**2) < 1e-4 * max(1.0, abs(tt))
    x = 0.7
    pt = (gen.phi(x, tau + dt)[0] - gen.phi(x, tau - dt)[0]) / (2 * dt)
    p3 = gen.phi([x - h, x, x + h], tau)
    assert abs(pt - gen.gamma * (p3[0] - 2 * p3[1] + p3[2]) / h**2) < 1e-4 * max(1.0, abs(pt))


def test_mass_balance_mode_by_mode(gen):
    """d/dtau [int theta + K int phi] = -theta_xi(0) holds for every truncation."""
    b = gen
    for tau in (0.02, 0.3, 2.0):
        e = b.c * np.exp(-b.lam**2 * tau)
        dM = np.sum(-e * b.lam**2 * (b.A * b.I1 + b.K * b.B * b.I2))
        inflow = -np.sum(e * b.A * b.lam)
        assert abs(dM - inflow) <= 1e-12 * abs(inflow)


def test_parseval_deficit_decays_like_one_over_n():
    out = []
    for n in (100, 400):
        b = Benchmark(**REF, n_terms=n)
        norm = b.A**2 * b.S1 + b.K * b.B**2 * b.S2
        total = (b.theta0 - 1) ** 2 * b.alpha + b.K * (b.phi0 - 1) ** 2 * (1 - b.alpha)
        out.append(total - np.sum(b.c**2 * norm))
    assert 0 < out[1] < out[0]
    assert out[0] / out[1] == pytest.approx(4.0, rel=0.02)


def test_equilibrium_and_initial_state(gen):
    assert np.allclose(gen.theta(np.linspace(0, gen.alpha, 7), 60.0), 1.0, atol=1e-10)
    assert np.allclose(gen.phi(np.linspace(gen.alpha, 1, 7), 60.0), 1.0, atol=1e-10)
    # far from the boundaries the series reproduces the initial state at small tau
    assert gen.theta(0.15, 1e-4)[0] == pytest.approx(GENERAL["theta0"], abs=1e-6)
    assert gen.phi(0.9, 1e-4)[0] == pytest.approx(GENERAL["phi0"], abs=1e-6)


def test_ldf_estimate_kinetic_regime():
    for Da, g in ((0.1, 0.1), (0.01, 1.0)):
        b = Benchmark(alpha=0.5, gamma=g, K=2.0, Da=Da, n_terms=3)
        assert abs(b.lam[0] ** 2 * ldf_time(0.5, g, 2.0, Da) - 1) < 1.5e-3


def test_limits_quoted_in_paper():
    lf = brentq(lambda x: x * np.tan(0.5 * x) - 1.0, 1e-6, np.pi - 1e-6) ** 2   # fluid-side control
    assert lf == pytest.approx(1.70705, abs=1e-5)
    assert Benchmark(alpha=0.5, gamma=1e4, K=2, Da=1e8, n_terms=2).lam[0] ** 2 == pytest.approx(1.70703, abs=1e-5)
    assert Benchmark(alpha=0.5, gamma=1e-4, K=2, Da=1e8, n_terms=2).lam[0] ** 2 == pytest.approx(9.8657e-4, rel=1e-4)


def test_truncation_rule():
    n = terms_for(0.5, 0.1, 1e-3)
    b = Benchmark(**REF, n_terms=n)
    assert b.lam[-1] ** 2 * 1e-3 > 40
    big = Benchmark(**REF, n_terms=2 * n)
    xs = np.linspace(0, 0.5, 11)
    assert np.max(np.abs(b.theta(xs, 1e-3) - big.theta(xs, 1e-3))) < 1e-13


def test_error_norms(ref):
    xf, xs = np.linspace(0, 0.5, 11), np.linspace(0.5, 1, 11)
    e2, einf = ref.error_norms(xf, ref.theta(xf, 0.2), xs, ref.phi(xs, 0.2), 0.2)
    assert e2 == 0.0 and einf == 0.0
    e2, einf = ref.error_norms(xf, ref.theta(xf, 0.2) + 1e-3, xs, ref.phi(xs, 0.2), 0.2)
    assert einf == pytest.approx(1e-3) and e2 == pytest.approx(1e-3 / np.sqrt(2))


@pytest.mark.parametrize("kw", [dict(alpha=0.0), dict(alpha=1.0), dict(gamma=0.0), dict(K=-1.0), dict(Da=0.0),
                                dict(n_terms=0)])
def test_invalid_input(kw):
    with pytest.raises(ValueError):
        Benchmark(**kw)
