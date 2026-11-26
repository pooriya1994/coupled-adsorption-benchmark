"""The finite-volume reference solver converges to the exact solution at second order."""
import numpy as np
import pytest

from adsorption_benchmark import Benchmark, fv_reference

REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)


def _rms_error(b, n, tau):
    xf, xs, _, th, ph = fv_reference(**REF, nf=n, ns=n, taus=[tau])
    return np.sqrt(np.mean(np.r_[(b.theta(xf, tau) - th[:, 0]) ** 2, (b.phi(xs, tau) - ph[:, 0]) ** 2]))


def test_second_order_convergence():
    b = Benchmark(**REF, n_terms=400)
    e25, e50 = _rms_error(b, 25, 0.2), _rms_error(b, 50, 0.2)
    assert e25 == pytest.approx(1.11e-4, rel=0.02)          # value quoted in the paper
    assert np.log2(e25 / e50) == pytest.approx(2.0, abs=0.05)


def test_general_case_agreement():
    kw = dict(alpha=0.3, gamma=5.0, K=10.0, Da=25.0, theta0=0.4, phi0=0.1)
    b = Benchmark(**kw, n_terms=400)
    xf, xs, T, th, ph = fv_reference(**kw, nf=200, ns=200, taus=[0.1, 1.0])
    for k, t in enumerate(T):
        assert np.max(np.abs(b.theta(xf, t) - th[:, k])) < 2e-5
        assert np.max(np.abs(b.phi(xs, t) - ph[:, k])) < 2e-5


def test_langmuir_limits():
    _, _, _, th_lin, ph_lin = fv_reference(**REF, nf=30, ns=30, taus=[0.5])
    _, _, _, th_l, ph_l = fv_reference(**REF, sigma=1e-10, nf=30, ns=30, taus=[0.5])
    assert np.max(np.abs(ph_l - ph_lin)) < 1e-8
    # saturated steady state: phi -> 1/(1 + sigma)
    sigma = 0.5
    _, _, _, th, ph = fv_reference(**REF, sigma=sigma, nf=20, ns=20, taus=[60.0])
    assert np.allclose(th[:, 0], 1.0, atol=1e-6)
    assert np.allclose(ph[:, 0], 1.0 / (1.0 + sigma), atol=1e-6)
