"""Two exact representations (series and inverted Laplace transform) agree to round-off."""
import pytest

from adsorption_benchmark import Benchmark, LaplaceSolution


@pytest.mark.parametrize("kw", [dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0),
                                dict(alpha=0.3, gamma=5.0, K=10.0, Da=25.0, theta0=0.4, phi0=0.1)])
def test_series_matches_laplace_inversion(kw):
    b = Benchmark(**kw, n_terms=600)
    L = LaplaceSolution(**kw, dps=30)
    a = kw["alpha"]
    for tau in (0.05, 1.0):
        assert abs(L.theta(a / 2, tau) - b.theta(a / 2, tau)[0]) < 1e-12
        assert abs(L.phi((1 + a) / 2, tau) - b.phi((1 + a) / 2, tau)[0]) < 1e-12
        assert abs(L.phi_mean(tau) - b.phi_mean(tau)[0]) < 1e-12
        assert abs(L.flux(tau) - b.flux(tau)[0]) < 1e-12
