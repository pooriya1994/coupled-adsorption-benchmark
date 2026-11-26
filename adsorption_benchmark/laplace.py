"""Laplace-domain solution (resistances in series) and its numerical inversion.

    J_hat(s) = [ (theta0 - phi0)/s + (1 - theta0)/(s cosh(sqrt(s) alpha)) ] / (1/Da + Z_f + Z_s)
    Z_f = tanh(sqrt(s) alpha)/sqrt(s),   Z_s = coth(sqrt(s/gamma) (1 - alpha)) / (K sqrt(gamma s))

The kinetic (1/Da), fluid-side (Z_f) and solid-side (Z_s) resistances act in series. The
inversion (Talbot's method, multiprecision) needs no eigenvalues, which makes it an
independent check of the series solution.
"""
import mpmath as mp

__all__ = ["LaplaceSolution"]


class LaplaceSolution:
    """Exact solution evaluated by numerical inversion of the Laplace transforms.

    Slower than ``Benchmark`` but independent of root finding.

    Parameters
    ----------
    alpha, gamma, K, Da, theta0, phi0 : float
        As in ``Benchmark``.
    dps : int
        Decimal digits used by mpmath.
    """

    def __init__(self, alpha=0.5, gamma=0.1, K=2.0, Da=2.0, theta0=0.0, phi0=0.0, dps=30):
        self.dps = int(dps)
        with mp.workdps(self.dps):
            self.a, self.g, self.K, self.Da, self.th0, self.ph0 = map(
                mp.mpf, (alpha, gamma, K, Da, theta0, phi0))

    # transforms -----------------------------------------------------------------
    def J_hat(self, s):
        a, g, K, Da, th0, ph0 = self.a, self.g, self.K, self.Da, self.th0, self.ph0
        w, mu = mp.sqrt(s), mp.sqrt(s / g)
        Zf = mp.tanh(w * a) / w
        Zs = mp.coth(mu * (1 - a)) / (K * mp.sqrt(g * s))
        drive = (th0 - ph0) / s + (1 - th0) / (s * mp.cosh(w * a))
        return drive / (1 / Da + Zf + Zs)

    def theta_hat(self, xi, s):
        a, th0 = self.a, self.th0
        w = mp.sqrt(s)
        return (th0 / s + (1 - th0) * mp.cosh(w * (a - xi)) / (s * mp.cosh(w * a))
                - self.J_hat(s) * mp.sinh(w * xi) / (w * mp.cosh(w * a)))

    def phi_hat(self, xi, s):
        a, g, K = self.a, self.g, self.K
        mu = mp.sqrt(s / g)
        return self.ph0 / s + self.J_hat(s) * mp.cosh(mu * (1 - xi)) / (K * mp.sqrt(g * s) * mp.sinh(mu * (1 - a)))

    def phi_mean_hat(self, s):
        return self.ph0 / s + self.J_hat(s) / (self.K * (1 - self.a) * s)

    # inversion ------------------------------------------------------------------
    def _inv(self, F, tau):
        with mp.workdps(self.dps):
            return float(mp.invertlaplace(F, mp.mpf(tau), method="talbot"))

    def theta(self, xi, tau):
        """theta(xi, tau) for scalar xi in [0, alpha] and tau > 0."""
        return self._inv(lambda s: self.theta_hat(mp.mpf(xi), s), tau)

    def phi(self, xi, tau):
        """phi(xi, tau) for scalar xi in [alpha, 1] and tau > 0."""
        return self._inv(lambda s: self.phi_hat(mp.mpf(xi), s), tau)

    def phi_mean(self, tau):
        """Mean solid loading phi_bar(tau), tau > 0."""
        return self._inv(self.phi_mean_hat, tau)

    def flux(self, tau):
        """Interfacial flux J*(tau), tau > 0."""
        return self._inv(self.J_hat, tau)
