"""Exact solution of the coupled fluid-solid adsorption benchmark.

Dimensionless problem (xi = x/l, tau = D_f t / l^2, theta = C/C_in, phi = N/(K C_in)):

    fluid  0 < xi < alpha :  theta_tau = theta_xixi,        theta(0, tau) = 1
    solid  alpha < xi < 1 :  phi_tau = gamma * phi_xixi,    phi_xi(1, tau) = 0
    xi = alpha            :  -theta_xi = -gamma*K*phi_xi = Da*(theta - phi)
    tau = 0               :  theta = theta0,  phi = phi0

Groups: alpha = a/l, gamma = D_s/D_f, K = k_a/k_d (Henry constant), Da = k_a l/D_f.
C and N are concentrations per unit volume of their own phase, so k_a and k_d are
surface rate constants (m/s) and K is dimensionless.
"""
import numpy as np
from scipy.optimize import brentq

__all__ = ["Benchmark", "PhysicalBenchmark", "ldf_time", "terms_for"]


def _check_groups(alpha, gamma, K, Da):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    for name, v in (("gamma", gamma), ("K", K), ("Da", Da)):
        if not v > 0.0:
            raise ValueError(f"{name} must be positive")


def ldf_time(alpha, gamma, K, Da):
    """Linear-driving-force estimate of the equilibration time scale.

    tau_LDF = K (1 - alpha) [1/Da + alpha + (1 - alpha)/(3 gamma K)], so that
    phi_bar ~ 1 - (1 - phi0) exp(-tau/tau_LDF) and lambda_1^2 ~ 1/tau_LDF.
    """
    _check_groups(alpha, gamma, K, Da)
    return K * (1.0 - alpha) * (1.0 / Da + alpha + (1.0 - alpha) / (3.0 * gamma * K))


def terms_for(alpha, gamma, tau_min, target=40.0):
    """Number of series terms for which lambda_N^2 * tau_min exceeds ``target``.

    High eigenvalues are spaced by about pi / (alpha + (1 - alpha)/sqrt(gamma)); a margin
    is added. With target = 40 the truncation error is below machine precision.
    """
    if not tau_min > 0.0:
        raise ValueError("tau_min must be positive (the series does not converge at tau = 0)")
    w = alpha + (1.0 - alpha) / np.sqrt(gamma)
    return int(np.ceil(1.2 * w * np.sqrt(target / tau_min) / np.pi)) + 20


class Benchmark:
    """Eigenfunction-series solution of the dimensionless benchmark.

    Parameters
    ----------
    alpha, gamma, K, Da : float
        Fluid fraction a/l, diffusivity ratio D_s/D_f, Henry constant k_a/k_d and
        Damkoehler number k_a l / D_f.
    theta0, phi0 : float, optional
        Uniform initial state C(x,0)/C_in and N(x,0)/(K C_in). Default 0 (clean system).
    n_terms : int, optional
        Number of eigenvalues kept. The series converges exponentially for tau > 0;
        ``terms_for(alpha, gamma, tau_min)`` gives a safe number for the smallest tau needed.

    Notes
    -----
    theta(xi, tau) = 1 + sum_n c_n A_n sin(lam_n xi) exp(-lam_n^2 tau)
    phi(xi, tau)   = 1 + sum_n c_n B_n cos(lam_n (1 - xi)/sqrt(gamma)) exp(-lam_n^2 tau)
    with lam_n the roots of the characteristic function ``char``.
    """

    def __init__(self, alpha=0.5, gamma=0.1, K=2.0, Da=2.0, theta0=0.0, phi0=0.0, n_terms=300):
        _check_groups(alpha, gamma, K, Da)
        if int(n_terms) < 1:
            raise ValueError("n_terms must be at least 1")
        self.alpha, self.gamma, self.K, self.Da = float(alpha), float(gamma), float(K), float(Da)
        self.theta0, self.phi0 = float(theta0), float(phi0)
        self.sg = np.sqrt(self.gamma)
        self._roots(int(n_terms))
        self._coeffs()

    # ------------------------------------------------------------------ eigenvalues
    def char(self, lam):
        """Characteristic function F(lambda); its positive roots are the eigenvalues."""
        p = lam * self.alpha
        q = lam * (1 - self.alpha) / self.sg
        kg = self.K * self.sg
        return (self.Da * np.cos(p) * np.cos(q)
                - kg * lam * np.cos(p) * np.sin(q)
                - self.Da * kg * np.sin(p) * np.sin(q))

    def _roots(self, n):
        # F has no poles; scan with a step well below the shortest oscillation, refine with Brent
        w = self.alpha + (1 - self.alpha) / self.sg
        h = 0.01 / w
        out, lo, flo = [], 1e-10, self.char(1e-10)
        while len(out) < n:
            hi = lo + h
            fhi = self.char(hi)
            if flo * fhi < 0:
                out.append(brentq(self.char, lo, hi, xtol=1e-15))
            elif fhi == 0.0:
                out.append(hi)
            lo, flo = hi, fhi
        self.lam = np.array(out)

    def _coeffs(self):
        lam, a, sg, K = self.lam, self.alpha, self.sg, self.K
        p, q = lam * a, lam * (1 - a) / sg
        A = K * sg * np.sin(q)
        B = np.cos(p)
        # non-generic modes (cos p = sin q = 0): amplitudes from the kinetic condition
        deg = (np.abs(A) < 1e-10) & (np.abs(B) < 1e-10)
        A = np.where(deg, np.cos(q), A)
        B = np.where(deg, np.sin(p), B)
        I1 = (1 - np.cos(p)) / lam
        I2 = sg * np.sin(q) / lam
        S1 = a / 2 - np.sin(2 * p) / (4 * lam)
        S2 = (1 - a) / 2 + sg * np.sin(2 * q) / (4 * lam)
        num = (self.theta0 - 1) * A * I1 + K * (self.phi0 - 1) * B * I2
        den = A**2 * S1 + K * B**2 * S2
        self.A, self.B, self.I1, self.I2, self.S1, self.S2 = A, B, I1, I2, S1, S2
        self.c = num / den

    @property
    def eigenvalues(self):
        """The eigenvalues lambda_n (modes decay as exp(-lambda_n^2 tau))."""
        return self.lam.copy()

    # ------------------------------------------------------------------ fields
    def theta(self, xi, tau):
        """Fluid concentration theta = C/C_in at positions xi (0 <= xi <= alpha), scalar tau > 0."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))[:, None]
        return 1 + (self.A * np.sin(self.lam * xi)) @ (self.c * np.exp(-self.lam**2 * tau))

    def phi(self, xi, tau):
        """Adsorbed loading phi = N/(K C_in) at positions xi (alpha <= xi <= 1), scalar tau > 0."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))[:, None]
        X = self.B * np.cos(self.lam * (1 - xi) / self.sg)
        return 1 + X @ (self.c * np.exp(-self.lam**2 * tau))

    def phi_mean(self, tau):
        """Mean solid loading (uptake curve) phi_bar(tau), for scalar or array tau."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))[:, None]
        return 1 + np.exp(-self.lam**2 * tau) @ (self.c * self.B * self.I2) / (1 - self.alpha)

    def flux(self, tau):
        """Dimensionless interfacial adsorption rate J* = J l/(D_f C_in) = Da [theta(alpha) - phi(alpha)]."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))[:, None]
        p, q = self.lam * self.alpha, self.lam * (1 - self.alpha) / self.sg
        return self.Da * (np.exp(-self.lam**2 * tau) @ (self.c * (self.A * np.sin(p) - self.B * np.cos(q))))

    def interface_jump(self, tau):
        """Kinetic jump theta(alpha^-) - phi(alpha^+) = J*/Da."""
        return self.flux(tau) / self.Da

    def ldf_time(self):
        """Linear-driving-force time scale tau_LDF of this case (see ``ldf_time``)."""
        return ldf_time(self.alpha, self.gamma, self.K, self.Da)

    # ------------------------------------------------------------------ verification helpers
    def error_norms(self, xi_fluid, theta_num, xi_solid, phi_num, tau):
        """E_2 (root-mean-square) and E_inf errors of a numerical solution at time tau.

        ``theta_num`` are values at fluid nodes ``xi_fluid`` and ``phi_num`` values at solid
        nodes ``xi_solid``. Returns (E2, Einf), taken over all nodes together.
        """
        ef = np.asarray(theta_num, dtype=float) - self.theta(xi_fluid, tau)
        es = np.asarray(phi_num, dtype=float) - self.phi(xi_solid, tau)
        e = np.r_[ef, es]
        return float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))


class PhysicalBenchmark:
    """The same exact solution in physical units (or lattice units, with dx = dt = 1).

    Parameters
    ----------
    Df, Ds : float
        Diffusivities of the fluid species and of the adsorbed phase.
    ka, kd : float
        Adsorption and desorption rate constants of J = ka C - kd N (C and N per unit
        volume of their own phase, so ka and kd have units of length/time).
    a, l : float
        Fluid-layer thickness and total thickness (interface at x = a, impermeable wall at x = l).
    C_in : float
        Fluid concentration imposed at x = 0. May be zero (desorption into a clean stream).
    c0, N0 : float, optional
        Uniform initial fluid concentration and loading.
    n_terms : int, optional
        Number of series terms (see ``Benchmark``).
    """

    def __init__(self, Df, Ds, ka, kd, a, l, C_in, c0=0.0, N0=0.0, n_terms=300):
        for name, v in (("Df", Df), ("Ds", Ds), ("ka", ka), ("kd", kd), ("l", l)):
            if not v > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < a < l:
            raise ValueError("the interface position a must satisfy 0 < a < l")
        self.Df, self.Ds, self.ka, self.kd = float(Df), float(Ds), float(ka), float(kd)
        self.a, self.l, self.C_in = float(a), float(l), float(C_in)
        self.c0, self.N0 = float(c0), float(N0)
        self.K = self.ka / self.kd
        groups = dict(alpha=self.a / self.l, gamma=self.Ds / self.Df, K=self.K, Da=self.ka * self.l / self.Df,
                      n_terms=n_terms)
        if self.C_in != 0.0:
            self._scale = self.C_in
            self._b = Benchmark(theta0=self.c0 / self.C_in, phi0=self.N0 / (self.K * self.C_in), **groups)
            self._b0 = None
        else:
            # linearity: inlet 0 = (inlet 1, initial state c0, N0) - (inlet 1, clean initial state)
            self._scale = 1.0
            self._b = Benchmark(theta0=self.c0, phi0=self.N0 / self.K, **groups)
            self._b0 = Benchmark(theta0=0.0, phi0=0.0, **groups)
        self.groups = {k: v for k, v in groups.items() if k != "n_terms"}

    @property
    def decay_rates(self):
        """Decay rates beta_n^2 = lambda_n^2 D_f / l^2 (1/time) of the modes."""
        return self._b.eigenvalues**2 * self.Df / self.l**2

    def tau(self, t):
        """Dimensionless time D_f t / l^2."""
        return self.Df * np.asarray(t, dtype=float) / self.l**2

    def time(self, tau):
        """Physical time for a dimensionless time tau."""
        return np.asarray(tau, dtype=float) * self.l**2 / self.Df

    def _eval(self, fun, *args):
        v = getattr(self._b, fun)(*args)
        if self._b0 is not None:
            v = v - getattr(self._b0, fun)(*args)
        return v

    def C(self, x, t):
        """Fluid concentration at positions x (0 <= x <= a), scalar time t > 0."""
        return self._scale * self._eval("theta", np.asarray(x, dtype=float) / self.l, float(self.tau(t)))

    def N(self, x, t):
        """Adsorbed loading at positions x (a <= x <= l), scalar time t > 0."""
        return self.K * self._scale * self._eval("phi", np.asarray(x, dtype=float) / self.l, float(self.tau(t)))

    def mean_loading(self, t):
        """Mean adsorbed loading over the adsorbent layer, scalar or array t."""
        return self.K * self._scale * self._eval("phi_mean", self.tau(t))

    def flux(self, t):
        """Interfacial adsorption rate J = ka C - kd N at x = a (positive for net adsorption)."""
        return self.Df * self._scale / self.l * self._eval("flux", self.tau(t))

    def ldf_time(self):
        """Linear-driving-force time scale t_LDF = K (l - a) [1/ka + a/Df + (l - a)/(3 K Ds)]."""
        return self.K * (self.l - self.a) * (1.0 / self.ka + self.a / self.Df + (self.l - self.a) / (3.0 * self.K * self.Ds))
