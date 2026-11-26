"""Second-order finite-volume reference solver (linear or Langmuir interface kinetics).

Used in the paper as an independent numerical check of the exact solution and as the
reference for the nonlinear (Langmuir) case, which has no closed form.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

__all__ = ["fv_reference"]


def fv_reference(alpha=0.5, gamma=0.1, K=2.0, Da=2.0, sigma=0.0, theta0=0.0, phi0=0.0,
                 nf=400, ns=400, taus=(0.1,), rtol=1e-10, atol=1e-12):
    """Cell-centred finite-volume solution of the dimensionless benchmark.

    The interface law is J* = Da [theta (1 - sigma phi) - phi]; sigma = 0 gives the linear
    benchmark and sigma = k_a C_in / k_d > 0 Langmuir kinetics (with K and Da defined from the
    Henry-limit rate constant). The interface values and J* are solved implicitly from the
    three interface relations; time integration uses a variable-order BDF method.

    Parameters
    ----------
    alpha, gamma, K, Da, theta0, phi0 : float
        As in ``Benchmark``.
    sigma : float
        Langmuir saturation parameter (0 = linear).
    nf, ns : int
        Number of cells in the fluid and in the solid layer.
    taus : sequence of float
        Output times (> 0).
    rtol, atol : float
        Tolerances of the time integrator.

    Returns
    -------
    xf, xs : ndarray
        Cell centres in the fluid and in the solid.
    t : ndarray
        Output times.
    theta, phi : ndarray
        Cell values, shape (nf, len(t)) and (ns, len(t)).
    """
    hf, hs = alpha / nf, (1 - alpha) / ns
    gf, gs = 2.0 / hf, 2.0 * gamma * K / hs          # half-cell conductances
    n = nf + ns

    def kin(th, ph):
        return Da * (th * (1 - sigma * ph) - ph)

    def interface(tc, pc):
        # J = gf (tc - ta) = gs (pa - pc) = kin(ta, pa)
        if sigma == 0.0:
            M = np.array([[gf, 0, 1], [0, -gs, 1], [-Da, Da, 1]])
            return np.linalg.solve(M, [gf * tc, -gs * pc, 0.0])[2]

        def res(J):
            return kin(tc - J / gf, pc + J / gs) - J
        return brentq(res, -1e3, 1e3, xtol=1e-14)

    def rhs(t, y):
        th, ph = y[:nf], y[nf:]
        J = interface(th[-1], ph[0])
        Ff = np.empty(nf + 1)
        Ff[0] = gf * (1.0 - th[0])
        Ff[1:-1] = (th[:-1] - th[1:]) / hf
        Ff[-1] = J
        Fs = np.empty(ns + 1)
        Fs[0] = J
        Fs[1:-1] = gamma * K * (ph[:-1] - ph[1:]) / hs
        Fs[-1] = 0.0
        return np.concatenate([(Ff[:-1] - Ff[1:]) / hf, (Fs[:-1] - Fs[1:]) / (K * hs)])

    S = lil_matrix((n, n))
    for i in range(n):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                S[i, j] = 1
    y0 = np.concatenate([np.full(nf, theta0), np.full(ns, phi0)])
    sol = solve_ivp(rhs, (0, max(taus)), y0, method="BDF", jac_sparsity=S,
                    t_eval=sorted(taus), rtol=rtol, atol=atol)
    xf = (np.arange(nf) + 0.5) * hf
    xs = alpha + (np.arange(ns) + 0.5) * hs
    return xf, xs, sol.t, sol.y[:nf], sol.y[nf:]
