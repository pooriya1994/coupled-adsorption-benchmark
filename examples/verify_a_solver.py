"""How to verify a numerical solver with the benchmark: errors and observed order.

Here the "solver under test" is the finite-volume reference solver of this package, run on
four grids. To test your own code (LB, FV, FE ...), replace ``run_solver`` by a function that
returns your node positions xi = x/l and values theta = C/C_in (fluid nodes) and
phi = N/(K C_in) (solid nodes) at time tau = D_f t / l^2, e.g. read from a CSV file:

    data = np.genfromtxt("my_output.csv", delimiter=",", names=True, dtype=None, encoding=None)
    fluid = data["phase"] == "fluid"
    xi_f, theta = data["xi"][fluid], data["value"][fluid]
    xi_s, phi = data["xi"][~fluid], data["value"][~fluid]

Remember to evaluate the exact solution with the alpha = a/l of each grid if the interface
position changes slightly with resolution.
"""
import numpy as np

from adsorption_benchmark import Benchmark, fv_reference

REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)
TAU = 0.2


def run_solver(n):
    """Solver under test on a grid with n cells per layer; returns xi_f, theta, xi_s, phi at TAU."""
    xf, xs, _, th, ph = fv_reference(**REF, nf=n, ns=n, taus=[TAU])
    return xf, th[:, 0], xs, ph[:, 0]


exact = Benchmark(**REF)
print(f"{'cells/layer':>11} {'E2':>11} {'Einf':>11} {'order':>6}")
prev = None
for n in (25, 50, 100, 200):
    e2, einf = exact.error_norms(*run_solver(n), TAU)
    order = "" if prev is None else f"{np.log(prev / e2) / np.log(2):6.2f}"
    print(f"{n:11d} {e2:11.3e} {einf:11.3e} {order:>6}")
    prev = e2
print("A second-order scheme, including its interface treatment, shows an order close to 2.")
