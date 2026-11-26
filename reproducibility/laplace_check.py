"""Independent check of Table 2: numerical inversion of the Laplace-domain solution (Talbot, 30 digits).

Run from the repository root:  python reproducibility/laplace_check.py
"""
from adsorption_benchmark import Benchmark, LaplaceSolution, ldf_time

REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)
L = LaplaceSolution(**REF, dps=30)
B = Benchmark(**REF, n_terms=600)

worst = 0.0
print("tau | theta(0.25) theta(0.5) phi(0.5) phi(0.75) phi(1) phi_bar J*   (Laplace inversion)")
for t in (0.05, 0.2, 1.0, 3.0):
    lap = [L.theta(0.25, t), L.theta(0.5, t), L.phi(0.5, t), L.phi(0.75, t), L.phi(1.0, t),
           L.phi_mean(t), L.flux(t)]
    ser = [B.theta(0.25, t)[0], B.theta(0.5, t)[0], B.phi(0.5, t)[0], B.phi(0.75, t)[0],
           B.phi(1.0, t)[0], B.phi_mean(t)[0], B.flux(t)[0]]
    d = max(abs(x - y) for x, y in zip(lap, ser))
    worst = max(worst, d)
    print(f"{t:g} | " + " ".join(f"{x:.10g}" for x in lap) + f"   max|series - Laplace| = {d:.1e}")
print(f"worst difference: {worst:.1e}")

# linear-driving-force (resistances in series) estimate of the slowest decay rate
print("lambda_1^2: exact", B.lam[0] ** 2, " LDF estimate", 1.0 / ldf_time(**REF))
