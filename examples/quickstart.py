"""Quick start: evaluate the exact solution and plot it.

python examples/quickstart.py      (the plot needs matplotlib: pip install -e ".[plot]")
"""
import numpy as np

from adsorption_benchmark import Benchmark, PhysicalBenchmark

# --- dimensionless form: alpha = a/l, gamma = Ds/Df, K = ka/kd, Da = ka l/Df -------------
b = Benchmark(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)          # reference case of the paper

print("first eigenvalues:", np.round(b.eigenvalues[:6], 6))
print("theta(0.25, tau=1) =", b.theta(0.25, 1.0)[0])         # fluid concentration C/C_in
print("phi(0.75, tau=1)   =", b.phi(0.75, 1.0)[0])           # adsorbed loading N/(K C_in)
print("phi_bar(tau=1)     =", b.phi_mean(1.0)[0])            # uptake curve
print("J*(tau=1)          =", b.flux(1.0)[0])                # interfacial adsorption rate
print("jump at interface  =", b.interface_jump(1.0)[0])      # theta(alpha-) - phi(alpha+)
print("LDF time scale     =", b.ldf_time())

# --- physical (or lattice) units -----------------------------------------------------------
# Lattice units of a 2D LB code with l = 99.5 spacings and D_f = 0.2 (dx = dt = 1):
l, Df = 99.5, 0.2
ka = 2.0 * Df / l                                            # Da = ka l / Df = 2
lb = PhysicalBenchmark(Df=Df, Ds=0.1 * Df, ka=ka, kd=ka / 2.0, a=49.5, l=l, C_in=1.0)
t = lb.time(1.0)                                             # number of time steps to tau = 1
print(f"tau = 1 is reached after {t:.0f} lattice time steps")
print("C at the interface (fluid side):", lb.C(49.5, t)[0], "  N (solid side):", lb.N(49.5, t)[0])

# --- plot ---------------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed: skipping the plot")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
xf, xs = np.linspace(0, b.alpha, 200), np.linspace(b.alpha, 1, 200)
for tau in (0.05, 0.2, 1.0, 3.0):
    line, = ax[0].plot(xf, b.theta(xf, tau), label=fr"$\tau={tau:g}$")
    ax[0].plot(xs, b.phi(xs, tau), color=line.get_color())
ax[0].axvline(b.alpha, color="k", ls="--", lw=0.8)
ax[0].set(xlabel=r"$\xi=x/l$", ylabel=r"$\theta$ (fluid), $\phi$ (solid)")
ax[0].legend()
tt = np.linspace(0.005, 3, 400)
ax[1].plot(tt, b.phi_mean(tt), label=r"$\bar\phi$")
ax[1].plot(tt, b.flux(tt), label=r"$J^*$")
ax[1].set(xlabel=r"$\tau$")
ax[1].legend()
fig.tight_layout()
fig.savefig("quickstart.png", dpi=150)
print("wrote quickstart.png")
