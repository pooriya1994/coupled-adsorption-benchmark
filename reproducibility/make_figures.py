"""Reproduce Figs. 2-6 of the paper and the verification numbers quoted in Sections 4-6.

Run from the repository root:  python reproducibility/make_figures.py
Output: reproducibility/output/fig_*.pdf and numbers.json  (takes about one minute)
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adsorption_benchmark import Benchmark, fv_reference

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
REF = dict(alpha=0.5, gamma=0.1, K=2.0, Da=2.0)
out = {}

# ---------- reference case: eigenvalues, verification against the FV solver ----------
B = Benchmark(**REF, n_terms=400)
out["lam"] = B.lam[:6].tolist()
taus = [0.01, 0.05, 0.2, 0.5, 1.0, 3.0]
t0 = time.time()
xf, xs, T, TH, PH = fv_reference(**REF, nf=400, ns=400, taus=taus)
out["fv_time"] = time.time() - t0
out["err"] = [[t, np.abs(B.theta(xf, t) - TH[:, k]).max(), np.abs(B.phi(xs, t) - PH[:, k]).max()]
              for k, t in enumerate(T)]

# grid convergence of the FV solver (the benchmark identifies the order of a discretisation)
conv = []
for nn in (25, 50, 100, 200):
    a, b_, T2, TH2, PH2 = fv_reference(**REF, nf=nn, ns=nn, taus=[0.2])
    e = np.sqrt(np.mean(np.r_[(B.theta(a, 0.2) - TH2[:, 0]) ** 2, (B.phi(b_, 0.2) - PH2[:, 0]) ** 2]))
    conv.append([nn, e])
out["conv"] = conv

# Table 2
out["table"] = [[t, float(B.theta(0.25, t)[0]), float(B.theta(0.5, t)[0]), float(B.phi(0.5, t)[0]),
                 float(B.phi(0.75, t)[0]), float(B.phi(1.0, t)[0]), float(B.phi_mean(t)[0]), float(B.flux(t)[0])]
                for t in (0.05, 0.2, 1.0, 3.0)]

# global mass balance: d/dtau [int theta + K int phi] = inflow, mode by mode
Bm = Benchmark(**REF, n_terms=2000)
merr = 0.0
for t in np.linspace(0.02, 3, 50):
    e = Bm.c * np.exp(-Bm.lam**2 * t)
    dM = np.sum(-e * Bm.lam**2 * (Bm.A * Bm.I1 + Bm.K * Bm.B * Bm.I2))
    inflow = -np.sum(e * Bm.A * Bm.lam)
    merr = max(merr, abs(dM - inflow) / abs(inflow))
out["mass_err"] = merr

# Fig. 2: profiles, exact vs FV
fig, ax = plt.subplots(figsize=(6.2, 4))
for k, t in enumerate(T):
    c = f"C{k}"
    ax.plot(xf, B.theta(xf, t), c, lw=1.4, label=fr"$\tau={t:g}$")
    ax.plot(xs, B.phi(xs, t), c, lw=1.4)
    ax.plot(xf[12::25], TH[12::25, k], "o", ms=4, mfc="none", color=c)
    ax.plot(xs[12::25], PH[12::25, k], "s", ms=4, mfc="none", color=c)
ax.axvline(0.5, color="k", ls="--", lw=0.8)
ax.text(0.22, 1.06, "fluid, $\\theta$", ha="center")
ax.text(0.75, 1.06, "solid, $\\phi$", ha="center")
ax.set(xlabel=r"$\xi = x/l$", ylabel=r"$\theta,\ \phi$", ylim=(-0.03, 1.13))
ax.legend(fontsize=8, loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_profiles.pdf"))
plt.close(fig)

# Fig. 3: series truncation error
fig, ax = plt.subplots(figsize=(4.6, 3.3))
Bbig = Benchmark(**REF, n_terms=2000)
xx = np.linspace(0, 1, 201)
for t in (1e-3, 1e-2, 1e-1):
    ex = np.r_[Bbig.theta(xx[xx <= .5], t), Bbig.phi(xx[xx > .5], t)]
    Ns = [5, 10, 20, 40, 80, 160, 320]
    es = []
    for Nt in Ns:
        b = Benchmark(**REF, n_terms=Nt)
        ap = np.r_[b.theta(xx[xx <= .5], t), b.phi(xx[xx > .5], t)]
        es.append(np.abs(ap - ex).max())
    ax.loglog(Ns, np.maximum(es, 1e-16), "o-", ms=3, label=fr"$\tau={t:g}$")
ax.set(xlabel="number of terms", ylabel="max truncation error")
ax.legend(fontsize=8)
ax.grid(alpha=.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_truncation.pdf"))
plt.close(fig)

# Fig. 4: uptake curves, effect of Da and gamma
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
tau = np.linspace(0, 8, 500)
for Da in (0.1, 1, 10, 100):
    b = Benchmark(alpha=0.5, gamma=0.1, K=2, Da=Da, n_terms=300)
    ax[0].plot(tau, b.phi_mean(tau), label=fr"$Da={Da:g}$")
b = Benchmark(alpha=0.5, gamma=0.1, K=2, Da=1e6, n_terms=300)
ax[0].plot(tau, b.phi_mean(tau), "k--", lw=1, label=r"$Da\to\infty$")
ax[0].set(xlabel=r"$\tau$", ylabel=r"$\bar\phi$", title=r"$\gamma=0.1,\ K=2$")
ax[0].legend(fontsize=8)
for g in (0.01, 0.1, 1, 10):
    b = Benchmark(alpha=0.5, gamma=g, K=2, Da=2, n_terms=300)
    ax[1].plot(tau, b.phi_mean(tau), label=fr"$\gamma={g:g}$")
ax[1].set(xlabel=r"$\tau$", title=r"$Da=2,\ K=2$")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_uptake.pdf"))
plt.close(fig)

# Fig. 5: regime map, slowest decay rate lambda_1^2 and the LDF estimate
fig, ax = plt.subplots(figsize=(4.6, 3.3))
Das = np.logspace(-2, 4, 60)
for k, g in enumerate((0.01, 0.1, 1, 10)):
    l1 = [Benchmark(alpha=0.5, gamma=g, K=2, Da=d, n_terms=1).lam[0] ** 2 for d in Das]
    ldf = [1 / (2 * 0.5 * (1 / d + 0.5 + 0.5 / (3 * g * 2))) for d in Das]
    ax.loglog(Das, l1, color=f"C{k}", label=fr"$\gamma={g:g}$")
    ax.loglog(Das, ldf, "--", color=f"C{k}", lw=1)
ax.plot([], [], "k-", label="exact")
ax.plot([], [], "k--", lw=1, label="LDF estimate")
ax.set(xlabel=r"$Da$", ylabel=r"$\lambda_1^2$ (equilibration rate)")
ax.legend(fontsize=8)
ax.grid(alpha=.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_regime.pdf"))
plt.close(fig)

# Fig. 6: Langmuir kinetics (FV) vs the linear exact solution
fig, ax = plt.subplots(figsize=(4.8, 3.4))
taus6 = list(np.linspace(0.05, 8, 40))
ax.plot(tau, B.phi_mean(tau), "k", lw=1.6, label="linear (analytical)")
lang = {}
for s, c in zip((0.01, 0.1, 0.5, 2.0), ("C0", "C1", "C2", "C3")):
    a_, b_, T6, TH6, PH6 = fv_reference(**REF, sigma=s, nf=100, ns=100, taus=taus6, rtol=1e-8, atol=1e-10)
    pm = PH6.mean(axis=0)
    pinf = 1 / (1 + s)
    ax.plot(T6, pm / pinf, "o", ms=3, color=c, label=fr"Langmuir, $\sigma={s:g}$")
    lang[s] = float(np.max(np.abs(pm / pinf - B.phi_mean(T6))))
out["lang"] = lang
ax.set(xlabel=r"$\tau$", ylabel=r"$\bar\phi/\phi_\infty$")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_langmuir.pdf"))
plt.close(fig)

json.dump(out, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
print("figures and numbers.json written to", OUT)
