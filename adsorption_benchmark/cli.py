"""Command-line interface: reference values and CSV files for any parameter set.

Examples
--------
adsorption-benchmark eigen -n 10
adsorption-benchmark table --tau 0.05 0.2 1 3
adsorption-benchmark profile --tau 0.05 0.2 1 3 --nodes 101 --out profiles.csv
adsorption-benchmark profile --tau 1 --xi-file my_nodes.txt --out ref_at_my_nodes.csv
adsorption-benchmark uptake --tau-max 3 --n 301 --out uptake.csv
"""
import argparse
import sys

import numpy as np

from . import __version__
from .exact import Benchmark, terms_for


def _add_groups(p):
    g = p.add_argument_group("benchmark parameters (defaults: reference case of the paper)")
    g.add_argument("--alpha", type=float, default=0.5, help="fluid fraction a/l (default 0.5)")
    g.add_argument("--gamma", type=float, default=0.1, help="diffusivity ratio Ds/Df (default 0.1)")
    g.add_argument("--K", type=float, default=2.0, help="Henry constant ka/kd (default 2)")
    g.add_argument("--Da", type=float, default=2.0, help="Damkoehler number ka*l/Df (default 2)")
    g.add_argument("--theta0", type=float, default=0.0, help="initial C/C_in (default 0)")
    g.add_argument("--phi0", type=float, default=0.0, help="initial N/(K C_in) (default 0)")
    g.add_argument("--terms", type=int, default=0, help="series terms (0 = chosen automatically)")


def _bench(args, tau_min):
    n = args.terms if args.terms > 0 else max(200, terms_for(args.alpha, args.gamma, tau_min))
    return Benchmark(alpha=args.alpha, gamma=args.gamma, K=args.K, Da=args.Da,
                     theta0=args.theta0, phi0=args.phi0, n_terms=n)


def _header(args, extra=""):
    return (f"# coupled adsorption benchmark v{__version__}: alpha={args.alpha:g} gamma={args.gamma:g} "
            f"K={args.K:g} Da={args.Da:g} theta0={args.theta0:g} phi0={args.phi0:g}{extra}\n")


def _write(text, out):
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)


def _positive_taus(taus):
    taus = [float(t) for t in taus]
    if min(taus) <= 0:
        raise SystemExit("all tau values must be positive (the series does not converge at tau = 0)")
    return taus


def cmd_eigen(args):
    b = Benchmark(alpha=args.alpha, gamma=args.gamma, K=args.K, Da=args.Da, n_terms=args.n)
    lines = [_header(args), "n,lambda_n,lambda_n^2\n"]
    lines += [f"{k + 1},{v:.15g},{v * v:.15g}\n" for k, v in enumerate(b.lam)]
    _write("".join(lines), args.out)


def cmd_table(args):
    taus = _positive_taus(args.tau)
    b = _bench(args, min(taus))
    xf = [x for x in args.xi if x <= args.alpha]
    xs = [x for x in args.xi if x >= args.alpha]
    cols = [f"theta({x:g})" for x in xf] + [f"phi({x:g})" for x in xs] + ["phi_bar", "J*"]
    lines = [_header(args), "tau," + ",".join(cols) + "\n"]
    for t in taus:
        vals = list(b.theta(xf, t)) if xf else []
        vals += list(b.phi(xs, t)) if xs else []
        vals += [b.phi_mean(t)[0], b.flux(t)[0]]
        lines.append(f"{t:g}," + ",".join(f"{v:.12g}" for v in vals) + "\n")
    _write("".join(lines), args.out)


def cmd_profile(args):
    taus = _positive_taus(args.tau)
    b = _bench(args, min(taus))
    if args.xi_file:
        xi = np.loadtxt(args.xi_file, ndmin=1, comments="#", usecols=0)
        if xi.min() < 0 or xi.max() > 1:
            raise SystemExit("xi values must lie in [0, 1] (xi = x/l)")
        xf, xs = xi[xi <= args.alpha], xi[xi >= args.alpha]
    else:
        xf = np.linspace(0.0, args.alpha, args.nodes)
        xs = np.linspace(args.alpha, 1.0, args.nodes)
    lines = [_header(args), "tau,xi,phase,value\n"]
    for t in taus:
        for x, v in zip(xf, b.theta(xf, t) if len(xf) else []):
            lines.append(f"{t:g},{x:.12g},fluid,{v:.12g}\n")
        for x, v in zip(xs, b.phi(xs, t) if len(xs) else []):
            lines.append(f"{t:g},{x:.12g},solid,{v:.12g}\n")
    _write("".join(lines), args.out)


def cmd_uptake(args):
    if args.tau_min <= 0 or args.tau_max <= args.tau_min:
        raise SystemExit("need 0 < tau-min < tau-max")
    taus = (np.geomspace(args.tau_min, args.tau_max, args.n) if args.log
            else np.linspace(args.tau_min, args.tau_max, args.n))
    b = _bench(args, taus.min())
    pb, js = b.phi_mean(taus), b.flux(taus)
    lines = [_header(args), "tau,phi_bar,J_star,jump\n"]
    lines += [f"{t:.12g},{p:.12g},{j:.12g},{j / args.Da:.12g}\n" for t, p, j in zip(taus, pb, js)]
    _write("".join(lines), args.out)


def build_parser():
    ap = argparse.ArgumentParser(prog="adsorption-benchmark",
                                 description="Exact coupled benchmark for adsorption codes: "
                                             "reference values and CSV files.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("eigen", help="eigenvalues lambda_n")
    _add_groups(p)
    p.add_argument("-n", type=int, default=10, help="number of eigenvalues (default 10)")
    p.add_argument("--out", help="CSV file (default: print)")
    p.set_defaults(func=cmd_eigen)

    p = sub.add_parser("table", help="point values, mean loading and flux at given times")
    _add_groups(p)
    p.add_argument("--tau", type=float, nargs="+", default=[0.05, 0.2, 1.0, 3.0])
    p.add_argument("--xi", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="positions; xi <= alpha gives theta, xi >= alpha gives phi")
    p.add_argument("--out", help="CSV file (default: print)")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("profile", help="theta and phi profiles at given times")
    _add_groups(p)
    p.add_argument("--tau", type=float, nargs="+", required=True)
    p.add_argument("--nodes", type=int, default=101, help="uniform points per layer (default 101)")
    p.add_argument("--xi-file", help="file with one xi = x/l per line (e.g. your node positions)")
    p.add_argument("--out", help="CSV file (default: print)")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("uptake", help="phi_bar(tau), J*(tau) and the interface jump")
    _add_groups(p)
    p.add_argument("--tau-min", type=float, default=0.01)
    p.add_argument("--tau-max", type=float, default=3.0)
    p.add_argument("--n", type=int, default=300, help="number of times (default 300)")
    p.add_argument("--log", action="store_true", help="logarithmic spacing in tau")
    p.add_argument("--out", help="CSV file (default: print)")
    p.set_defaults(func=cmd_uptake)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
