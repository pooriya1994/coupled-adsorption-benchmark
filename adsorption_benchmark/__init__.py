"""Exact coupled benchmark for lattice Boltzmann (and other) simulations of adsorption.

Transient diffusion in a fluid layer and an adsorbent layer coupled by first-order
adsorption-desorption kinetics at their interface.

Main objects
------------
Benchmark          exact eigenfunction-series solution (dimensionless)
PhysicalBenchmark  the same solution in physical or lattice units
LaplaceSolution    Laplace-domain form, inverted numerically (independent check)
fv_reference       second-order finite-volume solver, also for Langmuir kinetics
ldf_time           linear-driving-force (resistances-in-series) time scale
"""
__version__ = "1.0.0"

from .exact import Benchmark, PhysicalBenchmark, ldf_time, terms_for  # noqa: E402
from .fv import fv_reference  # noqa: E402

__all__ = ["Benchmark", "PhysicalBenchmark", "LaplaceSolution", "fv_reference", "ldf_time", "terms_for",
           "__version__"]


def __getattr__(name):
    # mpmath is only imported when the Laplace solution is used
    if name == "LaplaceSolution":
        from .laplace import LaplaceSolution
        return LaplaceSolution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
