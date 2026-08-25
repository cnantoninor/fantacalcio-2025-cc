"""Modulo D — Ottimizzatore MILP. Vedi `fantabuste.optimize.solver` per il
modello completo e le approssimazioni dichiarate."""

from fantabuste.optimize.errors import InfeasibleRosterError
from fantabuste.optimize.solver import (
    DEFAULT_N_ALTERNATIVE,
    DEFAULT_SOGLIA_INSTABILITA,
    DEFAULT_SOLVER_TIME_LIMIT_S,
    solve_bid_plan,
)
from fantabuste.optimize.types import AlternativeRoster, OptimizeResult, SensitivityReport

__all__ = [
    "DEFAULT_N_ALTERNATIVE",
    "DEFAULT_SOGLIA_INSTABILITA",
    "DEFAULT_SOLVER_TIME_LIMIT_S",
    "AlternativeRoster",
    "InfeasibleRosterError",
    "OptimizeResult",
    "SensitivityReport",
    "solve_bid_plan",
]
