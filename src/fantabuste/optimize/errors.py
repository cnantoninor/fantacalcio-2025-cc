"""Eccezioni del Modulo D.

Il MILP deve fallire in modo esplicito e comprensibile quando il problema è
infeasible — mai un crash con lo stack trace grezzo di CBC. Vedi il DoD
dell'Agente D in docs/DESIGN.md: "test che verificano che un caso infeasible
sia gestito con un messaggio chiaro (non un crash)".
"""

from __future__ import annotations


class InfeasibleRosterError(Exception):
    """Sollevata quando non esiste alcuna assegnazione ammissibile.

    Il messaggio elenca sempre la causa più probabile (giocatori insufficienti
    per ruolo, budget minimo insufficiente, vincolo per-squadra troppo
    stringente) così che chi la riceve — un umano o il Modulo E — capisca
    cosa correggere senza dover ispezionare il modello PuLP.
    """
