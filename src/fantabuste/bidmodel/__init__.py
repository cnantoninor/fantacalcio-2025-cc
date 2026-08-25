"""Modulo C — Modello delle offerte avversarie.

Trasforma `OpponentBidObservation[]` (o la loro assenza) in istanze di
`PriceDistribution` (schemas.py, contratto congelato in Fase 0), più la
simulazione Monte Carlo a livello di intera asta e la diagnostica per
giocatore. Vedi docs/DESIGN.md, Agente C, per il perché di ogni scelta.

⚠️ Stato v1 (verificato 2026-08-25, docs/OPEN_QUESTIONS.md §2, §2.1):
nessuno storico reale di offerte avversarie esiste per questa lega. Ogni
fascia gira quindi in modalità `prior`. La modalità `empirical` è
implementata e testata per intero (vedi `fitting.py`), ma non va
presentata come utilizzabile sui dati reali 2026/27.
"""

from __future__ import annotations

from .diagnostics import CurvaPWin, curva_p_win, curve_p_win
from .fasce import N_TIER_DEFAULT, assegna_fasce, nome_fascia
from .fitting import (
    PRIOR_MU_DEFAULT,
    PRIOR_SIGMA_DEFAULT,
    FasciaFitResult,
    costruisci_lot_max_ratios,
    fit_fascia,
    fit_price_distributions,
)
from .montecarlo import RisultatoMonteCarlo, simula_asta
from .normalizzazione import (
    RapportoNormalizzato,
    StagioneStorica,
    normalizza_stagione,
    pool_stagioni,
)
from .report import genera_report

__all__ = [
    "N_TIER_DEFAULT",
    "PRIOR_MU_DEFAULT",
    "PRIOR_SIGMA_DEFAULT",
    "CurvaPWin",
    "FasciaFitResult",
    "RapportoNormalizzato",
    "RisultatoMonteCarlo",
    "StagioneStorica",
    "assegna_fasce",
    "costruisci_lot_max_ratios",
    "curva_p_win",
    "curve_p_win",
    "fit_fascia",
    "fit_price_distributions",
    "genera_report",
    "nome_fascia",
    "normalizza_stagione",
    "pool_stagioni",
    "simula_asta",
]
