"""Tipi di output del Modulo D non presenti nel contratto congelato.

`BidPlan` (in `fantabuste.schemas`) è l'unico schema che il resto della
pipeline (Modulo E) consuma — è il contratto condiviso e resta invariato.
`AlternativeRoster`, `SensitivityReport` e `OptimizeResult` sono strutture
di supporto **interne al Modulo D**: incapsulano le top-5 rose alternative e
l'analisi di sensibilità richieste da docs/DESIGN.md (Agente D, punti 4-5),
che non hanno un equivalente nello schema congelato e quindi non lo
estendono — restano locali a `fantabuste.optimize`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantabuste.schemas import BidPlan


@dataclass(frozen=True)
class AlternativeRoster:
    """Una delle top-N rose ammissibili trovate dal solver, ordinate per
    utilità attesa decrescente. `rank=1` è l'ottimo."""

    rank: int
    bid_plan: list[BidPlan]
    utilita_attesa: float
    delta_vs_ottimo: float
    """utilita_attesa - utilita_attesa(rank=1). Sempre <= 0, 0 solo per rank=1."""


@dataclass(frozen=True)
class SensitivityReport:
    """Quanto cambia la rosa ottima se le proiezioni (VORP) si spostano
    sistematicamente di +-1 deviazione standard.

    **Approssimazione dichiarata sulla sigma**: `PlayerProjection` non porta
    una deviazione standard esplicita, solo un intervallo
    [punti_attesi_lo, punti_attesi_hi]. Trattiamo quell'intervallo come
    un inviluppo a +-1 sigma, cioè `sigma_i = (hi - lo) / 2`, e applichiamo
    lo stesso spostamento assoluto al VORP del giocatore (si assume che il
    VORP erediti l'incertezza assoluta dei punti attesi, essendone una
    trasformazione lineare rispetto a un replacement level fisso). Se
    l'ipotesi sull'intervallo (es. un intervallo al 95% invece che +-1
    sigma) fosse sbagliata, sigma_i sarebbe sovrastimata di un fattore
    circa 2 — dichiarato qui, non nascosto.

    Lo spostamento è **sistematico** (tutti i giocatori +sigma, poi tutti
    -sigma), non un campionamento indipendente per giocatore: è la verifica
    più economica ("cosa succede se le mie proiezioni sono uniformemente
    troppo ottimiste/pessimiste") ed è quella esplicitamente richiesta da
    docs/DESIGN.md. Non cattura instabilità dovuta a shock idiosincratici
    su un singolo giocatore.
    """

    utilita_ottimo: float
    utilita_shift_positivo: float
    utilita_shift_negativo: float
    roster_ottimo_ids: frozenset[str]
    roster_shift_positivo_ids: frozenset[str]
    roster_shift_negativo_ids: frozenset[str]
    jaccard_positivo: float
    """Frazione di giocatori in comune tra rosa ottima e rosa a +1 sigma."""
    jaccard_negativo: float
    """Frazione di giocatori in comune tra rosa ottima e rosa a -1 sigma."""
    soglia_instabilita: float
    instabile: bool
    """True se jaccard_positivo o jaccard_negativo < soglia_instabilita:
    la rosa ottima cambia sostanzialmente per uno spostamento plausibile
    delle proiezioni. Il chiamante deve mostrarlo esplicitamente, non
    seppellirlo in un report — vedi DESIGN.md Agente D punto 5."""
    note: str = ""


@dataclass(frozen=True)
class OptimizeResult:
    """Output completo di `solve_bid_plan`."""

    bid_plan: list[BidPlan]
    """La rosa ottima (== alternative[0].bid_plan), nel formato del
    contratto condiviso — questo è ciò che il Modulo E consuma."""
    utilita_attesa: float
    alternative: list[AlternativeRoster]
    """Fino a n_alternative rose, rank 1..N, ordinate per utilità
    decrescente. Può contenere meno di n_alternative elementi se non
    esistono così tante soluzioni ammissibili distinte."""
    sensitivity: SensitivityReport | None
    giocatori_esclusi: list[str] = field(default_factory=list)
    """player_id esclusi dal problema perché privi di PlayerProjection e/o
    PriceDistribution in input — non un errore, ma va mostrato: un
    giocatore escluso qui non può mai comparire in bid_plan."""
    warnings: list[str] = field(default_factory=list)
