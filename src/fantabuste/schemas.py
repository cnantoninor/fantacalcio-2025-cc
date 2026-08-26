"""Contratto dati condiviso — CONGELATO dopo la Fase 0.

Ogni modulo (A-F) legge e scrive esclusivamente questi schemi. Se il tuo
lavoro richiede di cambiarli, FERMATI e segnala il problema invece di
modificarli unilateralmente — vedi CLAUDE.md, "Confini di modulo".

Regola d'oro: ogni record porta `fonte` e `is_synthetic`. Nessuna offerta
reale può dipendere da un record con `is_synthetic=True` — vedi
`assert_no_synthetic_dependency` in fondo al file, usata dal Modulo E prima
di emettere un piano d'offerte finale.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Ruolo = Literal["P", "D", "C", "A"]
"""Portiere, Difensore, Centrocampista, Attaccante — nomenclatura della lega
(non GK/DF/MF/FW della pipeline legacy in antoninorau/)."""

Tornata = Literal[1, 2]
"""La fase busta chiusa ha esattamente 2 tornate, fisso — vedi
docs/LEAGUE_CONTEXT.md §2. Non generalizzare a un numero variabile."""

Fase = Literal["buste", "mercato"]
"""'buste' = tornate 1-2 a busta chiusa. 'mercato' = asta a tempo dopo il
3 settembre (il Modulo F la gestisce; nella v1 non è ancora implementata —
vedi docs/OPEN_QUESTIONS.md §0)."""


class SyntheticRecord(BaseModel):
    """Mixin: ogni schema del contratto eredita da questa classe."""

    fonte: str = Field(
        description="Provenienza del dato: nome file/URL/processo che l'ha prodotto."
    )
    is_synthetic: bool = Field(
        default=True,
        description=(
            "True finché il record non è stato sostituito da un dato con fonte "
            "primaria verificabile. Il default è True apposta: un modulo deve "
            "dichiarare esplicitamente is_synthetic=False, mai il contrario."
        ),
    )


class RosterSlots(BaseModel):
    """Composizione di rosa per ruolo. Usato sia per la rosa massima raggiungibile
    nella fase corrente sia per gli slot ancora liberi."""

    P: int = Field(ge=0)
    D: int = Field(ge=0)
    C: int = Field(ge=0)
    A: int = Field(ge=0)

    @property
    def totale(self) -> int:
        return self.P + self.D + self.C + self.A


# ---------------------------------------------------------------------------
# Modulo A — Ingestion
# ---------------------------------------------------------------------------


class Player(SyntheticRecord):
    """Prodotto da A. Consumato da B, C, D."""

    player_id: str
    nome: str
    ruolo: Ruolo
    squadra: str
    quotazione_listone: float = Field(gt=0)
    data_estrazione: datetime


class PlayerStats(SyntheticRecord):
    """Prodotto da A. Consumato da B."""

    player_id: str
    stagione: str = Field(description="Es. '2024/25'.")
    squadra: str = Field(
        description=(
            "Squadra del giocatore IN QUELLA STAGIONE — può differire da "
            "Player.squadra, che è solo lo snapshot alla data di estrazione. "
            "Necessario per l'aggiustamento da cambio squadra nel Modulo B "
            "(baseline.py, FATTORE_CAMBIO_SQUADRA): senza questo campo non è "
            "ricostruibile se un giocatore ha cambiato squadra tra le stagioni "
            "storiche. Vedi la segnalazione dell'Agente B, docs/DESIGN.md."
        )
    )
    presenze: int = Field(ge=0)
    minuti: int = Field(ge=0)
    gol: int = Field(ge=0)
    assist: int = Field(ge=0)
    xG: float = Field(ge=0)
    xA: float = Field(ge=0)
    fantamedia: float
    rigori_battuti: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Modulo B — Valutazione
# ---------------------------------------------------------------------------


class MetodoProiezione(StrEnum):
    BASELINE = "baseline"
    ML = "ml"


class PlayerProjection(SyntheticRecord):
    """Prodotto da B. Consumato da C, D.

    Con 2 sole stagioni storiche il baseline è il modello PRIMARIO, non un
    fallback — vedi docs/DESIGN.md Agente B. `metodo=ml` entra in gioco solo
    se ha battuto il baseline su un margine ampio e dichiarato altrove
    (valuation_report.md), mai per default.
    """

    player_id: str
    punti_attesi: float
    punti_attesi_lo: float
    punti_attesi_hi: float
    prob_titolarita: float = Field(ge=0, le=1)
    vorp: float
    metodo: MetodoProiezione
    confidence_flag: str = Field(
        description="Es. 'alta'|'media'|'bassa' — motivato in valuation_report.md."
    )

    @model_validator(mode="after")
    def _intervallo_coerente(self) -> PlayerProjection:
        if not (self.punti_attesi_lo <= self.punti_attesi <= self.punti_attesi_hi):
            raise ValueError(
                f"{self.player_id}: punti_attesi ({self.punti_attesi}) fuori "
                f"dall'intervallo [{self.punti_attesi_lo}, {self.punti_attesi_hi}]"
            )
        return self


# ---------------------------------------------------------------------------
# Modulo C — Modello delle offerte avversarie
# ---------------------------------------------------------------------------


class PriceDistributionMode(StrEnum):
    EMPIRICAL = "empirical"
    """Richiede K >= config.bidmodel_min_osservazioni (default 30) offerte
    osservate nella fascia. Verificato al 2026-08-25: NON esiste ancora
    storico reale di offerte avversarie per questa lega — vedi
    docs/OPEN_QUESTIONS.md §2. In v1 nessuna fascia gira in empirical."""
    PRIOR = "prior"
    """Nessuno storico sufficiente: distribuzione lognormale centrata sulla
    quotazione del listone, con varianza ampia e dichiarata. Modalità di
    default per la v1."""


class PriceDistribution(SyntheticRecord):
    """Prodotto da C. Consumato da D.

    Il rapporto modellato è sempre `offerta / quotazione_listone` (non il
    valore assoluto), così è confrontabile fra giocatori e fasce — vedi
    docs/DESIGN.md Agente C.
    """

    player_id: str
    fascia: str = Field(description="Chiave ruolo × tier di quotazione, es. 'A_tier1'.")
    mode: PriceDistributionMode
    n_osservazioni: int = Field(ge=0)

    # modalità PRIOR: lognormale sul rapporto offerta/quotazione
    prior_mu: float | None = Field(
        default=None, description="Media in log-spazio del rapporto offerta/quotazione."
    )
    prior_sigma: float | None = Field(
        default=None, description="Deviazione standard in log-spazio. Ampia per default."
    )

    # modalità EMPIRICAL: distribuzione empirica del rapporto, da
    # OpponentBidObservation osservate per la fascia (tutte le offerte, non
    # solo le vincenti — vedi docs/DESIGN.md Agente C).
    empirical_support: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parametri_coerenti_con_mode(self) -> PriceDistribution:
        if self.mode is PriceDistributionMode.PRIOR:
            if self.prior_mu is None or self.prior_sigma is None:
                raise ValueError(
                    f"{self.player_id}: mode=prior richiede prior_mu e prior_sigma"
                )
        elif self.mode is PriceDistributionMode.EMPIRICAL:
            if not self.empirical_support:
                raise ValueError(
                    f"{self.player_id}: mode=empirical richiede empirical_support non vuoto"
                )
        return self

    def p_win(self, b: float, quotazione_listone: float) -> float:
        """P(la mia offerta b vince) = P(offerta massima avversaria < b).

        Approssimazione dichiarata: tratta le offerte avversarie sui diversi
        giocatori come indipendenti — la correlazione via budget condiviso
        va catturata a parte dalla simulazione Monte Carlo di C (stretch
        goal in v1, non ancora implementata).
        """
        if quotazione_listone <= 0:
            raise ValueError("quotazione_listone deve essere positiva")
        ratio = b / quotazione_listone

        if self.mode is PriceDistributionMode.EMPIRICAL:
            support = sorted(self.empirical_support)
            idx = bisect.bisect_left(support, ratio)
            return idx / len(support)

        assert self.prior_mu is not None and self.prior_sigma is not None
        if ratio <= 0:
            return 0.0
        z = (math.log(ratio) - self.prior_mu) / self.prior_sigma
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Modulo D — Ottimizzatore MILP
# ---------------------------------------------------------------------------


class BidPlan(SyntheticRecord):
    """Prodotto da D. Consumato da E."""

    player_id: str
    offerta: float = Field(gt=0)
    p_win_stimata: float = Field(ge=0, le=1)
    vorp: float
    surplus_atteso: float
    tornata: Tornata


# ---------------------------------------------------------------------------
# Modulo E — Stato dell'asta, integrazione, tornate
# ---------------------------------------------------------------------------


class AuctionState(SyntheticRecord):
    """Prodotto e aggiornato da E dopo ogni tornata. Consumato da B, C, D, F.

    `slot_totali` è la rosa massima raggiungibile nella fase corrente:
    2-8-8-6 in fase 'buste' (tornate 1-2), 4-10-10-8 dopo il 7 settembre —
    vedi docs/LEAGUE_CONTEXT.md §1 e §3. Non hardcodare: viene da
    config/league.yaml via il modulo E.
    """

    fase: Fase
    tornata_corrente: Tornata | None = Field(
        default=None, description="None se fase='mercato' (asta a tempo, non a tornate)."
    )
    budget_totale: float = Field(gt=0)
    budget_residuo: float = Field(ge=0)
    slot_totali: RosterSlots
    slot_residui: RosterSlots
    giocatori_assegnati: list[str] = Field(default_factory=list)

    # Visibilità completa delle offerte (se confermata nell'app — vedi
    # docs/OPEN_QUESTIONS.md §2.1) rende questi ESATTI, non stimati.
    budget_residuo_avversari: dict[str, float] = Field(default_factory=dict)
    slot_residui_avversari: dict[str, RosterSlots] = Field(default_factory=dict)


class OpponentBidObservation(SyntheticRecord):
    """Prodotto da E (a partire dai risultati di tornata caricati dall'utente).
    Consumato da C.

    Una riga per OGNI offerta osservata in una tornata, vincente o perdente —
    non solo il prezzo pagato dal vincitore. È l'unico modo per cui il Modulo
    C potrà mai girare in modalità 'empirical' l'anno prossimo: vedi
    docs/OPEN_QUESTIONS.md §2. Catturarle quest'anno è l'investimento singolo
    col ritorno più alto del progetto — non va tagliato per fretta.
    """

    player_id: str
    stagione: str = Field(
        description=(
            "Es. '2026/27'. `tornata` (1|2) da sola non distingue le stagioni: "
            "senza questo campo non è possibile poolare/normalizzare offerte "
            "di anni diversi per fittare le fasce del Modulo C su più stagioni. "
            "Vedi la segnalazione dell'Agente C, docs/DESIGN.md."
        )
    )
    tornata: Tornata
    avversario_id: str
    offerta: float = Field(gt=0)
    vincente: bool


# ---------------------------------------------------------------------------
# Modulo F — Asta di riparazione (fuori scope v1, schema congelato comunque)
# ---------------------------------------------------------------------------


class Rilancio(BaseModel):
    offerente_id: str
    importo: float = Field(gt=0)
    orario: datetime


class RepairLotState(SyntheticRecord):
    """Interno a F. Una collezione di questi = i lotti aperti in parallelo,
    non un singolo stato — vedi docs/DESIGN.md Modulo F."""

    player_id: str
    prezzo_corrente: float = Field(gt=0)
    offerente_corrente: str | None
    orario_chiusura: datetime = Field(
        description="Assoluto, non un countdown relativo — necessario per "
        "confrontare lotti simultanei."
    )
    storico_rilanci: list[Rilancio] = Field(default_factory=list)


class RepairLotResult(SyntheticRecord):
    """Prodotto da F. Consumato da E (per run_log.json).

    Una riga per ogni lotto chiuso, vinto o perso — serve al backtest
    dell'anno prossimo. Meccanismo d'asta diverso (English auction) da
    OpponentBidObservation (FPSBA): il Modulo C non deve mai fittare le sue
    fasce su questi dati, a meno di una decisione esplicita e separata —
    vedi docs/DESIGN.md Modulo E punto 7.
    """

    player_id: str
    prezzo_finale: float = Field(gt=0)
    vinto_da_me: bool
    avversario_vincitore_id: str | None = None
    orario_chiusura: datetime


# ---------------------------------------------------------------------------
# Guardrail sui dati sintetici (usato dal Modulo E)
# ---------------------------------------------------------------------------


def assert_no_synthetic_dependency(
    *records: SyntheticRecord, allow_synthetic: bool = False
) -> None:
    """Rifiuta di procedere se una qualsiasi dipendenza a monte è sintetica.

    Il Modulo E lo chiama prima di emettere un BidPlan finale. Con
    allow_synthetic=True il chiamante deve comunque stampare un banner
    inequivocabile nel report — questa funzione non lo fa da sola.
    """
    if allow_synthetic:
        return
    synthetic = [r for r in records if r.is_synthetic]
    if synthetic:
        kinds = sorted({type(r).__name__ for r in synthetic})
        raise ValueError(
            f"{len(synthetic)} record sintetici tra le dipendenze ({', '.join(kinds)}). "
            "Rifiuto di emettere un piano d'offerte reale da dati sintetici. "
            "Passa allow_synthetic=True solo per test/dry-run, mai per un'offerta vera."
        )
