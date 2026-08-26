"""Normalizzazione dell'inflazione tra stagioni.

`OpponentBidObservation` porta ora un campo `stagione` (aggiunto al
contratto congelato dopo la segnalazione dell'Agente C in Fase 1 — vedi
docs/DESIGN.md). `StagioneStorica.label` resta comunque il raggruppamento
esplicito passato dal chiamante, per due motivi che il campo da solo non
risolve: (1) il contesto di normalizzazione (`quotazioni`, `budget_totale`,
`n_partecipanti` di QUELLA stagione) non è nello schema di
`OpponentBidObservation` e va comunque fornito a parte — il listone cambia
stagione per stagione e `Player.quotazione_listone` porta solo il valore
corrente; (2) `__post_init__` verifica ora che ogni osservazione in
`osservazioni` abbia `stagione == label`, così un raggruppamento con dati di
stagioni mescolate per errore fallisce esplicitamente invece di produrre una
normalizzazione silenziosamente sbagliata.

Metodo di normalizzazione — **approssimazione dichiarata**, non un modello
econometrico rigoroso: si assume che il livello dei prezzi (il rapporto
`offerta / quotazione_listone`) scali linearmente con il **budget
pro-capite** della stagione (`budget_totale / n_partecipanti`). Una
stagione con budget pro-capite più alto di quella di riferimento tende a
produrre rapporti più alti a parità di giocatore; si riporta ogni rapporto
alla scala della stagione di riferimento dividendo per il rapporto fra i
budget pro-capite delle due stagioni. Non cattura altre fonti di
inflazione (es. cambio di popolarità di un ruolo), dichiarato come limite.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantabuste.schemas import OpponentBidObservation


@dataclass(frozen=True)
class StagioneStorica:
    """Le osservazioni di UNA stagione passata + il contesto (quotazioni,
    budget, partecipanti) necessario per normalizzarle prima di poolarle
    con altre stagioni. Vedi il docstring di modulo per il perché questa
    classe esiste al posto di un campo `stagione` nello schema congelato.
    """

    label: str
    osservazioni: list[OpponentBidObservation]
    quotazioni: dict[str, float]
    """player_id -> quotazione_listone valida in QUESTA stagione (il
    listone cambia stagione per stagione, e Player.quotazione_listone
    porta solo il valore corrente — altro motivo per cui questo contesto
    va passato esplicitamente dal chiamante)."""
    budget_totale: float
    n_partecipanti: int

    def __post_init__(self) -> None:
        if self.budget_totale <= 0:
            raise ValueError(f"{self.label}: budget_totale deve essere positivo")
        if self.n_partecipanti <= 0:
            raise ValueError(f"{self.label}: n_partecipanti deve essere positivo")
        for oss in self.osservazioni:
            if oss.stagione != self.label:
                raise ValueError(
                    f"StagioneStorica(label={self.label!r}): osservazione "
                    f"player_id={oss.player_id!r} ha stagione={oss.stagione!r}, "
                    "diversa da label. Ogni osservazione deve appartenere alla "
                    "stagione dichiarata dal raggruppamento in cui è inserita."
                )

    @property
    def budget_procapite(self) -> float:
        return self.budget_totale / self.n_partecipanti


@dataclass(frozen=True)
class RapportoNormalizzato:
    """Un'osservazione ridotta al rapporto `offerta / quotazione_listone`,
    dopo la normalizzazione dell'inflazione tra stagioni — pronta per
    essere raggruppata per lotto e per fascia (vedi `fitting.py`)."""

    player_id: str
    tornata: int
    stagione: str
    rapporto_grezzo: float
    rapporto_normalizzato: float
    fonte: str
    is_synthetic: bool


def normalizza_stagione(
    stagione: StagioneStorica,
    budget_procapite_riferimento: float,
) -> list[RapportoNormalizzato]:
    """Converte le osservazioni di UNA stagione in rapporti normalizzati
    alla scala pro-capite della stagione di riferimento (tipicamente la
    stagione corrente, da `LeagueConfig`).

    `fattore = budget_procapite_riferimento / stagione.budget_procapite`
    `rapporto_normalizzato = rapporto_grezzo * fattore`

    Se un'osservazione riguarda un `player_id` assente da
    `stagione.quotazioni`, viene scartata silenziosamente rispetto al
    calcolo del rapporto (non solleva un'eccezione): può succedere
    fisiologicamente quando un giocatore non era più a listone in una
    stagione passata (retrocesso, svincolato, ecc.).
    """
    if budget_procapite_riferimento <= 0:
        raise ValueError("budget_procapite_riferimento deve essere positivo")

    fattore = budget_procapite_riferimento / stagione.budget_procapite
    risultato: list[RapportoNormalizzato] = []
    for oss in stagione.osservazioni:
        quot = stagione.quotazioni.get(oss.player_id)
        if quot is None or quot <= 0:
            continue
        grezzo = oss.offerta / quot
        risultato.append(
            RapportoNormalizzato(
                player_id=oss.player_id,
                tornata=oss.tornata,
                stagione=stagione.label,
                rapporto_grezzo=grezzo,
                rapporto_normalizzato=grezzo * fattore,
                fonte=oss.fonte,
                is_synthetic=oss.is_synthetic,
            )
        )
    return risultato


def pool_stagioni(
    stagioni: list[StagioneStorica],
    budget_procapite_riferimento: float,
) -> list[RapportoNormalizzato]:
    """Normalizza e poola più stagioni in un'unica lista di rapporti,
    pronta per il fitting empirico per fascia (`fitting.fit_price_distributions`).
    """
    pool: list[RapportoNormalizzato] = []
    for s in stagioni:
        pool.extend(normalizza_stagione(s, budget_procapite_riferimento))
    return pool
