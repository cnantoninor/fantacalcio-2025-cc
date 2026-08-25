"""Fitting delle fasce di prezzo — il cuore del Modulo C.

Trasforma osservazioni di offerte avversarie normalizzate (o la loro
assenza) in istanze di `PriceDistribution`, con la scelta fra modalità
`empirical` e `prior` governata ESCLUSIVAMENTE da
`config.bidmodel_min_osservazioni` (mai un numero hardcoded nel codice —
vedi CLAUDE.md, "Standard").

FATTO OPERATIVO PER LA v1 (verificato 2026-08-25, docs/OPEN_QUESTIONS.md
§2, §2.1): nessuno storico reale di offerte avversarie esiste per questa
lega — non solo "scarso": zero. Ogni fascia v1 gira quindi in `prior`, per
la stessa regola generale (n_osservazioni=0 < K per qualsiasi K positivo),
non per un ramo speciale. Il percorso `empirical` qui sotto è scritto e
testato per intero con osservazioni sintetiche costruite inline — serve
dall'anno prossimo, quando `OpponentBidObservation` verrà raccolto per la
prima volta durante l'asta vera (vedi CLAUDE.md, corollario operativo). Non
va presentato come utilizzabile sui dati 2026/27 reali: non lo è.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantabuste.config import LeagueConfig
from fantabuste.schemas import Player, PriceDistribution, PriceDistributionMode

from .fasce import N_TIER_DEFAULT, assegna_fasce
from .normalizzazione import RapportoNormalizzato

PRIOR_MU_DEFAULT = 0.0
"""log(1): rapporto mediano offerta/quotazione = 1, cioè il listone come
migliore stima puntuale in assenza di storico — vedi docs/DESIGN.md,
Agente C, punto 2 ("distribuzione centrata sulla quotazione del listone").
"""

PRIOR_SIGMA_DEFAULT = 0.5
"""Deviazione standard in log-spazio del prior. Con `prior_mu=0` implica un
intervallo al 90% di `[exp(-1.645*0.5), exp(1.645*0.5)]` ≈ `[0.44, 2.28]`
volte la quotazione — ampio e dichiarato per costruzione, come richiesto
per la modalità `prior` (docs/DESIGN.md, Agente C, punto 2): non nasconde
l'incertezza sotto un intervallo stretto solo perché mancano i dati.

Nota (docs/OPEN_QUESTIONS.md §2.2): il "Prezzo Medio Aste" di FantaLab, se
mai reso disponibile con fonte ed export verificati, potrebbe calibrare
`prior_mu` in modo più informativo di ratio=1 (vedi il parametro
`prior_mu_per_fascia` di `fit_price_distributions`) — non ancora fatto in
v1 perché nessun dato del genere è stato raccolto con fonte verificabile:
vedi CLAUDE.md, nessun numero da documento di ricerca entra nel modello
senza fonte primaria.
"""


@dataclass(frozen=True)
class FasciaFitResult:
    """Esito del fitting di UNA fascia, prima di essere replicato su ogni
    `player_id` che vi appartiene.

    `degradata_a_prior=True` segnala che la fascia AVEVA osservazioni ma
    sotto soglia K (distinto dal caso "zero osservazioni", la realtà della
    v1) — questa distinzione è quella che rende il report diagnostico
    utile anche quando iniziano ad arrivare le prime osservazioni reali
    l'anno prossimo, invece di limitarsi a dire "prior sì/no".
    """

    fascia: str
    mode: PriceDistributionMode
    n_osservazioni: int
    empirical_support: list[float] = field(default_factory=list)
    prior_mu: float | None = None
    prior_sigma: float | None = None
    degradata_a_prior: bool = False


def fit_fascia(
    fascia: str,
    lot_max_ratios: list[float],
    min_osservazioni: int,
    prior_mu: float = PRIOR_MU_DEFAULT,
    prior_sigma: float = PRIOR_SIGMA_DEFAULT,
) -> FasciaFitResult:
    """Fitta UNA fascia dati i suoi rapporti massimi per lotto.

    `lot_max_ratios` = un valore per ogni lotto (player, tornata,
    stagione) osservato in quella fascia — vedi `costruisci_lot_max_ratios`
    per il perché è il massimo per lotto e non ogni singola offerta.

    Rifiuta la modalità `empirical` sotto soglia `K = min_osservazioni`:
    degrada automaticamente a `prior`, mai un errore silenzioso, mai un
    `empirical_support` corto spacciato per distribuzione affidabile —
    vedi DoD dell'Agente C.
    """
    if min_osservazioni <= 0:
        raise ValueError("min_osservazioni deve essere positivo")

    n = len(lot_max_ratios)
    if n >= min_osservazioni:
        return FasciaFitResult(
            fascia=fascia,
            mode=PriceDistributionMode.EMPIRICAL,
            n_osservazioni=n,
            empirical_support=sorted(lot_max_ratios),
        )
    return FasciaFitResult(
        fascia=fascia,
        mode=PriceDistributionMode.PRIOR,
        n_osservazioni=n,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        degradata_a_prior=n > 0,
    )


def costruisci_lot_max_ratios(
    rapporti: list[RapportoNormalizzato],
) -> dict[tuple[str, int, str], float]:
    """Raggruppa i rapporti normalizzati per lotto `(player_id, tornata,
    stagione)` e prende il massimo: è il punto empirico di "offerta
    massima altrui" per quel lotto — vedi docs/DESIGN.md, Agente C, punto 1
    ("non serve elevare alla n [...] lo hai già osservato per davvero,
    tornata per tornata").

    Perché il massimo e non ogni singola offerta: `PriceDistribution.p_win`
    in modalità `empirical` (schemas.py, già scritto e testato in Fase 0)
    è un'ECDF su `empirical_support`. Perché quell'ECDF corrisponda
    davvero a `P(offerta massima altrui < b / quotazione)`,
    `empirical_support` deve contenere UN valore per lotto — il massimo
    fra tutte le offerte avversarie osservate su quel lotto — non ogni
    singola offerta individuale: altrimenti si stimerebbe
    `P(un singolo avversario a caso offre < ratio)`, un oggetto diverso
    (e sistematicamente più permissivo) da `P(vinco io con b)`.
    """
    per_lotto: dict[tuple[str, int, str], float] = {}
    for r in rapporti:
        chiave = (r.player_id, r.tornata, r.stagione)
        attuale = per_lotto.get(chiave)
        if attuale is None or r.rapporto_normalizzato > attuale:
            per_lotto[chiave] = r.rapporto_normalizzato
    return per_lotto


def fit_price_distributions(
    players: list[Player],
    config: LeagueConfig,
    fonte: str,
    rapporti_storici: list[RapportoNormalizzato] | None = None,
    n_tier: int = N_TIER_DEFAULT,
    prior_mu_per_fascia: dict[str, float] | None = None,
    prior_sigma: float = PRIOR_SIGMA_DEFAULT,
) -> list[PriceDistribution]:
    """Produce una `PriceDistribution` per ogni giocatore in `players`.

    Con `rapporti_storici=None` o vuoto (il caso reale della v1 — vedi
    docs/OPEN_QUESTIONS.md §2): ogni fascia degrada a `prior` perché ha
    zero osservazioni, sotto qualunque soglia K positiva. Non è un ramo
    speciale nel codice: è `fit_fascia` chiamata con `n=0`.

    `prior_mu_per_fascia`: calibrazione opzionale del centro del prior per
    fascia (in log-spazio), invece del default neutro `ratio=1`
    (`PRIOR_MU_DEFAULT`) — predisposto per un futuro dato con fonte
    verificata (es. FantaLab PMA, docs/OPEN_QUESTIONS.md §2.2), non
    popolato con nessun valore in v1 perché nessuna fonte del genere è
    stata verificata: passare questo parametro con numeri presi da un
    documento di ricerca violerebbe CLAUDE.md.

    Soglia K = `config.bidmodel_min_osservazioni`, mai hardcodata qui.
    """
    fasce_per_player = assegna_fasce(players, n_tier=n_tier)
    lot_max = costruisci_lot_max_ratios(rapporti_storici or [])
    player_by_id = {p.player_id: p for p in players}

    ratios_per_fascia: dict[str, list[float]] = {}
    for (player_id, _tornata, _stagione), ratio in lot_max.items():
        if player_id not in player_by_id:
            continue  # lotto storico di un giocatore non più a listone
        fascia = fasce_per_player.get(player_id)
        if fascia is None:
            continue
        ratios_per_fascia.setdefault(fascia, []).append(ratio)

    is_synthetic_storico_per_fascia: dict[str, bool] = {}
    for r in rapporti_storici or []:
        fascia = fasce_per_player.get(r.player_id)
        if fascia is None:
            continue
        is_synthetic_storico_per_fascia[fascia] = (
            is_synthetic_storico_per_fascia.get(fascia, False) or r.is_synthetic
        )

    prior_mu_per_fascia = prior_mu_per_fascia or {}
    fit_per_fascia: dict[str, FasciaFitResult] = {}
    for fascia in sorted(set(fasce_per_player.values())):
        fit_per_fascia[fascia] = fit_fascia(
            fascia,
            ratios_per_fascia.get(fascia, []),
            config.bidmodel_min_osservazioni,
            prior_mu=prior_mu_per_fascia.get(fascia, PRIOR_MU_DEFAULT),
            prior_sigma=prior_sigma,
        )

    distribuzioni: list[PriceDistribution] = []
    for p in players:
        fascia = fasce_per_player[p.player_id]
        fit = fit_per_fascia[fascia]
        # is_synthetic non ha mai un default silenzioso: True se il
        # giocatore stesso è sintetico, o se la fascia è stata fittata (in
        # tutto o in parte) su osservazioni storiche sintetiche.
        is_synth = p.is_synthetic or is_synthetic_storico_per_fascia.get(fascia, False)
        distribuzioni.append(
            PriceDistribution(
                player_id=p.player_id,
                fascia=fascia,
                mode=fit.mode,
                n_osservazioni=fit.n_osservazioni,
                prior_mu=fit.prior_mu,
                prior_sigma=fit.prior_sigma,
                empirical_support=fit.empirical_support,
                fonte=fonte,
                is_synthetic=is_synth,
            )
        )
    return distribuzioni
