"""Modulo D — Ottimizzatore MILP per l'asta a buste chiuse (fase 1).

Interfaccia pubblica: `solve_bid_plan`. Prende `Player[]` + `PlayerProjection[]`
+ `PriceDistribution[]` + `LeagueConfig` e restituisce un `OptimizeResult`
(la rosa ottima come `BidPlan[]`, le top-N rose alternative e un'analisi di
sensibilità). Non dipende da alcuna CLI: è pensata per essere chiamata sia
dal Modulo E in v1, sia dal Modulo F in una versione futura che riusa questo
stesso codice (non solo lo schema `BidPlan`) per l'asta di riparazione — vedi
docs/DESIGN.md, "Modulo F".

## Modello

Variabili binarie `x[i,b]` = "offro `b` crediti per il giocatore `i`", per
`b` in una griglia discretizzata di livelli candidati (vedi `grid.py` per il
trade-off di discretizzazione).

    max  sum_i sum_b (VORP_i * p_win_i(b)) * x[i,b]

Vincoli:
- al più un'offerta per giocatore: `sum_b x[i,b] <= 1`;
- budget: `sum_i sum_b b * x[i,b] <= budget` (default `budget_totale_fase1`);
- composizione di rosa ESATTA per ruolo, da `RosterSlots` (default
  `config.rosa_fase1`): `sum_{i in ruolo r} sum_b x[i,b] == richiesti_r`;
- `sum_{i in squadra s} sum_b x[i,b] <= max_giocatori_per_squadra_serie_a`,
  solo se il parametro non è `None` in `LeagueConfig` (in
  `league.example.yaml` è `null`, cioè nessun vincolo — vedi
  docs/OPEN_QUESTIONS.md §4).

`p_win_i(b)` non viene reimplementata qui: è `PriceDistribution.p_win`, già
fornita da `schemas.py` (Modulo C).

## ⚠️ Vincolo di onestà — approssimazione dichiarata, non ottimo globale

`p_win` è trattata come **indipendente tra giocatori**: il modello massimizza
la somma di prodotti `VORP_i * p_win_i(b)` come se vincere la busta sul
giocatore A non avesse alcuna relazione con la probabilità di vincere la
busta sul giocatore B. Nella realtà è falso — il budget degli avversari è
condiviso tra tutti i lotti, quindi le `p_win` sono correlate (spendere molto
su A riduce la probabilità che lo stesso avversario offra alto anche su B).
Questo MILP è quindi una **euristica ad approssimazione dichiarata**, non un
ottimo globale sull'intera asta — esattamente come previsto in
docs/DESIGN.md (A2) e ribadito nel work order dell'Agente D. La simulazione
Monte Carlo a livello di intera asta che stimerebbe questa correlazione è
compito del Modulo C (stretch goal, non ancora implementata in v1): quando
sarà disponibile, potrà eventualmente correggere le `p_win` qui usate, ma
questo modulo non la anticipa né la simula autonomamente.

## ⚠️ Scope v1 — vincoli speciali NON modellati (decisione presa, non un bug)

Per decisione esplicita del proprietario del progetto (docs/OPEN_QUESTIONS.md
§0, 2026-08-25), questo solver **non** modella:

- il regolamento portieri Top 8 (esclusività fra Top 8 di squadre diverse,
  diritto sul secondo portiere della stessa squadra, garanzia del titolare
  d'ufficio — LEAGUE_CONTEXT §4-5): sono vincoli condizionali/disgiuntivi,
  costosi da linearizzare, e la "garanzia del titolare" è un'opzione di
  fallback gratuita che altera il vero valore atteso di un terzo portiere
  economico in modi che questo modello non cattura;
- i modificatori difesa/centrocampo/attacco (LEAGUE_CONTEXT §7): rendono il
  valore di un giocatore dipendente dal resto della rosa (es. il modificatore
  difesa usa portiere + 3 migliori difensori), il che rompe la separabilità
  lineare dell'obiettivo qui sopra;
- la ricarica di budget/slot del 7 settembre (+50 crediti, +2 slot per
  ruolo): questo solver opera solo sulla rosa di fase 1
  (`config.rosa_fase1`, `config.budget_totale_fase1`), non sull'intera
  stagione di mercato.

Questi sono gap noti e accettati per v1, non dimenticanze: vanno segnalati a
chi consuma `OptimizeResult` (tipicamente come avvertenza testuale nel report
del Modulo E), non lavorati attorno silenziosamente qui.

## `tornata`

Questo solver produce **un singolo piano** per l'intera `rosa_fase1` in un
colpo solo; non gestisce la logica stateful delle 2 tornate (budget/slot
residui che cambiano tra tornata 1 e 2) — quella è responsabilità del Modulo
E, che in v1 è "ridotto" (docs/OPEN_QUESTIONS.md §0) e non ancora
implementato. Il parametro `tornata` esiste solo per etichettare i
`BidPlan` prodotti nel campo richiesto dallo schema congelato; di default è
`1`. Un chiamante che gestisce le tornate (Modulo E futuro) può richiamare
`solve_bid_plan` una seconda volta con `tornata=2`, `budget` e
`roster_slots` aggiornati per riflettere quanto è già stato assegnato.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace

import pulp

from fantabuste.config import LeagueConfig
from fantabuste.optimize.errors import InfeasibleRosterError
from fantabuste.optimize.grid import MIN_OFFERTA, griglia_offerte
from fantabuste.optimize.types import AlternativeRoster, OptimizeResult, SensitivityReport
from fantabuste.schemas import (
    BidPlan,
    Player,
    PlayerProjection,
    PriceDistribution,
    RosterSlots,
    Tornata,
)

logger = logging.getLogger(__name__)

DEFAULT_N_ALTERNATIVE = 5
DEFAULT_SOLVER_TIME_LIMIT_S = 20.0
DEFAULT_SOGLIA_INSTABILITA = 0.7
"""Sotto questa frazione di sovrapposizione (Jaccard) fra la rosa ottima e
quella ricalcolata a +-1 sigma, la rosa ottima viene dichiarata instabile.
Parametro dell'algoritmo di reporting (non di regolamento), documentato qui;
il chiamante può cambiarlo passando `soglia_instabilita` a `solve_bid_plan`."""

_RUOLI: tuple[str, ...] = ("P", "D", "C", "A")

_Joined = dict[str, tuple[Player, PlayerProjection, PriceDistribution]]
_Candidato = tuple[int, float, float, pulp.LpVariable]
"""(offerta, p_win, coefficiente_obiettivo, variabile)."""


def _join_input(
    players: Sequence[Player],
    projections: Sequence[PlayerProjection],
    price_distributions: Sequence[PriceDistribution],
) -> tuple[_Joined, list[str]]:
    """Unisce i tre input per player_id. Un giocatore privo di proiezione e/o
    distribuzione di prezzo viene escluso dal problema (non può mai comparire
    in un'offerta) invece di far fallire l'intero solve — l'esclusione viene
    riportata in `OptimizeResult.giocatori_esclusi`."""
    proj_by_id = {p.player_id: p for p in projections}
    dist_by_id = {d.player_id: d for d in price_distributions}

    joined: _Joined = {}
    esclusi: list[str] = []
    for player in players:
        proj = proj_by_id.get(player.player_id)
        dist = dist_by_id.get(player.player_id)
        if proj is None or dist is None:
            esclusi.append(player.player_id)
            continue
        joined[player.player_id] = (player, proj, dist)
    return joined, esclusi


def _costruisci_problema(
    joined: _Joined,
    config: LeagueConfig,
    roster_slots: RosterSlots,
    budget: float,
    vorp_override: dict[str, float] | None = None,
) -> tuple[pulp.LpProblem, dict[str, list[_Candidato]]]:
    """Costruisce il modello PuLP. `vorp_override` permette all'analisi di
    sensibilità di risolvere lo stesso problema con VORP perturbati senza
    duplicare la logica di costruzione dei vincoli."""
    prob = pulp.LpProblem("fantabuste_bid_plan", pulp.LpMaximize)

    candidati_per_player: dict[str, list[_Candidato]] = {}
    vars_per_ruolo: dict[str, list[pulp.LpVariable]] = {r: [] for r in _RUOLI}
    vars_per_squadra: dict[str, list[pulp.LpVariable]] = {}
    tutte_le_var_con_costo: list[tuple[int, pulp.LpVariable]] = []
    obiettivo_terms: list[pulp.LpAffineExpression] = []

    for player_id, (player, proj, dist) in joined.items():
        vorp = vorp_override[player_id] if vorp_override is not None else proj.vorp
        livelli = griglia_offerte(player.quotazione_listone, budget)

        cand_list: list[_Candidato] = []
        for b in livelli:
            var = pulp.LpVariable(f"x_{player_id}_{b}", cat="Binary")
            p_win = dist.p_win(b, player.quotazione_listone)
            coeff = vorp * p_win

            cand_list.append((b, p_win, coeff, var))
            obiettivo_terms.append(coeff * var)
            vars_per_ruolo[player.ruolo].append(var)
            vars_per_squadra.setdefault(player.squadra, []).append(var)
            tutte_le_var_con_costo.append((b, var))

        candidati_per_player[player_id] = cand_list

    prob += pulp.lpSum(obiettivo_terms), "utilita_attesa_totale"

    for player_id, cand_list in candidati_per_player.items():
        if cand_list:
            prob += pulp.lpSum(v for *_, v in cand_list) <= 1, f"al_piu_una_offerta_{player_id}"

    prob += (
        pulp.lpSum(b * v for b, v in tutte_le_var_con_costo) <= budget,
        "vincolo_budget",
    )

    for ruolo in _RUOLI:
        richiesti = getattr(roster_slots, ruolo)
        prob += (
            pulp.lpSum(vars_per_ruolo[ruolo]) == richiesti,
            f"rosa_esatta_{ruolo}",
        )

    max_per_squadra = config.max_giocatori_per_squadra_serie_a
    if max_per_squadra is not None:
        for squadra, vars_squadra in vars_per_squadra.items():
            prob += (
                pulp.lpSum(vars_squadra) <= max_per_squadra,
                f"max_per_squadra_{squadra}",
            )

    return prob, candidati_per_player


def _verifica_fattibilita_necessaria(
    joined: _Joined,
    roster_slots: RosterSlots,
    budget: float,
    candidati_per_player: dict[str, list[_Candidato]],
) -> None:
    """Controlli rapidi (senza invocare CBC) su condizioni necessarie ma non
    sufficienti di fattibilità, per dare un messaggio diagnostico specifico
    invece del generico "Infeasible" del solver quando la causa è ovvia."""
    per_ruolo: dict[str, list[tuple[str, int]]] = {r: [] for r in _RUOLI}
    for player_id, cand_list in candidati_per_player.items():
        if not cand_list:
            continue
        ruolo = joined[player_id][0].ruolo
        costo_min = min(b for b, *_ in cand_list)
        per_ruolo[ruolo].append((player_id, costo_min))

    problemi: list[str] = []
    costo_minimo_totale = 0.0
    for ruolo in _RUOLI:
        richiesti = getattr(roster_slots, ruolo)
        disponibili = per_ruolo[ruolo]
        if len(disponibili) < richiesti:
            problemi.append(
                f"ruolo {ruolo}: servono {richiesti} giocatori ma solo "
                f"{len(disponibili)} hanno proiezione e distribuzione di "
                "prezzo valide in input"
            )
        else:
            costo_minimo_totale += sum(
                costo for _, costo in sorted(disponibili, key=lambda t: t[1])[:richiesti]
            )

    if problemi:
        raise InfeasibleRosterError(
            "Composizione rosa richiesta non raggiungibile con i dati disponibili:\n"
            + "\n".join(f"  - {p}" for p in problemi)
        )

    if costo_minimo_totale > budget:
        raise InfeasibleRosterError(
            "Budget insufficiente per la rosa richiesta: anche scegliendo, per "
            "ogni ruolo, i giocatori con l'offerta minima più bassa nella "
            f"griglia discretizzata, servirebbero almeno {costo_minimo_totale:.0f} "
            f"crediti, ma il budget disponibile è {budget:.0f}."
        )


def _risolvi(prob: pulp.LpProblem, time_limit_s: float) -> str:
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def _estrai_selezione(
    candidati_per_player: dict[str, list[_Candidato]],
) -> list[tuple[str, int, float, float]]:
    """(player_id, offerta, p_win, coefficiente) per ogni variabile scelta
    dall'ultima `.solve()` sul problema che ha generato `candidati_per_player`."""
    selezionati = []
    for player_id, cand_list in candidati_per_player.items():
        for b, p_win, coeff, var in cand_list:
            valore = var.value()
            if valore is not None and valore > 0.5:
                selezionati.append((player_id, b, p_win, coeff))
    return selezionati


def _bid_plan_da_selezione(
    selezionati: list[tuple[str, int, float, float]],
    joined: _Joined,
    tornata: Tornata,
) -> list[BidPlan]:
    piani = []
    for player_id, offerta, p_win, coeff in selezionati:
        player, proj, dist = joined[player_id]
        is_synthetic = player.is_synthetic or proj.is_synthetic or dist.is_synthetic
        fonte = (
            "fantabuste.optimize.solver v1 "
            f"(player={player.fonte}; projection={proj.fonte}; "
            f"price_distribution={dist.fonte})"
        )
        piani.append(
            BidPlan(
                player_id=player_id,
                offerta=float(offerta),
                p_win_stimata=p_win,
                vorp=proj.vorp,
                surplus_atteso=coeff,
                tornata=tornata,
                fonte=fonte,
                is_synthetic=is_synthetic,
            )
        )
    return piani


def _top_k_alternative(
    prob: pulp.LpProblem,
    candidati_per_player: dict[str, list[_Candidato]],
    joined: _Joined,
    tornata: Tornata,
    n_alternative: int,
    time_limit_s: float,
) -> list[AlternativeRoster]:
    """Enumera fino a `n_alternative` soluzioni ammissibili distinte, per
    utilità attesa decrescente (rank=1 è l'ottimo globale del MILP), via
    "no-good cut" successivi: dopo ogni soluzione trovata, si vieta di
    riselezionare esattamente la stessa combinazione di (giocatore, offerta),
    forzando il solve successivo a trovare la miglior alternativa diversa.
    Ogni cut lavora sullo stesso oggetto `prob` (PuLP supporta risolvere più
    volte lo stesso problema dopo avergli aggiunto vincoli)."""
    alternative: list[AlternativeRoster] = []
    utilita_ottimo: float | None = None

    for rank in range(1, n_alternative + 1):
        status = _risolvi(prob, time_limit_s)
        if status != "Optimal":
            break

        selezionati = _estrai_selezione(candidati_per_player)
        if not selezionati:
            break

        utilita = sum(coeff for *_, coeff in selezionati)
        if utilita_ottimo is None:
            utilita_ottimo = utilita

        bid_plan = _bid_plan_da_selezione(selezionati, joined, tornata)
        alternative.append(
            AlternativeRoster(
                rank=rank,
                bid_plan=bid_plan,
                utilita_attesa=utilita,
                delta_vs_ottimo=utilita - utilita_ottimo,
            )
        )

        selected_vars = [
            var
            for cand_list in candidati_per_player.values()
            for _, _, _, var in cand_list
            if (v := var.value()) is not None and v > 0.5
        ]
        prob += (
            pulp.lpSum(selected_vars) <= len(selected_vars) - 1,
            f"no_good_cut_rank_{rank}",
        )

    return alternative


def _sigma_da_proiezione(proj: PlayerProjection) -> float:
    """Deviazione standard implicita di un giocatore, dall'intervallo
    [punti_attesi_lo, punti_attesi_hi] — vedi il disclaimer completo in
    `SensitivityReport`."""
    return max(0.0, (proj.punti_attesi_hi - proj.punti_attesi_lo) / 2)


def _sensitivity(
    joined: _Joined,
    config: LeagueConfig,
    roster_slots: RosterSlots,
    budget: float,
    time_limit_s: float,
    ottimo_ids: frozenset[str],
    utilita_ottimo: float,
    soglia_instabilita: float,
) -> SensitivityReport:
    sigma_by_id = {pid: _sigma_da_proiezione(proj) for pid, (_, proj, _) in joined.items()}

    risultati: dict[str, tuple[frozenset[str], float]] = {}
    for segno, chiave in ((1, "positivo"), (-1, "negativo")):
        vorp_override = {
            pid: proj.vorp + segno * sigma_by_id[pid] for pid, (_, proj, _) in joined.items()
        }
        prob, candidati = _costruisci_problema(
            joined, config, roster_slots, budget, vorp_override=vorp_override
        )
        status = _risolvi(prob, time_limit_s)
        if status != "Optimal":
            raise InfeasibleRosterError(
                f"Analisi di sensibilità (spostamento {chiave}): il problema "
                "perturbato è infeasible. Non dovrebbe accadere, dato che i "
                "vincoli di rosa/budget non dipendono dal VORP — controllare "
                "i dati di input (proiezioni con hi/lo incoerenti?)."
            )
        selezionati = _estrai_selezione(candidati)
        ids = frozenset(pid for pid, *_ in selezionati)
        utilita = sum(coeff for *_, coeff in selezionati)
        risultati[chiave] = (ids, utilita)

    ids_pos, utilita_pos = risultati["positivo"]
    ids_neg, utilita_neg = risultati["negativo"]

    def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 1.0
        unione = a | b
        return len(a & b) / len(unione) if unione else 1.0

    j_pos = jaccard(ottimo_ids, ids_pos)
    j_neg = jaccard(ottimo_ids, ids_neg)
    instabile = j_pos < soglia_instabilita or j_neg < soglia_instabilita

    if instabile:
        note = (
            "ROSA INSTABILE: uno spostamento di +-1 sigma nelle proiezioni "
            f"cambia una parte sostanziale della rosa ottima (Jaccard "
            f"positivo={j_pos:.2f}, negativo={j_neg:.2f}, soglia="
            f"{soglia_instabilita:.2f}). Trattare l'ottimo come indicativo, "
            "non come piano da eseguire alla lettera."
        )
    else:
        note = (
            f"Rosa stabile rispetto a +-1 sigma nelle proiezioni (Jaccard "
            f"positivo={j_pos:.2f}, negativo={j_neg:.2f}, soglia="
            f"{soglia_instabilita:.2f})."
        )

    return SensitivityReport(
        utilita_ottimo=utilita_ottimo,
        utilita_shift_positivo=utilita_pos,
        utilita_shift_negativo=utilita_neg,
        roster_ottimo_ids=ottimo_ids,
        roster_shift_positivo_ids=ids_pos,
        roster_shift_negativo_ids=ids_neg,
        jaccard_positivo=j_pos,
        jaccard_negativo=j_neg,
        soglia_instabilita=soglia_instabilita,
        instabile=instabile,
        note=note,
    )


def _applica_offerte_non_tonde(
    bid_plan: list[BidPlan],
    budget: float,
    joined: _Joined,
) -> list[BidPlan]:
    """Sposta le offerte multiple di 5 (e quindi anche di 10) di 1 credito,
    per rompere a proprio favore un eventuale pareggio con un avversario che
    offre un valore tondo — attiva solo se
    `config.assegna_buste_se_uguali is False` (pareggio = giocatore non
    assegnato) e `config.offerte_non_tonde is True`. Vedi
    docs/DESIGN.md Agente D, punto 3.

    Direzione preferita +1 (si scommette di superare un avversario che offre
    lo stesso tondo). Se +1 farebbe sforare il budget residuo del piano si
    tenta -1 (comunque fuori dal multiplo tondo, e riduce la spesa quindi non
    può mai sforare il budget). Se nemmeno -1 è applicabile (offerta già al
    minimo strutturale) l'offerta resta invariata: è un'eccezione rara e
    accettata, non un bug — un'offerta a 5 crediti non ha uno spazio di
    manovra sotto di sé.

    Ricalcola `p_win_stimata` e `surplus_atteso` per ogni offerta modificata
    usando la stessa `PriceDistribution` del giocatore, così i due campi
    restano coerenti con l'offerta effettiva anche dopo l'aggiustamento.
    """
    aggiustati: list[BidPlan] = []
    totale_corrente = sum(bp.offerta for bp in bid_plan)

    for bp in bid_plan:
        offerta = bp.offerta
        if offerta % 5 != 0:
            aggiustati.append(bp)
            continue

        player, proj, dist = joined[bp.player_id]
        nuova_offerta = offerta
        if totale_corrente + 1 <= budget:
            nuova_offerta = offerta + 1
            totale_corrente += 1
        elif offerta - 1 >= MIN_OFFERTA:
            nuova_offerta = offerta - 1
            totale_corrente -= 1

        if nuova_offerta == offerta:
            aggiustati.append(bp)
            continue

        nuovo_p_win = dist.p_win(nuova_offerta, player.quotazione_listone)
        aggiustati.append(
            bp.model_copy(
                update={
                    "offerta": float(nuova_offerta),
                    "p_win_stimata": nuovo_p_win,
                    "surplus_atteso": proj.vorp * nuovo_p_win,
                }
            )
        )

    totale_finale = sum(bp.offerta for bp in aggiustati)
    if totale_finale > budget + 1e-9:
        raise RuntimeError(
            "Bug interno: il post-processing offerte non tonde ha rotto il "
            f"vincolo di budget ({totale_finale} > {budget}). Non dovrebbe "
            "poter accadere dato l'aggiustamento greedy sopra — segnalare."
        )
    return aggiustati


def solve_bid_plan(
    players: Sequence[Player],
    projections: Sequence[PlayerProjection],
    price_distributions: Sequence[PriceDistribution],
    config: LeagueConfig,
    *,
    roster_slots: RosterSlots | None = None,
    budget: float | None = None,
    tornata: Tornata = 1,
    n_alternative: int = DEFAULT_N_ALTERNATIVE,
    calcola_sensitivity: bool = True,
    soglia_instabilita: float = DEFAULT_SOGLIA_INSTABILITA,
    solver_time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_S,
) -> OptimizeResult:
    """Risolve il MILP di allocazione del budget su buste chiuse. Vedi il
    docstring del modulo per il modello, i vincoli e le approssimazioni
    dichiarate (indipendenza di `p_win` fra giocatori; vincoli speciali di
    regolamento fuori scope in v1).

    Parametri opzionali:
    - `roster_slots`, `budget`: di default `config.rosa_fase1` e
      `config.budget_totale_fase1`. Esposti espliciti (invece di leggerli
      solo internamente da `config`) perché un chiamante che gestisce le 2
      tornate o un futuro Modulo F che riusa questo solver su un
      sottoinsieme di lotti aperti deve poter passare slot/budget residui
      diversi da quelli "di listino" — vedi il docstring di modulo su
      `tornata`.
    - `n_alternative`: quante rose (rank 1..N) restituire in
      `OptimizeResult.alternative`. Default 5, come da DoD dell'Agente D.
    - `calcola_sensitivity`: disattivabile per velocità (es. in un ciclo di
      test) — richiede 2 risolve aggiuntive.
    - `solver_time_limit_s`: timeout per singola chiamata a CBC. Con la
      griglia discretizzata di `grid.py` e 600 giocatori la soluzione ottima
      si trova tipicamente in pochi secondi; il timeout è una rete di
      sicurezza, non il caso atteso.

    Solleva `InfeasibleRosterError` (mai un crash grezzo di PuLP/CBC) se non
    esiste alcuna rosa ammissibile — per esempio se il pool di giocatori in
    input non ha abbastanza titolari di un ruolo, o se il budget minimo
    necessario a riempire la rosa richiesta eccede il budget disponibile.
    """
    with warnings.catch_warnings():
        # La versione di PuLP installata annuncia con DeprecationWarning
        # un'API futura (v4: LpVariable diretta e PULP_CBC_CMD verranno
        # sostituite) che qui non riguarda la correttezza del modello — solo
        # rumore nell'output dei test (una per ogni variabile creata, quindi
        # decine di migliaia su 600 giocatori). Silenziata solo per la durata
        # di questa chiamata, non globalmente per il processo.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")
        return _solve_bid_plan_impl(
            players,
            projections,
            price_distributions,
            config,
            roster_slots=roster_slots,
            budget=budget,
            tornata=tornata,
            n_alternative=n_alternative,
            calcola_sensitivity=calcola_sensitivity,
            soglia_instabilita=soglia_instabilita,
            solver_time_limit_s=solver_time_limit_s,
        )


def _solve_bid_plan_impl(
    players: Sequence[Player],
    projections: Sequence[PlayerProjection],
    price_distributions: Sequence[PriceDistribution],
    config: LeagueConfig,
    *,
    roster_slots: RosterSlots | None,
    budget: float | None,
    tornata: Tornata,
    n_alternative: int,
    calcola_sensitivity: bool,
    soglia_instabilita: float,
    solver_time_limit_s: float,
) -> OptimizeResult:
    roster_slots = roster_slots if roster_slots is not None else config.rosa_fase1
    budget = budget if budget is not None else config.budget_totale_fase1

    joined, esclusi = _join_input(players, projections, price_distributions)
    warnings: list[str] = []
    if esclusi:
        warnings.append(
            f"{len(esclusi)} giocatori esclusi dal problema: privi di "
            "PlayerProjection e/o PriceDistribution in input."
        )

    prob, candidati = _costruisci_problema(joined, config, roster_slots, budget)
    _verifica_fattibilita_necessaria(joined, roster_slots, budget, candidati)

    alternative = _top_k_alternative(
        prob, candidati, joined, tornata, n_alternative, solver_time_limit_s
    )
    if not alternative:
        raise InfeasibleRosterError(
            "Nessuna assegnazione ammissibile trovata dal solver CBC pur "
            "avendo superato i controlli rapidi su conteggi minimi per ruolo "
            "e budget minimo. La causa più probabile è "
            "max_giocatori_per_squadra_serie_a combinato con la "
            "distribuzione per squadra dei giocatori disponibili in input — "
            "verificare quel parametro in LeagueConfig."
        )

    ottimo = alternative[0]
    bid_plan = ottimo.bid_plan

    if config.offerte_non_tonde and not config.assegna_buste_se_uguali:
        bid_plan = _applica_offerte_non_tonde(bid_plan, budget, joined)
        alternative[0] = replace(ottimo, bid_plan=bid_plan)

    sensitivity: SensitivityReport | None = None
    if calcola_sensitivity:
        ottimo_ids = frozenset(bp.player_id for bp in ottimo.bid_plan)
        sensitivity = _sensitivity(
            joined,
            config,
            roster_slots,
            budget,
            solver_time_limit_s,
            ottimo_ids,
            ottimo.utilita_attesa,
            soglia_instabilita,
        )

    return OptimizeResult(
        bid_plan=bid_plan,
        utilita_attesa=ottimo.utilita_attesa,
        alternative=alternative,
        sensitivity=sensitivity,
        giocatori_esclusi=esclusi,
        warnings=warnings,
    )
