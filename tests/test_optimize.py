"""Test del Modulo D — ottimizzatore MILP.

Le fixture "grandi" (600 giocatori) leggono `data/fixtures/players.csv` in
sola lettura (mai modificato qui) e costruiscono `PlayerProjection`/
`PriceDistribution` sintetiche a partire dalla `quotazione_listone` reale
di quel CSV — così il test di correttezza/performance gira sulla stessa
popolazione di giocatori usata dagli altri moduli in Fase 1, non su una
fixture inventata ad-hoc. `PlayerProjection` e `PriceDistribution` restano
comunque sintetiche (Moduli B e C non sono ancora implementati): tutto è
marcato `is_synthetic=True` e la fonte lo dichiara esplicitamente.
"""

from __future__ import annotations

import csv
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fantabuste.config import EXAMPLE_CONFIG_PATH, LeagueConfig
from fantabuste.optimize import InfeasibleRosterError, solve_bid_plan
from fantabuste.optimize.grid import MIN_OFFERTA, griglia_offerte
from fantabuste.optimize.solver import _applica_offerte_non_tonde
from fantabuste.schemas import (
    BidPlan,
    MetodoProiezione,
    Player,
    PlayerProjection,
    PriceDistribution,
    PriceDistributionMode,
    RosterSlots,
)

PLAYERS_CSV = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "players.csv"


# ---------------------------------------------------------------------------
# Helper di costruzione — nessuno di questi legge/scrive dati reali.
# ---------------------------------------------------------------------------


def _carica_players_da_csv() -> list[Player]:
    """Legge data/fixtures/players.csv in sola lettura (mai scritto qui)."""
    players: list[Player] = []
    with PLAYERS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            players.append(
                Player(
                    player_id=row["player_id"],
                    nome=row["nome"],
                    ruolo=row["ruolo"],
                    squadra=row["squadra"],
                    quotazione_listone=float(row["quotazione_listone"]),
                    fonte=row["fonte"],
                    is_synthetic=row["is_synthetic"] == "True",
                    data_estrazione=row["data_estrazione"],
                )
            )
    return players


def _proiezione_sintetica(player: Player, rng: random.Random) -> PlayerProjection:
    """PlayerProjection sintetica plausibile, derivata dalla quotazione_listone
    reale del CSV (il Modulo B non è ancora implementato)."""
    punti = 5.0 + player.quotazione_listone * 0.35 + rng.uniform(-1.0, 1.0)
    ampiezza = 2.0 + player.quotazione_listone * 0.05
    vorp = player.quotazione_listone * rng.uniform(0.6, 1.4) - 4.0
    return PlayerProjection(
        player_id=player.player_id,
        punti_attesi=punti,
        punti_attesi_lo=punti - ampiezza,
        punti_attesi_hi=punti + ampiezza,
        prob_titolarita=rng.uniform(0.5, 0.95),
        vorp=vorp,
        metodo=MetodoProiezione.BASELINE,
        confidence_flag="bassa",
        fonte="test_optimize_projection_sintetica",
        is_synthetic=True,
    )


def _distribuzione_prior_sintetica(player: Player, *, sigma: float = 0.35) -> PriceDistribution:
    """PriceDistribution sintetica in modalità 'prior', centrata sulla
    quotazione_listone reale del CSV (prior_mu=0.0 in log-spazio => rapporto
    offerta/quotazione atteso = 1.0). Il Modulo C non è ancora implementato
    e comunque degraderebbe a 'prior' in v1 — vedi docs/OPEN_QUESTIONS.md §2."""
    return PriceDistribution(
        player_id=player.player_id,
        fascia=f"{player.ruolo}_prior_test",
        mode=PriceDistributionMode.PRIOR,
        n_osservazioni=0,
        prior_mu=0.0,
        prior_sigma=sigma,
        fonte="test_optimize_price_distribution_sintetica",
        is_synthetic=True,
    )


def _mini_player(
    player_id: str,
    *,
    ruolo: str = "A",
    squadra: str = "SQX",
    quotazione: float = 40.0,
    is_synthetic: bool = True,
) -> Player:
    return Player(
        player_id=player_id,
        nome=f"Test {player_id}",
        ruolo=ruolo,
        squadra=squadra,
        quotazione_listone=quotazione,
        fonte="test",
        is_synthetic=is_synthetic,
        data_estrazione=datetime(2026, 8, 25, tzinfo=UTC),
    )


def _mini_projection(
    player_id: str, *, vorp: float = 5.0, is_synthetic: bool = True
) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id,
        punti_attesi=10,
        punti_attesi_lo=8,
        punti_attesi_hi=12,
        prob_titolarita=0.8,
        vorp=vorp,
        metodo=MetodoProiezione.BASELINE,
        confidence_flag="bassa",
        fonte="test",
        is_synthetic=is_synthetic,
    )


def _mini_distribuzione(player_id: str, *, is_synthetic: bool = True) -> PriceDistribution:
    return PriceDistribution(
        player_id=player_id,
        fascia="A_test",
        mode=PriceDistributionMode.PRIOR,
        n_osservazioni=0,
        prior_mu=0.0,
        prior_sigma=0.3,
        fonte="test",
        is_synthetic=is_synthetic,
    )


# ---------------------------------------------------------------------------
# Fixture pytest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> LeagueConfig:
    return LeagueConfig.load(EXAMPLE_CONFIG_PATH)


@pytest.fixture(scope="module")
def players_600() -> list[Player]:
    players = _carica_players_da_csv()
    assert len(players) == 600
    return players


@pytest.fixture(scope="module")
def proiezioni_e_distribuzioni_600(
    players_600: list[Player],
) -> tuple[list[PlayerProjection], list[PriceDistribution]]:
    rng = random.Random(20260825)
    projections = [_proiezione_sintetica(p, rng) for p in players_600]
    distributions = [_distribuzione_prior_sintetica(p) for p in players_600]
    return projections, distributions


# ---------------------------------------------------------------------------
# DoD: risolve in <30s su 600 giocatori, rispetta i vincoli di rosa
# ---------------------------------------------------------------------------


def test_solve_600_giocatori_rispetta_vincoli_e_tempo(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600

    t0 = time.monotonic()
    result = solve_bid_plan(players_600, projections, distributions, config)
    elapsed = time.monotonic() - t0

    assert elapsed < 30.0, f"solve troppo lento: {elapsed:.1f}s (DoD: <30s su 600 giocatori)"

    assert len(result.bid_plan) == config.rosa_fase1.totale
    assert len({bp.player_id for bp in result.bid_plan}) == len(result.bid_plan)

    by_id = {p.player_id: p for p in players_600}
    conteggio_ruoli = Counter(by_id[bp.player_id].ruolo for bp in result.bid_plan)
    assert conteggio_ruoli["P"] == config.rosa_fase1.P
    assert conteggio_ruoli["D"] == config.rosa_fase1.D
    assert conteggio_ruoli["C"] == config.rosa_fase1.C
    assert conteggio_ruoli["A"] == config.rosa_fase1.A

    assert sum(bp.offerta for bp in result.bid_plan) <= config.budget_totale_fase1 + 1e-9

    for bp in result.bid_plan:
        assert bp.offerta > 0
        assert 0.0 <= bp.p_win_stimata <= 1.0
        assert bp.tornata == 1
        assert bp.fonte  # mai vuoto
        # tutti gli input di questo test sono sintetici -> ogni BidPlan deve
        # dichiararlo, mai un default silenzioso a False.
        assert bp.is_synthetic is True


def test_solve_600_produce_top5_alternative_ordinate(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    result = solve_bid_plan(players_600, projections, distributions, config)

    assert 1 <= len(result.alternative) <= 5
    assert [alt.rank for alt in result.alternative] == list(range(1, len(result.alternative) + 1))
    assert result.alternative[0].delta_vs_ottimo == 0.0
    assert result.alternative[0].utilita_attesa == pytest.approx(result.utilita_attesa)

    utilita = [alt.utilita_attesa for alt in result.alternative]
    assert utilita == sorted(utilita, reverse=True), "alternative non ordinate per utilità"
    for alt in result.alternative[1:]:
        assert alt.delta_vs_ottimo <= 1e-9
        assert len(alt.bid_plan) == config.rosa_fase1.totale

    # le rose alternative devono essere effettivamente diverse dall'ottimo
    roster_ottimo = {bp.player_id for bp in result.alternative[0].bid_plan}
    for alt in result.alternative[1:]:
        roster_alt = {bp.player_id for bp in alt.bid_plan}
        assert roster_alt != roster_ottimo or {
            (bp.player_id, bp.offerta) for bp in alt.bid_plan
        } != {(bp.player_id, bp.offerta) for bp in result.alternative[0].bid_plan}


def test_solve_600_analisi_di_sensibilita(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    result = solve_bid_plan(players_600, projections, distributions, config)

    assert result.sensitivity is not None
    sens = result.sensitivity
    assert 0.0 <= sens.jaccard_positivo <= 1.0
    assert 0.0 <= sens.jaccard_negativo <= 1.0
    assert len(sens.roster_ottimo_ids) == config.rosa_fase1.totale
    assert len(sens.roster_shift_positivo_ids) == config.rosa_fase1.totale
    assert len(sens.roster_shift_negativo_ids) == config.rosa_fase1.totale
    # la rosa ottima riportata nel report di sensibilità deve coincidere con
    # quella del piano principale
    assert sens.roster_ottimo_ids == {bp.player_id for bp in result.bid_plan}
    # instabile deve essere coerente con la soglia dichiarata
    soglia = sens.soglia_instabilita
    atteso_instabile = sens.jaccard_positivo < soglia or sens.jaccard_negativo < soglia
    assert sens.instabile is atteso_instabile
    assert sens.note  # sempre una spiegazione testuale, mai vuota


def test_max_giocatori_per_squadra_rispettato(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    cfg = config.model_copy(update={"max_giocatori_per_squadra_serie_a": 2})

    result = solve_bid_plan(
        players_600, projections, distributions, cfg, calcola_sensitivity=False, n_alternative=1
    )

    by_id = {p.player_id: p for p in players_600}
    conteggio_squadre = Counter(by_id[bp.player_id].squadra for bp in result.bid_plan)
    assert all(c <= 2 for c in conteggio_squadre.values())


# ---------------------------------------------------------------------------
# Infeasibility: mai un crash, sempre un messaggio comprensibile
# ---------------------------------------------------------------------------


def test_budget_troppo_basso_solleva_errore_chiaro(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    with pytest.raises(InfeasibleRosterError, match="Budget insufficiente"):
        solve_bid_plan(
            players_600,
            projections,
            distributions,
            config,
            budget=1.0,
            calcola_sensitivity=False,
        )


def test_ruolo_insufficiente_solleva_errore_chiaro(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    by_id = {p.player_id: p for p in players_600}
    portieri_ids = {pid for pid, p in by_id.items() if p.ruolo == "P"}
    projections_senza_portieri = [p for p in projections if p.player_id not in portieri_ids]

    with pytest.raises(InfeasibleRosterError, match="ruolo P"):
        solve_bid_plan(
            players_600,
            projections_senza_portieri,
            distributions,
            config,
            calcola_sensitivity=False,
        )


def test_max_per_squadra_troppo_stringente_solleva_errore_chiaro(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    """Caso in cui i controlli rapidi (conteggi per ruolo, budget minimo)
    passano ma il MILP è comunque infeasible per via del vincolo per
    squadra: deve arrivare comunque un InfeasibleRosterError leggibile,
    non un crash di CBC/PuLP."""
    projections, distributions = proiezioni_e_distribuzioni_600
    cfg = config.model_copy(update={"max_giocatori_per_squadra_serie_a": 0})

    with pytest.raises(InfeasibleRosterError, match="max_giocatori_per_squadra"):
        solve_bid_plan(
            players_600,
            projections,
            distributions,
            cfg,
            calcola_sensitivity=False,
        )


def test_nessun_giocatore_in_input_solleva_errore_chiaro(config: LeagueConfig) -> None:
    with pytest.raises(InfeasibleRosterError):
        solve_bid_plan([], [], [], config, calcola_sensitivity=False)


# ---------------------------------------------------------------------------
# Join fra Player/PlayerProjection/PriceDistribution ed esclusioni
# ---------------------------------------------------------------------------


def _config_minima(**overrides) -> LeagueConfig:
    base = LeagueConfig.load(EXAMPLE_CONFIG_PATH)
    update = {"rosa_fase1": RosterSlots(P=1, D=0, C=0, A=0), "budget_totale_fase1": 50.0}
    update.update(overrides)
    return base.model_copy(update=update)


def test_giocatori_esclusi_se_mancano_proiezione_o_distribuzione() -> None:
    cfg = _config_minima()
    con_dati = _mini_player("PZ1", ruolo="P", quotazione=10.0)
    senza_proiezione = _mini_player("PZ2", ruolo="P", quotazione=10.0)
    proj = _mini_projection("PZ1")
    dist1 = _mini_distribuzione("PZ1")
    dist2 = _mini_distribuzione("PZ2")  # PZ2 non ha una PlayerProjection

    result = solve_bid_plan(
        [con_dati, senza_proiezione],
        [proj],
        [dist1, dist2],
        cfg,
        calcola_sensitivity=False,
        n_alternative=1,
    )

    assert result.giocatori_esclusi == ["PZ2"]
    assert any("esclusi" in w for w in result.warnings)
    assert {bp.player_id for bp in result.bid_plan} == {"PZ1"}


def test_bid_plan_is_synthetic_true_se_una_qualsiasi_dipendenza_e_sintetica() -> None:
    cfg = _config_minima()
    player = _mini_player("PY1", ruolo="P", quotazione=10.0, is_synthetic=False)
    proj = _mini_projection("PY1", is_synthetic=True)  # <- l'unica dipendenza sintetica
    dist = _mini_distribuzione("PY1", is_synthetic=False)

    result = solve_bid_plan(
        [player], [proj], [dist], cfg, calcola_sensitivity=False, n_alternative=1
    )

    assert len(result.bid_plan) == 1
    assert result.bid_plan[0].is_synthetic is True


def test_bid_plan_is_synthetic_false_se_tutte_le_dipendenze_sono_reali() -> None:
    cfg = _config_minima()
    player = _mini_player("PY2", ruolo="P", quotazione=10.0, is_synthetic=False)
    proj = _mini_projection("PY2", is_synthetic=False)
    dist = _mini_distribuzione("PY2", is_synthetic=False)

    result = solve_bid_plan(
        [player], [proj], [dist], cfg, calcola_sensitivity=False, n_alternative=1
    )

    assert result.bid_plan[0].is_synthetic is False


# ---------------------------------------------------------------------------
# Post-processing: offerte non tonde (config.offerte_non_tonde)
# ---------------------------------------------------------------------------


def _bid_plan_singolo(
    player_id: str, offerta: float, dist: PriceDistribution, proj: PlayerProjection
) -> list[BidPlan]:
    p_win = dist.p_win(offerta, 40.0)
    return [
        BidPlan(
            player_id=player_id,
            offerta=offerta,
            p_win_stimata=p_win,
            vorp=proj.vorp,
            surplus_atteso=proj.vorp * p_win,
            tornata=1,
            fonte="test",
            is_synthetic=True,
        )
    ]


def test_offerte_non_tonde_sposta_verso_alto_se_ce_margine_di_budget() -> None:
    player = _mini_player("PX1", quotazione=40.0)
    proj = _mini_projection("PX1")
    dist = _mini_distribuzione("PX1")
    joined = {"PX1": (player, proj, dist)}
    bid_plan = _bid_plan_singolo("PX1", 40.0, dist, proj)

    aggiustato = _applica_offerte_non_tonde(bid_plan, budget=100.0, joined=joined)

    assert aggiustato[0].offerta == 41.0
    assert aggiustato[0].p_win_stimata == pytest.approx(dist.p_win(41, 40.0))
    assert aggiustato[0].surplus_atteso == pytest.approx(proj.vorp * dist.p_win(41, 40.0))


def test_offerte_non_tonde_sposta_verso_basso_se_budget_esaurito() -> None:
    player = _mini_player("PX2", quotazione=40.0)
    proj = _mini_projection("PX2")
    dist = _mini_distribuzione("PX2")
    joined = {"PX2": (player, proj, dist)}
    bid_plan = _bid_plan_singolo("PX2", 40.0, dist, proj)

    # budget == offerta corrente: nessun margine per +1
    aggiustato = _applica_offerte_non_tonde(bid_plan, budget=40.0, joined=joined)

    assert aggiustato[0].offerta == 39.0


def test_offerte_non_tonde_non_tocca_offerte_gia_non_tonde() -> None:
    player = _mini_player("PX3", quotazione=40.0)
    proj = _mini_projection("PX3")
    dist = _mini_distribuzione("PX3")
    joined = {"PX3": (player, proj, dist)}
    bid_plan = _bid_plan_singolo("PX3", 37.0, dist, proj)

    aggiustato = _applica_offerte_non_tonde(bid_plan, budget=100.0, joined=joined)

    assert aggiustato[0] is bid_plan[0]


def test_offerte_non_tonde_rispetta_sempre_il_budget_totale() -> None:
    player_a = _mini_player("PX4", quotazione=40.0)
    player_b = _mini_player("PX5", quotazione=40.0)
    proj_a = _mini_projection("PX4")
    proj_b = _mini_projection("PX5")
    dist_a = _mini_distribuzione("PX4")
    dist_b = _mini_distribuzione("PX5")
    joined = {"PX4": (player_a, proj_a, dist_a), "PX5": (player_b, proj_b, dist_b)}
    bid_plan = _bid_plan_singolo("PX4", 30.0, dist_a, proj_a) + _bid_plan_singolo(
        "PX5", 30.0, dist_b, proj_b
    )

    aggiustato = _applica_offerte_non_tonde(bid_plan, budget=60.0, joined=joined)

    assert sum(bp.offerta for bp in aggiustato) <= 60.0


def test_solve_disattiva_post_processing_se_offerte_non_tonde_false(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    cfg = config.model_copy(update={"offerte_non_tonde": False})

    result = solve_bid_plan(
        players_600, projections, distributions, cfg, calcola_sensitivity=False
    )

    # con offerte_non_tonde=False il piano finale è esattamente l'output
    # grezzo del MILP: nessuna nuova lista viene costruita dal post-processing.
    assert result.bid_plan is result.alternative[0].bid_plan


def test_solve_disattiva_post_processing_se_assegna_buste_se_uguali_true(
    config: LeagueConfig,
    players_600: list[Player],
    proiezioni_e_distribuzioni_600: tuple[list[PlayerProjection], list[PriceDistribution]],
) -> None:
    projections, distributions = proiezioni_e_distribuzioni_600
    cfg = config.model_copy(update={"assegna_buste_se_uguali": True})

    result = solve_bid_plan(
        players_600, projections, distributions, cfg, calcola_sensitivity=False
    )

    assert result.bid_plan is result.alternative[0].bid_plan


# ---------------------------------------------------------------------------
# Griglia di discretizzazione (grid.py)
# ---------------------------------------------------------------------------


def test_griglia_offerte_ordinata_unica_e_nei_limiti() -> None:
    livelli = griglia_offerte(quotazione_listone=40.0, budget_massimo=500.0)
    assert livelli == sorted(set(livelli))
    assert all(MIN_OFFERTA <= b <= 500 for b in livelli)
    assert len(livelli) > 1


def test_griglia_offerte_vuota_se_budget_insufficiente() -> None:
    assert griglia_offerte(quotazione_listone=10.0, budget_massimo=0.5) == []


def test_griglia_offerte_rispetta_il_tetto_di_budget() -> None:
    livelli = griglia_offerte(quotazione_listone=200.0, budget_massimo=50.0)
    assert all(b <= 50 for b in livelli)
    assert livelli  # deve comunque restare almeno un livello (clippato al tetto)


def test_griglia_offerte_piu_densa_per_fascia_alta() -> None:
    bassa = griglia_offerte(quotazione_listone=5.0, budget_massimo=500.0)
    alta = griglia_offerte(quotazione_listone=50.0, budget_massimo=500.0)
    assert len(alta) >= len(bassa)
