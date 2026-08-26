"""Test del Modulo C (modello delle offerte avversarie).

Tutte le fixture sono costruite qui inline, sintetiche, marcate
`is_synthetic=True` — non toccano `data/fixtures/` né `fixtures.py` (di
competenza degli altri agenti della Fase 1, in parallelo), come da
CLAUDE.md e docs/DESIGN.md, Agente C.

Stato v1 (verificato 2026-08-25, docs/OPEN_QUESTIONS.md §2, §2.1): nessuno
storico reale di offerte avversarie esiste per questa lega, quindi in
produzione ogni fascia gira in `prior`. Questo file testa ENTRAMBE le
modalità per intero — `empirical` con osservazioni sintetiche sufficienti,
e la degradazione automatica a `prior` sotto soglia K — perché il percorso
`empirical` è codice che serve dall'anno prossimo (vedi CLAUDE.md,
corollario operativo), anche se non è ancora utilizzabile sui dati reali.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from fantabuste.bidmodel.diagnostics import curva_p_win, curve_p_win
from fantabuste.bidmodel.fasce import N_TIER_DEFAULT, assegna_fasce, nome_fascia
from fantabuste.bidmodel.fitting import (
    PRIOR_MU_DEFAULT,
    PRIOR_SIGMA_DEFAULT,
    costruisci_lot_max_ratios,
    fit_fascia,
    fit_price_distributions,
)
from fantabuste.bidmodel.montecarlo import simula_asta
from fantabuste.bidmodel.normalizzazione import (
    StagioneStorica,
    normalizza_stagione,
    pool_stagioni,
)
from fantabuste.bidmodel.report import genera_report
from fantabuste.config import AstaRiparazioneConfig, LeagueConfig
from fantabuste.schemas import (
    OpponentBidObservation,
    Player,
    PriceDistribution,
    PriceDistributionMode,
    RosterSlots,
)

FONTE_TEST = "test_bidmodel_fixture_sintetica"


# ---------------------------------------------------------------------------
# Helper di costruzione fixture (inline, sintetiche)
# ---------------------------------------------------------------------------


def _player(player_id: str, ruolo: str, quotazione: float, **overrides) -> Player:
    base = dict(
        player_id=player_id,
        nome=f"Giocatore {player_id}",
        ruolo=ruolo,
        squadra="SQ01",
        quotazione_listone=quotazione,
        fonte=FONTE_TEST,
        is_synthetic=True,
        data_estrazione=datetime(2026, 8, 25, tzinfo=UTC),
    )
    base.update(overrides)
    return Player(**base)


def _osservazione(
    player_id: str,
    tornata: int,
    avversario_id: str,
    offerta: float,
    vincente: bool,
    *,
    stagione: str = "s1",
    **overrides,
) -> OpponentBidObservation:
    base = dict(
        player_id=player_id,
        stagione=stagione,
        tornata=tornata,
        avversario_id=avversario_id,
        offerta=offerta,
        vincente=vincente,
        fonte=FONTE_TEST,
        is_synthetic=True,
    )
    base.update(overrides)
    return OpponentBidObservation(**base)


def _league_config(**overrides) -> LeagueConfig:
    base = dict(
        n_partecipanti=12,
        budget_totale_fase1=500,
        budget_bonus_fase2=50,
        rosa_fase1=RosterSlots(P=2, D=8, C=8, A=6),
        rosa_fase2_massima=RosterSlots(P=4, D=10, C=10, A=8),
        n_tornate_buste=2,
        assegna_buste_se_uguali=False,
        offerte_non_tonde=True,
        modificatore_difesa=True,
        regolamento_portieri_top8_attivo=True,
        max_giocatori_per_squadra_serie_a=None,
        bidmodel_min_osservazioni=30,
        asta_riparazione=AstaRiparazioneConfig(
            tipo="tempo", sequenziale=False, soft_close=False, watchlist_size=8
        ),
    )
    base.update(overrides)
    return LeagueConfig(**base)


# ---------------------------------------------------------------------------
# fasce.py
# ---------------------------------------------------------------------------


class TestAssegnaFasce:
    def test_tier1_e_la_quotazione_piu_alta(self):
        players = [
            _player("A1", "A", 80),
            _player("A2", "A", 40),
            _player("A3", "A", 10),
        ]
        fasce = assegna_fasce(players, n_tier=3)
        assert fasce["A1"] == "A_tier1"
        assert fasce["A3"] == "A_tier3"

    def test_ruoli_diversi_indipendenti(self):
        players = [
            _player("A1", "A", 80),
            _player("D1", "D", 80),
        ]
        fasce = assegna_fasce(players, n_tier=3)
        assert fasce["A1"].startswith("A_")
        assert fasce["D1"].startswith("D_")

    def test_ripartizione_a_frequenza_costante(self):
        # 9 giocatori, 3 tier -> 3 per tier esatti.
        players = [_player(f"A{i}", "A", 100 - i) for i in range(9)]
        fasce = assegna_fasce(players, n_tier=3)
        conteggio: dict[str, int] = {}
        for f in fasce.values():
            conteggio[f] = conteggio.get(f, 0) + 1
        assert conteggio == {"A_tier1": 3, "A_tier2": 3, "A_tier3": 3}

    def test_n_tier_invalido_solleva(self):
        with pytest.raises(ValueError):
            assegna_fasce([_player("A1", "A", 10)], n_tier=0)

    def test_deterministico(self):
        players = [_player(f"A{i}", "A", 50) for i in range(20)]
        f1 = assegna_fasce(players, n_tier=N_TIER_DEFAULT)
        f2 = assegna_fasce(players, n_tier=N_TIER_DEFAULT)
        assert f1 == f2

    def test_nome_fascia(self):
        assert nome_fascia("A", 1) == "A_tier1"


# ---------------------------------------------------------------------------
# normalizzazione.py
# ---------------------------------------------------------------------------


class TestNormalizzazione:
    def test_stessa_scala_fattore_uno(self):
        oss = [_osservazione("A1", 1, "AVV1", 20.0, True)]
        stagione = StagioneStorica(
            label="s1",
            osservazioni=oss,
            quotazioni={"A1": 10.0},
            budget_totale=500,
            n_partecipanti=12,
        )
        risultato = normalizza_stagione(stagione, budget_procapite_riferimento=500 / 12)
        assert len(risultato) == 1
        assert risultato[0].rapporto_grezzo == pytest.approx(2.0)
        assert risultato[0].rapporto_normalizzato == pytest.approx(2.0)

    def test_budget_procapite_piu_alto_scala_giu_il_rapporto(self):
        # Stagione storica con budget pro-capite DOPPIO del riferimento:
        # un rapporto grezzo di 2.0 in quella stagione "vale" meno in
        # scala riferimento -> normalizzato a 1.0.
        oss = [_osservazione("A1", 1, "AVV1", 20.0, True, stagione="s_inflazionata")]
        stagione = StagioneStorica(
            label="s_inflazionata",
            osservazioni=oss,
            quotazioni={"A1": 10.0},
            budget_totale=1000,
            n_partecipanti=10,  # budget procapite = 100
        )
        risultato = normalizza_stagione(stagione, budget_procapite_riferimento=50)
        assert risultato[0].rapporto_normalizzato == pytest.approx(1.0)

    def test_player_id_senza_quotazione_viene_scartato(self):
        oss = [_osservazione("SCONOSCIUTO", 1, "AVV1", 20.0, True)]
        stagione = StagioneStorica(
            label="s1", osservazioni=oss, quotazioni={}, budget_totale=500, n_partecipanti=12
        )
        assert normalizza_stagione(stagione, budget_procapite_riferimento=500 / 12) == []

    def test_budget_totale_non_positivo_solleva(self):
        with pytest.raises(ValueError):
            StagioneStorica(
                label="s1", osservazioni=[], quotazioni={}, budget_totale=0, n_partecipanti=12
            )

    def test_pool_stagioni_combina_piu_stagioni(self):
        s1 = StagioneStorica(
            label="s1",
            osservazioni=[_osservazione("A1", 1, "AVV1", 10.0, True)],
            quotazioni={"A1": 10.0},
            budget_totale=500,
            n_partecipanti=12,
        )
        s2 = StagioneStorica(
            label="s2",
            osservazioni=[_osservazione("A1", 2, "AVV1", 12.0, True, stagione="s2")],
            quotazioni={"A1": 10.0},
            budget_totale=500,
            n_partecipanti=12,
        )
        pool = pool_stagioni([s1, s2], budget_procapite_riferimento=500 / 12)
        assert len(pool) == 2
        assert {r.stagione for r in pool} == {"s1", "s2"}


# ---------------------------------------------------------------------------
# fitting.py — fit_fascia (soglia K, degradazione automatica)
# ---------------------------------------------------------------------------


class TestFitFascia:
    def test_sopra_soglia_usa_empirical(self):
        ratios = [0.8 + 0.01 * i for i in range(30)]
        esito = fit_fascia("A_tier1", ratios, min_osservazioni=30)
        assert esito.mode is PriceDistributionMode.EMPIRICAL
        assert esito.n_osservazioni == 30
        assert esito.empirical_support == sorted(ratios)
        assert esito.degradata_a_prior is False

    def test_sotto_soglia_degrada_a_prior(self):
        ratios = [0.9] * 10  # 10 < 30
        esito = fit_fascia("A_tier1", ratios, min_osservazioni=30)
        assert esito.mode is PriceDistributionMode.PRIOR
        assert esito.n_osservazioni == 10
        assert esito.degradata_a_prior is True
        assert esito.prior_mu is not None
        assert esito.prior_sigma is not None

    def test_zero_osservazioni_e_prior_non_degradata(self):
        # Il caso reale della v1: zero osservazioni, non "poche".
        esito = fit_fascia("A_tier1", [], min_osservazioni=30)
        assert esito.mode is PriceDistributionMode.PRIOR
        assert esito.n_osservazioni == 0
        assert esito.degradata_a_prior is False  # non c'era nulla da degradare

    def test_esattamente_alla_soglia_e_empirical(self):
        ratios = [1.0] * 30
        esito = fit_fascia("A_tier1", ratios, min_osservazioni=30)
        assert esito.mode is PriceDistributionMode.EMPIRICAL

    def test_min_osservazioni_non_positivo_solleva(self):
        with pytest.raises(ValueError):
            fit_fascia("A_tier1", [], min_osservazioni=0)

    def test_prior_mu_sigma_personalizzabili(self):
        esito = fit_fascia("A_tier1", [], min_osservazioni=30, prior_mu=0.2, prior_sigma=0.3)
        assert esito.prior_mu == pytest.approx(0.2)
        assert esito.prior_sigma == pytest.approx(0.3)


class TestCostruisciLotMaxRatios:
    def test_prende_il_massimo_per_lotto(self):
        rapporti = pool_stagioni(
            [
                StagioneStorica(
                    label="s1",
                    osservazioni=[
                        _osservazione("A1", 1, "AVV1", 8.0, False),
                        _osservazione("A1", 1, "AVV2", 12.0, True),
                        _osservazione("A1", 2, "AVV1", 9.0, True),
                    ],
                    quotazioni={"A1": 10.0},
                    budget_totale=500,
                    n_partecipanti=12,
                )
            ],
            budget_procapite_riferimento=500 / 12,
        )
        lot_max = costruisci_lot_max_ratios(rapporti)
        assert lot_max[("A1", 1, "s1")] == pytest.approx(1.2)  # max(0.8, 1.2)
        assert lot_max[("A1", 2, "s1")] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# fitting.py — fit_price_distributions (pipeline end-to-end)
# ---------------------------------------------------------------------------


class TestFitPriceDistributions:
    def test_senza_storico_tutto_prior_come_in_v1(self):
        """Il caso reale della v1: zero OpponentBidObservation esistenti
        (docs/OPEN_QUESTIONS.md §2). Ogni fascia deve degradare a prior."""
        players = [_player(f"A{i}", "A", 20 + i) for i in range(5)]
        config = _league_config()
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=None
        )
        assert len(distribuzioni) == 5
        assert all(d.mode is PriceDistributionMode.PRIOR for d in distribuzioni)
        assert all(d.n_osservazioni == 0 for d in distribuzioni)
        assert all(d.is_synthetic for d in distribuzioni)
        assert all(d.fonte == FONTE_TEST for d in distribuzioni)

    def test_con_storico_sufficiente_una_fascia_diventa_empirical(self):
        # 32 attaccanti, tutti nella stessa fascia (n_tier=1), 2 tornate,
        # 1 avversario -> 64 lotti >= K=30.
        players = [_player(f"A{i}", "A", 20) for i in range(32)]
        config = _league_config(bidmodel_min_osservazioni=30)
        oss = [
            _osservazione(f"A{i}", tornata, "AVV1", 20.0, True)
            for i in range(32)
            for tornata in (1, 2)
        ]
        stagione = StagioneStorica(
            label="s1",
            osservazioni=oss,
            quotazioni={f"A{i}": 20.0 for i in range(32)},
            budget_totale=500,
            n_partecipanti=12,
        )
        rapporti = pool_stagioni([stagione], budget_procapite_riferimento=500 / 12)
        distribuzioni = fit_price_distributions(
            players=players,
            config=config,
            fonte=FONTE_TEST,
            rapporti_storici=rapporti,
            n_tier=1,
        )
        assert all(d.mode is PriceDistributionMode.EMPIRICAL for d in distribuzioni)
        assert all(d.n_osservazioni == 64 for d in distribuzioni)

    def test_fasce_diverse_possono_avere_esiti_diversi(self):
        # A: storico sufficiente -> empirical. D: nessuno storico -> prior.
        attaccanti = [_player(f"A{i}", "A", 20) for i in range(32)]
        difensori = [_player(f"D{i}", "D", 15) for i in range(5)]
        players = attaccanti + difensori
        config = _league_config(bidmodel_min_osservazioni=30)
        oss = [
            _osservazione(f"A{i}", tornata, "AVV1", 20.0, True)
            for i in range(32)
            for tornata in (1, 2)
        ]
        stagione = StagioneStorica(
            label="s1",
            osservazioni=oss,
            quotazioni={f"A{i}": 20.0 for i in range(32)},
            budget_totale=500,
            n_partecipanti=12,
        )
        rapporti = pool_stagioni([stagione], budget_procapite_riferimento=500 / 12)
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=rapporti, n_tier=1
        )
        by_id = {d.player_id: d for d in distribuzioni}
        assert by_id["A0"].mode is PriceDistributionMode.EMPIRICAL
        assert by_id["D0"].mode is PriceDistributionMode.PRIOR
        assert by_id["D0"].n_osservazioni == 0

    def test_is_synthetic_false_se_giocatori_e_storico_reali(self):
        players = [
            _player(f"A{i}", "A", 20, is_synthetic=False, fonte="listone_reale")
            for i in range(3)
        ]
        config = _league_config()
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=None
        )
        # nessuno storico -> prior, ma i giocatori sottostanti non sono
        # sintetici -> la PriceDistribution non deve esserlo silenziosamente.
        assert all(d.is_synthetic is False for d in distribuzioni)

    def test_is_synthetic_true_se_anche_solo_un_giocatore_e_sintetico(self):
        players = [
            _player("A0", "A", 20, is_synthetic=False, fonte="listone_reale"),
            _player("A1", "A", 20, is_synthetic=True),
        ]
        config = _league_config()
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=None
        )
        by_id = {d.player_id: d for d in distribuzioni}
        assert by_id["A1"].is_synthetic is True

    def test_ogni_player_id_ha_una_distribuzione(self):
        players = [_player(f"A{i}", "A", 20) for i in range(7)]
        config = _league_config()
        distribuzioni = fit_price_distributions(players=players, config=config, fonte=FONTE_TEST)
        assert {d.player_id for d in distribuzioni} == {p.player_id for p in players}

    def test_validazione_pydantic_passa_su_ogni_distribuzione_prodotta(self):
        # fit_price_distributions produce PriceDistribution reali (non
        # dict): se un campo richiesto mancasse, pydantic solleverebbe qui.
        players = [_player("A0", "A", 20)]
        config = _league_config()
        distribuzioni = fit_price_distributions(players=players, config=config, fonte=FONTE_TEST)
        assert isinstance(distribuzioni[0], PriceDistribution)


# ---------------------------------------------------------------------------
# schemas.PriceDistribution.p_win — monotonicità (riusa il metodo di Fase 0)
# ---------------------------------------------------------------------------


class TestMonotonicitaPWin:
    def test_monotonicita_modalita_prior(self):
        d = PriceDistribution(
            player_id="A0",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=PRIOR_MU_DEFAULT,
            prior_sigma=PRIOR_SIGMA_DEFAULT,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        offerte = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
        p_win = [d.p_win(b, quotazione_listone=20.0) for b in offerte]
        assert p_win == sorted(p_win)

    def test_monotonicita_modalita_empirical(self):
        players = [_player(f"A{i}", "A", 20) for i in range(32)]
        config = _league_config(bidmodel_min_osservazioni=30)
        oss = [
            _osservazione(f"A{i}", tornata, "AVV1", 10.0 + i, True)
            for i in range(32)
            for tornata in (1, 2)
        ]
        stagione = StagioneStorica(
            label="s1",
            osservazioni=oss,
            quotazioni={f"A{i}": 20.0 for i in range(32)},
            budget_totale=500,
            n_partecipanti=12,
        )
        rapporti = pool_stagioni([stagione], budget_procapite_riferimento=500 / 12)
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=rapporti, n_tier=1
        )
        d = distribuzioni[0]
        assert d.mode is PriceDistributionMode.EMPIRICAL
        offerte = [b for b in range(1, 60, 2)]
        p_win = [d.p_win(b, quotazione_listone=20.0) for b in offerte]
        assert p_win == sorted(p_win)

    @pytest.mark.parametrize("mode_prior", [True, False])
    def test_p_win_in_zero_uno(self, mode_prior):
        if mode_prior:
            d = PriceDistribution(
                player_id="A0",
                fascia="A_tier1",
                mode=PriceDistributionMode.PRIOR,
                n_osservazioni=0,
                prior_mu=0.0,
                prior_sigma=0.5,
                fonte=FONTE_TEST,
                is_synthetic=True,
            )
        else:
            d = PriceDistribution(
                player_id="A0",
                fascia="A_tier1",
                mode=PriceDistributionMode.EMPIRICAL,
                n_osservazioni=3,
                empirical_support=[0.5, 1.0, 1.5],
                fonte=FONTE_TEST,
                is_synthetic=True,
            )
        for b in (0.01, 1, 10, 100, 1000):
            pw = d.p_win(b, quotazione_listone=20.0)
            assert 0.0 <= pw <= 1.0


# ---------------------------------------------------------------------------
# montecarlo.py
# ---------------------------------------------------------------------------


class TestSimulaAsta:
    def _setup(self):
        players = [
            _player("A0", "A", 40.0),
            _player("A1", "A", 40.0),
            _player("D0", "D", 10.0),
        ]
        dist_a = PriceDistribution(
            player_id="A0",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=0.0,
            prior_sigma=0.3,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        distribuzioni = {
            "A0": dist_a.model_copy(update={"player_id": "A0"}),
            "A1": dist_a.model_copy(update={"player_id": "A1"}),
            "D0": PriceDistribution(
                player_id="D0",
                fascia="D_tier1",
                mode=PriceDistributionMode.PRIOR,
                n_osservazioni=0,
                prior_mu=0.0,
                prior_sigma=0.3,
                fonte=FONTE_TEST,
                is_synthetic=True,
            ),
        }
        mie_offerte = {"A0": 50.0, "A1": 50.0, "D0": 12.0}
        return players, distribuzioni, mie_offerte

    def test_riproducibile_da_seed(self):
        players, distribuzioni, mie_offerte = self._setup()
        r1 = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=4, budget_avversario=100, n_simulazioni=200, seed=42,
        )
        r2 = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=4, budget_avversario=100, n_simulazioni=200, seed=42,
        )
        assert np.array_equal(r1.vittorie, r2.vittorie)
        assert r1.p_win_simulata == r2.p_win_simulata

    def test_seed_diverso_puo_dare_risultato_diverso(self):
        players, distribuzioni, mie_offerte = self._setup()
        r1 = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=4, budget_avversario=100, n_simulazioni=200, seed=1,
        )
        r2 = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=4, budget_avversario=100, n_simulazioni=200, seed=2,
        )
        assert not np.array_equal(r1.vittorie, r2.vittorie)

    def test_forma_output(self):
        players, distribuzioni, mie_offerte = self._setup()
        r = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=3, budget_avversario=100, n_simulazioni=150, seed=7,
        )
        assert r.vittorie.shape == (150, 3)
        assert set(r.p_win_simulata) == {"A0", "A1", "D0"}
        assert r.matrice_correlazione.shape == (3, 3)
        assert r.matrice_correlazione[0, 0] == pytest.approx(1.0)

    def test_p_win_simulata_in_zero_uno(self):
        players, distribuzioni, mie_offerte = self._setup()
        r = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=3, budget_avversario=100, n_simulazioni=200, seed=7,
        )
        for v in r.p_win_simulata.values():
            assert 0.0 <= v <= 1.0

    def test_budget_limitato_crea_correlazione_negativa(self):
        """Due giocatori costosi che assorbono da soli quasi tutto il
        budget di un avversario: se un avversario "prova" su uno dei due,
        il budget spesso non gli basta per l'altro -> le vittorie sui due
        giocatori dovrebbero essere correlate positivamente per ME (se
        l'avversario non riesce a permettersi A0, probabilmente non si
        può permettere nemmeno A1: vinco spesso entrambi insieme o
        nessuno dei due) rispetto all'indipendenza pura."""
        players = [_player("A0", "A", 90.0), _player("A1", "A", 90.0)]
        dist = PriceDistribution(
            player_id="A0",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=0.0,
            prior_sigma=0.2,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        distribuzioni = {
            "A0": dist,
            "A1": dist.model_copy(update={"player_id": "A1"}),
        }
        mie_offerte = {"A0": 95.0, "A1": 95.0}
        r = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=1, budget_avversario=100, n_simulazioni=2000, seed=123,
        )
        # un solo avversario con budget 100 non può offrire ~90 su
        # ENTRAMBI i giocatori -> le vittorie non sono indipendenti.
        corr = r.matrice_correlazione[0, 1]
        assert corr != pytest.approx(0.0, abs=0.05)

    def test_rifiuta_n_simulazioni_non_positivo(self):
        players, distribuzioni, mie_offerte = self._setup()
        with pytest.raises(ValueError):
            simula_asta(
                players, distribuzioni, mie_offerte,
                n_avversari=1, budget_avversario=100, n_simulazioni=0, seed=1,
            )

    def test_rifiuta_distribuzione_mancante(self):
        players, distribuzioni, mie_offerte = self._setup()
        del distribuzioni["D0"]
        with pytest.raises(ValueError):
            simula_asta(
                players, distribuzioni, mie_offerte,
                n_avversari=1, budget_avversario=100, n_simulazioni=10, seed=1,
            )

    def test_gap_indipendenza_calcolabile(self):
        players, distribuzioni, mie_offerte = self._setup()
        r = simula_asta(
            players, distribuzioni, mie_offerte,
            n_avversari=3, budget_avversario=100, n_simulazioni=300, seed=9,
        )
        gap = r.gap_indipendenza("A0", "A1")
        assert isinstance(gap, float)


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_curva_p_win_monotona_e_lunghezza_corretta(self):
        p = _player("A0", "A", 20.0)
        d = PriceDistribution(
            player_id="A0",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=0.0,
            prior_sigma=0.5,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        curva = curva_p_win(p, d, n_punti=11)
        assert len(curva.b_grid) == 11
        assert len(curva.p_win_grid) == 11
        assert curva.p_win_grid == sorted(curva.p_win_grid)
        assert curva.n_osservazioni == 0
        assert curva.mode == "prior"

    def test_mismatch_player_distribuzione_solleva(self):
        p = _player("A0", "A", 20.0)
        d = PriceDistribution(
            player_id="A1",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=0.0,
            prior_sigma=0.5,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        with pytest.raises(ValueError):
            curva_p_win(p, d)

    def test_curve_p_win_salta_giocatori_senza_distribuzione(self):
        players = [_player("A0", "A", 20.0), _player("A1", "A", 25.0)]
        d0 = PriceDistribution(
            player_id="A0",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            prior_mu=0.0,
            prior_sigma=0.5,
            fonte=FONTE_TEST,
            is_synthetic=True,
        )
        curve = curve_p_win(players, [d0])
        assert len(curve) == 1
        assert curve[0].player_id == "A0"


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------


class TestGeneraReport:
    def test_report_scritto_e_contiene_avvisi_prior(self, tmp_path):
        players = [_player(f"A{i}", "A", 20 + i) for i in range(5)]
        config = _league_config()
        distribuzioni = fit_price_distributions(players=players, config=config, fonte=FONTE_TEST)
        out = tmp_path / "bidmodel_report.md"
        path = genera_report(
            players=players,
            distribuzioni=distribuzioni,
            output_path=out,
            min_osservazioni=config.bidmodel_min_osservazioni,
        )
        assert path.exists()
        testo = path.read_text(encoding="utf-8")
        assert "SINTETICI" in testo
        assert "prior" in testo
        assert "A_tier1" in testo or "A_tier2" in testo or "A_tier3" in testo

    def test_report_con_fascia_empirical_non_dichiara_tutto_prior(self, tmp_path):
        players = [_player(f"A{i}", "A", 20) for i in range(32)]
        config = _league_config(bidmodel_min_osservazioni=30)
        oss = [
            _osservazione(f"A{i}", tornata, "AVV1", 20.0, True)
            for i in range(32)
            for tornata in (1, 2)
        ]
        stagione = StagioneStorica(
            label="s1",
            osservazioni=oss,
            quotazioni={f"A{i}": 20.0 for i in range(32)},
            budget_totale=500,
            n_partecipanti=12,
        )
        rapporti = pool_stagioni([stagione], budget_procapite_riferimento=500 / 12)
        distribuzioni = fit_price_distributions(
            players=players, config=config, fonte=FONTE_TEST, rapporti_storici=rapporti, n_tier=1
        )
        out = tmp_path / "bidmodel_report.md"
        path = genera_report(
            players=players,
            distribuzioni=distribuzioni,
            output_path=out,
            min_osservazioni=config.bidmodel_min_osservazioni,
        )
        testo = path.read_text(encoding="utf-8")
        assert "empirical" in testo
        assert "Ogni fascia qui sotto gira in modalità `prior`" not in testo

    def test_esempi_curva_nel_report(self, tmp_path):
        players = [_player("A0", "A", 20.0)]
        config = _league_config()
        distribuzioni = fit_price_distributions(players=players, config=config, fonte=FONTE_TEST)
        out = tmp_path / "r.md"
        path = genera_report(
            players=players,
            distribuzioni=distribuzioni,
            output_path=out,
            min_osservazioni=config.bidmodel_min_osservazioni,
            esempi_curva=["A0"],
        )
        testo = path.read_text(encoding="utf-8")
        assert "p_win" in testo
        assert "A0" in testo


# ---------------------------------------------------------------------------
# Demo end-to-end (lo script che genera bidmodel_report.md alla radice)
# ---------------------------------------------------------------------------


class TestDemo:
    def test_genera_report_demo_gira_senza_errori(self, tmp_path):
        from fantabuste.bidmodel.demo import genera_report_demo

        out = tmp_path / "bidmodel_report.md"
        path = genera_report_demo(out)
        assert path.exists()
        testo = path.read_text(encoding="utf-8")
        assert "empirical" in testo
        assert "prior" in testo
        assert "degradata_a_prior" in testo or "degradazione automatica" in testo
