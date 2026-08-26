"""Test del Modulo B (Valutazione).

Girano solo su fixture esistenti (`fantabuste.fixtures.genera_fixture_giocatori`)
o su piccoli dataset sintetici costruiti qui inline — mai su
`data/fixtures/players.csv`/`player_stats.csv` modificati, mai su dati reali.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fantabuste.config import LeagueConfig
from fantabuste.fixtures import genera_fixture_giocatori
from fantabuste.schemas import MetodoProiezione, Player, PlayerProjection, PlayerStats, RosterSlots
from fantabuste.valuation import baseline as baseline_mod
from fantabuste.valuation import ml as ml_mod
from fantabuste.valuation import report as report_mod
from fantabuste.valuation import vorp as vorp_mod
from fantabuste.valuation.features import costruisci_features
from fantabuste.valuation.pipeline import esegui_valutazione

DATA_ESTRAZIONE = datetime(2026, 8, 25, tzinfo=UTC)


def _player(**overrides) -> Player:
    base = dict(
        player_id="P0001",
        nome="Test",
        ruolo="A",
        squadra="SQ01",
        quotazione_listone=10,
        fonte="test",
        is_synthetic=True,
        data_estrazione=DATA_ESTRAZIONE,
    )
    base.update(overrides)
    return Player(**base)


def _stats(**overrides) -> PlayerStats:
    base = dict(
        player_id="P0001",
        stagione="2024/25",
        squadra="SQ01",
        presenze=30,
        minuti=2500,
        gol=10,
        assist=5,
        xG=9.0,
        xA=4.5,
        fantamedia=7.0,
        rigori_battuti=0,
        fonte="test",
        is_synthetic=True,
    )
    base.update(overrides)
    return PlayerStats(**base)


def _config(**overrides) -> LeagueConfig:
    raw = dict(
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
        asta_riparazione=dict(
            tipo="tempo", sequenziale=False, soft_close=False, timer_secondi=None, watchlist_size=8
        ),
    )
    raw.update(overrides)
    return LeagueConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_costruisci_features_unisce_player_e_stats():
    players = [_player(player_id="P1")]
    stats = [
        _stats(player_id="P1", stagione="2024/25", minuti=900, gol=9, fantamedia=6.5),
        _stats(player_id="P1", stagione="2025/26", minuti=1800, gol=18, fantamedia=7.5),
    ]
    features = costruisci_features(players, stats)
    assert len(features) == 1
    f = features[0]
    assert f.n_stagioni_osservate == 2
    assert f.stagioni[0].stagione == "2024/25"
    assert f.stagioni[1].stagione == "2025/26"
    assert f.minuti_totali == 2700
    # gol/90 stagione1 = 9 * 90 / 900 = 0.9
    assert f.stagioni[0].gol_per90 == pytest.approx(0.9)


def test_costruisci_features_giocatore_senza_stats():
    players = [_player(player_id="P1")]
    features = costruisci_features(players, [])
    assert len(features) == 1
    assert features[0].stagioni == ()
    assert features[0].minuti_totali == 0


# ---------------------------------------------------------------------------
# Baseline: shrinkage, intervalli, prob_titolarita
# ---------------------------------------------------------------------------


def test_shrinkage_pochi_minuti_regredisce_verso_media_ruolo():
    """L'esempio letterale di docs/DESIGN.md: un attaccante con 400 minuti e
    fantamedia 8 non deve risultare proiettato come un attaccante da 8."""
    players = [
        _player(player_id="P1", ruolo="A"),  # 400 minuti, fantamedia 8
        _player(player_id="P2", ruolo="A"),  # tanti minuti, fantamedia 6 (tipico)
        _player(player_id="P3", ruolo="A"),
        _player(player_id="P4", ruolo="A"),
    ]
    stats = [
        _stats(player_id="P1", stagione="2025/26", minuti=400, presenze=6, fantamedia=8.0),
        _stats(player_id="P2", stagione="2025/26", minuti=3000, presenze=34, fantamedia=6.0),
        _stats(player_id="P3", stagione="2025/26", minuti=3000, presenze=34, fantamedia=6.0),
        _stats(player_id="P4", stagione="2025/26", minuti=3000, presenze=34, fantamedia=6.0),
    ]
    features = costruisci_features(players, stats)
    proiezioni, medie_ruolo = baseline_mod.proietta_baseline_tutti(features)
    by_id = {p.player_id: p for p in proiezioni}

    # La fantamedia pesata grezza di P1 resta 8 (nessuna manipolazione a
    # monte), ma lo shrinkage deve tenere il punteggio ben sotto la media di
    # ruolo pesata dalla sola prob_titolarita di un giocatore da 400 minuti.
    assert by_id["P1"].fantamedia_pesata_grezza == pytest.approx(8.0)
    media_ruolo_a = medie_ruolo["A"].media
    assert media_ruolo_a < 8.0
    # Il punto chiave: la fantamedia "pre-titolarità" di P1 deve essere
    # sostanzialmente più vicina alla media di ruolo che a 8, essendo il
    # campione di minuti minuscolo (400 << K_SHRINKAGE_MINUTI=900).
    punteggio_pre_prob, _, _ = baseline_mod.punteggio_atteso_pre_titolarita(
        [f for f in features if f.player_id == "P1"][0], medie_ruolo
    )
    assert punteggio_pre_prob < 7.0  # ben lontano da 8, shrinkato verso ~6
    assert punteggio_pre_prob > media_ruolo_a  # ma non shrinkage totale


def test_shrinkage_totale_senza_stagioni_osservate():
    players = [_player(player_id="P1", ruolo="D"), _player(player_id="P2", ruolo="D")]
    stats = [_stats(player_id="P2", stagione="2025/26", minuti=3000, fantamedia=6.2)]
    features = costruisci_features(players, stats)
    proiezioni, medie_ruolo = baseline_mod.proietta_baseline_tutti(features)
    p1 = next(p for p in proiezioni if p.player_id == "P1")
    assert p1.prob_titolarita == 0.0
    assert p1.punti_attesi == 0.0  # nessun minuto storico -> prob_titolarita 0
    assert p1.confidence_flag == "bassa"


def test_intervalli_sempre_coerenti_su_fixture():
    players, stats = genera_fixture_giocatori(seed=42)
    features = costruisci_features(players, stats)
    proiezioni, _ = baseline_mod.proietta_baseline_tutti(features)
    for p in proiezioni:
        assert p.punti_attesi_lo <= p.punti_attesi <= p.punti_attesi_hi
        assert p.punti_attesi_lo >= 0.0


def test_bassa_confidenza_produce_intervallo_piu_largo():
    players = [
        _player(player_id="TITOLARE", ruolo="C"),
        _player(player_id="PANCHINARO", ruolo="C"),
    ]
    stats = [
        _stats(player_id="TITOLARE", stagione="2024/25", minuti=3000, fantamedia=6.5),
        _stats(player_id="TITOLARE", stagione="2025/26", minuti=3000, fantamedia=6.6),
        _stats(player_id="PANCHINARO", stagione="2024/25", minuti=150, presenze=4, fantamedia=6.5),
        _stats(player_id="PANCHINARO", stagione="2025/26", minuti=150, presenze=4, fantamedia=6.6),
    ]
    features = costruisci_features(players, stats)
    proiezioni, _ = baseline_mod.proietta_baseline_tutti(features)
    by_id = {p.player_id: p for p in proiezioni}
    largo_titolare = by_id["TITOLARE"].punti_attesi_hi - by_id["TITOLARE"].punti_attesi_lo
    largo_panchinaro = by_id["PANCHINARO"].punti_attesi_hi - by_id["PANCHINARO"].punti_attesi_lo
    assert by_id["TITOLARE"].confidence_flag == "alta"
    assert by_id["PANCHINARO"].confidence_flag == "bassa"
    assert largo_panchinaro > largo_titolare


def test_media_pesata_privilegia_stagione_recente():
    players = [_player(player_id="P1", ruolo="D")]
    stats = [
        _stats(player_id="P1", stagione="2024/25", minuti=3000, fantamedia=5.0),
        _stats(player_id="P1", stagione="2025/26", minuti=3000, fantamedia=7.0),
    ]
    features = costruisci_features(players, stats)
    media = baseline_mod._media_pesata_stagioni(features[0])
    punto_medio = (5.0 + 7.0) / 2
    assert media > punto_medio  # più vicino a 7 (recente) che alla media semplice


# ---------------------------------------------------------------------------
# VORP
# ---------------------------------------------------------------------------


def test_vorp_replacement_level_su_ranking_noto():
    # 3 partecipanti, 1 slot per D -> rank di replacement = 3. Con 5
    # difensori ordinati per punti_attesi, il replacement è il 3° migliore
    # (0-indexed: idx=3 -> il 4° valore della lista ordinata desc).
    cfg = _config(n_partecipanti=3, rosa_fase1=RosterSlots(P=1, D=1, C=1, A=1))
    punti = {"D1": 10.0, "D2": 8.0, "D3": 6.0, "D4": 4.0, "D5": 2.0}
    ruoli = {k: "D" for k in punti}
    vorp, replacement = vorp_mod.calcola_vorp(punti, ruoli, cfg)
    assert replacement["D"].rank == 3
    assert replacement["D"].valore == pytest.approx(4.0)  # 4° valore (idx=3)
    assert replacement["D"].degenerato is False
    assert vorp["D1"] == pytest.approx(10.0 - 4.0)
    assert vorp["D5"] == pytest.approx(2.0 - 4.0)


def test_vorp_replacement_rank_scala_con_n_partecipanti():
    cfg_piccola = _config(n_partecipanti=2, rosa_fase1=RosterSlots(P=1, D=1, C=1, A=1))
    cfg_grande = _config(n_partecipanti=10, rosa_fase1=RosterSlots(P=1, D=1, C=1, A=1))
    rank_piccola = vorp_mod.calcola_replacement_rank(cfg_piccola)
    rank_grande = vorp_mod.calcola_replacement_rank(cfg_grande)
    assert rank_piccola["D"] == 2
    assert rank_grande["D"] == 10


def test_vorp_caso_degenerato_campione_piccolo():
    # rank teorico 12*8=96 ma solo 3 difensori disponibili: degenerato=True.
    cfg = _config()  # default 12 partecipanti, 8 D in rosa_fase1
    punti = {"D1": 10.0, "D2": 8.0, "D3": 6.0}
    ruoli = {k: "D" for k in punti}
    _, replacement = vorp_mod.calcola_vorp(punti, ruoli, cfg)
    assert replacement["D"].degenerato is True
    assert replacement["D"].valore == pytest.approx(6.0)  # peggior valore disponibile


def test_vorp_override_rank_e_valore():
    cfg = _config(n_partecipanti=12, rosa_fase1=RosterSlots(P=2, D=8, C=8, A=6))
    punti = {"D1": 10.0, "D2": 8.0, "D3": 6.0}
    ruoli = {k: "D" for k in punti}
    _, replacement = vorp_mod.calcola_vorp(
        punti, ruoli, cfg, override_valore={"D": 5.0}
    )
    assert replacement["D"].valore == pytest.approx(5.0)
    assert replacement["D"].degenerato is False


# ---------------------------------------------------------------------------
# ML: vincoli di feature, rifiuto automatico, accettazione
# ---------------------------------------------------------------------------


def test_ml_massimo_8_feature():
    assert len(ml_mod.FEATURE_NAMES) == ml_mod.MASSIMO_FEATURE == 8


def test_ml_non_valutato_con_pochi_giocatori():
    players = [_player(player_id=f"P{i}", ruolo="D") for i in range(6)]
    stats = []
    for i in range(6):
        stats.append(_stats(player_id=f"P{i}", stagione="2024/25", fantamedia=6.0))
        stats.append(_stats(player_id=f"P{i}", stagione="2025/26", fantamedia=6.1))
    features = costruisci_features(players, stats)
    risultato = ml_mod.valuta_sfidante_ml(features)
    assert risultato.accettato is False
    assert "insufficienti" in risultato.motivo.lower()


def test_ml_rifiutato_su_rumore_puro():
    """Con target puramente casuale (nessuna relazione con le feature), l'ML
    non deve battere il baseline con margine ampio: il codice deve
    rifiutarlo da solo, non limitarsi a segnalarlo."""
    import random

    rng = random.Random(7)
    players = []
    stats = []
    for i in range(80):
        pid = f"N{i:03d}"
        players.append(_player(player_id=pid, ruolo="C"))
        fm1 = 6.0 + rng.uniform(-0.3, 0.3)
        fm2 = 6.0 + rng.uniform(-3.0, 3.0)  # rumore indipendente da fm1
        stats.append(_stats(player_id=pid, stagione="2024/25", minuti=2500, fantamedia=fm1))
        stats.append(_stats(player_id=pid, stagione="2025/26", minuti=2500, fantamedia=fm2))
    features = costruisci_features(players, stats)
    risultato = ml_mod.valuta_sfidante_ml(features)
    assert risultato.accettato is False


def test_ml_accettato_su_segnale_forte_fixture():
    """Sulle fixture (qualità latente che guida sia le feature sia il
    target) l'ML ha un segnale reale da sfruttare e deve battere il
    baseline con margine — verifica che l'accettazione NON sia sempre
    rifiutata a prescindere."""
    players, stats = genera_fixture_giocatori(seed=42)
    features = costruisci_features(players, stats)
    risultato = ml_mod.valuta_sfidante_ml(features)
    assert risultato.n_osservazioni > 0
    assert risultato.mae_ml < risultato.mae_baseline


def test_ml_margine_configurabile_puo_far_rifiutare_anche_un_miglioramento_reale():
    players, stats = genera_fixture_giocatori(seed=42)
    features = costruisci_features(players, stats)
    risultato_permissivo = ml_mod.valuta_sfidante_ml(features, margine_miglioramento_minimo=0.01)
    risultato_severo = ml_mod.valuta_sfidante_ml(features, margine_miglioramento_minimo=0.99)
    assert risultato_permissivo.accettato is True
    assert risultato_severo.accettato is False


# ---------------------------------------------------------------------------
# Pipeline end-to-end e guardrail sui dati sintetici
# ---------------------------------------------------------------------------


def test_pipeline_baseline_produce_proiezioni_valide_su_fixture():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=False)
    assert len(risultato.proiezioni) == len(players)
    assert risultato.metodo_produzione == MetodoProiezione.BASELINE
    assert all(isinstance(p, PlayerProjection) for p in risultato.proiezioni)
    assert all(p.metodo == MetodoProiezione.BASELINE for p in risultato.proiezioni)
    assert all(p.is_synthetic for p in risultato.proiezioni)  # fixture = tutte sintetiche
    assert all(p.fonte for p in risultato.proiezioni)


def test_pipeline_is_synthetic_mai_default_silenzioso_a_false():
    """Un giocatore reale (is_synthetic=False) con statistiche sintetiche
    deve comunque risultare is_synthetic=True: la dipendenza sintetica si
    propaga, non sparisce per errore di default."""
    players = [_player(player_id="P1", ruolo="A", is_synthetic=False, fonte="listone_reale")]
    stats = [
        _stats(
            player_id="P1",
            stagione="2024/25",
            fantamedia=6.5,
            is_synthetic=True,
            fonte="fixture",
        )
    ]
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=False)
    assert risultato.proiezioni[0].is_synthetic is True


def test_pipeline_giocatore_e_stats_reali_produce_proiezione_non_sintetica():
    players = [_player(player_id="P1", ruolo="A", is_synthetic=False, fonte="listone_reale")]
    stats = [
        _stats(
            player_id="P1",
            stagione="2024/25",
            fantamedia=6.5,
            is_synthetic=False,
            fonte="fbref_reale",
        )
    ]
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=False)
    assert risultato.proiezioni[0].is_synthetic is False


def test_pipeline_enable_ml_ricade_su_baseline_se_rifiutato():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    # margine impossibile da superare -> ML sempre rifiutato
    risultato = esegui_valutazione(
        players, stats, cfg, enable_ml=True, margine_ml=0.999999
    )
    assert risultato.risultato_ml is not None
    assert risultato.risultato_ml.accettato is False
    assert risultato.metodo_produzione == MetodoProiezione.BASELINE
    assert all(p.metodo == MetodoProiezione.BASELINE for p in risultato.proiezioni)


def test_pipeline_enable_ml_usa_ml_se_accettato():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=True, margine_ml=0.05)
    assert risultato.risultato_ml is not None
    assert risultato.risultato_ml.accettato is True
    assert risultato.metodo_produzione == MetodoProiezione.ML
    assert all(p.metodo == MetodoProiezione.ML for p in risultato.proiezioni)
    # Gli intervalli restano coerenti anche con la proiezione ML
    for p in risultato.proiezioni:
        assert p.punti_attesi_lo <= p.punti_attesi <= p.punti_attesi_hi


def test_pipeline_ml_disabilitato_non_valuta_sfidante():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=False)
    assert risultato.risultato_ml is None
    assert risultato.ml_non_disponibile_motivo is None


def test_pipeline_vorp_vari_con_n_partecipanti():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg_piccola = _config(n_partecipanti=2)
    cfg_grande = _config(n_partecipanti=12)
    r_piccola = esegui_valutazione(players, stats, cfg_piccola, enable_ml=False)
    r_grande = esegui_valutazione(players, stats, cfg_grande, enable_ml=False)
    assert r_piccola.replacement["D"].rank != r_grande.replacement["D"].rank


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_contiene_sezioni_obbligatorie(tmp_path):
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=True, margine_ml=0.10)
    md = risultato.report_markdown
    assert "Perché questo risultato è debole" in md
    assert "MAE" in md
    assert "RMSE" in md
    assert "Sfidante ML" in md
    assert "VORP" in md
    assert str(len(players)) in md  # n. osservazioni riportato

    path = report_mod.scrivi_report(md, tmp_path / "valuation_report.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == md


def test_report_segnala_tutti_sintetici():
    players, stats = genera_fixture_giocatori(seed=42)
    cfg = _config()
    risultato = esegui_valutazione(players, stats, cfg, enable_ml=False)
    assert "sintetic" in risultato.report_markdown.lower()
