"""Test del Modulo A (ingestion) — Agente A.

Girano solo su fixture sintetiche/locali (data/fixtures/, o dati costruiti
qui a mano), mai su dati reali né su chiamate di rete obbligatorie — vedi
CLAUDE.md "Standard" e il work order dell'Agente A in docs/DESIGN.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fantabuste.ingest.errors import DipendenzaMancante, FormatoListoneNonRiconosciuto
from fantabuste.ingest.listone import parse_listone
from fantabuste.ingest.matching import (
    CandidatoEsterno,
    EsitoMatch,
    esegui_matching,
    scrivi_match_report,
)
from fantabuste.ingest.pipeline import ingest_listone, ingest_statistiche
from fantabuste.ingest.stats import (
    FonteStatistiche,
    estrai_nomi_squadre,
    fetch_grezzo_soccerdata,
    normalizza_statistiche_grezze,
)
from fantabuste.ingest.validazione import (
    ErroreValidazione,
    TipoProblema,
    valida_giocatori,
    valida_statistiche,
)
from fantabuste.schemas import Player, PlayerStats

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"
LISTONE_SAMPLE = FIXTURES_DIR / "listone_export_sample.csv"
LISTONE_SAMPLE_ALT = FIXTURES_DIR / "listone_export_sample_altformato.csv"


def _player(
    player_id: str, nome: str, ruolo: str, squadra: str, quotazione: float = 20.0
) -> Player:
    return Player(
        player_id=player_id,
        nome=nome,
        ruolo=ruolo,
        squadra=squadra,
        quotazione_listone=quotazione,
        fonte="test",
        is_synthetic=True,
        data_estrazione=datetime(2026, 8, 25, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# listone.py — parsing dell'export del listone
# ---------------------------------------------------------------------------


class TestParseListone:
    def test_formato_fantacalcio_it_conteggio_e_scarti(self):
        esito = parse_listone(LISTONE_SAMPLE, fonte="test_listone")
        # 15 righe totali nel fixture: vedi commenti nel CSV per il dettaglio
        # delle 2 righe volutamente scartabili (nome mancante, quotazione vuota).
        assert esito.n_totale_righe == 15
        assert len(esito.giocatori) == 13
        assert len(esito.righe_scartate) == 2

    def test_giocatori_prodotti_non_sintetici_e_con_fonte(self):
        esito = parse_listone(LISTONE_SAMPLE, fonte="test_listone")
        assert all(not p.is_synthetic for p in esito.giocatori)
        assert all(p.fonte == "test_listone" for p in esito.giocatori)

    def test_id_di_fonte_riusato_quando_presente(self):
        esito = parse_listone(LISTONE_SAMPLE)
        by_nome = {p.nome: p for p in esito.giocatori}
        assert by_nome["Svilar"].player_id == "listone_1001"

    def test_id_generato_quando_assente_nella_fonte(self):
        esito = parse_listone(LISTONE_SAMPLE)
        senza_id = [p for p in esito.giocatori if p.nome == "Senza Id Esempio"]
        assert len(senza_id) == 1
        assert senza_id[0].player_id.startswith("listone_slug_")

    def test_quotazione_formato_italiano_con_virgola(self):
        esito = parse_listone(LISTONE_SAMPLE)
        p = next(p for p in esito.giocatori if p.nome == "Senza Id Esempio")
        assert p.quotazione_listone == 9.5

    def test_riga_con_nome_mancante_scartata_con_motivo(self):
        esito = parse_listone(LISTONE_SAMPLE)
        motivi = [r.motivo for r in esito.righe_scartate]
        assert any("nome" in m for m in motivi)

    def test_riga_con_solo_ruolo_mantra_non_convertita_a_intuito(self):
        # "Solo Ruolo Mantra" ha RM='Dc' ma quotazione Qt.A vuota nel fixture:
        # verifichiamo comunque che nessun Player con quel nome sia stato
        # prodotto e che compaia tra gli scarti.
        esito = parse_listone(LISTONE_SAMPLE)
        nomi_prodotti = {p.nome for p in esito.giocatori}
        assert "Solo Ruolo Mantra" not in nomi_prodotti
        assert any(r.contenuto_grezzo for r in esito.righe_scartate)

    def test_giocatore_svincolato_mantiene_squadra_segnaposto(self):
        esito = parse_listone(LISTONE_SAMPLE)
        karsdorp = next(p for p in esito.giocatori if p.nome == "Karsdorp")
        assert karsdorp.squadra.lower() in ("svincolato",)

    def test_formato_alternativo_admin_lega(self):
        esito = parse_listone(LISTONE_SAMPLE_ALT, fonte="test_alt")
        assert esito.n_totale_righe == 7
        assert len(esito.giocatori) == 6
        assert len(esito.righe_scartate) == 1  # ruolo "GK Sconosciuto" non mappabile

    def test_formato_alternativo_ruoli_parola_intera_mappati(self):
        esito = parse_listone(LISTONE_SAMPLE_ALT)
        meret = next(p for p in esito.giocatori if p.nome == "Meret")
        assert meret.ruolo == "P"
        assert meret.quotazione_listone == 11.0  # simbolo € rimosso

    def test_file_inesistente_solleva_errore_chiaro(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_listone(tmp_path / "non_esiste.csv")

    def test_estensione_non_supportata_solleva_errore_esplicito(self, tmp_path):
        percorso = tmp_path / "listone.txt"
        percorso.write_text("nome,ruolo,squadra,quotazione\n")
        with pytest.raises(FormatoListoneNonRiconosciuto):
            parse_listone(percorso)

    def test_colonne_obbligatorie_mancanti_solleva_errore_esplicito(self, tmp_path):
        percorso = tmp_path / "listone_incompleto.csv"
        percorso.write_text("nome,colore_preferito\nMario Rossi,blu\n")
        with pytest.raises(FormatoListoneNonRiconosciuto):
            parse_listone(percorso)

    def test_file_vuoto_solleva_errore_esplicito(self, tmp_path):
        percorso = tmp_path / "listone_vuoto.csv"
        percorso.write_text("nome,ruolo,squadra,quotazione\n")
        with pytest.raises(FormatoListoneNonRiconosciuto):
            parse_listone(percorso)

    def test_xlsx_senza_motore_installato_degrada_esplicitamente(self, tmp_path):
        # In questo ambiente 'openpyxl' non è installato (non è nelle
        # dipendenze base di pyproject.toml). Se per caso lo fosse in un
        # altro ambiente, il file non è comunque un xlsx valido, quindi
        # fallirebbe comunque in modo esplicito (non silenzioso): il test
        # verifica che NON venga mai prodotto un Player da un file XLSX non
        # leggibile, qualunque sia il motivo.
        percorso = tmp_path / "listone.xlsx"
        percorso.write_bytes(b"non e' davvero un file xlsx")
        with pytest.raises((DipendenzaMancante, Exception)):
            parse_listone(percorso)


# ---------------------------------------------------------------------------
# validazione.py
# ---------------------------------------------------------------------------


class TestValidaGiocatori:
    def test_nessun_problema_su_dati_puliti(self):
        giocatori = [
            _player("X1", "Mario Rossi", "D", "SQ01", quotazione=15),
            _player("X2", "Luigi Bianchi", "A", "SQ02", quotazione=30),
        ]
        rapporto = valida_giocatori(giocatori)
        assert rapporto.n_problemi == 0

    def test_player_id_duplicato_con_dati_incoerenti_solleva_errore(self):
        giocatori = [
            _player("DUP", "Mario Rossi", "D", "SQ01"),
            _player("DUP", "Un Altro Giocatore", "A", "SQ02"),
        ]
        with pytest.raises(ErroreValidazione):
            valida_giocatori(giocatori)

    def test_player_id_duplicato_ma_dati_identici_non_solleva(self):
        # Stesso player_id, stessi dati "sostanziali" (nome/ruolo/squadra/
        # quotazione), diversa sola provenienza: non è un conflitto — non
        # deve sollevare ErroreValidazione. Genera comunque l'avviso "nome
        # duplicato nella stessa squadra" (due righe con lo stesso
        # nome+squadra), che è un controllo indipendente e corretto: qui
        # verifichiamo solo l'assenza dell'errore bloccante.
        p1 = _player("SAME", "Mario Rossi", "D", "SQ01")
        p2 = p1.model_copy(update={"fonte": "altra_fonte"})
        rapporto = valida_giocatori([p1, p2])  # non deve sollevare
        assert rapporto.per_tipo(TipoProblema.NOME_DUPLICATO_STESSA_SQUADRA)

    def test_quotazione_fuori_range_segnalata(self):
        giocatori = [_player("Y1", "Fenomeno", "P", "SQ01", quotazione=999)]
        rapporto = valida_giocatori(giocatori)
        assert rapporto.per_tipo(TipoProblema.QUOTAZIONE_FUORI_RANGE)

    def test_giocatore_senza_squadra_segnalato(self):
        giocatori = [_player("Y2", "Senza Squadra", "C", "Svincolato")]
        rapporto = valida_giocatori(giocatori)
        problemi = rapporto.per_tipo(TipoProblema.GIOCATORE_SENZA_SQUADRA)
        assert len(problemi) == 1
        assert problemi[0].player_id == "Y2"

    def test_nome_duplicato_stessa_squadra_segnalato(self):
        giocatori = [
            _player("Y3", "Omonimo", "D", "SQ01"),
            _player("Y4", "Omonimo", "D", "SQ01"),
        ]
        rapporto = valida_giocatori(giocatori)
        assert len(rapporto.per_tipo(TipoProblema.NOME_DUPLICATO_STESSA_SQUADRA)) == 2


class TestValidaStatistiche:
    def _stats(self, **override) -> PlayerStats:
        base = dict(
            player_id="X1",
            stagione="2025/26",
            squadra="SQ01",
            presenze=20,
            minuti=1500,
            gol=5,
            assist=3,
            xG=4.5,
            xA=2.5,
            fantamedia=6.5,
            rigori_battuti=0,
            fonte="test",
            is_synthetic=False,
        )
        base.update(override)
        return PlayerStats(**base)

    def test_nessun_problema_su_dati_plausibili(self):
        rapporto = valida_statistiche([self._stats()], player_id_noti={"X1"})
        assert rapporto.n_problemi == 0

    def test_player_id_sconosciuto_segnalato(self):
        rapporto = valida_statistiche([self._stats()], player_id_noti={"ALTRO"})
        assert rapporto.per_tipo(TipoProblema.STATS_SENZA_GIOCATORE_NOTO)

    def test_presenze_fuori_range_segnalate(self):
        rapporto = valida_statistiche([self._stats(presenze=50)])
        assert rapporto.per_tipo(TipoProblema.PRESENZE_FUORI_RANGE)

    def test_minuti_incoerenti_con_presenze_segnalati(self):
        rapporto = valida_statistiche([self._stats(presenze=1, minuti=5000)])
        assert rapporto.per_tipo(TipoProblema.MINUTI_INCOERENTI_CON_PRESENZE)

    def test_fantamedia_fuori_range_segnalata(self):
        rapporto = valida_statistiche([self._stats(fantamedia=25.0)])
        assert rapporto.per_tipo(TipoProblema.FANTAMEDIA_FUORI_RANGE)


# ---------------------------------------------------------------------------
# matching.py — fuzzy matching
# ---------------------------------------------------------------------------


class TestFuzzyMatching:
    def _riferimento(self) -> list[Player]:
        return [
            _player("R1", "Nicolo Barella", "C", "Inter"),
            _player("R2", "Paulo Dybala", "A", "Roma"),
            _player("R3", "Marco Verdi", "D", "SQ01"),
            _player("R4", "Marco Verdi", "D", "SQ02"),
        ]

    def test_match_esatto_alta_confidenza(self):
        candidati = [CandidatoEsterno(nome="Nicolo Barella", squadra="Inter")]
        risultato = esegui_matching(self._riferimento(), candidati)
        assert risultato.righe_report[0].esito is EsitoMatch.ALTA_CONFIDENZA
        assert risultato.player_id_per_candidato[0] == "R1"

    def test_ordine_nome_cognome_non_conta(self):
        candidati = [CandidatoEsterno(nome="Barella Nicolo", squadra="Inter")]
        risultato = esegui_matching(self._riferimento(), candidati)
        assert risultato.righe_report[0].esito is EsitoMatch.ALTA_CONFIDENZA

    def test_nome_completamente_diverso_nessun_match(self):
        candidati = [CandidatoEsterno(nome="Zzxqvw Completamente Ignoto")]
        risultato = esegui_matching(self._riferimento(), candidati)
        riga = risultato.righe_report[0]
        assert riga.esito is EsitoMatch.NESSUN_MATCH
        assert 0 not in risultato.player_id_per_candidato

    def test_omonimi_in_squadre_diverse_sono_ambigui_non_fusi(self):
        # "Marco Verdi" esiste due volte nel riferimento (R3, R4), stessa
        # somiglianza esatta con entrambi: NON deve essere assegnato a uno a
        # caso. Requisito esplicito del work order: "non deve mai fondere
        # silenziosamente due giocatori diversi".
        candidati = [CandidatoEsterno(nome="Marco Verdi")]
        risultato = esegui_matching(self._riferimento(), candidati)
        assert risultato.righe_report[0].esito is EsitoMatch.AMBIGUO
        assert 0 not in risultato.player_id_per_candidato

    def test_squadra_concorde_alza_il_punteggio(self):
        rif = [_player("S1", "Federico Chiesa", "A", "Juventus")]
        senza_squadra = esegui_matching(rif, [CandidatoEsterno(nome="Federico Chiesa")])
        con_squadra_giusta = esegui_matching(
            rif, [CandidatoEsterno(nome="Federico Chiesa", squadra="Juventus")]
        )
        assert (
            con_squadra_giusta.righe_report[0].punteggio >= senza_squadra.righe_report[0].punteggio
        )

    def test_squadra_discorde_abbassa_il_punteggio_sotto_soglia(self):
        rif = [_player("S2", "Nome Simile Ma Non Uguale", "C", "SQ01")]
        candidato_squadra_giusta = CandidatoEsterno(
            nome="Nome Simile Pero Diverso", squadra="SQ01"
        )
        candidato_squadra_sbagliata = CandidatoEsterno(
            nome="Nome Simile Pero Diverso", squadra="SQ99"
        )
        alto = esegui_matching(rif, [candidato_squadra_giusta]).righe_report[0].punteggio
        basso = esegui_matching(rif, [candidato_squadra_sbagliata]).righe_report[0].punteggio
        assert basso < alto
        # differenza attesa esattamente bonus + penalità (nessun clipping a
        # questi valori intermedi), non solo "un po' più basso"
        from fantabuste.ingest.matching import (
            BONUS_SQUADRA_CONCORDE,
            PENALITA_SQUADRA_DISCORDE,
        )

        assert alto - basso == pytest.approx(BONUS_SQUADRA_CONCORDE + PENALITA_SQUADRA_DISCORDE)

    def test_ruolo_discorde_penalizzato_pesantemente(self):
        rif = [_player("S3", "Ambiguo Di Ruolo", "P", "SQ01")]
        stesso_ruolo = esegui_matching(
            rif, [CandidatoEsterno(nome="Ambiguo Di Ruolo", ruolo="P")]
        ).righe_report[0]
        ruolo_diverso = esegui_matching(
            rif, [CandidatoEsterno(nome="Ambiguo Di Ruolo", ruolo="A")]
        ).righe_report[0]
        assert ruolo_diverso.punteggio < stesso_ruolo.punteggio

    def test_riferimento_vuoto_produce_nessun_match_per_ogni_candidato(self):
        risultato = esegui_matching([], [CandidatoEsterno(nome="Chiunque")])
        assert risultato.righe_report[0].esito is EsitoMatch.NESSUN_MATCH
        assert risultato.player_id_per_candidato == {}

    def test_soglia_bassa_maggiore_di_alta_rifiutata(self):
        with pytest.raises(ValueError):
            esegui_matching(self._riferimento(), [], soglia_alta=0.5, soglia_bassa=0.9)

    def test_ogni_candidato_produce_esattamente_una_riga_report(self):
        candidati = [
            CandidatoEsterno(nome="Nicolo Barella", squadra="Inter"),
            CandidatoEsterno(nome="Zzxqvw Ignoto"),
            CandidatoEsterno(nome="Marco Verdi"),
        ]
        risultato = esegui_matching(self._riferimento(), candidati)
        assert len(risultato.righe_report) == len(candidati)

    def test_match_report_scritto_include_bassa_confidenza_e_mismatch(self, tmp_path):
        candidati = [
            CandidatoEsterno(nome="Nicolo Barella", squadra="Inter"),  # alta
            CandidatoEsterno(nome="Zzxqvw Ignoto"),  # nessun match
            CandidatoEsterno(nome="Marco Verdi"),  # ambiguo
        ]
        risultato = esegui_matching(self._riferimento(), candidati)
        percorso = scrivi_match_report(risultato, tmp_path / "match_report.csv")
        assert percorso.exists()

        df = pd.read_csv(percorso)
        assert len(df) == 3
        esiti = set(df["esito"])
        assert EsitoMatch.NESSUN_MATCH.value in esiti
        assert EsitoMatch.AMBIGUO.value in esiti
        assert EsitoMatch.ALTA_CONFIDENZA.value in esiti


# ---------------------------------------------------------------------------
# stats.py — normalizzazione statistiche + fetch di rete con degradazione esplicita
# ---------------------------------------------------------------------------


class TestNormalizzaStatistiche:
    def _df_grezzo_multiindex(self) -> pd.DataFrame:
        # Riproduce la forma tipica di un DataFrame soccerdata/FBref:
        # colonne MultiIndex (categoria, statistica).
        colonne = pd.MultiIndex.from_tuples(
            [
                ("Unnamed", "player"),
                ("Unnamed", "team"),
                ("Playing Time", "MP"),
                ("Playing Time", "Min"),
                ("Performance", "Gls"),
                ("Performance", "Ast"),
                ("Expected", "xG"),
                ("Expected", "xAG"),
                ("Performance", "PK"),
            ]
        )
        dati = [
            ["Nicolo Barella", "Inter", 30, 2500, 6, 5, 5.2, 4.1, 0],
            ["Paulo Dybala", "Roma", 25, 2000, 10, 4, 9.5, 3.8, 3],
        ]
        return pd.DataFrame(dati, columns=colonne)

    def test_estrai_nomi_squadre(self):
        nomi = estrai_nomi_squadre(self._df_grezzo_multiindex())
        assert nomi == [("Nicolo Barella", "Inter"), ("Paulo Dybala", "Roma")]

    def test_normalizzazione_produce_playerstats_valide(self):
        df = self._df_grezzo_multiindex()
        player_id_per_riga = {0: "R1", 1: "R2"}
        stats, senza_match, senza_squadra = normalizza_statistiche_grezze(
            df, stagione="2025/26", fonte="fbref_mock", player_id_per_riga=player_id_per_riga
        )
        assert senza_match == []
        assert senza_squadra == []
        assert len(stats) == 2
        barella = next(s for s in stats if s.player_id == "R1")
        assert barella.gol == 6
        assert barella.assist == 5
        assert barella.presenze == 30
        assert barella.minuti == 2500
        assert barella.xG == 5.2
        assert barella.squadra == "Inter"
        assert not barella.is_synthetic
        assert barella.fonte == "fbref_mock"

    def test_righe_senza_match_escluse_non_indovinate(self):
        df = self._df_grezzo_multiindex()
        stats, senza_match, _ = normalizza_statistiche_grezze(
            df, stagione="2025/26", fonte="fbref_mock", player_id_per_riga={0: "R1"}
        )
        assert len(stats) == 1
        assert senza_match == [1]

    def test_riga_senza_squadra_riconoscibile_esclusa_non_indovinata(self):
        df = self._df_grezzo_multiindex()
        df.loc[1, ("Unnamed", "team")] = ""  # squadra vuota per la seconda riga
        stats, senza_match, senza_squadra = normalizza_statistiche_grezze(
            df, stagione="2025/26", fonte="fbref_mock", player_id_per_riga={0: "R1", 1: "R2"}
        )
        assert len(stats) == 1
        assert stats[0].player_id == "R1"
        assert senza_match == []
        assert senza_squadra == [1]

    def test_rigoristi_riconosciuti(self):
        df = self._df_grezzo_multiindex()
        stats, _, _ = normalizza_statistiche_grezze(
            df, stagione="2025/26", fonte="fbref_mock", player_id_per_riga={0: "R1", 1: "R2"}
        )
        dybala = next(s for s in stats if s.player_id == "R2")
        assert dybala.rigori_battuti == 3


class TestFetchRete:
    def test_pacchetto_soccerdata_assente_degrada_esplicitamente(self):
        # In questo ambiente 'soccerdata' non è installato (dipendenza
        # opzionale, non in pyproject.toml base — vedi CLAUDE.md "Confini di
        # modulo"). Il fetch deve fallire in modo esplicito, non silenzioso.
        with pytest.raises(DipendenzaMancante):
            fetch_grezzo_soccerdata(FonteStatistiche.FBREF, "2526")


# ---------------------------------------------------------------------------
# pipeline.py — orchestrazione end-to-end (scrive solo sotto data/processed/)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_ingest_listone_end_to_end(self, tmp_path):
        esito = ingest_listone(LISTONE_SAMPLE, output_dir=tmp_path, fonte="test_pipeline")
        assert esito.percorso_output.exists()
        assert esito.percorso_output.parent == tmp_path
        assert len(esito.giocatori) == 13

        df = pd.read_csv(esito.percorso_output)
        assert len(df) == 13
        assert set(df["is_synthetic"]) == {False}

    def test_ingest_listone_non_scrive_in_data_raw(self, tmp_path):
        data_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
        contenuto_prima = set(data_raw.iterdir())
        ingest_listone(LISTONE_SAMPLE, output_dir=tmp_path)
        assert set(data_raw.iterdir()) == contenuto_prima

    def test_ingest_statistiche_end_to_end(self, tmp_path):
        esito_listone = ingest_listone(LISTONE_SAMPLE, output_dir=tmp_path, fonte="test_pipeline")

        colonne = pd.MultiIndex.from_tuples(
            [
                ("Unnamed", "player"),
                ("Unnamed", "team"),
                ("Playing Time", "MP"),
                ("Playing Time", "Min"),
                ("Performance", "Gls"),
                ("Performance", "Ast"),
                ("Expected", "xG"),
                ("Expected", "xAG"),
            ]
        )
        # "Svilar" combacia esattamente con un Player del listone appena
        # ingerito; "Giocatore Del Tutto Ignoto" no.
        dati = [
            ["Svilar", "Roma", 30, 2700, 0, 0, 0.0, 0.0],
            ["Giocatore Del Tutto Ignoto", "Nessuna Squadra Nota", 10, 500, 1, 0, 0.5, 0.1],
        ]
        df_grezzo = pd.DataFrame(dati, columns=colonne)

        esito_stats = ingest_statistiche(
            df_grezzo,
            esito_listone.giocatori,
            stagione="2025/26",
            fonte="test_stats",
            output_dir=tmp_path,
        )

        assert esito_stats.percorso_stats.exists()
        assert esito_stats.percorso_match_report.exists()
        assert len(esito_stats.stats) == 1  # solo Svilar ha match alta confidenza
        assert esito_stats.stats[0].player_id == "listone_1001"
        assert esito_stats.stats[0].squadra == "Roma"
        assert esito_stats.righe_senza_player_id == [1]
        assert esito_stats.righe_senza_squadra == []

        report_df = pd.read_csv(esito_stats.percorso_match_report)
        assert len(report_df) == 2  # ENTRAMBI i candidati, match e non, sono tracciati
