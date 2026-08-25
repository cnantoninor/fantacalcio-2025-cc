"""Orchestrazione end-to-end del Modulo A: listone -> statistiche -> match -> validazione.

Vincolo esplicito (docs/DESIGN.md, work order Agente A): `data/raw/` è
**read-only**, ogni trasformazione scrive un file NUOVO sotto
`data/processed/`. Questo modulo non apre mai un file in `data/raw/` in
scrittura — legge solo il percorso passato da chi chiama (che tipicamente
sarà sotto `data/raw/`, ma la funzione non lo assume: accetta qualunque
percorso).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fantabuste.ingest.listone import RigaScartata, parse_listone
from fantabuste.ingest.matching import (
    CandidatoEsterno,
    RisultatoMatching,
    esegui_matching,
    scrivi_match_report,
)
from fantabuste.ingest.stats import estrai_nomi_squadre, normalizza_statistiche_grezze
from fantabuste.ingest.validazione import (
    RapportoValidazione,
    valida_giocatori,
    valida_statistiche,
)
from fantabuste.schemas import Player, PlayerStats

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


@dataclass
class EsitoIngestListone:
    giocatori: list[Player]
    righe_scartate: list[RigaScartata]
    validazione: RapportoValidazione
    percorso_output: Path


def ingest_listone(
    percorso_listone: str | Path,
    *,
    output_dir: str | Path = PROCESSED_DIR,
    fonte: str | None = None,
    nome_file_output: str = "players.csv",
) -> EsitoIngestListone:
    """Parsa e valida un export del listone, scrive `Player[]` normalizzati
    in `<output_dir>/<nome_file_output>` (mai in `data/raw/`)."""
    output_dir = Path(output_dir)
    esito_parse = parse_listone(percorso_listone, fonte=fonte)
    rapporto = valida_giocatori(esito_parse.giocatori)

    output_dir.mkdir(parents=True, exist_ok=True)
    percorso_output = output_dir / nome_file_output
    pd.DataFrame([p.model_dump() for p in esito_parse.giocatori]).to_csv(
        percorso_output, index=False
    )

    return EsitoIngestListone(
        giocatori=esito_parse.giocatori,
        righe_scartate=esito_parse.righe_scartate,
        validazione=rapporto,
        percorso_output=percorso_output,
    )


@dataclass
class EsitoIngestStatistiche:
    stats: list[PlayerStats]
    matching: RisultatoMatching
    validazione: RapportoValidazione
    righe_senza_player_id: list[int] = field(default_factory=list)
    percorso_stats: Path | None = None
    percorso_match_report: Path | None = None


def ingest_statistiche(
    df_grezzo: pd.DataFrame,
    riferimento_giocatori: list[Player],
    *,
    stagione: str,
    fonte: str,
    output_dir: str | Path = PROCESSED_DIR,
    nome_file_stats: str = "player_stats.csv",
    nome_file_match_report: str = "match_report.csv",
    **soglie_matching: float,
) -> EsitoIngestStatistiche:
    """Collega un DataFrame di statistiche grezzo (es. da
    `fantabuste.ingest.stats.fetch_grezzo_soccerdata`, o già scaricato) ai
    `player_id` del listone via fuzzy matching, normalizza in
    `PlayerStats[]`, valida e scrive `player_stats.csv` + `match_report.csv`
    sotto `output_dir`.

    Le righe senza un match ad alta confidenza NON producono `PlayerStats`
    (mai un player_id indovinato) — restano in `matching.righe_report` /
    `righe_senza_player_id` per revisione manuale, esattamente il
    comportamento richiesto dal work order dell'Agente A.
    """
    output_dir = Path(output_dir)

    nomi_squadre = estrai_nomi_squadre(df_grezzo)
    candidati = [CandidatoEsterno(nome=nome, squadra=squadra) for nome, squadra in nomi_squadre]

    risultato_matching = esegui_matching(riferimento_giocatori, candidati, **soglie_matching)

    player_id_per_riga = dict(risultato_matching.player_id_per_candidato)
    stats, righe_senza_match = normalizza_statistiche_grezze(
        df_grezzo, stagione=stagione, fonte=fonte, player_id_per_riga=player_id_per_riga
    )

    id_noti = {p.player_id for p in riferimento_giocatori}
    rapporto_validazione = valida_statistiche(stats, player_id_noti=id_noti)

    output_dir.mkdir(parents=True, exist_ok=True)
    percorso_stats = output_dir / nome_file_stats
    pd.DataFrame([s.model_dump() for s in stats]).to_csv(percorso_stats, index=False)
    percorso_report = scrivi_match_report(risultato_matching, output_dir / nome_file_match_report)

    return EsitoIngestStatistiche(
        stats=stats,
        matching=risultato_matching,
        validazione=rapporto_validazione,
        righe_senza_player_id=righe_senza_match,
        percorso_stats=percorso_stats,
        percorso_match_report=percorso_report,
    )
