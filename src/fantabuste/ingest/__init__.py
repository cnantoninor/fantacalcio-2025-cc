"""Modulo A — Ingestion & normalizzazione (Agente A).

Vedi docs/DESIGN.md, work order Agente A, per lo scope completo. Espone:

- `parse_listone` — export listone (CSV/XLSX) -> `Player[]`.
- `fetch_grezzo_soccerdata` / `normalizza_statistiche_grezze` — statistiche
  storiche -> `PlayerStats[]`, con degradazione esplicita se la rete non è
  raggiungibile.
- `esegui_matching` / `scrivi_match_report` — fuzzy matching dei nomi tra
  fonti, mai una fusione silenziosa.
- `valida_giocatori` / `valida_statistiche` — range plausibili, duplicati,
  giocatori senza squadra.
- `ingest_listone` / `ingest_statistiche` — orchestrazione end-to-end che
  scrive sotto `data/processed/` (mai in `data/raw/`, read-only).
"""

from fantabuste.ingest.errors import (
    DipendenzaMancante,
    FonteStatisticheNonRaggiungibile,
    FormatoListoneNonRiconosciuto,
    IngestError,
    RigaListoneScartata,
)
from fantabuste.ingest.listone import RigaScartata, RisultatoParseListone, parse_listone
from fantabuste.ingest.matching import (
    CandidatoEsterno,
    EsitoMatch,
    RigaMatchReport,
    RisultatoMatching,
    esegui_matching,
    scrivi_match_report,
)
from fantabuste.ingest.pipeline import (
    EsitoIngestListone,
    EsitoIngestStatistiche,
    ingest_listone,
    ingest_statistiche,
)
from fantabuste.ingest.stats import (
    FonteStatistiche,
    estrai_nomi_squadre,
    fetch_grezzo_soccerdata,
    normalizza_statistiche_grezze,
)
from fantabuste.ingest.validazione import (
    ErroreValidazione,
    ProblemaValidazione,
    RapportoValidazione,
    TipoProblema,
    valida_giocatori,
    valida_statistiche,
)

__all__ = [
    "CandidatoEsterno",
    "DipendenzaMancante",
    "ErroreValidazione",
    "EsitoIngestListone",
    "EsitoIngestStatistiche",
    "EsitoMatch",
    "FonteStatistiche",
    "FonteStatisticheNonRaggiungibile",
    "FormatoListoneNonRiconosciuto",
    "IngestError",
    "ProblemaValidazione",
    "RapportoValidazione",
    "RigaListoneScartata",
    "RigaMatchReport",
    "RigaScartata",
    "RisultatoMatching",
    "RisultatoParseListone",
    "TipoProblema",
    "esegui_matching",
    "estrai_nomi_squadre",
    "fetch_grezzo_soccerdata",
    "ingest_listone",
    "ingest_statistiche",
    "normalizza_statistiche_grezze",
    "parse_listone",
    "scrivi_match_report",
    "valida_giocatori",
    "valida_statistiche",
]
