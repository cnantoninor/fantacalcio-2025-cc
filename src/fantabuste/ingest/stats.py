"""Fetch e normalizzazione delle statistiche storiche -> `PlayerStats[]`.

Fonte prevista da docs/DESIGN.md (Agente A, punto 2): `soccerdata`
(FBref/Understat, Serie A). Due responsabilità tenute deliberatamente
separate:

1. `fetch_grezzo_soccerdata` — tocca la rete, richiede il pacchetto
   opzionale `soccerdata` (non presente nelle dipendenze base di
   `pyproject.toml`, che è read-only per l'Agente A — vedi CLAUDE.md
   "Confini di modulo"). Degrada in modo **esplicito**, mai silenzioso: se
   il pacchetto manca o la rete/il proxy blocca l'accesso, solleva
   un'eccezione dedicata con un messaggio chiaro sul motivo, invece di
   restituire dati sintetici o una cache scaduta senza dirlo.
2. `normalizza_statistiche_grezze` — pura, non tocca la rete. Prende in
   ingresso un `pandas.DataFrame` già ottenuto (da `soccerdata` o da un
   file già scaricato) e lo normalizza in `PlayerStats[]`. È questa la
   funzione testata contro dati mock/fixture locali (vedi
   `tests/test_ingest.py`), non una chiamata di rete live obbligatoria —
   esattamente come richiesto dal work order dell'Agente A.

Il collegamento tra le righe di statistiche (identificate per nome/squadra
nella fonte esterna) e il `player_id` canonico del listone passa dal fuzzy
matching di `fantabuste.ingest.matching`, non da un id diretto: fonti
diverse quasi mai condividono lo stesso schema di id.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

import pandas as pd

from fantabuste.ingest.errors import DipendenzaMancante, FonteStatisticheNonRaggiungibile
from fantabuste.schemas import PlayerStats

LEGA_SERIE_A = "ITA-Serie A"


class FonteStatistiche(StrEnum):
    FBREF = "fbref"
    UNDERSTAT = "understat"


# Alias delle colonne attese in un DataFrame di statistiche "grezzo", dopo
# l'appiattimento di eventuali MultiIndex (vedi `_appiattisci_colonne`).
# `soccerdata` raggruppa le colonne FBref per categoria (es. Playing Time >
# Min, Performance > Gls): l'appiattimento le rende confrontabili con questi
# alias indipendentemente dal livello superiore.
_ALIAS_NOME = ("player", "nome", "giocatore")
_ALIAS_SQUADRA = ("team", "squadra", "squad")
_ALIAS_PRESENZE = ("mp", "playingtimemp", "presenze", "matchesplayed")
_ALIAS_MINUTI = ("min", "playingtimemin", "minuti", "minutes")
_ALIAS_GOL = ("gls", "performancegls", "gol", "goals")
_ALIAS_ASSIST = ("ast", "performanceast", "assist", "assists")
_ALIAS_XG = ("xg", "expectedxg", "xg")
_ALIAS_XA = ("xag", "xa", "expectedxag", "expectedxa")
_ALIAS_RIGORI = (
    "pk",
    "performancepk",
    "rigoribattuti",
    "penaltiestaken",
    "pkatt",
    "performancepkatt",
)


def _normalizza_header(colonna: str) -> str:
    s = unicodedata.normalize("NFKD", str(colonna)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", s.lower().strip())


def _appiattisci_colonne(df: pd.DataFrame) -> pd.DataFrame:
    """Un DataFrame `soccerdata`/FBref tipicamente ha colonne MultiIndex
    (categoria, statistica), es. ('Performance', 'Gls'). Le appiattisce in
    stringhe singole concatenate ('Performance Gls') così gli alias sotto
    possono confrontarle indipendentemente dal numero di livelli."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        nuove_colonne = []
        for tupla in df.columns.to_flat_index():
            livelli_significativi = [
                str(livello)
                for livello in tupla
                if str(livello)
                and str(livello).lower() != "nan"
                and not str(livello).lower().startswith("unnamed")
            ]
            # Se TUTTI i livelli sono segnaposto (es. colonna non raggruppata
            # in nessuna categoria, tipica di 'player'/'team' in FBref), il
            # nome utile è quello originale senza il prefisso 'Unnamed'.
            nuove_colonne.append(
                " ".join(livelli_significativi)
                if livelli_significativi
                else "_".join(map(str, tupla))
            )
        df.columns = nuove_colonne
    return df


def _trova_colonna(colonne_normalizzate: dict[str, str], alias: tuple[str, ...]) -> str | None:
    inversa: dict[str, str] = {}
    for originale, normalizzata in colonne_normalizzate.items():
        inversa.setdefault(normalizzata, originale)
    for a in alias:
        if a in inversa:
            return inversa[a]
    return None


def _numero(valore: object, *, default: float = 0.0) -> float:
    if valore is None or (isinstance(valore, float) and pd.isna(valore)):
        return default
    try:
        return float(valore)
    except (TypeError, ValueError):
        return default


def normalizza_statistiche_grezze(
    df: pd.DataFrame,
    *,
    stagione: str,
    fonte: str,
    player_id_per_riga: dict[int, str],
) -> tuple[list[PlayerStats], list[int], list[int]]:
    """Normalizza un DataFrame di statistiche grezze in `PlayerStats[]`.

    `player_id_per_riga` collega l'indice di riga del DataFrame (posizione,
    0-based dopo `reset_index`) al `player_id` canonico — tipicamente
    l'output di `fantabuste.ingest.matching.esegui_matching` sui nomi. Le
    righe SENZA un player_id noto (match non risolto, in revisione) vengono
    escluse e i loro indici ritornati separatamente: non produciamo mai una
    `PlayerStats` orfana con un player_id indovinato.

    `PlayerStats.squadra` (la squadra del giocatore IN QUESTA STAGIONE, non
    quella attuale — vedi schemas.py) è obbligatoria nel contratto: una riga
    con player_id risolto ma senza colonna/valore squadra riconoscibile non
    produce comunque una `PlayerStats` (mai una squadra indovinata o vuota),
    e il suo indice va nel terzo elemento restituito, separato dalle righe
    senza player_id — sono due fallimenti diversi, non vanno confusi.

    Ogni `PlayerStats` prodotta ha `is_synthetic=False`.
    """
    df = _appiattisci_colonne(df).reset_index(drop=True)
    colonne_normalizzate = {c: _normalizza_header(c) for c in df.columns}

    col_squadra = _trova_colonna(colonne_normalizzate, _ALIAS_SQUADRA)
    col_presenze = _trova_colonna(colonne_normalizzate, _ALIAS_PRESENZE)
    col_minuti = _trova_colonna(colonne_normalizzate, _ALIAS_MINUTI)
    col_gol = _trova_colonna(colonne_normalizzate, _ALIAS_GOL)
    col_assist = _trova_colonna(colonne_normalizzate, _ALIAS_ASSIST)
    col_xg = _trova_colonna(colonne_normalizzate, _ALIAS_XG)
    col_xa = _trova_colonna(colonne_normalizzate, _ALIAS_XA)
    col_rigori = _trova_colonna(colonne_normalizzate, _ALIAS_RIGORI)

    risultati: list[PlayerStats] = []
    righe_senza_match: list[int] = []
    righe_senza_squadra: list[int] = []

    for indice, riga in df.iterrows():
        player_id = player_id_per_riga.get(int(indice))
        if player_id is None:
            righe_senza_match.append(int(indice))
            continue

        squadra = str(riga.get(col_squadra, "") or "").strip() if col_squadra else ""
        if not squadra:
            righe_senza_squadra.append(int(indice))
            continue

        gol = int(round(_numero(riga.get(col_gol)) if col_gol else 0.0))
        assist = int(round(_numero(riga.get(col_assist)) if col_assist else 0.0))
        presenze = int(round(_numero(riga.get(col_presenze)) if col_presenze else 0.0))
        minuti = int(round(_numero(riga.get(col_minuti)) if col_minuti else 0.0))
        xg = round(_numero(riga.get(col_xg)) if col_xg else 0.0, 2)
        xa = round(_numero(riga.get(col_xa)) if col_xa else 0.0, 2)
        rigori = int(round(_numero(riga.get(col_rigori)) if col_rigori else 0.0))

        # fantamedia non è quasi mai presente in una fonte "expected goals"
        # come FBref/Understat (è una metrica editoriale italiana, non un
        # dato oggettivo di partita): senza una fonte dedicata la
        # approssimiamo dal voto-base 6 aggiustato dal contributo realistico
        # di gol/assist per presenza. Approssimazione dichiarata, non un
        # dato osservato — chi consuma questo campo lo sa da qui.
        bonus_per_presenza = (gol * 3 + assist) / presenze if presenze > 0 else 0.0
        fantamedia = round(6.0 + bonus_per_presenza, 2)

        risultati.append(
            PlayerStats(
                player_id=player_id,
                stagione=stagione,
                squadra=squadra,
                presenze=max(0, presenze),
                minuti=max(0, minuti),
                gol=max(0, gol),
                assist=max(0, assist),
                xG=max(0.0, xg),
                xA=max(0.0, xa),
                fantamedia=fantamedia,
                rigori_battuti=max(0, rigori),
                fonte=fonte,
                is_synthetic=False,
            )
        )

    return risultati, righe_senza_match, righe_senza_squadra


def estrai_nomi_squadre(df: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Estrae (nome, squadra) da un DataFrame di statistiche grezzo, uno per
    riga, nello stesso ordine posizionale che userà
    `normalizza_statistiche_grezze` (appiattimento colonne + reset_index).

    Serve a costruire i `CandidatoEsterno` per `fantabuste.ingest.matching`
    PRIMA della normalizzazione vera e propria: il fuzzy matching lavora sui
    nomi, la normalizzazione ha bisogno del `player_id` già risolto.
    """
    df = _appiattisci_colonne(df).reset_index(drop=True)
    colonne_normalizzate = {c: _normalizza_header(c) for c in df.columns}
    col_nome = _trova_colonna(colonne_normalizzate, _ALIAS_NOME)
    col_squadra = _trova_colonna(colonne_normalizzate, _ALIAS_SQUADRA)

    if col_nome is None:
        raise ValueError(
            f"Nessuna colonna nome riconoscibile nel DataFrame grezzo "
            f"(colonne presenti: {list(df.columns)}). Impossibile costruire "
            "i candidati per il fuzzy matching."
        )

    risultati: list[tuple[str, str | None]] = []
    for _, riga in df.iterrows():
        nome = str(riga.get(col_nome, "") or "").strip()
        squadra = str(riga.get(col_squadra, "") or "").strip() if col_squadra else None
        risultati.append((nome, squadra or None))
    return risultati


def fetch_grezzo_soccerdata(
    fonte: FonteStatistiche,
    stagione: str,
    *,
    lega: str = LEGA_SERIE_A,
) -> pd.DataFrame:
    """Scarica statistiche stagionali via `soccerdata`. Tocca la rete.

    Solleva `DipendenzaMancante` se il pacchetto `soccerdata` non è
    installato (opzionale, non nelle dipendenze base — vedi il docstring del
    modulo) e `FonteStatisticheNonRaggiungibile` se la rete/il proxy
    impedisce il download. Nessun fallback silenzioso a dati sintetici o a
    una cache scaduta in nessuno dei due casi.
    """
    try:
        import soccerdata as sd  # type: ignore[import-not-found]
    except ImportError as e:
        raise DipendenzaMancante(
            "Il pacchetto 'soccerdata' non è installato in questo ambiente. "
            "Non è nelle dipendenze base di pyproject.toml (read-only per "
            "l'Agente A): segnalato nel report finale, va deciso se "
            "aggiungerlo come dipendenza opzionale del progetto. "
            "Nessun fallback silenzioso a dati sintetici."
        ) from e

    try:
        if fonte is FonteStatistiche.FBREF:
            reader = sd.FBref(leagues=lega, seasons=stagione)
            return reader.read_player_season_stats(stat_type="standard")
        if fonte is FonteStatistiche.UNDERSTAT:
            reader = sd.Understat(leagues=lega, seasons=stagione)
            return reader.read_player_season_stats()
        raise ValueError(f"Fonte statistiche non gestita: {fonte}")
    except (DipendenzaMancante, ValueError):
        raise
    except Exception as e:  # rete/proxy bloccato, dominio irraggiungibile, ecc.
        raise FonteStatisticheNonRaggiungibile(
            f"Impossibile ottenere le statistiche da {fonte.value} per la "
            f"stagione {stagione} (lega {lega}): {type(e).__name__}: {e}. "
            "Questo ambiente instrada il traffico HTTPS in uscita attraverso "
            "un proxy che potrebbe bloccare il dominio della fonte — vedi la "
            "nota nell'environment del progetto. Nessuna degradazione "
            "silenziosa a dati sintetici: riprova quando la fonte è "
            "raggiungibile, o fornisci un DataFrame già scaricato a "
            "`normalizza_statistiche_grezze`."
        ) from e
