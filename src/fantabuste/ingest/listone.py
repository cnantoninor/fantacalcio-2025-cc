"""Parser per l'export del listone (quotazioni) -> `Player[]`.

Le quotazioni reali 2026/27 non sono ancora disponibili (l'unico file su
Drive è del 2018 — vedi docs/LEAGUE_CONTEXT.md §9 e docs/OPEN_QUESTIONS.md
§4). Il formato più diffuso e pubblicamente documentato è quello del file
"Quotazioni Fantacalcio" pubblicato ogni anno da Fantacalcio.it
(scaricabile liberamente, non dietro autenticazione, non la piattaforma
leghe.fantacalcio.it il cui scraping è vietato — vedi CLAUDE.md):

    Id;R;RM;Nome;Squadra;Qt.A;Qt.I;Diff.;Qt.A M;Qt.I M;Diff.M;FVM;FVM M

dove `R` è il ruolo "classico" (P/D/C/A, lo stesso schema della nostra
lega) e `RM` il ruolo "Mantra" (schema diverso, a grana più fine: non lo
convertiamo mai in classico per non inventare un dato — vedi
`_estrai_ruolo`). L'export dell'admin di lega (sezione "Gestione Rose") ha
in generale un set di colonne più semplice (nome/ruolo/squadra/prezzo).

Dato che il formato reale potrebbe differire leggermente da un anno
all'altro o tra fonte e fonte, questo parser non si aggancia a nomi di
colonna fissi: normalizza gli header e li confronta con un elenco di alias
noti per ciascun campo obbligatorio. Se un campo obbligatorio non ha
nessuna colonna riconoscibile, il parse fallisce in modo esplicito
(`FormatoListoneNonRiconosciuto`) invece di produrre `Player` con dati
mancanti spacciati per validi.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from fantabuste.ingest.errors import (
    DipendenzaMancante,
    FormatoListoneNonRiconosciuto,
)
from fantabuste.schemas import Player, Ruolo

# ---------------------------------------------------------------------------
# Alias delle colonne, per campo obbligatorio. Chiavi e valori già passati da
# `_normalizza_header` (minuscolo, senza spazi/punteggiatura) al confronto.
# ---------------------------------------------------------------------------

_ALIAS_ID: tuple[str, ...] = ("id", "codice", "codiceid", "playerid")
_ALIAS_RUOLO: tuple[str, ...] = ("r", "ruolo", "ruoloclassico", "role", "rol")
_ALIAS_NOME: tuple[str, ...] = ("nome", "calciatore", "giocatore", "player", "name")
_ALIAS_SQUADRA: tuple[str, ...] = ("squadra", "team", "sq", "club")
# Quotazione: preferiamo la corrente (Qt.A / prezzo) alla iniziale (Qt.I),
# l'ordine della tupla è l'ordine di preferenza in `_prima_colonna_alias`.
_ALIAS_QUOTAZIONE: tuple[str, ...] = (
    "qta",
    "quotazioneattuale",
    "quotazione",
    "prezzo",
    "valore",
    "qti",
    "fvm",
)

# Mappa ruoli "classici" alternativi -> Ruolo della lega. Il ruolo Mantra
# (colonna RM: Por/Ds/Dd/Dc/B/E/M/C/W/T/A/Pc) NON è in questa mappa: non è
# una conversione 1:1 col ruolo classico e indovinarla produrrebbe un dato
# sbagliato spacciato per vero — una riga con solo RM viene scartata, non
# convertita a intuito.
_ALIAS_VALORE_RUOLO: dict[str, Ruolo] = {
    "p": "P",
    "por": "P",
    "portiere": "P",
    "gk": "P",
    "d": "D",
    "dif": "D",
    "difensore": "D",
    "df": "D",
    "c": "C",
    "cen": "C",
    "centrocampista": "C",
    "mf": "C",
    "a": "A",
    "att": "A",
    "attaccante": "A",
    "fw": "A",
}

_SQUADRE_SVINCOLATO: frozenset[str] = frozenset(
    {"", "svincolato", "svincolati", "free agent", "sv", "n/a", "nan", "none", "-"}
)


def _normalizza_header(colonna: str) -> str:
    """'Qt.A' -> 'qta', 'Nome ' -> 'nome', 'Ruolo_Classico' -> 'ruoloclassico'."""
    s = unicodedata.normalize("NFKD", str(colonna)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    return re.sub(r"[^a-z0-9]", "", s)


def _prima_colonna_alias(
    colonne_normalizzate: dict[str, str], alias: tuple[str, ...]
) -> str | None:
    """Ritorna il nome ORIGINALE della prima colonna del file che, una volta
    normalizzata, combacia con un alias — nell'ordine di preferenza di `alias`."""
    inversa: dict[str, str] = {}  # normalizzata -> originale (prima occorrenza)
    for originale, normalizzata in colonne_normalizzate.items():
        inversa.setdefault(normalizzata, originale)
    for a in alias:
        if a in inversa:
            return inversa[a]
    return None


def _normalizza_stringa(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _estrai_ruolo(valore_grezzo: object) -> Ruolo | None:
    chiave = _normalizza_stringa(valore_grezzo).lower()
    chiave = re.sub(r"[^a-z]", "", chiave)
    return _ALIAS_VALORE_RUOLO.get(chiave)


def _estrai_quotazione(valore_grezzo: object) -> float | None:
    """Accetta sia '12', '12.5' sia '12,5' (formato italiano) sia '€12'."""
    s = _normalizza_stringa(valore_grezzo)
    if not s:
        return None
    s = s.replace("€", "").replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        valore = float(s)
    except ValueError:
        return None
    if valore <= 0:
        return None
    return valore


def _genera_player_id(id_grezzo: object, nome: str, ruolo: str, squadra: str) -> str:
    id_normalizzato = _normalizza_stringa(id_grezzo)
    if id_normalizzato:
        return f"listone_{id_normalizzato}"
    # Fallback deterministico se la fonte non ha una colonna Id stabile:
    # slug di nome+ruolo+squadra. Non è stabile tra stagioni quanto un vero
    # Id di fonte, ma è deterministico e riproducibile all'interno dello
    # stesso file.
    slug = _normalizza_header(f"{nome}{ruolo}{squadra}")
    return f"listone_slug_{slug}"


@dataclass
class RigaScartata:
    indice_riga: int
    motivo: str
    contenuto_grezzo: dict = field(default_factory=dict)


@dataclass
class RisultatoParseListone:
    """Esito del parse: i `Player` validi più le righe scartate con motivo,
    così una revisione manuale sa esattamente cosa è stato perso e perché
    (mai uno scarto silenzioso)."""

    giocatori: list[Player]
    righe_scartate: list[RigaScartata]

    @property
    def n_totale_righe(self) -> int:
        return len(self.giocatori) + len(self.righe_scartate)


def _leggi_tabella(percorso: Path) -> pd.DataFrame:
    suffisso = percorso.suffix.lower()
    if suffisso == ".csv":
        # sep=None + engine="python" fa sniffing automatico tra ',' e ';'
        # (il file Fantacalcio.it storicamente usa ';').
        return pd.read_csv(percorso, sep=None, engine="python", dtype=str)
    if suffisso in (".xlsx", ".xls"):
        try:
            return pd.read_excel(percorso, dtype=str)
        except ImportError as e:
            raise DipendenzaMancante(
                f"Lettura di {percorso.name} richiede il motore per file Excel "
                "(es. 'openpyxl'), non presente in questo ambiente e non incluso "
                "nelle dipendenze di pyproject.toml (read-only per l'Agente A — "
                "vedi CLAUDE.md 'Confini di modulo'). Segnalato nel report finale: "
                "va deciso se aggiungerlo come dipendenza opzionale del progetto. "
                "In alternativa, esporta il listone in formato CSV."
            ) from e
    raise FormatoListoneNonRiconosciuto(
        f"Estensione non supportata: '{suffisso}' (file: {percorso}). Formati "
        "supportati: .csv, .xlsx, .xls."
    )


def parse_listone(
    percorso: str | Path,
    *,
    fonte: str | None = None,
    data_estrazione: datetime | None = None,
) -> RisultatoParseListone:
    """Parsa un export del listone (CSV o XLSX) in `Player[]`.

    Robusto a variazioni di formato: gli header sono confrontati contro un
    elenco di alias noti (vedi cima del file), non nomi di colonna fissi.
    Una riga con un campo obbligatorio non ricostruibile viene scartata e
    riportata in `RisultatoParseListone.righe_scartate` — non fa fallire
    l'intero parse, ma non produce nemmeno un `Player` con dati inventati.

    Ogni `Player` prodotto ha `is_synthetic=False`: è un dato con fonte
    primaria (il file passato), non una fixture — vedi `SyntheticRecord`.
    """
    percorso = Path(percorso)
    if not percorso.exists():
        raise FileNotFoundError(f"Listone non trovato: {percorso}")

    df = _leggi_tabella(percorso)
    if df.empty:
        raise FormatoListoneNonRiconosciuto(f"{percorso}: file vuoto (nessuna riga).")

    colonne_normalizzate = {c: _normalizza_header(c) for c in df.columns}

    col_ruolo = _prima_colonna_alias(colonne_normalizzate, _ALIAS_RUOLO)
    col_nome = _prima_colonna_alias(colonne_normalizzate, _ALIAS_NOME)
    col_squadra = _prima_colonna_alias(colonne_normalizzate, _ALIAS_SQUADRA)
    col_quotazione = _prima_colonna_alias(colonne_normalizzate, _ALIAS_QUOTAZIONE)
    col_id = _prima_colonna_alias(colonne_normalizzate, _ALIAS_ID)

    mancanti = [
        etichetta
        for etichetta, col in (
            ("ruolo", col_ruolo),
            ("nome", col_nome),
            ("squadra", col_squadra),
            ("quotazione", col_quotazione),
        )
        if col is None
    ]
    if mancanti:
        raise FormatoListoneNonRiconosciuto(
            f"{percorso}: nessuna colonna riconoscibile per {mancanti}. "
            f"Colonne presenti nel file: {list(df.columns)}. "
            "Aggiungi un alias in fantabuste.ingest.listone se il formato è "
            "legittimo ma non ancora coperto."
        )

    fonte_effettiva = fonte or f"listone:{percorso.name}"
    estrazione_effettiva = data_estrazione or datetime.now(tz=UTC)

    giocatori: list[Player] = []
    righe_scartate: list[RigaScartata] = []

    for indice_riga, riga in df.iterrows():
        grezzo = riga.to_dict()
        nome = _normalizza_stringa(grezzo.get(col_nome))
        squadra = _normalizza_stringa(grezzo.get(col_squadra))
        ruolo = _estrai_ruolo(grezzo.get(col_ruolo))
        quotazione = _estrai_quotazione(grezzo.get(col_quotazione))

        if not nome:
            righe_scartate.append(RigaScartata(int(indice_riga), "nome mancante o vuoto", grezzo))
            continue
        if ruolo is None:
            righe_scartate.append(
                RigaScartata(
                    int(indice_riga),
                    f"ruolo '{grezzo.get(col_ruolo)!r}' non mappabile a P/D/C/A "
                    "(un ruolo Mantra da solo non viene convertito a intuito)",
                    grezzo,
                )
            )
            continue
        if quotazione is None:
            righe_scartate.append(
                RigaScartata(
                    int(indice_riga),
                    f"quotazione '{grezzo.get(col_quotazione)!r}' non numerica o <= 0",
                    grezzo,
                )
            )
            continue

        player_id = _genera_player_id(
            grezzo.get(col_id) if col_id else None, nome, ruolo, squadra
        )

        try:
            giocatori.append(
                Player(
                    player_id=player_id,
                    nome=nome,
                    ruolo=ruolo,
                    squadra=squadra if squadra else "SVINCOLATO",
                    quotazione_listone=quotazione,
                    fonte=fonte_effettiva,
                    is_synthetic=False,
                    data_estrazione=estrazione_effettiva,
                )
            )
        except ValidationError as e:
            righe_scartate.append(
                RigaScartata(int(indice_riga), f"schema Player rifiutato: {e}", grezzo)
            )

    return RisultatoParseListone(giocatori=giocatori, righe_scartate=righe_scartate)
