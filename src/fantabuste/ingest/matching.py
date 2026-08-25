"""Fuzzy matching dei nomi tra fonti diverse (listone <-> fonte statistiche).

Questo è, per esperienza documentata in docs/DESIGN.md (Agente A, punto 3),
il punto in cui questi progetti falliscono più spesso: fondere silenziosamente
due giocatori omonimi/simili ma diversi (es. due "Rossi" in squadre diverse, o
varianti di trascrizione di un nome straniero) inquina ogni fase a valle senza
lasciare traccia dell'errore.

Regola d'oro di questo modulo: **nessun match entra in produzione senza essere
tracciato**. Ogni candidato esterno finisce nel `match_report.csv`, con il suo
esito (`alta_confidenza`, `bassa_confidenza`, `ambiguo`, `nessun_match`). Solo
`alta_confidenza` (non ambiguo) viene collegato automaticamente a un
`player_id`; tutto il resto è per revisione manuale — mai un default silenzioso
sull'ipotesi migliore.

Le soglie di confidenza sono parametri dell'algoritmo di matching, non del
regolamento di lega: non hanno posto in `config/league.yaml` (che è
read-only per l'Agente A) e restano quindi argomenti espliciti con default
qui documentati, non "numeri magici" nascosti.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Any

from fantabuste.schemas import Player, Ruolo

# Soglie di default: sopra SOGLIA_ALTA_DEFAULT un match è considerato
# sufficientemente sicuro per essere applicato senza revisione (salvo
# ambiguità — vedi MARGINE_AMBIGUITA_DEFAULT). Sotto SOGLIA_BASSA_DEFAULT il
# candidato è considerato "nessun match" plausibile nel listone di
# riferimento.
SOGLIA_ALTA_DEFAULT = 0.90
SOGLIA_BASSA_DEFAULT = 0.65
# Se il miglior punteggio e il secondo miglior punteggio (su riferimenti
# diversi) distano meno di questo margine, il match è ambiguo: due o più
# giocatori del listone sono candidati plausibili altrettanto buoni, e
# scegliere il primo silenziosamente rischierebbe di fondere due persone
# diverse. Va sempre in revisione manuale, mai auto-applicato.
MARGINE_AMBIGUITA_DEFAULT = 0.03

BONUS_SQUADRA_CONCORDE = 0.05
PENALITA_SQUADRA_DISCORDE = 0.10
PENALITA_RUOLO_DISCORDE = 0.30


class EsitoMatch(StrEnum):
    ALTA_CONFIDENZA = "alta_confidenza"
    BASSA_CONFIDENZA = "bassa_confidenza"
    AMBIGUO = "ambiguo"
    NESSUN_MATCH = "nessun_match"


@dataclass(frozen=True)
class CandidatoEsterno:
    """Un'identità da un'altra fonte (es. una riga di statistiche FBref) da
    riconciliare con un `Player` del listone. `payload` porta con sé
    qualunque dato grezzo serva a chi chiama dopo il match (es. la riga
    originale di statistiche), senza che questo modulo debba conoscerne la
    struttura."""

    nome: str
    squadra: str | None = None
    ruolo: Ruolo | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RigaMatchReport:
    candidato_nome: str
    candidato_squadra: str | None
    esito: EsitoMatch
    player_id_riferimento: str | None
    nome_riferimento: str | None
    squadra_riferimento: str | None
    punteggio: float
    punteggio_secondo_migliore: float | None
    note: str


@dataclass
class RisultatoMatching:
    """`player_id_per_candidato` contiene SOLO i match ad alta confidenza e
    non ambigui — è la mappa sicura da usare per collegare dati esterni a un
    `player_id`. `righe_report` copre invece OGNI candidato, compresi i
    mismatch e le bassa-confidenza, per la revisione manuale (DoD: mai
    ometterli dal report)."""

    player_id_per_candidato: dict[int, str]  # indice in `candidati` -> player_id
    righe_report: list[RigaMatchReport]

    @property
    def n_alta_confidenza(self) -> int:
        return sum(1 for r in self.righe_report if r.esito is EsitoMatch.ALTA_CONFIDENZA)

    @property
    def n_da_rivedere(self) -> int:
        return sum(
            1
            for r in self.righe_report
            if r.esito in (EsitoMatch.BASSA_CONFIDENZA, EsitoMatch.AMBIGUO)
        )


def _normalizza_nome(nome: str) -> str:
    """Rimuove accenti/punteggiatura e riordina i token alfabeticamente, così
    'Rossi Mario' e 'Mario Rossi' producono la stessa chiave di confronto."""
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    token = sorted(t for t in s.split() if t)
    return " ".join(token)


def _similarita(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalizza_nome(a), _normalizza_nome(b)).ratio()


def _normalizza_squadra(squadra: str) -> str:
    """Come `_normalizza_nome` ma SENZA rimuovere le cifre: i nomi di
    giocatore raramente ne contengono, i codici squadra sì (es. le fixture
    sintetiche 'SQ01'..'SQ20' di docs/DESIGN.md, o sigle come 'AC1908').
    Usare `_normalizza_nome` anche qui confonderebbe 'SQ01' e 'SQ99'."""
    s = unicodedata.normalize("NFKD", squadra).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _punteggio(candidato: CandidatoEsterno, riferimento: Player) -> float:
    punteggio = _similarita(candidato.nome, riferimento.nome)

    if candidato.squadra and riferimento.squadra:
        if _normalizza_squadra(candidato.squadra) == _normalizza_squadra(riferimento.squadra):
            punteggio = min(1.0, punteggio + BONUS_SQUADRA_CONCORDE)
        else:
            punteggio = max(0.0, punteggio - PENALITA_SQUADRA_DISCORDE)

    if candidato.ruolo and candidato.ruolo != riferimento.ruolo:
        punteggio = max(0.0, punteggio - PENALITA_RUOLO_DISCORDE)

    return punteggio


def esegui_matching(
    riferimento: list[Player],
    candidati: list[CandidatoEsterno],
    *,
    soglia_alta: float = SOGLIA_ALTA_DEFAULT,
    soglia_bassa: float = SOGLIA_BASSA_DEFAULT,
    margine_ambiguita: float = MARGINE_AMBIGUITA_DEFAULT,
) -> RisultatoMatching:
    """Confronta ogni `CandidatoEsterno` contro l'elenco `riferimento`
    (tipicamente il listone di Modulo A) e classifica l'esito.

    Non fonde mai silenziosamente: solo i match `alta_confidenza` non
    ambigui finiscono in `player_id_per_candidato`. Tutto il resto va
    rivisto a mano tramite `match_report.csv` (vedi `scrivi_match_report`).
    """
    if soglia_bassa > soglia_alta:
        raise ValueError("soglia_bassa non può superare soglia_alta")

    player_id_per_candidato: dict[int, str] = {}
    righe: list[RigaMatchReport] = []

    for indice, candidato in enumerate(candidati):
        punteggi = [(_punteggio(candidato, rif), rif) for rif in riferimento]
        punteggi.sort(key=lambda coppia: coppia[0], reverse=True)

        if not punteggi:
            righe.append(
                RigaMatchReport(
                    candidato_nome=candidato.nome,
                    candidato_squadra=candidato.squadra,
                    esito=EsitoMatch.NESSUN_MATCH,
                    player_id_riferimento=None,
                    nome_riferimento=None,
                    squadra_riferimento=None,
                    punteggio=0.0,
                    punteggio_secondo_migliore=None,
                    note="elenco di riferimento vuoto",
                )
            )
            continue

        migliore_punteggio, migliore_rif = punteggi[0]
        secondo_punteggio = punteggi[1][0] if len(punteggi) > 1 else None

        ambiguo = (
            secondo_punteggio is not None
            and migliore_punteggio - secondo_punteggio < margine_ambiguita
            and secondo_punteggio >= soglia_bassa
        )

        if migliore_punteggio < soglia_bassa:
            esito = EsitoMatch.NESSUN_MATCH
            note = "nessun riferimento sopra la soglia minima"
        elif ambiguo:
            esito = EsitoMatch.AMBIGUO
            note = (
                f"almeno due riferimenti con punteggio ravvicinato "
                f"(margine < {margine_ambiguita}): revisione manuale necessaria "
                "per evitare di fondere due giocatori diversi"
            )
        elif migliore_punteggio >= soglia_alta:
            esito = EsitoMatch.ALTA_CONFIDENZA
            note = ""
            player_id_per_candidato[indice] = migliore_rif.player_id
        else:
            esito = EsitoMatch.BASSA_CONFIDENZA
            note = "sotto la soglia di auto-match: revisione manuale consigliata"

        righe.append(
            RigaMatchReport(
                candidato_nome=candidato.nome,
                candidato_squadra=candidato.squadra,
                esito=esito,
                player_id_riferimento=migliore_rif.player_id,
                nome_riferimento=migliore_rif.nome,
                squadra_riferimento=migliore_rif.squadra,
                punteggio=round(migliore_punteggio, 4),
                punteggio_secondo_migliore=(
                    round(secondo_punteggio, 4) if secondo_punteggio is not None else None
                ),
                note=note,
            )
        )

    return RisultatoMatching(player_id_per_candidato=player_id_per_candidato, righe_report=righe)


def scrivi_match_report(risultato: RisultatoMatching, percorso: str | Path) -> Path:
    """Scrive `match_report.csv`: una riga per OGNI candidato (match sicuri
    inclusi, per trasparenza), cosicché mismatch e bassa-confidenza siano
    elencati esplicitamente per revisione — mai solo i match riusciti."""
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    intestazione = [
        "candidato_nome",
        "candidato_squadra",
        "esito",
        "player_id_riferimento",
        "nome_riferimento",
        "squadra_riferimento",
        "punteggio",
        "punteggio_secondo_migliore",
        "note",
    ]
    with percorso.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(intestazione)
        for riga in risultato.righe_report:
            writer.writerow(
                [
                    riga.candidato_nome,
                    riga.candidato_squadra or "",
                    riga.esito.value,
                    riga.player_id_riferimento or "",
                    riga.nome_riferimento or "",
                    riga.squadra_riferimento or "",
                    riga.punteggio,
                    riga.punteggio_secondo_migliore
                    if riga.punteggio_secondo_migliore is not None
                    else "",
                    riga.note,
                ]
            )
    return percorso
