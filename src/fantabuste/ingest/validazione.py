"""Validazione di `Player[]` / `PlayerStats[]` dopo il parsing.

Distingue due categorie, deliberatamente:

- **errori bloccanti** (`ErroreValidazione`, sollevato): violano
  un'invariante che nessun modulo a valle può gestire in sicurezza — oggi
  solo `player_id` duplicato con dati incoerenti (due `Player` diversi con
  lo stesso id rompe ogni join a valle in modo silenzioso).
- **avvisi** (`ProblemaValidazione`, raccolti in `RapportoValidazione`, MAI
  sollevati): condizioni plausibili nella realtà di un listone vero (un
  giocatore svincolato a fine mercato, una quotazione ai margini del range
  atteso) che vanno segnalate per revisione, non bloccate — l'unico modo
  di scoprire un formato-sorgente inatteso è vederlo, non farlo fallire e
  basta.

Nessun range qui è "il" range ufficiale del regolamento (non ce n'è uno in
`config/league.yaml`, che comunque è read-only per l'Agente A): sono soglie
di plausibilità statistica per intercettare errori di parsing (es. una
quotazione letta come 1200 invece di 12.0 per un errore di formato
migliaia/decimali), documentate qui come tali.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from fantabuste.schemas import Player, PlayerStats, Ruolo

# Range di quotazione plausibili per ruolo — ampi apposta: servono a
# intercettare errori grossolani di parsing, non a validare un vero
# regolamento (che non esiste ancora per le quotazioni 2026/27 — vedi
# docs/OPEN_QUESTIONS.md §4).
QUOTAZIONE_RANGE_PLAUSIBILE: dict[Ruolo, tuple[float, float]] = {
    "P": (1, 40),
    "D": (1, 60),
    "C": (1, 90),
    "A": (1, 120),
}

PRESENZE_MASSIME_STAGIONE = 38
FANTAMEDIA_RANGE_PLAUSIBILE = (3.0, 10.0)

_SQUADRE_SVINCOLATO = frozenset({"", "svincolato", "svincolati", "free agent", "sv", "n/a", "-"})


class SeveritaProblema(StrEnum):
    AVVISO = "avviso"


class TipoProblema(StrEnum):
    QUOTAZIONE_FUORI_RANGE = "quotazione_fuori_range"
    GIOCATORE_SENZA_SQUADRA = "giocatore_senza_squadra"
    NOME_DUPLICATO_STESSA_SQUADRA = "nome_duplicato_stessa_squadra"
    PRESENZE_FUORI_RANGE = "presenze_fuori_range"
    MINUTI_INCOERENTI_CON_PRESENZE = "minuti_incoerenti_con_presenze"
    FANTAMEDIA_FUORI_RANGE = "fantamedia_fuori_range"
    STATS_SENZA_GIOCATORE_NOTO = "stats_senza_giocatore_noto"


class ErroreValidazione(Exception):
    """Bloccante: un `player_id` è usato per due `Player` con dati diversi.
    Nessun modulo a valle (join su `player_id`) può funzionare in modo
    sensato in questo caso."""


@dataclass(frozen=True)
class ProblemaValidazione:
    tipo: TipoProblema
    player_id: str | None
    descrizione: str
    severita: SeveritaProblema = SeveritaProblema.AVVISO


@dataclass
class RapportoValidazione:
    problemi: list[ProblemaValidazione] = field(default_factory=list)

    def per_tipo(self, tipo: TipoProblema) -> list[ProblemaValidazione]:
        return [p for p in self.problemi if p.tipo is tipo]

    @property
    def n_problemi(self) -> int:
        return len(self.problemi)


def _verifica_player_id_duplicati(giocatori: list[Player]) -> None:
    visti: dict[str, Player] = {}
    for p in giocatori:
        precedente = visti.get(p.player_id)
        if precedente is None:
            visti[p.player_id] = p
            continue
        if precedente.model_dump(exclude={"fonte", "is_synthetic", "data_estrazione"}) != (
            p.model_dump(exclude={"fonte", "is_synthetic", "data_estrazione"})
        ):
            raise ErroreValidazione(
                f"player_id '{p.player_id}' duplicato con dati incoerenti: "
                f"{precedente.nome!r} ({precedente.squadra}) vs {p.nome!r} ({p.squadra}). "
                "Ogni join a valle su player_id sarebbe ambiguo: correggi la fonte o "
                "il parser prima di procedere."
            )


def valida_giocatori(giocatori: list[Player]) -> RapportoValidazione:
    """Valida un `Player[]` appena parsato. Solleva `ErroreValidazione` per
    player_id duplicati con dati incoerenti; raccoglie il resto come avvisi
    nel `RapportoValidazione` ritornato."""
    _verifica_player_id_duplicati(giocatori)

    rapporto = RapportoValidazione()

    conteggio_nome_squadra: dict[tuple[str, str], int] = {}
    for p in giocatori:
        chiave = (p.nome.strip().lower(), p.squadra.strip().lower())
        conteggio_nome_squadra[chiave] = conteggio_nome_squadra.get(chiave, 0) + 1

    for p in giocatori:
        lo, hi = QUOTAZIONE_RANGE_PLAUSIBILE[p.ruolo]
        if not (lo <= p.quotazione_listone <= hi):
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.QUOTAZIONE_FUORI_RANGE,
                    player_id=p.player_id,
                    descrizione=(
                        f"{p.nome} ({p.ruolo}): quotazione {p.quotazione_listone} fuori dal "
                        f"range plausibile [{lo}, {hi}] per il ruolo — verificare la riga sorgente"
                    ),
                )
            )

        if p.squadra.strip().lower() in _SQUADRE_SVINCOLATO:
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.GIOCATORE_SENZA_SQUADRA,
                    player_id=p.player_id,
                    descrizione=f"{p.nome} ({p.ruolo}): nessuna squadra assegnata nella fonte",
                )
            )

        chiave = (p.nome.strip().lower(), p.squadra.strip().lower())
        if conteggio_nome_squadra[chiave] > 1:
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.NOME_DUPLICATO_STESSA_SQUADRA,
                    player_id=p.player_id,
                    descrizione=(
                        f"{p.nome} compare {conteggio_nome_squadra[chiave]} volte nella stessa "
                        f"squadra ({p.squadra}) con player_id diversi: possibile doppia riga "
                        "per lo stesso giocatore nella fonte"
                    ),
                )
            )

    return rapporto


def valida_statistiche(
    stats: list[PlayerStats], *, player_id_noti: set[str] | None = None
) -> RapportoValidazione:
    """Valida un `PlayerStats[]` appena normalizzato. `player_id_noti`, se
    passato (tipicamente l'insieme dei `player_id` del listone), segnala le
    righe di statistiche che non corrispondono a nessun giocatore noto —
    sintomo tipico di un fuzzy matching incompleto a monte."""
    rapporto = RapportoValidazione()

    for s in stats:
        if player_id_noti is not None and s.player_id not in player_id_noti:
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.STATS_SENZA_GIOCATORE_NOTO,
                    player_id=s.player_id,
                    descrizione=(
                        f"stagione {s.stagione}: player_id '{s.player_id}' non presente "
                        "nell'elenco giocatori noto — controllare il fuzzy matching a monte"
                    ),
                )
            )

        if not (0 <= s.presenze <= PRESENZE_MASSIME_STAGIONE):
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.PRESENZE_FUORI_RANGE,
                    player_id=s.player_id,
                    descrizione=(
                        f"stagione {s.stagione}: {s.presenze} presenze fuori dal range "
                        f"plausibile [0, {PRESENZE_MASSIME_STAGIONE}]"
                    ),
                )
            )

        # margine oltre i 90' per i supplementari
        minuti_massimi_plausibili = PRESENZE_MASSIME_STAGIONE * 120
        if s.minuti > max(s.presenze, 1) * 120 or s.minuti > minuti_massimi_plausibili:
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.MINUTI_INCOERENTI_CON_PRESENZE,
                    player_id=s.player_id,
                    descrizione=(
                        f"stagione {s.stagione}: {s.minuti} minuti non plausibili per "
                        f"{s.presenze} presenze"
                    ),
                )
            )

        lo, hi = FANTAMEDIA_RANGE_PLAUSIBILE
        if not (lo <= s.fantamedia <= hi):
            rapporto.problemi.append(
                ProblemaValidazione(
                    tipo=TipoProblema.FANTAMEDIA_FUORI_RANGE,
                    player_id=s.player_id,
                    descrizione=(
                        f"stagione {s.stagione}: fantamedia {s.fantamedia} fuori dal range "
                        f"plausibile [{lo}, {hi}]"
                    ),
                )
            )

    return rapporto
