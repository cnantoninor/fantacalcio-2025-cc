"""VORP (Value Over Replacement Player) — docs/DESIGN.md, Agente B, punto 6.

Definizione: per ogni ruolo, il "replacement level" è il `punti_attesi` del
miglior giocatore ancora disponibile una volta che tutte le squadre di lega
hanno riempito i loro slot di quel ruolo con i migliori giocatori disponibili
(un modello a mercato "efficiente" di libro paga, non l'esito osservato di
un'asta reale — vedi limiti sotto). Con `n_partecipanti` squadre e `slot`
posti per ruolo, il rank di replacement è `n_partecipanti * slot`: i primi
`n_partecipanti * slot` giocatori del ruolo (ordinati per punti_attesi
decrescente) sono "rosterabili" da qualche squadra, il successivo è il
replacement.

`VORP_i = punti_attesi_i - replacement_level(ruolo_i)`

**Limiti noti (da documentare sempre nel report, non nel codice soltanto)**:
- Il replacement level qui è calcolato sulla popolazione di giocatori
  disponibile a B (fixture sintetiche in v1, listone reale in produzione),
  non da un mercato osservato: con un campione piccolo o sbilanciato per
  ruolo il rank di replacement può cadere oltre la fine della lista
  disponibile (vedi `_replacement_level_singolo_ruolo`, caso degenerato
  gestito esplicitamente, mai in silenzio).
- Il mercato dell'asta busta chiusa non è liquido nel senso classico (niente
  prezzo di mercato continuo, 2 sole tornate): il VORP qui è un ranking
  relativo di valore atteso, non una previsione di prezzo — quella è compito
  del Modulo C (`PriceDistribution`) e del Modulo D (`BidPlan`), non di B.
- v1 usa `rosa_fase1` (2-8-8-6, la composizione obbligatoria entro il 7
  settembre — docs/OPEN_QUESTIONS.md §0) come base del replacement level per
  default, non `rosa_fase2_massima`: gli slot di fase 2 sono opzionali e
  aggiungerli al denominatore sovrastimerebbe quanti giocatori "buoni" per
  ruolo servono davvero. Override esplicito disponibile per chi vuole
  valutare la fase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantabuste.config import LeagueConfig
from fantabuste.schemas import RosterSlots, Ruolo

RUOLI: tuple[Ruolo, ...] = ("P", "D", "C", "A")


@dataclass(frozen=True)
class ReplacementLevel:
    ruolo: str
    rank: int
    """Posizione (1-indexed) nel ranking di ruolo che definisce il
    replacement: n_partecipanti * slot_per_ruolo, prima di eventuale override."""
    valore: float
    n_disponibili_nel_ruolo: int
    degenerato: bool
    """True se `rank` supera il numero di giocatori disponibili nel ruolo:
    il replacement level è stato preso dal peggior giocatore disponibile
    invece che dalla posizione teorica — segnale che il campione è troppo
    piccolo per quel ruolo (tipico con fixture ridotte o listoni incompleti)."""


def calcola_replacement_rank(
    config: LeagueConfig,
    rosa: RosterSlots | None = None,
    override_rank: dict[str, int] | None = None,
) -> dict[str, int]:
    """Rank (1-indexed) di replacement per ruolo: n_partecipanti * slot.

    `rosa` di default è `config.rosa_fase1` (vedi docstring di modulo).
    `override_rank` permette di forzare il rank per uno o più ruoli
    (DESIGN.md: "con override manuale possibile") — es. per modellare un
    numero di titolari per ruolo diverso dagli slot di rosa nudi e crudi.
    """
    base = rosa or config.rosa_fase1
    rank = {ruolo: config.n_partecipanti * getattr(base, ruolo) for ruolo in RUOLI}
    if override_rank:
        rank.update(override_rank)
    return rank


def _replacement_level_singolo_ruolo(
    ruolo: str, valori_ordinati_desc: list[float], rank: int
) -> ReplacementLevel:
    n = len(valori_ordinati_desc)
    if rank <= 0:
        raise ValueError(f"rank di replacement non valido per ruolo {ruolo}: {rank}")
    idx = rank  # 0-indexed: il rank-esimo giocatore (1-indexed) è rosterato,
    # l'indice `rank` (0-indexed) è il primo NON rosterato = il replacement.
    if idx < n:
        return ReplacementLevel(
            ruolo=ruolo,
            rank=rank,
            valore=valori_ordinati_desc[idx],
            n_disponibili_nel_ruolo=n,
            degenerato=False,
        )
    # Campione troppo piccolo per questo ruolo rispetto al rank teorico:
    # degrada al peggior giocatore disponibile, dichiarato esplicitamente.
    valore = valori_ordinati_desc[-1] if valori_ordinati_desc else 0.0
    return ReplacementLevel(
        ruolo=ruolo, rank=rank, valore=valore, n_disponibili_nel_ruolo=n, degenerato=True
    )


def calcola_replacement_level(
    punti_attesi_per_id: dict[str, float],
    ruolo_per_id: dict[str, str],
    config: LeagueConfig,
    rosa: RosterSlots | None = None,
    override_rank: dict[str, int] | None = None,
    override_valore: dict[str, float] | None = None,
) -> dict[str, ReplacementLevel]:
    """Calcola il replacement level per ciascuno dei 4 ruoli.

    `override_valore` bypassa completamente il calcolo per ruoli specifici
    (override manuale diretto sul valore, non solo sul rank) — usato es. per
    incollare un replacement level noto da fuori (mercato di riparazione,
    stagione precedente), coerente con "override manuale possibile" di
    DESIGN.md.
    """
    rank_per_ruolo = calcola_replacement_rank(config, rosa=rosa, override_rank=override_rank)

    valori_per_ruolo: dict[str, list[float]] = {r: [] for r in RUOLI}
    for pid, punti in punti_attesi_per_id.items():
        ruolo = ruolo_per_id[pid]
        if ruolo in valori_per_ruolo:
            valori_per_ruolo[ruolo].append(punti)

    out: dict[str, ReplacementLevel] = {}
    for ruolo in RUOLI:
        if override_valore and ruolo in override_valore:
            valori_ordinati = sorted(valori_per_ruolo[ruolo], reverse=True)
            out[ruolo] = ReplacementLevel(
                ruolo=ruolo,
                rank=rank_per_ruolo[ruolo],
                valore=override_valore[ruolo],
                n_disponibili_nel_ruolo=len(valori_ordinati),
                degenerato=False,
            )
            continue
        valori_ordinati = sorted(valori_per_ruolo[ruolo], reverse=True)
        out[ruolo] = _replacement_level_singolo_ruolo(
            ruolo, valori_ordinati, rank_per_ruolo[ruolo]
        )
    return out


def calcola_vorp(
    punti_attesi_per_id: dict[str, float],
    ruolo_per_id: dict[str, str],
    config: LeagueConfig,
    rosa: RosterSlots | None = None,
    override_rank: dict[str, int] | None = None,
    override_valore: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, ReplacementLevel]]:
    """VORP_i = punti_attesi_i - replacement_level(ruolo_i), per ogni giocatore.

    Ritorna (vorp_per_id, replacement_level_per_ruolo) — il secondo va
    sempre riportato nel report per trasparenza (vedi limiti in cima al file).
    """
    replacement = calcola_replacement_level(
        punti_attesi_per_id,
        ruolo_per_id,
        config,
        rosa=rosa,
        override_rank=override_rank,
        override_valore=override_valore,
    )
    vorp = {
        pid: round(punti - replacement[ruolo_per_id[pid]].valore, 4)
        for pid, punti in punti_attesi_per_id.items()
    }
    return vorp, replacement


__all__ = [
    "RUOLI",
    "ReplacementLevel",
    "calcola_replacement_rank",
    "calcola_replacement_level",
    "calcola_vorp",
]
