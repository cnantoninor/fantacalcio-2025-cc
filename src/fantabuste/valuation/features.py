"""Feature engineering — Modulo B, punto 1 di docs/DESIGN.md.

Costruisce, per ogni giocatore, le feature derivate da `Player` + `PlayerStats`
usate sia dal baseline (`baseline.py`) sia dallo sfidante ML opzionale
(`ml.py`). Non tocca `schemas.py`: `PlayerFeatures` è un tipo interno del
modulo, non un contratto condiviso.

Nota sul contratto dati (limite noto, non risolvibile qui): `PlayerStats` non
porta un campo squadra per stagione, e `Player.squadra` è solo lo snapshot
alla data di estrazione corrente. Non è quindi possibile ricostruire da questi
due schemi se e quando un giocatore ha cambiato squadra tra le stagioni
storiche osservate — l'informazione semplicemente non è nel contratto. Vedi
`baseline.py::FATTORE_CAMBIO_SQUADRA` per come questo modulo dichiara (invece
di nascondere) la conseguenza.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantabuste.schemas import Player, PlayerStats, Ruolo

# Minuti di una stagione "piena" per un giocatore sempre titolare: 38 giornate
# di campionato Serie A a 90 minuti. Usato per stimare prob_titolarita dai
# minuti storici (proxy dichiarata: non abbiamo un dato di formazione futura,
# solo minuti passati — vedi baseline.py).
MINUTI_STAGIONE_PIENA = 38 * 90


@dataclass(frozen=True)
class StagioneFeatures:
    """Statistiche di una singola stagione, derivate da un PlayerStats."""

    stagione: str
    presenze: int
    minuti: int
    fantamedia: float
    gol_per90: float
    assist_per90: float
    xG_per90: float
    xA_per90: float
    rigori_battuti: int
    quota_minuti_stagione: float = field(
        metadata={"doc": "minuti / MINUTI_STAGIONE_PIENA, clampato a [0, 1]"}
    )


@dataclass(frozen=True)
class PlayerFeatures:
    """Feature aggregate di un giocatore su tutte le stagioni disponibili
    (ordinate dalla più vecchia alla più recente)."""

    player_id: str
    nome: str
    ruolo: Ruolo
    squadra: str
    quotazione_listone: float
    fonte_player: str
    is_synthetic: bool
    stagioni: tuple[StagioneFeatures, ...]

    @property
    def minuti_totali(self) -> int:
        return sum(s.minuti for s in self.stagioni)

    @property
    def n_stagioni_osservate(self) -> int:
        return len(self.stagioni)


def _per90(valore: float, minuti: int) -> float:
    if minuti <= 0:
        return 0.0
    return valore * 90.0 / minuti


def _stagione_features(ps: PlayerStats) -> StagioneFeatures:
    quota_minuti = MINUTI_STAGIONE_PIENA
    return StagioneFeatures(
        stagione=ps.stagione,
        presenze=ps.presenze,
        minuti=ps.minuti,
        fantamedia=ps.fantamedia,
        gol_per90=_per90(ps.gol, ps.minuti),
        assist_per90=_per90(ps.assist, ps.minuti),
        xG_per90=_per90(ps.xG, ps.minuti),
        xA_per90=_per90(ps.xA, ps.minuti),
        rigori_battuti=ps.rigori_battuti,
        quota_minuti_stagione=max(0.0, min(1.0, ps.minuti / quota_minuti)),
    )


def costruisci_features(
    players: list[Player], stats: list[PlayerStats]
) -> list[PlayerFeatures]:
    """Unisce Player e PlayerStats per player_id in PlayerFeatures.

    Le stagioni sono ordinate per stringa `stagione` (es. '2024/25' < '2025/26'
    ordina correttamente in ASCII per il formato usato dalle fixture e da
    LEAGUE_CONTEXT). Giocatori senza alcuna riga di PlayerStats sono inclusi
    con `stagioni=()`: il baseline li gestisce come shrinkage totale verso la
    media di ruolo (vedi baseline.py).
    """
    stats_by_player: dict[str, list[PlayerStats]] = {}
    for s in stats:
        stats_by_player.setdefault(s.player_id, []).append(s)

    out: list[PlayerFeatures] = []
    for p in players:
        righe = sorted(stats_by_player.get(p.player_id, []), key=lambda s: s.stagione)
        out.append(
            PlayerFeatures(
                player_id=p.player_id,
                nome=p.nome,
                ruolo=p.ruolo,
                squadra=p.squadra,
                quotazione_listone=p.quotazione_listone,
                fonte_player=p.fonte,
                is_synthetic=p.is_synthetic,
                stagioni=tuple(_stagione_features(s) for s in righe),
            )
        )
    return out
