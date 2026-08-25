"""Diagnostica per giocatore: curva `p_win(b)` e `n_osservazioni` su cui è
basata — vedi docs/DESIGN.md, Agente C, punto 4. Serve sia al report
markdown (`report.py`) sia a un consumo programmatico futuro (es. Modulo E
per il report finale della lega).
"""

from __future__ import annotations

from dataclasses import dataclass

from fantabuste.schemas import Player, PriceDistribution

N_PUNTI_CURVA_DEFAULT = 21
RAPPORTO_MAX_CURVA_DEFAULT = 2.5
"""Griglia di default per `curva_p_win`: da 0 a 2.5x la quotazione, 21
punti. Parametri di VISUALIZZAZIONE del report, non del modello — non
influenzano `p_win`, solo dove viene campionata per disegnare la curva.
Per questo non vivono in `league.yaml` (non sono regolamento di lega)."""


@dataclass(frozen=True)
class CurvaPWin:
    """Curva diagnostica di un giocatore: `p_win` valutata su una griglia
    di offerte, insieme alla fascia/modalità/numerosità che la sostengono
    — così l'incertezza (specialmente in modalità `prior`) è visibile
    accanto al numero, mai nascosta (vedi CLAUDE.md)."""

    player_id: str
    fascia: str
    mode: str
    n_osservazioni: int
    b_grid: list[float]
    p_win_grid: list[float]


def curva_p_win(
    player: Player,
    distribuzione: PriceDistribution,
    n_punti: int = N_PUNTI_CURVA_DEFAULT,
    rapporto_max: float = RAPPORTO_MAX_CURVA_DEFAULT,
) -> CurvaPWin:
    """Curva `p_win(b)` su una griglia di offerte da 0 a
    `rapporto_max * quotazione_listone` del giocatore.

    Riusa `PriceDistribution.p_win` (schemas.py, scritto e testato in
    Fase 0) — questo modulo non reimplementa la CDF/ECDF, produce solo la
    griglia di valutazione e il confezionamento diagnostico.
    """
    if distribuzione.player_id != player.player_id:
        raise ValueError(
            f"distribuzione per {distribuzione.player_id} passata insieme a "
            f"player {player.player_id}: mismatch"
        )
    if n_punti < 2:
        raise ValueError("n_punti deve essere >= 2")
    if rapporto_max <= 0:
        raise ValueError("rapporto_max deve essere positivo")

    step = (rapporto_max * player.quotazione_listone) / (n_punti - 1)
    b_grid = [round(step * i, 4) for i in range(n_punti)]
    p_win_grid = [
        distribuzione.p_win(b, player.quotazione_listone) if b > 0 else 0.0 for b in b_grid
    ]
    return CurvaPWin(
        player_id=player.player_id,
        fascia=distribuzione.fascia,
        mode=distribuzione.mode.value,
        n_osservazioni=distribuzione.n_osservazioni,
        b_grid=b_grid,
        p_win_grid=p_win_grid,
    )


def curve_p_win(
    players: list[Player],
    distribuzioni: list[PriceDistribution],
    n_punti: int = N_PUNTI_CURVA_DEFAULT,
    rapporto_max: float = RAPPORTO_MAX_CURVA_DEFAULT,
) -> list[CurvaPWin]:
    """`curva_p_win` per ogni giocatore in `players` che ha una
    `PriceDistribution` corrispondente in `distribuzioni`. Giocatori senza
    distribuzione vengono saltati silenziosamente (nessuna curva da
    produrre, non è un errore)."""
    dist_by_id = {d.player_id: d for d in distribuzioni}
    curve: list[CurvaPWin] = []
    for p in players:
        d = dist_by_id.get(p.player_id)
        if d is None:
            continue
        curve.append(curva_p_win(p, d, n_punti=n_punti, rapporto_max=rapporto_max))
    return curve
