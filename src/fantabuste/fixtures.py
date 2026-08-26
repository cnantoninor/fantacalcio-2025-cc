"""Generatore di fixture sintetiche per i test di ogni modulo.

Produce Player[] e PlayerStats[] (2 stagioni) con distribuzioni plausibili,
tutti marcati is_synthetic=True. Riproducibile da seed: stesso seed, stesso
output byte-per-byte. Squadre indicate come SQ01..SQ20 (non nomi reali di
Serie A) apposta, per non lasciare ambiguità sul fatto che questi non sono
dati veri — vedi l'avviso sui dati in CLAUDE.md.

Uso: python -m fantabuste.fixtures
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fantabuste.schemas import Player, PlayerStats, Ruolo

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"
FONTE = "fixture_sintetica_v1"
STAGIONI = ("2024/25", "2025/26")
N_SQUADRE = 20

# Giocatori per squadra e per ruolo: 4 P + 9 D + 9 C + 8 A = 30 -> 600 totali.
RUOLI_PER_SQUADRA: dict[Ruolo, int] = {"P": 4, "D": 9, "C": 9, "A": 8}

# Range di quotazione plausibili per ruolo (min, max) — puramente illustrativi,
# vedi CLAUDE.md: nessun numero da fixture entra mai in un'offerta reale.
QUOTAZIONE_RANGE: dict[Ruolo, tuple[int, int]] = {
    "P": (1, 20),
    "D": (1, 35),
    "C": (1, 60),
    "A": (1, 80),
}


@dataclass(frozen=True)
class _Qualita:
    """Variabile latente sintetica in [0, 1] che guida sia la quotazione sia
    le statistiche, così un giocatore "forte" nella fixture è coerentemente
    forte sia nel prezzo sia nel rendimento."""

    valore: float


def _genera_qualita(rng: random.Random) -> _Qualita:
    # Beta-like via due uniformi: concentra la massa su valori medio-bassi,
    # con una coda di top player, plausibile per una popolazione di 600.
    v = min(rng.random(), rng.random(), rng.random()) ** 0.5
    return _Qualita(valore=v)


PROB_CAMBIO_SQUADRA_STAGIONE_PIU_VECCHIA = 0.10
"""Probabilità che un giocatore risulti con una squadra diversa nella
stagione storica più vecchia rispetto a quella attuale — dà a
PlayerStats.squadra qualcosa di reale da testare per l'aggiustamento cambio
squadra del Modulo B, invece di ripetere sempre lo stesso valore."""


def _squadra_storica(
    squadra_attuale: str, indice_stagione: int, n_stagioni: int, rng: random.Random
) -> str:
    e_la_piu_vecchia = indice_stagione == 0 and n_stagioni > 1
    if not e_la_piu_vecchia or rng.random() >= PROB_CAMBIO_SQUADRA_STAGIONE_PIU_VECCHIA:
        return squadra_attuale
    altra_squadra = rng.randint(1, N_SQUADRE)
    while f"SQ{altra_squadra:02d}" == squadra_attuale:
        altra_squadra = rng.randint(1, N_SQUADRE)
    return f"SQ{altra_squadra:02d}"


def _quotazione(ruolo: Ruolo, qualita: _Qualita, rng: random.Random) -> float:
    lo, hi = QUOTAZIONE_RANGE[ruolo]
    base = lo + qualita.valore * (hi - lo)
    rumore = rng.uniform(0.85, 1.15)
    return round(max(lo, min(hi, base * rumore)))


def _stats_stagione(ruolo: Ruolo, qualita: _Qualita, rng: random.Random) -> dict:
    presenze = round(rng.uniform(5, 38) * (0.4 + 0.6 * qualita.valore))
    presenze = max(0, min(38, presenze))
    minuti = round(presenze * rng.uniform(45, 90))

    gol_atteso = {"P": 0.0, "D": 0.02, "C": 0.10, "A": 0.35}[ruolo] * qualita.valore
    assist_atteso = {"P": 0.0, "D": 0.03, "C": 0.15, "A": 0.15}[ruolo] * qualita.valore

    gol = sum(1 for _ in range(presenze) if rng.random() < gol_atteso)
    assist = sum(1 for _ in range(presenze) if rng.random() < assist_atteso)

    xG = round(gol * rng.uniform(0.8, 1.3), 2)
    xA = round(assist * rng.uniform(0.8, 1.3), 2)

    fantamedia_base = {"P": 6.0, "D": 6.0, "C": 6.0, "A": 6.0}[ruolo]
    fantamedia = round(fantamedia_base + qualita.valore * 2.5 + rng.uniform(-0.4, 0.4), 2)

    rigoristi_prob = 0.08 if ruolo in ("C", "A") else 0.0
    rigori_battuti = (
        round(rng.uniform(1, 8)) if qualita.valore > 0.75 and rng.random() < rigoristi_prob else 0
    )

    return dict(
        presenze=presenze,
        minuti=minuti,
        gol=gol,
        assist=assist,
        xG=xG,
        xA=xA,
        fantamedia=fantamedia,
        rigori_battuti=rigori_battuti,
    )


def genera_fixture_giocatori(seed: int = 42) -> tuple[list[Player], list[PlayerStats]]:
    """Genera Player[] e PlayerStats[] (2 stagioni) deterministicamente da seed."""
    rng = random.Random(seed)
    data_estrazione = datetime(2026, 8, 25, tzinfo=UTC)

    players: list[Player] = []
    stats: list[PlayerStats] = []

    player_counter = 0
    for squadra_idx in range(1, N_SQUADRE + 1):
        squadra = f"SQ{squadra_idx:02d}"
        for ruolo, n in RUOLI_PER_SQUADRA.items():
            for _ in range(n):
                player_counter += 1
                player_id = f"P{player_counter:04d}"
                qualita = _genera_qualita(rng)

                players.append(
                    Player(
                        player_id=player_id,
                        nome=f"Giocatore Sintetico {player_counter}",
                        ruolo=ruolo,
                        squadra=squadra,
                        quotazione_listone=_quotazione(ruolo, qualita, rng),
                        fonte=FONTE,
                        is_synthetic=True,
                        data_estrazione=data_estrazione,
                    )
                )

                for indice_stagione, stagione in enumerate(STAGIONI):
                    s = _stats_stagione(ruolo, qualita, rng)
                    squadra_stagione = _squadra_storica(
                        squadra, indice_stagione, len(STAGIONI), rng
                    )
                    stats.append(
                        PlayerStats(
                            player_id=player_id,
                            stagione=stagione,
                            squadra=squadra_stagione,
                            fonte=FONTE,
                            is_synthetic=True,
                            **s,
                        )
                    )

    return players, stats


def scrivi_fixture(
    players: list[Player], stats: list[PlayerStats], out_dir: Path = FIXTURES_DIR
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    players_path = out_dir / "players.csv"
    stats_path = out_dir / "player_stats.csv"

    pd.DataFrame([p.model_dump() for p in players]).to_csv(players_path, index=False)
    pd.DataFrame([s.model_dump() for s in stats]).to_csv(stats_path, index=False)

    return players_path, stats_path


def main() -> None:
    players, stats = genera_fixture_giocatori()
    players_path, stats_path = scrivi_fixture(players, stats)
    print(f"Generati {len(players)} giocatori -> {players_path}")
    print(f"Generate {len(stats)} righe di statistiche ({len(STAGIONI)} stagioni) -> {stats_path}")


if __name__ == "__main__":
    main()
