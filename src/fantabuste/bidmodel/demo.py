"""Genera un `bidmodel_report.md` dimostrativo, su fixture sintetiche
costruite QUI (non tocca `data/fixtures/` né `fixtures.py` — vedi
CLAUDE.md, "Confini di modulo": quelle fixture sono di competenza degli
altri 3 agenti della Fase 1 in parallelo).

Mostra ENTRAMBE le modalità fianco a fianco:
- una fascia (`A_tier1`) con abbastanza osservazioni sintetiche per
  superare la soglia reale `K = config.bidmodel_min_osservazioni = 30`
  -> `empirical`;
- le altre fasce (`P_tier1`, `D_tier1`, `C_tier1`) con ZERO osservazioni
  storiche -> `prior`. Questa è la situazione REALE della v1 (vedi
  docs/OPEN_QUESTIONS.md §2): nessuno storico di offerte avversarie esiste
  per questa lega, quindi nella pipeline vera ogni fascia gira in `prior`,
  non solo quelle di questo esempio.

Usa `n_tier=1` (una sola fascia per ruolo) apposta, solo per tenere
piccola la fixture dimostrativa: il binning per quantili con
`N_TIER_DEFAULT=3` è invece quello esercitato dai test unitari di
`fasce.py`.

Uso: `python -m fantabuste.bidmodel.demo [output_path]`
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from fantabuste.config import (
    AstaRiparazioneConfig,
    LeagueConfig,
)
from fantabuste.schemas import OpponentBidObservation, Player, RosterSlots

from .fitting import fit_fascia, fit_price_distributions
from .normalizzazione import StagioneStorica, pool_stagioni
from .report import genera_report

FONTE_DEMO = "bidmodel_demo_v1"
STAGIONE_DEMO = "2025/26"
REPORT_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "bidmodel_report.md"


def _config_demo() -> LeagueConfig:
    """Stessa configurazione della lega reale (vedi config/league.example.yaml
    e config/league.yaml) — in particolare `bidmodel_min_osservazioni=30`,
    la soglia K reale, NON un valore ridotto per comodità della demo."""
    return LeagueConfig(
        n_partecipanti=12,
        budget_totale_fase1=500,
        budget_bonus_fase2=50,
        rosa_fase1=RosterSlots(P=2, D=8, C=8, A=6),
        rosa_fase2_massima=RosterSlots(P=4, D=10, C=10, A=8),
        n_tornate_buste=2,
        assegna_buste_se_uguali=False,
        offerte_non_tonde=True,
        modificatore_difesa=True,
        regolamento_portieri_top8_attivo=True,
        max_giocatori_per_squadra_serie_a=None,
        bidmodel_min_osservazioni=30,
        asta_riparazione=AstaRiparazioneConfig(
            tipo="tempo", sequenziale=False, soft_close=False, watchlist_size=8
        ),
    )


def _players_demo() -> list[Player]:
    """32 attaccanti (per superare K=30 lotti in A_tier1 con 2 tornate:
    32*2=64 >= 30) + 6 giocatori per ciascuno degli altri ruoli, senza
    storico -> restano in prior. Tutti marcati is_synthetic=True,
    fonte=FONTE_DEMO: mai un default silenzioso, vedi CLAUDE.md."""
    oggi = datetime.now(UTC)
    players: list[Player] = []
    for i in range(32):
        players.append(
            Player(
                player_id=f"DEMO_A{i:03d}",
                nome=f"Attaccante Demo {i}",
                ruolo="A",
                squadra=f"SQ{(i % 8) + 1:02d}",
                quotazione_listone=10 + (i % 20) * 3.0,
                fonte=FONTE_DEMO,
                is_synthetic=True,
                data_estrazione=oggi,
            )
        )
    for ruolo, n in (("P", 6), ("D", 6), ("C", 6)):
        for i in range(n):
            players.append(
                Player(
                    player_id=f"DEMO_{ruolo}{i:03d}",
                    nome=f"{ruolo} Demo {i}",
                    ruolo=ruolo,  # type: ignore[arg-type]
                    squadra=f"SQ{(i % 8) + 1:02d}",
                    quotazione_listone=5 + i * 2.0,
                    fonte=FONTE_DEMO,
                    is_synthetic=True,
                    data_estrazione=oggi,
                )
            )
    return players


def _osservazioni_storiche_demo(attaccanti: list[Player]) -> list[OpponentBidObservation]:
    """2 avversari sintetici, 2 tornate, per ognuno dei 32 attaccanti:
    64 lotti, 128 righe di offerta. Rapporto medio ~1.0 con dispersione,
    così A_tier1 mostra una distribuzione empirica non banale."""
    oss: list[OpponentBidObservation] = []
    for p in attaccanti:
        for tornata in (1, 2):
            for k, avversario in enumerate(("AVV_1", "AVV_2")):
                # variazione deterministica (no RNG): riproducibile senza seed.
                rapporto = 0.7 + 0.1 * tornata + 0.05 * k + 0.15 * ((hash(p.player_id) % 5) - 2)
                rapporto = max(0.2, rapporto)
                oss.append(
                    OpponentBidObservation(
                        player_id=p.player_id,
                        stagione=STAGIONE_DEMO,
                        tornata=tornata,  # type: ignore[arg-type]
                        avversario_id=avversario,
                        offerta=round(rapporto * p.quotazione_listone, 2),
                        vincente=(k == 0),
                        fonte=FONTE_DEMO,
                        is_synthetic=True,
                    )
                )
    return oss


def genera_report_demo(output_path: Path | str = REPORT_DEFAULT_PATH) -> Path:
    config = _config_demo()
    players = _players_demo()
    attaccanti = [p for p in players if p.ruolo == "A"]
    osservazioni = _osservazioni_storiche_demo(attaccanti)

    quotazioni = {p.player_id: p.quotazione_listone for p in players}
    stagione = StagioneStorica(
        label=STAGIONE_DEMO,
        osservazioni=osservazioni,
        quotazioni=quotazioni,
        budget_totale=config.budget_totale_fase1,
        n_partecipanti=config.n_partecipanti,
    )
    budget_procapite_riferimento = config.budget_totale_fase1 / config.n_partecipanti
    rapporti = pool_stagioni([stagione], budget_procapite_riferimento)

    distribuzioni = fit_price_distributions(
        players=players,
        config=config,
        fonte=FONTE_DEMO,
        rapporti_storici=rapporti,
        n_tier=1,  # una fascia per ruolo, solo per tenere la demo piccola
    )

    # Dimostrazione esplicita della degradazione automatica sotto soglia K:
    # una fascia con osservazioni (10) ma sotto K=30 -> degrada a prior.
    fit_sotto_soglia = fit_fascia(
        fascia="DEMO_sotto_soglia",
        lot_max_ratios=[0.9 + 0.01 * i for i in range(10)],
        min_osservazioni=config.bidmodel_min_osservazioni,
    )
    note_extra = [
        "**Dimostrazione della degradazione automatica sotto soglia K** "
        f"(`config.bidmodel_min_osservazioni = {config.bidmodel_min_osservazioni}`): "
        f"una fascia ipotetica con {fit_sotto_soglia.n_osservazioni} osservazioni "
        f"(< {config.bidmodel_min_osservazioni}) ottiene "
        f"`mode={fit_sotto_soglia.mode.value}`, `degradata_a_prior="
        f"{fit_sotto_soglia.degradata_a_prior}` — nessun errore, nessuna "
        "distribuzione empirica corta spacciata per affidabile.",
        "",
        "Fasce di questo report: `assegna_fasce` è stata chiamata con "
        "`n_tier=1` (una fascia per ruolo) solo per tenere piccola la "
        "fixture dimostrativa — il binning per quantili con "
        "`N_TIER_DEFAULT=3` è quello usato in produzione ed è esercitato "
        "dai test unitari di `fasce.py`.",
    ]

    return genera_report(
        players=players,
        distribuzioni=distribuzioni,
        output_path=output_path,
        min_osservazioni=config.bidmodel_min_osservazioni,
        esempi_curva=["DEMO_A000", "DEMO_P000"],
        note_extra=note_extra,
    )


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT_DEFAULT_PATH
    path = genera_report_demo(dest)
    print(f"Report scritto in {path}")
