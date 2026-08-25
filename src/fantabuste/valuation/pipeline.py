"""Orchestratore del Modulo B: Player[] + PlayerStats[] -> PlayerProjection[].

Punto di ingresso pensato per essere chiamato sia dai test sia (in futuro)
dalla CLI del Modulo E (`src/fantabuste/cli.py`, non di competenza di questo
modulo — vedi CLAUDE.md "Confini di modulo"). Espone anche un piccolo entry
point standalone (`python -m fantabuste.valuation.pipeline`) per generare
`valuation_report.md` sulle fixture senza dipendere da E, coerente con la
richiesta di lavorare in parallelo sulle fixture (docs/DESIGN.md).

Il flag `--enable-ml` di docs/DESIGN.md vive qui come parametro
`enable_ml: bool` di `esegui_valutazione` — è la CLI di E che dovrà esporlo
come flag da riga di comando quando i moduli verranno integrati; questo
modulo lo espone già pronto all'uso, non lo nasconde dietro codice interno.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantabuste.config import EXAMPLE_CONFIG_PATH, LeagueConfig
from fantabuste.schemas import (
    MetodoProiezione,
    Player,
    PlayerProjection,
    PlayerStats,
    RosterSlots,
)
from fantabuste.valuation import baseline as baseline_mod
from fantabuste.valuation import ml as ml_mod
from fantabuste.valuation import report as report_mod
from fantabuste.valuation import vorp as vorp_mod
from fantabuste.valuation.features import costruisci_features

FONTE_MODULO = "fantabuste.valuation.pipeline"


@dataclass(frozen=True)
class RisultatoValutazione:
    proiezioni: list[PlayerProjection]
    metodo_produzione: MetodoProiezione
    risultato_ml: ml_mod.RisultatoSfidanteML | None
    ml_non_disponibile_motivo: str | None
    replacement: dict[str, vorp_mod.ReplacementLevel]
    medie_ruolo: dict[str, baseline_mod.MediaDispersioneRuolo]
    report_markdown: str


def _stats_per_player(stats: list[PlayerStats]) -> dict[str, list[PlayerStats]]:
    out: dict[str, list[PlayerStats]] = {}
    for s in stats:
        out.setdefault(s.player_id, []).append(s)
    return out


def _fonte_e_synthetic(
    player: Player, stats_giocatore: list[PlayerStats], metodo: MetodoProiezione
) -> tuple[str, bool]:
    """Nessun default silenzioso: is_synthetic è sempre calcolato esplicitamente
    dalle dipendenze a monte (Player + le sue PlayerStats), mai lasciato al
    default=True di SyntheticRecord senza motivazione — vedi CLAUDE.md."""
    is_synthetic = player.is_synthetic or any(s.is_synthetic for s in stats_giocatore)
    fonti_stats = sorted({s.fonte for s in stats_giocatore})
    fonte = (
        f"{FONTE_MODULO}; metodo={metodo.value}; player_fonte={player.fonte}; "
        f"stats_fonte={'+'.join(fonti_stats) if fonti_stats else 'nessuna'}"
    )
    return fonte, is_synthetic


def esegui_valutazione(
    players: list[Player],
    stats: list[PlayerStats],
    config: LeagueConfig,
    *,
    enable_ml: bool = False,
    margine_ml: float = ml_mod.MARGINE_MIGLIORAMENTO_MINIMO_DEFAULT,
    rosa_vorp: RosterSlots | None = None,
    override_rank_vorp: dict[str, int] | None = None,
    override_valore_vorp: dict[str, float] | None = None,
) -> RisultatoValutazione:
    """Esegue l'intera pipeline B: feature engineering -> baseline (sempre)
    -> ML opzionale dietro `enable_ml` -> VORP -> report.

    Il baseline è SEMPRE calcolato (è il modello primario, docs/DESIGN.md):
    anche con `enable_ml=True`, se l'ML non batte il baseline con margine
    sufficiente le proiezioni prodotte restano `metodo=baseline`.
    """
    features = costruisci_features(players, stats)
    stats_by_player = _stats_per_player(stats)
    players_by_id = {p.player_id: p for p in players}

    proiezioni_baseline, medie_ruolo = baseline_mod.proietta_baseline_tutti(features)
    punti_baseline = {p.player_id: p.punti_attesi for p in proiezioni_baseline}

    risultato_ml: ml_mod.RisultatoSfidanteML | None = None
    ml_non_disponibile_motivo: str | None = None
    metodo_produzione = MetodoProiezione.BASELINE
    punti_ml_produzione: dict[str, float] = {}

    if enable_ml:
        try:
            risultato_ml = ml_mod.valuta_sfidante_ml(features, margine_ml)
        except ml_mod.MLNonDisponibile as exc:
            ml_non_disponibile_motivo = str(exc)
        else:
            if risultato_ml.accettato:
                modello = ml_mod.fitta_modello_produzione(features)
                fantamedia_ml = ml_mod.proietta_ml_produzione(features, modello)
                # L'ML predice la fantamedia attesa (stesso target del backtest,
                # vedi ml.py): la moltiplicazione per prob_titolarita e lo stesso
                # concetto di "aggiustamento cambio squadra" restano quelli del
                # baseline — l'ML sostituisce SOLO la stima della fantamedia
                # attesa, non l'intera pipeline di 4 passi di DESIGN.md.
                by_id_features = {f.player_id: f for f in features}
                for pid, fantamedia_attesa in fantamedia_ml.items():
                    prob_tit = baseline_mod.stima_prob_titolarita(by_id_features[pid])
                    punti_ml_produzione[pid] = round(
                        fantamedia_attesa * baseline_mod.FATTORE_CAMBIO_SQUADRA * prob_tit, 4
                    )
                metodo_produzione = MetodoProiezione.ML

    if metodo_produzione is MetodoProiezione.ML:
        punti_produzione = punti_baseline | punti_ml_produzione
    else:
        punti_produzione = punti_baseline

    ruolo_per_id = {f.player_id: f.ruolo for f in features}
    replacement_rosa = rosa_vorp or config.rosa_fase1
    vorp_per_id, replacement = vorp_mod.calcola_vorp(
        punti_produzione,
        ruolo_per_id,
        config,
        rosa=replacement_rosa,
        override_rank=override_rank_vorp,
        override_valore=override_valore_vorp,
    )

    baseline_by_id = {p.player_id: p for p in proiezioni_baseline}
    proiezioni: list[PlayerProjection] = []
    confidence_counts: Counter = Counter()
    for f in features:
        player = players_by_id[f.player_id]
        stats_giocatore = stats_by_player.get(f.player_id, [])
        fonte, is_synthetic = _fonte_e_synthetic(player, stats_giocatore, metodo_produzione)

        pb = baseline_by_id[f.player_id]
        if metodo_produzione is MetodoProiezione.ML and f.player_id in punti_ml_produzione:
            punti_attesi = punti_ml_produzione[f.player_id]
            # Gli intervalli restano derivati dalla dispersione storica di
            # ruolo (DESIGN.md punto 5: mai da una quantile regression
            # fittata sullo stesso campione), quindi si riusa la stessa
            # larghezza calcolata dal baseline, ricentrata sulla previsione ML.
            largo_lo = pb.punti_attesi - pb.punti_attesi_lo
            largo_hi = pb.punti_attesi_hi - pb.punti_attesi
            punti_attesi_lo = max(0.0, punti_attesi - largo_lo)
            punti_attesi_hi = punti_attesi + largo_hi
            prob_titolarita = pb.prob_titolarita
            confidence_flag = pb.confidence_flag
        else:
            punti_attesi = pb.punti_attesi
            punti_attesi_lo = pb.punti_attesi_lo
            punti_attesi_hi = pb.punti_attesi_hi
            prob_titolarita = pb.prob_titolarita
            confidence_flag = pb.confidence_flag

        confidence_counts[confidence_flag] += 1

        proiezioni.append(
            PlayerProjection(
                player_id=f.player_id,
                punti_attesi=punti_attesi,
                punti_attesi_lo=punti_attesi_lo,
                punti_attesi_hi=punti_attesi_hi,
                prob_titolarita=prob_titolarita,
                vorp=vorp_per_id[f.player_id],
                metodo=metodo_produzione,
                confidence_flag=confidence_flag,
                fonte=fonte,
                is_synthetic=is_synthetic,
            )
        )

    n_sintetici = sum(1 for p in proiezioni if p.is_synthetic)
    report_markdown = report_mod.genera_report(
        n_giocatori_totale=len(proiezioni),
        n_giocatori_sintetici=n_sintetici,
        confidence_counts=confidence_counts,
        medie_ruolo=medie_ruolo,
        replacement=replacement,
        config=config,
        risultato_ml=risultato_ml,
        ml_non_disponibile_motivo=ml_non_disponibile_motivo,
        metodo_produzione=metodo_produzione.value,
    )

    return RisultatoValutazione(
        proiezioni=proiezioni,
        metodo_produzione=metodo_produzione,
        risultato_ml=risultato_ml,
        ml_non_disponibile_motivo=ml_non_disponibile_motivo,
        replacement=replacement,
        medie_ruolo=medie_ruolo,
        report_markdown=report_markdown,
    )


# ---------------------------------------------------------------------------
# Entry point standalone (non la CLI ufficiale, di competenza del Modulo E)
# ---------------------------------------------------------------------------


def _carica_players_stats(
    players_csv: Path, stats_csv: Path
) -> tuple[list[Player], list[PlayerStats]]:
    players_df = pd.read_csv(players_csv)
    stats_df = pd.read_csv(stats_csv)
    players = [Player.model_validate(row) for row in players_df.to_dict(orient="records")]
    stats = [PlayerStats.model_validate(row) for row in stats_df.to_dict(orient="records")]
    return players, stats


def _carica_config(path: Path | None) -> LeagueConfig:
    if path is not None:
        return LeagueConfig.load(path)
    try:
        return LeagueConfig.load()
    except FileNotFoundError:
        print(
            "config/league.yaml non trovato: uso config/league.example.yaml "
            "(solo per esecuzione standalone di questo modulo — la CLI ufficiale "
            "del Modulo E deve richiedere league.yaml vero)."
        )
        return LeagueConfig.load(EXAMPLE_CONFIG_PATH)


def main() -> None:
    default_fixtures = Path(__file__).resolve().parents[3] / "data" / "fixtures"
    parser = argparse.ArgumentParser(
        description="Modulo B — genera PlayerProjection e valuation_report.md"
    )
    parser.add_argument("--enable-ml", action="store_true", help="Valuta lo sfidante ML")
    parser.add_argument(
        "--margine-ml",
        type=float,
        default=ml_mod.MARGINE_MIGLIORAMENTO_MINIMO_DEFAULT,
        help="Margine minimo di miglioramento MAE richiesto all'ML (default 0.10 = 10%%)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Percorso a league.yaml")
    parser.add_argument("--players-csv", type=Path, default=default_fixtures / "players.csv")
    parser.add_argument("--stats-csv", type=Path, default=default_fixtures / "player_stats.csv")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=report_mod.DEFAULT_REPORT_PATH,
        help="Percorso di output per valuation_report.md",
    )
    args = parser.parse_args()

    config = _carica_config(args.config)
    players, stats = _carica_players_stats(args.players_csv, args.stats_csv)

    risultato = esegui_valutazione(
        players, stats, config, enable_ml=args.enable_ml, margine_ml=args.margine_ml
    )
    path = report_mod.scrivi_report(risultato.report_markdown, args.report_out)

    print(f"Proiezioni generate: {len(risultato.proiezioni)}")
    print(f"Metodo di produzione: {risultato.metodo_produzione.value}")
    if risultato.risultato_ml is not None:
        print(f"ML: {'accettato' if risultato.risultato_ml.accettato else 'rifiutato'}")
    print(f"Report scritto in: {path}")


if __name__ == "__main__":
    main()
