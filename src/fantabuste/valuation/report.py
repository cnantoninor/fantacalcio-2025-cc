"""Genera `valuation_report.md` — obbligatorio per docs/DESIGN.md, Agente B,
punto 4: MAE/RMSE di baseline e ML, numero di osservazioni, e una sezione
esplicita "perché questo risultato è debole" che dichiara il limite del
singolo fold. Non è un nice-to-have: è la Definition of Done del modulo.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fantabuste.config import LeagueConfig
from fantabuste.valuation.baseline import MediaDispersioneRuolo
from fantabuste.valuation.ml import RisultatoSfidanteML
from fantabuste.valuation.vorp import ReplacementLevel

DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "valuation_report.md"
)


def _sezione_ml(
    risultato: RisultatoSfidanteML | None, ml_non_disponibile_motivo: str | None
) -> str:
    if ml_non_disponibile_motivo is not None:
        return (
            "## Sfidante ML\n\n"
            f"**Non valutato**: {ml_non_disponibile_motivo}\n\n"
            "Il baseline resta l'unico metodo di produzione. Nessuna proiezione in questo "
            "run porta `metodo=ml`.\n"
        )
    if risultato is None:
        return (
            "## Sfidante ML\n\n"
            "Non richiesto in questo run (flag `--enable-ml` non passato). Il baseline è "
            "l'unico metodo di produzione — coerente con docs/DESIGN.md: il baseline è il "
            "default, l'ML è uno sfidante opzionale.\n"
        )
    esito = "ACCETTATO" if risultato.accettato else "RIFIUTATO"
    return (
        "## Sfidante ML\n\n"
        f"| Metrica | Baseline | ML |\n"
        f"|---|---|---|\n"
        f"| MAE | {risultato.mae_baseline:.4f} | {risultato.mae_ml:.4f} |\n"
        f"| RMSE | {risultato.rmse_baseline:.4f} | {risultato.rmse_ml:.4f} |\n\n"
        f"Osservazioni nel fold di backtest (giocatori con 2 stagioni complete): "
        f"**{risultato.n_osservazioni}**.\n\n"
        f"Margine minimo richiesto: MAE ML inferiore di almeno "
        f"**{risultato.margine_richiesto:.0%}** rispetto al baseline.\n\n"
        f"**Esito: {esito}.** {risultato.motivo}\n\n"
        + (
            "Le proiezioni di produzione in questo run usano `metodo=ml`.\n"
            if risultato.accettato
            else "Le proiezioni di produzione in questo run restano `metodo=baseline` "
            "(rifiuto automatico, non una scelta manuale).\n"
        )
    )


def _sezione_perche_debole(risultato: RisultatoSfidanteML | None, n_giocatori_totale: int) -> str:
    righe = [
        "## Perché questo risultato è debole\n",
        "Con **2 sole stagioni storiche** disponibili, il confronto baseline/ML sopra si "
        "basa su un **unico fold temporale non replicabile**: si allena (o si applica la "
        "formula) usando la stagione più vecchia come storico e si misura l'errore contro "
        "la stagione più recente come target. Non è validazione nel senso classico — è un "
        "singolo esperimento. Se per caso quella singola coppia di stagioni contenesse una "
        "particolarità (un anno anomalo per infortuni, cambi di modulo, un singolo "
        "outlier con molte presenze), il risultato del confronto cambierebbe interamente "
        "senza che nessun dato lo segnali.",
        "",
        "La cross-validation K-fold usata per l'ML (vedi `ml.py`) stima l'errore fuori "
        "campione **tra giocatori**, non aggiunge fold temporali: quelli restano uno. "
        "Serve solo a evitare di valutare il modello sugli stessi dati su cui è stato "
        "fittato, non a simulare una validazione su più stagioni che i dati non hanno.",
        "",
        f"La popolazione osservata in questo run è di **{n_giocatori_totale} giocatori** "
        "(fixture sintetiche in v1). Con dati reali il numero di giocatori con 2 stagioni "
        "complete sarà probabilmente inferiore (infortuni, esordienti, trasferimenti da "
        "campionati esteri senza storico comparabile), riducendo ulteriormente la potenza "
        "del confronto.",
        "",
        "Gli intervalli di incertezza (`punti_attesi_lo/hi`) sono derivati dalla dispersione "
        "storica per fascia di ruolo osservata su questa stessa popolazione, non da una "
        "quantile regression fittata sul campione: sono quindi larghi per costruzione, "
        "onesti sul fatto che con 2 stagioni non c'è modo di stimare l'incertezza con "
        "precisione, non uno strumento di calibrazione fine.",
        "",
        "**Conclusione**: qualunque esito sopra (ML accettato o rifiutato) va trattato come "
        "un'indicazione debole, non come una conferma statistica. Il baseline resta il "
        "metodo di produzione di default proprio per questo motivo, indipendentemente "
        "dall'esito del confronto su questo singolo fold.",
    ]
    if risultato is not None and risultato.accettato:
        righe.append(
            "\n> Nota: in questo run l'ML ha superato la soglia e viene usato in produzione. "
            "Questo NON significa che il risultato sopra sia forte — significa solo che ha "
            "superato la soglia di margine ampio scelta apposta per essere conservativa. "
            "Ri-verificare quando saranno disponibili più stagioni reali."
        )
    return "\n".join(righe) + "\n"


def _sezione_intervalli(medie_ruolo: dict[str, MediaDispersioneRuolo]) -> str:
    righe = [
        "## Dispersione di ruolo (base degli intervalli di incertezza)\n",
        "| Ruolo | Media fantamedia pesata | Deviazione standard | N giocatori |",
        "|---|---|---|---|",
    ]
    for ruolo in ("P", "D", "C", "A"):
        m = medie_ruolo.get(ruolo)
        if m is None:
            righe.append(f"| {ruolo} | n/d | n/d | 0 |")
        else:
            righe.append(
                f"| {ruolo} | {m.media:.3f} | {m.deviazione_standard:.3f} | {m.n_giocatori} |"
            )
    righe.append(
        "\n`punti_attesi_lo/hi` è derivato da questa deviazione standard per ruolo "
        "(vedi `baseline.Z_INTERVALLO`), non da un modello fittato sullo stesso campione."
    )
    return "\n".join(righe) + "\n"


def _sezione_vorp(replacement: dict[str, ReplacementLevel], config: LeagueConfig) -> str:
    righe = [
        "## VORP — replacement level\n",
        f"`n_partecipanti` = {config.n_partecipanti}, slot di riferimento = "
        "`rosa_fase1` (2-8-8-6, la composizione obbligatoria entro il 7 settembre — "
        "vedi docs/OPEN_QUESTIONS.md §0).\n",
        "| Ruolo | Rank replacement | Valore replacement | N disponibili | Degenerato |",
        "|---|---|---|---|---|",
    ]
    for ruolo in ("P", "D", "C", "A"):
        r = replacement.get(ruolo)
        if r is None:
            continue
        righe.append(
            f"| {ruolo} | {r.rank} | {r.valore:.4f} | {r.n_disponibili_nel_ruolo} | "
            f"{'SÌ — campione troppo piccolo per il ruolo' if r.degenerato else 'no'} |"
        )
    righe.append(
        "\nLimite noto (vedi anche `vorp.py`): il replacement level è calcolato sulla "
        "popolazione disponibile a questo run (fixture sintetiche in v1), non da un "
        "mercato osservato — il mercato busta chiusa non è liquido (2 sole tornate, "
        "nessun prezzo continuo). È un ranking di valore relativo, non una previsione "
        "di prezzo: quella è compito dei Moduli C e D."
    )
    return "\n".join(righe) + "\n"


def genera_report(
    *,
    n_giocatori_totale: int,
    n_giocatori_sintetici: int,
    confidence_counts: Counter,
    medie_ruolo: dict[str, MediaDispersioneRuolo],
    replacement: dict[str, ReplacementLevel],
    config: LeagueConfig,
    risultato_ml: RisultatoSfidanteML | None,
    ml_non_disponibile_motivo: str | None,
    metodo_produzione: str,
) -> str:
    ora = datetime.now(tz=UTC).isoformat()
    header = (
        "# Valuation report — Modulo B (FANTABUSTE)\n\n"
        f"Generato: {ora}\n\n"
        f"Giocatori valutati: **{n_giocatori_totale}** "
        f"({n_giocatori_sintetici} sintetici, "
        f"{n_giocatori_totale - n_giocatori_sintetici} reali).\n\n"
    )
    if n_giocatori_sintetici == n_giocatori_totale:
        header += (
            "> ⚠️ **Tutti i giocatori in questo run sono sintetici** (`is_synthetic=True`). "
            "Nessun numero qui dentro deve entrare in un'offerta reale — vedi l'avviso sui "
            "dati in CLAUDE.md e `assert_no_synthetic_dependency` in `schemas.py`.\n\n"
        )
    header += (
        f"Metodo di produzione in questo run: **`{metodo_produzione}`**.\n\n"
        "Distribuzione `confidence_flag`: "
        + ", ".join(f"{k}={v}" for k, v in sorted(confidence_counts.items()))
        + "\n\n---\n\n"
    )

    return (
        header
        + _sezione_ml(risultato_ml, ml_non_disponibile_motivo)
        + "\n---\n\n"
        + _sezione_perche_debole(risultato_ml, n_giocatori_totale)
        + "\n---\n\n"
        + _sezione_intervalli(medie_ruolo)
        + "\n---\n\n"
        + _sezione_vorp(replacement, config)
    )


def scrivi_report(contenuto: str, path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contenuto, encoding="utf-8")
    return path


__all__ = ["DEFAULT_REPORT_PATH", "genera_report", "scrivi_report"]
