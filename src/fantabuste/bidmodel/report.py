"""Genera `bidmodel_report.md` — il report diagnostico obbligatorio del
Modulo C (vedi DoD dell'Agente C in docs/DESIGN.md).

Per ogni fascia: modalità, `n_osservazioni`, e la curva fittata (quantili
del supporto empirico, o mu/sigma + quantili impliciti in `prior`).
L'incertezza di `prior` è sempre visibile qui, mai nascosta — vedi
CLAUDE.md, "Avviso sui dati" e "Standard".
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from fantabuste.schemas import Player, PriceDistribution, PriceDistributionMode

# Quantili standard della normale (0,1) — costanti matematiche, non
# parametri di regolamento o di modello: servono solo a mostrare i
# quantili impliciti di una lognormale(mu, sigma) senza dipendere da
# scipy.stats.norm.ppf (scipy non è fra le dipendenze del progetto).
_Z_QUANTILI_STANDARD: dict[float, float] = {
    0.10: -1.2815515655446004,
    0.25: -0.6744897501960817,
    0.50: 0.0,
    0.75: 0.6744897501960817,
    0.90: 1.2815515655446004,
}


def _quantili_empirical(support: list[float]) -> dict[str, float]:
    valori = sorted(support)
    n = len(valori)

    def _quantile(q: float) -> float:
        if n == 1:
            return valori[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return valori[lo] * (1 - frac) + valori[hi] * frac

    return {f"p{int(q * 100)}": _quantile(q) for q in _Z_QUANTILI_STANDARD}


def _quantili_prior(mu: float, sigma: float) -> dict[str, float]:
    return {f"p{int(q * 100)}": math.exp(mu + sigma * z) for q, z in _Z_QUANTILI_STANDARD.items()}


def genera_report(
    players: list[Player],
    distribuzioni: list[PriceDistribution],
    output_path: Path | str,
    min_osservazioni: int,
    esempi_curva: list[str] | None = None,
    note_extra: list[str] | None = None,
) -> Path:
    """Scrive `bidmodel_report.md` in `output_path` e lo ritorna.

    `esempi_curva`: lista opzionale di `player_id`, per stampare qualche
    riga della curva `p_win(b)` di quei giocatori specifici (oltre alla
    tabella per fascia, che è sempre presente).
    `note_extra`: righe markdown aggiuntive (es. dimostrazioni della
    degradazione automatica sotto soglia K) inserite in fondo al report.
    """
    output_path = Path(output_path)
    by_fascia: dict[str, list[PriceDistribution]] = defaultdict(list)
    for d in distribuzioni:
        by_fascia[d.fascia].append(d)

    tutte_synthetic = all(d.is_synthetic for d in distribuzioni) if distribuzioni else True
    tutte_prior = (
        all(d.mode is PriceDistributionMode.PRIOR for d in distribuzioni) if distribuzioni else True
    )

    righe: list[str] = []
    righe.append("# Modulo C — Report diagnostico del modello di offerte avversarie")
    righe.append("")
    righe.append(f"Generato: {datetime.now(UTC).isoformat(timespec='seconds')}")
    righe.append("")

    if tutte_synthetic:
        righe.append(
            "> ⚠️ **Tutti i dati in questo report sono SINTETICI** — fixture "
            "costruite inline dai test/demo dell'Agente C, non dati reali "
            "della lega. Vedi CLAUDE.md, \"Avviso sui dati\"."
        )
        righe.append("")
    if tutte_prior:
        righe.append(
            "> ⚠️ **Ogni fascia qui sotto gira in modalità `prior`.** Questo è "
            f"lo stato reale del Modulo C per la stagione 2026/27: nessuno "
            f"storico reale di offerte avversarie esiste per questa lega "
            f"(soglia richiesta: >= {min_osservazioni} osservazioni per "
            "fascia — verificato 2026-08-25, vedi docs/OPEN_QUESTIONS.md §2 "
            "e §2.1). Le curve sottostanti sono centrate sul listone con "
            "varianza ampia e DICHIARATA, non un edge informativo sulle "
            "offerte reali degli avversari — non vanno presentate come tali."
        )
        righe.append("")

    righe.append("## Fasce")
    righe.append("")
    righe.append("| Fascia | Modalità | n_osservazioni | p10 | p25 | p50 (mediana) | p75 | p90 |")
    righe.append("|---|---|---|---|---|---|---|---|")
    for fascia in sorted(by_fascia):
        d = by_fascia[fascia][0]
        if d.mode is PriceDistributionMode.EMPIRICAL:
            q = _quantili_empirical(d.empirical_support)
        else:
            assert d.prior_mu is not None and d.prior_sigma is not None
            q = _quantili_prior(d.prior_mu, d.prior_sigma)
        righe.append(
            f"| `{fascia}` | {d.mode.value} | {d.n_osservazioni} | "
            f"{q['p10']:.2f} | {q['p25']:.2f} | {q['p50']:.2f} | {q['p75']:.2f} | {q['p90']:.2f} |"
        )
    righe.append("")
    righe.append(
        "Quantili del rapporto `offerta / quotazione_listone` (1.00 = offerta "
        "pari alla quotazione del listone). In modalità `prior` sono impliciti "
        "dalla lognormale (`prior_mu`, `prior_sigma`), non osservati — in "
        "modalità `empirical` sono i quantili del supporto empirico "
        "(un valore per lotto osservato: il massimo fra le offerte avversarie "
        "su quel lotto, vedi `fitting.costruisci_lot_max_ratios`)."
    )
    righe.append("")

    if esempi_curva:
        righe.append("## Esempi di curva p_win(b) per giocatore")
        righe.append("")
        player_by_id = {p.player_id: p for p in players}
        dist_by_id = {d.player_id: d for d in distribuzioni}
        for pid in esempi_curva:
            p = player_by_id.get(pid)
            d = dist_by_id.get(pid)
            if p is None or d is None:
                continue
            righe.append(
                f"### {p.nome} ({p.player_id}, fascia `{d.fascia}`, "
                f"{d.mode.value}, n_osservazioni={d.n_osservazioni})"
            )
            righe.append("")
            righe.append("| offerta | rapporto offerta/quotazione | p_win |")
            righe.append("|---|---|---|")
            for frac in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
                b = frac * p.quotazione_listone
                pw = d.p_win(b, p.quotazione_listone)
                righe.append(f"| {b:.1f} | {frac:.2f} | {pw:.3f} |")
            righe.append("")

    if note_extra:
        righe.append("## Note")
        righe.append("")
        righe.extend(note_extra)
        righe.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(righe) + "\n", encoding="utf-8")
    return output_path
