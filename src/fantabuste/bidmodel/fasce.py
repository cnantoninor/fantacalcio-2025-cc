"""Fasce di quotazione — la chiave di raggruppamento del Modulo C.

Una fascia è `ruolo × tier di quotazione` (es. `"A_tier1"`, `"D_tier3"`) —
vedi docs/DESIGN.md, Agente C, e il docstring di `PriceDistribution.fascia`
in schemas.py.

Scelta di modellazione dichiarata: i tier sono per QUANTILI di
`quotazione_listone` calcolati separatamente per ciascun ruolo
(equal-frequency binning), non per soglie assolute di crediti. Soglie
assolute (es. "tier1 = quotazione >= 30") non si generalizzerebbero a
budget di lega diversi o a listoni di stagioni diverse — lo stesso motivo
per cui il Modulo C modella un *rapporto* (`offerta / quotazione_listone`)
invece di un valore assoluto. `tier1` = quotazione più alta del ruolo (i
big-name), `tier{N}` = quotazione più bassa.
"""

from __future__ import annotations

from fantabuste.schemas import Player, Ruolo

N_TIER_DEFAULT = 3
"""Numero di fasce di quotazione per ruolo, di default.

Scelta di modellazione dichiarata (NON un parametro di regolamento di
lega, quindi non vive in `league.yaml`): 3 fasce (es. "top", "medio",
"scarto") è un compromesso fra granularità — fasce più fini frazionano
ulteriormente osservazioni già scarse, il problema esatto che
`config.bidmodel_min_osservazioni` esiste per intercettare — e potere
discriminante. Chi vuole sperimentare può passare `n_tier` esplicito alle
funzioni di questo modulo.
"""


def nome_fascia(ruolo: Ruolo, tier: int) -> str:
    """Chiave canonica di una fascia: `'{ruolo}_tier{N}'`."""
    return f"{ruolo}_tier{tier}"


def assegna_fasce(players: list[Player], n_tier: int = N_TIER_DEFAULT) -> dict[str, str]:
    """Ritorna `player_id -> fascia`, con `tier1` = quotazione più alta del ruolo.

    Binning per quantili (equal-frequency) di `quotazione_listone`,
    calcolato separatamente per ogni ruolo sulla popolazione passata in
    `players`. Con meno di `n_tier` giocatori in un ruolo, alcuni tier
    restano semplicemente vuoti (nessun player_id vi è assegnato) — non è
    un errore, è il comportamento atteso su popolazioni piccole (es. i
    test di questo modulo).

    Deterministico a parità di input: a parità di quotazione, l'ordine è
    spareggiato da `player_id` per garantire un output riproducibile.
    """
    if n_tier < 1:
        raise ValueError("n_tier deve essere >= 1")

    per_ruolo: dict[Ruolo, list[Player]] = {}
    for p in players:
        per_ruolo.setdefault(p.ruolo, []).append(p)

    fasce: dict[str, str] = {}
    for ruolo, gruppo in per_ruolo.items():
        ordinati = sorted(gruppo, key=lambda p: (-p.quotazione_listone, p.player_id))
        n = len(ordinati)
        for idx, p in enumerate(ordinati):
            # idx=0 -> quotazione più alta -> tier1. Ripartizione a
            # frequenza costante: ogni tier riceve circa n/n_tier giocatori.
            tier = 1 + (idx * n_tier) // n
            tier = min(tier, n_tier)
            fasce[p.player_id] = nome_fascia(ruolo, tier)
    return fasce
