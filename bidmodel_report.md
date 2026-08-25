# Modulo C — Report diagnostico del modello di offerte avversarie

Generato: 2026-08-25T22:14:38+00:00

> ⚠️ **Tutti i dati in questo report sono SINTETICI** — fixture costruite inline dai test/demo dell'Agente C, non dati reali della lega. Vedi CLAUDE.md, "Avviso sui dati".

## Fasce

| Fascia | Modalità | n_osservazioni | p10 | p25 | p50 (mediana) | p75 | p90 |
|---|---|---|---|---|---|---|---|
| `A_tier1` | empirical | 64 | 0.65 | 0.70 | 0.82 | 1.02 | 1.15 |
| `C_tier1` | prior | 0 | 0.53 | 0.71 | 1.00 | 1.40 | 1.90 |
| `D_tier1` | prior | 0 | 0.53 | 0.71 | 1.00 | 1.40 | 1.90 |
| `P_tier1` | prior | 0 | 0.53 | 0.71 | 1.00 | 1.40 | 1.90 |

Quantili del rapporto `offerta / quotazione_listone` (1.00 = offerta pari alla quotazione del listone). In modalità `prior` sono impliciti dalla lognormale (`prior_mu`, `prior_sigma`), non osservati — in modalità `empirical` sono i quantili del supporto empirico (un valore per lotto osservato: il massimo fra le offerte avversarie su quel lotto, vedi `fitting.costruisci_lot_max_ratios`).

## Esempi di curva p_win(b) per giocatore

### Attaccante Demo 0 (DEMO_A000, fascia `A_tier1`, empirical, n_osservazioni=64)

| offerta | rapporto offerta/quotazione | p_win |
|---|---|---|
| 5.0 | 0.50 | 0.000 |
| 8.0 | 0.80 | 0.391 |
| 10.0 | 1.00 | 0.688 |
| 12.0 | 1.20 | 0.906 |
| 15.0 | 1.50 | 1.000 |
| 20.0 | 2.00 | 1.000 |

### P Demo 0 (DEMO_P000, fascia `P_tier1`, prior, n_osservazioni=0)

| offerta | rapporto offerta/quotazione | p_win |
|---|---|---|
| 2.5 | 0.50 | 0.083 |
| 4.0 | 0.80 | 0.328 |
| 5.0 | 1.00 | 0.500 |
| 6.0 | 1.20 | 0.642 |
| 7.5 | 1.50 | 0.791 |
| 10.0 | 2.00 | 0.917 |

## Note

**Dimostrazione della degradazione automatica sotto soglia K** (`config.bidmodel_min_osservazioni = 30`): una fascia ipotetica con 10 osservazioni (< 30) ottiene `mode=prior`, `degradata_a_prior=True` — nessun errore, nessuna distribuzione empirica corta spacciata per affidabile.

Fasce di questo report: `assegna_fasce` è stata chiamata con `n_tier=1` (una fascia per ruolo) solo per tenere piccola la fixture dimostrativa — il binning per quantili con `N_TIER_DEFAULT=3` è quello usato in produzione ed è esercitato dai test unitari di `fasce.py`.

