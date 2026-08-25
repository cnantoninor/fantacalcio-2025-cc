"""Discretizzazione dei livelli di offerta per il MILP del Modulo D.

Il problema "vero" avrebbe una variabile binaria x[i,b] per ogni giocatore i
e ogni intero b da 1 al budget totale. Con budget_totale_fase1 = 500 e 600
giocatori (fixture) questo darebbe fino a 600 * 500 = 300.000 variabili
binarie — la "esplosione combinatoria" che docs/DESIGN.md (Agente D, punto 2)
chiede esplicitamente di evitare.

**Trade-off dichiarato**: invece di enumerare ogni intero, generiamo una
griglia di livelli di offerta *relativi alla quotazione del giocatore*
(es. 0.5x, 1.0x, 1.5x la quotazione), e la densità della griglia dipende
dalla **fascia di prezzo** del giocatore — pochi punti per i giocatori a
basso costo (dove il margine di manovra assoluto in crediti è comunque
piccolo), più punti per i giocatori costosi (dove offrire 60 invece di 65
può davvero cambiare la p_win e vale la pena distinguerli). Questo è ciò
che DESIGN.md intende per "griglia adattiva per fascia di prezzo".

Conseguenza accettata: l'ottimo trovato dal MILP è l'ottimo **sulla griglia
discretizzata**, non l'ottimo sull'insieme continuo (o intero-completo) dei
possibili valori di offerta. Un'offerta ottima "vera" che cadesse a metà tra
due punti della griglia non verrà mai proposta. In pratica l'errore
introdotto è piccolo perché p_win(b) è una funzione smooth (CDF lognormale o
ECDF) e i punti della griglia sono relativamente fitti attorno a 1x la
quotazione, dove la parte interessante della curva si trova.

Il credito è comunque un'unità intera nella piattaforma (si offrono crediti,
non frazioni di credito): ogni livello della griglia viene quindi arrotondato
all'intero più vicino, e 1 credito è il minimo strutturale di un'offerta
positiva — non un parametro di regolamento, quindi non va cercato in
LeagueConfig.
"""

from __future__ import annotations

MIN_OFFERTA = 1
"""Minimo strutturale: un'offerta è espressa in crediti interi positivi."""

# Confini di fascia in crediti di quotazione. Costanti dell'algoritmo di
# discretizzazione, non parametri di regolamento di lega — non vanno in
# LeagueConfig (a differenza di budget, rosa, max_giocatori_per_squadra).
_FASCIA_BASSA_MAX = 10.0
_FASCIA_MEDIA_MAX = 30.0

# Moltiplicatori relativi alla quotazione, uno per fascia. Più fitti per le
# fasce alte: lì la posta in gioco assoluta è più alta e la differenza tra
# offrire 0.9x o 1.1x la quotazione conta di più.
_GRIGLIA_BASSA: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
_GRIGLIA_MEDIA: tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5)
_GRIGLIA_ALTA: tuple[float, ...] = (
    0.3,
    0.5,
    0.7,
    0.85,
    1.0,
    1.15,
    1.3,
    1.5,
    1.8,
    2.2,
    2.6,
    3.0,
)


def _moltiplicatori_per_fascia(quotazione: float) -> tuple[float, ...]:
    if quotazione < _FASCIA_BASSA_MAX:
        return _GRIGLIA_BASSA
    if quotazione < _FASCIA_MEDIA_MAX:
        return _GRIGLIA_MEDIA
    return _GRIGLIA_ALTA


def griglia_offerte(quotazione_listone: float, budget_massimo: float) -> list[int]:
    """Livelli di offerta candidati (interi, crediti) per un giocatore.

    Ritorna una lista ordinata di interi unici in [MIN_OFFERTA,
    floor(budget_massimo)], derivati come `quotazione_listone * moltiplicatore`
    per ogni moltiplicatore della fascia di prezzo del giocatore (vedi modulo
    docstring). Può ritornare una lista vuota se `budget_massimo < MIN_OFFERTA`
    (nessuna offerta possibile) — il chiamante deve gestirlo.
    """
    tetto = int(budget_massimo)
    if tetto < MIN_OFFERTA:
        return []

    moltiplicatori = _moltiplicatori_per_fascia(quotazione_listone)
    valori = {
        max(MIN_OFFERTA, min(tetto, round(quotazione_listone * m))) for m in moltiplicatori
    }
    return sorted(valori)
