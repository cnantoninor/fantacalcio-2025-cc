"""Simulazione Monte Carlo a livello di intera asta.

Il MILP del Modulo D tratta `p_win(b)` come indipendente fra giocatori
(vedi il docstring di `PriceDistribution.p_win` in schemas.py, già scritto
in Fase 0). Non lo è: gli avversari hanno un budget CONDIVISO fra tutti i
lotti, quindi vincere un lotto riduce la capacità di offrire su un altro
— la `p_win` di due giocatori "costosi" per lo stesso profilo di
avversario è correlata negativamente. Questo modulo stima quella
correlazione simulando N aste COMPLETE (ogni avversario alloca il proprio
budget su una lista di preferenze), non giocatore per giocatore — vedi
docs/DESIGN.md, Agente C, punto 3. È implementazione reale e testata, non
un placeholder: è lo strumento che quantifica il gap che D dichiara come
approssimazione, non lo elimina.

Approssimazione dichiarata nel modello di comportamento degli avversari:
allocazione greedy del budget in ordine di offerta candidata decrescente,
senza vincoli di composizione di rosa per ruolo (quelli sono lo scope del
MILP di D, non di questa simulazione). Basta a catturare l'effetto
principale — budget condiviso implica correlazione negativa fra `p_win` —
non pretende di replicare il comportamento avversario reale, che è
sconosciuto (vedi docs/OPEN_QUESTIONS.md §2: zero osservazioni reali per
calibrarlo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fantabuste.schemas import Player, PriceDistribution, PriceDistributionMode


def _campiona_rapporti(
    dist: PriceDistribution, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Campiona `n` rapporti offerta/quotazione per un avversario "medio"
    della fascia di `dist`. Bootstrap dal supporto empirico in modalità
    `empirical`, lognormale in modalità `prior` — la stessa distribuzione
    che `PriceDistribution.p_win` usa per la CDF, qui campionata invece
    che integrata in forma chiusa."""
    if dist.mode is PriceDistributionMode.EMPIRICAL:
        support = np.asarray(dist.empirical_support, dtype=float)
        return rng.choice(support, size=n, replace=True)
    assert dist.prior_mu is not None and dist.prior_sigma is not None
    return rng.lognormal(mean=dist.prior_mu, sigma=dist.prior_sigma, size=n)


@dataclass(frozen=True)
class RisultatoMonteCarlo:
    """Esito di `simula_asta`: N aste complete simulate, con la matrice
    booleana delle vittorie (una riga per simulazione, una colonna per
    giocatore) e le statistiche derivate."""

    player_ids: list[str]
    n_simulazioni: int
    n_avversari: int
    seed: int
    vittorie: np.ndarray
    """[n_simulazioni, n_players] booleano: True se `mie_offerte[player]`
    avrebbe vinto il lotto in quella simulazione."""
    p_win_simulata: dict[str, float]
    p_win_indipendente: dict[str, float]
    """`PriceDistribution.p_win` calcolata direttamente (l'approssimazione
    indipendente che il MILP di D usa) — il termine di paragone per
    quantificare il gap che la simulazione cattura."""
    matrice_correlazione: np.ndarray
    """[n_players, n_players]: correlazione di Pearson fra gli indicatori
    di vittoria simulati. La diagonale è 1 per costruzione; le fuori
    diagonale negative sono il segnale del budget condiviso."""

    def indice(self, player_id: str) -> int:
        return self.player_ids.index(player_id)

    def gap_indipendenza(self, player_id_a: str, player_id_b: str) -> float:
        """`P(vinco entrambi)` simulata meno `P(vinco entrambi)` assumendo
        indipendenza (prodotto delle marginali) — quantifica esattamente
        il gap che il MILP di D ignora per costruzione (vedi il suo
        docstring del solver). Negativo quando i due giocatori competono
        per lo stesso budget avversario più spesso di quanto
        l'indipendenza predirebbe."""
        i, j = self.indice(player_id_a), self.indice(player_id_b)
        congiunta_simulata = float(np.mean(self.vittorie[:, i] & self.vittorie[:, j]))
        congiunta_indipendente = (
            self.p_win_indipendente[player_id_a] * self.p_win_indipendente[player_id_b]
        )
        return congiunta_simulata - congiunta_indipendente


def simula_asta(
    players: list[Player],
    price_distributions: dict[str, PriceDistribution],
    mie_offerte: dict[str, float],
    n_avversari: int,
    budget_avversario: float,
    n_simulazioni: int,
    seed: int,
) -> RisultatoMonteCarlo:
    """Simula `n_simulazioni` aste complete e stima, per ogni giocatore in
    `players`, la probabilità che `mie_offerte[player_id]` vinca — insieme
    alla correlazione fra le vittorie sui diversi giocatori.

    Per ogni simulazione e ogni avversario simulato (`n_avversari`, tipicamente
    `config.n_partecipanti - 1`):
    1. campiona un rapporto offerta/quotazione per OGNI giocatore in
       `players` dalla `PriceDistribution` della sua fascia (il "quanto
       sarebbe disposto a offrire se ci provasse" di un avversario medio
       di quella fascia);
    2. converte in un'offerta candidata = rapporto × quotazione_listone;
    3. alloca il budget dell'avversario in modo greedy, in ordine di
       offerta candidata decrescente, finché il budget residuo lo
       consente — un avversario con budget limitato NON può "provare" su
       tutti i giocatori che vorrebbe: è esattamente il meccanismo che
       genera la correlazione che questo modulo vuole quantificare. Chi
       non entra nel budget non fa un'offerta ridotta sul lotto — la
       rinuncia, non tronca l'offerta al budget residuo (troncare
       sarebbe irrealistico: un'offerta bassa a caso non è come non
       partecipare);
    4. per ogni giocatore, confronta `mie_offerte[i]` con il massimo fra
       le offerte EFFETTIVAMENTE piazzate (dopo il vincolo di budget) dai
       `n_avversari` simulati su quel giocatore.

    Riproducibile da seed: usa `numpy.random.Generator` seedato
    esplicitamente (mai lo stato globale di `numpy.random`), quindi stesso
    seed => stesso risultato byte-per-byte.
    """
    if n_simulazioni <= 0:
        raise ValueError("n_simulazioni deve essere positivo")
    if n_avversari <= 0:
        raise ValueError("n_avversari deve essere positivo")
    if budget_avversario <= 0:
        raise ValueError("budget_avversario deve essere positivo")
    if not players:
        raise ValueError("players non può essere vuoto")

    mancanti_dist = [p.player_id for p in players if p.player_id not in price_distributions]
    if mancanti_dist:
        raise ValueError(f"PriceDistribution mancante per: {mancanti_dist}")
    mancanti_offerte = [p.player_id for p in players if p.player_id not in mie_offerte]
    if mancanti_offerte:
        raise ValueError(f"offerta mancante per: {mancanti_offerte}")

    rng = np.random.default_rng(seed)
    player_ids = [p.player_id for p in players]
    quotazioni = np.array([p.quotazione_listone for p in players])
    n_players = len(players)
    mie = np.array([mie_offerte[pid] for pid in player_ids])

    vittorie = np.zeros((n_simulazioni, n_players), dtype=bool)

    for s in range(n_simulazioni):
        max_avversario = np.zeros(n_players)
        for _ in range(n_avversari):
            rapporti = np.array(
                [_campiona_rapporti(price_distributions[pid], rng, 1)[0] for pid in player_ids]
            )
            candidate = rapporti * quotazioni
            ordine = np.argsort(-candidate)  # priorità: offerta candidata decrescente
            budget_residuo = budget_avversario
            offerta_avversario = np.zeros(n_players)
            for j in ordine:
                if candidate[j] <= budget_residuo:
                    offerta_avversario[j] = candidate[j]
                    budget_residuo -= candidate[j]
                # altrimenti: rinuncia a questo giocatore, prova il successivo
            max_avversario = np.maximum(max_avversario, offerta_avversario)
        vittorie[s] = mie > max_avversario

    p_win_simulata = {pid: float(np.mean(vittorie[:, j])) for j, pid in enumerate(player_ids)}
    p_win_indipendente = {
        pid: price_distributions[pid].p_win(mie_offerte[pid], q)
        for pid, q in zip(player_ids, quotazioni.tolist(), strict=True)
    }

    if n_players > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            matrice_correlazione = np.corrcoef(vittorie.T.astype(float))
        # colonne costanti (sempre vinto o sempre perso in tutte le
        # simulazioni) danno deviazione standard nulla -> NaN in
        # corrcoef; sono per definizione incorrelate con tutto il resto.
        matrice_correlazione = np.nan_to_num(matrice_correlazione, nan=0.0)
        np.fill_diagonal(matrice_correlazione, 1.0)
    else:
        matrice_correlazione = np.array([[1.0]])

    return RisultatoMonteCarlo(
        player_ids=player_ids,
        n_simulazioni=n_simulazioni,
        n_avversari=n_avversari,
        seed=seed,
        vittorie=vittorie,
        p_win_simulata=p_win_simulata,
        p_win_indipendente=p_win_indipendente,
        matrice_correlazione=matrice_correlazione,
    )
