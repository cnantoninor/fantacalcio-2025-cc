"""Baseline trasparente — modello PRIMARIO del Modulo B (metodo=baseline).

Vedi docs/DESIGN.md, Agente B: con 2 sole stagioni storiche un walk-forward
degenera in un solo fold, quindi il baseline non è un fallback ma il default
di produzione. Nessun training, nessun parametro fittato sui dati: ogni
costante qui sotto è una scelta di modellazione dichiarata (non un numero di
regolamento — quelli vengono sempre da LeagueConfig/league.yaml), ispezionabile
e motivata nel commento accanto.

Pipeline per giocatore (docs/DESIGN.md, Agente B, punto 2):
1. media pesata delle 2 stagioni (peso maggiore alla più recente);
2. shrinkage verso la media di ruolo per chi ha pochi minuti;
3. aggiustamento per cambio squadra (vedi limite di contratto sotto);
4. moltiplicazione per prob_titolarita.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantabuste.valuation.features import MINUTI_STAGIONE_PIENA, PlayerFeatures

# --- Costanti di modellazione (non di regolamento: qui non c'entra league.yaml) ---

PESO_STAGIONE_RECENTE = 0.65
"""Peso della stagione più recente nella media pesata a 2 stagioni (punto 1).
Il resto (0.35) va alla stagione precedente. Scelta dichiarata: la stagione
più recente è più informativa sul livello attuale del giocatore, ma 2
stagioni sono comunque 2 osservazioni — non fittato, non ottimizzato su dati."""

K_SHRINKAGE_MINUTI = 900.0
"""Costante di shrinkage (in minuti) verso la media di ruolo: circa 10
partite intere. Peso attribuito al dato individuale = minuti / (minuti + K).
Con K=900: un giocatore con 900 minuti pesa il suo dato individuale al 50%,
uno con 400 minuti (l'esempio di DESIGN.md, "attaccante con 400 minuti e
fantamedia 8") pesa il suo dato solo ~31%, il resto è media di ruolo — evita
di scambiare un campione minuscolo per un livello di rendimento vero."""

MINUTI_CONFIDENZA_ALTA = 2 * K_SHRINKAGE_MINUTI
"""Soglia di minuti totali (su tutte le stagioni disponibili) sopra la quale
il baseline dichiara confidence_flag='alta'."""
MINUTI_CONFIDENZA_BASSA = 0.5 * K_SHRINKAGE_MINUTI
"""Sotto questa soglia di minuti totali, confidence_flag='bassa'."""

Z_INTERVALLO = 1.28
"""Moltiplicatore della deviazione standard di ruolo per costruire
punti_attesi_lo/hi (punto 5): 1.28 ~ intervallo all'80% sotto normalità
approssimata. Con solo 2 stagioni non pretendiamo calibrazione esatta —
l'obiettivo è un intervallo onestamente largo, non un valore di copertura
preciso al percento."""

FATTORE_ALLARGAMENTO_BASSA_CONFIDENZA = 1.6
"""Moltiplica ulteriormente la larghezza dell'intervallo per i giocatori a
bassa confidenza (pochi minuti, shrinkage forte): la dispersione di ruolo da
sola sottostima l'incertezza quando il dato individuale è quasi assente."""

FATTORE_CAMBIO_SQUADRA = 1.0
"""Aggiustamento per cambio squadra (punto 3 di DESIGN.md): NO-OP
DELIBERATO, non più un limite di contratto. `PlayerStats.squadra` porta ora
la squadra per stagione (aggiunta dopo la segnalazione dell'Agente B in Fase
1 — vedi docs/DESIGN.md), quindi un cambio squadra fra le stagioni storiche
è rilevabile confrontando `StagioneFeatures.squadra` fra le stagioni in
`PlayerFeatures.stagioni` (features.py). Il fattore resta 1.0 perché
tradurre "ha cambiato squadra" in
uno sconto numerico sulla proiezione è una scelta di modellazione (di
quanto? in che direzione? dipende dal campionato di provenienza?) non
ancora specificata, non perché il dato manchi. Segnalato nel report
(valuation_report.md), non nascosto."""


@dataclass(frozen=True)
class MediaDispersioneRuolo:
    """Media e deviazione standard della fantamedia pesata, per ruolo, sulla
    popolazione osservata — usata sia per lo shrinkage sia per gli intervalli
    di incertezza (dispersione storica per fascia di ruolo, non una quantile
    regression fittata sullo stesso campione minuscolo — DESIGN.md punto 5)."""

    ruolo: str
    media: float
    deviazione_standard: float
    n_giocatori: int


@dataclass(frozen=True)
class ProiezioneBaseline:
    """Output grezzo del baseline per un giocatore, prima della validazione
    Pydantic in pipeline.py (che assembla il PlayerProjection finale)."""

    player_id: str
    punti_attesi: float
    punti_attesi_lo: float
    punti_attesi_hi: float
    prob_titolarita: float
    confidence_flag: str
    minuti_totali: int
    fantamedia_pesata_grezza: float


def _media_pesata_stagioni(pf: PlayerFeatures) -> float | None:
    """Media pesata delle fantamedie disponibili, peso maggiore alla più
    recente. Con 1 sola stagione disponibile la media è semplicemente quella
    stagione (degradazione esplicita, non un errore). Con 0 stagioni None."""
    if not pf.stagioni:
        return None
    if len(pf.stagioni) == 1:
        return pf.stagioni[0].fantamedia
    # Le stagioni sono ordinate dalla più vecchia alla più recente
    # (features.py); con 2 stagioni: [vecchia, recente].
    recente = pf.stagioni[-1]
    vecchia = pf.stagioni[-2]
    return (
        PESO_STAGIONE_RECENTE * recente.fantamedia
        + (1 - PESO_STAGIONE_RECENTE) * vecchia.fantamedia
    )


def _prob_titolarita(pf: PlayerFeatures) -> float:
    """Stima di prob_titolarita dai minuti storici (proxy dichiarata: nessuno
    schema fornisce una probabilità di titolarità futura, la stimiamo dal
    tasso di utilizzo recente, più peso alla stagione più recente, coerente
    col resto del baseline). Clampata a [0, 1]."""
    if not pf.stagioni:
        return 0.0
    if len(pf.stagioni) == 1:
        return pf.stagioni[0].quota_minuti_stagione
    recente = pf.stagioni[-1]
    vecchia = pf.stagioni[-2]
    quota = (
        PESO_STAGIONE_RECENTE * recente.quota_minuti_stagione
        + (1 - PESO_STAGIONE_RECENTE) * vecchia.quota_minuti_stagione
    )
    return max(0.0, min(1.0, quota))


def stima_prob_titolarita(pf: PlayerFeatures) -> float:
    """Alias pubblico di `_prob_titolarita`, per riuso da altri moduli (es.
    `pipeline.py` quando l'ML sostituisce solo la stima di fantamedia attesa
    e riusa comunque questa stima di prob_titolarita — vedi pipeline.py)."""
    return _prob_titolarita(pf)


def calcola_medie_dispersione_ruolo(
    features: list[PlayerFeatures],
) -> dict[str, MediaDispersioneRuolo]:
    """Media e deviazione standard (campionaria) della fantamedia pesata a 2
    stagioni, raggruppata per ruolo, sui soli giocatori con almeno una
    stagione osservata. Usata per lo shrinkage e per gli intervalli."""
    valori_per_ruolo: dict[str, list[float]] = {}
    for pf in features:
        media = _media_pesata_stagioni(pf)
        if media is None:
            continue
        valori_per_ruolo.setdefault(pf.ruolo, []).append(media)

    out: dict[str, MediaDispersioneRuolo] = {}
    for ruolo, valori in valori_per_ruolo.items():
        n = len(valori)
        media = sum(valori) / n
        if n > 1:
            varianza = sum((v - media) ** 2 for v in valori) / (n - 1)
        else:
            varianza = 0.0
        out[ruolo] = MediaDispersioneRuolo(
            ruolo=ruolo, media=media, deviazione_standard=varianza**0.5, n_giocatori=n
        )
    return out


def punteggio_atteso_pre_titolarita(
    pf: PlayerFeatures, medie_ruolo: dict[str, MediaDispersioneRuolo]
) -> tuple[float, float, int]:
    """Passi 1-3 del baseline (media pesata, shrinkage, aggiustamento cambio
    squadra) SENZA il passo 4 (moltiplicazione per prob_titolarita).

    Isolato in una funzione propria perché riusato dal backtest di `ml.py`
    per confrontare baseline ed ML sullo stesso target (la fantamedia
    osservata della stagione di test, che non è "scontata" per probabilità
    di titolarità — quella è una stima senza ground truth nel contratto dati,
    vedi ml.py). Ritorna (punteggio_aggiustato, fantamedia_grezza, minuti_totali).
    """
    media_ruolo = medie_ruolo.get(pf.ruolo)
    media_ruolo_valore = media_ruolo.media if media_ruolo else 6.0
    # 6.0 = voto di sufficienza standard Fantacalcio, usato SOLO come pavimento
    # di fallback se un ruolo non ha alcuna osservazione nel dataset corrente
    # (non dovrebbe accadere con le fixture, ma un ruolo raro in dati reali
    # potrebbe arrivare vuoto qui) — dichiarato, non un numero di regolamento.

    fantamedia_grezza = _media_pesata_stagioni(pf)
    minuti_totali = pf.minuti_totali

    # --- passo 1+2: media pesata + shrinkage verso la media di ruolo -------
    if fantamedia_grezza is None:
        # Nessuna stagione osservata: shrinkage totale, alpha=0.
        punteggio_shrunk = media_ruolo_valore
        fantamedia_grezza_out = media_ruolo_valore
    else:
        alpha = minuti_totali / (minuti_totali + K_SHRINKAGE_MINUTI)
        punteggio_shrunk = alpha * fantamedia_grezza + (1 - alpha) * media_ruolo_valore
        fantamedia_grezza_out = fantamedia_grezza

    # --- passo 3: aggiustamento cambio squadra (vedi FATTORE_CAMBIO_SQUADRA) -
    punteggio_aggiustato = punteggio_shrunk * FATTORE_CAMBIO_SQUADRA
    return punteggio_aggiustato, fantamedia_grezza_out, minuti_totali


def proietta_baseline(
    pf: PlayerFeatures, medie_ruolo: dict[str, MediaDispersioneRuolo]
) -> ProiezioneBaseline:
    """Applica i 4 passi del baseline (docstring del modulo) a un giocatore."""
    media_ruolo = medie_ruolo.get(pf.ruolo)
    sigma_ruolo = media_ruolo.deviazione_standard if media_ruolo else 1.0

    punteggio_aggiustato, fantamedia_grezza_out, minuti_totali = (
        punteggio_atteso_pre_titolarita(pf, medie_ruolo)
    )

    # --- passo 4: moltiplicazione per prob_titolarita -----------------------
    prob_tit = _prob_titolarita(pf)
    punti_attesi = punteggio_aggiustato * prob_tit

    # --- intervalli di incertezza (punto 5) ---------------------------------
    # Dispersione storica per fascia di ruolo, allargata se la confidenza è
    # bassa (shrinkage forte = dato individuale quasi assente).
    largo = Z_INTERVALLO * sigma_ruolo * prob_tit
    if minuti_totali < MINUTI_CONFIDENZA_BASSA:
        largo *= FATTORE_ALLARGAMENTO_BASSA_CONFIDENZA
    largo = max(largo, 0.05)  # evita intervalli degeneri a punto singolo
    punti_attesi_lo = max(0.0, punti_attesi - largo)
    punti_attesi_hi = punti_attesi + largo

    if minuti_totali >= MINUTI_CONFIDENZA_ALTA and pf.n_stagioni_osservate == 2:
        confidence_flag = "alta"
    elif minuti_totali < MINUTI_CONFIDENZA_BASSA or pf.n_stagioni_osservate == 0:
        confidence_flag = "bassa"
    else:
        confidence_flag = "media"

    return ProiezioneBaseline(
        player_id=pf.player_id,
        punti_attesi=round(punti_attesi, 4),
        punti_attesi_lo=round(punti_attesi_lo, 4),
        punti_attesi_hi=round(punti_attesi_hi, 4),
        prob_titolarita=round(prob_tit, 4),
        confidence_flag=confidence_flag,
        minuti_totali=minuti_totali,
        fantamedia_pesata_grezza=round(fantamedia_grezza_out, 4),
    )


def proietta_baseline_tutti(
    features: list[PlayerFeatures],
) -> tuple[list[ProiezioneBaseline], dict[str, MediaDispersioneRuolo]]:
    """Applica proietta_baseline a tutta la popolazione, calcolando prima le
    medie/dispersioni di ruolo su cui si basano shrinkage e intervalli."""
    medie_ruolo = calcola_medie_dispersione_ruolo(features)
    proiezioni = [proietta_baseline(pf, medie_ruolo) for pf in features]
    return proiezioni, medie_ruolo


__all__ = [
    "MINUTI_STAGIONE_PIENA",
    "PESO_STAGIONE_RECENTE",
    "K_SHRINKAGE_MINUTI",
    "Z_INTERVALLO",
    "FATTORE_CAMBIO_SQUADRA",
    "MediaDispersioneRuolo",
    "ProiezioneBaseline",
    "calcola_medie_dispersione_ruolo",
    "stima_prob_titolarita",
    "punteggio_atteso_pre_titolarita",
    "proietta_baseline",
    "proietta_baseline_tutti",
]
