"""Sfidante ML opzionale — metodo=ml (docs/DESIGN.md, Agente B, punto 3).

Vincolo centrale del modulo (vedi anche baseline.py e docs/DESIGN.md): con 2
sole stagioni storiche un walk-forward ha **un solo fold temporale** (allena
su stagione 1, testa su stagione 2 — l'unica coppia storica disponibile). Un
modello con tante feature su un campione così piccolo overfitta quasi per
costruzione, e un miglioramento del 2-3% di MAE su un fold solo è rumore, non
segnale. Questo modulo perciò:

- usa al massimo `MASSIMO_FEATURE` feature (vedi `FEATURE_NAMES`), con
  regolarizzazione forte (Ridge, `RIDGE_ALPHA` alto);
- per stimare l'errore fuori campione usa K-fold cross-sezionale TRA
  GIOCATORI dentro l'unico fold temporale S1->S2 disponibile — questo non
  moltiplica i fold temporali (ce n'è comunque solo uno), serve solo a non
  valutare il modello sugli stessi dati su cui è stato fittato;
- confronta l'errore ML con l'errore del baseline sullo STESSO target
  (fantamedia osservata in stagione 2, calcolata dal baseline con la sola
  stagione 1 come storico — vedi `baseline.punteggio_atteso_pre_titolarita`);
- **rifiuta automaticamente** l'ML se non batte il baseline con un margine
  ampio (default: MAE inferiore di almeno il 10%, `MARGINE_MIGLIORAMENTO_MINIMO`)
  — il codice rifiuta da solo, non si limita a segnalarlo nel report.

Richiede scikit-learn (dipendenza opzionale `ml` in pyproject.toml, gruppo
`[project.optional-dependencies]`). Se non installato, `valuta_sfidante_ml`
solleva `MLNonDisponibile` con un messaggio chiaro: il chiamante (pipeline.py)
deve trattarlo come "ML non valutato" in modo esplicito, mai come "ML
rifiutato" (sono due esiti diversi, non vanno confusi nel report).
"""

from __future__ import annotations

from dataclasses import dataclass

from fantabuste.valuation.baseline import (
    calcola_medie_dispersione_ruolo,
    punteggio_atteso_pre_titolarita,
)
from fantabuste.valuation.features import PlayerFeatures, StagioneFeatures

MASSIMO_FEATURE = 8
"""Tetto esplicito da docs/DESIGN.md ('poche feature (≤8)'). FEATURE_NAMES
sotto ne usa esattamente 8 — se qualcuno prova ad aggiungerne una nona il test
di modulo deve fallire (vedi tests/test_valuation.py)."""

FEATURE_NAMES: tuple[str, ...] = (
    "fantamedia",
    "quota_minuti",
    "gol_per90",
    "assist_per90",
    "xG_per90",
    "is_D",
    "is_C",
    "is_A",
)
"""8 feature dalla sola stagione più vecchia delle due disponibili (quella
usata come "storico" nel fold S1->S2). Ruolo codificato one-hot con P come
categoria di riferimento (is_D/is_C/is_A, 3 dummy per 4 ruoli) invece di un
intero ordinale: i ruoli non hanno un ordine naturale e un ordinale
implicherebbe erroneamente che C è "tra" D e A."""

RIDGE_ALPHA = 10.0
"""Forza di regolarizzazione L2 (Ridge). Alto di proposito: con ~600
osservazioni e 8 feature il rischio di overfitting su questo singolo fold è
comunque concreto — vedi il vincolo centrale del modulo. Non ottimizzato via
grid-search: farlo significherebbe fittare l'iperparametro sullo stesso
campione minuscolo che DESIGN.md ci mette in guardia dal fidarci."""

N_FOLD_CV = 5
"""Numero di fold della cross-validation cross-sezionale (tra giocatori)
usata per stimare l'errore fuori campione dell'ML. Non è un secondo fold
temporale: il fold temporale resta uno solo (S1->S2)."""

SEED_CV = 42
"""Seed fisso per la K-fold: risultati riproducibili, coerente con lo
standard 'riproducibile da seed' del resto del progetto."""

MARGINE_MIGLIORAMENTO_MINIMO_DEFAULT = 0.10
"""Margine minimo di miglioramento del MAE (frazione) richiesto all'ML sul
baseline per essere accettato in produzione — default 10%, da DESIGN.md:
'un miglioramento del 2-3% su un fold solo è rumore, non segnale'."""


class MLNonDisponibile(RuntimeError):
    """scikit-learn non è installato. Vedi `pip install -e '.[ml]'` in
    CLAUDE.md / le istruzioni dell'Agente B."""


@dataclass(frozen=True)
class RisultatoSfidanteML:
    """Esito del confronto baseline vs ML sul fold singolo S1->S2."""

    n_osservazioni: int
    mae_baseline: float
    rmse_baseline: float
    mae_ml: float
    rmse_ml: float
    margine_richiesto: float
    accettato: bool
    motivo: str
    predizioni_ml_per_player_id: dict[str, float]
    """Predizioni ML fuori campione (cross-val) sul fold di backtest — usate
    solo per ispezione/report, MAI per produrre PlayerProjection: le
    proiezioni di produzione ri-fittano il modello (se accettato) su
    entrambe le stagioni, vedi pipeline.py."""


def _vettore_feature(s: StagioneFeatures, ruolo: str) -> list[float]:
    return [
        s.fantamedia,
        s.quota_minuti_stagione,
        s.gol_per90,
        s.assist_per90,
        s.xG_per90,
        1.0 if ruolo == "D" else 0.0,
        1.0 if ruolo == "C" else 0.0,
        1.0 if ruolo == "A" else 0.0,
    ]


def _dataset_backtest(
    features: list[PlayerFeatures],
) -> tuple[list[str], list[list[float]], list[float]]:
    """Coppie (feature da stagione più vecchia, target = fantamedia stagione
    più recente) per i soli giocatori con entrambe le stagioni disponibili —
    è l'unico fold temporale che i dati permettono."""
    player_ids: list[str] = []
    X: list[list[float]] = []
    y: list[float] = []
    for pf in features:
        if len(pf.stagioni) != 2:
            continue
        vecchia, recente = pf.stagioni[0], pf.stagioni[1]
        player_ids.append(pf.player_id)
        X.append(_vettore_feature(vecchia, pf.ruolo))
        y.append(recente.fantamedia)
    return player_ids, X, y


def _features_solo_prima_stagione(features: list[PlayerFeatures]) -> list[PlayerFeatures]:
    """Copia le PlayerFeatures troncando allo storico disponibile nel fold di
    backtest (solo la stagione più vecchia) — usata per calcolare la
    previsione del BASELINE nello stesso confronto, così i due modelli
    vedono esattamente lo stesso storico."""
    out = []
    for pf in features:
        if len(pf.stagioni) < 2:
            continue
        out.append(
            PlayerFeatures(
                player_id=pf.player_id,
                nome=pf.nome,
                ruolo=pf.ruolo,
                squadra=pf.squadra,
                quotazione_listone=pf.quotazione_listone,
                fonte_player=pf.fonte_player,
                is_synthetic=pf.is_synthetic,
                stagioni=(pf.stagioni[0],),
            )
        )
    return out


def _mae_rmse(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    n = len(y_true)
    errori = [yt - yp for yt, yp in zip(y_true, y_pred, strict=True)]
    mae = sum(abs(e) for e in errori) / n
    rmse = (sum(e**2 for e in errori) / n) ** 0.5
    return mae, rmse


def valuta_sfidante_ml(
    features: list[PlayerFeatures],
    margine_miglioramento_minimo: float = MARGINE_MIGLIORAMENTO_MINIMO_DEFAULT,
) -> RisultatoSfidanteML:
    """Esegue il backtest a fold singolo S1->S2 e decide se l'ML batte il
    baseline con margine sufficiente. Non modifica `features`.

    Solleva `MLNonDisponibile` se scikit-learn non è installato — il
    chiamante deve distinguere questo caso da un ML "valutato e rifiutato".
    """
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise MLNonDisponibile(
            "scikit-learn non installato: installa con `pip install -e '.[ml]'` "
            "(o `pip install scikit-learn`) per usare --enable-ml."
        ) from exc

    player_ids, X_list, y_list = _dataset_backtest(features)
    n = len(player_ids)
    if n < N_FOLD_CV * 2:
        return RisultatoSfidanteML(
            n_osservazioni=n,
            mae_baseline=float("nan"),
            rmse_baseline=float("nan"),
            mae_ml=float("nan"),
            rmse_ml=float("nan"),
            margine_richiesto=margine_miglioramento_minimo,
            accettato=False,
            motivo=(
                f"Solo {n} giocatori con 2 stagioni complete: insufficienti per "
                f"{N_FOLD_CV}-fold cross-validation. ML non valutato, baseline resta primario."
            ),
            predizioni_ml_per_player_id={},
        )

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    assert X.shape[1] == MASSIMO_FEATURE == len(FEATURE_NAMES)

    modello = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    cv = KFold(n_splits=N_FOLD_CV, shuffle=True, random_state=SEED_CV)
    y_pred_ml = cross_val_predict(modello, X, y, cv=cv)

    # Baseline sullo stesso identico fold e target: shrinkage/media-ruolo
    # calcolati SOLO sulla stagione più vecchia (lo storico visibile nel
    # backtest), non sulle 2 stagioni intere (sarebbe una fuga di
    # informazione dal futuro).
    features_storiche = _features_solo_prima_stagione(features)
    medie_ruolo_storiche = calcola_medie_dispersione_ruolo(features_storiche)
    by_id = {pf.player_id: pf for pf in features_storiche}
    y_pred_baseline = [
        punteggio_atteso_pre_titolarita(by_id[pid], medie_ruolo_storiche)[0] for pid in player_ids
    ]

    mae_baseline, rmse_baseline = _mae_rmse(y_list, y_pred_baseline)
    mae_ml, rmse_ml = _mae_rmse(y_list, list(y_pred_ml))

    soglia = mae_baseline * (1 - margine_miglioramento_minimo)
    if mae_baseline <= 0:
        accettato = False
        motivo = "MAE baseline nulla o negativa: confronto non significativo, ML rifiutato."
    elif mae_ml < soglia:
        riduzione = (mae_baseline - mae_ml) / mae_baseline
        accettato = True
        motivo = (
            f"MAE ML ({mae_ml:.4f}) inferiore alla soglia richiesta ({soglia:.4f} = "
            f"baseline {mae_baseline:.4f} × (1 - {margine_miglioramento_minimo:.0%})): "
            f"riduzione effettiva {riduzione:.1%}. Accettato."
        )
    else:
        accettato = False
        motivo = (
            f"MAE ML ({mae_ml:.4f}) non inferiore alla soglia richiesta ({soglia:.4f} = "
            f"baseline {mae_baseline:.4f} × (1 - {margine_miglioramento_minimo:.0%})). "
            "Rifiutato automaticamente: il baseline resta il metodo di produzione."
        )

    return RisultatoSfidanteML(
        n_osservazioni=n,
        mae_baseline=round(mae_baseline, 4),
        rmse_baseline=round(rmse_baseline, 4),
        mae_ml=round(mae_ml, 4),
        rmse_ml=round(rmse_ml, 4),
        margine_richiesto=margine_miglioramento_minimo,
        accettato=accettato,
        motivo=motivo,
        predizioni_ml_per_player_id=dict(
            zip(player_ids, (round(float(v), 4) for v in y_pred_ml), strict=True)
        ),
    )


def fitta_modello_produzione(features: list[PlayerFeatures]):
    """Rifitta l'ML su TUTTI i dati disponibili (entrambe le stagioni come
    'storico', fantamedia più recente come target implicito nella stessa
    parametrizzazione feature) da usare per proiettare la stagione futura,
    SOLO dopo che `valuta_sfidante_ml` ha accettato il modello. Ritorna una
    pipeline scikit-learn fittata; il chiamante applica `_vettore_feature`
    sull'ultima stagione disponibile per proiettare in avanti.

    Nota di onestà: qui non esiste un target osservabile per la stagione
    futura (2026/27) per definizione — questo rifit usa la stessa
    parametrizzazione validata nel backtest, applicata all'ultima stagione
    osservata come input, esattamente come il baseline usa la stagione più
    recente come base della sua media pesata.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise MLNonDisponibile(
            "scikit-learn non installato: installa con `pip install -e '.[ml]'`."
        ) from exc

    player_ids, X_list, y_list = _dataset_backtest(features)
    if not X_list:
        raise ValueError("Nessun giocatore con 2 stagioni complete: impossibile fittare l'ML.")
    modello = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    modello.fit(X_list, y_list)
    return modello


def proietta_ml_produzione(
    features: list[PlayerFeatures], modello
) -> dict[str, float]:
    """Applica il modello fittato (`fitta_modello_produzione`) all'ultima
    stagione disponibile di ogni giocatore, per proiettare in avanti.
    Giocatori senza alcuna stagione osservata non sono predetti qui: la
    pipeline li lascia al baseline (shrinkage totale verso la media di
    ruolo), coerente col fatto che l'ML non ha alcuna feature da leggere per
    loro."""
    out: dict[str, float] = {}
    for pf in features:
        if not pf.stagioni:
            continue
        ultima = pf.stagioni[-1]
        vettore = [_vettore_feature(ultima, pf.ruolo)]
        pred = modello.predict(vettore)[0]
        out[pf.player_id] = round(float(pred), 4)
    return out


__all__ = [
    "MASSIMO_FEATURE",
    "FEATURE_NAMES",
    "RIDGE_ALPHA",
    "MARGINE_MIGLIORAMENTO_MINIMO_DEFAULT",
    "MLNonDisponibile",
    "RisultatoSfidanteML",
    "valuta_sfidante_ml",
    "fitta_modello_produzione",
    "proietta_ml_produzione",
]
