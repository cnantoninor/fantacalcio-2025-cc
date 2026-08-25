# FANTABUSTE — Piano di progetto per agenti Claude Code

> ⚠️ **AVVISO SUI DATI — LEGGERE PRIMA DI TUTTO**
> Tutti i dati numerici presenti in questo documento e in qualsiasi documento di ricerca allegato al progetto (percentuali di titolarità, gol attesi, prezzi medi d'asta, tabelle di allocazione del budget per reparto, nomi/prezzi di giocatori) sono **puramente esemplificativi**. Servono a definire *forma, tipo e unità di misura* dei dati, **non** a essere usati come input reali.
> Ogni agente deve trattarli come **fixture sintetiche**. Nessun numero di questo documento entra in un modello, in un backtest o in un'offerta reale senza essere prima sostituito da un dato con **fonte primaria verificabile** (export ufficiale del listone, dataset con URL diretto, storico d'asta esportato dalla propria lega).
> Se un modulo non ha dati reali disponibili, deve **fallire in modo esplicito o degradare a una modalità dichiaratamente approssimata** — mai proseguire silenziosamente su fixture.

---

## PARTE A — Fondamenta concettuali (solo ciò che ha superato il vaglio critico)

Queste sono le uniche assunzioni su cui il codice può poggiare. Tutto il resto (architettura multi-agente LangGraph, formula di equilibrio di Nash come regola operativa, claim di performance di tool terzi, tabelle di allocazione budget "empiriche" non verificate) è stato **rimosso** e non va reintrodotto.

### A1. Principi solidi
- **Bid shading**: in un'asta a busta chiusa first-price, l'offerta ottimale è *sotto* la propria valutazione reale. Offrire esattamente il valore stimato azzera il surplus.
- **Winner's curse**: il rischio principale non è perdere una busta, è vincerne una pagando sopra il valore. L'obiettivo del sistema è **massimizzare il surplus atteso aggregato sulla rosa**, non il tasso di vittoria delle singole buste.
- **Regola dei pareggi**: nel regolamento standard Leghe Fantacalcio, offerte migliori pari ⇒ giocatore **non assegnato**, torna nella tornata successiva. Da qui l'euristica delle offerte non tonde. **Altre piattaforme usano first-come-first-served o sorteggio**: la regola è un parametro di configurazione, non una costante.
- **Dinamica delle tornate**: dopo il round 1 il budget residuo si concentra su un bacino di giocatori più povero ⇒ inflazione sui comprimari. Il sistema deve essere **stateful tra le tornate**.

### A2. Principi validi ma con vincolo esplicito
- **VORP (Value Over Replacement Player)** — `VORP_i = E[Punti_i] − E[Punti_rimpiazzo_ruolo]`. Concettualmente corretto per confrontare ruoli diversi, **ma** il "livello di rimpiazzo" in una lega da 8-10 persone non è un mercato liquido: dipende interamente dalla composizione della tua lega. ⇒ Il replacement level dev'essere un **parametro configurabile e calibrabile sulla propria lega**, mai una costante hardcoded.
- **MILP per l'allocazione del budget** — solido e pronto all'uso; è ciò che i progetti open source reali effettivamente fanno. **Ma** tratta `P(win|b)` come indipendente tra giocatori, mentre in realtà il budget avversario è condiviso tra tutti i lotti (le probabilità sono correlate). ⇒ Va documentato nel codice come **euristica ad approssimazione dichiarata**, non come "soluzione globale ottima".
- **Modellazione della distribuzione delle offerte avversarie (lognormale / KDE)** — rigorosa **solo in presenza di storico reale di prezzi pagati** (non le quotazioni del listone, che sono valori suggeriti). Senza storico, produce precisione finta. ⇒ Il modulo deve avere due modalità nettamente separate e etichettate: `empirical` (con dati) e `prior` (senza dati, con incertezza dichiaratamente ampia).

### A4. Struttura reale della tua asta (confermata)

- **Fase busta chiusa**: esattamente **2 tornate** (non un numero variabile). Ogni giocatore non assegnato dopo la tornata 2 passa alla fase successiva.
- **Visibilità completa**: dopo l'apertura di ogni tornata **vedi tutte le offerte, vincenti e perdenti**, non solo il prezzo pagato dal vincitore. Questo è il dato più prezioso del progetto — vedi il punto A2 sopra e il Modulo C aggiornato.
- **Asta di riparazione**: dopo le 2 tornate, i giocatori rimasti si assegnano con un'**asta classica a tempo, giocatore per giocatore** — non più busta chiusa. È un meccanismo **strutturalmente diverso**: è un'asta inglese/ascendente (English auction) con prezzo corrente visibile e rilanci, non una FPSBA. La teoria del bid shading **non si applica** a questa fase: in un'asta ascendente a valore privato la strategia debolmente dominante è offrire fino al proprio valore reale e fermarsi, non sotto-offrire. Il problema qui non è "quanto nascondere" ma **quale sia il vero valore netto**, dato che il budget è condiviso con tutti i giocatori ancora da acquisire in questa fase e nelle successive. Va progettato come modulo separato (Modulo F), non forzato dentro l'apparato lognormale/KDE/MILP costruito per le buste chiuse.
  - **Meccanica confermata**: **più lotti "a tempo" sono aperti in contemporanea**, non un giocatore alla volta. Il timer è **fisso, senza reset al rilancio** (nessun soft-close).
  - **Niente protezione da sniping**: dato che un rilancio non estende il timer, non c'è penalità a offrire all'ultimo istante utile — anzi c'è un vantaggio (non inneschi contro-rilanci avversari con tempo di reagire). È lo stesso fenomeno documentato nella letteratura sulle aste online a scadenza fissa stile eBay (Roth & Ockenfels): con timer fisso conviene offrire tardi e una sola volta, non presto e per gradini.
  - **Ma con più lotti simultanei e budget condiviso, lo sniping puro crea un problema di coordinamento**: se due lotti che vuoi entrambi chiudono in una finestra ravvicinata, rischi di vincerli entrambi sforando il budget, o di dover scegliere quale snipare sapendo che l'altro andrà perso. Serve una **priorità pre-calcolata**, non solo un tetto per lotto isolato.
  - Il Modulo F va quindi ridisegnato attorno a un **riottimizzatore di portafoglio sui lotti correntemente aperti**, non attorno a un singolo tetto sequenziale — vedi sezione Modulo F aggiornata.
  - **Decisione presa sul carico di inserimento**: non tracci tutti i lotti aperti — tracci solo una **watchlist di obiettivi prioritari**, generata automaticamente (non curata a mano da te prima dell'asta) dal Modulo F stesso a partire da `AuctionState` + `PlayerProjection`: i giocatori ancora disponibili con il maggior surplus atteso, dimensione configurabile. Questo risolve anche il problema di conflitto-di-tempo del punto sopra: i conflitti da segnalare sono solo quelli **dentro la watchlist**, non su tutto il listone residuo.
- **Piattaforma confermata**: Leghe Fantacalcio® Serie A (Fantacalcio s.r.l., app ufficiale). Questo risolve una domanda aperta: il regolamento ufficiale di questa piattaforma è quello già verificato in fase di ricerca — **pareggio ⇒ giocatore non assegnato**, torna nella tornata successiva. `regola_pareggio: nessuna_assegnazione` è confermato, non più ipotetico.
- **Nessuna API pubblica, e scraping esplicitamente vietato.** Fantacalcio s.r.l. non espone un'API di terze parti. La piattaforma web (`leghe.fantacalcio.it`) esiste ed è tecnicamente raggiungibile da un browser automatizzato (es. Playwright) — è stata valutata questa opzione e **scartata deliberatamente**: i Termini di Utilizzo di Fantacalcio vietano esplicitamente l'uso di "programmi software o altri meccanismi automatici o manuali per copiare o accedere alle pagine della Piattaforma... ivi compresi sistemi atti a effettuare il c.d. scraping" senza autorizzazione scritta, con conseguenze che includono sospensione/cessazione dell'account e una clausola di manleva. Il rischio concreto — un ban rilevato **durante l'asta vera**, nel momento peggiore possibile — non vale il guadagno di comodità. **Decisione presa e definitiva: Modulo F a inserimento manuale, nessuno scraping, nessuna automazione di lettura della piattaforma.** Non riaprire questa opzione senza una ragione nuova e esplicita.
- **Canale utile per il resto della pipeline**: la piattaforma offre import/export di rose in CSV/file (sezione "Gestione Rose", import a fine asta) — verificare con l'admin di lega se è disponibile anche un export dello storico di mercato delle stagioni passate (per il Modulo A) oltre alle rose correnti.

### A3. Sanità di base
- Nessuna metrica del sistema è mai stata validata causalmente ("chi usa questo approccio vince più leghe"). Il valore atteso realistico è **ridurre errori sistematici** (sovrapagamento, squilibrio di rosa), non garantire vittorie.
- Il vantaggio è **relativo agli avversari**: si erode se anche loro adottano approcci simili.

---

## PARTE B — Architettura del repo

Architettura **volutamente semplice**: pipeline a moduli con contratti dati espliciti. Niente orchestrazione ad agenti runtime, niente state machine — la complessità di quel tipo non ha superato il vaglio critico.

```
fantabuste/
├── CLAUDE.md                  # regole di progetto + regole della MIA lega
├── pyproject.toml
├── config/
│   ├── league.example.yaml    # regolamento lega (parametrizzato)
│   └── league.yaml            # gitignored — il mio vero regolamento
├── data/
│   ├── raw/                   # gitignored — export reali
│   ├── fixtures/              # dati SINTETICI per test (versionati)
│   └── processed/             # gitignored
├── src/fantabuste/
│   ├── schemas.py             # ⚠️ CONTRATTO CONDIVISO — congelato dopo Fase 0
│   ├── ingest/                # Modulo A
│   ├── valuation/             # Modulo B
│   ├── bidmodel/              # Modulo C
│   ├── optimize/              # Modulo D
│   ├── auction/               # Modulo E (integrazione + tornate)
│   ├── repair/                # Modulo F (asta di riparazione, live)
│   └── cli.py                 # Modulo E
└── tests/
    ├── test_ingest.py  test_valuation.py  test_bidmodel.py
    ├── test_optimize.py  test_integration.py  test_repair.py
```

### Contratti dati (`schemas.py`) — da definire in Fase 0 e **congelare**

Questo è il punto critico per la parallelizzazione: se i contratti sono stabili, gli agenti lavorano in isolamento senza collisioni.

| Schema | Prodotto da | Consumato da | Campi essenziali |
|---|---|---|---|
| `Player` | A | B, C, D | `player_id, nome, ruolo, squadra, quotazione_listone, fonte, data_estrazione` |
| `PlayerStats` | A | B | `player_id, stagione, presenze, minuti, gol, assist, xG, xA, fantamedia, rigori_battuti` |
| `PlayerProjection` | B | C, D | `player_id, punti_attesi, punti_attesi_lo, punti_attesi_hi, prob_titolarita, vorp, metodo, confidence_flag` |
| `PriceDistribution` | C | D | `player_id, mode ('empirical'\|'prior'), p_win(b) -> float, supporto, n_osservazioni` |
| `BidPlan` | D | E | `player_id, offerta, p_win_stimata, vorp, surplus_atteso, tornata` |
| `AuctionState` | E | B, C, D, F | `budget_residuo, slot_residui_per_ruolo, giocatori_assegnati, budget_residuo_avversari (esatto, non stimato), slot_residui_avversari_per_ruolo` |
| `OpponentBidObservation` | E | C | `player_id, tornata, avversario_id, offerta, vincente: bool` — **una riga per OGNI offerta osservata, non solo le vincenti** |
| `RepairLotState` | F (interno) | F | `player_id, prezzo_corrente, offerente_corrente, orario_chiusura (assoluto, non un countdown relativo — necessario per confrontare lotti simultanei), storico_rilanci` — **collezione di più lotti aperti in parallelo**, non un singolo stato |
| `RepairLotResult` | F | E (per `run_log.json`) | `player_id, prezzo_finale, vinto_da_me: bool, avversario_vincitore_id (se non io), orario_chiusura` — **una riga per ogni lotto chiuso**, vinto o perso, per il backtest dell'anno prossimo |

**Regola d'oro per ogni schema**: ogni record porta `fonte` e `is_synthetic: bool`. Il sistema **rifiuta di emettere un `BidPlan` finale** se una qualsiasi dipendenza a monte ha `is_synthetic=True`, a meno di flag esplicito `--allow-synthetic`.

---

## PARTE C — Work order per agenti paralleli

### Meccanica di parallelizzazione su Claude Code
- **Fase 0 sequenziale, un solo agente.** Nessuno parte prima che i contratti siano congelati.
- Fasi 1: **un git worktree + branch per agente** (`git worktree add ../fantabuste-A feat/ingest`). Evita conflitti sul filesystem tra sessioni concorrenti.
- Ogni agente **possiede** i suoi file e **non tocca** quelli altrui. `schemas.py` è **read-only per tutti** dopo la Fase 0: se un agente ritiene che il contratto sia sbagliato, **si ferma e lo segnala** invece di modificarlo unilateralmente.
- Ogni agente scrive test che girano **solo sulle fixture sintetiche**, così è indipendente dalla disponibilità di dati reali.

---

### FASE 0 — Fondamenta (sequenziale, 1 agente)

**Scope**: scaffold repo, `pyproject.toml`, `schemas.py` con tutti i dataclass/Pydantic model della tabella sopra, `config/league.example.yaml` completo di tutti i parametri di regolamento, generatore di fixture sintetiche (~600 giocatori finti con distribuzioni plausibili, tutti marcati `is_synthetic=True`), setup pytest + ruff. **Copia questo intero documento di progetto in `docs/DESIGN.md`** (o un estratto fedele): è la memoria del perché delle scelte — baseline primario in B, K≥30 in C, niente scraping in F — e senza di esso un agente Claude Code che riprende il lavoro tra un anno riparte da zero.

**Parametri obbligatori in `league.yaml`**: `budget_totale`, `n_partecipanti`, `rosa: {P,D,C,A}` (+ `T` se la lega usa il trequartista), `regola_pareggio: nessuna_assegnazione|primo_arrivato|sorteggio` *(ancora da confermare — vedi Parte F)*, `n_tornate_buste: 2` (fisso, confermato), `modificatore_difesa: bool`, `max_giocatori_per_squadra`, `offerte_non_tonde: bool`, `asta_riparazione: {tipo: tempo, sequenziale: bool, soft_close: bool, timer_secondi}` *(valori di default da confermare — vedi Parte F)*.

**Definition of done**: `pytest` passa su test placeholder; `python -m fantabuste.cli --help` funziona; le fixture si generano riproducibilmente da un seed; `CLAUDE.md` contiene la sezione "Avviso sui dati" in cima.

---

### FASE 1 — Quattro moduli parallelizzabili (A-D)

**Decisione presa: F riusa direttamente il codice del solver di D**, quindi non è più parte del parallelismo di Fase 1 — aspetta che D sia mergiato. Il parallelismo reale di questa fase è A, B, C, D (4 agenti, non 5).

#### 🅰️ AGENTE A — Ingestion & normalizzazione
**Possiede**: `src/fantabuste/ingest/`, `tests/test_ingest.py`
**Input**: file locali caricati dall'utente (CSV/XLSX del listone) + eventuali fetch da fonti pubbliche
**Output**: `Player[]` + `PlayerStats[]` validati contro schema

Compiti:
1. Parser per l'export del listone (CSV/XLSX) → `Player[]`, robusto a colonne mancanti e a variazioni di formato tra stagioni.
2. Fetch statistiche storiche via `soccerdata` (FBref/Understat, Serie A) → `PlayerStats[]`.
3. **Fuzzy matching dei nomi tra fonti** — è il punto in cui questi progetti falliscono più spesso. Deve produrre un `match_report.csv` che elenca esplicitamente i mismatch e i match a bassa confidenza per revisione manuale. Non deve mai fondere silenziosamente due giocatori diversi.
4. Validazione: range plausibili, duplicati, giocatori senza squadra.

**Vincoli espliciti**: i file in `data/raw/` sono **read-only**; ogni trasformazione scrive un nuovo file in `data/processed/`. Rispettare i ToS delle fonti, rate limiting e caching locale delle risposte.
**DoD**: gira end-to-end sulle fixture; `match_report.csv` prodotto; ≥80% coverage sul modulo; nessuna scrittura in `data/raw/`.

---

#### 🅱️ AGENTE B — Valutazione (punti attesi + VORP)
**Possiede**: `src/fantabuste/valuation/`, `tests/test_valuation.py`
**Input**: `Player[]`, `PlayerStats[]` (usa le fixture, non aspettare A)
**Output**: `PlayerProjection[]`

> 🔒 **Vincolo dati confermato: 2 sole stagioni storiche.** Questo cambia lo scope del modulo. Con 2 stagioni il walk-forward degenera in **un solo fold** (allena su stagione 1, testa su stagione 2): non è validazione, è un singolo esperimento non replicabile. Un gradient boosting con decine di feature su un campione così piccolo **overfitta quasi per costruzione** e non c'è modo di accorgersene. Di conseguenza il baseline non è il fallback: **è il default**, e l'ML è un candidato sfidante che deve superare un bar alto.

Compiti:
1. Feature engineering: fantamedia storica, minuti, xG/xA, gol/assist per 90, ruolo, squadra.
2. **Baseline trasparente come modello PRIMARIO** (`metodo: baseline`): media pesata delle 2 stagioni (peso maggiore alla più recente), regressione verso la media di ruolo per chi ha pochi minuti (shrinkage — un attaccante con 400 minuti e fantamedia 8 non è un attaccante da 8), aggiustamento per cambio squadra, moltiplicazione per `prob_titolarita`. Nessun training, nessun overfitting, output ispezionabile a mano.
3. **Modello ML come sfidante opzionale** (`metodo: ml`), dietro flag `--enable-ml`, con **poche feature (≤8) e regolarizzazione forte**. Entra in produzione **solo se** batte il baseline sul fold singolo con un margine configurabile e ampio (default: MAE inferiore di almeno il 10%). Un miglioramento del 2-3% su un fold solo è rumore, non segnale — il codice deve rifiutarlo.
4. **Report obbligatorio** `valuation_report.md`: MAE/RMSE di baseline e ML, numero di osservazioni, e una sezione esplicita "perché questo risultato è debole" che dichiara il limite del singolo fold. Serve a te fra sei mesi, quando avrai dimenticato quanto era sottile il campione.
5. Intervalli di incertezza → `punti_attesi_lo/hi`. Con 2 stagioni gli intervalli devono essere **larghi e onesti**: derivarli dalla dispersione storica osservata per fascia di ruolo, non da una quantile regression addestrata sullo stesso campione minuscolo.
6. **VORP con replacement level configurabile**: parametrizzato su `n_partecipanti` e slot di rosa da `league.yaml`, con override manuale. Documentare nel codice il limite noto (campione piccolo, mercato non liquido).

**DoD**: baseline funzionante e primario; ML dietro flag e rifiutato automaticamente se non supera il margine; `valuation_report.md` con la sezione sui limiti; intervalli di incertezza larghi e giustificati; VORP configurabile; nessun numero hardcoded preso da documenti di ricerca.

---

#### 🅲 AGENTE C — Modello delle offerte avversarie
**Possiede**: `src/fantabuste/bidmodel/`, `tests/test_bidmodel.py`
**Input**: `PlayerProjection[]` + (opzionale) storico prezzi reali
**Output**: `PriceDistribution[]` con funzione `p_win(b)`

Compiti:
> 🔒 **Aggiornamento con dato confermato: vedi TUTTE le offerte, non solo quella vincente, dopo ognuna delle 2 tornate.** Questo elimina il problema di selezione che affliggerebbe un dataset di soli prezzi vincenti (dove osserveresti solo il massimo della distribuzione, non la distribuzione). Con offerte complete su 2-3 stagioni, per fascia hai centinaia di osservazioni **della distribuzione vera**, non solo della sua coda destra. Cambia il design in meglio: puoi fittare la distribuzione empirica direttamente, e in più hai il materiale per un modulo opzionale di profilazione per-avversario (chi è aggressivo su che ruolo) — utile ma non indispensabile al primo giro, va tenuto come stretch goal per non scivolare di nuovo nell'over-engineering già bocciato in Parte 2 del documento di ricerca originale.

1. **Modalità `empirical` — modellazione a livello di FASCIA, con dati completi.**
   - Variabile target: il **rapporto** `offerta / quotazione_listone` (o `offerta / VORP`), confrontabile tra giocatori e stagioni.
   - Raggruppa per **fascia** (ruolo × tier di quotazione). Con offerte complete (vincenti + perdenti) su 2-3 stagioni, ogni fascia ha decine o centinaia di osservazioni: **abbastanza per usare la distribuzione empirica direttamente** (ECDF) come base, con lognormale/KDE come smoothing sopra, non come sostituto necessario per scarsità di dati.
   - `p_win(b)` per il giocatore i = probabilità empirica (fascia di i) che l'offerta massima altrui sia < `b / quotazione_i`, stimata **direttamente dalle offerte multiple osservate per tornata** — non serve elevare alla n per approssimare il massimo tra n avversari: **lo hai già osservato per davvero**, tornata per tornata. Questo è un salto di qualità reale rispetto alla versione precedente del progetto.
   - **Normalizzazione dell'inflazione tra stagioni**: se budget o partecipanti sono cambiati, normalizzare prima di poolare.
   - **Stretch goal (facoltativo, solo dopo che il resto funziona)**: profilo per-avversario (`OpponentBidObservation` aggregato per `avversario_id`) — quanto sopra/sotto fascia offre tipicamente ciascun avversario, per ruolo. Non bloccare la Fase 1 su questo.
2. **Modalità `prior`** — attiva se manca lo storico: distribuzione centrata sulla quotazione del listone con varianza **ampia e dichiarata**. Ogni output di questa modalità porta un flag visibile e l'incertezza NON va nascosta nel report finale. Non spacciare precisione che non c'è.
3. **Simulazione Monte Carlo a livello di intera asta**, non giocatore per giocatore: gli avversari hanno un budget condiviso tra tutti i lotti, quindi simulare N aste complete in cui ogni avversario alloca il proprio budget su una lista di preferenze. Serve a stimare la **correlazione** tra le `p_win` che il MILP non cattura.
4. Output di diagnostica: per ogni giocatore, curva `p_win(b)` e `n_osservazioni` su cui è basata.

**DoD**: entrambe le modalità implementate e chiaramente etichettate; il codice **rifiuta di fittare una fascia con meno di K osservazioni** (K configurabile, default 30) e degrada quella fascia a `prior`; prezzi storici normalizzati tra stagioni; Monte Carlo riproducibile da seed; test che verificano la monotonicità di `p_win(b)`; `bidmodel_report.md` con, per ogni fascia, `n_osservazioni` e la curva fittata.

---

#### 🅳 AGENTE D — Ottimizzatore MILP
**Possiede**: `src/fantabuste/optimize/`, `tests/test_optimize.py`
**Input**: `PlayerProjection[]` + `PriceDistribution[]` (usa fixture)
**Output**: `BidPlan[]`

Compiti:
1. Modello MILP con PuLP (solver CBC di default):
   - Variabili binarie `x[i,b]` = offro `b` crediti per il giocatore `i`.
   - Obiettivo: `max Σ (VORP_i × p_win(b)) · x[i,b]`
   - Vincoli: max un'offerta per giocatore; `Σ b·x[i,b] ≤ budget`; composizione rosa esatta da `league.yaml`; `max_giocatori_per_squadra`.
2. **Discretizzazione dei livelli di offerta**: non enumerare tutti gli interati da 1 a 500 per 600 giocatori (esplosione combinatoria). Usare una griglia adattiva per fascia di prezzo. Documentare il trade-off.
3. **Post-processing offerte non tonde**: se `league.yaml` ha `regola_pareggio: nessuna_assegnazione` e `offerte_non_tonde: true`, spostare le offerte lontano dai multipli di 5/10 (es. 40 → 41). Verificare che il vincolo di budget regga dopo l'aggiustamento.
4. Produrre **le top-5 rose alternative**, non solo l'ottimo, con il delta di utilità attesa. Un ottimo singolo su input incerti è fragile.
5. **Analisi di sensibilità**: quanto cambia la rosa se le proiezioni si spostano di ±1 deviazione standard? Se l'ottimo è instabile, dirlo esplicitamente.

**Vincolo di onestà nel codice**: il docstring del solver deve dichiarare che `p_win` è trattata come indipendente tra giocatori mentre nella realtà è correlata via budget avversario ⇒ **euristica, non ottimo globale**.
**DoD**: risolve in <30s su 600 giocatori con fixture; rispetta tutti i vincoli di rosa; top-5 alternative; analisi di sensibilità; test che verificano infeasibility gestita con messaggio chiaro (non crash).

---

### FASE 2 — E e F in parallelo (dopo il merge di A-D)

E non dipende dal codice di F, e F ha già tutto ciò che gli serve una volta che D è mergiato (il solver) e gli schemi sono congelati (`AuctionState`, `RepairLotState`). Possono procedere insieme.

#### 🅴 AGENTE E — CLI, stato tornate, report
**Possiede**: `src/fantabuste/auction/`, `src/fantabuste/cli.py`, `tests/test_integration.py`

Compiti:
1. CLI end-to-end: `fantabuste prepara --config league.yaml` → pipeline A→B→C→D → report.
2. **Gestione delle 2 tornate busta chiusa (stateful)**: `fantabuste tornata --risultati round1.csv` — il file risultati contiene **tutte le offerte, vincenti e perdenti** (schema `OpponentBidObservation`), non solo il prezzo pagato. Aggiorna `AuctionState`: budget residuo proprio, giocatori rimossi, slot rimasti, e **budget residuo esatto di ogni avversario** (non stimato — è calcolabile con precisione dalla somma di quanto ha speso, dato che si vede tutto), poi ri-esegue C+D per la tornata 2 con i vincoli aggiornati.
3. **Chiusura della fase busta chiusa e handoff al Modulo F**: dopo la tornata 2, `fantabuste chiudi-buste` congela `AuctionState` finale (budget e slot residui, propri e di ogni avversario, esatti) e lo passa al Modulo F per l'asta di riparazione. Questo stato è il pezzo di maggior valore per F: sapere con precisione quanto budget ha ancora ciascun avversario prima di un'asta a tempo è un vantaggio informativo reale.
4. **Guardrail sui dati sintetici**: il comando finale rifiuta di emettere offerte se una dipendenza a monte ha `is_synthetic=True`, salvo `--allow-synthetic`, e in quel caso stampa un banner inequivocabile su ogni pagina del report.
5. Report finale (Markdown o HTML) per la fase busta chiusa: per ogni obiettivo → offerta consigliata, `p_win` stimata, VORP, surplus atteso, **flag di confidenza della fonte**, e una sezione "cosa NON so".
6. Un `run_log.json` per stagione — **ora comprende sia le 2 tornate busta chiusa (via `OpponentBidObservation`) sia l'esito di ogni lotto della riparazione (via `RepairLotResult` da F)**, così il backtest dell'anno prossimo copre l'intera asta, non solo la fase busta chiusa. **Non tagliare questo pezzo.**
7. **Attenzione a non mischiare i due formati quando li riusi**: le offerte busta chiusa (FPSBA) e i prezzi finali della riparazione (asta ascendente) sono generati da meccanismi diversi — un prezzo di riparazione non è un'offerta busta chiusa "bassa", è il risultato di una dinamica diversa. Il Modulo C dell'anno prossimo deve fittare le sue fasce **solo su `OpponentBidObservation`**, mai su `RepairLotResult`, a meno di una decisione esplicita e separata.

**DoD**: `fantabuste prepara` gira end-to-end su fixture; guardrail sintetici testato; ciclo delle 2 tornate testato; handoff a `AuctionState` finale verificato con test dedicato.

---

#### 🅵 AGENTE F — Asta di riparazione (live, lotti simultanei, timer fisso)

**Possiede**: `src/fantabuste/repair/`, `tests/test_repair.py`
**Input**: `PlayerProjection[]` (Modulo B), `AuctionState` finale da E, riusa direttamente il **solver del Modulo D** (dipendenza di codice, non solo di schema — per questo F parte dopo il merge di D, non in Fase 1)
**Output**: una **watchlist di obiettivi prioritari** (generata dal modulo, non curata a mano da te), e per ogni lotto della watchlist attualmente aperto un **tetto di offerta** (`max_bid`) + una **priorità** relativa agli altri lotti della watchlist, ricalcolati a ogni chiusura di lotto; per ogni lotto della watchlist chiuso, un `RepairLotResult`

Meccanica confermata: **più giocatori "a tempo" sono aperti in contemporanea**, timer **fisso, senza reset al rilancio**. **Decisione presa sul carico operativo**: non monitori tutti i lotti aperti — troppi per un inserimento manuale sostenibile — ma solo una watchlist di dimensione limitata (default configurabile, es. 10-15) generata dal modulo stesso, non da te a mano. Questo NON è più riducibile a "decidi un tetto per il lotto che hai davanti in questo momento" — è un problema di **allocazione di portafoglio con budget condiviso**, ma ristretto alla watchlist, non all'intero mercato residuo. Il disegno cambia di conseguenza:

0. **Generazione della watchlist**: dopo la chiusura delle 2 tornate (e ricalcolata dopo ogni lotto di riparazione chiuso, perché budget e slot residui cambiano), classifica i giocatori ancora disponibili per surplus atteso (`VORP_i` al netto di una stima del prezzo di riparazione, es. una frazione bassa della quotazione — i prezzi di riparazione tendono a essere più bassi delle tornate) e prendi i primi K (default **8**, configurabile in `league.yaml`: `asta_riparazione: {watchlist_size: 8}` — basso perché operi da solo). Tutto ciò che è fuori watchlist è deliberatamente ignorato: se lo vince un avversario, non è un problema che questo modulo debba vedere.
1. **Riusa il solver di D, non reinventarlo.** A ogni istante rilevante (apertura/chiusura di un lotto della watchlist, o un tick periodico se i prezzi salgono su lotti della watchlist), esegui un **re-solve ristretto** dello stesso MILP: stessi vincoli (budget residuo, slot residui per ruolo, max giocatori per squadra), ma limitato ai soli lotti **della watchlist attualmente aperti + non ancora assegnati**, con il prezzo corrente di ogni lotto come **vincolo di minimo** (non puoi offrire meno del prezzo corrente). L'output non è "l'offerta da fare adesso" ma **il tetto oltre il quale, dato tutto il resto, conviene lasciar perdere quel lotto**.
2. **Priorità tra lotti della watchlist in conflitto di tempo.** Se due o più lotti della watchlist chiudono entro una finestra ravvicinata (configurabile, es. 2 minuti), il solver deve segnalarlo esplicitamente come **conflitto**, con un ordine di priorità basato sul surplus atteso, così la decisione "quale snipare" è presa con calma prima, non nel panico all'ultimo secondo.
3. **Aggiornamento budget avversari esatto a ogni chiusura di lotto della watchlist**: se la piattaforma mostra chi vince ogni lotto a che prezzo (tipico in un'asta a tempo pubblica), il tracciamento resta preciso — niente stime — e il re-solve successivo parte da numeri veri, non inferiti. **Marca ogni dato con l'orario dell'ultimo aggiornamento manuale, e mostra un avviso visibile se un lotto non viene aggiornato da più di N secondi** (configurabile) — con inserimento manuale, i dati invecchiano, e un tetto calcolato su un prezzo vecchio mostrato con la stessa sicurezza di uno fresco è peggio di nessun tetto.
4. **Nessuna sottostima del vincolo di esecuzione umana**: il timer fisso senza reset premia chi offre una sola volta, all'ultimo istante utile — ma è un istante che un umano deve comunque cliccare. Lo scope di questo modulo è **decision support**, non invio automatico di offerte: calcola e mostra i tetti e le priorità in anticipo così che al momento dello sniping non ci sia calcolo da fare, solo un numero da inserire. **L'invio automatico delle offerte sulla piattaforma è deliberatamente fuori scope** — oltre al rischio di errore in una finestra temporale stretta, dipende dai Termini di Servizio della piattaforma specifica, che vanno verificati separatamente prima di considerarlo.
5. **Interfaccia minima, a inserimento manuale, ristretta alla watchlist — per scelta deliberata, non per limite tecnico.** Playwright potrebbe tecnicamente leggere `leghe.fantacalcio.it`, ma i Termini di Utilizzo vietano esplicitamente lo scraping e le conseguenze (sospensione account, possibile durante l'asta stessa) non sono accettabili — vedi A4. **Nessuno scraping, nessuna automazione di lettura, in nessuna versione di questo modulo — questo vale per qualsiasi tool, Playwright, Claude in Chrome o altro: il divieto riguarda la categoria (accesso automatizzato senza autorizzazione), non lo strumento specifico.** Una tabella live (CLI o file che si aggiorna) — solo i K lotti della watchlist: prezzo corrente, chiusura tra, il tuo tetto, priorità/conflitti — aggiornata a mano da te (o da un compagno di lega) guardando l'app. Digitare K numeri, non il listone residuo intero, è il punto di questa decisione.
6. **Cattura dei risultati per il backtest**: quando un lotto della watchlist chiude (vinto da te o da un avversario), registra un `RepairLotResult`. I lotti fuori watchlist non vengono tracciati — scelta consapevole, non lacuna: il backtest dell'anno prossimo copre "come sono andati i miei obiettivi", non l'intero mercato di riparazione. Alla fine dell'asta, esporta tutti i `RepairLotResult` per il `run_log.json` del Modulo E (punto 6-7 sopra).

**Vincolo di onestà**: documentare nel modulo che il re-solve ristretto è comunque un'approssimazione (ottimizza dato lo stato attuale, non l'intera sequenza futura di chiusure non ancora note) — stessa disciplina già applicata al MILP di D. Documentare anche che **la watchlist stessa è un'approssimazione**: un giocatore fuori dai primi K per surplus atteso potrebbe rivelarsi un affare se il suo prezzo scende inaspettatamente — il modulo accetta questo compromesso deliberatamente in cambio di un carico di lavoro sostenibile.

**DoD**: dato un `AuctionState` di fixture e una sequenza simulata di eventi, la generazione della watchlist produce sempre esattamente K candidati (o meno se i disponibili sono meno di K) ordinati per surplus atteso; il re-solve ristretto alla watchlist produce tetti coerenti (mai sopra budget residuo, mai tali da scoprire slot obbligatori) e segnala correttamente i conflitti di tempo **dentro la watchlist**; tempo di risposta del re-solve sotto 1 secondo (la watchlist è piccola per costruzione); ogni lotto della watchlist chiuso produce un `RepairLotResult`, vinto o perso.

---

## PARTE D — Contenuto di `CLAUDE.md` (da mettere in cima al repo)

```markdown
# FANTABUSTE — regole di progetto

## ⚠️ Dati
Tutti i numeri nei documenti di ricerca allegati sono ESEMPLIFICATIVI.
Nessun numero preso da un documento generato da un LLM entra in un modello.
Ogni record porta `fonte` e `is_synthetic`. Nessuna offerta reale da dati sintetici.

## ⚠️ Nessuno scraping, nessuna automazione della piattaforma
Deciso e definitivo: il Modulo F (asta di riparazione) è a inserimento manuale.
I ToS di Fantacalcio s.r.l. vietano esplicitamente scraping e accesso automatizzato
alla piattaforma (leghe.fantacalcio.it), con sospensione dell'account come conseguenza
possibile — un rischio inaccettabile se scatta durante l'asta vera. Non proporre,
implementare o abilitare automazioni di lettura/scrittura verso quella piattaforma
in nessun modulo, nemmeno "solo in lettura" o "solo per test".

## Regole della mia lega
[compilare da config/league.yaml — budget, rosa, regola pareggi, n. partecipanti, tornate]

## Confini di modulo
Se il tuo lavoro richiede di modificare `schemas.py` o file di un altro modulo:
FERMATI e segnalalo. Non modificare unilateralmente i contratti condivisi.

## Standard
- Nessun numero magico hardcoded: tutto in config/.
- Ogni modello ML ha un baseline trasparente come confronto e come fallback.
- Ogni approssimazione è dichiarata nel docstring, non nascosta.
- Test girano su fixture sintetiche, mai su dati reali.
- data/raw/ è read-only.
```

---

## PARTE E — Ordine di esecuzione consigliato

1. **Fase 0** (1 agente, ~1 sessione, forse 2 — vedi nota sotto) → merge su `main`, contratti congelati (inclusi `AuctionState` esteso, `OpponentBidObservation`, `RepairLotResult`).
2. **Fase 1**: lanciare A, B, C, D **in parallelo** su worktree separati (4 agenti). B è il più leggero (baseline come default); C è il più prezioso data la visibilità completa delle offerte; D condiziona anche F, quindi la qualità della sua interfaccia conta doppio.
3. Merge A→B→C→D, risolvendo eventuali attriti sui contratti.
4. **Fase 2**: E e F **in parallelo** (2 agenti) — entrambi partono solo dopo il merge di D. F riusa direttamente il solver di D.
5. **Solo dopo**: sostituire le fixture con i dati reali, un modulo alla volta, verificando `match_report.csv` a mano.

**Nota sulla stima della Fase 0**: lo scope (7 schemi, config completo, fixture per 600 giocatori, setup CI) è denso per "una sessione". Se non chiude in una sessione, meglio spezzarla in due (schemi + config in una, generatore di fixture nell'altra) che congelare contratti fatti in fretta — un errore qui si paga moltiplicato per 4 agenti a valle.

---

## PARTE F — Vincoli risolti e conseguenze sul design

| Domanda | Risposta | Conseguenza già recepita nel piano |
|---|---|---|
| Storico prezzi pagati nella lega | **Sì, 2-3 stagioni** | Modulo C attivabile in modalità `empirical`. |
| Visibilità delle offerte | **Completa — vedo tutte, vincenti e perdenti** | Elimina il bias di selezione. Modulo C fitta la distribuzione vera per fascia, non solo il massimo. Budget residuo avversari **esatto**, non stimato, in `AuctionState`. |
| Numero di tornate busta chiusa | **2, fisso** | `n_tornate_buste: 2` in config; niente logica per un numero variabile. |
| Dopo le tornate | **Asta di riparazione classica a tempo, giocatore per giocatore, lotti multipli simultanei, timer fisso senza reset** | Modulo F ridisegnato come riottimizzatore di portafoglio (riusa il solver di D), non euristica sequenziale. |
| Piattaforma | **Leghe Fantacalcio® Serie A (Fantacalcio s.r.l.)** | Regola pareggio confermata: nessuna assegnazione. Nessuna API pubblica confermata → Modulo F a inserimento manuale. |
| Automazione di lettura (Playwright, Claude in Chrome, o altro) | **No — il divieto ToS è sulla categoria, non sullo strumento** | Modulo F resta a inserimento manuale in ogni sua versione, qualunque tool si consideri in futuro. |
| Modulo F: riusa il solver di D o indipendente? | **Riusa il solver di D** | F non è più parte del parallelismo di Fase 1; parte in Fase 2, in parallelo con E, dopo il merge di D. |
| Carico di inserimento manuale in F | **Watchlist di obiettivi prioritari (K, generata dal modulo), non tutti i lotti** | F genera e ricalcola la watchlist da `AuctionState`+`PlayerProjection`; il re-solve, i conflitti e la cattura risultati sono ristretti alla watchlist. |
| Cattura risultati riparazione per backtest | **Sì, da subito** | Nuovo schema `RepairLotResult`; `run_log.json` del Modulo E ora copre l'intera asta, non solo le tornate busta chiusa. |
| Stagioni storiche per il modello di valutazione | **2** | Modulo B: baseline come modello primario, ML come sfidante dietro flag con margine ampio. |
| Agenti in parallelo | **4 in Fase 1 (A-D), 2 in Fase 2 (E, F)** | Struttura fissata, non più un gate aperto. |

### Dove è finito lo sforzo, dati questi vincoli

Il bilanciamento è cambiato due volte in questa conversazione. Prima: da B (valutazione ML) verso C (modello di prezzo), perché 2 stagioni non bastano per un ML robusto mentre 2-3 stagioni di prezzi reali sono oro. Ora, con la visibilità completa delle offerte: **C è ulteriormente rafforzato** (distribuzione vera, non solo il massimo osservato) e si aggiunge **F come nuovo centro di valore**, perché conoscere il budget *esatto* degli avversari all'inizio di un'asta a tempo è un vantaggio informativo concreto, non un'approssimazione statistica.

### Gate di decisione a fine Fase 0 (ora riguarda solo A-D)

La struttura F-dopo-D e la coppia E+F sono fissate. Resta un solo grado di libertà, sulla Fase 1:

- **4 agenti** (A, B, C, D) se i contratti sono usciti puliti e le fixture coprono tutti i casi, incluso `AuctionState`.
- **2 agenti**: (A+B) e (C+D). Coppie naturali — A e B condividono il dominio "dati e valutazione", C e D quello "prezzi e decisione".
- **1 sequenziale** se durante la Fase 0 sono emerse ambiguità sui contratti.

Segnale pratico: se alla fine della Fase 0 stai ancora discutendo *cosa contiene* `PlayerProjection` o l'interfaccia del solver di D, non sei pronto per il parallelismo massimo — e ricorda che un'interfaccia di D fatta male si ripaga su F con un secondo giro di attrito, dato che F ne dipende direttamente.

### Restano aperte

1. **Formato dell'export del listone** (Fantacalcio.it, Leghe, altro) — determina il parser dell'Agente A.
2. **Formato dello storico prezzi/offerte** (export della piattaforma, foglio tenuto a mano, screenshot da trascrivere) — se è manuale, va nello scope dell'Agente A prima che C possa lavorare.
3. ~~Dimensione di default della watchlist (K)~~ — **risolto: K = 8** come default, configurabile in `league.yaml` (`asta_riparazione: {watchlist_size: 8}`). Scelto basso perché lavori da solo — tara al rialzo dopo la prima asta reale se vedi margine.
4. ~~Chi inserisce i dati durante l'asta di riparazione~~ — **risolto: da solo.** Nessun compagno di lega dedicato; l'interfaccia di F va ottimizzata per un solo paio d'occhi e una sola tastiera.
