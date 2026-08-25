# Divergenze e questioni aperte — da risolvere prima della Fase 0

> Redatto il 2026-08-25 confrontando `docs/DESIGN.md` (piano di progetto),
> `CLAUDE.md` (regole sempre attive) e `docs/LEAGUE_CONTEXT.md` (regolamento
> reale della lega), più una verifica diretta dei file storici su Google Drive.
>
> **Perché esiste questo file**: DESIGN.md dichiara "confermato, non ipotetico"
> una serie di fatti sulla struttura dell'asta. Il regolamento di lega ne
> contraddice diversi. Congelare `schemas.py` sopra i fatti sbagliati significa
> propagare l'errore su 4 agenti a valle — esattamente il rischio che DESIGN.md
> §PARTE E segnala sulla Fase 0.

---

## 0. Decisioni prese (2026-08-25)

Scope della v1, decisa dal proprietario del progetto dato il vincolo di 8 giorni:

- **Obiettivo v1: fase buste completa, 2 tornate.** Pipeline A→D più la gestione
  stateful delle due tornate (Modulo E ridotto). Il Modulo F (mercato ad asta)
  **non** è nella v1: il mercato resta aperto fino al 7 settembre e oltre, quindi
  c'è tempo per aggiungerlo dopo la tornata 2.
- **Nessun vincolo speciale nel modello v1.** Regolamento portieri Top 8,
  modificatori e ricarica +50/+2 slot **non** entrano nel MILP: vengono emessi
  come **avvertenze testuali nel report**, da gestire a mano. Le divergenze §1.2,
  §1.3 e §1.4 qui sotto restano quindi **note ma deliberatamente non risolte nel
  codice** della v1 — sono il primo candidato per la v2.
- **`budget_totale: 500`** in prima fase.
- Il Modulo C parte in modalità **`prior`** (vedi §2): nessuno storico di offerte
  avversarie esiste, e i privilegi da amministratore non lo sbloccano (§2.1).

**Conseguenza da tenere presente**: con i vincoli speciali fuori dal modello, le
offerte prodotte dalla v1 sono un *punto di partenza da correggere a mano*, non
un piano da eseguire alla lettera. Il report deve dirlo in modo inequivocabile,
in particolare sui portieri (2 soli slot in fase 1 + regola Top 8 + garanzia del
titolare rendono la strategia sui portieri il punto dove il modello semplificato
sbaglia di più).

---

## 1. Divergenze regolamento ↔ DESIGN.md

### 1.1 🔴 Il mercato ad asta NON è sequenziale rispetto alle buste

- **DESIGN.md** (A4, Modulo E punto 3, Modulo F): dopo le 2 tornate si chiude la
  fase busta chiusa (`fantabuste chiudi-buste`), si congela `AuctionState` e si
  passa il testimone al Modulo F.
- **Regolamento** (§2, §6): il mercato ad asta apre **il 3 settembre alle 20:00,
  contemporaneamente alla seconda tornata a buste chiuse**.

**Conseguenza**: l'handoff E→F sequenziale non regge. Durante la tornata 2 il
budget è impegnato **simultaneamente** su buste chiuse e su lotti a tempo. Il
Modulo D deve allocare budget su due meccanismi d'asta concorrenti nello stesso
istante, non uno dopo l'altro. Va deciso se: (a) riservare esplicitamente una
quota di budget alla tornata 2 prima di aprire F, (b) modellare le due fasi in un
unico problema di allocazione, o (c) accettare la sequenzialità come
approssimazione dichiarata (e documentarla come tale).

### 1.2 🔴 Composizione della rosa e seconda fase con ricarica di budget

- **DESIGN.md / `league.example.yaml`**: `rosa: {P,D,C,A}` generico, budget unico.
- **Regolamento** (§1, §3): fase 1 = **2 P – 8 D – 8 C – 6 A = 24**; il 7 settembre
  arrivano **+50 crediti** e **+2 slot per ruolo**, rosa massima **4-10-10-8 = 32**,
  slot aggiuntivi **non obbligatori**.

**Conseguenza**: due vincoli di rosa distinti in due momenti, e un budget che
**non è fisso** ma riceve una ricarica nota a data nota. Un ottimizzatore che
spende tutto entro il 7 settembre ignora 50 crediti garantiti; uno che ne
accantona troppi lascia slot obbligatori scoperti. Serve nel modello: budget
fase 1, budget fase 2, e il fatto che gli slot fase 2 sono **opzionali**.
Nota anche che 2 portieri in fase 1 è insolito e interagisce con §4–5 (sotto).

### 1.3 🔴 Regolamento speciale portieri — vincoli condizionali non modellati

Il regolamento §4–5 introduce vincoli che DESIGN.md non menziona affatto:

- lista **Top 8** portieri con valori fissati;
- **non** si possono tenere due Top 8 di squadre diverse (eccezione: stessa squadra);
- chi prende il titolare di una squadra rappresentata in Top 8 può **reclamare il
  secondo portiere della stessa squadra**, ma poi non può comprare portieri di
  altre squadre finché tutti non hanno completato i portieri;
- strategia alternativa: **nessun Top 8 e due titolari liberi**;
- **garanzia**: chi finisce senza titolare riceve d'ufficio il portiere titolare
  disponibile di valore più basso.

**Conseguenza**: sono vincoli disgiuntivi/condizionali, non lineari in forma
naturale. Richiedono variabili binarie ausiliarie nel MILP del Modulo D. La
"garanzia del titolare" è inoltre un'**opzione di fallback gratuita** che cambia
il valore di offrire sui portieri bassi: se il peggior esito è ricevere gratis il
titolare più economico, il surplus atteso di un terzo portiere a 1 credito è
quasi nullo. Va modellato, non ignorato.

### 1.4 🟠 I modificatori rendono il valore non separabile per giocatore

Regolamento §7: modificatore **difesa** = media aritmetica del portiere + i **3
migliori difensori**; **centrocampo** su differenziale di reparto; **attacco** su
costanza del voto per attaccanti senza bonus pesanti.

**Conseguenza**: il contributo di un difensore dipende da *quali altri difensori
possiedi* (è nei tuoi top 3 o no?) e dal portiere. L'obiettivo del MILP di
DESIGN.md — `max Σ (VORP_i × p_win(b)) · x[i,b]` — è **lineare e separabile**, e
quindi non può rappresentare questo. Opzioni: (a) approssimare il modificatore
con un bonus marginale precalcolato per fascia di voto atteso (approssimazione
dichiarata), (b) linearizzare "i 3 migliori" con variabili di ordinamento (costoso),
(c) valutare la rosa a posteriori con il modificatore vero e usare il MILP solo
per generare candidati, poi ri-classificare le top-N rose. La (c) si sposa bene
con "top-5 rose alternative" già previsto nel Modulo D.

### 1.5 🟠 Numero di tornate: coerente, ma il totale delle fasi no

`n_tornate_buste: 2` è confermato dal regolamento. Ma DESIGN.md tratta la
riparazione come **una** fase post-buste, mentre il regolamento ne ha **due**
(3–7 settembre con rosa obbligatoria, poi dal 7 settembre con slot opzionali e
+50 crediti). Il Modulo F va parametrizzato su due finestre con vincoli diversi.

---

## 2. 🔴 Il dato che regge il Modulo C non esiste (verificato)

DESIGN.md §PARTE F dichiara:

| Domanda | Risposta dichiarata in DESIGN.md |
|---|---|
| Storico prezzi pagati nella lega | "**Sì, 2-3 stagioni**" → Modulo C in modalità `empirical` |
| Visibilità delle offerte | "**Completa — vedo tutte, vincenti e perdenti**" → elimina il bias di selezione |

e ne deriva che C è "il centro di valore del progetto", con `K ≥ 30` osservazioni
per fascia.

**Verifica su Google Drive (2026-08-25) — cosa c'è davvero:**

- `fantacalcio - buste #1` (set. 2017): contiene **solo le mie offerte** — colonne
  `nome, busta (crediti), quot. attuale, % peso, squadra, fantamedia, ...`.
  **Nessuna offerta avversaria. Nessun esito vinto/perso.** ~40 giocatori, **1 sola
  tornata, 1 sola stagione, di 9 anni fa.**
- Piani busta 25/26 (`rifai le buste...`, `tattico_buste_..._v2.csv`): di nuovo
  **solo offerte mie**, per giunta *pianificate* e generate da un LLM, non
  offerte realmente presentate né esiti. Contengono affermazioni sugli avversari
  ("X spende il 7.4% del budget in difesa") che sono **testo generato, non dati**.
- `F2016_voti.pdf`: **non trovato** su Drive.
- Nessun file con offerte avversarie per nessuna stagione.

**Conclusione**: la visibilità completa delle offerte è probabilmente vera *dal
vivo dentro l'app*, ma **non è mai stata archiviata**. Allo stato attuale il
Modulo C **non può girare in modalità `empirical`**: zero osservazioni di offerte
avversarie, contro le ≥30 per fascia richieste. Per la sua stessa regola,
degrada interamente a `prior` — distribuzione centrata sul listone con varianza
ampia e dichiarata.

Questo **non affossa il progetto**, ma ne sposta il baricentro: il vantaggio
atteso torna a essere quello dichiarato in DESIGN.md §A3 ("ridurre errori
sistematici"), non un edge informativo sulle offerte altrui. E rende la
**cattura dei dati di quest'anno** (`OpponentBidObservation` dopo ogni tornata)
il singolo investimento con il ritorno più alto — è ciò che rende `empirical`
possibile l'anno prossimo.

### 2.1 L'accesso da amministratore non recupera lo storico (ricerca 2026-08-25)

Verificato sulla documentazione pubblica di Leghe Fantacalcio (solo ricerca web:
il dominio `leghe.fantacalcio.it` è bloccato dal proxy di rete di questo
ambiente, il che è comunque coerente con la regola "nessun accesso automatizzato
alla piattaforma"):

- **Il presidente di lega non può vedere le offerte dei partecipanti.** La guida
  ufficiale è esplicita: *"Il Presidente di Lega ricordiamo che non può in nessun
  caso vedere le offerte dei partecipanti alla sua Lega"*. Esiste solo un
  **riepilogo aggregato**, limitato al mercato iniziale, che mostra per ogni
  squadra il **numero di giocatori offerti** e i **kapitals impegnati** — non le
  singole offerte, non per giocatore.
- **L'export CSV esiste per le rose, non per le offerte.** Menù → Rose → Esporta
  CSV, e l'import corrispondente in ADMIN → Gestione Rose. Serve a ricostruire
  *chi ha preso chi e a quanto*, cioè i prezzi pagati dai vincitori, **non** le
  offerte perdenti.

**Conclusione**: i privilegi di amministratore **non** sbloccano lo storico delle
offerte avversarie. Il massimo recuperabile a posteriori è l'export delle rose
(prezzi vincenti), che soffre del bias di selezione descritto in DESIGN.md §A2 —
si osserva solo il massimo della distribuzione, non la distribuzione.

### 2.2 FantaLab "Prezzo Medio Aste" (PMA) — utile, ma non è ciò che serve a `empirical` (ricerca 2026-08-25)

Verificato via ricerca web (il dominio `fantalab.it` è bloccato dal proxy di
rete di questo ambiente, quindi solo tramite fonti secondarie — SOS Fanta,
FantaMaster, TuttoFantacalcio):

- FantaLab pubblica un **Prezzo Medio Aste (PMA)** per giocatore: dichiarato
  come dato reale ("non è una stima, è quello che i FantaPresidenti hanno
  realmente speso"), aggregato dalle leghe che usano FantaLab/FantaCalcio-
  Online **in tutta Italia**, **filtrabile per numero di partecipanti (8/10/
  12/14) e budget (250/300/500/1000)** — combinazione 12 partecipanti/500
  crediti disponibile, cioè esattamente la configurazione della nostra lega.
- **Ma è solo il prezzo VINCENTE**, non tutte le offerte. Soffre dello stesso
  bias di selezione descritto in DESIGN.md §A2 e in §2 qui sopra: si osserva
  il massimo della distribuzione, non la distribuzione intera. **Non
  sblocca la modalità `empirical`** di C, che richiede offerte vincenti +
  perdenti per fittare la distribuzione vera.
- **Probabile mismatch di meccanismo d'asta**: il PMA aggrega quasi certamente
  soprattutto aste classiche/live (ascendenti), non buste chiuse (FPSBA). Sono
  meccanismi diversi con dinamiche di prezzo diverse — è lo stesso motivo per
  cui DESIGN.md tiene separati `OpponentBidObservation` (buste) e
  `RepairLotResult` (asta a tempo) e vieta di fittare C sul secondo. Pooling
  diretto del PMA nella distribuzione busta-chiusa richiede quindi una
  cautela esplicita, non un uso 1:1.
- **Nessun export CSV/API confermato**: è presentato come consultazione per
  giocatore dentro l'app/sito durante la preparazione dell'asta, non come
  dataset scaricabile. Verificare se serve l'abbonamento Premium (~13-17€)
  per vederlo per intero o se la versione gratuita basta.
- ToS di fantalab.it non verificati (sito irraggiungibile da questo ambiente).
  Finché non sono confermati, vale la stessa cautela già adottata per
  leghe.fantacalcio.it: **niente accesso automatizzato/scraping**, solo
  consultazione manuale dall'utente.

**Uso corretto**: il PMA non fa diventare C `empirical`, ma può fare un
**`prior` molto più informativo** — invece di centrare `prior_mu` su
`ratio=1` (offerta = quotazione, pura assunzione), calibrarlo sul rapporto
osservato `PMA / quotazione_listone` per fascia, con `prior_sigma` che resta
ampio per la varianza non catturata (mismatch di meccanismo, singola stagione
aggregata, ecc.). È un miglioramento reale e concreto, da inserire come
parametro calibrato in v1, non un dato "grezzo" da spacciare per empirico —
va marcato `fonte="fantalab_pma_2026"`, `is_synthetic=False` ma con
`confidence_flag` bassa/media nel report per il mismatch di meccanismo.

Restano due cose che vale la pena verificare **dentro l'app** (a mano, non da
codice), perché sono decisioni di configurazione della lega e risolvono due
questioni aperte:

1. **Impostazione "Assegna buste se uguali"** → risolve `regola_pareggio` (§4).
   Se **disattivata**: offerte pari ⇒ giocatore non assegnato, torna alla tornata
   successiva (default della piattaforma, quello che DESIGN.md assume). Se
   **attivata**: il giocatore va a chi ha presentato la busta **per primo** — e in
   quel caso l'euristica delle **offerte non tonde** del Modulo D non serve più,
   mentre diventa rilevante *quando* si invia la busta.
2. **Cosa vedono i partecipanti a tornata chiusa** — se davvero tutte le offerte
   di tutti (premessa di DESIGN.md §A4) o solo le vincenti. La documentazione
   pubblica non lo specifica: dipende dalla configurazione. Determina se
   `OpponentBidObservation` quest'anno raccoglierà la distribuzione completa o
   solo la coda destra.

---

## 3. 🔴 Vincolo di tempo

Il regolamento data la prima busta chiusa al **2 settembre ore 20:00**. Oggi è il
**25 agosto 2026**: mancano **8 giorni**, di cui ~7 utili.

Il piano di DESIGN.md (Fase 0 → 4 agenti paralleli A-D → merge → Fase 2 con E+F
→ sostituzione fixture con dati reali) è dimensionato su settimane, non giorni, e
la sostituzione delle fixture con dati reali è esplicitamente l'**ultimo** passo.
Va deciso esplicitamente cosa entra nella v1 utilizzabile il 2 settembre.

---

## 4. Parametri di regolamento ancora mancanti

Non presenti in nessuno dei tre documenti, necessari per `config/league.yaml`:

- ~~**`budget_totale`** di fase 1~~ — **risolto: 500 crediti**, con il +50 del
  7 settembre che si somma sopra (→ 550 complessivi in seconda fase).
- **`n_partecipanti`** per la stagione 26/27 (l'esempio storico 2016/17 è "lega a 10";
  i piani 25/26 nominano ~10-12 squadre). **Ancora da confermare** — entra nel
  replacement level del VORP (Modulo B).
- **`max_giocatori_per_squadra`** di Serie A: esiste un tetto?
- **`regola_pareggio`**: DESIGN.md la dà per confermata (`nessuna_assegnazione`)
  in base ai regolamenti standard della piattaforma, ma il regolamento di lega
  **non la cita**. Se la lega usa una variante (sorteggio, primo arrivato), cade
  l'euristica delle **offerte non tonde** del Modulo D.
- **`offerte_non_tonde`**: dipende dalla risposta precedente.
- **Asta di riparazione**: `timer_secondi` e conferma del "timer fisso senza
  soft-close" — DESIGN.md lo dà per confermato, il regolamento non lo specifica.
- **Fonte del listone 26/27** e suo formato di export (determina il parser del
  Modulo A). Le quotazioni su Drive sono del 2018.
- **Statistiche storiche**: DESIGN.md assume 2 stagioni. Su Drive ci sono
  statistiche 15/16–20/21, che **non coprono** 24/25 e 25/26. Da chiarire da dove
  arrivano le 2 stagioni recenti (FBref via `soccerdata`?).
