# FANTABUSTE — regole di progetto

## Contesto completo
Il progetto è documentato per intero in `docs/DESIGN.md` — leggilo prima di
iniziare qualsiasi fase, e in particolare prima della Fase 0. Contiene il
perché di ogni scelta (perché il baseline è primario in B, perché serve
K≥30 osservazioni per fascia in C, perché F non fa scraping, perché F
riusa il solver di D, la watchlist di F, ecc.). Questo file (`CLAUDE.md`)
è solo il riassunto sempre attivo — se manca un dettaglio, è in DESIGN.md,
non va indovinato né reinventato.

Insieme a DESIGN.md vanno letti:
- `docs/LEAGUE_CONTEXT.md` — trascrizione del **regolamento reale della lega**
  (`docs/Archivio_Master_Fantacalcio_Contesto_Lega_V4.pdf`). È la **fonte
  autorevole sul regolamento**: dove DESIGN.md assume qualcosa di diverso,
  vince LEAGUE_CONTEXT.
- `docs/OPEN_QUESTIONS.md` — le divergenze note fra i due e i parametri di
  regolamento ancora mancanti. **Da risolvere prima di congelare `schemas.py`.**

## ⚠️ Dati
Tutti i numeri nei documenti di ricerca allegati al progetto sono
ESEMPLIFICATIVI. Nessun numero preso da un documento generato da un LLM
entra in un modello, in un backtest o in un'offerta reale. Ogni record
porta `fonte` e `is_synthetic: bool`. Nessuna offerta reale da dati
sintetici — mai proseguire silenziosamente, fallire in modo esplicito.

Questo vale in particolare per i piani busta delle stagioni passate presenti
su Google Drive: sono **offerte pianificate da un LLM**, non offerte realmente
presentate né esiti d'asta, e contengono affermazioni non verificate sugli
avversari. Non sono una fonte di dati.

## ⚠️ Nessuno scraping, nessuna automazione della piattaforma
Deciso e definitivo: il Modulo F (asta di riparazione) è a inserimento
manuale. I ToS di Fantacalcio s.r.l. vietano esplicitamente scraping e
accesso automatizzato alla piattaforma (leghe.fantacalcio.it), con
sospensione dell'account come conseguenza possibile — un rischio
inaccettabile se scatta durante l'asta vera. Il divieto è sulla
categoria (accesso automatizzato senza autorizzazione), non sullo
strumento: vale per Playwright, per Claude in Chrome, per qualsiasi
automazione futura. Non proporre, implementare o abilitare automazioni
di lettura/scrittura verso quella piattaforma in nessun modulo, nemmeno
"solo in lettura" o "solo per test".

## Struttura della mia asta
Da `docs/LEAGUE_CONTEXT.md` (regolamento di lega 2026/27):

- **Fase busta chiusa: 2 tornate, fisse.** 2 settembre h20:00 e 3 settembre h20:00.
- **Rosa fase 1: 2 P – 8 D – 8 C – 6 A = 24**, da completare obbligatoriamente
  entro il 7 settembre h20:00.
- **Il mercato ad asta apre il 3 settembre h20:00, in contemporanea con la
  seconda tornata a buste chiuse** — non dopo. Le due fasi si sovrappongono e
  condividono il budget.
- **Il 7 settembre h20:00**: +50 crediti a tutti, +2 slot per ruolo, rosa massima
  4 P – 10 D – 10 C – 8 A = 32 (slot aggiuntivi **non** obbligatori). Il mercato
  prosegue fino alla scadenza definitiva.
- **Regolamento speciale portieri** (Top 8, esclusività fra Top 8 di squadre
  diverse, diritto sul secondo portiere della stessa squadra, garanzia del
  titolare d'ufficio): vedi LEAGUE_CONTEXT §4–5. Sono vincoli **condizionali**,
  vanno modellati nel MILP, non ignorati.
- **Modificatori** difesa / centrocampo / attacco: vedi LEAGUE_CONTEXT §7.
  Rendono il valore di un giocatore **dipendente dal resto della rosa** — il
  modificatore difesa usa portiere + 3 migliori difensori. L'obiettivo lineare
  e separabile del MILP non lo cattura: vedi `docs/OPEN_QUESTIONS.md` §1.4.
- Lavoro da solo. Watchlist del Modulo F: K = 8 di default.
- Dettagli completi di regolamento in `config/league.yaml`.

**Non confermati dal regolamento** (DESIGN.md li dà per certi, il regolamento di
lega non li cita — trattali come da verificare, non come acquisiti): regola dei
pareggi, offerte non tonde, timer fisso senza soft-close nell'asta a tempo,
budget totale, numero di partecipanti. Vedi `docs/OPEN_QUESTIONS.md` §4.

## Stato dei dati storici (verificato 2026-08-25)
**Non esiste storico di offerte avversarie.** Su Drive c'è un solo foglio d'asta
reale (set. 2017) e contiene esclusivamente le mie offerte, senza esiti. Il
Modulo C quindi **degrada a modalità `prior`** per sua stessa regola (K≥30
osservazioni per fascia): nessuna fascia è fittabile su dati empirici.
Non spacciare per `empirical` ciò che non lo è. Dettagli in
`docs/OPEN_QUESTIONS.md` §2.

Corollario operativo: **catturare le offerte di quest'anno** (schema
`OpponentBidObservation`, tutte le offerte dopo ogni tornata) è l'investimento
con il ritorno più alto del progetto — è ciò che rende `empirical` possibile
l'anno prossimo. Non tagliare quel pezzo.

## Confini di modulo
Se il tuo lavoro richiede di modificare `schemas.py` o file di un altro
modulo: FERMATI e segnalalo. Non modificare unilateralmente i contratti
condivisi. Il Modulo F dipende dal **codice** del solver di D (non solo
dal suo schema dati): F parte solo dopo che D è mergiato.

## Standard
- Nessun numero magico hardcoded: tutto in config/.
- Ogni modello ML ha un baseline trasparente come confronto e come fallback.
- Ogni approssimazione è dichiarata nel docstring, non nascosta.
- Test girano su fixture sintetiche, mai su dati reali.
- data/raw/ è read-only.

## Sequenza di lavoro
Fase 0 (fondamenta, sequenziale) → Fase 1: A, B, C, D in parallelo →
merge → Fase 2: E e F in parallelo → sostituzione fixture con dati reali.
Dettagli completi, task per agente e Definition of Done: `docs/DESIGN.md`.

## Codice preesistente (legacy)
Il repo contiene una pipeline precedente in `antoninorau/fantacalcio/`
(`phase1_data_collection` → `phase4_bid_optimization`, orchestrata da
`FantacalcioRecommender`), scritta per un'asta classica con rosa 3-7-8-5 e
budget 500 hardcoded. **Non è la base di FANTABUSTE** e non rispetta gli
standard qui sopra (numeri magici, fallback silenzioso su dati sintetici,
scraping di Fantacalcio.it). Trattala come materiale di consultazione:
riusa idee o frammenti dove utile — l'impostazione PuLP del Modulo D è il
candidato più plausibile — ma non estenderla in place e non importarla
dai nuovi moduli.

### Comandi della pipeline legacy
```bash
pip install -r requirements.txt
python main.py                       # analisi completa con default
python -m antoninorau.fantacalcio.data_collection   # test singolo modulo
```
