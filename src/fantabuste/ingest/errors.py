"""Eccezioni del Modulo A (ingestion).

Filosofia del progetto (CLAUDE.md): "fallire in modo esplicito, non
silenziosamente". Questo modulo non ha MAI un percorso che degrada a dati
sintetici o a una cache scaduta senza dirlo forte e chiaro con
un'eccezione dedicata.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base per tutti gli errori del Modulo A."""


class FormatoListoneNonRiconosciuto(IngestError):
    """Il file del listone non ha nessuna colonna riconoscibile per un campo
    obbligatorio (nome, ruolo, squadra, quotazione). Non tentare di indovinare
    oltre l'elenco di alias noto: meglio fallire con un messaggio chiaro che
    produrre un `Player` con dati inventati."""


class RigaListoneScartata(IngestError):
    """Una singola riga del listone non è convertibile in un `Player` valido
    (es. ruolo non mappabile, quotazione non numerica). Usata internamente
    per accumulare gli scarti riga per riga senza abortire l'intero parse."""


class DipendenzaMancante(IngestError):
    """Una libreria opzionale richiesta da questa funzionalità non è
    installata nell'ambiente corrente. `pyproject.toml` è read-only per
    l'Agente A (vedi CLAUDE.md, "Confini di modulo"): l'assenza va segnalata
    nel report finale, non risolta installando pacchetti non dichiarati nel
    contratto condiviso di progetto."""


class FonteStatisticheNonRaggiungibile(IngestError):
    """Fetch di rete verso una fonte di statistiche storiche fallito (proxy,
    dominio bloccato, timeout, fonte irraggiungibile). Non degrada mai a dati
    sintetici o a una cache scaduta senza dirlo esplicitamente."""
