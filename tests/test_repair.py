"""Placeholder — Modulo F (asta di riparazione) è FUORI SCOPE per la v1
(decisione 2026-08-25, vedi docs/OPEN_QUESTIONS.md §0): il mercato ad asta
resta aperto oltre il 7 settembre, quindi F si costruisce dopo la v1 delle
buste. Dipende dal codice del solver di D, non solo dal suo schema — parte
solo dopo che D è mergiato. Non implementare qui prematuramente."""

import pytest


def test_placeholder_modulo_f_fuori_scope_v1():
    pytest.skip("Modulo F: fuori scope v1, rimandato a dopo la tornata 2 — vedi OPEN_QUESTIONS §0")
