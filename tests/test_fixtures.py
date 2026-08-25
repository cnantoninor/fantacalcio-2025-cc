from fantabuste.fixtures import N_SQUADRE, RUOLI_PER_SQUADRA, genera_fixture_giocatori


def test_fixture_riproducibile_da_seed():
    players_a, stats_a = genera_fixture_giocatori(seed=42)
    players_b, stats_b = genera_fixture_giocatori(seed=42)
    assert [p.model_dump() for p in players_a] == [p.model_dump() for p in players_b]
    assert [s.model_dump() for s in stats_a] == [s.model_dump() for s in stats_b]


def test_fixture_seed_diverso_produce_output_diverso():
    players_a, _ = genera_fixture_giocatori(seed=1)
    players_b, _ = genera_fixture_giocatori(seed=2)
    assert players_a != players_b


def test_fixture_conteggio_giocatori():
    players, _ = genera_fixture_giocatori()
    attesi = N_SQUADRE * sum(RUOLI_PER_SQUADRA.values())
    assert len(players) == attesi == 600


def test_fixture_tutti_marcati_sintetici():
    players, stats = genera_fixture_giocatori()
    assert all(p.is_synthetic for p in players)
    assert all(s.is_synthetic for s in stats)


def test_fixture_due_stagioni_per_giocatore():
    players, stats = genera_fixture_giocatori()
    by_player = {}
    for s in stats:
        by_player.setdefault(s.player_id, set()).add(s.stagione)
    assert len(by_player) == len(players)
    assert all(len(stagioni) == 2 for stagioni in by_player.values())


def test_fixture_player_id_univoci():
    players, _ = genera_fixture_giocatori()
    ids = [p.player_id for p in players]
    assert len(ids) == len(set(ids))


def test_fixture_ruoli_plausibili():
    players, _ = genera_fixture_giocatori()
    ruoli = {p.ruolo for p in players}
    assert ruoli == {"P", "D", "C", "A"}
