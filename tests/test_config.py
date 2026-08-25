import pytest
from pydantic import ValidationError

from fantabuste.config import EXAMPLE_CONFIG_PATH, LeagueConfig


def test_example_config_carica_ed_e_valido():
    cfg = LeagueConfig.load(EXAMPLE_CONFIG_PATH)
    assert cfg.n_partecipanti == 12
    assert cfg.budget_totale_fase1 == 500
    assert cfg.n_tornate_buste == 2


def test_budget_totale_fase2_somma_il_bonus():
    cfg = LeagueConfig.load(EXAMPLE_CONFIG_PATH)
    assert cfg.budget_totale_fase2 == cfg.budget_totale_fase1 + cfg.budget_bonus_fase2


def test_rosa_fase1_coerente_con_regolamento():
    cfg = LeagueConfig.load(EXAMPLE_CONFIG_PATH)
    assert cfg.rosa_fase1.totale == 24
    assert cfg.rosa_fase2_massima.totale == 32


def test_load_file_mancante_solleva_errore_chiaro(tmp_path):
    with pytest.raises(FileNotFoundError, match="league.example.yaml"):
        LeagueConfig.load(tmp_path / "non_esiste.yaml")


def test_rosa_fase2_inferiore_a_fase1_rifiutata():
    raw = LeagueConfig.load(EXAMPLE_CONFIG_PATH).model_dump()
    raw["rosa_fase2_massima"]["P"] = 1  # < rosa_fase1.P (2)
    with pytest.raises(ValidationError):
        LeagueConfig.model_validate(raw)
