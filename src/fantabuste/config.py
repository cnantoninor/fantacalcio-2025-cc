"""Caricamento tipizzato di config/league.yaml.

Nessun modulo deve leggere il file YAML direttamente né hardcodare un
parametro di regolamento — passa sempre da LeagueConfig.load().
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from fantabuste.schemas import RosterSlots

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "league.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "league.example.yaml"


class AstaRiparazioneConfig(BaseModel):
    tipo: str
    sequenziale: bool
    soft_close: bool
    timer_secondi: int | None = None
    watchlist_size: int = Field(gt=0)


class LeagueConfig(BaseModel):
    n_partecipanti: int = Field(gt=1)

    budget_totale_fase1: float = Field(gt=0)
    budget_bonus_fase2: float = Field(ge=0)

    rosa_fase1: RosterSlots
    rosa_fase2_massima: RosterSlots

    n_tornate_buste: int = Field(gt=0)
    assegna_buste_se_uguali: bool
    offerte_non_tonde: bool

    modificatore_difesa: bool
    regolamento_portieri_top8_attivo: bool

    max_giocatori_per_squadra_serie_a: int | None = None

    bidmodel_min_osservazioni: int = Field(gt=0)

    asta_riparazione: AstaRiparazioneConfig

    @model_validator(mode="after")
    def _rosa_fase2_non_inferiore_a_fase1(self) -> LeagueConfig:
        for ruolo in ("P", "D", "C", "A"):
            if getattr(self.rosa_fase2_massima, ruolo) < getattr(self.rosa_fase1, ruolo):
                raise ValueError(
                    f"rosa_fase2_massima.{ruolo} non può essere inferiore a rosa_fase1.{ruolo}"
                )
        return self

    @property
    def budget_totale_fase2(self) -> float:
        return self.budget_totale_fase1 + self.budget_bonus_fase2

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> LeagueConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} non trovato. Copia config/league.example.yaml in "
                "config/league.yaml e correggi i valori DA CONFERMARE."
            )
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
