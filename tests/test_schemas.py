from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fantabuste.schemas import (
    AuctionState,
    BidPlan,
    OpponentBidObservation,
    Player,
    PlayerProjection,
    PriceDistribution,
    PriceDistributionMode,
    RepairLotResult,
    RosterSlots,
    assert_no_synthetic_dependency,
)


def _player(**overrides) -> Player:
    base = dict(
        player_id="P0001",
        nome="Test Player",
        ruolo="A",
        squadra="SQ01",
        quotazione_listone=10,
        fonte="test",
        is_synthetic=True,
        data_estrazione=datetime(2026, 8, 25, tzinfo=UTC),
    )
    base.update(overrides)
    return Player(**base)


def test_player_default_is_synthetic_true():
    p = Player.model_validate(
        {
            "player_id": "P0001",
            "nome": "X",
            "ruolo": "P",
            "squadra": "SQ01",
            "quotazione_listone": 5,
            "fonte": "test",
            "data_estrazione": datetime(2026, 8, 25, tzinfo=UTC),
        }
    )
    assert p.is_synthetic is True


def test_player_quotazione_deve_essere_positiva():
    with pytest.raises(ValidationError):
        _player(quotazione_listone=0)


def test_player_projection_intervallo_incoerente_rifiutato():
    with pytest.raises(ValidationError):
        PlayerProjection(
            player_id="P0001",
            punti_attesi=10,
            punti_attesi_lo=12,  # > punti_attesi: incoerente
            punti_attesi_hi=15,
            prob_titolarita=0.8,
            vorp=3.0,
            metodo="baseline",
            confidence_flag="bassa",
            fonte="test",
        )


def test_price_distribution_prior_richiede_mu_sigma():
    with pytest.raises(ValidationError):
        PriceDistribution(
            player_id="P0001",
            fascia="A_tier1",
            mode=PriceDistributionMode.PRIOR,
            n_osservazioni=0,
            fonte="test",
        )


def test_price_distribution_prior_p_win_monotona_in_b():
    dist = PriceDistribution(
        player_id="P0001",
        fascia="A_tier1",
        mode=PriceDistributionMode.PRIOR,
        n_osservazioni=0,
        prior_mu=0.0,  # centrata su ratio=1 (offerta = quotazione)
        prior_sigma=0.3,
        fonte="test",
    )
    quotazione = 20.0
    p_bassa = dist.p_win(5, quotazione)
    p_media = dist.p_win(20, quotazione)
    p_alta = dist.p_win(60, quotazione)
    assert 0 <= p_bassa < p_media < p_alta <= 1
    # centrata sul listone: offrire esattamente la quotazione è vicino al 50%
    assert 0.4 < p_media < 0.6


def test_price_distribution_empirical_p_win_ecdf():
    dist = PriceDistribution(
        player_id="P0001",
        fascia="A_tier1",
        mode=PriceDistributionMode.EMPIRICAL,
        n_osservazioni=4,
        empirical_support=[0.5, 0.8, 1.0, 1.2],  # rapporti offerta/quotazione osservati
        fonte="test",
    )
    quotazione = 10.0
    assert dist.p_win(4, quotazione) == 0.0  # ratio 0.4 < tutti gli osservati
    assert dist.p_win(15, quotazione) == 1.0  # ratio 1.5 > tutti gli osservati
    assert dist.p_win(9, quotazione) == pytest.approx(0.5)  # ratio 0.9: 2 osservazioni sotto


def test_roster_slots_totale():
    slots = RosterSlots(P=2, D=8, C=8, A=6)
    assert slots.totale == 24


def test_bid_plan_offerta_deve_essere_positiva():
    with pytest.raises(ValidationError):
        BidPlan(
            player_id="P0001",
            offerta=0,
            p_win_stimata=0.5,
            vorp=3.0,
            surplus_atteso=1.0,
            tornata=1,
            fonte="test",
        )


def test_opponent_bid_observation_tornata_valida():
    with pytest.raises(ValidationError):
        OpponentBidObservation(
            player_id="P0001",
            tornata=3,  # solo 1 o 2 sono ammesse
            avversario_id="AVV1",
            offerta=10,
            vincente=True,
            fonte="test",
        )


def test_auction_state_richiede_slot_totali_e_residui():
    state = AuctionState(
        fase="buste",
        tornata_corrente=1,
        budget_totale=500,
        budget_residuo=500,
        slot_totali=RosterSlots(P=2, D=8, C=8, A=6),
        slot_residui=RosterSlots(P=2, D=8, C=8, A=6),
        fonte="test",
    )
    assert state.slot_residui.totale == 24


def test_repair_lot_result_prezzo_finale_positivo():
    with pytest.raises(ValidationError):
        RepairLotResult(
            player_id="P0001",
            prezzo_finale=0,
            vinto_da_me=False,
            orario_chiusura=datetime(2026, 9, 10, tzinfo=UTC),
            fonte="test",
        )


def test_assert_no_synthetic_dependency_blocca_se_sintetico():
    p = _player(is_synthetic=True)
    with pytest.raises(ValueError, match="sintetici"):
        assert_no_synthetic_dependency(p)


def test_assert_no_synthetic_dependency_passa_se_reale():
    p = _player(is_synthetic=False)
    assert assert_no_synthetic_dependency(p) is None


def test_assert_no_synthetic_dependency_bypass_esplicito():
    p = _player(is_synthetic=True)
    assert assert_no_synthetic_dependency(p, allow_synthetic=True) is None
