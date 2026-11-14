"""Preflop decision features, strict nuts, and frozen v4 contracts."""
import json
from pathlib import Path

import numpy as np
import pytest

from hunl_escher import strategic_features as sf, v4_strategic_features as old
from hunl_escher.features import StrategicV4HUNLFeatureEncoder, encoder_from_metadata
from test_hunl import state_at_street
from test_strategic_features import cards


def test_strict_nuts_excludes_ties_and_preflop():
    for h,b,unbeatable,strict in [('2c 3d','As Ks Qs Js Ts',1,0),
                                ('As Ks','Qs Js Ts 2c 3d',1,1),
                                ('As Ad','',0,0)]:
        f = dict(zip(sf.CARD_NAMES,sf.card_features(cards(h),cards(b))))
        assert f['currently_unbeatable'] == unbeatable
        assert f['currently_strict_nuts'] == strict


def test_preflop_interactions_are_priced_and_observer_safe():
    for street in range(4):
        state=state_at_street(street)
        for observer in (0,1):
            view=state.decision_view(observer)
            own=sf.card_features(view.hole_cards,view.board_cards)
            actual=dict(zip(sf.economics_names(view.action_menu),sf.betting_features(view,own)))
            previous=dict(zip(old.economics_names(view.action_menu),old.betting_features(view,own[:62])))
            for name,value in previous.items(): assert actual[name] == value
            np.testing.assert_array_equal(own[:62],old.card_features(view.hole_cards,view.board_cards))
            available=street == 0 and observer == view.actor
            equity=own[sf.CARD_NAMES.index('preflop_equity')]
            for name,expected in [('minus_pot_odds',equity-actual['call_pot_odds']),
                                  ('times_call_fraction',equity*actual['call_stack_fraction']),
                                  ('times_stack_share',equity*actual['effective_stack_pot_share'])]:
                assert actual['preflop_equity_'+name] == pytest.approx(expected if available else 0)


def test_v4_metadata_is_frozen():
    metadata=json.loads((Path(__file__).parent/'fixtures/encoder_v4_metadata.json').read_text())
    assert StrategicV4HUNLFeatureEncoder().metadata() == metadata
    assert encoder_from_metadata(metadata).metadata() == metadata
