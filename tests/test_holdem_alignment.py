"""Protect the agreed cross-variant specification from accidental drift."""
import json
from pathlib import Path
import pytest


def test_only_game_size_and_representation_change_the_baseline():
    parent = Path(__file__).resolve().parents[2]
    paths = [parent / f"{tag}-ucv-escher/{tag}_escher/data/baseline.json"
             for tag in ("hunl", "hulhe")]
    if not all(path.exists() for path in paths):
        pytest.skip("Sibling Hold'em checkout is not available")
    nl, limit = [json.loads(path.read_text()) for path in paths]
    limit["regret_network_layers"] = [limit.pop("num_hiddens")] * limit.pop("num_layers")
    width = limit.pop("structured_branch_width")
    for role in ("regret", "critic", "average_policy"):
        limit[f"{role}_branch_width"] = width
    allowed = {"game_name", "feature_encoder_id", "action_menu", "advantage_buffer_size",
               "regret_network_layers", "average_policy_network_layers",
               "regret_branch_width", "average_policy_branch_width"}
    differences = {key for key in nl.keys() | limit.keys() if nl.get(key) != limit.get(key)}
    assert differences == allowed
    for tag in ("hunl", "hulhe"):
        base = parent / f"{tag}-ucv-escher"
        other = parent / ("hulhe-ucv-escher" if tag == "hunl" else "hunl-ucv-escher")
        other_tag = "hulhe" if tag == "hunl" else "hunl"
        for path, other_path in (
            ("unbiased_escher/holdem_fitting.py", "unbiased_escher/holdem_fitting.py"),
            (f"{tag}_escher/optimization.py", f"{other_tag}_escher/optimization.py"),
            (f"{tag}_escher/checkpoint_schedule.py", f"{other_tag}_escher/checkpoint_schedule.py"),
            (f"{tag}_escher/recovery_retention.py", f"{other_tag}_escher/recovery_retention.py"),
        ):
            assert (base / path).read_bytes() == (other / other_path).read_bytes()
