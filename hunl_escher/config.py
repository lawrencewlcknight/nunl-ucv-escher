"""Selected source learning settings and deliberately small engineering smoke."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .features import ENCODER_ID
from .game import validate_action_menu, load_hunl_game


def baseline_config():
    return json.loads((Path(__file__).parent / "data/baseline.json").read_text())


def smoke_config():
    config = baseline_config()
    for key in ("advantage_buffer_size", "ave_policy_buffer_size",
                "baseline_buffer_size", "calibration_buffer_size"):
        config[key] = 256
    for key in ("advantage_batch_size", "ave_policy_batch_size",
                "baseline_batch_size", "calibration_batch_size"):
        config[key] = 8
    for key in ("advantage_network_train_steps", "ave_policy_network_train_steps",
                "baseline_network_train_steps", "calibration_train_steps",
                "average_policy_train_steps"):
        config[key] = 2
    config["policy_validation_steps"] = [1, 2]
    config["num_traversals"] = 16
    config["critic_min_train_steps"] = 1
    config["calibration_min_train_steps"] = 1
    return config


def validate_config(config):
    validate_action_menu(config["action_menu"])
    expected = baseline_config()
    if set(config) != set(expected):
        raise ValueError("Configuration keys differ from the versioned baseline")
    for role in ("regret", "critic", "calibration", "average_policy"):
        width_key = f"{role}_branch_width"
        if type(config[width_key]) is not int or config[width_key] <= 0:
            raise ValueError(f"{width_key} must be a positive integer")
        layers_key = f"{role}_network_layers"
        layers = config[layers_key]
        if (not isinstance(layers, list) or not layers
                or any(type(width) is not int or width <= 0 for width in layers)):
            raise ValueError(f"{layers_key} must be a nonempty list of positive integers")
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size",
                "calibration_buffer_size", "num_traversals", "advantage_batch_size",
                "ave_policy_batch_size", "baseline_batch_size", "calibration_batch_size",
                "advantage_network_train_steps", "baseline_network_train_steps",
                "calibration_train_steps", "average_policy_train_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key, maximum_key in (("critic_min_train_steps", "baseline_network_train_steps"),
                             ("calibration_min_train_steps", "calibration_train_steps")):
        if type(config[key]) is not int or not 1 <= config[key] <= config[maximum_key]:
            raise ValueError(f"{key} must be a positive integer no larger than {maximum_key}")
    clip = config["regret_policy_gradient_clip_norm"]
    if type(clip) not in (float, int) or not math.isfinite(clip) or clip <= 0:
        raise ValueError("regret_policy_gradient_clip_norm must be positive and finite")
    if config["game_name"] != "HUNL" or config["feature_encoder_id"] != ENCODER_ID:
        raise ValueError("Use the versioned HUNL game and encoder")
    if config["q_ensemble_size"] != 2 or config["use_instantaneous_predictor"]:
        raise ValueError("Baseline requires two cross-fit critics and no predictor")
    if config["evaluation_frequency"] != 0 or config["evaluate_initial_policy"]:
        raise ValueError("The HUNL runner owns evaluation and checkpoint boundaries")

    for key in ("cache_frozen_regret_targets", "uniform_regret_ties"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    fraction = config["policy_validation_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("policy_validation_fraction must be between zero and one")
    steps = config["policy_validation_steps"]
    if (not isinstance(steps, list) or not steps
            or any(type(step) is not int or step <= 0 for step in steps)
            or steps != sorted(set(steps))):
        raise ValueError("policy_validation_steps must be increasing positive integers")


def make_solver(config=None, *, seed=0, workers=1):
    from unbiased_escher.hunl_structured_solver import StructuredHUNLGroupedWideUCVEscher
    from unbiased_escher.grouped_parallel_solver import ParallelStructuredGroupedUCVEscher
    from vr_deep_cfr.logger import Logger

    config = deepcopy(baseline_config() if config is None else config)
    validate_config(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    kwargs = dict(config)
    cache = kwargs.pop("cache_frozen_critic_targets")
    runtime = {key: kwargs.pop(key) for key in (
        "max_num_iterations", "preserve_evaluation_rng", "evaluate_initial_policy",
        "early_evaluation_node_thresholds")}
    kwargs.update(seed=seed, logger=Logger(), num_episodes=2 * config["num_traversals"])
    if workers == 1:
        solver = StructuredHUNLGroupedWideUCVEscher(**kwargs)
    else:
        solver = ParallelStructuredGroupedUCVEscher(
            **kwargs, parallel_num_workers=workers, parallel_run_seed=seed,
            parallel_collection_chunk_size=min(1200, config["num_traversals"],
                max(1, 28800 // (load_hunl_game(config["action_menu"]).max_game_length() + 1))),
            parallel_ray_object_store_memory=128 * 1024**2,
            parallelize_independent_learners=False, parallel_cache_actor_snapshots=True)
    for key, value in runtime.items():
        setattr(solver, key, value)
    for member in solver.q_value_trainer.members:
        member.cache_frozen_targets = cache
    return solver
