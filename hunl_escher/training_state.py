"""Completed-iteration continuation, including all private replay and actor RNGs."""
from copy import deepcopy

from . import state_storage


def _replay_buffers(solver):
    return [solver.ave_policy_trainer.buffer,
            *[t.buffer for t in solver.regret_trainers],
            *[m.buffer for m in solver.q_value_trainer.members]]


def build_training_state(solver, **kwargs):
    result = state_storage.build_training_state(solver, **kwargs)
    result["game"] = solver.game.definition()
    result["encoder"] = solver.feature_encoder.metadata()
    result["replay_rng"] = [deepcopy(b.rng.bit_generator.state) for b in _replay_buffers(solver)]
    workers = getattr(solver, "_workers", [])
    result["actors"] = (solver._ray.get([w.continuation_state.remote() for w in workers])
                        if workers else [])
    result["parallel_contract"] = {name: getattr(solver, name, None) for name in (
        "_parallel_num_workers", "_parallel_run_seed", "_parallel_collection_chunk_size",
        "_parallel_cache_actor_snapshots", "_parallelize_independent_learners")}
    return result


def restore_training_state(solver, payload, **kwargs):
    if payload["game"] != solver.game.definition():
        raise ValueError("Continuation game mismatch")
    if payload["encoder"] != solver.feature_encoder.metadata():
        raise ValueError("Continuation encoder mismatch")
    if payload["parallel_contract"] != {name: getattr(solver, name, None)
                                       for name in payload["parallel_contract"]}:
        raise ValueError("Continuation worker configuration mismatch")
    workers = getattr(solver, "_workers", [])
    if len(workers) != len(payload["actors"]):
        raise ValueError("Continuation worker count mismatch")
    result = state_storage.restore_training_state(solver, payload, **kwargs)
    for buffer, rng in zip(_replay_buffers(solver), payload["replay_rng"], strict=True):
        buffer.rng.bit_generator.state = deepcopy(rng)
    if workers:
        solver._ray.get([w.restore_continuation_state.remote(state)
                        for w, state in zip(workers, payload["actors"], strict=True)])
    return result


save_training_state = state_storage.save_sharded_training_state
read_training_state = state_storage.read_sharded_training_state
