"""Per-fit regret caching and information-set-disjoint policy validation."""
from copy import deepcopy
from hashlib import blake2b
import math
import random
import time

import numpy as np
import torch

from .replay_grouping import CHUNK_ROWS, reservoir_counts, retained_context_counts


class RegretFitMixin:
    """Identical cached/uncached fitting, clipping and skip semantics for both games."""

    def train_model(self, iteration):
        started = time.perf_counter()
        self.last_fit_stats = {"rows": len(self.buffer), "replay": reservoir_counts(self.buffer), "updates": 0,
                               "clipped_updates": 0, "max_gradient_norm": 0.0}
        self.last_fit_stats["replay"]["retained_by_player_street"] = retained_context_counts(
            self.buffer, self.feature_encoder.replay_context_columns())
        self.last_target_cache_stats = {}
        # A deterministic opponent can terminate every hand before this player
        # acts. Preserve the network/optimizer instead of fitting an empty batch.
        if not len(self.buffer) or self.train_steps <= 0:
            self.last_fit_stats.update(loss=None, seconds=0.0)
            return None
        loss = (fit_cached_regrets(self, iteration) if self.cache_frozen_targets
                else super().train_model(iteration))
        self.last_fit_stats.update(loss=loss, updates=self.train_steps,
                                   seconds=time.perf_counter() - started)
        return loss

    def _clip_gradients(self):
        if self.gradient_clip_norm is not None:
            norm = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
            ).item())
            self.last_fit_stats["max_gradient_norm"] = max(
                self.last_fit_stats["max_gradient_norm"], norm)
            self.last_fit_stats["clipped_updates"] += int(norm > self.gradient_clip_norm)



def fit_cached_regrets(trainer, iteration):
    size = min(trainer.buffer.cur_id, trainer.buffer.buffer_size)
    if size == 0:
        return None
    batch = min(trainer.batch_size, size)
    started = time.perf_counter()
    targets = torch.empty((size, trainer.output_size), device=trainer.device)
    age = math.pow(iteration - 1, trainer.alpha)
    # No sampling or RNG consumption while constructing the frozen labels.
    with torch.no_grad():
        for start in range(0, size, batch):
            stop = min(start + batch, size)
            indices = np.resize(np.arange(start, stop), batch)
            features = torch.as_tensor(trainer.buffer.infostate_buf[indices], device=trainer.device)
            masks = torch.as_tensor(trainer.buffer.q_value_mask_buf[indices], device=trainer.device)
            previous = trainer.predict(trainer.target_model, features, masks)
            values = torch.as_tensor(trainer.buffer.q_value_buf[indices], device=trainer.device)
            targets[start:stop] = (previous.clamp_min(0) * age / (age + 1.5) + values)[:stop-start]
    if not torch.isfinite(targets).all():
        raise FloatingPointError("Nonfinite frozen regret targets")
    trainer.last_target_cache_stats = {
        "cached_rows": size, "cache_bytes": targets.numel() * targets.element_size(),
        "cache_build_seconds": time.perf_counter() - started, "cache_lifetime": "this_fit_only",
    }
    for _ in range(trainer.train_steps):
        indices = (slice(0, size) if trainer.batch_size >= size else
                   trainer.buffer.rng.choice(size, size=trainer.batch_size, replace=False))
        features = torch.as_tensor(trainer.buffer.infostate_buf[indices], device=trainer.device)
        masks = torch.as_tensor(trainer.buffer.q_value_mask_buf[indices], device=trainer.device)
        loss = trainer.loss_fn(trainer.predict(trainer.model, features, masks), targets[indices])
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer._clip_gradients()
        trainer.optimizer.step()
    trainer.target_model.load_state_dict(trainer.model.state_dict())
    return float(loss.item())


def policy_validation_mask(features, fraction):
    """Stable membership across fits; grouped identical infosets cannot leak."""
    threshold = int(fraction * 2**64)
    heldout = np.empty(len(features), dtype=bool)
    for start in range(0, len(features), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(features))
        array = (features.numpy_batch(start, stop) if hasattr(features, "numpy_batch")
                 else features[start:stop].detach().cpu().numpy().copy())
        array[array == 0] = 0
        heldout[start:stop] = [int.from_bytes(blake2b(row.tobytes(), digest_size=8,
            person=b"holdem-policy-v1").digest(), "little") < threshold for row in array]
    return heldout


def weighted_policy_ce(model, features, targets, masks, weights, indices, batch_size):
    total, mass = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = torch.as_tensor(indices[start:start+batch_size], device=features.device)
            logits = model(features[idx]).masked_fill(masks[idx] != 1, -1e20)
            losses = -(targets[idx] * torch.log_softmax(logits, -1)).sum(-1)
            total += float((losses.double() * weights[idx].double()).sum().item())
            mass += float(weights[idx].double().sum().item())
    return total / mass


def fit_validated_policy(trainer, iteration):
    features, targets, masks, weights = trainer._grouped_training_data(iteration)
    heldout = policy_validation_mask(features, trainer.validation_fraction)
    train_ids, validation_ids = np.flatnonzero(~heldout), np.flatnonzero(heldout)
    # Tiny smoke/early replay can contain no validation or no training groups.
    # Do not change stable membership to manufacture a validation result.
    validating = bool(len(train_ids) and len(validation_ids))
    if not validating:
        train_ids = np.arange(len(features))
    batch_size = min(trainer.batch_size, len(train_ids))
    checkpoints = {step for step in trainer.validation_steps if step <= trainer.train_steps}
    checkpoints.add(trainer.train_steps)
    history, best, best_score = [], None, float("inf")
    trainer.model.train()
    for step in range(1, trainer.train_steps + 1):
        selected = (train_ids if batch_size == len(train_ids) else
                    train_ids[random.sample(range(len(train_ids)), batch_size)])
        idx = torch.as_tensor(selected, device=features.device)
        logits = trainer.model(features[idx]).masked_fill(masks[idx] != 1, -1e20)
        losses = -(targets[idx] * torch.log_softmax(logits, -1)).sum(-1)
        loss = (losses * weights[idx]).mean()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        if step in checkpoints and validating:
            trainer.model.eval()
            score = weighted_policy_ce(trainer.model, features, targets, masks, weights,
                                       validation_ids, trainer.batch_size)
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite validation policy loss")
            history.append({"update": step, "weighted_ce": score})
            if score < best_score:
                best_score = score
                best = (deepcopy(trainer.model.state_dict()), deepcopy(trainer.optimizer.state_dict()),
                        step, float(loss.item()))
            trainer.model.train()
    if best is not None:
        model, optimizer, selected_step, selected_loss = best
        trainer.model.load_state_dict(model)
        trainer.optimizer.load_state_dict(optimizer)
    else:
        selected_step, selected_loss = trainer.train_steps, float(loss.item())
    trainer.model.eval()
    trainer.last_fit_diagnostics = {
        "split": "canonical_feature_blake2b_v1", "validation_fraction": trainer.validation_fraction,
        "training_groups": len(train_ids), "validation_groups": len(validation_ids) if validating else 0,
        "validation_available": validating, "updates_completed": trainer.train_steps,
        "selected_update": selected_step, "validation_weighted_ce": best_score if validating else None,
        "validation_history": history,
    }
    return selected_loss
