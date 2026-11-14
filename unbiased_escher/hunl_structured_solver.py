"""Lossless suit-canonical HUNL variant of grouped-wide UCV-ESCHER."""

from __future__ import annotations

from copy import deepcopy
import random
import time

import numpy as np
import torch

from hunl_escher.game import DEFAULT_ACTION_MENU, load_hunl_game
from hunl_escher.optimization import ReplayScaledFitMixin
from hunl_escher.features import (
    HUNLFeatureEncoder,
    ENCODER_ID,
    make_feature_encoder,
    FeatureLayout,
    StructuredHUNLMLP,
)
from vr_deep_cfr.solver import AvePolicyTrainer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .grouped_wide_solver import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
    GroupedWideUnbiasedControlVariateEscher,
    TemporallyAveragedCrossFittedQEnsemble,
    TemporallyAveragedCrossFittedQMember,
)
from .efficient_replay import CompactCircularBuffer, CompactReservoirBuffer
from .solver import ResidualCalibrationTrainer, CalibrationBuffer
class _StructuredModelMixin:
    feature_layout: FeatureLayout
    structured_branch_width: int

    def init_model(self):
        return StructuredHUNLMLP(
            self.feature_layout,
            self.network_layers,
            self.output_size,
            branch_width=self.structured_branch_width,
        ).to(self.device)


class StructuredHUNLRegretTrainer(_StructuredModelMixin, VRDCFRPlusRegretTrainer):
    def __init__(
        self,
        *args,
        feature_encoder: HUNLFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        gradient_clip_norm=None,
        **kwargs,
    ):
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.policy_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        self.gradient_clip_norm = gradient_clip_norm
        super().__init__(*args, **kwargs)
        self.predictor_enabled = False

    def train_model(self, iteration):
        started = time.perf_counter()
        self.last_fit_stats = {"rows": len(self.buffer), "updates": 0,
                               "clipped_updates": 0, "max_gradient_norm": 0.0}
        # A deterministic opponent can terminate every hand before this player
        # acts. Preserve the network/optimizer instead of fitting an empty batch.
        if not len(self.buffer) or self.train_steps <= 0:
            self.last_fit_stats.update(loss=None, seconds=0.0)
            return None
        loss = super().train_model(iteration)
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

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def init_buffer(self):
        return CompactReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHUNLAveragePolicyTrainer(
    _StructuredModelMixin, GroupedSoftTargetCrossEntropyAvePolicyTrainer
):
    def __init__(
        self,
        *args,
        feature_encoder: HUNLFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        **kwargs,
    ):
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.policy_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        super().__init__(*args, **kwargs)

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def init_buffer(self):
        return CompactReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHUNLQMember(
    ReplayScaledFitMixin, _StructuredModelMixin, TemporallyAveragedCrossFittedQMember
):
    def __init__(
        self,
        *args,
        feature_encoder: HUNLFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        minimum_train_steps: int,
        **kwargs,
    ):
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.full_state_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        self.minimum_train_steps = int(minimum_train_steps)
        super().__init__(*args, **kwargs)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)

    def init_buffer(self):
        return CompactCircularBuffer(
            self.buffer_size,
            self.input_size,
            self.state_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHUNLQEnsemble(TemporallyAveragedCrossFittedQEnsemble):
    def __init__(
        self,
        *,
        feature_encoder: HUNLFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        target_average_window: int,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        total_buffer_size: int,
        batch_size: int,
        train_steps: int,
        minimum_train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
    ):
        self.ensemble_size = int(ensemble_size)
        self.target_average_window = int(target_average_window)
        if self.ensemble_size < 1 or self.target_average_window < 1:
            raise ValueError("Ensemble size and target-average window must be positive")
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            StructuredHUNLQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                member_buffer_size,
                batch_size,
                train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
                target_average_window=self.target_average_window,
                minimum_train_steps=minimum_train_steps,
                feature_encoder=feature_encoder,
                branch_width=branch_width,
                replay_seed=int(replay_seed) + index,
            )
            for index in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0


class StructuredHUNLResidualCalibrationTrainer(ReplayScaledFitMixin, ResidualCalibrationTrainer):
    def __init__(
        self,
        *,
        infostate_size: int,
        action_size: int,
        hidden_layers,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        minimum_train_steps: int,
        device: str,
        minimum_variance: float,
        feature_encoder: HUNLFeatureEncoder,
        branch_width: int,
    ):
        self.feature_encoder = feature_encoder
        self.minimum_train_steps = int(minimum_train_steps)
        super().__init__(
            infostate_size=infostate_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            train_steps=train_steps,
            device=device,
            minimum_variance=minimum_variance,
        )
        self.feature_encoder = feature_encoder
        self.feature_layout = FeatureLayout(
            "residual_calibration",
            self.feature_size,
            feature_encoder.policy_layout.card_size,
            feature_encoder.policy_layout.strategic_size,
        )
        hidden_layers = list(hidden_layers)
        self.model = StructuredHUNLMLP(
            self.feature_layout,
            hidden_layers,
            2,
            branch_width=int(branch_width),
        ).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(learning_rate)
        )


class StructuredHUNLGroupedWideUCVEscher(GroupedWideUnbiasedControlVariateEscher):
    """Selected UCV learner with four-street HUNL neural features."""

    def __init__(
        self,
        *args,
        regret_network_layers=(256, 256, 256),
        critic_network_layers=(192, 192),
        calibration_network_layers=(128, 128),
        average_policy_network_layers=(256, 256, 256),
        regret_branch_width: int = 128,
        critic_branch_width: int = 96,
        calibration_branch_width: int = 64,
        average_policy_branch_width: int = 128,
        critic_min_train_steps: int = 250,
        calibration_min_train_steps: int = 100,
        feature_encoder_id: str = ENCODER_ID,
        action_menu=DEFAULT_ACTION_MENU,
        **kwargs,
    ):
        self.feature_encoder = make_feature_encoder(feature_encoder_id, action_menu)
        self.regret_network_layers = list(regret_network_layers)
        self.critic_network_layers = list(critic_network_layers)
        self.calibration_network_layers = list(calibration_network_layers)
        self.regret_branch_width = regret_branch_width
        self.critic_branch_width = critic_branch_width
        self.calibration_branch_width = calibration_branch_width
        self.average_policy_branch_width = average_policy_branch_width
        self.critic_min_train_steps = int(critic_min_train_steps)
        self.calibration_min_train_steps = int(calibration_min_train_steps)
        self.feature_replay_seed = int(kwargs.get("seed", 0))
        super().__init__(*args, average_policy_network_layers=average_policy_network_layers,
                         **kwargs)
        if self.average_policy_loss != GROUPED_SOFT_TARGET_CROSS_ENTROPY:
            raise ValueError("The structured experiment requires grouped policy CE")

    def load_game(self):
        return load_hunl_game(self.feature_encoder.action_menu)

    def _reset_architecture_diagnostics(self):
        super()._reset_architecture_diagnostics()
        for player in range(2):
            for street in range(4):
                self._architecture_stats[f"regret_rows_player_{player}_street_{street}"] = 0

    def dfs(self, state, traverser, *args, **kwargs):
        if not state.is_terminal() and state.current_player() == traverser:
            street = state.decision_view().street
            self._architecture_stats[f"regret_rows_player_{traverser}_street_{street}"] += 1
        return super().dfs(state, traverser, *args, **kwargs)

    def init_ave_policy_trainer(self):
        # DeepCumuAdv initially discovers OpenSpiel's raw size. From this hook
        # onwards every learner consistently uses the versioned HUNL encoder.
        self.raw_open_spiel_infostate_size = int(self.infostate_size)
        self.infostate_size = self.feature_encoder.policy_size
        # Keep the inherited generic architecture attribute aligned with regret.
        # Every HUNL trainer below uses its own explicit role configuration.
        self.network_layers = list(self.regret_network_layers)

        # Preserve Experiment 35's legacy RNG anchor so the algorithmic change
        # is limited to representation and structured model capacity.
        legacy_rng_anchor = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            list(self.paired_legacy_network_layers),
            self.paired_legacy_learning_rate,
            1,
            self.ave_policy_batch_size,
            self.paired_legacy_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )
        rng_after_legacy = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_after_legacy["cuda"] = torch.cuda.get_rng_state_all()
        self.ave_policy_trainer = StructuredHUNLAveragePolicyTrainer(
            self.infostate_size,
            self.action_size,
            list(self.average_policy_network_layers),
            self.average_policy_learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.average_policy_train_steps,
            self.logger,
            self.device,
            self.gamma,
            feature_encoder=self.feature_encoder,
            branch_width=self.average_policy_branch_width,
            replay_seed=self.feature_replay_seed + 100,
        )
        random.setstate(rng_after_legacy["python"])
        np.random.set_state(rng_after_legacy["numpy"])
        torch.random.set_rng_state(rng_after_legacy["torch"])
        if "cuda" in rng_after_legacy:
            torch.cuda.set_rng_state_all(rng_after_legacy["cuda"])
        del legacy_rng_anchor

    def init_regret_trainers(self):
        if self.use_instantaneous_predictor:
            raise ValueError("Structured HUNL requires non-predictive UCV")
        self.regret_trainers = [
            StructuredHUNLRegretTrainer(
                self.infostate_size,
                self.action_size,
                self.regret_network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
                feature_encoder=self.feature_encoder,
                branch_width=self.regret_branch_width,
                replay_seed=self.feature_replay_seed + 200 + player,
                gradient_clip_norm=self.regret_policy_gradient_clip_norm,
            )
            for player in range(self.num_players)
        ]

    def init_q_value_trainer(self):
        self.q_value_trainer = StructuredHUNLQEnsemble(
            feature_encoder=self.feature_encoder,
            branch_width=self.critic_branch_width,
            replay_seed=self.feature_replay_seed + 300,
            target_average_window=self.critic_target_average_window,
            ensemble_size=self.q_ensemble_size,
            history_size=self.feature_encoder.full_state_size,
            state_size=self.feature_encoder.policy_size,
            action_size=self.action_size,
            network_layers=self.critic_network_layers,
            learning_rate=self.learning_rate,
            total_buffer_size=self.baseline_buffer_size,
            batch_size=self.baseline_batch_size,
            train_steps=self.baseline_network_train_steps,
            minimum_train_steps=self.critic_min_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
        )

    def init_calibration_trainer(self):
        self.calibration_trainer = StructuredHUNLResidualCalibrationTrainer(
            infostate_size=self.infostate_size,
            action_size=self.action_size,
            hidden_layers=self.calibration_network_layers,
            learning_rate=self.calibration_learning_rate,
            buffer_size=self.calibration_buffer_size,
            batch_size=self.calibration_batch_size,
            train_steps=self.calibration_train_steps,
            minimum_train_steps=self.calibration_min_train_steps,
            device=self.device,
            minimum_variance=self.calibration_minimum_variance,
            feature_encoder=self.feature_encoder,
            branch_width=self.calibration_branch_width,
        )

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)

    def evaluate(self, **kwargs):
        self.logger.record("feature_encoder_id", self.feature_encoder.encoder_id)
        self.logger.record("policy_feature_size", self.feature_encoder.policy_size)
        self.logger.record("critic_feature_size", self.feature_encoder.full_state_size)
        self.logger.record("raw_open_spiel_feature_size", self.raw_open_spiel_infostate_size)
        replay_bytes = sum(trainer.buffer.nbytes() for trainer in self.regret_trainers)
        replay_bytes += self.ave_policy_trainer.buffer.nbytes()
        replay_bytes += sum(member.buffer.nbytes() for member in self.q_value_trainer.members)
        self.logger.record("primary_replay_allocated_bytes", replay_bytes)
        return super().evaluate(**kwargs)


__all__ = [
    "StructuredHUNLGroupedWideUCVEscher",
    "StructuredHUNLAveragePolicyTrainer",
    "StructuredHUNLQEnsemble",
    "StructuredHUNLRegretTrainer",
]
