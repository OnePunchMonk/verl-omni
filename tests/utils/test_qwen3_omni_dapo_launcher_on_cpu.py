# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS = (
    "python3 -m verl_omni.trainer.main_omni",
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
    "actor_rollout_ref.actor.clip_ratio_low=0.2",
    "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.actor.clip_ratio_c=10.0",
    "actor_rollout_ref.actor.loss_agg_mode=token-mean",
    "actor_rollout_ref.actor.use_kl_loss=false",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=false",
    "algorithm.filter_groups.enable=false",
    "reward.reward_manager.source=register",
    "reward.reward_manager.name=dapo",
    'engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"',
)


def _assert_dapo_without_dynamic_sampling_contract(script: str) -> None:
    assert all(setting in script for setting in DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS)
    assert "policy_loss.loss_mode=gspo" not in script
    assert "algorithm.filter_groups.enable=true" not in script
    assert "overlong_buffer_cfg" not in script


def test_dapo_example_launcher_has_phase_one_contract():
    repo_root = Path(__file__).parents[2]
    launcher = (repo_root / "examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh").read_text(
        encoding="utf-8"
    )

    _assert_dapo_without_dynamic_sampling_contract(launcher)
    assert ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" in launcher
    assert "actor_rollout_ref.actor.freeze_vision_tower=true" in launcher


def test_dapo_tiny_random_smoke_matches_example_contract():
    repo_root = Path(__file__).parents[2]
    smoke = (repo_root / "tests/special_e2e/run_dapo_qwen3_omni_thinker_lora_v1_smoke.sh").read_text(encoding="utf-8")

    _assert_dapo_without_dynamic_sampling_contract(smoke)
    assert "build_qwen3_omni_tiny_random.py" in smoke
    assert "SKIP_COMPAT_DEPS_INSTALL:-0" in smoke
    assert 'trainer.total_training_steps="${TOTAL_TRAIN_STEPS}"' in smoke
