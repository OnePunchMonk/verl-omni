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
"""CPU tests for verl_omni.utils.config validation."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.config import validate_config


def _config(**trainer):
    return OmegaConf.create({"trainer": {"resume_mode": "disable", **trainer}})


def test_validate_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="Available options"):
        validate_config(_config(resume_mode="resumee"))


def test_validate_config_requires_resume_path():
    with pytest.raises(ValueError, match="resume_from_path"):
        validate_config(_config(resume_mode="resume_path"))


def test_validate_config_warns_when_dapo_reward_manager_is_used_with_gspo():
    config = OmegaConf.create(
        {
            "trainer": {"resume_mode": "disable"},
            "reward": {"reward_manager": {"name": "dapo"}},
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "gspo"}}},
        }
    )

    with pytest.warns(UserWarning, match="still runs GSPO, not DAPO"):
        validate_config(config)


def test_validate_config_accepts_dapo_reward_manager_with_vanilla_loss():
    config = OmegaConf.create(
        {
            "trainer": {"resume_mode": "disable"},
            "reward": {"reward_manager": {"name": "dapo"}},
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "vanilla"}}},
        }
    )

    validate_config(config)
