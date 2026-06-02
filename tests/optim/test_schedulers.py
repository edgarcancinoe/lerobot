# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import torch
import pytest
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
    XVLAStagedPromptWarmupSchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)
from lerobot.utils.constants import SCHEDULER_STATE


def test_diffuser_scheduler(optimizer):
    config = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()  # so that we don't get torch warning
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0002],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_vqbet_scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_cosine_decay_with_warmup_scheduler(optimizer):
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001818181818181819],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()


def test_xvla_staged_prompt_warmup_scheduler():
    params = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(4)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": 5e-5, "name": "vlm"},
            {"params": [params[1]], "lr": 1e-4, "name": "transformer_core"},
            {"params": [params[2]], "lr": 5e-5, "name": "soft_prompts"},
            {"params": [params[3]], "lr": 1e-4, "name": "action_heads"},
        ]
    )
    scheduler = XVLAStagedPromptWarmupSchedulerConfig(
        freeze_steps=2,
        num_warmup_steps=2,
        num_decay_steps=10,
        peak_lr=1e-4,
        decay_lr=1e-5,
        learning_coef=0.5,
    ).build(optimizer, num_training_steps=10)
    assert isinstance(scheduler, LambdaLR)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.0, 0.0, 5e-5, 1e-4])
    for expected in (
        [0.0, 0.0, 5e-5, 1e-4],
        [0.0, 0.0, 0.0, 0.0],
        [2.5e-5, 5e-5, 2.5e-5, 5e-5],
    ):
        optimizer.step()
        scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(5e-6)
    assert optimizer.param_groups[3]["lr"] == pytest.approx(1e-5)
