from lerobot.optim.schedulers import XVLAStagedPromptWarmupSchedulerConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


def test_xvla_staged_training_preset_normalizes_freeze_flags():
    config = XVLAConfig(
        adaptation_mode="staged_prompt_warmup",
        freeze_vision_encoder=True,
        freeze_language_encoder=True,
        train_policy_transformer=False,
        train_soft_prompts=True,
    )
    assert config.freeze_vision_encoder is False
    assert config.freeze_language_encoder is False
    assert config.train_policy_transformer is True
    assert config.train_soft_prompts is True
    optimizer = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()
    assert optimizer.adaptation_mode == "staged_prompt_warmup"
    assert optimizer.learning_coef == config.learning_coef
    assert isinstance(scheduler, XVLAStagedPromptWarmupSchedulerConfig)
    assert scheduler.freeze_steps == config.freeze_steps
    assert scheduler.num_warmup_steps == config.warmup_steps
