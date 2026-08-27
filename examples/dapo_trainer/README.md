# Qwen3-Omni Thinker DAPO Trainer

Last updated: 08/27/2026

This example provides the first Qwen3-Omni Thinker DAPO milestone on the V1
omni trainer: GPU LoRA training on GSM8K with clip-higher, token-level policy
gradient, GRPO advantages, and the registered DAPO reward manager.

The Phase 1 launcher intentionally disables dynamic sampling:

```text
algorithm.filter_groups.enable=false
```

It also does not enable the overlong reward buffer. Those components are kept
out of this baseline so the token-level DAPO policy path can be validated
independently. Do not use the reward-manager name alone to identify an
algorithm: `reward.reward_manager.name=dapo` with `policy_loss.loss_mode=gspo`
still runs GSPO and now emits a configuration warning.

## Run

Prepare GSM8K parquet files under `~/data/gsm8k`, then launch from the
repository root:

```bash
bash examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh
```

The default model is `~/models/Qwen/Qwen3-Omni-30B-A3B-Instruct`. Override the
model, data, or any Hydra setting without editing the script:

```bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
bash examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh \
    trainer.total_training_steps=2
```

Only the Thinker LoRA adapters are trained. Talker, code2wav, code predictor,
visual projection, and audio-tower modules are excluded, and the vision tower
is frozen, matching the existing GSPO V1 baseline.

The corresponding tiny-random two-step smoke is:

```bash
bash tests/special_e2e/run_dapo_qwen3_omni_thinker_lora_v1_smoke.sh
```
