#!/usr/bin/env bash
# Paper-aligned OPSD setup for verl PR #6833; this is not an exact reproduction.
# Current gaps: top-k partial forward KL at KD temperature 1.0, aggregate loss
# clipping, a constant LR schedule, dedicated teacher GPUs, and no 20k teacher
# prompt truncation. See the launch summary printed below.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/run_qwen3_8b_fsdp.sh"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-${MODEL_PATH}}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/openthoughts_math_opsd/train.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/OPSD/qwen3_1p7b_opsd_verl}
LOG_FILE=${LOG_FILE-"${OUTPUT_DIR}/train.log"}

# PR #6833 gives the teacher a separate resource pool. The defaults retain the
# official four-GPU budget by splitting it into 2 actor/rollout + 2 teacher GPUs.
# Use 4 + 4 GPUs to retain the official four actor ranks with this architecture.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-18975}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

ACTOR_LR=${ACTOR_LR:-5e-6}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
USE_CHUNKED_TOPK=${USE_CHUNKED_TOPK:-False}
CHUNKED_TOPK_SIZE=${CHUNKED_TOPK_SIZE:-4096}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.6}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-40960}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-25}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}
PROJECT_NAME=${PROJECT_NAME:-OPSD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1p7b_opsd_verl_pr6833}
LOGGER=${LOGGER:-'["console","wandb"]'}
RESUME_MODE=${RESUME_MODE:-disable}

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Missing training parquet: ${TRAIN_FILE}" >&2
    echo "Set TRAIN_FILE to the train.parquet produced by prepare_verl_opsd.py." >&2
    exit 2
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "Missing base launcher: ${BASE_SCRIPT}" >&2
    exit 2
fi

if [[ -n "${TOTAL_TRAINING_STEPS}" && ! "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_TRAINING_STEPS must be empty or a positive integer." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

export STUDENT_MODEL="${MODEL_PATH}"
export TEACHER_MODEL
export NNODES NGPUS_PER_NODE TEACHER_WORLD_SIZE ROLLOUT_TP TEACHER_TP
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export PPO_MAX_TOKEN_LEN_PER_GPU ACTOR_LR ROLLOUT_GPU_MEM_UTIL TEACHER_GPU_MEM_UTIL
export TOTAL_EPOCHS SAVE_FREQ PROJECT_NAME EXPERIMENT_NAME
export DISTILLATION_LOSS_MODE=forward_kl_topk
export USE_POLICY_GRADIENT=False
export DISTILLATION_TOPK
export TEST_FREQ=-1

OVERRIDES=(
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TRAIN_FILE}']"
    data.val_max_samples=1
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.shuffle=True
    data.seed=42
    data.validation_shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
    actor_rollout_ref.rollout.temperature=1.1
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.weight_decay=0.0
    'actor_rollout_ref.actor.optim.betas=[0.9,0.999]'
    actor_rollout_ref.actor.optim.clip_grad=0.1
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True
    distillation.self_distillation=True
    distillation.privileged_mode=chat_turn
    distillation.privileged_solution_key=extra_info.solution
    distillation.privileged_problem_key=extra_info.problem
    distillation.privileged_enable_thinking=True
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}"
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=0.05
    distillation.distillation_loss.log_prob_min_clamp=null
    +distillation.distillation_loss.use_chunked_topk="${USE_CHUNKED_TOPK}"
    +distillation.distillation_loss.chunked_topk_chunk_size="${CHUNKED_TOPK_SIZE}"
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.teacher_models.teacher_model.inference.max_model_len="${TEACHER_MAX_MODEL_LEN}"
    trainer.logger="${LOGGER}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.resume_mode="${RESUME_MODE}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.log_val_generations=0
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    OVERRIDES+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
    STOP_CONDITION="${TOTAL_TRAINING_STEPS} steps (explicit cap; epoch cap ${TOTAL_EPOCHS})"
else
    STOP_CONDITION="${TOTAL_EPOCHS} epochs"
fi

cat <<EOF
OPSD launch mapping
  model:             ${MODEL_PATH}
  data:              ${TRAIN_FILE}
  actor GPUs/node:   ${NGPUS_PER_NODE}
  teacher GPUs/node: ${TEACHER_WORLD_SIZE}
  global batch:      ${TRAIN_BATCH_SIZE}
  stop condition:    ${STOP_CONDITION}
  output:            ${OUTPUT_DIR}

This matches the published OPSD hyperparameters where current verl supports
them. It is not loss- or topology-equivalent to the reference implementation.
EOF

run_training() {
    bash "${BASE_SCRIPT}" "${OVERRIDES[@]}" "$@"
}

if [[ -n "${LOG_FILE}" ]]; then
    mkdir -p "$(dirname -- "${LOG_FILE}")"
    set +e
    run_training "$@" 2>&1 | tee -a "${LOG_FILE}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[0] != 0 )); then
        exit "${pipeline_status[0]}"
    fi
    exit "${pipeline_status[1]}"
fi

run_training "$@"
