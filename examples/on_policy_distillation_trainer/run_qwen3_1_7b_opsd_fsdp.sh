#!/usr/bin/env bash
# Qwen3-1.7B OPSD on 4 GPUs: 2 actor/rollout + 2 frozen-teacher GPUs.
# This matches the reference hyperparameters where PR #6833 supports them;
# the local loss remains top-k forward KL rather than exact full-vocabulary OPSD.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-/host_home/models/Qwen3-1.7B}
TRAIN_FILE=${TRAIN_FILE:-/host_home/opd-data/train/verl-math-30k-opsd/train.parquet}

ACTOR_GPUS=${ACTOR_GPUS:-2}
TEACHER_GPUS=${TEACHER_GPUS:-2}

PROJECT_NAME=${PROJECT_NAME:-OPSD}
MODEL_TAG=qwen3_1p7b
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/${PROJECT_NAME}/${MODEL_TAG}}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_TAG}_opsd_${RUN_TAG}}

LOGGER=${LOGGER:-'["console","tensorboard","file"]'}

LOG_ROOT=${LOG_ROOT:-${OUTPUT_DIR}/logs}
RUN_LOG_DIR=${RUN_LOG_DIR:-${LOG_ROOT}/${EXPERIMENT_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}}

LOG_FILE=${LOG_FILE:-${RUN_LOG_DIR}/train.log}
FILE_METRICS_PATH=${FILE_METRICS_PATH:-${RUN_LOG_DIR}/metrics.jsonl}
TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR:-${RUN_LOG_DIR}/tensorboard}

# verl file/tensorboard env
export TENSORBOARD_DIR="${TENSORBOARD_LOG_DIR}"
export VERL_FILE_LOGGER_PATH="${FILE_METRICS_PATH}"

[[ -f "${TRAIN_FILE}" ]] || { echo "Missing training parquet: ${TRAIN_FILE}" >&2; exit 2; }
mkdir -p \
    "${OUTPUT_DIR}" \
    "${RUN_LOG_DIR}" \
    "${TENSORBOARD_LOG_DIR}" \
    "${CHECKPOINT_DIR}" \
    "$(dirname -- "${LOG_FILE}")" \
    "$(dirname -- "${FILE_METRICS_PATH}")"

# The remaining ppo_* names are legacy batch controls; no PPO loss is enabled.
run_training() {
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TRAIN_FILE}" \
        data.val_max_samples=1 \
        data.train_batch_size=32 \
        data.max_prompt_length=18975 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        data.seed=42 \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.lora_rank=64 \
        actor_rollout_ref.model.lora_alpha=128 \
        'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.actor.optim.weight_decay=0 \
        actor_rollout_ref.actor.optim.clip_grad=0.1 \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_model_len=20000 \
        actor_rollout_ref.rollout.temperature=1.1 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        distillation.enabled=True \
        distillation.n_gpus_per_node="${TEACHER_GPUS}" \
        distillation.nnodes=1 \
        distillation.teacher_models.teacher_model.model_path="${MODEL_PATH}" \
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
        distillation.teacher_models.teacher_model.inference.temperature=1.0 \
        distillation.teacher_models.teacher_model.inference.max_model_len=40960 \
        distillation.self_distillation=True \
        distillation.privileged_mode=chat_turn \
        distillation.privileged_solution_key=extra_info.solution \
        distillation.privileged_problem_key=extra_info.problem \
        distillation.privileged_enable_thinking=True \
        distillation.distillation_loss.loss_mode=forward_kl_topk \
        distillation.distillation_loss.topk=64 \
        distillation.distillation_loss.use_task_rewards=False \
        distillation.distillation_loss.use_policy_gradient=False \
        distillation.distillation_loss.loss_max_clamp=0.05 \
        distillation.distillation_loss.log_prob_min_clamp=null \
        trainer.n_gpus_per_node="${ACTOR_GPUS}" \
        trainer.nnodes=1 \
        trainer.logger="${LOGGER}" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.total_epochs=1 \
        trainer.total_training_steps=100 \
        trainer.save_freq=25 \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.resume_mode=disable \
        trainer.max_actor_ckpt_to_keep=5 \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        "$@"
}

# Append Hydra overrides after the script, for example:
#   bash run_qwen3_1_7b_opsd_fsdp.sh trainer.save_freq=50
set +e
run_training "$@" 2>&1 | tee -a "${LOG_FILE}"
status=("${PIPESTATUS[@]}")
set -e
(( status[0] != 0 )) && exit "${status[0]}"
exit "${status[1]}"
