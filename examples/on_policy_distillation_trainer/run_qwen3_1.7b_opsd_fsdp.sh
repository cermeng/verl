#!/usr/bin/env bash
# Privileged-context OPSD | text | vLLM teacher/rollout | eager FSDP
#
# This maps the OPSD repository's optional teacher-top-k objective to verl.
# It is not the paper's full-vocabulary main objective: both distributions are
# renormalized on the teacher's top-k support before generalized JSD is applied.

set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}
# Fixed-teacher OPSD requires the same initial checkpoint as the student.
TEACHER_MODEL=${TEACHER_MODEL:-$STUDENT_MODEL}
OPSD_DATA_DIR=${OPSD_DATA_DIR:-$HOME/data/opsd}

NNODES=${NNODES:-1}
# Actor and teacher use distinct pools; defaults consume 4 + 4 GPUs per node.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
# The reference collator allows up to 20K privileged-context tokens.
max_teacher_prompt_length=${MAX_TEACHER_PROMPT_LENGTH:-20000}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

temperature=${TEMPERATURE:-1.1}
distillation_topk=${DISTILLATION_TOPK:-128}
opsd_beta=${OPSD_BETA:-0.0}
opsd_jsd_token_clip=${OPSD_JSD_TOKEN_CLIP:-0.05}
actor_lr=${ACTOR_LR:-5e-6}
lora_rank=${LORA_RANK:-64}
lora_alpha=${LORA_ALPHA:-128}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-20}
project_name=${PROJECT_NAME:-verl_opsd_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_1.7b_fixed_teacher_opsd_topk}

train_file="$OPSD_DATA_DIR/train.parquet"
validation_file="$OPSD_DATA_DIR/validation.parquet"
student_max_model_len=$(( max_prompt_length + max_response_length + 1 ))
teacher_max_model_len=$(( max_teacher_prompt_length + max_response_length + 1 ))

DATA=(
    data.train_files="['$train_file']"
    data.val_files="['$validation_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.continuous_token.enable=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    # Match the reference fixed-teacher setup: the student trains fresh LoRA
    # adapters while the separate teacher stays at the base checkpoint.
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.max_model_len=${student_max_model_len}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.opsd.enabled=True
    distillation.opsd.teacher_prompt_key=opsd_teacher_prompt
    distillation.opsd.max_teacher_prompt_length=${max_teacher_prompt_length}
    +distillation.opsd.teacher_apply_chat_template_kwargs.enable_thinking=True
    distillation.opsd.beta=${opsd_beta}
    distillation.opsd.temperature=${temperature}
    distillation.opsd.jsd_token_clip=${opsd_jsd_token_clip}
    distillation.opsd.require_same_model=True
    distillation.distillation_loss.loss_mode=opsd_gjsd_topk
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
)

TRAINER=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${TRAINER[@]}" \
    "$@"
