# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_1.7b_opsd_fsdp.sh`   | self     | text       | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`

## Privileged-context OPSD

`run_qwen3_1.7b_opsd_fsdp.sh` ports the privileged-context self-distillation
data flow from [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD): the
student sees only the problem, while a frozen copy of the same initial model
scores the student's response after also reading a verified solution. The
provided Qwen3-1.7B launch defaults to the reference setup's fresh rank-64 LoRA
student and a frozen base-checkpoint teacher.

Prepare the two-prompt parquet schema first:

```bash
python3 examples/data_preprocess/opsd.py --local_save_dir "$HOME/data/opsd"
```

Then launch on a GPU cluster:

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_1.7b_opsd_fsdp.sh
```

The current verl implementation maps the reference trainer's optional
teacher-top-k objective: student and teacher are separately renormalized on
the teacher top-k support, followed by generalized JSD and vocabulary-entry
pointwise clipping. It does **not** claim equivalence to the paper's main
full-vocabulary runs. See [`docs/algo/opd.md`](../../docs/algo/opd.md) for the
exact objective, alignment rules, and unsupported combinations.
