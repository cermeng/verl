# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def generalized_jsd_pointwise(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float,
    token_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute OPSD generalized-JSD contributions on a shared support.

    ``student_log_probs`` and ``teacher_log_probs`` must each be normalized on
    their last dimension.  The beta endpoints intentionally use KL rather than
    the degenerate endpoint of the mixture expression, matching the OPSD
    objective.  Clipping is applied to each vocabulary-entry contribution
    *before* summation; negative contributions are preserved.

    Returns the clipped per-position loss, the unclipped per-position loss, and
    the fraction of vocabulary entries clipped at each position.
    """
    student_log_probs = student_log_probs.float()
    teacher_log_probs = teacher_log_probs.detach().float()
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    if beta == 0.0:
        contributions = teacher_probs * (teacher_log_probs - student_log_probs)
    elif beta == 1.0:
        contributions = student_probs * (student_log_probs - teacher_log_probs)
    else:
        log_beta = torch.log(student_log_probs.new_tensor(beta))
        log_one_minus_beta = torch.log(student_log_probs.new_tensor(1.0 - beta))
        mixture_log_probs = torch.logaddexp(
            student_log_probs + log_one_minus_beta,
            teacher_log_probs + log_beta,
        )
        contributions = beta * teacher_probs * (teacher_log_probs - mixture_log_probs) + (
            1.0 - beta
        ) * student_probs * (student_log_probs - mixture_log_probs)

    unclipped_losses = contributions.sum(dim=-1)
    if token_clip is None:
        clipped_fraction = torch.zeros_like(unclipped_losses)
        clipped_contributions = contributions
    else:
        clipped_mask = contributions > token_clip
        clipped_fraction = clipped_mask.float().mean(dim=-1)
        clipped_contributions = contributions.clamp(max=token_clip)
    return clipped_contributions.sum(dim=-1), unclipped_losses, clipped_fraction


def compute_opsd_gjsd_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute the teacher-top-k approximation of the OPSD generalized JSD.

    The teacher chooses the support.  Student and teacher distributions are
    separately renormalized on that same support before the divergence is
    evaluated, so logits outside the support receive no OPSD gradient.  This is
    distinct from ``compute_forward_kl_topk``, which uses truncated full-vocab
    probabilities without renormalization.

    ``student_logits`` have already been divided by rollout temperature by the
    actor engine.  Teacher prompt log-probabilities are returned at temperature
    one, so they are divided by ``config.opsd.temperature`` before top-k
    renormalization.
    """
    del data_format  # FSDP handles THD/BSHD identically at this stage.
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).long()

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # Selecting logits before log_softmax is important: OPSD's optional top-k
    # objective renormalizes on the teacher support, unlike truncated FKL.
    selected_student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)
    student_subset_log_probs = F.log_softmax(selected_student_logits.float(), dim=-1)

    opsd_config = config.opsd
    teacher_subset_log_probs = F.log_softmax(
        teacher_topk_log_probs.detach().float() / opsd_config.temperature,
        dim=-1,
    )
    distillation_losses, unclipped_losses, clipped_fraction = generalized_jsd_pointwise(
        student_log_probs=student_subset_log_probs,
        teacher_log_probs=teacher_subset_log_probs,
        beta=opsd_config.beta,
        token_clip=opsd_config.jsd_token_clip,
    )

    # These masses are diagnostics only and intentionally use the distributions
    # available from the servers before top-k renormalization.
    student_log_z = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    selected_student_full_log_probs = selected_student_logits.float() - student_log_z
    student_mass = selected_student_full_log_probs.exp().sum(dim=-1).detach()
    teacher_mass = teacher_topk_log_probs.detach().float().exp().sum(dim=-1)

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "opsd_unclipped_losses": unclipped_losses.detach(),
        "opsd_clipped_fraction": clipped_fraction.detach(),
    }


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
