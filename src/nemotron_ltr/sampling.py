"Rejection sampling with retained categorical proposal distributions."

from __future__ import annotations


def probs(logits):
    return torch.softmax(logits.float(), dim=-1)


def draw(p):
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])


def mode(model, diffusion):
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = diffusion


import copy
import torch
from transformers import DynamicCache
from .runtime_attention import (
    extract_quadratic_roots,
    install_runtime_attention,
    packed_position_ids,
    packed_visibility,
    quadratic_position_ids,
    quadratic_visibility,
    runtime_attention,
)


def reject(p, q, candidates):
    """p has k+1 rows (including bonus); q/candidates have exactly k."""
    assert p.shape[1] == q.shape[1] + 1 == candidates.shape[1] + 1
    numerator = p[:, :-1].gather(-1, candidates[..., None]).squeeze(-1)
    denominator = q.gather(-1, candidates[..., None]).squeeze(-1)
    accepted = torch.rand_like(numerator) < (numerator / denominator).clamp(max=1)
    n = int(accepted.long().cumprod(-1).sum().item())
    if n == candidates.shape[1]:
        extra = draw(p[:, -1:])
    else:
        residual = (p[:, n : n + 1] - q[:, n : n + 1]).clamp(min=0)
        extra = draw(residual / residual.sum(-1, keepdim=True))
    return (torch.cat((candidates[:, :n], extra), dim=1), n)


@torch.inference_mode()
def decode(model, prompt, eos, method, width, limit, *, audit=False, rejection_fn=None):
    if method not in {"linear", "packed", "quadratic"}:
        raise ValueError(method)
    if (
        prompt.ndim != 2
        or prompt.shape[0] != 1
        or prompt.shape[1] == 0
        or width < 2
        or (limit < 1)
    ):
        raise ValueError("Requires batch one, width >= 2 and positive token limit")
    rejection_fn = reject if rejection_fn is None else rejection_fn
    install_runtime_attention(model)
    mode(model, False)
    out = model.encoder(
        input_ids=prompt,
        past_key_values=DynamicCache(),
        use_cache=True,
        use_causal_mask=True,
    )
    cache = out.past_key_values
    seed = draw(probs(model.diffusion_head(out.last_hidden_state[:, -1:])))
    emitted = [seed]
    (count, nfe, rounds, accepted_total, offered) = (1, 1, 0, 0, 0)
    ended = int(seed.item()) in eos
    q = None
    checks = []
    row_counts = []
    while count < limit and (not ended):
        c = int(cache.get_seq_length())
        assert c == prompt.shape[1] + count - 1
        block = torch.full(
            (1, width),
            model.config.mask_token_id,
            dtype=torch.long,
            device=prompt.device,
        )
        block[:, :1] = seed
        if method == "linear" or q is None:
            mode(model, True)
            proposal = model.encoder(
                input_ids=block, past_key_values=cache, use_cache=False
            )
            q = probs(model.diffusion_head(proposal.last_hidden_state[:, 1:]))
            candidates = draw(q)
            nfe += 1
            row_counts.append(width)
        k = candidates.shape[1]
        block[:, 1 : k + 1] = candidates
        mode(model, False)
        if audit:
            reference_cache = copy.deepcopy(cache)
            native = model.encoder(
                input_ids=block,
                past_key_values=reference_cache,
                use_cache=True,
                use_causal_mask=True,
            )
            native_p = probs(model.diffusion_head(native.last_hidden_state[:, : k + 1]))
        if method == "linear":
            out = model.encoder(
                input_ids=block,
                past_key_values=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            target_hidden = out.last_hidden_state[:, : k + 1]
            row_counts.append(width)
        else:
            if method == "packed":
                clean = torch.full_like(block, model.config.mask_token_id)
                clean[:, :1] = seed
                query = torch.cat((block, clean), dim=1)
                positions = packed_position_ids(c, width, prompt.device)
                mask = packed_visibility(c, width)
            else:
                masks = torch.full(
                    (1, width, width),
                    model.config.mask_token_id,
                    dtype=torch.long,
                    device=prompt.device,
                )
                query = torch.cat((block[..., None], masks), dim=2).reshape(1, -1)
                positions = quadratic_position_ids(c, width, prompt.device)
                mask = quadratic_visibility(c, width)
            with runtime_attention(model, mask.to(prompt.device)):
                out = model.encoder(
                    input_ids=query,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
            target_hidden = (
                out.last_hidden_state[:, : k + 1]
                if method == "packed"
                else out.last_hidden_state[:, :: width + 1][:, : k + 1]
            )
            row_counts.append(query.shape[1])
        cache = out.past_key_values
        p = probs(model.diffusion_head(target_hidden))
        if audit:
            reference_ids = torch.cat((prompt, *emitted, candidates), dim=1)
            ref = model.encoder(
                input_ids=reference_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                use_causal_mask=True,
            )
            ref_p = probs(model.diffusion_head(ref.last_hidden_state[:, -(k + 1) :]))
            tv = float((p - native_p).abs().sum(-1).mul(0.5).max().item())
            full_tv = float((p - ref_p).abs().sum(-1).mul(0.5).max().item())
            control_tv = float((native_p - ref_p).abs().sum(-1).mul(0.5).max().item())
            checks.append(
                dict(
                    round=rounds,
                    target_vs_same_cache_native_tv=tv,
                    target_vs_full_recompute_tv=full_tv,
                    native_cache_vs_full_recompute_tv=control_tv,
                )
            )
            if tv > 0.001 or full_tv > 0.001:
                raise AssertionError(
                    f"Fused target differs from native control: TV={tv}; full={full_tv}; control={control_tv}"
                )
        (segment, n) = rejection_fn(p, q, candidates)
        advance = n + 1
        if method == "quadratic":
            extract_quadratic_roots(cache, c, width)
        cache.crop(c + advance)
        nfe += 1
        rounds += 1
        accepted_total += n
        offered += k
        if method == "packed":
            h = out.last_hidden_state[:, width + advance + 1 :]
            q = probs(model.diffusion_head(h)) if h.shape[1] else p[:, :0]
        elif method == "quadratic":
            start = n * (width + 1) + 2
            q = probs(
                model.diffusion_head(
                    out.last_hidden_state[:, start : start + width - 1]
                )
            )
        if audit and method != "linear" and q.shape[1]:
            clean = torch.full_like(block, model.config.mask_token_id)
            if method == "packed":
                reference_cache.crop(c)
                clean[:, :1] = block[:, :1]
            else:
                reference_cache.crop(c + advance)
            mode(model, True)
            native_q_out = model.encoder(
                input_ids=clean, past_key_values=reference_cache, use_cache=False
            )
            mode(model, False)
            selected = (
                native_q_out.last_hidden_state[:, advance + 1 :]
                if method == "packed"
                else native_q_out.last_hidden_state[:, 1:]
            )
            native_q = probs(model.diffusion_head(selected))
            q_tv = float((q - native_q).abs().sum(-1).mul(0.5).max().item())
            checks[-1]["proposal_vs_native_tv"] = q_tv
            if q_tv > 0.001:
                raise AssertionError(
                    f"Selected proposal differs from native reference: TV={q_tv}"
                )
        if method != "linear":
            candidates = draw(q) if q.shape[1] else block[:, :0]
        segment = segment[:, : limit - count]
        for i, token in enumerate(segment[0].tolist()):
            if token in eos:
                segment = segment[:, : i + 1]
                ended = True
                break
        emitted.append(segment)
        count += segment.shape[1]
        seed = segment[:, -1:]
    ids = torch.cat(emitted, dim=1)[0].tolist()
    return dict(
        output_ids=ids,
        generated_tokens=len(ids),
        nfe=nfe,
        rounds=rounds,
        accepted_candidates=accepted_total,
        offered_candidates=offered,
        stopped_on_eos=ended,
        decode_rows_per_forward=row_counts,
        audit=checks,
    )
