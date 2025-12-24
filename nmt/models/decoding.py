# nmt/decoding.py
from typing import List, Tuple
import torch
import torch.nn.functional as F

from .rnn_nmt import RNNNMT, HiddenState


@torch.no_grad()
def greedy_decode(
    model: RNNNMT,
    src_ids: torch.Tensor,      # [B,S]
    src_lens: torch.Tensor,     # [B]
    max_len: int = 256
) -> torch.Tensor:
    model.eval()
    enc_out, state, enc_mask = model.encode(src_ids, src_lens)

    B = src_ids.size(0)
    cur = torch.full((B,), model.bos_id, dtype=torch.long, device=src_ids.device)
    outputs = []

    finished = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

    for _ in range(max_len):
        logits, state, _ = model.decoder.forward_step(cur, state, enc_out, enc_mask)
        next_tok = torch.argmax(logits, dim=-1)  # [B]
        outputs.append(next_tok)
        cur = next_tok
        finished |= (next_tok == model.eos_id)
        if finished.all():
            break

    return torch.stack(outputs, dim=1)  # [B, T]


@torch.no_grad()
def beam_search_decode_one(
    model: RNNNMT,
    src_ids_1: torch.Tensor,     # [1,S]
    src_lens_1: torch.Tensor,    # [1]
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 0.0
) -> List[int]:
    """
    Beam search for a single sample. Returns best token list (excluding BOS).
    """
    model.eval()
    enc_out, state, enc_mask = model.encode(src_ids_1, src_lens_1)

    device = src_ids_1.device

    # Hypothesis: (tokens, score, state, finished)
    hyps: List[Tuple[List[int], float, HiddenState, bool]] = [
        ([], 0.0, state, False)
    ]

    for t in range(max_len):
        new_hyps: List[Tuple[List[int], float, HiddenState, bool]] = []
        for tokens, score, st, fin in hyps:
            if fin:
                new_hyps.append((tokens, score, st, fin))
                continue

            prev = torch.tensor([model.bos_id if t == 0 else tokens[-1]], device=device, dtype=torch.long)
            logits, st2, _ = model.decoder.forward_step(prev, st, enc_out, enc_mask)
            logp = F.log_softmax(logits, dim=-1).squeeze(0)  # [V]

            topk_logp, topk_ids = torch.topk(logp, k=beam_size)
            for lp, wid in zip(topk_logp.tolist(), topk_ids.tolist()):
                tok_list = tokens + [wid]
                fin2 = (wid == model.eos_id)
                new_hyps.append((tok_list, score + lp, st2, fin2))

        # prune
        def norm_score(s: float, length: int) -> float:
            if length_penalty <= 0:
                return s
            return s / ((length + 1) ** length_penalty)

        new_hyps.sort(key=lambda x: norm_score(x[1], len(x[0])), reverse=True)
        hyps = new_hyps[:beam_size]

        if all(h[3] for h in hyps):
            break

    # pick best
    best = max(hyps, key=lambda x: x[1] / ((len(x[0]) + 1) ** length_penalty) if length_penalty > 0 else x[1])
    return best[0]
