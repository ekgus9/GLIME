import numpy as np
import torch
import gc
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

from .glime_hparams import GLIMEHyperParams
from ...util.generate import generate_fast


RANDOM_SEED = 7

GLOBAL_GRAD_BASIS = {}
CONTEXT_TEMPLATES_CACHE = None
GLOBAL_REPLAY_BUFFER = deque()


def tokenize(tok, prompts, targets, device):
    assert len(prompts) == len(targets)

    full_texts = [p + t for p, t in zip(prompts, targets)]
    enc = tok(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    prompt_lens = [len(tok(p, add_special_tokens=True).input_ids) for p in prompts]

    labels = enc["input_ids"].clone()
    if tok.pad_token_id is not None:
        labels[labels == tok.pad_token_id] = -100

    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100

    return enc, labels


def _build_prefix_messages(chosen_msgs):
    if isinstance(chosen_msgs, list) and len(chosen_msgs) > 0:
        if isinstance(chosen_msgs[-1], dict) and chosen_msgs[-1].get("role") == "assistant":
            return chosen_msgs[:-1]
    return chosen_msgs


def tokenize_chat(tok, msgs, device, max_length=8192):
    prefix_msgs = _build_prefix_messages(msgs)

    prefix_txt = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
    full_txt   = tok.apply_chat_template(msgs,        tokenize=False, add_generation_prompt=False)

    enc_prefix = tok(prefix_txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    enc_full   = tok(full_txt,   return_tensors="pt", truncation=True, max_length=max_length).to(device)

    prefix_len = enc_prefix["input_ids"].shape[1]

    labels = enc_full["input_ids"].clone()
    labels[:, :prefix_len] = -100
    if tok.pad_token_id is not None:
        labels[labels == tok.pad_token_id] = -100

    return enc_full, labels
    

def add_to_replay_buffer(new_requests: List[Dict], max_size: int = 128) -> None:
    if max_size <= 0:
        return
    for r in new_requests:
        GLOBAL_REPLAY_BUFFER.append(deepcopy(r))
    while len(GLOBAL_REPLAY_BUFFER) > max_size:
        GLOBAL_REPLAY_BUFFER.popleft()


def sample_from_replay_buffer(k: int, rng: Optional[np.random.Generator] = None) -> List[Dict]:
    if k <= 0 or len(GLOBAL_REPLAY_BUFFER) == 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    buf_list = list(GLOBAL_REPLAY_BUFFER)
    n = min(k, len(buf_list))
    idx = rng.choice(len(buf_list), size=n, replace=False)
    return [deepcopy(buf_list[i]) for i in idx]


def compress_basis_qr(basis: torch.Tensor, max_rank: int = 256) -> torch.Tensor:
    if basis is None:
        return basis
    if basis.ndim != 2 or basis.shape[1] <= 1:
        return basis
    Q, _ = torch.linalg.qr(basis, mode="reduced")  # [in_dim, r]
    return Q[:, : min(Q.shape[1], max_rank)]


def grad_to_right_basis_sketch_qr(
    grad_2d: torch.Tensor,
    k: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    assert grad_2d.ndim == 2
    out_dim, in_dim = grad_2d.shape

    k = max(1, min(k, in_dim))
    if generator is None:
        R = torch.randn(out_dim, k, device=grad_2d.device, dtype=grad_2d.dtype)
    else:
        R = torch.randn(out_dim, k, device=grad_2d.device, dtype=grad_2d.dtype, generator=generator)

    Y = grad_2d.transpose(0, 1) @ R   # [in_dim, k]
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ][0]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def get_logps(logits, labels, ignore_index=-100):
    logps = F.log_softmax(logits, dim=-1)

    shift_logits = logps[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    mask = (shift_labels != ignore_index).float()

    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == ignore_index] = 0

    per_token_logps = torch.gather(shift_logits, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * mask).sum() / (mask.sum() + 1e-8)
    

def apply_glime_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Any, 
    copy=False,
    return_orig_weights=True,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    device = torch.device(f'cuda:{hparams.device}')
    model.config.use_cache = False
    model.eval()

    epoch = getattr(hparams, "epoch", 3)

    replay_max_size = getattr(hparams, "replay_max_size", 1000)
    replay_batch = getattr(hparams, "replay_batch", 3)
    replay_loss_weight = getattr(hparams, "replay_loss_weight", 0.1)
    sft_loss_weight = getattr(hparams, "sft_loss_weight", 1.0)
    dpo_loss_weight = getattr(hparams, "dpo_loss_weight", 1.0)

    basis_k = getattr(hparams, "basis_k", 128)
    basis_max_rank = getattr(hparams, "basis_max_rank", 1024)

    rng = np.random.default_rng(RANDOM_SEED)

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    context_templates = get_context_templates(model, tok)

    target_mlp_prefixes = [hparams.mlp_module_tmp.format(l) for l in hparams.layers]
    target_layer_names = []
    params_to_optimize = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix + ".") for prefix in target_mlp_prefixes):
            if return_orig_weights:
                weights_copy[name] = param.detach().clone()
            param.requires_grad = True
            params_to_optimize.append(param)
            target_layer_names.append(name)
        else:
            param.requires_grad = False

    def compute_edit_loss(req: Dict) -> torch.Tensor:
        base_prompt = req["prompt"]
        prompts = [ctx.format(base_prompt) for ctx in context_templates]
        target_new = " " + req["target_new"].strip()

        enc_c, labels_c = tokenize(
            tok,
            prompts=prompts,
            targets=[target_new] * len(prompts),
            device=device
        )
        out_c = model(**enc_c, labels=labels_c)
        loss_sft = out_c.loss
        return loss_sft
    
    def get_hook(name):
        def hook(grad):
            if name in GLOBAL_GRAD_BASIS and GLOBAL_GRAD_BASIS[name] is not None:
                    B = GLOBAL_GRAD_BASIS[name].to(grad.device).to(grad.dtype)
                    grad = grad - torch.mm(torch.mm(grad, B), B.t())
            return grad
        return hook

    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hooks.append(param.register_hook(get_hook(name)))

    optimizer = AdamW(params_to_optimize, lr=hparams.lr if hasattr(hparams, 'lr') else 5e-5, weight_decay=0)

    model.gradient_checkpointing_enable()
    
    for epoch in tqdm(range(epoch), desc="Epochs"):
        for request in requests:
            replay_samples = sample_from_replay_buffer(replay_batch, rng=rng)

            total_loss_rep = 0.0
            for r_req in replay_samples:
                loss_rep = compute_edit_loss(r_req)
                total_loss_rep += loss_rep

            base_prompt = request["prompt"]

            prompt = [ctx.format(base_prompt) for ctx in context_templates]
            target_new = " " + request["target_new"].strip()
            enc_c, labels_c = tokenize(
                tok, prompts=prompt, targets=[target_new]*len(prompt), device=device
            )
            out_c = model(**enc_c, labels=labels_c)
            loss_sft = out_c.loss

            with torch.no_grad():
                enc_dr, labels_dr = tokenize_chat(tok, request["dpo"]["rejected"], device)
                out_dr = model(**enc_dr)
                logps_dr = get_logps(out_dr.logits, labels_dr)
            enc_dc, labels_dc = tokenize_chat(tok, request["dpo"]["chosen"], device)
            out_dc = model(**enc_dc)
            logps_dc = get_logps(out_dc.logits, labels_dc)

            loss_dpo = -F.logsigmoid((logps_dc - logps_dr))
            loss = sft_loss_weight * loss_sft + replay_loss_weight * total_loss_rep + dpo_loss_weight * loss_dpo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if epoch == 0: 
                add_to_replay_buffer([request], max_size=replay_max_size)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(RANDOM_SEED))

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not (param.requires_grad and param.grad is not None):
                continue

            g = param.grad.detach()
            if g.ndim != 2:
                continue
            g2d = g.float()

            new_basis = grad_to_right_basis_sketch_qr(g2d, k=basis_k, generator=gen).cuda()

            if name not in GLOBAL_GRAD_BASIS or GLOBAL_GRAD_BASIS[name] is None:
                GLOBAL_GRAD_BASIS[name] = compress_basis_qr(new_basis, max_rank=basis_max_rank)
            else:
                old = GLOBAL_GRAD_BASIS[name]
                combined = torch.cat([old, new_basis], dim=1)          # [in_dim, r_old + basis_k]
                GLOBAL_GRAD_BASIS[name] = compress_basis_qr(combined, max_rank=basis_max_rank)

    for h in hooks:
        h.remove()

    for param in model.parameters():
        param.requires_grad = True

    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return model, weights_copy
