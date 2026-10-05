#!/usr/bin/env python3
"""
run_gpu.py: GPU experiments for the workshop papers (Isambard-AI or termitech).

Replaces the Colab-only notebook cells (Drive mount, repo sync) with plain
commands. Every prediction goes through the same decoding path as
`model.predict_action_dist`, and is logged with the original readout (`c*`,
identical to the dissertation's log) plus a corrected one:

  * `cf*`: expected bin centre with token 31744 included. OpenVLA's decoder
    clips that token to bin 254, but the original slice dropped it.
  * `m*`:  probability mass on all 256 action tokens, per dimension. The
    original log kept only the minimum across dimensions, which the gripper
    dominates.

Subcommands
  smoke    load the model, print the action space, run the readout gate
  replay   re-predict logged stimuli (cross-machine reproducibility; precision)
  ladder   instruction ladder: positive controls and a paraphrase noise floor
  natural  antonym swap on the unedited validation frames

Shard a run over GPUs with --shard i --nshards n; each shard writes its own
CSV (`<out>.shard<i>.csv`) and skips work already logged, so runs resume.

Options added for the BtP experiments (docs/btp_experiments_plan.md):
  --save-dist  append each prediction's 7x256 action-token probabilities (float16,
               unnormalised, i.e. as a share of the full vocabulary) to
               `<csv>.dist.f16`; the CSV column `dist_row` indexes the record
  --placebo    ladder: add grab/take pairs that name the same twin for the
               prenominal, object and table-side wordings (the original wording's
               pair is the existing `paraphrase` condition)
  --lowercase  natural: lower-case every instruction, as OpenVLA's training does

    source scripts/isambard/env.sh      # or scripts/termitech/env.sh
    python scripts/run_gpu.py smoke
    python scripts/run_gpu.py replay --precision nf4 --out $SGVLA_ROOT/runs/replay_nf4.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

# Machine-specific paths come from scripts/<machine>/env.sh (SGVLA_ROOT, HF_HOME,
# and on termitech HF_ENDPOINT for the Hugging Face mirror).
ROOT = os.environ.get("SGVLA_ROOT", "/data/sgx/SGVLA")
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATA = os.environ.get("SGVLA_DATA", f"{ROOT}/v2")

# Extra columns carried from the original probe log onto replayed rows.
CARRY = ("scene_id", "pair_id", "base_scene_id", "role", "frame", "scene_source",
         "condition", "image_transform", "image_scene_id", "configuration",
         "expected_sign_image", "target_sign_a_image", "target_sign_b_image",
         "spatial_term", "axis", "axis_index", "category", "feasible_both",
         "duplicate_target", "sample_idx")


# --------------------------------------------------------------------------- #
# Model and readout
# --------------------------------------------------------------------------- #
PRECISIONS = ("nf4", "bf16", "fp4_official", "int8", "fp16", "fp32")


def load(precision: str, attn: str = "eager"):
    import torch
    from model import load_openvla, run_metadata

    if precision in ("nf4", "bf16") and attn == "eager":
        processor, vla, dtype = load_openvla(quantize_4bit=(precision == "nf4"), precision="bf16")
    else:
        processor, vla, dtype = load_custom(precision, attn)
    meta = run_metadata(dtype)
    meta["precision"] = precision
    meta["attn"] = attn
    torch.backends.cuda.matmul.allow_tf32 = False
    if IMAGE_ROOT:
        meta["image_root"] = IMAGE_ROOT
    return processor, vla, dtype, meta


def load_custom(precision: str, attn: str):
    """BtP plan B4: precisions and attention kernels beyond the reference settings.

    fp4_official reproduces OpenVLA's own evaluation path, `load_in_4bit=True` with bitsandbytes defaults
    (FP4, no double quantisation); int8 is LLM.int8(); fp16/fp32/bf16 load unquantised.
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from model import MODEL_ID

    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)
    kw = dict(attn_implementation=attn, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    if precision == "fp4_official":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    elif precision == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kw)
    if "quantization_config" not in kw:
        vla = vla.to("cuda:0")
    vla.eval()
    print(f"[load_custom] precision={precision} attn={attn} dtype={dtype}", flush=True)
    return processor, vla, dtype


def predict(processor, vla, image, instruction, dtype, unnorm_key="bridge_orig"):
    """`model.predict_action_dist` plus the corrected readout from the same logits."""
    import torch
    from action_bins import EMPTY_TOKEN_ID, denormalise_action, readout_from_logits
    from model import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image.convert("RGB")).to("cuda:0", dtype=dtype)
    inputs.pop("attention_mask", None)
    input_ids = inputs.pop("input_ids")
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        pad = torch.tensor([[EMPTY_TOKEN_ID]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad), dim=1)
    action_dim = vla.get_action_dim(unnorm_key)
    with torch.inference_mode():
        out = vla.generate(input_ids, max_new_tokens=action_dim, do_sample=False,
                           output_scores=True, return_dict_in_generate=True, **inputs)
    tokens = out.sequences[0, -action_dim:].cpu().numpy()
    logits = torch.stack(out.scores, dim=0)[:, 0, :].float().cpu().numpy()
    stats = vla.get_action_stats(unnorm_key)
    centers = np.asarray(vla.bin_centers, dtype=np.float64)
    vocab = int(vla.vocab_size)
    readout = readout_from_logits(tokens, logits, centers, vocab, stats)

    # Corrected readout: all 256 action tokens, 31744 folded into bin 254.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    ids = np.arange(vocab - 256, vocab)                       # 31744 ... 31999
    bins = np.clip(vocab - ids - 1, 0, len(centers) - 1)      # upstream decode
    p = probs[:, ids]
    mass = p.sum(axis=-1)
    expected = (p / np.where(mass > 0, mass, 1.0)[:, None]) @ centers[bins]
    fixed = denormalise_action(expected, stats)
    extra = {f"cf{i}": float(v) for i, v in enumerate(fixed)}
    extra.update({f"m{i}": float(v) for i, v in enumerate(mass)})
    return readout, extra, p


def gate(processor, vla, dtype, samples, n=50):
    """Readout gate: the reimplemented decode must reproduce predict_action."""
    from model import verify_readout
    return verify_readout(processor, vla, samples[:n], dtype)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def done_keys(path, key_cols):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="") as f:
        return {tuple(r.get(k, "") for k in key_cols) for r in csv.DictReader(f)}


def log_row(path, readout, extra_readout, instruction, meta, fields):
    from model import append_prediction_log
    append_prediction_log(path, readout.action, instruction, meta, readout=readout,
                          **fields, **extra_readout)


def shard_items(items, shard, nshards):
    return [it for k, it in enumerate(items) if k % nshards == shard]


def out_path(base, shard, nshards):
    if nshards == 1:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}.shard{shard}{ext}"


DIST_RECORD_BYTES = 7 * 256 * 2   # one prediction's float16 action-token probabilities


def run(items, path, key_cols, processor, vla, dtype, meta, label, save_dist=False):
    """items: list of (image_loader, instruction, fields). Resume-safe.

    With save_dist, each prediction's 7x256 action-token probabilities are appended
    to `<path>.dist.f16` and the CSV row records its index as `dist_row`.
    """
    done = done_keys(path, key_cols)
    todo = [it for it in items if tuple(str(it[2].get(k, "")) for k in key_cols) not in done]
    print(f"[{label}] {len(items)} items, {len(items) - len(todo)} already logged, {len(todo)} to run -> {path}",
          flush=True)
    t0 = time.time()
    for k, (loader, instruction, fields) in enumerate(todo, 1):
        readout, extra, dist = predict(processor, vla, loader(), instruction, dtype)
        if save_dist:
            dist_path = path + ".dist.f16"
            row = os.path.getsize(dist_path) // DIST_RECORD_BYTES if os.path.exists(dist_path) else 0
            with open(dist_path, "ab") as f:
                f.write(np.asarray(dist, dtype=np.float16).tobytes())
            fields = {**fields, "dist_row": row}
        log_row(path, readout, extra, instruction, meta, fields)
        if k % 100 == 0 or k == len(todo):
            rate = k / (time.time() - t0)
            print(f"[{label}] {k}/{len(todo)}  {rate:.2f} it/s  eta {(len(todo) - k) / rate / 60:.1f} min",
                  flush=True)


# --------------------------------------------------------------------------- #
# Stimuli
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


# B5: pre-processed images, already transformed (scripts/termitech/preprocess_b5.py)
IMAGE_ROOT = os.environ.get("SGVLA_IMAGE_ROOT")


def constructed_image(construct_id, transform):
    if IMAGE_ROOT:
        key = ("p", construct_id, transform)
        if key not in _CACHE:
            _CACHE[key] = Image.open(os.path.join(IMAGE_ROOT, f"{construct_id}_{transform}.png")).convert("RGB")
        return _CACHE[key]
    from controls import apply_image_transform
    key = ("c", construct_id)
    if key not in _CACHE:
        _CACHE[key] = Image.open(os.path.join(DATA, "constructed", "frames", f"{construct_id}.png")).convert("RGB")
    return apply_image_transform(transform, _CACHE[key])


def bridge_image(rel_path, transform="original"):
    from controls import apply_image_transform
    key = ("b", rel_path)
    if key not in _CACHE:
        _CACHE[key] = Image.open(os.path.join(DATA, "bridge", rel_path)).convert("RGB")
    return apply_image_transform(transform, _CACHE[key])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def probe_rows():
    return read_csv(os.path.join(DATA, "probe_predictions.csv"))


def frozen_scenes():
    """One record per probed scene: id, noun, base frame, resolved arrangement."""
    manifest = {r["construct_id"]: r for r in read_csv(os.path.join(DATA, "constructed", "constructed_manifest.csv"))}
    scenes = {}
    for r in probe_rows():
        if r["condition"] == "baseline" and r["role"] == "a":
            m = manifest[r["scene_id"]]
            scenes[r["scene_id"]] = {**{k: r[k] for k in CARRY if k in r}, "noun": m["noun"]}
    return [scenes[k] for k in sorted(scenes)]


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_smoke(args):
    from model import describe_action_space
    processor, vla, dtype, meta = load(args.precision, args.attn)
    print(meta)
    space = describe_action_space(vla)
    for k, v in space.items():
        print(f"  {k:20} {v}")
    rows = [r for r in probe_rows() if r["condition"] == "baseline"][: 2 * args.n]
    samples = [(constructed_image(r["image_scene_id"], r["image_transform"]), r["instruction"]) for r in rows]
    print(gate(processor, vla, dtype, samples, n=args.n))
    # The corrected readout must agree with the original wherever no mass sits on 31744.
    r = rows[0]
    readout, extra, _ = predict(processor, vla, samples[0][0], samples[0][1], dtype)
    print("c :", np.round(readout.expected, 6))
    print("cf:", np.round([extra[f"cf{i}"] for i in range(7)], 6))
    print("m :", np.round([extra[f"m{i}"] for i in range(7)], 4))
    print("logged a1 / c1:", r["a1"], r["c1"], "| now:", readout.action[1], readout.expected[1])


def cmd_replay(args):
    processor, vla, dtype, meta = load(args.precision, args.attn)
    meta["run_tag"] = f"replay_{args.precision}"
    rows = probe_rows()
    if args.conditions != "all":
        keep = set(args.conditions.split(","))
        rows = [r for r in rows if r["condition"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    items = [((lambda r=r: constructed_image(r["image_scene_id"], r["image_transform"])),
              r["instruction"], {k: r[k] for k in CARRY}) for r in rows]
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "condition", "role", "image_scene_id"), processor, vla, dtype, meta, "replay",
        save_dist=args.save_dist)


PLACEBO = [
    # Same referent (the left twin) in two verbs, one pair per named-twin wording; with the
    # existing `paraphrase` pair these give the placebo contrast O4 for every wording.
    ("placebo_prenominal", "grab the left {noun}", "take the left {noun}"),
    ("placebo_object", "grab the object on the left", "take the object on the left"),
    ("placebo_table_side", "grab the {noun} on the left side of the table",
     "take the {noun} on the left side of the table"),
]


def ladder_instructions(noun, other, placebo=False):
    """(condition, role a, role b). Roles a/b name the left/right twin where a term is used."""
    extra = [(c, a.format(noun=noun), b.format(noun=noun)) for c, a, b in PLACEBO] if placebo else []
    return extra + [
        ("prenominal", f"pick up the left {noun}", f"pick up the right {noun}"),
        ("table_side", f"pick up the {noun} on the left side of the table",
         f"pick up the {noun} on the right side of the table"),
        ("move", "move left", "move right"),
        ("object", "pick up the object on the left", "pick up the object on the right"),
        ("absent_noun", f"pick up the {other} on the left", f"pick up the {other} on the right"),
        # Same referent (the left twin) in two other wordings: a noise floor for
        # how much a meaning-preserving rewording moves the action.
        ("paraphrase", f"grab the {noun} on the left", f"take the {noun} on the left"),
    ]


def cmd_ladder(args):
    processor, vla, dtype, meta = load(args.precision, args.attn)
    meta["run_tag"] = f"ladder{'_placebo' if args.placebo else ''}_{args.precision}"
    scenes = frozen_scenes()
    nouns = sorted({s["noun"] for s in scenes})
    rng = random.Random(0)
    items = []
    for s in scenes:
        other = rng.choice([n for n in nouns if n != s["noun"]])
        for transform in ("original", "mirror") if args.mirror else ("original",):
            for condition, instr_a, instr_b in ladder_instructions(s["noun"], other, placebo=args.placebo):
                for role, instruction in (("a", instr_a), ("b", instr_b)):
                    fields = {k: s.get(k, "") for k in CARRY}
                    fields.update({"role": role, "condition": f"ladder_{condition}",
                                   "image_transform": transform, "image_scene_id": s["scene_id"],
                                   "ladder_other_noun": other})
                    items.append(((lambda sid=s["scene_id"], t=transform: constructed_image(sid, t)),
                                  instruction, fields))
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "condition", "role", "image_transform"), processor, vla, dtype, meta, "ladder",
        save_dist=args.save_dist)


def cmd_natural(args):
    from controls import strip_spatial_term
    from data import make_pair
    processor, vla, dtype, meta = load(args.precision, args.attn)
    meta["run_tag"] = f"natural{'_lower' if args.lowercase else ''}_{args.precision}"
    manifest = read_csv(os.path.join(DATA, "bridge", "manifest.csv"))
    items = []
    for r in manifest:
        if r["split"] != "validation":
            continue
        made = make_pair(r["instruction"])
        if made is None:
            continue
        term, swapped = made
        neutral = strip_spatial_term(r["instruction"], term)
        variants = [("a", r["instruction"]), ("b", swapped)] + ([("n", neutral)] if neutral else [])
        if args.lowercase:   # OpenVLA lower-cases instructions in training and in its own evaluation
            variants = [(role, text.lower()) for role, text in variants]
        for transform in ("original", "mirror") if args.mirror else ("original",):
            for role, instruction in variants:
                fields = {"scene_id": f"b{int(r['episode_index']):06d}", "pair_id": f"b{int(r['episode_index']):06d}",
                          "base_scene_id": r["episode_index"], "role": role, "frame": "initial",
                          "scene_source": "bridge", "condition": "natural_swap", "image_transform": transform,
                          "image_scene_id": f"b{int(r['episode_index']):06d}", "spatial_term": term,
                          "axis": "lateral", "axis_index": 1, "category": r["category"], "sample_idx": 0,
                          "text_case": "lower" if args.lowercase else "raw"}
                items.append(((lambda p=r["image_path"], t=transform: bridge_image(p, t)), instruction, fields))
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "role", "image_transform"), processor, vla, dtype, meta, "natural",
        save_dist=args.save_dist)


# --------------------------------------------------------------------------- #
# Prefix control (BtP plan B2): dy with the dx token held fixed
# --------------------------------------------------------------------------- #
# OpenVLA decodes dx before dy, so a word can move dy directly or through the dx
# token it changes. Each unit is one scene x wording x image transform, with three
# prompts: left (a), right (b) and the term-free instruction (n). For every prompt,
# dy is read after each of the three greedy dx tokens (own, the other's, the
# neutral one), and as mixtures over dx: weighted by p(dx | neutral) for a marginal
# direct effect, and by the prompt's own p(dx) for the total effect under sampling.
# All steps run at batch size 1 through the cached path `generate` uses.
PREFIX_WORDINGS = {
    "baseline": ("pick up the {noun} on the left", "pick up the {noun} on the right", "pick up the {noun}"),
    "prenominal": ("pick up the left {noun}", "pick up the right {noun}", "pick up the {noun}"),
    "object": ("pick up the object on the left", "pick up the object on the right", "pick up the object"),
    "table_side": ("pick up the {noun} on the left side of the table",
                   "pick up the {noun} on the right side of the table", "pick up the {noun}"),
}
PREFIX_COVER = 0.99   # dx tokens covering this share of the action-token mass enter the mixtures


def prompt_pass(processor, vla, image, instruction, dtype):
    """The prefill `generate` runs: next-token (dx) logits and the KV cache as a legacy tuple."""
    import torch
    from action_bins import EMPTY_TOKEN_ID
    from model import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image.convert("RGB")).to("cuda:0", dtype=dtype)
    inputs.pop("attention_mask", None)
    input_ids = inputs.pop("input_ids")
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        pad = torch.tensor([[EMPTY_TOKEN_ID]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad), dim=1)
    with torch.inference_mode():
        # `generate` builds an all-ones attention mask for the prefill; pass the same.
        out = vla(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True, **inputs)
    cache = out.past_key_values
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return out.logits[0, -1, :].float().cpu().numpy(), cache


def forced_step(vla, cache, token_id):
    """dy logits after feeding one dx token: the cached step `generate` takes (batch size 1)."""
    import torch
    t = torch.tensor([[int(token_id)]], device="cuda:0")
    with torch.inference_mode():
        out = vla(input_ids=t, past_key_values=cache, use_cache=True)
    return out.logits[0, -1, :].float().cpu().numpy()


def _softmax(x):
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class DyReadout:
    """Folded expected value and argmax of the lateral dimension (dy) from logits."""

    def __init__(self, vla, unnorm_key="bridge_orig"):
        self.vocab = int(vla.vocab_size)
        self.centers = np.asarray(vla.bin_centers, dtype=np.float64)
        self.ids = np.arange(self.vocab - 256, self.vocab)
        self.bins = np.clip(self.vocab - self.ids - 1, 0, len(self.centers) - 1)
        stats = vla.get_action_stats(unnorm_key)
        self.lo, self.hi = float(stats["q01"][1]), float(stats["q99"][1])

    def denorm(self, x):
        return 0.5 * (x + 1) * (self.hi - self.lo) + self.lo

    def action_probs(self, logits):
        """Probabilities of the 256 action tokens, as a share of the full vocabulary."""
        return _softmax(np.asarray(logits, dtype=np.float64))[..., self.ids]

    def expected(self, p256):
        p256 = np.asarray(p256, dtype=np.float64)
        return self.denorm((p256 / p256.sum(axis=-1, keepdims=True)) @ self.centers[self.bins])

    def argmax_value(self, logits):
        tok = int(np.argmax(logits))
        return self.denorm(self.centers[int(np.clip(self.vocab - tok - 1, 0, len(self.centers) - 1))]), tok


def _cover(p256, ids, share=PREFIX_COVER):
    """Action tokens (ids) covering `share` of the action-token mass, with their probabilities."""
    order = np.argsort(p256)[::-1]
    cum = np.cumsum(p256[order]) / p256.sum()
    k = int(np.searchsorted(cum, share) + 1)
    return ids[order[:k]], p256[order[:k]]


def prefix_unit(processor, vla, dtype, ro, image, texts):
    """Rows for one scene x wording x transform; texts = {'a': left, 'b': right, 'n': neutral}."""
    passes = {r: prompt_pass(processor, vla, image, t, dtype) for r, t in texts.items()}
    dx_tok = {r: int(np.argmax(lg)) for r, (lg, _) in passes.items()}
    dx_p = {r: ro.action_probs(lg) for r, (lg, _) in passes.items()}
    cover = {r: _cover(dx_p[r], ro.ids) for r in passes}
    union = sorted(set().union(*[set(c[0].tolist()) for c in cover.values()]))
    rows, dists = {}, {}
    for r, (_, cache) in passes.items():
        # Every forced step runs alone, at batch size 1: a batched step over the dx tokens
        # changed dy by up to 3.6 bins in bf16 (gate, 3 Oct), while this path reproduces
        # `generate` exactly.
        step = {tok: forced_step(vla, cache, tok) for tok in sorted(set(union) | set(dx_tok.values()))}
        forced = {src: step[dx_tok[src]] for src in ("a", "b", "n")}
        mix_p = {tok: ro.action_probs(step[tok]) for tok in union}
        row = {"dx_token": dx_tok[r], "dx_token_p": float(dx_p[r][dx_tok[r] - ro.ids[0]] / dx_p[r].sum())}
        for src, lg in forced.items():
            row[f"ct1_from_{src}"] = float(ro.expected(ro.action_probs(lg)))
            row[f"at1_from_{src}"] = float(ro.argmax_value(lg)[0])
        row["ct1_own"] = row[f"ct1_from_{r}"]
        mixes = {}
        for name, weights_of in (("npi", "n"), ("own", r)):
            toks, w = cover[weights_of]
            pdy = sum(wi * mix_p[int(t)] for t, wi in zip(toks, w)) / w.sum()
            mixes[name] = pdy
            row[f"cm1_{name}"] = float(ro.expected(pdy))
            row[f"cover_k_{name}"] = int(len(toks))
            row[f"cover_mass_{name}"] = float(w.sum() / dx_p[weights_of].sum())
        rows[r] = row
        dists[r] = np.stack([ro.action_probs(forced[s]) for s in ("a", "b", "n")] + [mixes["npi"], mixes["own"]])
    return rows, dists, passes, union


def prefix_gate(processor, vla, dtype, ro, units, n=20):
    """The hand decode must reproduce `generate` (own dx token, then dy)."""
    worst, sizes, t0 = 0.0, [], time.time()
    for image_loader, texts, _ in units[:n]:
        image = image_loader()
        rows, _, _, union = prefix_unit(processor, vla, dtype, ro, image, texts)
        sizes.append(len(union))
        for r, t in texts.items():
            _, extra, _ = predict(processor, vla, image, t, dtype)
            worst = max(worst, abs(extra["cf1"] - rows[r]["ct1_own"]))
    one_bin = (ro.hi - ro.lo) / 255
    return {"units": min(n, len(units)), "max_abs_dev_generate_bins": worst / one_bin,
            "dx_tokens_per_unit_median": float(np.median(sizes)), "dx_tokens_per_unit_max": int(max(sizes)),
            "seconds_per_unit_incl_gate": (time.time() - t0) / max(1, min(n, len(units)))}


def cmd_prefix(args):
    processor, vla, dtype, meta = load(args.precision, args.attn)
    meta["run_tag"] = f"prefix_{args.precision}"
    ro = DyReadout(vla)
    scenes = frozen_scenes()
    units = []
    for s in scenes:
        for transform in ("original", "mirror") if args.mirror else ("original",):
            for wording, (ta, tb, tn) in PREFIX_WORDINGS.items():
                texts = {"a": ta.format(noun=s["noun"]), "b": tb.format(noun=s["noun"]), "n": tn.format(noun=s["noun"])}
                fields = {k: s.get(k, "") for k in CARRY}
                fields.update({"condition": f"prefix_{wording}", "image_transform": transform,
                               "image_scene_id": s["scene_id"]})
                units.append(((lambda sid=s["scene_id"], t=transform: constructed_image(sid, t)), texts, fields))
    units = shard_items(units, args.shard, args.nshards)
    if args.gate:
        import json
        print(json.dumps(prefix_gate(processor, vla, dtype, ro, units, n=args.gate)), flush=True)
        return
    path = out_path(args.out, args.shard, args.nshards)
    done = done_keys(path, ("scene_id", "condition", "role", "image_transform"))
    todo = [u for u in units if any((str(u[2]["scene_id"]), u[2]["condition"], r, u[2]["image_transform"]) not in done
                                    for r in "abn")]
    print(f"[prefix] {len(units)} units, {len(units) - len(todo)} done, {len(todo)} to run -> {path}", flush=True)
    t0 = time.time()
    for k, (loader, texts, fields) in enumerate(todo, 1):
        rows, dists, _, _ = prefix_unit(processor, vla, dtype, ro, loader(), texts)
        for r in "abn":
            row = {**meta, **fields, "role": r, "instruction": texts[r], **rows[r]}
            dist_path = path + ".dist.f16"
            row["dist_row"] = os.path.getsize(dist_path) // (5 * 256 * 2) if os.path.exists(dist_path) else 0
            with open(dist_path, "ab") as f:
                f.write(np.asarray(dists[r], dtype=np.float16).tobytes())
            from prediction_log import append_row
            append_row(path, row)
        if k % 50 == 0 or k == len(todo):
            rate = k / (time.time() - t0)
            print(f"[prefix] {k}/{len(todo)}  {rate:.2f} units/s  eta {(len(todo) - k) / rate / 60:.1f} min", flush=True)


def cmd_manifest(args):
    """BtP plan B6: left/right on composites outside the frozen set, scored by recorded arrangement."""
    processor, vla, dtype, meta = load(args.precision, args.attn)
    meta["run_tag"] = f"manifest_{args.precision}"
    man = read_csv(os.path.join(DATA, "constructed", "constructed_manifest.csv"))
    review = {r["construct_id"]: r for r in read_csv(os.path.join(DATA, "constructed", "constructed_review.csv"))}
    keep = set(args.decisions.split(","))
    items = []
    for m in man:
        decision = review.get(m["construct_id"], {}).get("decision", "") or "unscreened"
        if decision not in keep:
            continue
        for role, text in (("a", m["instr_a"]), ("b", m["instr_b"])):
            fields = {"scene_id": m["construct_id"], "pair_id": m["construct_id"], "base_scene_id": m["base_scene_id"],
                      "role": role, "condition": "manifest", "image_transform": "original",
                      "image_scene_id": m["construct_id"], "configuration": m["configuration"],
                      "screen_decision": decision, "gripper_source": m["gripper_source"],
                      "target_sign_a_image": m["target_sign_a_image"], "target_sign_b_image": m["target_sign_b_image"],
                      "expected_sign_image": m["expected_sign_image"], "axis": "lateral", "axis_index": 1}
            items.append(((lambda cid=m["construct_id"]: constructed_image(cid, "original")), text, fields))
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "role", "image_transform"), processor, vla, dtype, meta, "manifest", save_dist=args.save_dist)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "replay", "ladder", "natural", "prefix", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--precision", choices=PRECISIONS, default="nf4")
        p.add_argument("--attn", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
        p.add_argument("--decisions", default="rejected,approved", help="manifest: screening decisions to run")
        p.add_argument("--out", default=f"{ROOT}/runs/{name}.csv")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--nshards", type=int, default=1)
        p.add_argument("--n", type=int, default=50)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--conditions", default="all")
        p.add_argument("--mirror", action="store_true")
        p.add_argument("--save-dist", action="store_true", help="append 7x256 float16 action-token probabilities")
        p.add_argument("--placebo", action="store_true", help="ladder: add same-twin grab/take pairs per wording")
        p.add_argument("--lowercase", action="store_true", help="natural: lower-case the instructions")
        p.add_argument("--gate", type=int, default=0, help="prefix: run the gate on N units and exit")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    {"smoke": cmd_smoke, "replay": cmd_replay, "ladder": cmd_ladder, "natural": cmd_natural,
     "prefix": cmd_prefix, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
