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
def load(precision: str):
    import torch
    from model import load_openvla, run_metadata

    processor, vla, dtype = load_openvla(quantize_4bit=(precision == "nf4"), precision="bf16")
    meta = run_metadata(dtype)
    meta["precision"] = precision
    torch.backends.cuda.matmul.allow_tf32 = False
    return processor, vla, dtype, meta


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
    return readout, extra


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


def run(items, path, key_cols, processor, vla, dtype, meta, label):
    """items: list of (image_loader, instruction, fields). Resume-safe."""
    done = done_keys(path, key_cols)
    todo = [it for it in items if tuple(str(it[2].get(k, "")) for k in key_cols) not in done]
    print(f"[{label}] {len(items)} items, {len(items) - len(todo)} already logged, {len(todo)} to run -> {path}",
          flush=True)
    t0 = time.time()
    for k, (loader, instruction, fields) in enumerate(todo, 1):
        readout, extra = predict(processor, vla, loader(), instruction, dtype)
        log_row(path, readout, extra, instruction, meta, fields)
        if k % 100 == 0 or k == len(todo):
            rate = k / (time.time() - t0)
            print(f"[{label}] {k}/{len(todo)}  {rate:.2f} it/s  eta {(len(todo) - k) / rate / 60:.1f} min",
                  flush=True)


# --------------------------------------------------------------------------- #
# Stimuli
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def constructed_image(construct_id, transform):
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
    processor, vla, dtype, meta = load(args.precision)
    print(meta)
    space = describe_action_space(vla)
    for k, v in space.items():
        print(f"  {k:20} {v}")
    rows = [r for r in probe_rows() if r["condition"] == "baseline"][: 2 * args.n]
    samples = [(constructed_image(r["image_scene_id"], r["image_transform"]), r["instruction"]) for r in rows]
    print(gate(processor, vla, dtype, samples, n=args.n))
    # The corrected readout must agree with the original wherever no mass sits on 31744.
    r = rows[0]
    readout, extra = predict(processor, vla, samples[0][0], samples[0][1], dtype)
    print("c :", np.round(readout.expected, 6))
    print("cf:", np.round([extra[f"cf{i}"] for i in range(7)], 6))
    print("m :", np.round([extra[f"m{i}"] for i in range(7)], 4))
    print("logged a1 / c1:", r["a1"], r["c1"], "| now:", readout.action[1], readout.expected[1])


def cmd_replay(args):
    processor, vla, dtype, meta = load(args.precision)
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
        ("scene_id", "condition", "role", "image_scene_id"), processor, vla, dtype, meta, "replay")


def ladder_instructions(noun, other):
    """(condition, role a, role b). Roles a/b name the left/right twin where a term is used."""
    return [
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
    processor, vla, dtype, meta = load(args.precision)
    meta["run_tag"] = f"ladder_{args.precision}"
    scenes = frozen_scenes()
    nouns = sorted({s["noun"] for s in scenes})
    rng = random.Random(0)
    items = []
    for s in scenes:
        other = rng.choice([n for n in nouns if n != s["noun"]])
        for transform in ("original", "mirror") if args.mirror else ("original",):
            for condition, instr_a, instr_b in ladder_instructions(s["noun"], other):
                for role, instruction in (("a", instr_a), ("b", instr_b)):
                    fields = {k: s.get(k, "") for k in CARRY}
                    fields.update({"role": role, "condition": f"ladder_{condition}",
                                   "image_transform": transform, "image_scene_id": s["scene_id"],
                                   "ladder_other_noun": other})
                    items.append(((lambda sid=s["scene_id"], t=transform: constructed_image(sid, t)),
                                  instruction, fields))
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "condition", "role", "image_transform"), processor, vla, dtype, meta, "ladder")


def cmd_natural(args):
    from controls import strip_spatial_term
    from data import make_pair
    processor, vla, dtype, meta = load(args.precision)
    meta["run_tag"] = f"natural_{args.precision}"
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
        for transform in ("original", "mirror") if args.mirror else ("original",):
            for role, instruction in variants:
                fields = {"scene_id": f"b{int(r['episode_index']):06d}", "pair_id": f"b{int(r['episode_index']):06d}",
                          "base_scene_id": r["episode_index"], "role": role, "frame": "initial",
                          "scene_source": "bridge", "condition": "natural_swap", "image_transform": transform,
                          "image_scene_id": f"b{int(r['episode_index']):06d}", "spatial_term": term,
                          "axis": "lateral", "axis_index": 1, "category": r["category"], "sample_idx": 0}
                items.append(((lambda p=r["image_path"], t=transform: bridge_image(p, t)), instruction, fields))
    items = shard_items(items, args.shard, args.nshards)
    run(items, out_path(args.out, args.shard, args.nshards),
        ("scene_id", "role", "image_transform"), processor, vla, dtype, meta, "natural")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "replay", "ladder", "natural"):
        p = sub.add_parser(name)
        p.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
        p.add_argument("--out", default=f"{ROOT}/runs/{name}.csv")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--nshards", type=int, default=1)
        p.add_argument("--n", type=int, default=50)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--conditions", default="all")
        p.add_argument("--mirror", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    {"smoke": cmd_smoke, "replay": cmd_replay, "ladder": cmd_ladder, "natural": cmd_natural}[args.cmd](args)


if __name__ == "__main__":
    main()
