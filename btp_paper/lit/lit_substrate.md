# Literature review C: the substrate

Topics: action tokenisation, numerical precision, image transforms, robot-data language, and spatial language.

*Compiled 2 Oct 2026 for the anonymous CoRL 2026 BtP workshop paper. Companion file: `lit_substrate.bib`, which holds verified entries only.*

**How this file was verified.** Every entry below was checked against a primary source fetched in this session:

- **arXiv:** the export API for metadata, plus the PDF read with PyMuPDF for quotes.
- **Venues:** proceedings pages at PMLR, roboticsproceedings.org, CVF open access, papers.nips.cc, ACL Anthology, OpenReview and AAAI OJS, or Crossref/IEEE metadata.
- **Code:** raw files at a pinned commit.

Quotes are verbatim, including the authors' spelling and typos. Page numbers refer to the arXiv PDF version that was fetched (shown as vN). Anything derived by us rather than stated by a source is marked **[derived]**.

---

## Read this first: points that refine or contradict our current description

1. **The paper and the code disagree on bin count; our description matches the code.**
   - The OpenVLA paper says actions are discretised "into one of 256 bins … to uniformly divide the interval between the 1st and 99th quantile", giving "N discrete integers ∈[0 . . . 255]".
   - The released code instead builds `np.linspace(-1, 1, 256)`, which gives 256 *edges* and **255 bin centres**. `np.digitize` returns 1…256, and decoding clips index 255 to 254. So the top action token (id 31744) and the next one (31745) decode to the *same* centre.
   - The decodable grid step is therefore (q99 − q01)/255, which is what we state. A literal reading of the paper would imply /256.
   - The code docstring admits this: "there are actually only (# bins - 1) bin intervals".
2. **OpenVLA's "4-bit matches bf16" claim is weaker than it sounds, and "4-bit" is not one setting.**
   - Table 2 (71.9% int4 vs 71.3% bf16) comes from 8 Bridge tasks with 80 rollouts per precision. It used a *different checkpoint*: per footnote 4, Sections 5.3–5.4 use a SigLIP-only model pretrained on the Octo mixture, not the released `openvla-7b`.
   - Footnote 5 says int8 and int4 reach "comparable token accuracy" offline and points to App. D.4. But App. D.4 reports only blocking-control success rates (70.0/74.4/68.8%), not token accuracy.
   - The paper does not name the 4-bit data type. It cites QLoRA, which introduced NF4, and LLM.int8(). The official evaluation code passes `load_in_4bit=True` straight to `from_pretrained`. In the pinned transformers 4.40.1 this builds a `BitsAndBytesConfig` whose default is `bnb_4bit_quant_type="fp4"` **[derived from code]**.
   - We ran NF4, so we should report NF4 and its compute dtype explicitly.
3. **Bridge's annotation protocol explains our destination-dominance finding.** BridgeData V2 instructions were written *post hoc* by crowd workers, who "were asked to describe the task being performed by the robot in each trajectory, with particular emphasis on the final location of any moved objects". The acknowledgements add that Microsoft Research helped label "parts of the data with language". Collectors were not required to label tasks during collection. Nothing here contradicts our description.
4. **Neither the Bridge paper nor the released dataset metadata defines the action frame or sign; the convention has to be inferred.**
   - The paper only says "continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose".
   - The official TFDS `features.json` describes the 7-D `action` as "[7x joint velocities, 2x gripper velocities, 1x terminate episode]", which is a template string and wrong for this 7-D feature. It labels all four images "Main camera RGB observation".
   - The robot-side code says translation axes "are the same as world".
   - OpenVLA does not train on Bridge's commanded actions. It re-derives them as differences of successive proprioceptive states (`relabel_bridge_actions`).
   - Independent corroboration of our sign convention: ECoT's Bridge labelling code maps the y-axis as `{-1: "right", 0: None, 1: "left"}`, "assuming a fixed camera".
5. **Decoding defaults.**
   - Action dimensions are generated autoregressively, one token per dimension (7 for `bridge_orig`). So dy is conditioned on the dx token.
   - Greedy decoding is the de facto default. `generation_config.json` sets no sampling flags, and every official call passes `do_sample=False`.
   - `predict_action` silently appends token 29871 to the prompt "to match the inputs seen at training time". This is part of the autoregressive prefix.
   - It does *not* restrict generation to the 256 action tokens. RT-2 did impose an output constraint.

---

## (a) Verified facts we can state

### A1. OpenVLA, the paper (Kim et al.; arXiv 2406.09246v3; CoRL 2024, PMLR 270:2679–2713)

Sources: https://arxiv.org/abs/2406.09246 and https://proceedings.mlr.press/v270/kim25c.html

| # | Fact | Verbatim quote (location) |
|---|---|---|
| A1.1 | Per-dimension 256-bin discretisation over the 1st–99th quantile | "we discretize each dimension of the robot actions separately into one of 256 bins. For each action dimension, we set the bin width to uniformly divide the interval between the 1st and 99th quantile of the actions in the training data. Using quantiles instead of the min-max bounds Brohan et al. [7] used allows us to ignore outlier actions in the data that could otherwise drastically expand the discretization interval and reduce the effective granularity of our action discretization." (§3.2, p.5) |
| A1.2 | Integer range | "Using this discretization, we obtain N discrete integers ∈[0 . . . 255] for an N-dimensional robot action." (§3.2, p.5) |
| A1.3 | Which tokens are overwritten | "the Llama tokenizer [10], only reserves 100 “special tokens” for tokens newly introduced during fine-tuning, which is too few for the 256 tokens of our action discretization. Instead, we again opt for simplicity and follow Brohan et al. [7]’s approach by simply overwriting the 256 least used tokens in the Llama tokenizer’s vocabulary (which corresponds to the last 256 tokens) with our action tokens." (§3.2, p.5) |
| A1.4 | Training objective | "OpenVLA is trained with a standard next-token prediction objective, evaluating the cross-entropy loss on the predicted action tokens only." (§3.2, p.5) |
| A1.5 | bf16 memory and speed | "During inference, OpenVLA requires 15GB of GPU memory when loaded in bfloat16 precision (i.e., without quantization) and runs at approximately 6Hz on one NVIDIA RTX 4090 GPU (without compilation, speculative decoding, or other inference speed-up tricks)." (§3.5) |
| A1.6 | Precision default | "We follow best-practices from LLM serving by saving and loading OpenVLA in bfloat16 precision for inference (our default approach), which cuts the memory footprint in half" (§5.4, p.10) |
| A1.7 | Table 2 (quantised inference) | bfloat16 "71.3 ± 4.8%", "16.8 GB"; int8 "58.1 ± 5.1%", "10.2 GB"; int4 "71.9 ± 4.7%", "7.0 GB". Caption: "4-bit quantization matches the performance of bfloat16 inference (our default approach) while reducing the GPU memory footprint by more than half. Mean success ± StdErr computed across 8 representative BridgeData V2 tasks [6] and 80 rollouts per approach" (p.10) |
| A1.8 | Speed effects of quantisation | "We observe that 8-bit quantization slows down inference across most GPUs, due to the overhead of the added quantization operations. 4-bit inference achieves higher throughput, since reduced GPU memory transfer compensates for the quantization overhead." … "on the A5000 GPU we use for our evaluations, we can only run the model at 1.2Hz, which significantly changes the system dynamics compared to the training dataset for the 5Hz non-blocking controller used in the BridgeData V2 tasks." … "4-bit quantized models can run at 3Hz on the A5000" (§5.4, p.11) |
| A1.9 | The quantisation results come from a different model | Footnote 4: "In Section 5.3 and Section 5.4, we experiment with a version of the OpenVLA model that is pretrained with a smaller robot data mixture (the same OpenX dataset mixture as Octo) and has a slightly smaller architecture which only uses a SigLIP [79] vision backbone instead of the fused DinoSigLIP encoder." (p.10) |
| A1.10 | Action-level agreement is claimed but not shown | Footnote 5: "We attribute the performance loss to low inference speed, since both 8-bit and 4-bit quantization achieve comparable token accuracy to bfloat16 inference when evaluated offline on training data. See Appendix D.4 for supporting details." (p.11). App. D.4 / Table 11 (blocking control) reports success only: bf16 "70.0 ± 5.1%", int8 "74.4 ± 4.9%", int4 "68.8 ± 5.2%". The caption reads "All average success rates have overlapping error bars, which suggests that all methods perform comparably." (pp.34–35) |
| A1.11 | Prompt shown in the paper's figure | Fig. 2 shows “What should the robot do to {task}? A:” (p.4). The released code uses a different template (A2.8). |
| A1.12 | Quantisation references | [27] is QLoRA (Dettmers et al., NeurIPS 36) and [88] is GPT3.int8() / LLM.int8() (NeurIPS 35) (references list). |

### A2. OpenVLA, the official code

Pinned commits:

- **GitHub:** `openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9`. The tokenizer file has been unchanged since the release commit `9423e9c`.
- **Hugging Face:** `openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f`.

**A2.1 Bins and centres, training side.** Source: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py#L30-L36

```python
31  self.bins = np.linspace(min_action, max_action, self.n_bins)
32  self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
36  self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
```

The defaults are `bins: int = 256, min_action: int = -1, max_action: int = 1` (L15).

**A2.2 Token-id mapping, encoding.** Source: L39–45 of the same file.

```python
40  action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
41  discretized_action = np.digitize(action, self.bins)
45  return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
```

**A2.3 Decoding and the clip that folds the top token.** Source: L49–68.

> "NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the digitization returns bin indices between [1, # bins], inclusive, when there are actually only (# bins - 1) bin intervals. … EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There is still one index (i==255) that would cause an out-of-bounds error if used to index into self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of the last bin center. We implement this simply via clipping between [0, 255 - 1]."

```python
65  discretized_actions = self.tokenizer.vocab_size - action_token_ids
66  discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
68  return self.bin_centers[discretized_actions]
```

**A2.4 The same logic in the Hugging Face model that we actually run.**

- Blob: https://huggingface.co/openvla/openvla-7b/blob/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py#L499-L536
- Raw: https://huggingface.co/openvla/openvla-7b/raw/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py

```python
500  self.bins = np.linspace(-1, 1, config.n_action_bins)
501  self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
504  self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
510  # If the special empty token ('') does not already appear after the colon (':') token in the prompt
511  # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
512  if not torch.all(input_ids[:, -1] == 29871):
518  generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
521  predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
522  discretized_actions = self.vocab_size - predicted_action_token_ids
523  discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
524  normalized_actions = self.bin_centers[discretized_actions]
528  mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
529  action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
532  0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
```

The relevant `config.json` lines (https://huggingface.co/openvla/openvla-7b/blob/47a0ec7fc4ec123775a391911046cf33cf9ed83f/config.json):

- L19: `"n_action_bins": 256`
- L3148: `"pad_to_multiple_of": 64`
- L3154: `"vocab_size": 32064`

That gives `vocab_size` = 32000.

**[derived]** The action tokens are ids 31744–31999 (32000 − {1…256}), and they decode to **255** distinct centres: id 31999 maps to centre 0, and ids 31745 and 31744 both map to centre 254. The decodable unnormalised grid step is (q99 − q01)/255.

Neither q01 nor q99 can itself be decoded; the outermost centres sit half a step inside them. Because `generate` is unconstrained, any non-action token would be clipped to an extreme centre: ids < 31744 go to centre 254 and pad ids ≥ 32000 go to centre 0. This follows from L522–523.

**A2.5 Bridge statistics we decode with.** Source: `config.json`, `bridge_orig`, L896–963.

- dy (index 1): q01 = −0.04170349963009357 (L936) and q99 = 0.040855254605412394 (L945).
- `mask` = [true×6, false] (gripper not unnormalised).
- `"num_trajectories": 60064` (L962).

**[derived]** The dy argmax grid step is (0.040855 + 0.041703)/255 ≈ 3.24 × 10⁻⁴ Bridge action units. The dx step is ≈ 2.24 × 10⁻⁴.

**A2.6 Greedy decoding by default.**

- `generation_config.json` contains only `bos_token_id`, `eos_token_id`, `pad_token_id` and `transformers_version: 4.40.1`. No sampling flags are set, so transformers falls back to greedy decoding **[derived]**. Source: https://huggingface.co/openvla/openvla-7b/blob/47a0ec7fc4ec123775a391911046cf33cf9ed83f/generation_config.json
- The official examples all decode greedily:
  - GitHub README L71: `action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)` (https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/README.md?plain=1#L67-L71)
  - HF README L80 has the same call.
  - Evaluation helper `experiments/robot/openvla_utils.py` L169: `action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)` (https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/openvla_utils.py#L163-L169)

**A2.7 Autoregressive generation of action dimensions.**

- Code: `max_new_tokens=self.get_action_dim(unnorm_key)` (A2.4 L518). `get_action_dim` returns `len(self.norm_stats[unnorm_key]["action"]["q01"])`, which is 7 for `bridge_orig`.
- Action layout: `ActionEncoding.EEF_POS`, commented "EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)" (https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/oxe/configs.py#L44-L49). So dy is the 2nd token, generated after dx.
- OpenVLA-OFT says this explicitly: "The original OpenVLA training scheme includes autoregressive decoding, discrete actions, and next-token prediction" (Fig. 2 caption, arXiv 2502.19645v2 p.3). It also says parallel decoding works "by replacing 7 sequential forward passes through the decoder portion of the policy with a single pass" (p.6).

**A2.8 Prompt templates: training, inference and evaluation differ in form.**

- Training (`prismatic/vla/datasets/datasets.py` L42, L47):
  - `lang = rlds_batch["task"]["language_instruction"].decode().lower()`
  - `{"from": "human", "value": f"What action should the robot take to {lang}?"}`
  - Source: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/datasets.py#L42-L48
- Inference (README L67): `prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"`
- Evaluation (openvla_utils L163): `prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"`
- So instructions are lower-cased at both training and evaluation.

**A2.9 Normalisation.** Source: `prismatic/vla/datasets/rlds/utils/data_utils.py`.

- L53: `BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]`
- L90: `tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)` with `low = metadata[key]["q01"]` and `high = metadata[key]["q99"]` (L82–83).
- L247–248: `"q01": np.quantile(actions, 0.01, axis=0).tolist()` and `"q99": np.quantile(actions, 0.99, axis=0).tolist()`.
- Source: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/utils/data_utils.py#L76-L93

**A2.10 What a Bridge "action" is inside OpenVLA.** It is the achieved end-effector displacement, not the command.

- `relabel_bridge_actions` (data_utils L166–172): "Relabels actions to use reached proprioceptive state; discards last timestep (no-action)." The code is `movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]`.
- It is applied in `bridge_orig_dataset_transform` (`oxe/transforms.py` L61–85; the registry at L847–848 maps both `"bridge_orig"` and `"bridge_dataset"` to it). The OXE variant `bridge_oxe_dataset_transform` (L31–58, L55) also calls it, so every Bridge path in OpenVLA relabels. That transform also uses `binarize_gripper_actions` and drops the first all-zero step: "In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!"
- `configs.py` L79–84 makes `image_0` the primary camera: `"bridge_orig": {  # Original version of Bridge V2 from project website` / `"image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None}`.

**A2.11 Image transforms.**

- **Training augmentation, when enabled** (`datasets.py` L122–135): `random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0])`, `random_brightness=[0.2]`, `random_contrast=[0.8, 1.2]`, `random_saturation=[0.8, 1.2]` and `random_hue=[0.05]`. There is **no horizontal flip**.
- The pretraining script defaults to `image_aug: bool = False` (`vla-scripts/train.py` L72).
- **Evaluation:** the README warns "**NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop)." This note is for LIBERO models.
- The Bridge evaluation script asserts `assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"` (https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/bridge/run_bridgev2_eval.py#L46-L49 and #L84). Its flags `load_in_8bit` and `load_in_4bit` both default to False.
- **Numerical nondeterminism:**

  > "The results reported in our paper were obtained using **Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, and flash-attn 2.5.5** on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions. Note that results may vary slightly if you use a different GPU for evaluation due to GPU nondeterminism in large models (though we have tested that results were consistent across different machines with A100 GPUs)."

  Source: GitHub README, LIBERO notes, around L583–593.

**A2.12 The official 4-bit path.** The code is:

```python
vla = AutoModelForVision2Seq.from_pretrained(..., torch_dtype=torch.bfloat16, load_in_8bit=cfg.load_in_8bit, load_in_4bit=cfg.load_in_4bit, ...)
```

Source: `openvla_utils.py` L43–51.

In the pinned transformers v4.40.1:

- `modeling_utils.py` L3090–3103 turns these kwargs into a `BitsAndBytesConfig` ("handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation").
- `quantization_config.py` L243–245 sets the defaults `bnb_4bit_compute_dtype=None`, `bnb_4bit_quant_type="fp4"` and `bnb_4bit_use_double_quant=False`.
- L263–264 resolve `if bnb_4bit_compute_dtype is None: self.bnb_4bit_compute_dtype = torch.float32`.
- Sources: https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/utils/quantization_config.py#L236-L264 and https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/modeling_utils.py#L3090-L3107

**[derived]** OpenVLA's own 4-bit path is therefore bitsandbytes FP4 unless the user overrides it. NF4 is the QLoRA data type ("4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights", arXiv 2305.14314v1 abstract). So "4-bit" names at least two configurations.

### A3. The action representations around OpenVLA

| Paper | Quote (location) |
|---|---|
| RT-1 (RSS 2023) | "To tokenize actions, each action dimension in RT-1 is discretized into 256 bins. … For each variable, we map the target to one of the 256 bins, where the bins are uniformly distributed within the bounds of each variable." (arXiv 2212.06817v2, "Action tokenization" paragraph, p.6) |
| RT-2 (CoRL 2023) | "The continuous dimensions (all dimensions except for the discrete termination command) are discretized into 256 bins uniformly." "For the PaLM-E model, which does not provide this convenient representation of numbers, we simply overwrite the 256 least frequently used tokens to represent the action vocabulary." **Output constraint:** "to ensure that RT-2 outputs valid action tokens during decoding, we constrain its output vocabulary via only sampling valid action tokens when the model is prompted with a robot-action task" (arXiv 2307.15818v1, §3.2, pp.5–6) |
| Open X-Embodiment (ICRA 2024) | "We use a coarsely aligned action and observation space across datasets." "We normalize each dataset’s actions prior to discretization." **"Similarly, for the action space, we do not align the coordinate frames across datasets in which the end-effector is controlled, and allow action values to represent either absolute or relative positions or velocities, as per the original control scheme chosen for each robot. Thus, the same action vector may induce very different motions for different robots."** "the action is tokenized into 256 bins uniformly distributed along each of eight dimensions" (arXiv 2310.08864v9, §IV-A/B, p.4). Language analysis: "We use the PaLM language model [3] to extract objects and behaviors from the instructions." (p.4) |
| Octo (RSS 2024) | "We use a conditional diffusion decoding head to predict continuous, multi-modal action distributions". Only the gripper convention was aligned: "align the gripper action spaces between the datasets such that a gripper command of +1 means “the gripper is open” and 0 means “the gripper is closed.”" Augmentation: "stochastic crops followed be a resize to 256×256, followed by color jitter" (arXiv 2405.12213v2, pp.5, 14) |
| FAST (RSS 2025) | "Prior robotic policies of this sort typically use naïve tokenization strategies based on a per-dimension, per-timestep binning scheme [9, 10, 39]. We find that such methods perform poorly when learning dexterous skills with high-frequency control". "Intuitively, in such cases low token prediction loss can often be achieved with mappings as trivial as simply copying the most recent action token". "OpenVLA worked well on the low-frequency BridgeV2 and RT-1 datasets, but has struggled to fit the higher-frequency DROID dataset [39]." FAST also normalises "such that the 1st and 99th quantile of values in the training dataset for each action dimension maps to the range [−1, . . . , 1]" (arXiv 2501.09747v1, pp.1–4) |
| OpenVLA-OFT (RSS 2025) | "We examine three key design choices: action decoding scheme (autoregressive vs. parallel generation), action representation (discrete vs. continuous), and learning objective". "continuous action representations further improve model quality compared to discrete representations". "For a chunk size of K timesteps and action dimensionality D, OpenVLA requires KD sequential decoder forward passes versus just D passes without chunking." (arXiv 2502.19645v2, pp.1, 3) |
| π0 (RSS 2025) | "Such models employ autoregressive discretization to represent actions in a manner analogous to text tokens. In contrast, our model employs a novel design that fine-tunes a VLM to produce actions via flow matching" (arXiv 2410.24164, p.3) |
| π0.5 (CoRL 2025) | "Robotic data uses the FAST action tokenizer to represent actions as discrete tokens [64]." "Our model is therefore trained to predict actions both through autoregressive sampling of tokens (using the FAST tokenizer) and iterative integration of the flow field" (arXiv 2504.16054v1, pp.4–5) |
| CogACT (arXiv 2024) | "works like [8, 30] directly quantize the continuous spectrum of robot actions into discrete bins … such a simple quantization, unlike sophisticated tokenizers such as those designed for images [65, 72] and audio [19, 73], poses difficulties in action learning and limits action precision." (arXiv 2411.19650v1, p.2) |
| SpatialVLA (RSS 2025) | "propose Adaptive Action Grids to represent spatial robot movement actions with adaptive discretized action grids"; "grids are split on each action variable according to the probability density function of fitted Gaussian distribution" (arXiv 2501.15830v5, pp.1, 4) |
| MiniVLA (blog, Dec 2024) | "OpenVLA and other works use a simple binning scheme, where each dimension of the action (usually 7 total dimensions) is binned within some minimum and maximum value into N bins (usually 256 bins)." It introduces "VQ action chunking" (https://ai.stanford.edu/blog/minivla/) |
| VQ-VLA (ICCV 2025), OAT (RSS 2026), VLA-0 (arXiv 2025) | These are learned or alternative tokenisers. OAT lists "high compression, total decodability, and a left-to-right causally ordered token space" as desiderata. VLA-0 represents "actions directly as text" (arXiv abstracts 2507.01016, 2602.04215, 2510.13054) |
| ECoT (CoRL 2024) | It derives direction *words* from robot motion: "we use the robot proprioception to determine the movement direction for the next 4 time steps (assuming a fixed camera), and translate this into one of 729 templated movement primitives" (arXiv 2407.08693v3, p.5). Code: `{-1: "right", 0: None, 1: "left"},` is the y-axis entry (https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/scripts/generate_embodied_data/primitive_movements.py#L4-L13). It is computed from `diff = move[-1] - move[0]` over Bridge `observation/state` (L50–73). |

### A4. BridgeData V2: annotation, cameras and action space (Walke et al.; CoRL 2023, PMLR 229:1723–1736; arXiv 2308.12952v3)

| # | Fact | Verbatim quote (location / URL) |
|---|---|---|
| A4.1 | **Who wrote the language, when, and under what instruction** | "Since we do not annotate trajectories with task names during data collection, we used a crowdsourcing platform to label the data post-hoc. Annotators were asked to describe the task being performed by the robot in each trajectory, with particular emphasis on the final location of any moved objects." (§3.2, p.4) |
| A4.2 | Collectors did not label | "We also do not require the data collector to label trajectories with task names." (§3.2, p.4) |
| A4.3 | Third-party help | "as well as Microsoft Research for assistance in labeling parts of the data with language." (Acknowledgements) |
| A4.4 | Cameras | "For sensing, we use an RGBD camera that is fixed in an over-the-shoulder view, two RGB cameras with poses that are randomized during data collection, and an RGB camera attached to the robot’s wrist. The images are saved at a 640x480 resolution and the control frequency is 5 Hz. We collect demonstrations by teleoperating the robot with a VR controller." (§3.1, p.4) |
| A4.5 | Re-randomisation | "Every 50 trajectories, the collector randomizes the poses of the cameras, switches out the objects in the scene, and randomizes the position of the workspace relative to the robot." (§3.2). Fig. 6: "“Over-the-shoulder” refers to the primary fixed camera, and “randomized” refers to the two alternative camera views that are randomized by the data collectors every 50 trajectories. … More cameras were added to the hardware setup throughout data collection, so the majority of the data only includes the primary fixed camera view, and very little data currently includes all 4 views." |
| A4.6 | Hardware | "We use an Intel RealSense D435 RGBD camera as a fixed over-the-shoulder camera view and two Logitech C920 webcams to capture alternative camera views … We used a Meta Quest 2 VR headset to teleoperate the robot." (App. C) |
| A4.7 | Action space, with no frame stated | "The 7D action space of the robot consists of continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose, as well as a discrete dimension to control the opening and closing of the gripper." (§4, p.5) |
| A4.8 | Scale | "In total, BridgeData V2 contains 50,365 expert demonstrations and 9,731 trajectories from a scripted policy." (§3.3). The website lists "60,096 trajectories" and adds "Each trajectory is labeled with a natural langauge instruction corresponding to the task the robot is performing." Its example instructions include "move the cloth to the left" (https://rail-berkeley.github.io/bridgedata/) |
| A4.9 | Action frame (robot code) | `bridge_data_robot` `transformation_utils.py` L144 says: "actions: [xyz deltas, and euler rotation angles, grasp_action], the rotation is around the position of the end-effector (axes are the same as world)" (https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_envs/utils/transformation_utils.py#L142-L158) |
| A4.10 | Released metadata misdescribes the action | Official TFDS `bridge_dataset/1.0.0/features.json`: the `action` has shape `[7]` but is described as "Robot action, consists of [7x joint velocities, 2x gripper velocities, 1x terminate episode]." The `state` has shape `[7]` but is described as "[7x robot joint angles, 2x gripper position, 1x door opening angle]". `image_0` to `image_3` are each described as "Main camera RGB observation." (https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/features.json) |
| A4.11 | OpenVLA's Bridge actions are achieved deltas | See A2.10. |
| A4.12 | Label quality, from later work | NILS (CoRL 2024): "alleviating several shortcomings of crowdsourced human annotations, such as low data quality and diversity". Footnote: "BridgeV2 dataset contains this annotation: ”https://www.youtube.com/watch?v=JWA5hJl4Dv0”". Cost: "approximately 200 USD, compared to about 5000 USD for crowd-sourced labels" with footnote "Based on the reported labeling cost of approximately 10, 000 USD for the full dataset." (arXiv 2410.17772v2, pp.1, 2, 6) |
| A4.13 | Relabelled Bridge language | Steerable VLA policies (RSS 2026): "This pipeline allows us to expand from 38k “standard” Bridge task-level language labels to 206k subtasks and nearly 2M total steering commands." (arXiv 2602.13193, p.4) |

### A5. Human direction labels are unreliable, so later work derives direction words automatically

- **RT-H (RSS 2024):**

  > "we found that having humans provide these labels offline leads to language inconsistency across the dataset and even inaccuracy in the labeled skills. For example, humans would often mislabel the transitions between skills, or misjudge the direction of motion of the robot due to camera angles."

  Their fix is "an automated labeling scheme relying on robot proprioception information" (arXiv 2403.01823v2, §III-C, p.4).
- **ECoT:** it uses fixed-camera proprioceptive templates (A3).

### A6. Flip-and-swap: sources showing that a mirrored image needs mirrored language (and mirrored actions)

| Source | What it does | Verbatim (URL) |
|---|---|---|
| MDETR code (ICCV 2021) | Swaps the words on flip | `caption = target["caption"].replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")` (https://github.com/ashkamath/mdetr/blob/ea09acc44ca067072c4b143b726447ee7ff66f5f/datasets/transforms.py#L61-L79) |
| MDETR code | **Disables** flips for referring expressions | `horizontal = [] if cautious else [T.RandomHorizontalFlip()]` (`datasets/coco.py` L212), and the referring-expression builder calls `make_coco_transforms(image_set, cautious=True)` (`datasets/refexp.py` L112) |
| TransVG code (ICCV 2021) | Swaps the words on flip | `text = text.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')` (https://github.com/djiajunustc/TransVG/blob/c8624277018fc9786a39181e229f155f933b2b59/datasets/transforms.py#L146-L156) |
| One-stage grounding (FAOA, ICCV 2019) and ReSC (ECCV 2020) code | Swaps the words on flip | "## random horizontal flip" … `phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')` (https://github.com/zyang-ur/onestage_grounding/blob/2fff0943c3bc4ad22fc0b9751c2c893c074dd5f2/dataset/referit_loader.py#L271-L275; ReSC `dataset/data_loader.py` L269–273) |
| RefTR (NeurIPS 2021), paper text | Removes flips | "We remove the random horizontal flip augmentation used in previous work [50] since we notice it causes semantic ambiguity on RefCOCO, likely due to relative location (e.g., left of/right of) specific queries in the dataset." (arXiv 2106.03089v2, p.6) |
| Augment the Pairs (WACV 2024), paper text | Shows a naive swap fails | "Due to the complex interplay of the word combinations (see Figure 1(b)), simply replacing word containing “left” with “right” or vice versa when applying horizontal flipping in phrase grounding task would introduce errors. One way to address this limitation is to skip the horizontal flipping if the caption contains “left” or “right”. We term this method as THflip." Abstract: "To guarantee image-caption correspondence in the training samples, we modify the captions according to pre-defined keywords when applying horizontal flipping." (arXiv 2311.02536v1, pp.1, 4) |
| MirrorDuo (CoRL 2025), robotics | Mirrors images *and* actions | "To ensure consistency with image mirroring (horizontal flip), we define an analogous mirroring operation in pose space." "we randomly sample m trajectories and mirror their images, proprioception, and actions". Caveat: "the above reflection symmetry formulations do not account for common sources of visual asymmetry under image mirroring, such as non-uniform table textures, background patterns, or asymmetries in the robot’s design. As shown in Fig. 3, the robot’s wrist appears left-sided in the mirrored view despite being consistently right-sided in the real world. Such discrepancies introduce visual out-of-distribution (OOD) artifacts." (arXiv 2606.20048v1, pp.2–4) |

Two more relevant points:

- Neither OpenVLA (A2.11) nor Octo (A3) uses flips, so VLAs trained on OXE have not seen mirrored scenes.
- We did not locate a paper that evaluates a VLA's language response on mirrored images. The arXiv searches returned nothing on point, but they were not exhaustive.

### A7. Quantising VLAs, 2024–2026: what each paper reports

| Paper (venue) | Model(s) | Reports success rate? | Reports action-level agreement with full precision? | Key verbatim |
|---|---|---|---|---|
| OpenVLA (CoRL 2024) | SigLIP-only OpenVLA variant | Yes (8 Bridge tasks) | Claims "comparable token accuracy" but gives no numbers | A1.7–A1.10 |
| QAIL (arXiv 2412.01034) | OpenVLA on LIBERO; also CILRS driving | Yes. OpenVLA LIBERO average: "Baseline Bfloat16 … 74.0%", "AWQ INT4 … 70.8%", "QAIL + QBC INT4 … 73.1%" (Table 3) | Partly. It reports attention divergence ("AttDiv") between quantised and FP policies, plus action accuracy during training | Fig. 1: "Differences in robot action between INT4 quantization and Bfloat16 in OpenVLA on LIBERO. Bfloat16 successfully places the mug inside the microwave and closes the door, whereas INT4 quantization fails in precise action" |
| SQIL (ICCV 2025) | OpenVLA on LIBERO; CILRS | Yes: "FP 73.8 ± 0.7 INT4 PTQ 70.7 ± 0.5 QAT 71.6 ± 0.6 SQIL 73.2 ± 0.6" | Qualitative. "Quantization errors generally have minor impact across most timesteps, producing small action discrepancies … Certain critical states, however, experience large deviations in actions due to quantization errors" | arXiv 2505.15304v2, pp.1–2 |
| BitVLA (arXiv 2506.07530) | 1.58-bit VLA; PTQ baselines | Yes. OpenVLA LIBERO average: INT8 PTQ 76.9 and INT4 PTQ 72.7 (Table II) | No | "We evaluate the publicly released fine-tuned checkpoints from Hugging Face and quantize the model backbones to INT8 and INT4 using bitsandbytes" |
| QVLA (ICLR 2026) | OpenVLA, OpenVLA-OFT | Yes | **Yes**: action error drives per-channel bit allocation and is reported | "naively applying uniform-bit quantization from Large Language Models (LLMs) to robotics is flawed, as these methods prioritize passive data fidelity while ignoring how minor action deviations compound into catastrophic task failures." (arXiv 2602.03782v1, p.1) |
| QuantVLA (CVPR 2026) | VLA PTQ | Yes | None found by keyword search of the arXiv text (no hits for action error, MSE, L1, deviation, fidelity or cosine) | arXiv 2602.20309 |
| HBVLA (arXiv 2602.13710) | 1-bit PTQ | Yes | Yes, as a loss: "we measure the action deviation induced by binarization" | "even small quantization-induced action deviations can be amplified by contact dynamics and accumulated over long-horizon rollouts" |
| DyQ-VLA (arXiv 2603.07904) | Dynamic quantisation | Yes | Yes. "Fig. 2. (a) Non-linear relationship between local action error and success rate." | — |
| VLAQuantBench (arXiv 2609.25376, Sept 2026) | OpenVLA-OFT, π0, π0.5, X-VLA | Yes (closed loop, 94,574 episodes) | **Yes**, in a paired-observation replay | "Alternative LLM quantizers on OpenVLA-OFT reach 98.0–99.0% success … On one held-out trajectory, AWQ reduces mean absolute action deviation relative to RTN by factors of 1.44 at W4 and 1.15 at W3 (Table 23). This comparison reveals action differences that the near-ceiling task scores cannot resolve" (§5.5). Also: "A small numerical perturbation can therefore have consequences that are not apparent from an individual forward pass." (p.1). Its §6 "Evaluation-Stack Checks" adds: "An activation-quantization run can silently become weight-only for a component if its input hooks are omitted. Missing AH hooks made OpenVLA-OFT appear much more tolerant in earlier internal runs." (p.12) |

The trend runs from success-only reporting (2024–25) towards action-level fidelity (2026). **None of these papers tests the *direction* (sign) of a single action slot, or a language contrast, across precisions.** That is the gap our precision axis fills.

### A8. Substrate papers the BtP community will recognise, including the organisers' work

- **Bronars, Park, Agrawal, "Tune to Learn" (RSS 2026). These are organisers, and the paper is directly on-theme.**
  - "Yet a critical design decision remains understudied: how should we choose controller gains for policy learning?"
  - "These findings reveal that optimal gain selection depends not on the desired task behavior, but on the learning paradigm employed."
  - **"While controller gains fundamentally shape the learning interface, their configuration in existing large-scale datasets remains largely undocumented. … Although exact gain values are rarely reported, tracking behavior reveals controller characteristics."**
  - Source: arXiv 2604.02523v1, pp.1, 3.
  - *Why it matters to us:* it is the same move we make for language. A hidden dataset convention is recovered by auditing the data (OXE/DROID tracking there, Bridge instruction roles here), and that convention changes the conclusion.
- **Park, Margolis, Agrawal, environment-shaping position paper (ICML 2024). Organisers.**
  - "Most practitioners don’t tune the RL algorithm, but other environment parameters to obtain a desirable controller."
  - The paper says shaping covers "designing observations, actions, rewards and simulation dynamics".
  - Source: arXiv 2407.16186v1, abstract.
  - *Relevance:* moderate. It supports "the choices that matter sit beneath the algorithm".
- **Park et al., DART/DexHub (arXiv 2411.02214; ICRA 2025 as "DART: Dexterous Augmented Reality Teleoperation Platform…"). Organisers.**
  - This is a data-collection-interface paper.
  - *Relevance:* weak for our argument (we do not collect data). Cite it only if we say "collection interfaces shape datasets"; otherwise omit.
- **Aljalbout et al. (RA-L 2024). Possible organiser.**
  - "We study the choice of action space in robot manipulation learning and sim-to-real transfer … highlight the need for careful consideration of action spaces when training and transferring RL agents for real-world robotics."
  - Source: arXiv 2312.03673v2 abstract.
- **Martín-Martín et al. (IROS 2019).**
  - "While many studies in RL focus on varying the observation space or reward model, few efforts focused on the choice of action space (e.g. joint or end-effector space, position, velocity, etc.)."
  - Source: arXiv 1906.08880v2 abstract.
- Other organiser papers I checked (Sha Yi: robot-hand generation and tactile work; Kehlani Fay: "House of Dextra" hand co-design) do not bear on our argument and are **not** included.

---

## (b) Synthesis: spatial language and robot-data language

**Two uses of one word.** Classical grounding work already separates the *referent* of a command from its *goal*.

- Tellex et al.'s G3 model builds its graph from Spatial Description Clauses, each mapped to "an object, place, path or event". Its sample corpus contains both uses: "Go to the first crate on the left" (object-selecting) and "place them on the trailer to the left" (destination).
- Guadarrama et al. parse "Move the cup close to the robot to the area in front of the plate and behind the tea box", where one relation picks the object and another the goal.
- Hatori et al. build the split into the architecture. They "Identify the target object by using an encoder model that takes the textual instruction and the image". They "Identify the destination box with a classifier that takes the textual instruction (but not image)".
- Venkatesh et al. map an instruction "to the start and end co-ordinates corresponding to the locations where the robot must pick up and place the object respectively".
- RoboPoint separates "relational object reference" from "free space reference".

Our probe only asks the object-selecting question ("the {noun} on the left"). Most robot-learning benchmarks never separate the two roles, so a verdict about "spatial language" depends on which role was tested.

**Datasets decide which role the model sees.** Referring-expression corpora collect object-selecting uses on purpose. RefCOCO images "were selected to contain two or more objects of the same object category", and Yu et al. note that people say "the man on the left" for one object and "the man on the right" for the other. RefCOCO+ bans location words to force appearance-based descriptions "independent of viewer perspective".

BridgeData V2's protocol points the other way. Crowd workers labelled trajectories after the fact, "with particular emphasis on the final location of any moved objects". The project page's own example is "move the cloth to the left". Our count (3,333 of 3,574 lateral instructions name a destination; none chooses between identical objects) is what that instruction predicts.

The closest robot precedent is DIAL. On novel spatial instructions with duplicate objects ("knock down the right soda"), baselines trained on templated teleoperator commands "ignore the language instruction and instead repeat the same motions or randomly select a target object". Relabelling the data with a fine-tuned CLIP fixed this. More recent work makes the same diagnosis at scale:

- CAST blames "a lack of fine-grained task diversity for similar observations".
- LIBERO-CF finds VLAs "selecting objects frequently seen during training regardless of language intent".
- Xing et al. tie shortcut learning to the structure of pooled datasets.

The general point, as in Kamath et al.'s audit of LAION ("left of" and similar prepositions appear in only 0.2% of captions), is this: a model can only learn the uses of a spatial word that its corpus contains.

**Frames of reference make the same word mean different things.** Levinson's typology reduces spatial frames to "intrinsic, relative, and absolute" (as summarised in VSR). VSR notes that "intrinsic and relative frames are widely used, and present an important source of variation". VSR's annotators were told to accept "left"/"right" under *either* frame, which is a protocol decision that changes what the labels mean.

COMFORT shows that VLMs "lack the flexibility to accommodate multiple FoRs" while partly following English conventions. In human–robot instruction, Li et al. find "only about 42% of instructions contain perspective-independent spatial references". INGRESS has to handle "object-centric", "user-centric" and "robot-centric" expressions with a keyword-and-viewpoint transform. RoboSpatial poses every question from "Ego-centric", "World-centric" and "Object-centric" frames.

In Bridge, the fixed over-the-shoulder camera makes image-left and robot-left roughly coincide, so a relative frame is implied but never stated. The convention we rely on (image-right is negative dy) appears in nobody's documentation. ECoT's labelling code encodes it independently ("right" for −y, "left" for +y, "assuming a fixed camera").

**Direction words are themselves noisy annotations.** RT-H reports that human labellers "misjudge the direction of motion of the robot due to camera angles". It therefore derives "language motions" from proprioception; ECoT does the same. NILS documents low-quality Bridge crowd labels, including a YouTube URL used as an instruction. Steerable VLA policies replace Bridge's 38k task labels with about 2M synthetic commands. When several identical objects are present, it says language is "underspecified" and uses pointing instead.

So the language conventions of robot data are a design choice at three levels:

- which role a spatial word plays (protocol);
- whose frame it uses (camera placement and annotator instructions);
- how its direction is decided (human versus proprioceptive labelling).

None of the three is usually reported. This is exactly the kind of reporting gap BtP targets: the "Tune to Learn" authors make the same point about controller gains, which "remain largely undocumented" in large datasets.

**Mirroring tests the frame, but only if words and actions are mirrored together.** Vision-language grounding codebases handle flips in one of two ways. Some swap the words (MDETR, TransVG and the one-stage grounding code all do `replace('right', …)`). Others disable flips for referring expressions (MDETR's `cautious=True`; RefTR: flips cause "semantic ambiguity on RefCOCO, likely due to relative location (e.g., left of/right of) specific queries").

Yi et al. warn that naive string swaps "would introduce errors" and propose skipping such captions. In robotics, MirrorDuo mirrors "images, proprioception, and actions" together. It also cautions that mirrored robot images are visually out of distribution ("the robot’s wrist appears left-sided in the mirrored view").

For a VLA language probe, a mirrored scene therefore needs three changes to be a valid control:

1. swap which twin the word names;
2. negate the lateral action;
3. accept that the robot itself looks mirrored, a scene OpenVLA never saw because neither OpenVLA nor Octo trains with flips.

Each of these is a reporting item.

---

## (c) Annotated bibliography

Paper-section tags used in the "Where" column:

| Tag | Section |
|---|---|
| [Intro] | Introduction |
| [Probe] | Model-organism setup |
| [Slot] | Choice: lateral slot and sign |
| [Readout] | Choice: argmax vs expected value, bin width |
| [Map] | Choice: token-to-bin map |
| [Prec] | Choice: precision |
| [Prefix] | Choice: autoregressive prefix and decoding |
| [Mirror] | Choice: mirrored-image scoring |
| [Wording] | Choice: instruction wording |
| [Data] | Data conventions |
| [Card] | Reporting card |
| [RW] | Related work |

| Key | Citation | URL | What it shows | Where | Suggested sentence |
|---|---|---|---|---|---|
| kim2024openvla | Kim et al., OpenVLA, CoRL 2024 (PMLR 270) | https://proceedings.mlr.press/v270/kim25c.html | 256 action tokens over q01–q99, last 256 Llama tokens overwritten, bf16 default, 4-bit ≈ bf16 on 8 Bridge tasks | [Probe][Map][Prec] | "OpenVLA maps each action dimension to the last 256 Llama tokens, binned between the 1st and 99th training quantiles [kim2024openvla]." |
| brohan2023rt1 | Brohan et al., RT-1, RSS 2023 | https://www.roboticsproceedings.org/rss19/p025.html | Origin of 256-bin per-dimension binning (min–max bounds) | [Map][RW] | "Per-dimension 256-bin tokenisation dates to RT-1 [brohan2023rt1]." |
| brohan2023rt2 | Brohan et al., RT-2, arXiv 2307.15818 (author order as on arXiv) | https://arxiv.org/abs/2307.15818 | Actions as text tokens; overwrites least-frequent tokens; **constrains decoding to valid action tokens** | [Prefix][Map] | "RT-2 restricts decoding to valid action tokens [brohan2023rt2]; OpenVLA's `predict_action` does not." |
| zitkovich2023rt2 | Zitkovich et al., RT-2, CoRL 2023 (PMLR 229) | https://proceedings.mlr.press/v229/zitkovich23a.html | Same paper, venue version (PMLR lists Zitkovich first) | — | Use instead of brohan2023rt2 if citing the venue. Cite only one of the two. |
| oneill2024open | Open X-Embodiment Collaboration (O'Neill et al.), ICRA 2024 | https://doi.org/10.1109/ICRA57147.2024.10611477 | **Coordinate frames not aligned across datasets**; the same action vector moves robots differently | [Slot][Data] | "Open X-Embodiment pools datasets without aligning end-effector frames [oneill2024open], so the sign of a lateral slot is a per-dataset convention." |
| ghosh2024octo | Ghosh et al., Octo, RSS 2024 | https://www.roboticsproceedings.org/rss20/p090.html | Diffusion head; only the gripper sign aligned across datasets; crop and colour augmentation, no flip | [Slot][Mirror] | "Octo aligns only the gripper convention across datasets [ghosh2024octo]." |
| pertsch2025fast | Pertsch et al., FAST, RSS 2025 | https://www.roboticsproceedings.org/rss21/p012.html | Naive per-dimension binning fails at high frequency; OpenVLA struggled on DROID | [Map][RW] | "Per-dimension binning is a consequential choice: it degrades sharply on high-frequency data [pertsch2025fast]." |
| kim2025finetuning | Kim, Finn, Liang, OpenVLA-OFT, RSS 2025 | https://www.roboticsproceedings.org/rss21/p017.html | Names decoding scheme and action representation as design choices; OpenVLA is autoregressive with 7 sequential passes | [Prefix] | "OpenVLA generates its seven action tokens autoregressively [kim2025finetuning], so the lateral token is read conditional on the preceding dx token." |
| black2025pi0 | Black et al., π0, RSS 2025 | https://www.roboticsproceedings.org/rss21/p010.html | Flow-matching continuous actions as an alternative to autoregressive discretisation | [RW] | "Flow-based VLAs avoid token readouts altogether [black2025pi0]." |
| black2025pi05 | Black et al., π0.5, CoRL 2025 (PMLR 305) | https://proceedings.mlr.press/v305/black25a.html | Trains on FAST tokens and decodes with flow; discrete and continuous paths coexist | [RW][Card] | "Even within one model, actions may have both a token and a continuous readout [black2025pi05]." |
| li2024cogact | Li et al., CogACT, arXiv 2024 | https://arxiv.org/abs/2411.19650 | Binning "limits action precision" | [Readout] | "Bin quantisation limits action precision [li2024cogact]." |
| qu2025spatialvla | Qu et al., SpatialVLA, RSS 2025 | https://www.roboticsproceedings.org/rss21/p011.html | Non-uniform, Gaussian-fit action grids | [Map] | "The bin geometry itself varies across VLAs [qu2025spatialvla]." |
| belkhale2024minivla | Belkhale & Sadigh, MiniVLA (blog) | https://ai.stanford.edu/blog/minivla/ | Describes OpenVLA binning; VQ action chunks | [Map] | Optional; it is a blog post. |
| wang2025vqvla | Wang et al., VQ-VLA, ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.html | Learned VQ action tokeniser | [RW] | "…and learned vector-quantised tokenisers [wang2025vqvla]." |
| liu2026oat | Liu et al., OAT, RSS 2026 | https://www.roboticsproceedings.org/rss22/p075.html | Token ordering and decodability as tokeniser desiderata | [Prefix][RW] | "Token ordering is now treated as a design axis [liu2026oat]." |
| goyal2025vla0 | Goyal et al., VLA-0, arXiv 2025 | https://arxiv.org/abs/2510.13054 | Actions written as plain text | [Map] | "Some VLAs emit actions as ordinary digits [goyal2025vla0], which makes the token-to-value map different again." |
| zawalski2024robotic | Zawalski et al., ECoT, CoRL 2024 (PMLR 270) | https://proceedings.mlr.press/v270/zawalski25a.html | Derives "move left/right" from Bridge proprioception "assuming a fixed camera"; code maps −y to "right" | [Slot] | "This sign convention matches ECoT's Bridge labelling code, which names negative y motion 'right' [zawalski2024robotic]." |
| walke2023bridgedata | Walke et al., BridgeData V2, CoRL 2023 (PMLR 229) | https://proceedings.mlr.press/v229/walke23a.html | Post-hoc crowd labels with "emphasis on the final location of any moved objects"; fixed over-the-shoulder camera plus randomised cameras; 5 Hz; relative-pose actions | [Data][Slot] | "Bridge's crowd annotators were asked to emphasise 'the final location of any moved objects' [walke2023bridgedata], so 'left' and 'right' mostly name destinations." |
| xiao2023robotic | Xiao et al., DIAL, RSS 2023 | https://www.roboticsproceedings.org/rss19/p029.html | Duplicate-object spatial tasks; baselines ignore language; relabelling helps | [Data][RW] | "With duplicate objects, policies trained on templated commands ignore 'left/right' unless the data is relabelled [xiao2023robotic]." |
| blank2024scaling | Blank et al., NILS, CoRL 2024 (PMLR 270) | https://proceedings.mlr.press/v270/blank25a.html | Crowd labels low in quality and diversity; Bridge contains a URL as an "instruction" | [Data] | "Bridge's crowd labels are noisy [blank2024scaling]." |
| belkhale2024rth | Belkhale et al., RT-H, RSS 2024 | https://www.roboticsproceedings.org/rss20/p049.html | Humans "misjudge the direction of motion of the robot due to camera angles" | [Data][Slot] | "Direction words written by humans are unreliable because of camera viewpoint [belkhale2024rth]." |
| chen2026steerable | Chen et al., Steerable VLA Policies, RSS 2026 | https://www.roboticsproceedings.org/rss22/p074.html | Bridge relabelled from 38k to about 2M commands; with multiple instances, "language underspecified", so they use pointing | [Data][RW] | "Recent relabelling replaces Bridge's 38k labels with millions of synthetic commands [chen2026steerable]." |
| glossop2025cast | Glossop et al., CAST, arXiv 2025 | https://arxiv.org/abs/2508.13446 | VLAs miss fine-grained commands due to "lack of fine-grained task diversity for similar observations" | [Data] | "Missing contrastive language in robot data limits instruction following [glossop2025cast]." |
| fang2026when | Fang et al., When Vision Overrides Language, arXiv 2026 | https://arxiv.org/abs/2602.17659 | Vision shortcuts "induced by dataset biases" override language | [Data][RW] | "VLAs fall back on dataset-induced visual shortcuts [fang2026when]." |
| xing2025shortcut | Xing et al., Shortcut Learning in Generalist Robot Policies, CoRL 2025 (PMLR 305) | https://proceedings.mlr.press/v305/xing25a.html | Dataset diversity and fragmentation cause shortcuts | [Data] | — |
| lynch2021language | Lynch & Sermanet, RSS 2021 | https://www.roboticsproceedings.org/rss17/p047.html | Hindsight crowd labelling protocol ("what instruction would you give the agent to get from first frame to last frame?") | [Data] | "Post-hoc crowd labelling of robot data has a long history [lynch2021language]; its prompt shapes the language." |
| dettmers2023qlora | Dettmers et al., QLoRA, NeurIPS 2023 | https://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html | Defines NF4 | [Prec] | "We use NF4 4-bit weights [dettmers2023qlora] with bf16 compute." (Set compute dtype to whatever was actually used.) |
| dettmers2022llmint8 | Dettmers et al., GPT3.int8(), NeurIPS 2022 | https://papers.nips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html | 8-bit inference | [Prec] | — |
| park2024quantization | Park et al., QAIL, arXiv 2024 | https://arxiv.org/abs/2412.01034 | INT4 PTQ lowers OpenVLA LIBERO success (74.0 → 70.8 with AWQ) | [Prec] | "Post-training INT4 costs fine-tuned OpenVLA a few points of LIBERO success [park2024quantization]…" |
| park2025saliency | Park et al., SQIL, ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html | Small action discrepancies at most steps, large ones at critical states | [Prec] | "…with errors concentrated in a few critical states [park2025saliency]." |
| wang2025bitvla | Wang et al., BitVLA, arXiv 2025 | https://arxiv.org/abs/2506.07530 | bitsandbytes INT8/INT4 PTQ of OpenVLA: 76.9 / 72.7 LIBERO average | [Prec] | — |
| xu2026qvla | Xu et al., QVLA, ICLR 2026 | https://openreview.net/forum?id=TpL2nXanru | Action-error-centric quantisation | [Prec] | "Recent work argues quantisation should be judged in action space [xu2026qvla]." |
| zhang2026quantvla | Zhang et al., QuantVLA, CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_QuantVLA_Scale-Calibrated_Post-Training_Quantization_for_Vision-Language-Action_Models_CVPR_2026_paper.html | PTQ for VLAs (success-rate evaluation) | [Prec] | — |
| yan2026hbvla | Yan et al., HBVLA, arXiv 2026 | https://arxiv.org/abs/2602.13710 | Action-deviation loss for 1-bit PTQ | [Prec] | — |
| zheng2026dyqvla | Zheng et al., DyQ-VLA, arXiv 2026 | https://arxiv.org/abs/2603.07904 | Non-linear link between local action error and success | [Prec][Card] | "Local action error and task success are related non-linearly [zheng2026dyqvla]." |
| xu2026vlaquantbench | Xu et al., VLAQuantBench, arXiv 2026 | https://arxiv.org/abs/2609.25376 | Near-ceiling success hides action deviations; paired-observation replay | [Prec][Card] | "Success rates near ceiling cannot resolve action-level differences between precisions [xu2026vlaquantbench]; we therefore compare the lateral action itself." |
| kamath2021mdetr | Kamath et al., MDETR, ICCV 2021 | https://openaccess.thecvf.com/content/ICCV2021/html/Kamath_MDETR_-_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.html | Code swaps left/right on flip, and disables flips for referring expressions | [Mirror] | "Grounding codebases either swap 'left'/'right' when flipping (e.g., MDETR, TransVG) or disable flips for referring expressions [kamath2021mdetr, deng2021transvg]." |
| deng2021transvg | Deng et al., TransVG, ICCV 2021 | https://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html | Code swaps left/right on flip | [Mirror] | (see above) |
| yang2019fast | Yang et al., One-stage grounding, ICCV 2019 | https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_A_Fast_and_Accurate_One-Stage_Approach_to_Visual_Grounding_ICCV_2019_paper.html | Code swaps left/right on flip (the earliest swap located) | [Mirror] | — |
| yang2020improving | Yang et al., ReSC, ECCV 2020 | https://doi.org/10.1007/978-3-030-58568-6_23 | Same swap in code | [Mirror] | — |
| li2021referring | Li & Sigal, RefTR, NeurIPS 2021 | https://papers.nips.cc/paper_files/paper/2021/hash/a376802c0811f1b9088828288eb0d3f0-Abstract.html | Removes flips because of left-of/right-of queries | [Mirror] | "RefTR removes flip augmentation because relative-location queries become ambiguous [li2021referring]." |
| yi2024augment | Yi et al., Augment the Pairs, WACV 2024 | https://openaccess.thecvf.com/content/WACV2024/html/Yi_Augment_the_Pairs_Semantics-Preserving_Image-Caption_Pair_Augmentation_for_Grounding-Based_Vision_WACV_2024_paper.html | A naive word swap introduces errors; text-conditioned flipping instead | [Mirror] | "Even word swapping is error-prone [yi2024augment]; we therefore swap target identities, not strings." |
| zhuang2025mirrorduo | Zhuang et al., MirrorDuo, CoRL 2025 (PMLR 305) | https://proceedings.mlr.press/v305/zhuang25a.html | Mirrors images, proprioception and actions; mirrored robot images are OOD | [Mirror] | "Mirroring a manipulation scene requires mirroring the actions [zhuang2025mirrorduo], and mirrored robot images are themselves out of distribution." |
| levinson2003space | Levinson, *Space in Language and Cognition*, CUP 2003 | https://doi.org/10.1017/CBO9780511613609 | Typology of spatial frames of reference | [Wording][RW] | "Spatial terms are interpreted within a frame of reference [levinson2003space]." |
| tellex2011understanding | Tellex et al., G3, AAAI 2011 | https://doi.org/10.1609/aaai.v25i1.7979 | Grounds objects, places and paths separately; corpus contains both uses of "left" | [Wording][RW] | "Grounding models have long distinguished objects from places [tellex2011understanding]." |
| guadarrama2013grounding | Guadarrama et al., IROS 2013 | https://doi.org/10.1109/IROS.2013.6696569 | Spatial prepositions for both object reference and goal | [RW] | — |
| shridhar2018interactive | Shridhar & Hsu, INGRESS, RSS 2018 | https://www.roboticsproceedings.org/rss14/p28.html | Self-referential vs relational expressions; perspective correction (object-, user-, robot-centric) | [Wording][RW] | — |
| hatori2018interactively | Hatori et al., ICRA 2018 | https://doi.org/10.1109/ICRA.2018.8460699 | Target object (image + text) vs destination box (text only) | [Wording] | "Earlier systems split the object-selecting and destination roles architecturally [hatori2018interactively, venkatesh2021spatial]." |
| venkatesh2021spatial | Venkatesh et al., ICRA 2021 | https://doi.org/10.1109/ICRA48506.2021.9560895 | Language mapped to pick (start) and place (end) coordinates | [Wording] | (see above) |
| li2016spatial | Li et al., RO-MAN 2016 | https://doi.org/10.1109/ROMAN.2016.7745089 | "only about 42% of instructions contain perspective-independent spatial references" (abstract) | [Wording] | — |
| kamath2023whats | Kamath et al., What's "up", EMNLP 2023 | https://aclanthology.org/2023.emnlp-main.568/ | VLMs fail left/right; LAION captions contain such prepositions "only 0.2% of the time" | [Data][RW] | "Like web corpora for VLMs [kamath2023whats], robot corpora determine which spatial uses are learnable." |
| liu2023visual | Liu et al., VSR, TACL 2023 | https://aclanthology.org/2023.tacl-1.37/ | Summarises Levinson's three frames; annotators accept either frame | [Wording] | — |
| zhang2025comfort | Zhang et al., COMFORT, ICLR 2025 | https://openreview.net/forum?id=84pDoCD4lH | FoR ambiguity; VLMs inflexible across FoRs | [Wording][RW] | "VLMs follow some English frame conventions but not others [zhang2025comfort]." |
| chen2024spatialvlm | Chen et al., SpatialVLM, CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/html/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.html | Attributes weak spatial reasoning to training data | [RW] | — |
| song2025robospatial | Song et al., RoboSpatial, CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/html/Song_RoboSpatial_Teaching_Spatial_Understanding_to_2D_and_3D_Vision-Language_Models_CVPR_2025_paper.html | Ego-, world- and object-centric frames for robotics | [Wording] | — |
| yuan2024robopoint | Yuan et al., RoboPoint, CoRL 2024 (PMLR 270) | https://proceedings.mlr.press/v270/yuan25c.html | Object reference vs free-space reference; relations computed "from the camera's perspective" | [Wording] | — |
| kazemzadeh2014referitgame | Kazemzadeh et al., ReferItGame, EMNLP 2014 | https://aclanthology.org/D14-1086/ | Two-player game behind RefCOCO and RefCOCO+ | [RW] | — |
| yu2016modeling | Yu et al., ECCV 2016 | https://doi.org/10.1007/978-3-319-46475-6_5 | RefCOCO scenes contain two or more same-category objects; RefCOCO+ bans location words | [Data][RW] | "Referring-expression datasets are built around same-category distractors [yu2016modeling]; Bridge is not." |
| mao2016generation | Mao et al., RefCOCOg, CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/html/Mao_Generation_and_Comprehension_CVPR_2016_paper.html | Non-interactive referring expressions | [RW] | — |
| bronars2026tune | Bronars, Park, Agrawal, Tune to Learn, RSS 2026 | https://www.roboticsproceedings.org/rss22/p139.html | Gains change what is learnable; undocumented in datasets | [Intro][Card] | "Like controller gains, which remain 'largely undocumented' in large datasets [bronars2026tune], the language conventions of robot data are rarely reported." |
| park2024automatic | Park, Margolis, Agrawal, ICML 2024 (position) | https://proceedings.mlr.press/v235/park24i.html | Practitioners tune environments, not algorithms | [Intro] | Optional. |
| park2025dart | Park et al., DART, ICRA 2025 (arXiv: "DexHub and DART") | https://doi.org/10.1109/ICRA55743.2025.11128299 | Collection-interface design | [RW] | Optional; weak fit. |
| aljalbout2024role | Aljalbout et al., RA-L 2024 | https://doi.org/10.1109/LRA.2024.3398428 | Action-space choice changes learning and transfer | [Intro][RW] | "Action-space design alone changes learning outcomes [aljalbout2024role, martinmartin2019variable]." |
| martinmartin2019variable | Martín-Martín et al., IROS 2019 | https://doi.org/10.1109/IROS40897.2019.8968201 | Few studies examine action-space choice | [Intro][RW] | (see above) |

**Code citations that are not bib entries.** Cite these in footnotes or the appendix with their permalinks:

- OpenVLA `action_tokenizer.py`;
- Hugging Face `modeling_prismatic.py`, `config.json` and `generation_config.json`;
- OpenVLA `data_utils.py`, `transforms.py`, `configs.py`, `datasets.py`, `openvla_utils.py` and `run_bridgev2_eval.py`;
- transformers v4.40.1 `quantization_config.py` and `modeling_utils.py`;
- ECoT `primitive_movements.py`;
- bridge_data_robot `transformation_utils.py`;
- the Bridge TFDS `features.json`;
- MDETR, TransVG and onestage_grounding/ReSC transforms.

---

## (d) Unverified, or verified only partially (kept out of the .bib unless noted)

**Verified only partially:**

1. **The 4-bit configuration behind OpenVLA Table 2.** The paper does not state FP4 or NF4, or the compute dtype. "Official code defaults to FP4" is our reading of the released script plus transformers 4.40.1 defaults. It is not a statement by the authors.
2. **Which camera `image_0` is.** We assume the Bridge TFDS `image_0` is the fixed over-the-shoulder camera. The metadata describes all four images as "Main camera RGB observation", and the Bridge paper does not name the TFDS keys.
3. **The orientation of Bridge's base frame.** The axes of the WidowX "world"/base frame (x forward, y left, z up) are not documented in the Bridge paper or TFDS metadata. Our "image-right = −dy" comes from our own data. It is consistent with ECoT's code naming, but no primary source states it.
4. **The source of NILS's "approximately 10,000 USD" figure.** NILS says it is "reported" but does not say where.
5. **Abstract-level verification only:**
   - Li et al. 2016 (RO-MAN): metadata via Crossref; abstract via the Semantic Scholar API; full text not read.
   - Guadarrama et al. 2013 (IROS): same as Li et al. 2016.
   - Levinson 2003: metadata and book description from Cambridge Core; the three-frame summary quoted from VSR and COMFORT, not from the book.
   - These entries are **in** the .bib because their metadata is verified, but do not quote them beyond their abstracts or descriptions.
6. **Peer-review status.** The following are in the .bib as preprints, so cite them as preprints:
   - MiniVLA (blog);
   - CogACT, VLA-0, CAST, "When Vision Overrides Language", QAIL, BitVLA, HBVLA, DyQ-VLA and VLAQuantBench (all arXiv only).

**Not verified and not used:**

7. A claim on the Open X-Embodiment website that RT-2-X responds to preposition changes ("on" vs "near"). It is not in the arXiv v9 text.
8. Any VQA paper that disables flips because of left/right. None was located; the referring-expression sources above stand in.
9. Any paper that evaluates a VLA's language response on mirrored images, or that counts destination-vs-object-selecting uses of "left/right" in robot datasets. None was found in arXiv searches, so our analyses appear to be new. Treat this as "to our knowledge".
10. "Interactive Language" (Lynch et al. 2022, arXiv 2210.06407) and Mittal et al. 2024 (symmetry, ICRA 2024). Their IDs were found but they were not read, so they are excluded.
