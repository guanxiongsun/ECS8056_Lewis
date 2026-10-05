# B7-lite: desk audit of measurement-choice reporting in ten VLA language evaluations

**Status: preliminary.** One coder did this audit: an AI agent working from the authors' own review list (lit_vla.md §(b)). It must be double-coded by humans, and agreement reported, before any number here goes into the paper. The codes measure what is *written in the papers' full texts*, not what the authors did. Code repositories, project pages and supplementary files were not consulted.

Coded 3 Oct 2026 from the arXiv HTML full texts (main text and appendices) fetched on 2 Oct 2026.

---

## 1. Frozen sample, sources and rules

**Sample (frozen before coding):** the ten works in the verified comparison table of lit_vla.md §(b).

| Short name | Paper | arXiv version coded |
|---|---|---|
| STAGE | Jin, Liang, Shen. STAGE: Diagnosing Semantic Transfer at Grounded Execution in Embodied Agents | 2609.13458v1 |
| LIBERO-CF | Fang et al. When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs | 2602.17659v2 |
| BeTTER | Xu et al. Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models | 2604.18000v1 |
| LangGap | Hou & Zhao. LangGap: Diagnosing and Closing the Language Gap in VLA Models | 2603.00592v1 |
| InstructMove | Zhao et al. InstructMove: A Text-Indispensable Benchmark for Instruction-Following Manipulation | 2608.22990v1 |
| RefGuard | Wei et al. RefGuard: Identity-Aware Language-Guided Robot Manipulation via Joint Target-Anchor-Frame Grounding | 2609.06221v1 |
| RoboIRGBench | Akelijiang et al. RoboIRGBench: Benchmarking Implicit Referential Grounding in VLA Models | 2609.34384v1 |
| Act2Answer | Kachaev et al. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in VLA Models | 2606.19297v1 |
| Kirouane | Kirouane, Giaples, Petrocheilos. Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 VLA Policy | 2609.07470v1 |
| Chen | Chen et al. When Instructions Retrieve Trajectories: Diagnosing and Mitigating Generalization Failures in VLA Models | 2609.39971v1 |

**Codes.** R = reported (quote or location below); P = partly (what is missing is stated); N = not reported; N/A = not applicable.

**N/A rules (decided before coding):**
- (given) Items 1–4 are N/A for papers that read no action values and score only closed-loop outcomes (success, contact, placement region, behaviour labels). Hidden-state probes are not action values.
- (given) Item 2 is N/A for continuous-action models without a token map (flow matching, diffusion, regression heads). If a paper reads both token and continuous models, item 2 applies to the token model.
- (added) Item 4 is N/A when every model whose actions are read generates all action dimensions jointly (non-autoregressive heads), because there is no within-step decoding order.
- (added) Item 10 is coded N, not N/A, when no screening is described: a missing description cannot be told apart from no screening.
- Applied to the sample: items 1–4 apply only to **STAGE** (OpenVLA single-step actions; Octo) and **Chen** (decoded π0.5 action chunks in the prefix-KV patching analysis, App. E). For Chen, items 2 and 4 are N/A because π0.5 and GR00T-N1.7 have continuous, jointly generated action heads. For **Kirouane**, items 1–4 are N/A: their training-loss instrument is computed from action predictions, but it is reported only as a rejected instrument, and every verdict rests on closed-loop success.

**R/P thresholds (frozen before coding):**

| # | Item | R requires | P |
|---|---|---|---|
| 1 | Action axis, sign, zero, units | dims read + sign convention + zero + units, frame stated relative to image/camera | some of these |
| 2 | Token↔bin map | mapping, bin grid, edge handling | some |
| 3 | Readout rule | rule (argmax / expected value / sampled; or fixed noise for flow models) AND ties or logging of distributions | rule only, or clearly implied |
| 4 | Decoding order | whether the probed dimension is read after the model's own earlier dims, or forced/marginalised | partial mention |
| 5 | Horizon and control rate | concrete horizon (single step / chunk length / episode step or time limit) AND control rate | one of the two |
| 6 | Precision | inference dtype/quantisation AND evaluation GPU AND library versions | any of these (training-only counts) |
| 7 | Observations | image source AND resolution AND preprocessing relative to training (+ edits/transforms and their scoring, if used) | some |
| 8 | Instructions | templates (or full list) AND their number AND text normalisation (case) | templates/examples and/or number, no normalisation |
| 9 | Training-data use of probed words | counts, frequency, or seen/unseen status per word or instruction | qualitative or coarse |
| 10 | Stimulus/scene screening | criteria AND numbers (approval/exclusion) or annotator agreement | criteria only |
| 11 | Decisive statistic | (a) direction-aware AND (b) unit of analysis + clustering/dependence addressed AND ≥1 of (c) multiplicity, (d) positive control | ≥1 of (a)–(d) |
| 12 | Sensitivity to analysis choices | main verdict recomputed under ≥1 alternative analysis/measurement choice, with stability reported | alternative shown without a stability statement, or robustness to non-analysis factors (seeds) only |

**Clarifications adopted during coding** (not in the frozen protocol; flagged for the human coders):
- (i) When a detail is reported for some but not all evaluated models, the item is coded P. This affected RefGuard items 5 and 7, InstructMove item 7, and RoboIRGBench item 9.
- (ii) A GPU count without a GPU model does not count as "GPU" for item 6 (BeTTER).
- (iii) A library named without a version does not count for item 6 (Chen, "JAX").

---

## 2. Codes (10 papers × 12 items)

| Paper | 1 Axis/sign | 2 Token map | 3 Readout | 4 Order | 5 Horizon/rate | 6 Precision | 7 Observations | 8 Instructions | 9 Training words | 10 Screening | 11 Statistic | 12 Sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STAGE | P | N | N | N | P | N | P | P | N | P | P | R |
| LIBERO-CF | N/A | N/A | N/A | N/A | N | P | P | P | R | P | P | N |
| BeTTER | N/A | N/A | N/A | N/A | N | N | N | P | P | P | P | N |
| LangGap | N/A | N/A | N/A | N/A | N | P | N | P | R | P | P | N |
| InstructMove | N/A | N/A | N/A | N/A | P | P | P | P | P | P | P | N |
| RefGuard | N/A | N/A | N/A | N/A | P | P | P | P | P | P | P | N |
| RoboIRGBench | N/A | N/A | N/A | N/A | P | N | P | P | P | R | P | N |
| Act2Answer | N/A | N/A | N/A | N/A | N | N | P | P | P | R | P | R |
| Kirouane | N/A | N/A | N/A | N/A | R | P | P | P | R | P | R | R |
| Chen | P | N/A | N | N/A | P | N | P | P | R | P | R | R |

## 3. Per-item summary (N/A excluded)

| # | Item | papers coded | R | P | N |
|---|---|---|---|---|---|
| 1 | Action axis, sign, zero, units | 2 | 0 | 2 | 0 |
| 2 | Token↔bin map | 1 | 0 | 0 | 1 |
| 3 | Readout rule | 2 | 0 | 0 | 2 |
| 4 | Decoding order | 1 | 0 | 0 | 1 |
| 5 | Horizon and control rate | 10 | 1 | 5 | 4 |
| 6 | Precision | 10 | 0 | 5 | 5 |
| 7 | Observations | 10 | 0 | 8 | 2 |
| 8 | Instruction templates, number, normalisation | 10 | 0 | 10 | 0 |
| 9 | Training-data use of probed words | 10 | 4 | 5 | 1 |
| 10 | Stimulus/scene screening | 10 | 2 | 8 | 0 |
| 11 | Decisive statistic | 10 | 2 | 8 | 0 |
| 12 | Sensitivity to analysis choices | 10 | 4 | 0 | 6 |
| | **All applicable cells** | **86** | **13** | **51** | **22** |

**Item 11 sub-criteria** (a direction-aware / b unit + clustering / c multiplicity / d positive control):

| | STAGE | LIBERO-CF | BeTTER | LangGap | InstructMove | RefGuard | RoboIRGBench | Act2Answer | Kirouane | Chen |
|---|---|---|---|---|---|---|---|---|---|---|
| a | no (headline statistic direction-blind; ActCheck secondary) | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| b | unit only | unit only | no | unit only | unit only | unit only | no | yes | yes | yes |
| c | no | no | no | no | no | no | no | no | yes (listed, Bonferroni discussed) | no |
| d | yes | no | no | no | no | yes (S1) | yes (explicit counterparts) | no | yes (English) | yes (scripted oracle) |

**What stands out:**
- Eight of ten papers read no action values at all, so they cannot report items 1–4.
- Of the two that do read action values, neither states its readout rule. The one that reads a token model (STAGE, OpenVLA) reports neither the token map nor the decoding order.
- No paper reports inference-time numerical precision: every P on item 6 is a training GPU or training dtype.
- No paper states text normalisation, so item 8 is P for all ten.
- Only two papers, both from September 2026, meet the item-11 bar.
- Six papers report no check of whether their verdict survives a different analysis choice.

---

## 4. Evidence for every R and P

Quotes are verbatim from the HTML full text; "§" locations follow each paper's own numbering.

### STAGE (2609.13458v1)
- **1 P.** Names the action ("the policy’s native seven-degree-of-freedom (7-DoF) action", §1) and uses "v=a_{1:3} the translational action", compared with simulator position differences p_t − p_e for ActCheck (§3.3). *Missing:* frame relative to the image or camera, sign convention, zero, units. Actions are normalised by "the per-dimension action standard deviation" (§3.3), with no units.
- **2 N.** No token-to-bin description for OpenVLA.
- **3 N.** Only "the decoded native continuous action" (App. G.3); no argmax/sampling/noise rule.
- **4 N.**
- **5 P.** Horizon: "One-step sensitivity" (§4.2); rollouts "of K=20 steps" (§4.2), also K=50/100 (§6.2). *Missing:* control rate.
- **6 N.**
- **7 P.** Sources: LIBERO observations; "We construct targeted BridgeData V2 real-robot diagnostics from the first public TFDS shard ... 165 sampled frames" (App. K). *Missing:* resolution, preprocessing relative to training, frame selection rule.
- **8 P.** Intervention families with examples and split sizes (Table 14: e.g. "Target-name swap 600", "Templated invalid instruction 1,800"); example relation instruction "pick up the object to the right of the black bowl" (App. C.3). *Missing:* full template list, normalisation.
- **9 N.** No description of how OpenVLA/Octo training data use the swapped target names or relations.
- **10 P.** Label validation: "blind external validation with three non-author annotators over 300 randomized items", with Fleiss’ κ per field (App. C.6, Table 19). *Missing:* criteria and rates for selecting the 600 LIBERO observations and the 165 Bridge frames.
- **11 P.** (a) The headline gap uses direction-blind sensitivity. The authors flag this: "Action sensitivity only asks whether an action changes; it does not determine whether the changed action is directed toward the intended target" (§3.3), and report ActCheck separately. (b) Unit is the example; "The intervals quantify sampling uncertainty over evaluated examples; they do not address benchmark-level distribution shift" (App., bootstrap CIs); no clustering. (c) None (McNemar p-values uncorrected, Table 7). (d) Positive control: "BridgeData V2 with Octo reaches 92.9 ..., showing that the metric can register stronger instruction-conditioned action separation when present" (§4.2).
- **12 R.** "To avoid making the diagnosis depend on a single calibrated cutoff, we additionally report target-versus-control AUC" (§3.3; App. L.1: OpenVLA AUC 0.573/0.583). Also: "Spatially separated target subsets give the same qualitative OpenVLA result" (App. D.5), and schema-backbone robustness, Qwen2.5-VL-7B vs 3B (Table 44).

### LIBERO-CF (2602.17659v2)
- **1–4 N/A** (closed-loop grounding/success; grasp-position heatmaps are rollout outcomes).
- **5 N.** Closed loop, but no step limit, chunk/execution length or control rate.
- **6 P.** "We use bf16 mixed precision" for X-VLA fine-tuning (App. B-B). *Missing:* inference precision, GPU, versions.
- **7 P.** Real robot: "a ZED 2i stereo camera as the exterior camera and a ZED mini camera as the wrist camera" (§VII-B). *Missing:* resolution, preprocessing, simulation camera details.
- **8 P.** Full real-world instruction list (Table VI, e.g. "Pick up the cup on the left") and suite sizes (CF-Spatial 15, CF-Object 10, CF-Long 10, CF-OOD 15 tasks; §IV-A). *Missing:* simulation instruction list, normalisation.
- **9 R.** Real robot: "we collect 20 demonstrations for each l∈L_o^in. For each l∈L_o^out, we collect only a single demonstration for minimal warm-up, while setting OOD tasks in a zero-shot manner" (§VII-B). Simulation: "VLAs are finetuned on the original LIBERO dataset, which covers only in-domain manipulation tasks" (§IV-A). *Note:* fine-tuning data only; pre-training corpora not characterised.
- **10 P.** Criterion only: "assigns alternative feasible language instructions under LIBERO scene layouts" (§I). No procedure, rates or agreement.
- **11 P.** (a) "grounding rate ... measures whether the gripper makes contact with the target object specified in the instruction" (§IV-A). (b) Unit is the trial ("we run 50 trials for each task", §VI-A; "10 trials per instruction", §VII-B); no clustering or intervals. (c) and (d) none.

### BeTTER (2604.18000v1)
- **1–4 N/A.**
- **5 N.** "short-horizon"/"long-horizon" are task types; no step limit or control rate.
- **6 N.** Only "trained for 100k steps across 16 GPUs" for VLM ablations (App.). No GPU model, dtype or versions; clarification (ii).
- **7 N.** No camera, resolution or preprocessing for the evaluated VLAs. The "224×224" mention (§5.2) is a generic claim about VLAs.
- **8 P.** Example instructions (spatial "top"/"bottom", semantic "red"/"blue", §4.1; "Put the lemon into the fruit basket", §6.1); VLM prompt templates for task generation (App.). *Missing:* full instruction set, number, normalisation.
- **9 P.** "All objects and locations are observed during training" and "During training, “red” is associated with a downward motion" (§4.1). *Missing:* counts or a training pairing table in the text.
- **10 P.** "we incorporate a human-in-the-loop verification step for targeted spatial refinements" (§3.1). No rates.
- **11 P.** (a) "We report the success rate of grasping the correct target under specific layout-instruction pairings" against "A 50% random-guess baseline" (Table 1). (b)–(d) none; trial counts not stated.

### LangGap (2603.00592v1)
- **1–4 N/A.**
- **5 N.**
- **6 P.** "fine-tune π0.5 with LoRA on a single RTX 4090 GPU" (§IV-A). Training only.
- **7 N.**
- **8 P.** One example per dimension (e.g. "the bowl to the right of the ramekin" → "the bowl to the right of the plate", §III-A) and task counts per dimension (Table III: 38/13/5/3). *Missing:* full list, normalisation.
- **9 R.** "Instruction-level split: Training tasks do not include all test tasks, ensuring that test evaluation contains language instructions not seen during training" (§III-C). Training tasks listed per suite (§III-D: "libero_spatial: 9 tasks ...").
- **10 P.** "All extended tasks are verified in the LIBERO simulator to ensure that the target object is graspable, the placement location is reachable, and the success condition is detectable"; "After physical feasibility checks, we obtain 59 valid extended tasks" (§III-C). *Missing:* number of candidates, so no approval rate.
- **11 P.** (a) Binary success on the instructed task. (b) "Each task is evaluated for 20 episodes with binary success" (§III-A); no clustering or intervals. (c) and (d) none. The "at most 1/k" chance bound (§III-C) is a reference level, not a positive control.

### InstructMove (2608.22990v1)
- **1–4 N/A** (Reach/Lift outcomes).
- **5 P.** Chunk lengths: π0/π0.5 "predict 50-step action chunks"; GR00T "30 valid action steps within a 40-step model horizon"; Motus chunks of 48 (App. C). *Missing:* control rate, episode limit.
- **6 P.** "5,000 steps on 8 NVIDIA RTX 5090 GPUs"; "BF16" for GR00T training (App. C). Training only.
- **7 P.** "The model consumes RGB observations resized to 224×224" (π0/π0.5); "Each model uses its own observation preprocessing" (App. C.1). Camera streams are named for GR00T. *Missing:* resolution for the other models; clarification (i).
- **8 P.** Templates such as "lift the {object attribute} {object category}" and "pick the {object category} {spatial relation} the {reference category}", plus the relation vocabulary "left of, right of, in front of, behind, near, and far from", defined "with respect to the robot-centric reference frame" (§3). *Missing:* number of templates, normalisation.
- **9 P.** Referring expressions "partitioned into seen and unseen sets for language-generalization evaluation" (App. A.2); "Training episodes are balanced within each task" (§4). *Missing:* per-expression status or counts; relation words not characterised.
- **10 P.** "Assets with low grasp success rates are flagged for review or excluded ... disagreements and consistency conflicts are routed to human annotators ... Only assets that pass scale, grasp, and consistency checks enter the final corpus" (App. A.2). *Missing:* exclusion rates, agreement.
- **11 P.** (a) "Reach indicates that the end-effector comes within a predefined distance of the instruction-consistent target" (§3.4). (b) Unit stated (100 held-out episodes; diagnostic over "100 fixed pick_category unseen-instance scenes, with each scene evaluated under every instruction condition", Table 4); no clustering or intervals. (c) and (d) none.

### RefGuard (2609.06221v1)
- **1–4 N/A** (closed-loop identity-switch / execution outcomes).
- **5 P.** π0.5: "We use an action horizon of 15"; Table V: "20 Hz low-level control frequency"; "This receding-horizon loop is repeated until the episode terminates" (App. A-B). GR00T: action chunks with temporal ensembling but no rate; clarification (i).
- **6 P.** "4 NVIDIA A100 80GB GPUs" for training (App. A-B). No inference dtype or versions.
- **7 P.** π0.5: main and wrist cameras, raw "640×480", "resized with padding to 224×224" at inference vs "images are resized to 224×224" in training (App. A-B, Table V). GR00T: top-view and wrist cameras with training-time colour jitter, resolution not stated; clarification (i).
- **8 P.** Templates per setting (App. A-D), e.g. S2 "Pick the [target] next to the [unique anchor]." and S4 "Pick the [target] left of the [anchor] from the robot’s perspective." *Missing:* normalisation.
- **9 P.** Baselines fine-tuned on "300 UF850 demonstrations collected via Meta Quest teleoperation across S1–S4 tasks" (App. A-B). No per-instruction or per-word breakdown.
- **10 P.** Layout criteria, e.g. S3 "Exactly one joint target-anchor assignment satisfies the RGB-D next-to relation" (App. A-D). No rates.
- **11 P.** (a) Identity switch = acting on the wrong instance. (b) Unit is the trial ("30 trials per method and setting", Table I; "no macro-averaging"); no clustering or intervals. (c) None. (d) "S1 is an unambiguous sanity check" (App. A-D).

### RoboIRGBench (2609.34384v1)
- **1–4 N/A.**
- **5 P.** Real robot: "both the action execution horizon and prediction horizon set to 20 steps" (§IV-F). *Missing:* control rate (the 30 Hz is the camera rate), simulation horizon.
- **6 N.**
- **7 P.** "a front-view RGBD camera and a wrist-mounted camera at 30 Hz, both of which are Intel RealSense 435IF" (§IV-F). *Missing:* resolution, preprocessing, simulation cameras.
- **8 P.** "rule-based instruction templates" (§III-A), examples ("the cube there", "one time fewer than that"), distribution of referring expressions (Fig. 2). *Missing:* template list and count, normalisation.
- **9 P.** Real robot only: "all policies are trained only with explicit instructions" and "Target colors, spatial arrangements, and execution counts are balanced across training and evaluation" (§IV-F). Simulation training data not described; clarification (i).
- **10 R.** "We exclude five RoboMME tasks whose instructions do not contain such variables" (§III-A). For the spatial setting, "StopCube, VideoRepick, VideoPlaceButton, and VideoPlaceOrder are excluded because their targets are defined primarily by action counts or temporal information" (§III-B). Criteria with counts.
- **11 P.** (a) Success rate. (b) "50 episodes under each of three random seeds ... averaged across the three seeds" (§IV-A); no clustering or intervals. (c) None. (d) "We therefore construct a paired explicit counterpart for each implicit instruction" (§IV-A).

### Act2Answer (2606.19297v1)
- **1–4 N/A** (placement-region outcome; hidden-state probes).
- **5 N.** "the required action is short-horizon" (§4.1); no step limit or control rate.
- **6 N.**
- **7 P.** Simulated scenes built on SimplerEnv (§4.3), with a "Visual Matching" variant using "real-background compositing" (App. A.3). Transforms are re-scored: "we evaluate each example in both its original and swapped left/right versions and report the average score across the two" (§4.4; App. F). *Missing:* VLA input resolution, preprocessing relative to training.
- **8 P.** "instructions of the form “Put the cube on …”", produced by an LLM followed by "human review and manual editing", "limited to template normalization" (App. B.4). *Missing:* template count, case normalisation.
- **9 P.** Models are grouped by whether they had VQA co-training ("VQA co-training is associated with stronger performance", §1/§4.6). Use of the probed words in training is not characterised.
- **10 R.** Filtering "by instruction length"; "human annotators remove examples in which the relevant objects are too small or visually ambiguous" (§4.3). Per-category "Initial pool" vs "Final eval" counts (App. B, Table 6). For OK-VQA, examples kept only with "stable annotator agreement" (App. B.3).
- **11 P.** (a) Soft success rate: probability of placing on the correct answer region (§4.4). (b) Dependence addressed: "The two swapped views of a single item are not fully independent ... Using the number of unique items as the effective sample size ... yields a slightly wider, more conservative margin" (App. E). (c) None. (d) None.
- **12 R.** "This appendix reports ablations that test whether the main Act2Answer conclusions are sensitive to evaluation choices" (App. A: resolution, prompt, texture, tile size, lighting, each with a stated conclusion). For the effective-N choice: "Our qualitative conclusions are unchanged under either choice" (App. E).

### Kirouane et al. (2609.07470v1)
- **1–4 N/A** (see the N/A rules).
- **5 R.** "action chunks of sixteen at 20 FPS" (§3, Training configuration); closed-loop "50 trials per task" (§3, Evaluation protocol). *Note:* stated in the training configuration; replanning and episode limits are not stated.
- **6 P.** "the stock Cosmos3 SFT trainer on 8× B200" (§3). Training only.
- **7 P.** "agentview and wrist cameras at 256²" (§3). *Missing:* preprocessing relative to training.
- **8 P.** The ten Greek training instructions and an independent set (Fig. 7); one vs "seven distinct Greek phrasings per task" (§5). *Missing:* case normalisation.
- **9 R.** Word-level account of training wording, e.g. "Goals 1, 5 and 7 show the glossary at work: stove becomes a word that also means kitchen" (Fig. 7); language sampling ratios and phrasings per arm (§3, §5).
- **10 P.** "Quality was audited by Greek-character ratio (mean ≥ 0.995), structure preservation, and spot review" (§3). *Missing:* exclusion counts.
- **11 R.**
  - (a) Margin of correct-language success over a wrong-instruction null (§3, Evaluation protocol).
  - (b) "We therefore treat the goal as the sampling unit and report a paired task-level bootstrap (20,000 resamples of the ten goals ...)" (§3, Statistical treatment).
  - (c) "we make at least twenty, and we apply no multiplicity correction; Appendix A lists them all with their sampling units so a reader can apply one" (§3; App. A discusses Bonferroni).
  - (d) English instructions act as a positive control ("correct English instructions, correct Greek instructions, and wrong instructions", §3).
- **12 R.** Instruments compared and verdicts retracted: "Five of our own conclusions did not survive contact with these controls" (abstract); "a ten-goal suite credited a policy with Greek instruction-following that a ninety-task suite shows to be marginal at best" (abstract).

### Chen et al. (2609.39971v1)
- **1 P.** Workspace axes are named for the mirror transforms ("x-axis mirror (front-back) ... y-axis mirror (left-right)", App. F, Fig. 11). The data builder stores "end-effector displacement as the translational action" (App. F). *Missing:* frame, sign, zero and units of the decoded chunks compared by Δ_L2 (App. E).
- **2 N/A**, **4 N/A** (continuous, jointly generated action heads).
- **3 N.** "decode the action chunk with the action expert unchanged" (App. E); flow noise and sampling not stated for the readout.
- **5 P.** Readout horizon: "We report it for the first step and the full chunk" (App. E); success is "the simulator predicate at episode termination" (App. A). *Missing:* control rate, episode limit.
- **6 N.** Only "converted back to JAX" (App. G), no version or dtype; clarification (iii).
- **7 P.** LIBERO-PRO renders; UR5e "recorded frames with images and proprioception held fixed" (App. E). Mirror and shift transforms are described for training data (App. F). *Missing:* resolution, preprocessing.
- **8 P.** Instruction delivery is made explicit: Semantic/Task cells are read "from the :language block", while other cells "keep the filename-derived wording used in the demonstrations" (App. B). *Missing:* template list and count, normalisation.
- **9 R.** "the four-suite training set shows the same destination word at different places, but every Spatial instruction names the same black bowl, so nothing in the data asks the policy to read the object word" (App. D). "In the original Spatial data each instruction’s action support lies on one side" (App. F).
- **10 P.** "Some generated cases were adjusted or dropped by hand to keep the scenes executable" (App. D). No counts. The reported κ = 0.78 (§3; App. D, Table 9) concerns outcome labels, not stimulus screening.
- **11 R.**
  - (a) Terminal-state success on the instructed task, plus human labels separating following from retrieval (§3).
  - (b) An "Evaluation unit" column for every analysis (App. A, Table 6). "Hierarchical-bootstrap intervals over seeds and rollouts are descriptive given only three seeds" (App. I).
  - (c) None.
  - (d) Positive control: "A scripted oracle parses the target object from the delivered instruction ... scoring 90/100 on the Object Task cell" (App. B).
- **12 R.** "The signal does not depend on how the representation is aggregated" (App. E, rank-1 recovery under alternative representations). An automated labeller is checked against human labels ("it marks 59 of 117 rollouts as source, where the human labels mark 52", App. D).

---

## 5. Incidental finding while coding (outside the twelve items)

Chen et al., App. B, report a silent instruction-delivery fault in LIBERO-PRO's released evaluation code:

> "For its Semantic and Task cells, LIBERO-PRO writes the perturbed instruction into the (:language ...) block of each regenerated BDDL file, but its released evaluation code reads the instruction from the file name, so the policy receives the original instruction while the success predicate scores the perturbed task. The maintainers have acknowledged this issue in the official LIBERO-PRO repository" (https://arxiv.org/html/2609.39971, App. B; their footnote cites LIBERO-PRO issue #14).

If this holds for the published numbers, it is exactly the kind of pitfall the BtP paper describes: a pipeline convention that silently changes a language verdict. I have not checked the issue or LIBERO-PRO's numbers myself, and I did not consult code repositories. Treat it as Chen et al.'s claim until a human reads the issue.

## 6. Limitations

- One AI coder. There are no agreement statistics, and some R/P boundaries are judgement calls (flagged above).
- Only the arXiv HTML full text was coded. Details that appear only in figures rendered as images, supplementary PDFs or repositories were missed by design. Two examples: Kirouane's Fig. 7 instruction list is an image, and BeTTER's training pairing is drawn in Fig. 1.
- The keyword searches used to locate evidence could miss unusual phrasing. N codes are the most likely to be wrong in the direction of under-crediting.
- The sample is the authors' own list of closest works, not a random sample of VLA evaluations. The counts describe this list, not the field.
