#!/usr/bin/env python3
"""Build the blind re-annotation tool for inter-annotator agreement (BtP plan B6 step 4; G2A annotation).

Draws 100 of the 400 frozen scenes, stratified by the original hand label, and writes
  sample.csv             the scene ids, in the order shown (no labels)
  annotation_tool.html   a self-contained page: each scene with the pipeline's gripper position
                         marked, four questions that start unanswered, autosave in the browser,
                         and a CSV export
Usage: .venv/bin/python docs/btp_paper/annotation/make_annotation_tool.py
"""
import base64, io, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "google_drive" / "v2" / "constructed"
SEED, N = 20261003, 100

val = pd.read_csv(DATA / "validation_sample.csv")
man = pd.read_csv(DATA / "constructed_manifest.csv").set_index("construct_id")
rng = np.random.default_rng(SEED)
shares = val["human_configuration"].value_counts()
quota = (shares / shares.sum() * N).round().astype(int)
quota.iloc[0] += N - quota.sum()
picked = []
for label, k in quota.items():
    ids = sorted(val.loc[val["human_configuration"] == label, "construct_id"])
    picked += list(rng.choice(ids, size=int(k), replace=False))
order = list(rng.permutation(picked))
keys = [f"s{i:03d}" for i in range(1, N + 1)]   # opaque: construct ids embed the recorded layout
pd.DataFrame({"position": range(1, N + 1), "scene_key": keys, "construct_id": order}).to_csv(HERE / "sample.csv", index=False)

scenes = []
for key, cid in zip(keys, order):
    m = man.loc[cid]
    img = Image.open(DATA / "frames" / f"{cid}.png").convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
    scenes.append({"id": key, "w": int(img.width), "h": int(img.height), "x": float(m["x_gripper"]),
                   "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})

QUESTIONS = [
    ("two_instances", "Are there exactly two instances of the same kind of object?", ["yes", "no"]),
    ("gripper_ok", "Is the dashed line at the horizontal position of the robot gripper?", ["yes", "no", "can't tell"]),
    ("arrangement", "Relative to the dashed line (image left/right, as displayed), where are the two instances?",
     ["both left", "one on each side", "both right", "unclear"]),
    ("plausible", "Are both instances placed in a physically possible way (e.g. resting on the table, not floating or off its edge)?",
     ["yes", "no"]),
]

page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Twin-scene re-annotation</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#d6d3d1;--accent:#0f766e;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--accent:#2dd4bf;--warn:#fbbf24}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 8px}.muted{color:var(--muted)}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0 16px}
.stage{position:relative;width:100%;max-width:640px;border:1px solid var(--line)}
.stage img{display:block;width:100%;height:auto}
.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.grid{display:grid;grid-template-columns:minmax(0,640px) minmax(0,1fr);gap:20px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
fieldset{border:1px solid var(--line);border-radius:8px;margin:0 0 12px;padding:8px 12px}
legend{font-weight:600;padding:0 4px}
label{display:inline-flex;gap:6px;align-items:center;margin:4px 14px 4px 0;cursor:pointer}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.45;cursor:not-allowed}
input[type=text]{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
.warn{color:var(--warn)}details{margin:0 0 12px}
</style></head><body><main>
<h1>Twin-scene re-annotation</h1>
<p class="muted">Label each scene from the image alone. Every answer starts unset. Do not consult the original labels or the other annotator. Your answers save in this browser as you go; export the CSV at the end and send it back.</p>
<details><summary>Criteria</summary><ul>
<li><b>Dashed line</b>: the horizontal position the pipeline uses for the gripper's start position. Answer whether it sits on the gripper.</li>
<li><b>Arrangement</b>: judge each instance's centre against the dashed line, in the image as displayed. Use "unclear" if an instance straddles the line or you cannot tell.</li>
<li><b>Plausible</b>: physically possible placement within the scene, e.g. not floating, not off the edge of the table, not intersecting other objects.</li></ul></details>
<div class="bar"><label>Annotator initials <input type="text" id="who" size="6" autocomplete="off"></label>
<span id="progress" class="muted"></span><span id="msg" class="warn"></span></div>
<div class="grid"><div><div class="stage"><img id="img" alt="scene"><svg id="ov"></svg></div>
<p class="muted" id="sid"></p></div><div><form id="form"></form>
<div class="bar"><button id="prev" type="button">Previous</button><button id="next" type="button" class="primary">Next</button>
<button id="export" type="button">Export CSV</button></div></div></div>
</main><script>
const SCENES = __SCENES__;
const QS = __QS__;
const store = {get(k){try{return JSON.parse(localStorage.getItem(k))}catch(e){return null}},
               set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const KEY = "twin-reannotation-v1";
let state = store.get(KEY) || {who:"", answers:{}, idx:0, t0:{}};
const $ = id => document.getElementById(id);
$("who").value = state.who || "";
$("who").addEventListener("input", e => {state.who = e.target.value.trim(); store.set(KEY, state)});
function render(){
  const s = SCENES[state.idx];
  $("img").src = s.img; $("sid").textContent = "Scene " + s.id + "  (" + (state.idx+1) + " of " + SCENES.length + ")";
  const x = s.x / s.w * 100;
  $("ov").setAttribute("viewBox", "0 0 100 100"); $("ov").setAttribute("preserveAspectRatio", "none");
  $("ov").innerHTML = '<line x1="'+x+'" x2="'+x+'" y1="0" y2="100" stroke="#fff" stroke-opacity="0.85" stroke-width="4" vector-effect="non-scaling-stroke"/>'
    + '<line x1="'+x+'" x2="'+x+'" y1="0" y2="100" stroke="#e11d48" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>';
  const a = state.answers[s.id] || {};
  $("form").innerHTML = QS.map(([k, q, opts]) => '<fieldset><legend>'+q+'</legend>' + opts.map(o =>
    '<label><input type="radio" name="'+k+'" value="'+o+'"'+(a[k]===o?' checked':'')+'> '+o+'</label>').join('') + '</fieldset>').join('');
  if (!state.t0[s.id]) state.t0[s.id] = Date.now();
  const done = Object.values(state.answers).filter(v => QS.every(([k]) => v[k])).length;
  $("progress").textContent = done + " of " + SCENES.length + " complete";
  $("prev").disabled = state.idx === 0;
  $("next").disabled = !QS.every(([k]) => a[k]) || state.idx === SCENES.length - 1;
  store.set(KEY, state);
}
$("form").addEventListener("change", e => {
  const s = SCENES[state.idx]; const a = state.answers[s.id] || {};
  a[e.target.name] = e.target.value; a.t_ms = Date.now() - state.t0[s.id]; state.answers[s.id] = a; render();
});
$("prev").onclick = () => {state.idx = Math.max(0, state.idx-1); render()};
$("next").onclick = () => {state.idx = Math.min(SCENES.length-1, state.idx+1); render()};
$("export").onclick = () => {
  if (!state.who) {$("msg").textContent = "Enter your initials first."; return}
  $("msg").textContent = "";
  const rows = [["position","scene_key","annotator",...QS.map(q=>q[0]),"seconds"]];
  SCENES.forEach((s,i) => {const a = state.answers[s.id] || {};
    rows.push([i+1, s.id, state.who, ...QS.map(([k]) => a[k]||""), a.t_ms ? (a.t_ms/1000).toFixed(1) : ""])});
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g,'""') + '"').join(",")).join("\\n");
  const url = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  const link = document.createElement("a"); link.href = url; link.download = "reannotation_" + state.who + ".csv"; link.click();
};
render();
</script></body></html>"""
html = page.replace("__SCENES__", json.dumps(scenes)).replace("__QS__", json.dumps(QUESTIONS))
(HERE / "annotation_tool.html").write_text(html, encoding="utf-8")
print("wrote", HERE / "annotation_tool.html", f"({len(html) / 1e6:.1f} MB);", "strata:", quota.to_dict())
