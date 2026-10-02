"""Rewrite sections d and e of src/ed_table4.md (Table S3 of the Patterns version) from predictor_check.json, adding the joint
baseline (depth + norm + random-direction gain) with and without sigma_1 and the paired difference between the two.
Run: python update_table_s3.py  (reads ../r5/predictor_check.json; rewrites src/ed_table4.md in place; prints the new rows)."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "..", "r5", "predictor_check.json")))
SETS = ["depth", "depth+norm", "depth+randgain", "depth+gain"]


def f(x):
    return f"{x:.2f}".replace("-", "−")


def cell(blk, key=None):
    if key is None:
        four = " / ".join(f(blk[k]["r2"]) for k in SETS); j = f"{f(blk['joint']['r2'])} → {f(blk['joint+gain']['r2'])}"
    else:
        four = " / ".join(f(blk[k]["by_set"][key]) for k in SETS); j = f"{f(blk['joint']['by_set'][key])} → {f(blk['joint+gain']['by_set'][key])}"
    return f"{four}; joint {j}"


def note(blk, key=None, first=""):
    b = blk["joint"]["delta_r2_joint_gain_minus_joint" + ("_by_set" if key else "")]
    if key: b = b[key]
    s = f"Δ(joint + σ₁ − joint) {f(b['point'])} ({f(b['ci'][0])} to {f(b['ci'][1])})"
    if key is None:
        a = blk["depth+randgain"]["delta_r2_depth_gain_minus_this"]
        s = f"Δ(σ₁ − random gain) {f(a['point'])} ({f(a['ci'][0])} to {f(a['ci'][1])}); " + s
    return (first + "; " if first else "") + s


def rows(sec, y, yr, label):
    c = d["controlled"]; p = d["pretrained"]
    first = "depth / + norm / + random-direction gain / + σ₁; joint = depth + norm + random-direction gain, → with σ₁" if sec == "d" else "the six sets as in d"
    R = [(f"**{sec}** {label}", "controlled, leave one model out", "50 folds", cell(c[y]["L1O"]), cell(c[yr]["L1O"]), note(c[y]["L1O"], first=first)),
         ("", "controlled, leave one condition out", "10 folds pooled", cell(c[y]["LOCO"]), cell(c[yr]["LOCO"]), note(c[y]["LOCO"])),
         ("", "", "4 text conditions pooled", cell(c[y]["LOCO"], "training_conditions"), cell(c[yr]["LOCO"], "training_conditions"), note(c[y]["LOCO"], "training_conditions")),
         ("", "", "6 k-gram orders pooled", cell(c[y]["LOCO"], "kgram"), cell(c[yr]["LOCO"], "kgram"), note(c[y]["LOCO"], "kgram")),
         ("", "controlled, leave one set out", "fit 20 text models, predict 30 k-gram", cell(c[y]["LOSO"], "kgram"), cell(c[yr]["LOSO"], "kgram"), note(c[y]["LOSO"], "kgram")),
         ("", "", "fit 30 k-gram models, predict 20 text", cell(c[y]["LOSO"], "training_conditions"), cell(c[yr]["LOSO"], "training_conditions"), note(c[y]["LOSO"], "training_conditions")),
         ("", "pretrained, leave one model out", "6 folds", cell(p[y]["L1O"]), cell(p[yr]["L1O"]), note(p[y]["L1O"])),
         ("", "pretrained, leave one family out", "5 folds", cell(p[y]["LOFO"]), cell(p[yr]["LOFO"]), note(p[y]["LOFO"], first="") + ("; per-family values with intervals in Figure 2g" if sec == "d" else ""))]
    return ["| " + " | ".join(r) + " |" for r in R]


src = os.path.join(HERE, "src", "ed_table4.md")
lines = open(src).read().split("\n")
# locate section d rows (from the '**d**' row to the row before '**e**') and section e rows up to the slope rows
i_d = next(i for i, l in enumerate(lines) if l.startswith("| **d**"))
i_e = next(i for i, l in enumerate(lines) if l.startswith("| **e**"))
i_e_end = next(i for i, l in enumerate(lines) if i > i_e and "pooled slope and within-model ρ" in l)   # first slope row of e
new_d = rows("d", "y_abs", "y_rel", "Predictor sets on identical folds")
new_e = rows("e", "y_abs_t", "y_rel_t", "Same-position readout D~t~")
out = lines[:i_d] + new_d + lines[i_e:i_e][:0] + new_e + lines[i_e_end:]
text = "\n".join(out)
text = text.replace("Section d compares four predictor sets on identical folds (predictor_check.py): block position; block position with the log activation norm; with the log gain along a random unit direction at ε = 0.01; and with log σ₁. Δ is the pooled R² of the σ₁ set minus that of the random-direction set on the same folds, with a 95% percentile interval from 2,000 resamples of the held-out models.",
                    "Section d compares six predictor sets on identical folds (predictor_check.py): block position; block position with the log activation norm; with the log gain along a random unit direction at ε = 0.01; with log σ₁; the joint baseline of block position with the norm and the random-direction gain; and the joint baseline with log σ₁. Δ(σ₁ − random gain) is the pooled R² of the σ₁ set minus that of the random-direction set on the same folds, and Δ(joint + σ₁ − joint) that of the joint baseline with σ₁ minus without, each with a 95% percentile interval from 2,000 resamples of the held-out models.")
open(src, "w").write(text)
print("\n".join(new_d + new_e))
