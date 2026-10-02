"""Plumbing test for the validation rerun: runs survey_sigma1_v2.run_validation on a tiny randomly
initialised GPT-2 (no download) and mamba_fullscan_validation on a tiny random Mamba, then checks
that the JSON files exist and carry the expected keys. Seconds on CPU."""
import os, sys, json, glob, shutil, subprocess
import torch
import survey_sigma1_v2 as sv

OUT = sys.argv[1] if len(sys.argv) > 1 else "smoke_out"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)


def tiny_bundle(cfg, dtype_name):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=101, n_embd=24, n_layer=3, n_head=2, n_positions=64,
                                   attn_implementation="eager")).to(sv.DEVICE).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return dict(arch="gpt2", hf_id="gpt2", cfg=cfg, dtype=torch.float32, cls_offset=0, kind="text", vocab_size=101,
                forward=lambda mm, inp: mm(input_ids=inp["input_ids"], use_cache=False), model=m,
                layers=sv.get_layers(m, "gpt2"))


def tiny_inputs(b, kind, n):
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, 101, (1, 32), generator=g).to(sv.DEVICE)} for _ in range(n)]


sv.load_bundle, sv.prepare_inputs = tiny_bundle, tiny_inputs
args = sv.parse_args(["--validate", "--results-dir", OUT, "--no-skip"])
sv.run_validation(sv.CFG["gpt2"], args)
f = os.path.join(OUT, "validation", "validation_gpt2.json")
d = json.load(open(f))
assert d["summary"]["n"] == len(d["rows"]) == 18, d["summary"]["n"]   # 3 blocks x 3 positions x 2 inputs
need = {"input", "block", "position", "d", "sigma1_exact", "sigma2_exact", "rho_exact", "sigma1_branch_exact",
        "sigma1_est", "sigma1_est_iters", "rho_est", "rho_est_iters", "random_probe_64"}
assert need <= set(d["rows"][0]), set(d["rows"][0])
assert d["summary"]["sigma1_max_rel_err"] < 1e-3, d["summary"]
print(f"Part A plumbing ok: {f}; max rel err {d['summary']['sigma1_max_rel_err']:.1e}")

r = subprocess.run([sys.executable, "mamba_fullscan_validation.py", "--smoke", "--seq-lens", "4", "--n-inputs", "1",
                    "--layers", "0,last", "--results-dir", OUT], capture_output=True, text=True)
print(r.stdout[-600:])
assert r.returncode == 0, r.stderr[-2000:]
rows = glob.glob(os.path.join(OUT, "fullscan", "rows", "*.json"))
assert len(rows) == 2, rows
row = json.load(open(rows[0]))
assert row["rel_err"] < 1e-3 and len(row["gains_v1"]) == 12 and len(row["gains_random"]) == 20, row["rel_err"]
assert os.path.exists(os.path.join(OUT, "fullscan", "fullscan_summary.json"))
assert os.path.exists(os.path.join(OUT, "fullscan", "edfig1_rerun.png"))
for z in glob.glob(os.path.join(os.getcwd(), "mamba_fullscan_results.zip")):
    os.remove(z)
print("Part B plumbing ok:", len(rows), "rows; summary and figure written")
print("SMOKE OK")
