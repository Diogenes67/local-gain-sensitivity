"""
Matched token-local perturbation assay (20 Sept 2026)
====================================================
Question. The gain profile is sigma_1 of the token-local block Jacobian, an infinitesimal
same-token quantity; the dependence measure is the loss change under a finite rotation of a
block's branch outputs at every position. A reviewer can ask whether the dissociation between
them is an artefact of comparing two different kinds of perturbation. This assay puts both on
the same footing: at the same token position t and block l used for sigma_1, the block input
h_{l,t} is displaced by a small vector delta = eps * ||h_{l,t}|| * u, with u either the top right
singular vector v_1 of that token-local Jacobian (the direction sigma_1 refers to) or a random
unit direction, and both the local amplification ||f(h + delta) - f(h)|| / ||delta|| at the block
output and the effect on the network's task loss and output distribution are measured, at
eps in {0.01, 0.03, 0.1, 0.3, 1}. Every quantity is measured on the same input, at the same
token, with the same perturbation norm, so sigma_1 and the task effect can be compared block
by block within a model and across models.

Models. The 50 September checkpoints on Drive: destroyed-data (d1 v2.1: real, shuffled,
random labels, bidirectional; five seeds each; hourglass/d1_v21/ckpt) and the k-gram sweep
(k in {1, 2, 3, 4, 5, 8}, five seeds; hourglass/reprofile_v2/kgram/ckpt). Inputs are five
held-out sequences of each model's own distribution (shuffled models on shuffled text,
random-label models on real text with the real-text readout, bidirectional models on
corrupted real text with a fixed corruption, k-gram models on held-out sequences of their own
source), four positions per input (t = 16, 48, 80, 112), all 12 blocks.

Per (model, block, input, position): sigma_1 and v_1 by power iteration on J^T J of the
token-local map (forward-mode jvp and reverse-mode vjp on the block, the estimator of the
paper), then one batched forward of 51 sequence copies (clean, plus 5 directions x 5 eps x
2 signs) with the perturbation applied to the block input at position t by a pre-hook,
recording the block output at t (local gain), the output distribution at t and at every
position from t onward (KL from the clean distribution) and the model's own loss over the
positions from t onward (delta-CE). Signs are averaged. Aggregates per block and per model
(middle-third to edge-third ratios of the task effect, alongside R_ex0 of sigma_1 on the same
inputs) are written with the raw per-sample values.

Outputs. One JSON per model under $MATCHED_OUT/results, atomic, resume-safe; summary table from
disk; zip export. Needs survey_sigma1_v2.py, factorial_v2.py, d1_v2.py and reprofile_v2.py beside
it (embedded in the notebook). About 30 min for all 50 models on an A100.

Usage:  python matched_perturbation.py               # all 50, resume-safe
        python matched_perturbation.py --exp kgram --cond 8 --seed 0
        python matched_perturbation.py --plumbing     # random-weight models, synthetic inputs, CPU
Environment: MATCHED_OUT, D1_CKPT, D1_DATA, REPRO_ROOT, MATCHED_QUICK=1.
"""
import os, sys, json, time, gc, argparse, shutil, datetime, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REPRO_OUT", os.environ.get("REPRO_ROOT", "reprofile_v2"))
import factorial_v2 as fv
import d1_v2 as d1
import reprofile_v2 as rp

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(os.environ.get("MATCHED_OUT", "matched"))
D1_CKPT = Path(os.environ.get("D1_CKPT", "d1_v21/ckpt"))
D1_DATA = Path(os.environ.get("D1_DATA", "data_cache/_wikitext103_128.pt"))
REPRO_ROOT = Path(os.environ.get("REPRO_ROOT", "reprofile_v2"))
SEED = 42
EPS = [0.01, 0.03, 0.1, 0.3, 1.0]
N_RAND = 4
POSITIONS = [16, 48, 80, 112]
N_INPUTS = 5
PI_ITERS, PI_RESTARTS, PI_TOL = 50, 2, 1e-6
D1_CONDS = {"real": 5, "shuffled": 5, "randlab": 5, "bidir": 5}
K_VALUES = [1, 2, 3, 4, 5, 8]
VERSION = "matched-v1-2026-09-20"


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------- model adapters
class Adapter:
    """Uniform access to a controlled model: block list, block call, full forward, loss."""

    def __init__(self, kind, model, cond=None):
        self.kind, self.model, self.cond = kind, model, cond
        T = fv.SEQ
        if kind == "d1":
            self.layers = list(model.layers)
            self.mask = fv.mask_for("MLM" if cond == "bidir" else "AR", T, DEVICE)
        else:
            self.layers = list(model.blocks)
            self.mask = rp.causal_mask(T, DEVICE)
        self.causal = not (kind == "d1" and cond == "bidir")

    def forward(self, ids):
        if self.kind == "d1":
            return self.model(ids, self.mask)
        return self.model(ids, attn_mask=self.mask)

    def block_call(self, li, x):
        if self.kind == "d1":
            return self.layers[li](x, mask=self.mask)
        return self.layers[li](x, attn_mask=self.mask)

    def loss_from(self, logits, ids, t, sel=None):
        """Per-row loss over the positions the perturbation can reach: next-token CE at positions
        t..T-2 (causal), or masked-token CE over the selected positions (bidirectional)."""
        V = logits.shape[-1]
        if self.causal:
            lg = logits[:, t:-1]; tg = ids[:, t + 1:]
            return F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1), reduction="none").view(lg.shape[0], -1).mean(1)
        out = []
        for i in range(logits.shape[0]):
            out.append(F.cross_entropy(logits[i][sel], ids[i][sel]))
        return torch.stack(out)


def load_d1(cond, seed):
    model = fv.Transformer()
    sd = torch.load(D1_CKPT / f"{cond}_s{seed}.pt", map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    return model.to(DEVICE).eval()


def load_kgram(k, seed):
    model = rp.D3Transformer()
    sd = torch.load(REPRO_ROOT / "kgram" / "ckpt" / f"kgram_{k}_s{seed}.pt", map_location="cpu")
    model.load_state_dict({k_: v.float() for k_, v in sd.items()})
    model.drop.p = 0.0
    return model.to(DEVICE).eval()


_D1_DATA = {}


def d1_inputs(cond, seed):
    """Five held-out sequences of the model's own distribution and, for bidir, a fixed corruption."""
    if "eval" not in _D1_DATA:
        data = torch.load(D1_DATA)
        _D1_DATA["eval"] = data[-(d1.N_EVAL * fv.BATCH):].clone()
        del data
    real_eval = _D1_DATA["eval"]
    ids = real_eval[:N_INPUTS]
    if cond == "shuffled":
        ids = d1.shuffle_within(real_eval, 500 + seed)[:N_INPUTS]
    sels, xs = None, None
    if cond == "bidir":
        gen = torch.Generator(device="cpu").manual_seed(4242 + seed)
        xs, sels = [], []
        for i in range(N_INPUTS):
            x, sel = fv.corrupt(ids[i:i + 1], gen)
            xs.append(x[0]); sels.append(sel[0])
        xs, sels = torch.stack(xs), torch.stack(sels)
    return ids, xs, sels


def kgram_inputs(k, seed):
    d = torch.load(REPRO_ROOT / "data" / f"kgram_k{k}_s{seed}.pt")
    return d["held"][-N_INPUTS:], None, None


# ---------------------------------------------------------------- token-local Jacobian: sigma_1 and v_1
def capture_block_input(ad, li, ids_1):
    store = {}
    h = ad.layers[li].register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach().clone()))
    try:
        with torch.no_grad():
            ad.forward(ids_1)
    finally:
        h.remove()
    return store["x"]          # (1, T, d)


def make_token_local_f(ad, li, hidden0, positions):
    """f: (P, d) -> (P, d). Copy p of the sequence has position positions[p] replaced by x[p];
    returns the block output at that position. Other positions keep their natural values."""
    P = len(positions)
    base = hidden0.expand(P, -1, -1).clone()
    idx = torch.arange(P, device=hidden0.device)
    pos = torch.tensor(positions, device=hidden0.device)

    def f(x):
        h = base.clone()
        h = h.index_put((idx, pos), x)
        out = ad.block_call(li, h)
        return out[idx, pos]
    return f


def top_singular(f, x0, iters=PI_ITERS, restarts=PI_RESTARTS, tol=PI_TOL, seed=SEED):
    """Batched power iteration on J^T J for each row of x0 (rows are independent copies).
    Returns sigma_1 (P,), v_1 (P, d) and iterations used."""
    P, d = x0.shape
    best_s = torch.zeros(P, device=x0.device); best_v = torch.zeros_like(x0); its = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for r in range(restarts):
        v = torch.randn(P, d, generator=g).to(x0.device)
        v = v / v.norm(dim=1, keepdim=True)
        n_it = iters
        for it in range(iters):
            _, Jv = torch.func.jvp(f, (x0,), (v,))
            _, vjp_fn = torch.func.vjp(f, x0)
            JtJv = vjp_fn(Jv)[0]
            v_new = JtJv / (JtJv.norm(dim=1, keepdim=True) + 1e-30)
            cos = (v_new * v).sum(1).abs()
            v = v_new
            if bool((1 - cos).max() < tol):
                n_it = it + 1; break
        _, Jv = torch.func.jvp(f, (x0,), (v,))
        s = Jv.norm(dim=1)
        better = s > best_s
        best_s = torch.where(better, s, best_s); best_v[better] = v[better]; its.append(n_it)
    return best_s.detach(), best_v.detach(), float(np.mean(its))


# ---------------------------------------------------------------- matched perturbation
@torch.no_grad()
def perturb_and_read(ad, li, ids_1, tgt_1, x_in, t, dirs, sel=None):
    """One batched forward: row 0 clean, then for each direction, eps and sign a copy with
    h_{l,t} += sign * eps * ||h_{l,t}|| * u. Returns per-row local gain, KL at t, KL over the
    reachable positions and delta-CE over the reachable positions."""
    h_t = x_in[0, t]                                  # (d,)
    norm = h_t.norm().item()
    rows = [torch.zeros_like(h_t)]
    meta = [("clean", 0.0, 0)]
    for name, u in dirs:
        for e in EPS:
            for sgn in (1, -1):
                rows.append(sgn * e * norm * u); meta.append((name, e, sgn))
    delta = torch.stack(rows)                         # (B, d)
    B = delta.shape[0]
    ids = ids_1.expand(B, -1)                         # model input (corrupted for the bidirectional models)
    tgt = tgt_1.expand(B, -1)                         # loss targets (the original tokens)
    out_store = {}

    def pre(m, args, kwargs):
        x = args[0].clone()
        x[:, t] = x[:, t] + delta
        return (x,) + tuple(args[1:]), kwargs

    hp = ad.layers[li].register_forward_pre_hook(pre, with_kwargs=True)
    ho = ad.layers[li].register_forward_hook(lambda m, a, o: out_store.__setitem__("y", o[:, t].detach().clone()))
    try:
        logits = ad.forward(ids).float()
    finally:
        hp.remove(); ho.remove()
    y = out_store["y"]                                 # (B, d) block output at t
    gain = (y[1:] - y[0:1]).norm(dim=1) / delta[1:].norm(dim=1)
    logp = F.log_softmax(logits, -1)                   # (B, T, V)
    cl = logp[0:1]
    kl_all = (cl.exp() * (cl - logp)).sum(-1)          # (B, T)
    kl_t = kl_all[:, t]
    if ad.causal:
        kl_seq = kl_all[:, t:-1].mean(1)
    else:
        kl_seq = kl_all.mean(1)
    loss = ad.loss_from(logits, tgt, t, sel)
    dce = loss - loss[0]
    # output-norm change at t, for the record
    return dict(meta=meta[1:], gain=gain.cpu().numpy(), kl_t=kl_t[1:].cpu().numpy(), kl_seq=kl_seq[1:].cpu().numpy(),
                dce=dce[1:].cpu().numpy(), h_norm=norm, clean_loss=float(loss[0]))


def aggregate(meta, arr, names):
    """Sign-averaged mean per (direction class, eps): returns {name: [per eps]}."""
    out = {}
    for name in names:
        vals = []
        for e in EPS:
            sel = [i for i, (n, ee, s) in enumerate(meta) if (n == name or (name == "rand" and n.startswith("rand"))) and ee == e]
            vals.append(float(np.mean(arr[sel])))
        out[name] = vals
    return out


def run_model(ad, ids, xs, sels, quick=False):
    """Full assay for one model. Returns the per-block aggregates and raw per-sample values."""
    L = len(ad.layers)
    blocks = range(L) if not quick else [0, L // 2, L - 1]
    positions = POSITIONS if not quick else POSITIONS[:2]
    n_in = ids.shape[0] if not quick else 1
    per_block = []
    raw = []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for li in blocks:
        sig_all, gain_v1, gain_rand, klt_v1, klt_rand, kls_v1, kls_rand, dce_v1, dce_rand, its_all = [], [], [], [], [], [], [], [], [], []
        for i in range(n_in):
            ids_1 = (xs[i:i + 1] if xs is not None else ids[i:i + 1]).to(DEVICE)   # model input
            tgt_1 = ids[i:i + 1].to(DEVICE)                                            # loss targets
            sel = sels[i].to(DEVICE) if sels is not None else None
            x_in = capture_block_input(ad, li, ids_1)
            f = make_token_local_f(ad, li, x_in, positions)
            x0 = x_in[0, positions].clone()
            sig, v1, n_it = top_singular(f, x0)
            its_all.append(n_it)
            for p, t in enumerate(positions):
                dirs = [("v1", v1[p])]
                for r in range(N_RAND):
                    u = torch.randn(x0.shape[1], generator=g).to(DEVICE); u = u / u.norm()
                    dirs.append((f"rand{r}", u))
                res = perturb_and_read(ad, li, ids_1, tgt_1, x_in, t, dirs, sel)
                ag = {k: aggregate(res["meta"], res[k], ["v1", "rand"]) for k in ("gain", "kl_t", "kl_seq", "dce")}
                sig_all.append(float(sig[p]))
                gain_v1.append(ag["gain"]["v1"]); gain_rand.append(ag["gain"]["rand"])
                klt_v1.append(ag["kl_t"]["v1"]); klt_rand.append(ag["kl_t"]["rand"])
                kls_v1.append(ag["kl_seq"]["v1"]); kls_rand.append(ag["kl_seq"]["rand"])
                dce_v1.append(ag["dce"]["v1"]); dce_rand.append(ag["dce"]["rand"])
                raw.append(dict(block=li, input=i, position=t, sigma1=float(sig[p]), h_norm=res["h_norm"], clean_loss=res["clean_loss"],
                                gain=ag["gain"], kl_t=ag["kl_t"], kl_seq=ag["kl_seq"], dce=ag["dce"]))
        pb = dict(block=li, sigma1=float(np.mean(sig_all)), sigma1_samples=sig_all, power_iters=float(np.mean(its_all)),
                  gain_v1=np.mean(gain_v1, 0).tolist(), gain_rand=np.mean(gain_rand, 0).tolist(),
                  kl_t_v1=np.mean(klt_v1, 0).tolist(), kl_t_rand=np.mean(klt_rand, 0).tolist(),
                  kl_seq_v1=np.mean(kls_v1, 0).tolist(), kl_seq_rand=np.mean(kls_rand, 0).tolist(),
                  dce_v1=np.mean(dce_v1, 0).tolist(), dce_rand=np.mean(dce_rand, 0).tolist())
        per_block.append(pb)
        log(f"    block {li:2d}: sigma1 {pb['sigma1']:.3f}  gain(v1, eps=0.01) {pb['gain_v1'][0]:.3f}  gain(rand) {pb['gain_rand'][0]:.3f}  "
            f"KLseq(v1) eps 0.1 {pb['kl_seq_v1'][2]:.2e} eps 1 {pb['kl_seq_v1'][4]:.2e}  dCE(v1) eps 0.1 {pb['dce_v1'][2]:+.2e}  rand {pb['dce_rand'][2]:+.2e}")
    return per_block, raw


def thirds_ratio(vals):
    p = np.asarray(vals, float)[1:]; n = len(p); t = n // 3
    if n < 3: return float("nan"), float("nan"), float("nan")
    early, mid, late = p[:t], p[t:n - t], p[n - t:]
    edge = np.concatenate([early, late]).mean()
    return (float(mid.mean() / edge) if edge > 0 else float("nan"), float(early.mean() / mid.mean()) if mid.mean() > 0 else float("nan"),
            float(late.mean() / mid.mean()) if mid.mean() > 0 else float("nan"))


def summarise(per_block):
    """Model-level summary: R_ex0 of sigma1 on these inputs and the same ratio for the task effects."""
    s = {}
    sig = [b["sigma1"] for b in per_block]
    s["R_ex0_sigma1"], s["EM_sigma1"], s["LM_sigma1"] = thirds_ratio(sig)
    for key in ("kl_seq_v1", "kl_seq_rand", "dce_v1", "dce_rand", "kl_t_v1", "gain_v1"):
        for ei, e in enumerate(EPS):
            prof = [b[key][ei] for b in per_block]
            r, em, lm = thirds_ratio(prof)
            s[f"R_ex0_{key}_eps{e}"] = r
            n = len(prof); t3 = (n - 1) // 3
            s[f"waist_{key}_eps{e}"] = float(np.mean(prof[1 + t3: n - t3]))
    # does sigma1 track the task effect across blocks within the model?
    from scipy import stats
    for key in ("kl_seq_v1", "dce_v1"):
        prof = [b[key][2] for b in per_block]           # eps = 0.1
        rho, p = stats.spearmanr(sig[1:], prof[1:]) if len(sig) > 3 else (float("nan"), float("nan"))
        s[f"spearman_sigma1_vs_{key}_eps0.1_ex0"] = float(rho)
    # gain along v1 at the smallest eps should reproduce sigma1: report the ratio
    s["gain_v1_eps0.01_over_sigma1"] = float(np.mean([b["gain_v1"][0] / b["sigma1"] for b in per_block if b["sigma1"] > 0]))
    return s


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def provenance():
    p = {"torch": torch.__version__, "python": platform.python_version(), "date": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE}
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name()
    return p


def jobs():
    out = []
    for cond, n in D1_CONDS.items():
        for s in range(n):
            out.append(("d1", cond, s))
    for k in K_VALUES:
        for s in range(5):
            out.append(("kgram", str(k), s))
    return out


def run_job(exp, cond, seed, res_dir, quick):
    fpath = res_dir / f"matched_{exp}_{cond}_s{seed}{'_quick' if quick else ''}.json"
    if fpath.exists():
        try:
            d = json.load(open(fpath))
            if d.get("version") == VERSION and len(d.get("per_block", [])) >= (12 if not quick else 3):
                log(f"[SKIP] {exp} {cond} s{seed}: done"); return
        except Exception:
            pass
    t0 = time.time()
    log("\n" + "-" * 78 + f"\n{exp} {cond} seed {seed}\n" + "-" * 78)
    if exp == "d1":
        model = load_d1(cond, seed); ad = Adapter("d1", model, cond); ids, xs, sels = d1_inputs(cond, seed)
    else:
        model = load_kgram(int(cond), seed); ad = Adapter("kgram", model); ids, xs, sels = kgram_inputs(int(cond), seed)
    per_block, raw = run_model(ad, ids, xs, sels, quick)
    summ = summarise(per_block)
    log(f"  R_ex0 sigma1 {summ['R_ex0_sigma1']:.3f} | R (KLseq v1, eps 0.1) {summ['R_ex0_kl_seq_v1_eps0.1']:.3f} | R (dCE v1, eps 0.1) {summ['R_ex0_dce_v1_eps0.1']:.3f} "
        f"| waist dCE v1 eps 0.1 {summ['waist_dce_v1_eps0.1']:+.3e} | gain/sigma1 at eps 0.01: {summ['gain_v1_eps0.01_over_sigma1']:.3f} | {time.time() - t0:.0f} s")
    out = dict(experiment=exp, condition=cond, seed=seed, version=VERSION, eps=EPS, positions=POSITIONS if not quick else POSITIONS[:2],
               n_inputs=ids.shape[0] if not quick else 1, n_random_directions=N_RAND,
               inputs="held-out sequences of the model's own distribution" + (" (corrupted, fixed masks; masked-token readout)" if cond == "bidir" else "") + (" with the real-text readout" if cond == "randlab" else ""),
               estimator="token-local J^T J power iteration on the block (jvp + vjp), same map as survey_sigma1_v2, block input at position t",
               perturbation="h_{l,t} += sign * eps * ||h_{l,t}|| * u, u in {v_1, 4 random unit vectors}, signs averaged; pre-hook on the block input",
               readouts=dict(gain="||block output change at t|| / ||delta||", kl_t="KL(clean || perturbed) of the output distribution at t",
                             kl_seq="mean KL over positions t..T-2 (causal) or all positions (bidirectional)",
                             dce="change of the model's own loss over positions t..T-2 (next-token CE; real-text CE for randlab) or over the masked positions (bidirectional)"),
               summary=summ, per_block=per_block, raw=raw, quick=quick, elapsed_seconds=time.time() - t0, **{f"prov_{k}": v for k, v in provenance().items()})
    write_json(fpath, out)
    log(f"  saved {fpath}")
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summary(res_dir):
    rows = []
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("matched_") and fn.endswith(".json"):
            d = json.load(open(os.path.join(res_dir, fn))); s = d["summary"]
            rows.append((d["experiment"], d["condition"], d["seed"], s["R_ex0_sigma1"], s["R_ex0_kl_seq_v1_eps0.1"], s["R_ex0_dce_v1_eps0.1"],
                         s["waist_dce_v1_eps0.1"], s["waist_dce_rand_eps0.1"], s["waist_kl_seq_v1_eps0.1"], s["gain_v1_eps0.01_over_sigma1"], d.get("quick", False)))
    log(f"\n{'exp':6s} {'cond':9s} {'s':>2s} {'R_s1':>6s} {'R_KL':>6s} {'R_dCE':>6s} {'wdCEv1':>10s} {'wdCErnd':>10s} {'wKLv1':>10s} {'g/s1':>6s}")
    for r in rows:
        log(f"{r[0]:6s} {r[1]:9s} {r[2]:2d} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:10.3e} {r[7]:10.3e} {r[8]:10.3e} {r[9]:6.3f}" + ("  QUICK" if r[10] else ""))
    return rows


def export(res_dir):
    stage = OUT / "matched_export"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    n = 0
    for fn in os.listdir(res_dir):
        if fn.endswith(".json"):
            shutil.copy(os.path.join(res_dir, fn), stage); n += 1
    z1 = shutil.make_archive(str(OUT / "matched_results"), "zip", stage)
    z2 = shutil.make_archive(os.path.join(os.getcwd(), "matched_results"), "zip", stage)
    for z in (z1, z2):
        log(f"export: {n} files -> {z} ({os.path.getsize(z) / 1e3:.0f} kB)")
    return z1, z2


# ---------------------------------------------------------------- plumbing test (no checkpoints, no data)
def plumbing():
    torch.manual_seed(0)
    fv.VOCAB = 2000; rp.VOCAB_SIZE = 2000; fv.EOS = 1999   # small vocabulary keeps the CPU test inside a few GB
    res_dir = OUT / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("d1", "bidir", "kgram"):
        xs = sels = None
        ids = torch.randint(0, 1000, (2, fv.SEQ))
        if kind == "d1":
            model = fv.Transformer().to(DEVICE).eval(); ad = Adapter("d1", model, "real")
        elif kind == "bidir":
            model = fv.Transformer().to(DEVICE).eval(); ad = Adapter("d1", model, "bidir")
            gen = torch.Generator(device="cpu").manual_seed(1)
            x, sel = fv.corrupt(ids, gen); xs, sels = x, sel
        else:
            model = rp.D3Transformer().to(DEVICE).eval(); model.drop.p = 0.0; ad = Adapter("kgram", model)
        per_block, raw = run_model(ad, ids, xs, sels, quick=True)
        summ = summarise(per_block)
        ratio = summ["gain_v1_eps0.01_over_sigma1"]
        log(f"  {kind}: gain along v1 at eps 0.01 / sigma1 = {ratio:.4f} (expect ~1); gain(rand)/sigma1 = "
            f"{np.mean([b['gain_rand'][0] / b['sigma1'] for b in per_block]):.3f} (expect < 1)")
        assert 0.8 < ratio < 1.2, "finite perturbation along v1 does not reproduce sigma1"
        write_json(res_dir / f"matched_{kind}_plumbing_s0_quick.json", dict(experiment=kind, condition="plumbing", seed=0, version=VERSION, summary=summ, per_block=per_block, quick=True))
    summary(res_dir); export(res_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["d1", "kgram"]); ap.add_argument("--cond"); ap.add_argument("--seed", type=int)
    ap.add_argument("--quick", action="store_true", default=bool(os.environ.get("MATCHED_QUICK")))
    ap.add_argument("--plumbing", action="store_true")
    args, _ = ap.parse_known_args()
    log(f"matched perturbation | {provenance()} | out={OUT}")
    if args.plumbing:
        plumbing(); return
    res_dir = OUT / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    if args.exp: todo = [j for j in todo if j[0] == args.exp]
    if args.cond: todo = [j for j in todo if j[1] == args.cond]
    if args.seed is not None: todo = [j for j in todo if j[2] == args.seed]
    log(f"{len(todo)} models")
    for exp, cond, seed in todo:
        try:
            run_job(exp, cond, seed, res_dir, args.quick)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"  FAILED {exp} {cond} s{seed}: {e}")
    summary(res_dir); export(res_dir)


if __name__ == "__main__":
    main()
