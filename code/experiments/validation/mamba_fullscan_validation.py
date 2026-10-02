"""
mamba_fullscan_validation.py  --  rerun of the Mamba full-scan estimator validation (ED Fig. 1)
==========================================================================================

The September 2026 paper reproduces Extended Data Fig. 1 from the original image because the
run's data files were not retained. This script reruns the three checks of that figure on the
same design, writes one JSON per (sequence length, layer, input) to persistent storage as soon
as it is computed, and regenerates the figure from the saved rows.

Design (as in the original run, patterns_repo/experiments/mamba_sigma1_validation.py):
  * Mamba-130M, layers 0, 12 and 23, sequence lengths T = 4 and 8, five inputs each
    (uniformly random token ids, seed 42 + 1000 * input index), float32, reference PyTorch
    selective scan (no mamba_ssm / causal_conv1d / kernels packages).
  * Check 1: the dense Jacobian of the whole-sequence block map, R^{Td} -> R^{Td}, is
    materialised (3,072 x 3,072 and 6,144 x 6,144) and decomposed exactly; the JVP/VJP power
    iteration on J^T J (survey_sigma1_v2.sigma1 settings: four restarts, 50 iterations,
    tolerance 1e-6) is compared with the exact sigma_1.
  * Check 2: finite-difference gain ||f(x + eps v) - f(x)|| / eps along the estimated v_1 and
    along 20 random unit directions at 12 scales, eps in [1e-6, 1e-1].
  * Check 3 (T = 8): the full-scan sigma_1 against the largest one-step cross-position gain.
    Three versions are recorded: the exact spectral norm of the Jacobian block
    d y_{t+1} / d x_t read off the materialised J (new, exact); the original eight-probe
    finite-difference estimate of the same quantity (eps = 1e-4); and, where the transformers
    internals allow it, the exact norm of the diagonal SSM state transition d s_t / d s_{t-1}
    (max over channels and states of exp(A dt_t)). The exact token-local sigma_1 at every
    position, the diagonal blocks of the same J, is recorded as well.

Outputs (all under --results-dir, which should be on Drive):
  fullscan/rows/T{T}_L{layer}_i{input}.json   one per pair, written atomically
  fullscan/fullscan_summary.json               regenerated from the rows by summary()
  fullscan/edfig1_rerun.{png,pdf}              regenerated from the rows by make_figure()
  fullscan/run_log.txt                         appended by the notebook's run cell

Resume: an existing row file is skipped, so a disconnect costs at most one pair.

Usage:
  python mamba_fullscan_validation.py --results-dir /content/drive/MyDrive/hourglass/validation_rerun
  python mamba_fullscan_validation.py --smoke --results-dir /tmp/x    # tiny random model, CPU ok
"""
import os, sys, json, time, gc, argparse, datetime, shutil, traceback
import numpy as np
import torch

import survey_sigma1_v2 as sv

VERSION = "mamba-fullscan-rerun-2026-09-25"  # summary() direction ratio on the 1e-4..1e-2 plateau (25 Sept, evening)
DEVICE = sv.DEVICE
SEED = sv.SEED


def log(msg):
    print(msg, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--seq-lens", default="4,8")
    ap.add_argument("--layers", default="0,mid,last")
    ap.add_argument("--n-inputs", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=sv.N_RESTARTS)
    ap.add_argument("--iters", type=int, default=sv.PI_ITERS)
    ap.add_argument("--tol", type=float, default=sv.PI_TOL)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny randomly initialised Mamba; no download")
    ap.add_argument("--figure-only", action="store_true", help="summary and figure from existing rows")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- model
def load_model(args):
    if args.smoke:
        from transformers import MambaConfig, MambaForCausalLM
        torch.manual_seed(0)
        cfg = MambaConfig(vocab_size=97, hidden_size=16, state_size=4, num_hidden_layers=3,
                          intermediate_size=32, conv_kernel=4, use_cache=False)
        model = MambaForCausalLM(cfg).to(DEVICE).float().eval()
        label = "tiny-random-mamba"
    else:
        cfg = sv.CFG[args.model]
        b = sv.load_bundle(cfg, "float32")
        model, label = b["model"], cfg["label"]
    for p in model.parameters():
        p.requires_grad_(False)
    layers = sv.get_layers(model, "mamba")
    bundle = dict(model=model, layers=layers, arch="mamba", kind="text", label=label,
                  vocab_size=int(model.config.vocab_size),
                  forward=lambda m, inp: m(input_ids=inp["input_ids"], use_cache=False))
    return bundle


def fast_path_report():
    """The paper's Mamba numbers use the reference PyTorch scan. Report which accelerated
    packages are importable so the log shows which path ran."""
    found = []
    for name in ("mamba_ssm", "causal_conv1d", "kernels", "mambapy"):
        try:
            __import__(name)
            found.append(name)
        except Exception:
            pass
    log(f"accelerated-scan packages importable: {found if found else 'none (reference PyTorch scan)'}")
    return found


def layer_indices(spec, L):
    out = []
    for s in spec.split(","):
        s = s.strip()
        out.append(L // 2 if s == "mid" else L - 1 if s == "last" else int(s))
    return sorted(set(out))


# ---------------------------------------------------------------- SSM transition capture
class TransitionCapture:
    """Wraps transformers' mamba_selective_scan to record, per timestep, the largest entry of
    the diagonal discretised transition exp(A dt_t), which is the spectral norm of
    d s_t / d s_{t-1}. Best effort: if the internals differ, .per_t stays None."""

    def __init__(self):
        self.per_t = None
        self.mod = None
        self.orig = None
        try:
            from transformers.models.mamba import modeling_mamba as mm
            self.mod, self.orig = mm, mm.mamba_selective_scan
        except Exception:
            pass

    def __enter__(self):
        if self.orig is None:
            return self
        cap = self

        def wrapped(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, *a, **kw):
            try:
                dt_ = dt
                if delta_bias is not None:
                    dt_ = dt_ + delta_bias.to(dt_.dtype)[..., None]
                if delta_softplus:
                    dt_ = torch.nn.functional.softplus(dt_)
                dA = torch.exp(A[None, :, None, :].float() * dt_[:, :, :, None].float())  # (b, inter, T, state)
                cap.per_t = dA[0].amax(dim=(0, 2)).detach().cpu().tolist()
            except Exception:
                cap.per_t = None
            return cap.orig(hidden_states, dt, A, B, C, D, z, delta_bias, delta_softplus, *a, **kw)

        self.mod.mamba_selective_scan = wrapped
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            self.mod.mamba_selective_scan = self.orig
        return False


# ---------------------------------------------------------------- estimators
def power_iteration_v(f, x0, n_restarts, iters, tol):
    """Batched power iteration on J^T J for one base point x0 (1, T, d); returns
    (sigma_1 estimate = max over restarts, its right singular vector (1, T, d), iterations)."""
    x = x0.repeat_interleave(n_restarts, dim=0)
    g = torch.Generator(device=x.device).manual_seed(SEED)
    v = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)
    v = v / sv._flat_norm(v)
    sigma = torch.zeros(x.shape[0], device=x.device)
    n_done = 0
    for it in range(iters):
        Jv = sv.jvp_func(f, x, v).float()
        s_new = sv._flat_norm(Jv).flatten()
        u = Jv / sv._flat_norm(Jv).clamp_min(1e-30)
        JTu = sv.vjp(f, x, u).float()
        v = JTu / sv._flat_norm(JTu).clamp_min(1e-30)
        n_done = it + 1
        if it >= 4:
            rel = ((s_new - sigma).abs() / s_new.clamp_min(1e-30)).max().item()
            sigma = s_new
            if rel < tol:
                break
        else:
            sigma = s_new
    k = int(sigma.argmax().item())
    return float(sigma[k].item()), v[k:k + 1].detach(), n_done


def exact_jacobian(f, x0):
    """Dense Jacobian of the flattened whole-sequence map at x0 (1, T, d): (Td, Td)."""
    T, d = x0.shape[1], x0.shape[2]
    x_flat = x0.reshape(-1).detach()

    def f_flat(z):
        return f(z.view(1, T, d)).reshape(-1)

    try:
        J = torch.autograd.functional.jacobian(f_flat, x_flat, vectorize=True)
    except Exception as e:
        log(f"    vectorised Jacobian failed ({type(e).__name__}); falling back to column-wise")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        J = torch.autograd.functional.jacobian(f_flat, x_flat)
    return J.detach()


def fd_gain(f, x0, direction, eps_values):
    with torch.no_grad():
        y0 = f(x0)
        out = []
        for eps in eps_values:
            y = f(x0 + eps * direction)
            out.append(((y - y0).norm() / eps).item())
    return out


def one_step_fd(f, x0, T, d, n_probes=8, eps=1e-4, seed=SEED):
    """Original check-3 estimate: perturb position t along n_probes random unit directions and
    read the response at position t + 1 only; max over probes of ||dy_{t+1}|| / eps."""
    g = torch.Generator(device=x0.device).manual_seed(seed)
    with torch.no_grad():
        y0 = f(x0)
        out = []
        for t in range(T - 1):
            best = 0.0
            for _ in range(n_probes):
                v = torch.randn(d, generator=g, device=x0.device)
                v = v / v.norm()
                xp = x0.clone()
                xp[0, t, :] += eps * v
                y = f(xp)
                best = max(best, ((y[0, t + 1] - y0[0, t + 1]).norm() / eps).item())
            out.append(best)
    return out


# ---------------------------------------------------------------- one pair
def run_pair(bundle, T, li, ii, args, rows_dir):
    path = os.path.join(rows_dir, f"T{T}_L{li}_i{ii}.json")
    if os.path.exists(path) and not args.no_skip:
        log(f"  skip T={T} layer {li} input {ii} (exists)")
        return
    t0 = time.time()
    torch.manual_seed(SEED + 1000 * ii)
    ids = torch.randint(0, bundle["vocab_size"], (1, T), device=DEVICE)
    inp = {"input_ids": ids}
    with TransitionCapture() as tc:
        captured = sv.capture_layer_inputs(bundle, inp)
        hidden0, rest, kw = captured[li]
        # the capture above ran the whole model; re-run the block alone so per_t is this block's
        call = sv.make_layer_call(bundle["layers"][li], rest, kw)
        with torch.no_grad():
            call(hidden0)
        dA_per_t = tc.per_t
    x0 = hidden0.float().detach()
    d = x0.shape[-1]
    f = sv.make_f_fullseq(call, False, torch.float32)

    # check 1: exact SVD against power iteration
    J = exact_jacobian(f, x0)
    Td = J.shape[0]
    svals = torch.linalg.svdvals(J)
    s1_exact, s2_exact = svals[0].item(), svals[1].item()
    s1_est, v1, n_it = power_iteration_v(f, x0, args.restarts, args.iters, args.tol)
    rel_err = abs(s1_est - s1_exact) / s1_exact

    # exact block norms from J: token-local (diagonal) and one-step cross-position (sub-diagonal)
    tokenlocal = [torch.linalg.svdvals(J[t * d:(t + 1) * d, t * d:(t + 1) * d])[0].item() for t in range(T)]
    onestep = [torch.linalg.svdvals(J[(t + 1) * d:(t + 2) * d, t * d:(t + 1) * d])[0].item() for t in range(T - 1)]
    del J

    # check 2: finite-difference gain along v1 and random directions
    eps_values = np.logspace(-6, -1, 12).tolist()
    gains_v1 = fd_gain(f, x0, v1, eps_values)
    g = torch.Generator(device=DEVICE).manual_seed(SEED + 17 + ii)
    gains_random = []
    for _ in range(args.n_random):
        r = torch.randn(x0.shape, generator=g, device=DEVICE)
        r = r / r.norm()
        gains_random.append(fd_gain(f, x0, r, eps_values))

    # check 3 (original estimate; the exact values are 'onestep_exact' above)
    onestep_fd = one_step_fd(f, x0, T, d) if T >= 8 else None

    row = dict(model=bundle["label"], T=T, layer=li, input=ii, Td=Td, d=d,
               sigma1_exact=s1_exact, sigma2_exact=s2_exact, sigma1_est=s1_est, rel_err=rel_err, iters=n_it,
               eps_values=eps_values, gains_v1=gains_v1, gains_random=gains_random,
               tokenlocal_sigma1_exact=tokenlocal, onestep_exact=onestep, onestep_fd8=onestep_fd,
               ssm_dA_max_per_t=dA_per_t,
               provenance=dict(version=VERSION, survey_version=sv.VERSION, seed_ids=SEED + 1000 * ii,
                               restarts=args.restarts, iters=args.iters, tol=args.tol, n_random=args.n_random,
                               device=(torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                               torch=torch.__version__, transformers=sv._tf_version(),
                               date=datetime.datetime.now().isoformat(timespec="seconds"),
                               time_s=time.time() - t0))
    tmp = path + ".tmp"
    json.dump(row, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    log(f"  T={T} layer {li:2d} input {ii}: sigma1 exact {s1_exact:.6f} est {s1_est:.6f} (rel err {rel_err:.1e}, {n_it} it)  "
        f"FD(v1) {gains_v1[7]:.4f}  random mean {np.mean([r[7] for r in gains_random]):.4f} (eps 2e-3)  "
        f"one-step exact max {max(onestep):.4f}  token-local max {max(tokenlocal):.4f}"
        + (f"  dA max {max(dA_per_t):.4f}" if dA_per_t else "") + f"  [{time.time() - t0:.0f} s]")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary and figure
def load_rows(rows_dir):
    rows = []
    for fn in sorted(os.listdir(rows_dir)) if os.path.isdir(rows_dir) else []:
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(rows_dir, fn))))
    return rows


def summary(out_dir):
    rows = load_rows(os.path.join(out_dir, "rows"))
    if not rows:
        log("no rows yet"); return None
    err = np.array([r["rel_err"] for r in rows])
    s = dict(version=VERSION, n_pairs=len(rows), model=rows[0]["model"],
             seq_lens=sorted({r["T"] for r in rows}), layers=sorted({r["layer"] for r in rows}),
             rel_err_median=float(np.median(err)), rel_err_max=float(err.max()),
             pearson_r=float(np.corrcoef([r["sigma1_exact"] for r in rows], [r["sigma1_est"] for r in rows])[0, 1]),
             iters_median=float(np.median([r["iters"] for r in rows])), by_layer={})
    log(f"check 1: {len(rows)} pairs; relative error median {s['rel_err_median']:.1e}, max {s['rel_err_max']:.1e}; "
        f"Pearson r {s['pearson_r']:.6f}; iterations median {s['iters_median']:.0f}")
    log(f"{'layer':>5} {'T':>3} {'n':>2} {'sigma1 exact':>13} {'rel err max':>12} {'v1/random plateau':>17} {'full / one-step':>16} {'full / dA':>10}")
    for li in s["layers"]:
        for T in s["seq_lens"]:
            sub = [r for r in rows if r["layer"] == li and r["T"] == T]
            if not sub:
                continue
            # direction ratio on the plateau of the finite-difference curve (1e-4 <= eps <= 1e-2), where the gain
            # along v1 equals sigma1; below about 1e-5 the differences are dominated by float32 rounding
            ratio_dir = []
            for r in sub:
                idx = [i for i, e in enumerate(r["eps_values"]) if 1e-4 <= e <= 1e-2]
                gv = np.array(r["gains_v1"])[idx]; gr = np.mean(np.array(r["gains_random"])[:, idx], axis=0)
                ratio_dir.append(float(np.mean(gv / gr)))
            full_one = [r["sigma1_exact"] / max(r["onestep_exact"]) for r in sub]
            full_dA = [r["sigma1_exact"] / max(r["ssm_dA_max_per_t"]) for r in sub if r.get("ssm_dA_max_per_t")]
            full_fd = [r["sigma1_exact"] / max(r["onestep_fd8"]) for r in sub if r.get("onestep_fd8")]
            e = dict(n=len(sub), sigma1_exact_mean=float(np.mean([r["sigma1_exact"] for r in sub])),
                     rel_err_max=float(max(r["rel_err"] for r in sub)),
                     v1_over_random_gain=[float(np.mean(ratio_dir)), float(np.std(ratio_dir))],
                     full_over_onestep_exact=[float(np.mean(full_one)), float(np.std(full_one))],
                     full_over_onestep_fd8=[float(np.mean(full_fd)), float(np.std(full_fd))] if full_fd else None,
                     full_over_ssm_dA=[float(np.mean(full_dA)), float(np.std(full_dA))] if full_dA else None,
                     tokenlocal_sigma1_max_mean=float(np.mean([max(r["tokenlocal_sigma1_exact"]) for r in sub])))
            s["by_layer"][f"L{li}_T{T}"] = e
            log(f"{li:>5} {T:>3} {len(sub):>2} {e['sigma1_exact_mean']:>13.4f} {e['rel_err_max']:>12.1e} "
                f"{e['v1_over_random_gain'][0]:>8.2f} +- {e['v1_over_random_gain'][1]:<5.2f} "
                f"{e['full_over_onestep_exact'][0]:>8.1f} +- {e['full_over_onestep_exact'][1]:<5.1f} "
                + (f"{e['full_over_ssm_dA'][0]:>10.1f}" if e['full_over_ssm_dA'] else f"{'n/a':>10}"))
    path = os.path.join(out_dir, "fullscan_summary.json")
    json.dump(s, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    log(f"summary saved {path}")
    return s


def make_figure(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load_rows(os.path.join(out_dir, "rows"))
    if not rows:
        return None
    layers = sorted({r["layer"] for r in rows})
    Tmax = max(r["T"] for r in rows)
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.2, "axes.titlesize": 6.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.35))
    colours = {li: c for li, c in zip(layers, ["#1f77b4", "#2ca02c", "#d62728"])}
    markers = {4: "o", 8: "s", 16: "^"}
    # left: exact vs estimate
    ax = axes[0]
    for T in sorted({r["T"] for r in rows}):
        for li in layers:
            sub = [r for r in rows if r["T"] == T and r["layer"] == li]
            ax.scatter([r["sigma1_exact"] for r in sub], [r["sigma1_est"] for r in sub], s=22,
                       marker=markers.get(T, "o"), color=colours[li], alpha=0.85,
                       label=f"layer {li}, T = {T}")
    lo = 0.9 * min(r["sigma1_exact"] for r in rows); hi = 1.1 * max(r["sigma1_exact"] for r in rows)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_1$, exact SVD of the full-scan Jacobian"); ax.set_ylabel(r"$\sigma_1$, power iteration")
    ax.legend(fontsize=5, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    err = np.array([r["rel_err"] for r in rows])
    ins = ax.inset_axes([0.62, 0.1, 0.34, 0.3])
    ins.hist(np.log10(np.clip(err, 1e-9, None)), bins=10, color="grey")
    ins.set_xlabel(r"$\log_{10}$ relative error", fontsize=5, labelpad=1); ins.tick_params(labelsize=5, pad=1)
    ins.set_title(f"median {np.median(err):.1e}", fontsize=5, pad=2)
    # middle: FD gain along v1 and random directions, T = Tmax, mean over inputs
    ax = axes[1]
    for li in layers:
        sub = [r for r in rows if r["T"] == Tmax and r["layer"] == li]
        if not sub:
            continue
        eps = sub[0]["eps_values"]
        gv = np.mean([r["gains_v1"] for r in sub], axis=0)
        gr = np.mean([np.mean(r["gains_random"], axis=0) for r in sub], axis=0)
        ax.plot(eps, gv, "-", color=colours[li], label=f"layer {li}, $v_1$")
        ax.plot(eps, gr, "--", color=colours[li], label=f"layer {li}, random")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"perturbation scale $\varepsilon$"); ax.set_ylabel(r"$\|f(x+\varepsilon v)-f(x)\|/\varepsilon$")
    ax.set_title(f"T = {Tmax}, mean over inputs", fontsize=6.5)
    ax.legend(fontsize=5, frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    # right: full-scan sigma1 over the largest one-step cross-position gain (exact), T = Tmax
    ax = axes[2]
    xs, ys, es = [], [], []
    for li in layers:
        sub = [r for r in rows if r["T"] == Tmax and r["layer"] == li]
        if not sub:
            continue
        ratios = [r["sigma1_exact"] / max(r["onestep_exact"]) for r in sub]
        xs.append(str(li)); ys.append(np.mean(ratios)); es.append(np.std(ratios))
    ax.bar(xs, ys, yerr=es, color=[colours[li] for li in layers if any(r["T"] == Tmax and r["layer"] == li for r in rows)],
           alpha=0.85, capsize=3)
    ax.set_xlabel("layer"); ax.set_ylabel(r"full-scan $\sigma_1$ / max$_t \|\partial y_{t+1}/\partial x_t\|$")
    ax.set_title(f"T = {Tmax}; error bars, s.d. across inputs", fontsize=6.5)
    axes[1].axvspan(min(eps) * 0.8, 1e-5, color="0.9", zorder=0)
    axes[1].text(1.1e-6, axes[1].get_ylim()[0] * 1.15, "float32\nrounding", fontsize=5, color="0.35")
    for a, lab in zip(axes, "abc"):
        a.text(-0.2, 1.04, lab, transform=a.transAxes, fontsize=8, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"edfig1_rerun.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"figure saved {os.path.join(out_dir, 'edfig1_rerun.png')}")
    return True


def export(out_dir, stem="mamba_fullscan_results"):
    """Zip rows + summary + figure beside the output directory and into the working directory."""
    z1 = shutil.make_archive(os.path.join(os.path.dirname(out_dir.rstrip("/")), stem), "zip", out_dir)
    z2 = shutil.copy(z1, os.path.join(os.getcwd(), os.path.basename(z1)))
    for z in (z1, z2):
        log(f"export: {z} ({os.path.getsize(z) / 1e6:.2f} MB)")
    return z1, z2


# ---------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.results_dir, "fullscan")
    rows_dir = os.path.join(out_dir, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    log(f"{VERSION} (estimator {sv.VERSION}); device {DEVICE}; torch {torch.__version__}; transformers {sv._tf_version()}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name} ({p.total_memory / 1e9:.0f} GB); TF32 matmul={torch.backends.cuda.matmul.allow_tf32}")
    if args.figure_only:
        summary(out_dir); make_figure(out_dir); return
    fast_path_report()
    bundle = load_model(args)
    L = len(bundle["layers"])
    lis = layer_indices(args.layers, L)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    log(f"model {bundle['label']}: {L} blocks; layers {lis}; T {seq_lens}; {args.n_inputs} inputs; "
        f"restarts {args.restarts}, iters {args.iters}, tol {args.tol}; rows -> {rows_dir}")
    t0 = time.time()
    for T in seq_lens:
        for li in lis:
            for ii in range(args.n_inputs):
                try:
                    run_pair(bundle, T, li, ii, args, rows_dir)
                except Exception as e:
                    log(f"  FAILED T={T} layer {li} input {ii}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
    log(f"total {time.time() - t0:.0f} s")
    summary(out_dir)
    try:
        make_figure(out_dir)
    except Exception as e:
        log(f"figure failed: {type(e).__name__}: {e}")
    export(out_dir)


if __name__ == "__main__":
    main()
