"""Assemble validation_rerun_colab.ipynb from the scripts in this folder (25 Sept 2026)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["survey_sigma1_v2.py", "mamba_fullscan_validation.py", "validation_summary.py", "validation_smoke.py"]


def md(s): return dict(cell_type="markdown", metadata={}, source=s)
def code(s): return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s)


cells = [md("""# Estimator validation rerun at full precision (JMLR version, Appendix H)

The exact-decomposition validation of Fig. 1d,e and the Mamba full-scan check of Extended Data Fig. 1 survive only as three-decimal printed logs and as an image. This notebook reruns both with every value written to Drive. Nothing is retrained and no manuscript number is changed by this notebook; the numbers are folded in afterwards from the saved files.

**Part A (about 30–60 min on an A100; a T4 falls back to the column-wise Jacobian on Phi-2 and takes hours):** `survey_sigma1_v2.py --validate`, unchanged (v2.3), on GPT-2 124M, Pythia-160M, Pythia-410M, Phi-2, ConvNeXt-tiny, ConvNeXt-base and Mamba-130M: the exact token-local Jacobian at five blocks × three positions × two natural inputs per model (210 pairs), exact σ₁, σ₂, ρ and σ₁(J − I) against the power-iteration estimates and 64 random probes. One JSON per model with every pair at full precision.

**Part B (about 10–15 min on an A100):** `mamba_fullscan_validation.py`, a port of the original ED Fig. 1 script with persistence: Mamba-130M, layers 0, 12 and 23, T = 4 and 8, five random-id inputs each (30 pairs); exact SVD of the 3,072² and 6,144² full-scan Jacobians against the power iteration; finite-difference gain along v₁ and 20 random directions at 12 scales; full-scan σ₁ against the largest one-step cross-position gain (exact from the same Jacobian, the original eight-probe finite-difference estimate, and the exact diagonal SSM transition norm). One JSON per pair, and the figure regenerated from them.

**Outputs (all on Drive under `hourglass/validation_rerun/`):** `validation/validation_<model>.json`, `run_log.txt`, `validation_rerun_summary.json`; `fullscan/rows/*.json`, `fullscan/fullscan_summary.json`, `fullscan/edfig1_rerun.{png,pdf}`, `fullscan_log.txt`; and `validation_rerun_results.zip` with all of it. Every file is written as soon as it is computed; a disconnect loses nothing; re-run the run cell. Upload `validation_rerun_results.zip` back (or paste the two summary printouts).

The Mamba numbers in the paper use the reference PyTorch selective scan, so do not install `mamba_ssm`, `causal_conv1d` or `kernels`; cell 6 prints which path ran. No gated models; no Hugging Face token needed.
"""),
code("""# 1. Drive, dependencies
import os, sys, subprocess
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content')
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.16,<6", "datasets>=3.0", "huggingface_hub", "pandas", "pyarrow", "pillow", "matplotlib"])
import torch, transformers
props = torch.cuda.get_device_properties(0)
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), round(getattr(props, 'total_memory', 0) / 1e9), 'GB')
for name in ("mamba_ssm", "causal_conv1d", "kernels"):
    try:
        __import__(name); print(f"WARNING: {name} is installed; the paper's Mamba values use the reference PyTorch scan")
    except Exception:
        pass
"""),
code("""# 2. Paths (everything under Drive)
H = "/content/drive/MyDrive/hourglass/validation_rerun"
for d in ("validation", "fullscan/rows"):
    os.makedirs(f"{H}/{d}", exist_ok=True)
import glob
print(H)
print("Part A files present:", sorted(os.path.basename(f) for f in glob.glob(f"{H}/validation/validation_*.json")))
print("Part B rows present:", len(glob.glob(f"{H}/fullscan/rows/*.json")))
""")]
for s in SCRIPTS:
    body = open(os.path.join(HERE, s)).read()
    cells.append(code(f"%%writefile /content/{s}\n" + body))
cells += [
code("""# 3. Smoke test (about 1 min, no downloads): tiny random GPT-2 through survey_sigma1_v2.run_validation and a tiny random Mamba
#    through mamba_fullscan_validation; checks the JSON layouts, the summary and the figure. Writes to /content/smoke, not Drive.
!cd /content && python validation_smoke.py /content/smoke 2>&1 | grep -v Warning | tail -8
"""),
code("""# 4. Part A: exact token-local validation, seven models (resume-safe: a model whose JSON is on Drive is skipped)
MODELS = ["gpt2", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "microsoft/phi-2",
          "facebook/convnext-tiny-224", "facebook/convnext-base-224", "state-spaces/mamba-130m-hf"]
for m in MODELS:
    !cd /content && python survey_sigma1_v2.py --validate --models {m} --results-dir {H} 2>&1 | grep -v Warning | tee -a {H}/run_log.txt
"""),
code("""# 5. Part A summary from the JSONs on Drive (safe to re-run; the reported values are the paper's three-decimal ones)
!cd /content && python validation_summary.py {H}
"""),
code("""# 6. Part B: Mamba full-scan validation (resume-safe, one JSON per pair; re-run after a disconnect)
!cd /content && python mamba_fullscan_validation.py --results-dir {H} 2>&1 | grep -v Warning | tee -a {H}/fullscan_log.txt
"""),
code("""# 7. Part B summary and figure from the rows on Drive (safe to re-run)
!cd /content && python mamba_fullscan_validation.py --figure-only --results-dir {H}
from IPython.display import Image, display
display(Image(f"{H}/fullscan/edfig1_rerun.png", width=1100))
"""),
code("""# 8. Export everything (re-run after a partial sweep): validation_rerun_results.zip on Drive and in /content
import shutil
z = shutil.make_archive("/content/validation_rerun_results", "zip", H)
shutil.copy(z, f"{H}/validation_rerun_results.zip")
print(z, f"{os.path.getsize(z) / 1e6:.2f} MB"); print(f"{H}/validation_rerun_results.zip")
"""),
]
nb = dict(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                 "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=0)
out = os.path.join(HERE, "validation_rerun_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(out, len(cells), "cells", os.path.getsize(out), "bytes")
