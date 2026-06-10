# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Fast single-sequence folding with ESMFold2-Fast
#
# [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) is the smaller, inference-
# optimized configuration of [ESMFold2](https://biohub.ai/esm/protein/about) from
# [Biohub](https://biohub.ai/). It is tuned for **very fast single-sequence folding** —
# well suited to high-throughput folding, designed sequences, metagenomic proteins, and
# targets with limited homology. This notebook folds one protein chain by default and
# includes an optional batch cell that folds several sequences in a loop.
#
# This is a notebook port of a Modal app; the model code is unchanged — only the
# scaffolding is restructured so it runs top-to-bottom in Colab. For complexes
# (protein + DNA + ligand) and maximum accuracy, use the companion
# `esmfold2_colab.ipynb` notebook (the full model).
#
# > **Before you run anything:** select a GPU runtime via
# > **Runtime → Change runtime type → Hardware accelerator → GPU**. ESMFold2-Fast is
# > small, so a free-tier **T4** is sufficient.

# %% [markdown]
# ## Setup
#
# Colab runtimes are ephemeral — they reset silently and nothing persists between
# sessions. So **every cold start** must reinstall the package and re-point the weight
# cache. This single Setup cell does all of that and **must run first, before any
# import**. We install `esm` from a pinned upstream commit (it pulls a custom
# `transformers` fork and the CUDA build of PyTorch) so builds are reproducible.

# %%
# Reinstall on every cold start (ephemeral runtime). Pinned for reproducibility.
ESM_REVISION = "81b3646c9429ea8458918415ad6a46178cb59833"
# !pip install -q "esm @ git+https://github.com/Biohub/esm.git@{ESM_REVISION}" py3Dmol

import os

# ESMFold2-Fast weights are small, so re-downloading each session is cheap; we default to
# ephemeral /content. Set USE_DRIVE = True to cache to Drive across sessions instead.
USE_DRIVE = False  #@param {type:"boolean"}
if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    HF_HOME = "/content/drive/MyDrive/esmfold2/hf_cache"  # persistent across sessions
else:
    HF_HOME = "/content/hf_cache"  # fast, but ephemeral

# HF_HOME must be set before transformers/huggingface_hub are imported.
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"  # speed up downloads
os.makedirs(HF_HOME, exist_ok=True)
HF_HOME

# %% [markdown]
# ## Imports
#
# All imports, once. We only need `ProteinInput` / `StructurePredictionInput` (Fast is
# single-sequence), the `ESMFold2InputBuilder`, the `ESMFold2Model` class, and `py3Dmol`
# for inline rendering.

# %%
from pathlib import Path

import torch
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

import py3Dmol

# %% [markdown]
# ## Config
#
# A notebook has no command line, so every tunable value lives here as a plain
# assignment. The `# parameters` tag lets tools like Papermill override these for
# programmatic runs; the `#@param` annotations render as interactive widgets in Colab.
#
# `SEQUENCE` is a single protein chain (here, the M.HhaI methyltransferase as an
# example). Note there is **no pinned revision** for ESMFold2-Fast below — it defaults to
# the repo's `main`; pin a commit if the model card publishes one.

# %% tags=["parameters"]
MODEL_REPO = "biohub/ESMFold2-Fast"  #@param {type:"string"}
MODEL_REVISION = ""  #@param {type:"string"}  empty -> repo default (main)

SEQUENCE = (  #@param {type:"string"}
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)

NUM_LOOPS = 3  #@param {type:"integer"}
NUM_SAMPLING_STEPS = 50  #@param {type:"integer"}
NUM_DIFFUSION_SAMPLES = 1  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}

# Output persistence: /content is fast but ephemeral; a Drive path survives sessions.
OUTPUT_DIR = "/content/esmfold2_fast_out"  #@param {type:"string"}

# %% [markdown]
# ## Load model
#
# Loading downloads the weights to `HF_HOME` and moves the model onto the GPU. This is
# the most expensive setup step, so it gets its own cell — re-run it only when you change
# the model. Passing `revision=None` when `MODEL_REVISION` is empty tells `transformers`
# to use the repo default. On Colab the GPU is NVIDIA, so `.cuda()` is correct.

# %%
model = (
    ESMFold2Model.from_pretrained(MODEL_REPO, revision=MODEL_REVISION or None)
    .cuda()
    .eval()
)
model.config

# %% [markdown]
# ## Inspect
#
# A quick look at the hardware and model size, so the runtime is self-documenting.
# ESMFold2-Fast should report far fewer parameters than the full model.

# %%
n_params = sum(p.numel() for p in model.parameters())
{
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "parameters_B": round(n_params / 1e9, 3),
    "hf_home": os.environ["HF_HOME"],
}

# %% [markdown]
# ## Fold a single sequence
#
# We define one helper, `fold_sequence`, that wraps the whole single-chain pipeline
# (build input → fold → extract mmCIF + metrics). Defining it once keeps the single run
# below and the batch loop further down from duplicating logic. This cell does the actual
# (expensive) prediction for `SEQUENCE`.

# %%
def fold_sequence(seq: str, seed: int = SEED):
    """Fold one protein chain; return (mmcif_text, metrics_dict)."""
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=seq.strip())]
    )
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=NUM_LOOPS,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        seed=seed,
    )
    metrics = {
        "plddt_mean": float(result.plddt.mean()),
        "ptm": float(result.ptm),
    }
    return result.complex.to_mmcif(), metrics


cif_text, metrics = fold_sequence(SEQUENCE)
metrics

# %% [markdown]
# ## Visualize
#
# Render the predicted structure inline so we can sanity-check the fold without leaving
# the notebook.

# %%
view = py3Dmol.view(width=800, height=600)
view.addModel(cif_text, "cif")
view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()
view.show()

# %% [markdown]
# **Reading the view:** the cartoon is colored N→C terminus by the spectrum. Cross-check
# against the `pLDDT` / `pTM` numbers above — low-confidence regions are where the model
# is least sure. For per-residue confidence coloring open the exported `.cif` in the
# [Mol* Viewer](https://molstar.org/viewer).

# %% [markdown]
# ## (Optional) High-throughput batch
#
# ESMFold2-Fast's purpose is folding **many** sequences quickly. This cell reuses
# `fold_sequence` to fold a small list and tabulate the confidence metrics. Edit
# `BATCH_SEQUENCES`, or skip this cell entirely — it does not feed the export below.

# %%
BATCH_SEQUENCES = {
    "ubiquitin": (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ),
    "trp_cage": "NLYIQWLKDGGPSSGRPPPS",
}

batch_metrics = {}
for name, seq in BATCH_SEQUENCES.items():
    _, m = fold_sequence(seq)
    batch_metrics[name] = m
    print(f"{name}: pLDDT={m['plddt_mean']:.3f} pTM={m['ptm']:.3f}")

batch_metrics

# %% [markdown]
# ## Conclusion and export
#
# We write the single-sequence prediction to `OUTPUT_DIR` and trigger a browser download
# so you keep a local copy — `/content` is **ephemeral** and vanishes when the runtime
# resets (set `USE_DRIVE = True` in Setup to persist instead).

# %%
out_path = Path(OUTPUT_DIR) / "prediction.cif"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(cif_text)

from google.colab import files

files.download(str(out_path))
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

# %% [markdown]
# ### What we did
#
# We loaded ESMFold2-Fast and folded a single protein chain (plus an optional batch),
# reporting pLDDT / pTM and saving the structure as mmCIF.
#
# **Next steps**
# - **Throughput**: extend `BATCH_SEQUENCES` (or read sequences from a FASTA on Drive) to
#   fold hundreds of designs; collect the metrics into a table for ranking.
# - **Complexes & max accuracy**: switch to `esmfold2_colab.ipynb` (the full ESMFold2
#   model) for protein–DNA–ligand multimers and MSA conditioning.
# - **Inspect**: open any exported `.cif` in the [Mol* Viewer](https://molstar.org/viewer).
