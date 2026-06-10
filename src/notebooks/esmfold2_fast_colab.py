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
# **ESMFold2-Fast** is the smaller, inference-optimized configuration of
# [ESMFold2](https://biohub.ai/esm/protein/about) from [Biohub](https://biohub.ai/). It
# is tuned for **very fast single-sequence folding** — well suited to high-throughput
# folding, designed sequences, metagenomic proteins, and targets with limited homology.
# This notebook folds one protein chain by default and includes an optional batch cell
# that folds several sequences in a loop.
#
# **New to this?** Just run the cells from top to bottom (**Runtime → Run all**). Each
# code cell has a short note above it explaining what it does — no prior setup, accounts,
# or local installs needed. For complexes (protein + DNA + ligand) and maximum accuracy,
# use the companion `esmfold2_colab.ipynb` notebook (the full model).
#
# > **Before you run anything:** turn on a GPU via
# > **Runtime → Change runtime type → Hardware accelerator → GPU**. ESMFold2-Fast is
# > small, so a free **T4** is plenty.

# %% [markdown]
# ## Setup
#
# Colab gives you a fresh, empty machine each session and forgets everything when the
# session ends. So the first cell installs the software we need. **Run it first** and let
# it finish before running anything else — installing takes a couple of minutes, which is
# normal.

# %%
# Installs ESMFold2 and a small 3D viewer. Runs once per session — wait for it to finish.
# !pip install -q "esm @ git+https://github.com/Biohub/esm.git@81b3646c9429ea8458918415ad6a46178cb59833" py3Dmol

# %% [markdown]
# ## Imports
#
# Load the pieces we need: tools to describe the protein we want to fold, the
# ESMFold2-Fast model itself, and `py3Dmol` to draw the result in 3D.

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
# Everything you might want to change is gathered here. In Colab these show up as little
# editable boxes next to the code (that's what the `#@param` comments do). `SEQUENCE` is
# one protein chain — here, the M.HhaI methyltransferase as an example. To fold your own,
# type it into the **`SEQUENCE`** box (edit the box, not the quotes in the code).

# %% tags=["parameters"]
MODEL = "biohub/ESMFold2-Fast"  #@param {type:"string"}

SEQUENCE = "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDHDILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELDYSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTNQEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDSYKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"  #@param {type:"string"}

NUM_LOOPS = 3  #@param {type:"integer"}
NUM_SAMPLING_STEPS = 50  #@param {type:"integer"}
NUM_DIFFUSION_SAMPLES = 1  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}

OUTPUT_DIR = "/content/esmfold2_fast_out"  #@param {type:"string"}

# %% [markdown]
# ## Download and load the model
#
# This line fetches the trained model from the internet the **first** time you run it and
# loads it onto the GPU. ESMFold2-Fast is small, so this is quick. It then stays in memory
# for the rest of the session, so you only wait once.

# %%
model = ESMFold2Model.from_pretrained(MODEL).cuda().eval()
model.config

# %% [markdown]
# ## Inspect
#
# A quick look at the GPU we got and the model's size, so the runtime is self-documenting.
# ESMFold2-Fast should report far fewer parameters than the full model.

# %%
n_params = sum(p.numel() for p in model.parameters())
{
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "parameters_B": round(n_params / 1e9, 3),
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
# We write the single-sequence prediction to a `.cif` file. Colab deletes everything when
# the session ends, so we also download the `.cif` to your own computer to keep it.

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
# - **Throughput**: extend `BATCH_SEQUENCES` (or read sequences from a FASTA file) to fold
#   hundreds of designs; collect the metrics into a table for ranking.
# - **Complexes & max accuracy**: switch to `esmfold2_colab.ipynb` (the full ESMFold2
#   model) for protein–DNA–ligand multimers and MSA conditioning.
# - **Inspect**: open any exported `.cif` in the [Mol* Viewer](https://molstar.org/viewer).
