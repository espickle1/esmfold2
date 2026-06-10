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
# # Fold proteins and biomolecular complexes with ESMFold2
#
# [ESMFold2](https://biohub.ai/esm/protein/about) is a state-of-the-art model for
# biomolecular complex structure prediction from [Biohub](https://biohub.ai/), built on
# ESMC representations. This notebook runs the **full ESMFold2** model — the larger,
# maximum-accuracy configuration — and by default folds a protein–DNA–ligand complex
# (the M.HhaI DNA methyltransferase bound to a methylated DNA duplex and its SAH
# cofactor, [PDB 1MHT](https://www.rcsb.org/structure/1MHT)) to exercise the model's
# full multimer capabilities. You can also fold any single protein chain by setting a
# sequence in the Config cell.
#
# This is a notebook port of a Modal app; the model code is unchanged — only the
# scaffolding is restructured so it runs top-to-bottom in Colab.
#
# > **Before you run anything:** select a GPU runtime via
# > **Runtime → Change runtime type → Hardware accelerator → GPU**. The full model is
# > large; an **A100 / High-RAM** runtime is recommended. A free-tier **T4** may run out
# > of memory on the default multimer (fold a single short sequence instead if so).

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

# Persist the (multi-GB) weight download across sessions by caching to Google Drive.
# Set USE_DRIVE = True to mount Drive; otherwise weights live in ephemeral /content and
# are re-downloaded each cold start. See the Config cell to change this.
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
# All imports, once. The `esm.models.esmfold2` classes describe the inputs (proteins,
# nucleic acids, ligands, modifications); `ESMFold2Model` is the model itself from the
# `transformers` fork; `py3Dmol` renders the predicted structure inline.

# %%
from pathlib import Path

import torch
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

import py3Dmol

# %% [markdown]
# ## Config
#
# A notebook has no command line, so every tunable value that the original
# `modal run ... --flags` exposed lives here as a plain assignment. The `# parameters`
# tag lets tools like Papermill override these for programmatic runs; the `#@param`
# annotations render as interactive widgets in Colab.
#
# Leave `SEQUENCE` empty to fold the default M.HhaI complex; set it to a single amino
# acid sequence to fold one protein chain instead.

# %% tags=["parameters"]
MODEL_REPO = "biohub/ESMFold2"  #@param {type:"string"}
MODEL_REVISION = "6234905"  #@param {type:"string"}  pinned for reproducibility

SEQUENCE = ""  #@param {type:"string"}  empty -> default protein/DNA/ligand complex

NUM_LOOPS = 3  #@param {type:"integer"}
NUM_SAMPLING_STEPS = 50  #@param {type:"integer"}
NUM_DIFFUSION_SAMPLES = 1  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}

# Output persistence: /content is fast but ephemeral; a Drive path survives sessions.
OUTPUT_DIR = "/content/esmfold2_out"  #@param {type:"string"}

# %% [markdown]
# ## Load model
#
# Loading downloads the weights to `HF_HOME` (several minutes on the **first** run; fast
# afterwards if cached to Drive) and moves the model onto the GPU. This is the most
# expensive setup step, so it gets its own cell — re-run it only when you change the
# model. On Colab the GPU is NVIDIA, so `.cuda()` is correct.

# %%
model = ESMFold2Model.from_pretrained(MODEL_REPO, revision=MODEL_REVISION).cuda().eval()
model.config

# %% [markdown]
# ## Inspect
#
# A quick look at the hardware we landed on and the model size, so the runtime is
# self-documenting (and so an out-of-memory failure later is easy to diagnose).

# %%
n_params = sum(p.numel() for p in model.parameters())
{
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "parameters_B": round(n_params / 1e9, 3),
    "hf_home": os.environ["HF_HOME"],
}

# %% [markdown]
# ## Build the input
#
# We assemble a `StructurePredictionInput`. If `SEQUENCE` is set we fold that single
# protein chain; otherwise we build the default **M.HhaI / DNA / SAH** complex — the
# cytosine-5 DNA methyltransferase from *Haemophilus haemolyticus* bound to a methylated
# DNA duplex (`C36` is the CCD code for 5-methylcytosine) and its
# S-adenosyl-L-homocysteine (`SAH`) cofactor.

# %%
MHHAI_SEQUENCE = (
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)

if SEQUENCE.strip():
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=SEQUENCE.strip())]
    )
else:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=MHHAI_SEQUENCE),
            DNAInput(
                id="B",
                sequence="GATAGCGCTATC",
                modifications=[Modification(position=5, ccd="C36")],
            ),
            DNAInput(
                id="C",
                sequence="TGATAGCGCTATC",
                modifications=[Modification(position=6, ccd="C36")],
            ),
            LigandInput(id="L", ccd=["SAH"]),
        ]
    )

[(s.id, type(s).__name__) for s in spi.sequences]

# %% [markdown]
# ## Fold
#
# The actual structure prediction. This is the expensive compute step (diffusion
# sampling over the structure), so it lives alone in its own cell — that way you can
# re-run downstream evaluation/visualization without paying to fold again. Higher
# `NUM_LOOPS` / `NUM_SAMPLING_STEPS` trade speed for accuracy.

# %%
result = ESMFold2InputBuilder().fold(
    model,
    spi,
    num_loops=NUM_LOOPS,
    num_sampling_steps=NUM_SAMPLING_STEPS,
    num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
    seed=SEED,
)

# %% [markdown]
# ## Evaluation
#
# ESMFold2 reports calibrated confidence metrics: **pLDDT** (mean per-residue
# confidence, 0–1), **pTM** (global fold confidence), and **ipTM** (interface
# confidence — only meaningful for complexes). Higher is better.

# %%
metrics = {
    "plddt_mean": float(result.plddt.mean()),
    "ptm": float(result.ptm),
    "iptm": float(result.iptm),
}
metrics

# %% [markdown]
# ## Visualize
#
# Render the predicted structure inline so we can sanity-check the fold without leaving
# the notebook. We export the result to mmCIF text and hand it to `py3Dmol`.

# %%
cif_text = result.complex.to_mmcif()

view = py3Dmol.view(width=800, height=600)
view.addModel(cif_text, "cif")
view.setStyle({"cartoon": {"color": "spectrum"}})
view.addStyle({"hetflag": True}, {"stick": {}})  # show the ligand / modified bases
view.zoomTo()
view.show()

# %% [markdown]
# **Reading the view:** the cartoon is colored N→C terminus by the spectrum; ligands and
# modified residues are drawn as sticks. Cross-check the fold against the `pLDDT`/`ipTM`
# numbers above — low-confidence regions are where the model is least sure. For a richer
# inspection (per-residue pLDDT coloring, measurements) open the exported `.cif` in the
# [Mol* Viewer](https://molstar.org/viewer).

# %% [markdown]
# ## Conclusion and export
#
# We fold once and produce: the confidence metrics above, an inline 3D view, and a
# portable mmCIF file. We write the `.cif` to `OUTPUT_DIR` and also trigger a browser
# download so you keep a copy locally — important because `/content` is **ephemeral** and
# vanishes when the runtime resets (set `USE_DRIVE = True` in Setup to persist instead).

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
# We loaded the full ESMFold2 model and folded the M.HhaI protein–DNA–ligand complex
# (or your single sequence), reporting pLDDT / pTM / ipTM and saving the structure as
# mmCIF.
#
# **Next steps**
# - **Scale up**: loop over many sequences, or raise `NUM_LOOPS` / `NUM_SAMPLING_STEPS`
#   for higher accuracy on difficult targets.
# - **MSA context**: ESMFold2 can take MSAs to improve accuracy on hard complexes.
# - **Speed**: for fast single-sequence / high-throughput folding, use the companion
#   `esmfold2_fast_colab.ipynb` notebook (ESMFold2-Fast).
# - **Design**: invert the model to design binders for a target.
