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
# maximum-accuracy configuration — and by default folds a **single protein chain** from a
# sequence you provide in the Config cell. It is still the full, multimer-capable model,
# so it can also fold protein–DNA–ligand complexes (e.g. the M.HhaI DNA methyltransferase
# bound to a methylated DNA duplex and its SAH cofactor, [PDB 1MHT](https://www.rcsb.org/structure/1MHT)) —
# see the optional example in the "Build the input" cell.
#
# **New to this?** Just run the cells from top to bottom (**Runtime → Run all**). Each
# code cell has a short note above it explaining what it does — you don't need any prior
# setup, accounts, or installs on your own computer.
#
# > **Before you run anything:** turn on a GPU via
# > **Runtime → Change runtime type → Hardware accelerator → GPU**. The full model is
# > large; an **A100 / High-RAM** runtime is recommended. A free **T4** may run out of
# > memory on long sequences (try a shorter one if so).

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
# Load the pieces we need: tools to describe the molecules we want to fold, the ESMFold2
# model itself, and `py3Dmol` to draw the result in 3D.

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
# ## Authenticate with Hugging Face
#
# The ESMFold2 weights live on the Hugging Face Hub and are **gated**, so you need to
# identify yourself before the download cell below can fetch them. Add your access token
# once as a Colab **secret** named `HF_TOKEN` (the 🔑 panel in the left sidebar — see the
# README for the exact steps); this cell reads it automatically, with nothing to paste here.

# %%
from google.colab import userdata
from huggingface_hub import login

try:
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    raise RuntimeError(
        "Couldn't read the 'HF_TOKEN' secret. Open the 🔑 Secrets panel in Colab's "
        "left sidebar, add a secret named HF_TOKEN with your Hugging Face access token, "
        "and toggle on 'Notebook access'. See the README for details."
    )

login(hf_token)

# %% [markdown]
# ## Config
#
# Everything you might want to change is gathered here. In Colab these show up as little
# editable boxes next to the code (that's what the `#@param` comments do). The
# **`SEQUENCE`** box starts empty — type one protein chain into it (edit the box, not the
# quotes in the code). To fold a complex instead, see the optional example in the "Build
# the input" cell.

# %% tags=["parameters"]
MODEL = "biohub/ESMFold2"  #@param {type:"string"}

SEQUENCE = ""  #@param {type:"string"}

NUM_LOOPS = 3  #@param {type:"integer"}
NUM_SAMPLING_STEPS = 50  #@param {type:"integer"}
NUM_DIFFUSION_SAMPLES = 1  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}

OUTPUT_DIR = "/content/esmfold2_out"  #@param {type:"string"}

# %% [markdown]
# ## Download and load the model
#
# This line fetches the trained model from the internet the **first** time you run it
# (the full model is several gigabytes, so the first run takes a few minutes) and loads
# it onto the GPU. It then stays in memory for the rest of the session, so you only wait
# once. If this cell seems slow, that's the download — let it finish.

# %%
model = ESMFold2Model.from_pretrained(MODEL).cuda().eval()
model.config

# %% [markdown]
# ## Inspect
#
# A quick look at the GPU we got and the model's size, so the runtime is self-documenting
# (and so an out-of-memory error later is easy to diagnose).

# %%
n_params = sum(p.numel() for p in model.parameters())
{
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "parameters_B": round(n_params / 1e9, 3),
}

# %% [markdown]
# ## Build the input
#
# We assemble a `StructurePredictionInput` from the single protein chain in `SEQUENCE`. If
# the box is empty we stop with a clear message, so a top-to-bottom **Run all** fails right
# here rather than deep inside the model. The optional commented block shows how to fold a
# protein–DNA–ligand complex instead — the **M.HhaI / DNA / SAH** example (the cytosine-5
# DNA methyltransferase from *Haemophilus haemolyticus* bound to a methylated DNA duplex,
# where `C36` is the CCD code for 5-methylcytosine, and its `SAH` cofactor, PDB 1MHT).

# %%
if not SEQUENCE.strip():
    raise ValueError(
        "SEQUENCE is empty. Type a protein sequence into the SEQUENCE box in the "
        "Config cell above, then re-run this cell. (To fold a complex instead, "
        "uncomment and edit the example block below.)"
    )

spi = StructurePredictionInput(
    sequences=[ProteinInput(id="A", sequence=SEQUENCE.strip())]
)

# --- Optional: fold a protein–DNA–ligand complex instead ----------------------
# The full ESMFold2 model also folds multimers. As an example, to fold the M.HhaI
# DNA methyltransferase bound to a methylated DNA duplex (`C36` = 5-methylcytosine)
# and its SAH cofactor (PDB 1MHT), comment out the single-chain `spi` above and
# uncomment this block:
#
# MHHAI_SEQUENCE = (
#     "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
#     "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
#     "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
#     "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
#     "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
# )
# spi = StructurePredictionInput(
#     sequences=[
#         ProteinInput(id="A", sequence=MHHAI_SEQUENCE),
#         DNAInput(id="B", sequence="GATAGCGCTATC",
#                  modifications=[Modification(position=5, ccd="C36")]),
#         DNAInput(id="C", sequence="TGATAGCGCTATC",
#                  modifications=[Modification(position=6, ccd="C36")]),
#         LigandInput(id="L", ccd=["SAH"]),
#     ]
# )

[(s.id, type(s).__name__) for s in spi.sequences]

# %% [markdown]
# ## Fold
#
# The actual structure prediction. This is the expensive compute step (diffusion
# sampling over the structure), so it lives alone in its own cell — that way you can
# re-run the evaluation/visualization below without paying to fold again. Higher
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
# confidence, 0–1) and **pTM** (global fold confidence). For complexes it also reports
# **ipTM** (interface confidence), which we add below only when folding more than one
# chain. Higher is better.

# %%
metrics = {
    "plddt_mean": float(result.plddt.mean()),
    "ptm": float(result.ptm),
}
if len(spi.sequences) > 1:  # ipTM (interface confidence) only applies to complexes
    metrics["iptm"] = float(result.iptm)
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
# modified residues are drawn as sticks. Cross-check the fold against the `pLDDT`/`pTM`
# numbers above (and `ipTM` for complexes) — low-confidence regions are where the model is
# least sure. For a richer inspection (per-residue pLDDT coloring, measurements) open the
# exported `.cif` in the [Mol* Viewer](https://molstar.org/viewer).

# %% [markdown]
# ## Conclusion and export
#
# We fold once and produce the confidence metrics above, an inline 3D view, and a
# portable `.cif` structure file. Colab deletes everything when the session ends, so we
# also download the `.cif` to your own computer to keep it.

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
# We loaded the full ESMFold2 model and folded a single protein chain (or a complex, if
# you uncommented the optional block), reporting pLDDT / pTM (plus ipTM for complexes) and
# saving the structure as mmCIF.
#
# **Next steps**
# - **Scale up**: loop over many sequences, or raise `NUM_LOOPS` / `NUM_SAMPLING_STEPS`
#   for higher accuracy on difficult targets.
# - **MSA context**: ESMFold2 can take MSAs to improve accuracy on hard complexes.
# - **Speed**: for fast single-sequence / high-throughput folding, use the companion
#   `esmfold2_fast_colab.ipynb` notebook (ESMFold2-Fast).
# - **Design**: invert the model to design binders for a target.
