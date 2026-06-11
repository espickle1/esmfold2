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
# **New to this?** Just run the cells from top to bottom (**Runtime → Run all**). Each
# code cell has a short note above it explaining what it does — you don't need any prior
# setup, accounts, or installs on your own computer.
#
# > **Before you run anything:** turn on a GPU via
# > **Runtime → Change runtime type → Hardware accelerator → GPU**. The full model is
# > large; an **A100 / High-RAM** runtime is recommended. A free **T4** may run out of
# > memory on the default multimer (fold a single short sequence instead if so).

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
# editable boxes next to the code (that's what the `#@param` comments do). Keep
# **`FOLD_EXAMPLE_COMPLEX`** checked to fold the example M.HhaI / DNA / SAH complex; to
# fold your own single protein instead, uncheck it and type your sequence into the
# **`SEQUENCE`** box (edit the box, not the quotes in the code).

# %% tags=["parameters"]
MODEL = "biohub/ESMFold2"  #@param {type:"string"}

FOLD_EXAMPLE_COMPLEX = True  #@param {type:"boolean"}
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  #@param {type:"string"}

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
# We assemble a `StructurePredictionInput`. With **`FOLD_EXAMPLE_COMPLEX`** checked we
# build the default **M.HhaI / DNA / SAH** complex — the cytosine-5 DNA methyltransferase
# from *Haemophilus haemolyticus* bound to a methylated DNA duplex (`C36` is the CCD code
# for 5-methylcytosine) and its S-adenosyl-L-homocysteine (`SAH`) cofactor. Unchecked, we
# fold the single protein chain in `SEQUENCE`.

# %%
MHHAI_SEQUENCE = (
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)

if FOLD_EXAMPLE_COMPLEX:
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
else:
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=SEQUENCE.strip())]
    )

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
