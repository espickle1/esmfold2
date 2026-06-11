# ESMFold2

Fold proteins and biomolecular complexes with [ESMFold2](https://biohub.ai/esm/protein/about)
on [Modal](https://modal.com)'s and [Google Colab](https://colab.research.google.com/)'s serverless GPU infrastructure.

ESMFold2 is a state-of-the-art model for biomolecular complex structure
prediction, developed by [Biohub](https://biohub.ai/) and released under an open
license. Built on ESMC representations, it produces leading accuracy for
protein–protein and antibody–antigen interactions at any given compute budget.

This repository is a self-contained example that shows how to build a
`modal.Image` with the right dependencies, cache model weights on a
`modal.Volume`, run inference on an H100 GPU, and save the predicted structure
locally. It also ships two Google Colab notebook ports (see
[Run on Colab](#run-on-colab)) for running the same models on a browser-hosted GPU.

## Project structure

```
.
├── src/
│   ├── scripts/
│   │   └── esmfold2.py                 # Modal app: image, weight caching, inference, CLI entrypoint
│   └── notebooks/
│       ├── esmfold2_colab.py / .ipynb       # Colab: full ESMFold2 (single protein chain; complexes via optional cell)
│       └── esmfold2_fast_colab.py / .ipynb  # Colab: ESMFold2-Fast (single-sequence, high-throughput)
├── data/
│   └── prediction.cif     # Example output — the predicted M.HhaI / DNA / SAH complex
├── LICENSE                # Apache 2.0
└── README.md
```

Each Colab notebook is a [jupytext](https://jupytext.readthedocs.io/) percent-format
`.py` (the source of truth) paired with a generated `.ipynb` (what you open in Colab).

## Model configurations

ESMFold2 is available in two configurations:

- **[ESMFold2](https://huggingface.co/biohub/ESMFold2)** — the larger model for
  maximum accuracy. It can be run from a single sequence or with MSA context,
  with MSAs improving performance on difficult complexes.
- **[ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast)** — a smaller
  model optimized for very fast single-sequence folding, well suited for
  high-throughput folding, designed sequences, metagenomic proteins, and targets
  with limited homologous sequence information.

## Prerequisites

- A [Modal account](https://modal.com) (the GPU runs in Modal's cloud, not locally).
- The Modal CLI installed and authenticated:

  ```shell
  pip install modal
  modal setup
  ```

- **A Hugging Face account and access token.** The model weights are downloaded from the
  Hugging Face Hub and require authentication. Create a free account, accept the access
  terms on the [ESMFold2](https://huggingface.co/biohub/ESMFold2) and
  [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) model pages, then create a
  read [access token](https://huggingface.co/settings/tokens). For Colab, see
  [Run on Colab](#run-on-colab); for the Modal run, attach the token to the function as
  the `HF_TOKEN` environment variable via a [Modal secret](https://modal.com/docs/guide/secrets).

> **Note:** The Colab notebooks read the `HF_TOKEN` secret automatically (see
> [Run on Colab](#run-on-colab)). The Modal script does not yet consume a token, so
> folding the gated weights through it currently requires adding a
> [Modal secret](https://modal.com/docs/guide/secrets) to the `@app.cls` yourself.

## Usage

By default the example folds a protein–DNA–ligand complex — the M.HhaI DNA
methyltransferase from *Haemophilus haemolyticus* ([PDB 1MHT](https://www.rcsb.org/structure/1MHT))
bound to a methylated DNA duplex and its S-adenosyl-L-homocysteine (SAH)
cofactor — which exercises the model's full multimer capabilities:

```shell
modal run src/scripts/esmfold2.py
```

To fold a single protein chain, pass a sequence:

```shell
modal run src/scripts/esmfold2.py --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQA..."
```

To choose where the structure is written:

```shell
modal run src/scripts/esmfold2.py --output-path data/prediction.cif
```

The first run downloads the model weights to a Modal Volume (several minutes);
subsequent runs reuse the cached weights and start up much faster.

## Run on Colab

Prefer an interactive, browser-hosted GPU and no Modal account? Two notebook ports
live in [`src/notebooks/`](src/notebooks/). They strip the Modal scaffolding into
linear cells that run top-to-bottom, with a first-run setup cell, `#@param` config
widgets, an inline [py3Dmol](https://github.com/3dmol/3Dmol.js) viewer, and `.cif`
export.

| Notebook | Model | Folds | Suggested runtime |
|---|---|---|---|
| [`esmfold2_colab`](src/notebooks/esmfold2_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/esmc-esmfold2/blob/main/src/notebooks/esmfold2_colab.ipynb) | full **ESMFold2** | a single protein chain by default (complexes supported via an optional cell) | A100 / High-RAM |
| [`esmfold2_fast_colab`](src/notebooks/esmfold2_fast_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/esmc-esmfold2/blob/main/src/notebooks/esmfold2_fast_colab.ipynb) | **ESMFold2-Fast** | a single protein sequence, plus an optional high-throughput batch | T4 (free tier) |

**Before you run** (one-time):

1. **Turn on a GPU** — Runtime → Change runtime type → Hardware accelerator → GPU.
2. **Add your Hugging Face token** — the weights download from the Hugging Face Hub, which
   requires authentication. Click the 🔑 **Secrets** panel in Colab's left sidebar, add a
   secret named **`HF_TOKEN`** with your [access token](https://huggingface.co/settings/tokens)
   (see [Prerequisites](#prerequisites)) as the value, and toggle on **Notebook access**.
   The notebook reads it automatically — nothing to paste into a cell.

Then **Runtime → Run all**. The notebooks `pip install` the `esm` package on each cold
start (a couple of minutes) before downloading the model.

## Output

The run prints quality metrics (mean pLDDT, pTM, ipTM) and writes the predicted
structure as a [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File)
(`.cif`). By default it is saved to `/tmp/esmfold2/prediction.cif`; pass
`--output-path` to override. A sample prediction for the default complex is
included at [`data/prediction.cif`](data/prediction.cif).

You can render the `.cif` file with the [Mol* Viewer](https://molstar.org/viewer).

![Folded complex in Mol* Viewer](https://modal-cdn.com/cdnbot/example-esmfold2-molviewerin7blk30_59122d5b.webp)

## License

Apache 2.0 — see [LICENSE](LICENSE).
