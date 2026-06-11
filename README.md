# ESMFold2

Fold proteins and biomolecular complexes with [ESMFold2](https://biohub.ai/esm/protein/about)
using free GPU compute on [Google Colab](https://colab.research.google.com/).

ESMFold2 is a state-of-the-art model for biomolecular complex structure
prediction, developed by [Biohub](https://biohub.ai/) and released under an open
license. Built on ESMC representations, it produces leading accuracy for
protein–protein and antibody–antigen interactions at any given compute budget.

## Model configurations

ESMFold2 is available in two configurations:

- **[ESMFold2](https://huggingface.co/biohub/ESMFold2)** — the larger model for
  maximum accuracy. It can be run from a single sequence or with MSA context,
  with MSAs improving performance on difficult complexes.
- **[ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast)** — a smaller
  model optimized for very fast single-sequence folding, well suited for
  high-throughput folding, designed sequences, metagenomic proteins, and targets
  with limited homologous sequence information.

## Run on Colab

Click a badge to open the notebook directly in Colab:

| Notebook | Model | Folds | Suggested runtime |
|---|---|---|---|
| [`esmfold2_colab`](src/notebooks/esmfold2_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/esmfold2/blob/main/src/notebooks/esmfold2_colab.ipynb) | full **ESMFold2** | a single protein chain by default (complexes supported via an optional cell) | A100 / High-RAM |
| [`esmfold2_fast_colab`](src/notebooks/esmfold2_fast_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/esmfold2/blob/main/src/notebooks/esmfold2_fast_colab.ipynb) | **ESMFold2-Fast** | a single protein sequence, plus an optional high-throughput batch | T4 (free tier) |

**One-time setup:**

1. **Turn on a GPU** — Runtime → Change runtime type → Hardware accelerator → GPU.
2. **Add your Hugging Face token** — the model weights are gated on the Hugging Face Hub
   and require a free account and access token:
   - Create a [Hugging Face account](https://huggingface.co/join) if you don't have one.
   - Accept the access terms on the [ESMFold2](https://huggingface.co/biohub/ESMFold2) and
     [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) model pages.
   - Create a read [access token](https://huggingface.co/settings/tokens).
   - In Colab, click the 🔑 **Secrets** panel in the left sidebar, add a secret named
     **`HF_TOKEN`** with your token as the value, and toggle on **Notebook access**.
     The notebook reads it automatically — nothing to paste into a cell.

Then **Runtime → Run all**. The notebook `pip install`s the `esm` package on the first run
(a couple of minutes) before downloading the model weights.

## Output

The notebook prints quality metrics (mean pLDDT, pTM, ipTM), renders an inline
[py3Dmol](https://github.com/3dmol/3Dmol.js) viewer, and writes the predicted structure as
a [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File)
(`.cif`) that you can download from Colab's file browser.

You can also render the `.cif` locally with the [Mol* Viewer](https://molstar.org/viewer).

![Folded complex in Mol* Viewer](https://modal-cdn.com/cdnbot/example-esmfold2-molviewerin7blk30_59122d5b.webp)

A sample prediction for the default M.HhaI / DNA / SAH complex is included at
[`data/prediction.cif`](data/prediction.cif).

## Advanced: Run on Modal

For running inference from the command line on [Modal](https://modal.com)'s cloud GPUs.

### Prerequisites

- A [Modal account](https://modal.com).
- The Modal CLI installed and authenticated:

  ```shell
  pip install modal
  modal setup
  ```

- A Hugging Face token (see above). Attach it to the `@app.cls` as the `HF_TOKEN`
  environment variable via a [Modal secret](https://modal.com/docs/guide/secrets).

### Usage

```shell
modal run src/scripts/esmfold2.py
```

```shell
modal run src/scripts/esmfold2.py --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQA..."
```

```shell
modal run src/scripts/esmfold2.py --output-path data/prediction.cif
```

The first run downloads the model weights to a Modal Volume (several minutes);
subsequent runs reuse the cached weights and start up much faster. The predicted
structure is saved to `/tmp/esmfold2/prediction.cif` by default.

### Project structure

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

## License

Apache 2.0 — see [LICENSE](LICENSE).
