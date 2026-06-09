# ESMFold2 on Modal

Fold proteins and biomolecular complexes with [ESMFold2](https://biohub.ai/esm/protein/about)
on [Modal](https://modal.com)'s serverless GPU infrastructure.

ESMFold2 is a state-of-the-art model for biomolecular complex structure
prediction, developed by [Biohub](https://biohub.ai/) and released under an open
license. Built on ESMC representations, it produces leading accuracy for
protein–protein and antibody–antigen interactions at any given compute budget.

This repository is a self-contained example that shows how to build a
`modal.Image` with the right dependencies, cache model weights on a
`modal.Volume`, run inference on an H100 GPU, and save the predicted structure
locally.

## Project structure

```
.
├── src/
│   └── esmfold2.py        # Modal app: image, weight caching, inference, CLI entrypoint
├── data/
│   └── prediction.cif     # Example output — the predicted M.HhaI / DNA / SAH complex
├── LICENSE                # Apache 2.0
└── README.md
```

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

## Usage

By default the example folds a protein–DNA–ligand complex — the M.HhaI DNA
methyltransferase from *Haemophilus haemolyticus* ([PDB 1MHT](https://www.rcsb.org/structure/1MHT))
bound to a methylated DNA duplex and its S-adenosyl-L-homocysteine (SAH)
cofactor — which exercises the model's full multimer capabilities:

```shell
modal run src/esmfold2.py
```

To fold a single protein chain, pass a sequence:

```shell
modal run src/esmfold2.py --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQA..."
```

To choose where the structure is written:

```shell
modal run src/esmfold2.py --output-path data/prediction.cif
```

The first run downloads the model weights to a Modal Volume (several minutes);
subsequent runs reuse the cached weights and start up much faster.

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
