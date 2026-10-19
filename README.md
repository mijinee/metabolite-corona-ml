# Chemically Engineered Carbon Nanotubes Map Class-Selective Metabolite Enrichment from Human Plasma

Analysis code for LC-MS/MS metabolomics of carbon nanotube coronas formed in human plasma.

## Usage

```bash
pip install -r requirements.txt
```

Set input paths in `config.py`, then run the scripts from the repository root in folder order:

1. `00_preprocessing` – combine Compound Discoverer exports; filtering, imputation, normalization
2. `01_classification` – metabolite class prediction for unannotated features
3. `02_ml_evaluation` – classifier performance and LIPID MAPS external validation
4. `03_log2fc` – corona vs plasma and QWD vs pristine fold changes
5. `04_batch_reproducibility` – inter-batch reproducibility
6. `05_figures` – figure data tables
7. `06_pathway_analysis` – mummichog pathway enrichment
8. `07_supplementary` – supplementary data and variance partition

Outputs are written to `output/`.

## Citation

If you use this code, please cite the accompanying paper.

## License

MIT (see `LICENSE`).
