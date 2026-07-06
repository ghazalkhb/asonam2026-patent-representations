# Auditing Semantic and Structural Representations for Patent Outcome Prediction in Heterogeneous Temporal Networks

**ASONAM 2026 — Camera-Ready Supplementary Code**

> G. Khodabandeh, L. Tahmooresnejad, N. Ezzati-Jivan, A. Ayanso  
> Brock University, St. Catharines, ON, Canada

[📄 Camera-ready PDF](paper/ASONAM2026_camera_ready.pdf)

---

## Overview

This repository contains the code and results for our ASONAM 2026 short paper. We present a controlled, leakage-aware benchmark of structural, semantic, metadata, and fused representations for patent outcome prediction using a large electric-vehicle (EV) patent knowledge graph.

**Three prediction tasks:**
- **4-year renewal** (`renewed_4y`) — binary maintenance-fee outcome
- **Triadic family membership** (`is_triadic`) — international patent family indicator
- **Citation impact** (`forward_citations_cat`) — three-class forward-citation discretization

**Four representation families:**
| Name | Description | Dim |
|------|-------------|-----|
| `TEXT` | all-mpnet-base-v2 Sentence-BERT on title + abstract | 768 |
| `KG` | node2vec on temporally masked heterogeneous KG | 64 |
| `META` | 9 grant-time bibliographic attributes | 9 |
| `NETFEAT` | 6 simple network statistics (no future leakage) | 6 |

Plus fusions: `HYBRID` (TEXT+KG), `KG+META`, `FULL` (TEXT+KG+META).

**Key findings:**
- Random splitting inflates renewal AUC by **6.6 pp** over temporal splitting.
- Structural (KG) embeddings outperform text on all three tasks under temporal evaluation.
- Metadata is a competitive low-cost baseline; fusion helps most for triadic prediction.

---

## Repository structure

```
.
├── code/
│   ├── asonam_pipeline.py   # Full reproducibility pipeline (Sections M1–M4, H1–H3)
│   └── smoke_test.py        # Quick data + model sanity check
├── results/                 # Pre-computed CSVs (reproduced by running the pipeline)
│   ├── main_results_aggregated.csv
│   ├── main_results_all_seeds.csv
│   ├── cold_warm_start.csv
│   ├── graph_structure_ablation.csv
│   ├── leakage_ablation.csv
│   ├── netfeat_baseline.csv
│   └── significance_tests.csv
├── figures/                 # Paper figures
│   ├── ai_framework_patent_analysis.png
│   ├── fig_main_results.pdf
│   ├── fig_coldwarm.pdf
│   └── fig_tsne_triadic.pdf
├── paper/
│   └── ASONAM2026_camera_ready.pdf
├── data/
│   └── README.md            # Instructions for obtaining the full dataset
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# Clone
git clone https://github.com/<your-org>/asonam2026-patent-representations.git
cd asonam2026-patent-representations

# Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Large data files are not included in this repository. See [data/README.md](data/README.md) for data sources and the expected folder layout.

Place the required files under `data/kg_exports/`:
```
data/kg_exports/
├── patent_full_table.parquet   # main feature table with KG + text embeddings
├── rels_cites.txt
├── rels_invented_by.txt
└── rels_assigned_to.txt
```

---

## Reproducing results

### Full pipeline

```bash
# Run all experiments (Main results, significance tests, cold/warm-start,
# leakage ablation, graph-structure ablation, network-feature baseline)
python code/asonam_pipeline.py --data-dir data/kg_exports

# Skip text embeddings for a much faster run (KG / META / NETFEAT only)
python code/asonam_pipeline.py --data-dir data/kg_exports --no-text

# Regenerate paper tables from existing CSVs (no re-training)
python code/asonam_pipeline.py --data-dir data/kg_exports --only-tables

# Custom output directory
python code/asonam_pipeline.py --data-dir data/kg_exports --out-dir my_results/
```

### Smoke test

```bash
python code/smoke_test.py --data-dir data/kg_exports
```

### Output files

All CSVs are written to the `--out-dir` (default: `results/`):

| File | Contents |
|------|----------|
| `main_results_all_seeds.csv` | Per-seed metrics for all representations × tasks × models |
| `main_results_aggregated.csv` | Mean ± std across 10 seeds |
| `significance_tests.csv` | Paired Wilcoxon tests (XGB, primary metric per task) |
| `cold_warm_start.csv` | AUC stratified by test-patent citation in-degree buckets |
| `leakage_ablation.csv` | Temporal vs. random split AUC comparison |
| `graph_structure_ablation.csv` | Citation-only vs. heterogeneous KG embeddings |
| `netfeat_baseline.csv` | Simple network-feature baseline vs. KG |

---

## Experimental setup

- **Temporal split:** 2014–2016 training (`n = 20,469`) → 2017 test (`n = 8,915`)
- **Models:** Logistic Regression (lbfgs, C=1) and XGBoost (100 trees, depth 6, lr 0.05, histogram)
- **Seeds:** 10 random seeds (XGB only; LR is deterministic on a fixed split)
- **Primary metrics:** AUC (renewal, triadic), Macro-F1 (citation impact)

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{khodabandeh2026auditing,
  title     = {Auditing Semantic and Structural Representations for Patent Outcome Prediction
               in Heterogeneous Temporal Networks},
  author    = {Khodabandeh, Ghazal and Tahmooresnejad, Leila and
               Ezzati-Jivan, Naser and Ayanso, Anteneh},
  booktitle = {Proceedings of the 2026 IEEE/ACM International Conference on
               Advances in Social Networks Analysis and Mining (ASONAM)},
  year      = {2026},
  publisher = {IEEE/ACM}
}
```

---

## License

Code is released under the [MIT License](LICENSE).  
The camera-ready PDF is © the authors and IEEE/ACM 2026; redistribution follows the conference's copyright policy.
