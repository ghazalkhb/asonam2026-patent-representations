"""Quick smoke test — runs only data loading + one task/representation.

Usage (from repo root):
    python code/smoke_test.py --data-dir path/to/kg_exports

Requires patent_full_table.parquet to be present in the data dir.
"""
import argparse
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

import asonam_pipeline as p
import numpy as np
import pandas as pd


def _init_paths(data_dir: str):
    """Inject paths into the pipeline module globals."""
    p.BASE_KG      = Path(data_dir)
    p.PARQUET_FILE = p.BASE_KG / "patent_full_table.parquet"
    p.CITES_FILE   = p.BASE_KG / "rels_cites.txt"
    p.INV_FILE     = p.BASE_KG / "rels_invented_by.txt"
    p.ASSG_FILE    = p.BASE_KG / "rels_assigned_to.txt"
    p.OUT_DIR      = Path(__file__).parent.parent / "results"
    p.OUT_DIR.mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=str,
    default=str(Path(__file__).parent.parent / "data" / "kg_exports"),
    help="Path to KG exports folder (must contain patent_full_table.parquet)",
)
args = parser.parse_args()
_init_paths(args.data_dir)

print("=== SMOKE TEST ===")

# 1. Load data
print("\n[1] Loading data...")
df = p.load_parquet_selective(p.PARQUET_FILE)
df = p.coerce_numerics(df)
print(f"Shape: {df.shape}")

year = pd.to_numeric(df["year"], errors="coerce")
mask_tr = year.isin(p.TRAIN_YEARS)
mask_te = year == p.TEST_YEAR
df_tr = df[mask_tr].reset_index(drop=True)
df_te = df[mask_te].reset_index(drop=True)
print(f"Train: {len(df_tr)} | Test: {len(df_te)}")

# 2. KG matrix
print("\n[2] KG matrix...")
X_kg_tr = p.build_kg_matrix(df_tr)
X_kg_te = p.build_kg_matrix(df_te)
print(f"X_kg_tr: {X_kg_tr.shape}, X_kg_te: {X_kg_te.shape}")

# 3. Labels
print("\n[3] Labels...")
y_tr, _ = p.task_setup(df_tr, "renewed_4y")
y_te, _ = p.task_setup(df_te, "renewed_4y")
print(f"y_tr: {np.unique(y_tr, return_counts=True)}")
print(f"y_te: {np.unique(y_te, return_counts=True)}")

y_tri_tr, _ = p.task_setup(df_tr, "is_triadic")
y_tri_te, _ = p.task_setup(df_te, "is_triadic")
print(f"triadic train: {np.unique(y_tri_tr, return_counts=True)}")
print(f"triadic test: {np.unique(y_tri_te, return_counts=True)}")

y_cat_tr, _ = p.task_setup(df_tr, "forward_citations_cat")
y_cat_te, _ = p.task_setup(df_te, "forward_citations_cat")
print(f"citations_cat train: {np.unique(y_cat_tr, return_counts=True)}")
print(f"citations_cat test: {np.unique(y_cat_te, return_counts=True)}")

# 4. Quick model run
print("\n[4] Quick XGBoost run (KG -> renewed_4y)...")
X_tr_s, X_te_s = p.scale(X_kg_tr, X_kg_te)
mdl = p.make_xgb(0, 2)
mdl.fit(X_tr_s, y_tr)
prob = mdl.predict_proba(X_te_s)[:, 1]
metrics = p.compute_binary_metrics(y_te, prob)
print(f"Metrics: {metrics}")

# 5. Meta matrix
print("\n[5] Meta matrix...")
_, drop_cols = p.task_setup(df_tr, "renewed_4y")
X_meta_tr = p.build_meta_matrix(df_tr, drop_cols)
print(f"X_meta_tr (if available): {X_meta_tr.shape if X_meta_tr is not None else None}")

print("\n=== SMOKE TEST PASSED ===")
