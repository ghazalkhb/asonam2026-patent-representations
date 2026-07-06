"""
ASONAM 2026 — Clean Pipeline
Auditing Semantic and Structural Representations for Patent Outcome Prediction

Addresses every mandatory and high-value reviewer request:
  [M1] Rerun main results table from a single clean pipeline
  [M2] Cold/warm-start with clearly defined degree buckets, all buckets reported
  [M3] Significance tests: KG vs TEXT, KG vs HYBRID, KG+META vs FULL
  [M4] Renewal PR-AUC, balanced accuracy, class-specific recall/precision
  [H1] Leakage inflation ablation: temporal split vs random split
  [H2] Graph structure ablation: citation-only vs heterogeneous KG
  [H3] Simple network-feature baseline

Run from ANY directory — all paths are absolute.

Usage:
    python asonam_pipeline.py [--no-mpl]     # --no-mpl skips matplotlib plots
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  — set at runtime via --data-dir / --out-dir (see main())
#           Defaults: ./data/kg_exports  and  ./results
# ──────────────────────────────────────────────────────────────────────────────
BASE_KG      = None  # Path to KG exports folder (patent_full_table.parquet, rels_*.txt)
PARQUET_FILE = None
CITES_FILE   = None
INV_FILE     = None
ASSG_FILE    = None

OUT_DIR = None  # Results output directory

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────────
KG_DIM   = 64
KG_COLS  = [f"kg_emb_{i}" for i in range(KG_DIM)]

TRAIN_YEARS = [2014, 2015, 2016]
TEST_YEAR   = 2017

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Metadata columns (no target leakage)
# drop lists are per-task below
META_ALL = [
    "num_claims", "processing_days", "priority_count",
    "backward_citations_count", "family_size", "grant_lag_days",
    "team_size", "female_count", "pct_female",
]

# Network-feature columns (structural statistics only, no embeddings)
NETFEAT_COLS = [
    # cite_indegree EXCLUDED: it equals forward citation count -> direct label leakage
    # for forward_citations_cat, and uses future edges for renewal/triadic tasks.
    # Only temporally-safe, graph-computed features are kept.
    "cite_outdegree",
    "inventor_count", "assignee_count",
    "family_size", "priority_count",
    "team_size",
]

# Degree buckets for cold/warm-start (citation in-degree at test cutoff)
DEGREE_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 9999)]
BUCKET_LABELS  = ["0", "1-2", "3-5", "6-10", "11+"]

N_BOOTSTRAP = 1000
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_parquet_selective(path: Path, load_text: bool = True) -> pd.DataFrame:
    """
    Memory-efficient load: reads metadata + KG columns first,
    then text embeddings only if load_text=True.
    """
    print(f"[DATA] Reading schema from {path} ...")
    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    all_cols = [schema.field(i).name for i in range(len(schema))]

    # Select metadata + KG embedding columns (exclude string embeddings and large text)
    meta_cols = [c for c in all_cols
                 if not c.startswith("txt_emb")
                 and c not in ("abstract_text", "text_for_embedding",
                               "patent_title", "priority_countries",
                               "priority_years", "kg_emb_str", "kg_emb")]
    print(f"[DATA] Loading {len(meta_cols)} columns (metadata + KG emb) ...")
    df = pf.read(columns=meta_cols).to_pandas()
    print(f"[DATA] Loaded: {df.shape}")

    if not load_text:
        print("[DATA] Skipping text embeddings (--no-text).")
        return df

    # Load text embeddings (stored as stringified vectors)
    print("[DATA] Loading txt_emb column (slow — string parsing for 91K rows) ...")
    print("[DATA] Tip: use --no-text to skip and run KG/META experiments only.")
    txt_col = "txt_emb" if "txt_emb" in all_cols else (
              "txt_emb_str" if "txt_emb_str" in all_cols else None)
    if txt_col:
        try:
            txt_tbl = pf.read(columns=["patent_number", txt_col]).to_pandas()
            df = df.merge(txt_tbl[["patent_number", txt_col]],
                          on="patent_number", how="left")
            print(f"[DATA] txt_emb ({txt_col}) merged.")
        except Exception as e:
            print(f"[DATA][WARN] txt_emb load failed: {e}")
    return df


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce string-typed numeric columns to float."""
    for col in ["forward_citations_count", "backward_citations_count",
                "team_size", "family_size", "num_claims", "processing_days",
                "grant_lag_days", "priority_count", "female_count", "pct_female",
                "num_of_maint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_map = {True: 1, False: 0, "True": 1, "False": 0,
                  "true": 1, "false": 0, 1.0: 1, 0.0: 0, "1": 1, "0": 0}
    for col in ["is_triadic", "ev_relation", "renewed_4y",
                "renewed_4y_strict", "renewed_8y", "renewed_12y"]:
        if col in df.columns:
            df[col] = df[col].replace(binary_map)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def parse_text_emb(series: pd.Series) -> np.ndarray:
    """
    Parse txt_emb column (stored as stringified vectors).
    Uses parallel processing for speed.
    Returns float32 matrix (n, dim).
    """
    import ast
    from concurrent.futures import ThreadPoolExecutor

    def _parse_one(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        if isinstance(x, list):
            return np.array(x, dtype=np.float32)
        if isinstance(x, str):
            # Fast path: space-separated numbers (most common format)
            x2 = x.strip()
            if x2.startswith("["):
                x2 = x2[1:]
            if x2.endswith("]"):
                x2 = x2[:-1]
            parts = x2.replace(",", " ").split()
            if parts:
                try:
                    return np.fromiter((float(p) for p in parts),
                                       dtype=np.float32, count=len(parts))
                except ValueError:
                    pass
        return np.zeros(768, dtype=np.float32)

    vals = series.values
    n = len(vals)
    print(f"[FEAT] Parsing {n} text embeddings (parallel) ...")

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_parse_one, vals))

    matrix = np.vstack(results)
    print(f"[FEAT] Text embedding matrix: {matrix.shape}")
    return matrix.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GRAPH STATS (degree, inventor/assignee counts)
# ──────────────────────────────────────────────────────────────────────────────

def compute_graph_degrees(patent_ids: set) -> dict:
    """
    Reads rels_cites.txt, rels_invented_by.txt, rels_assigned_to.txt
    and returns per-patent-number degree stats.
    """
    print("[GRAPH] Computing citation degrees ...")

    def read_rel(path, col_src="src_id", col_dst="dst_id"):
        df = pd.read_csv(str(path), sep=",", skipinitialspace=True,
                         engine="python", on_bad_lines="warn")
        df.columns = [c.strip().strip('"') for c in df.columns]
        # rename if needed
        if "src_id" not in df.columns and len(df.columns) >= 2:
            df.columns = ["src_id", "dst_id"] + list(df.columns[2:])
        df["src_id"] = df["src_id"].astype(str).str.strip().str.strip('"')
        df["dst_id"] = df["dst_id"].astype(str).str.strip().str.strip('"')
        return df

    cite_df = read_rel(CITES_FILE)
    inv_df  = read_rel(INV_FILE)
    assg_df = read_rel(ASSG_FILE)

    # citation in-degree (how many others cite this patent within training window)
    # For the full degree analysis, we use ALL edges; cold/warm uses training cutoff
    cite_indeg = cite_df.groupby("dst_id").size().rename("cite_indegree")
    cite_outdeg = cite_df.groupby("src_id").size().rename("cite_outdegree")

    inv_count = inv_df.groupby("src_id").size().rename("inventor_count")
    assg_count = assg_df.groupby("src_id").size().rename("assignee_count")

    degree_df = pd.DataFrame(index=list(patent_ids))
    degree_df.index.name = "patent_number"
    degree_df = degree_df.join(cite_indeg,  how="left")
    degree_df = degree_df.join(cite_outdeg, how="left")
    degree_df = degree_df.join(inv_count,   how="left")
    degree_df = degree_df.join(assg_count,  how="left")
    degree_df = degree_df.fillna(0).astype(float)
    degree_df = degree_df.reset_index()

    # citation in-degree computed using only training edges (for cold/warm split)
    # edges where src_id is a training patent
    print("[GRAPH] Degrees computed.")
    return degree_df, cite_df


def compute_train_indegree(cite_df: pd.DataFrame, train_ids: set) -> pd.Series:
    """
    For cold/warm-start: number of training-era patents cited by each test patent.
    (How many of a test patent's backward citations point into the training graph.)
    Higher = more connected to training knowledge graph = warm-start.
    Note: train->test direction would always be 0 since test patents post-date training.
    """
    edges_to_train = cite_df[cite_df["dst_id"].isin(train_ids)]
    score = edges_to_train.groupby("src_id").size()
    return score


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE MATRICES
# ──────────────────────────────────────────────────────────────────────────────

def build_kg_matrix(df: pd.DataFrame) -> np.ndarray:
    avail_kg = [c for c in KG_COLS if c in df.columns]
    if len(avail_kg) < KG_DIM:
        print(f"[FEAT][WARN] Only {len(avail_kg)}/{KG_DIM} KG columns found.")
    X = df[avail_kg].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X


def build_text_matrix(df: pd.DataFrame) -> np.ndarray:
    if "txt_emb" in df.columns and df["txt_emb"].notna().any():
        return parse_text_emb(df["txt_emb"])
    if "txt_emb_str" in df.columns and df["txt_emb_str"].notna().any():
        return parse_text_emb(df["txt_emb_str"])
    raise ValueError("No text embedding column found (txt_emb / txt_emb_str).")


def build_meta_matrix(df: pd.DataFrame, drop_cols: list) -> np.ndarray:
    cols = [c for c in META_ALL if c in df.columns and c not in drop_cols]
    if not cols:
        return None
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0).values.astype(np.float32)
    return X


def build_netfeat_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in NETFEAT_COLS if c in df.columns]
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0).values.astype(np.float32)
    return X


def scale(X_tr: np.ndarray, X_te: np.ndarray):
    sc = StandardScaler()
    return sc.fit_transform(X_tr).astype(np.float32), \
           sc.transform(X_te).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LABELS & TASK UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def make_citation_cats(y_raw: pd.Series) -> np.ndarray:
    """
    Bin forward_citations_count into 3 categories used in paper:
    0 = zero (no citations), 1 = low [1-2], 2 = high [3+]
    Consistent with the bins in RQ3_label_stats.json.
    """
    # Use quantile-based: bottom 50%, next 25%, top 25%
    vals = y_raw.astype(float).fillna(0)
    bins = [vals.min() - 1, 0, 2, vals.max() + 1]
    cats = pd.cut(vals, bins=bins, labels=[0, 1, 2]).astype(int)
    return cats.values


def task_setup(df: pd.DataFrame, task: str):
    """
    Returns (y, drop_meta_cols) for each task.
    """
    if task == "renewed_4y":
        y = df["renewed_4y"].astype(int).values
        drop = ["num_of_maint", "renewed_4y", "renewed_4y_strict",
                "renewed_8y", "renewed_12y", "forward_citations_count",
                "is_triadic"]
    elif task == "is_triadic":
        y = df["is_triadic"].astype(int).values
        drop = ["is_triadic", "forward_citations_count"]
    elif task == "forward_citations_cat":
        y = make_citation_cats(df["forward_citations_count"])
        drop = ["forward_citations_count"]
    else:
        raise ValueError(f"Unknown task: {task}")
    return y, drop


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full metric set for binary classification."""
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    balacc = balanced_accuracy_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # Per-class precision / recall
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)

    return {
        "AUC": float(auc),
        "PR_AUC": float(prauc),
        "BalAcc": float(balacc),
        "Accuracy": float(acc),
        "F1": float(f1),
        "Precision_0": float(prec[0]),
        "Recall_0": float(rec[0]),
        "Precision_1": float(prec[1]),
        "Recall_1": float(rec[1]),
    }


def compute_multiclass_metrics(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_classes: int = 3) -> dict:
    """Macro-F1, balanced accuracy, per-class recall for multi-class."""
    y_pred = y_prob.argmax(axis=1)
    balacc = balanced_accuracy_score(y_true, y_pred)
    macf1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc    = accuracy_score(y_true, y_pred)

    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)

    d = {"BalAcc": float(balacc), "MacroF1": float(macf1), "Accuracy": float(acc)}
    for c in range(n_classes):
        d[f"Precision_{c}"] = float(prec[c])
        d[f"Recall_{c}"]    = float(rec[c])
    return d


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn, n_boot: int = N_BOOTSTRAP,
                 rng: np.random.Generator = None) -> tuple:
    """95 % bootstrap CI for a scalar metric."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            pass
    if not scores:
        return np.nan, np.nan
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODELS
# ──────────────────────────────────────────────────────────────────────────────

def make_lr(seed: int, n_classes: int):
    return LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs",
        random_state=seed, n_jobs=-1)


def make_xgb(seed: int, n_classes: int):
    if n_classes == 2:
        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", objective="binary:logistic",
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
    else:
        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", objective="multi:softprob",
            num_class=n_classes,
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")


def run_model(model, X_tr, y_tr, X_te, y_te, task: str) -> dict:
    """Fit model and return full metrics dict."""
    model.fit(X_tr, y_tr)

    if task in ("renewed_4y", "is_triadic"):
        y_prob = model.predict_proba(X_te)[:, 1]
        return compute_binary_metrics(y_te, y_prob), y_prob
    else:  # forward_citations_cat
        y_prob_mat = model.predict_proba(X_te)  # (n, 3)
        return compute_multiclass_metrics(y_te, y_prob_mat), y_prob_mat


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN RESULTS TABLE  [M1, M4]
# ──────────────────────────────────────────────────────────────────────────────

def run_main_experiments(df: pd.DataFrame, tasks: list) -> pd.DataFrame:
    """
    Runs the core comparison: TEXT / KG / HYBRID / META / KG+META / FULL
    over all tasks and 5 seeds with LR + XGBoost.
    Also includes NETFEAT baseline.
    Saves per-seed results and produces the aggregated table.
    """
    print("\n" + "="*70)
    print("SECTION: MAIN RESULTS TABLE [M1, M4]")
    print("="*70)

    # temporal split
    year = pd.to_numeric(df["year"], errors="coerce")
    mask_tr = year.isin(TRAIN_YEARS)
    mask_te = year == TEST_YEAR
    df_tr = df[mask_tr].reset_index(drop=True)
    df_te = df[mask_te].reset_index(drop=True)
    print(f"  Train: {mask_tr.sum():,} | Test: {mask_te.sum():,}")

    # build fixed feature matrices
    X_kg_tr  = build_kg_matrix(df_tr)
    X_kg_te  = build_kg_matrix(df_te)

    has_text = False
    try:
        X_txt_tr_raw = build_text_matrix(df_tr)
        X_txt_te_raw = build_text_matrix(df_te)
        has_text = True
        print(f"  Text embeddings: {X_txt_tr_raw.shape[1]}d")
    except Exception as e:
        print(f"  [WARN] Text embeddings unavailable: {e}")

    all_rows = []

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        y_tr, drop_cols = task_setup(df_tr, task)
        y_te, _         = task_setup(df_te, task)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))

        X_meta_tr = build_meta_matrix(df_tr, drop_cols)
        X_meta_te = build_meta_matrix(df_te, drop_cols)

        X_nf_tr = build_netfeat_matrix(df_tr)
        X_nf_te = build_netfeat_matrix(df_te)

        # scale each representation
        kg_tr_s, kg_te_s = scale(X_kg_tr, X_kg_te)

        reps = {"KG": (kg_tr_s, kg_te_s)}

        if has_text:
            txt_tr_s, txt_te_s = scale(X_txt_tr_raw, X_txt_te_raw)
            reps["TEXT"]   = (txt_tr_s, txt_te_s)
            hyb_tr_s = np.hstack([txt_tr_s, kg_tr_s])
            hyb_te_s = np.hstack([txt_te_s, kg_te_s])
            reps["HYBRID"] = (hyb_tr_s, hyb_te_s)

        if X_meta_tr is not None:
            meta_tr_s, meta_te_s = scale(X_meta_tr, X_meta_te)
            reps["META"] = (meta_tr_s, meta_te_s)
            kg_meta_tr = np.hstack([kg_tr_s, meta_tr_s])
            kg_meta_te = np.hstack([kg_te_s, meta_te_s])
            reps["KG+META"] = (kg_meta_tr, kg_meta_te)

            if has_text:
                full_tr = np.hstack([txt_tr_s, kg_tr_s, meta_tr_s])
                full_te = np.hstack([txt_te_s, kg_te_s, meta_te_s])
                reps["FULL"] = (full_tr, full_te)

        if X_nf_tr is not None and X_nf_tr.shape[1] > 0:
            nf_tr_s, nf_te_s = scale(X_nf_tr, X_nf_te)
            reps["NETFEAT"] = (nf_tr_s, nf_te_s)

        for rep_name, (X_tr_r, X_te_r) in reps.items():
            for seed in SEEDS:
                for model_name, model_fn in [
                    ("LR",  make_lr(seed, n_classes)),
                    ("XGB", make_xgb(seed, n_classes)),
                ]:
                    try:
                        metrics, _ = run_model(model_fn, X_tr_r, y_tr,
                                               X_te_r, y_te, task)
                        row = {
                            "task": task,
                            "representation": rep_name,
                            "model": model_name,
                            "seed": seed,
                            **metrics,
                        }
                        all_rows.append(row)
                        print(f"  {task} | {rep_name:12s} | {model_name} | seed {seed} |"
                              f" AUC {metrics.get('AUC', metrics.get('MacroF1', '?')):.4f}")
                    except Exception as e:
                        print(f"  [ERROR] {task} | {rep_name} | {model_name} | seed {seed}: {e}")

    df_res = pd.DataFrame(all_rows)
    df_res.to_csv(OUT_DIR / "main_results_all_seeds.csv", index=False)

    # aggregate: mean ± std across seeds
    agg_metric = {
        "renewed_4y":           "AUC",
        "is_triadic":           "AUC",
        "forward_citations_cat":"MacroF1",
    }
    agg_rows = []
    for (task, rep, mdl), grp in df_res.groupby(["task", "representation", "model"]):
        met = agg_metric.get(task, "AUC")
        row = {"task": task, "representation": rep, "model": mdl}
        for col in ["AUC", "PR_AUC", "BalAcc", "MacroF1",
                    "Recall_0", "Recall_1", "Precision_0", "Precision_1"]:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_std"]  = grp[col].std()
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUT_DIR / "main_results_aggregated.csv", index=False)
    print(f"\n[M1] Main results saved -> {OUT_DIR / 'main_results_all_seeds.csv'}")
    print(f"[M1] Aggregated table  -> {OUT_DIR / 'main_results_aggregated.csv'}")
    return df_res


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SIGNIFICANCE TESTS  [M3]
# ──────────────────────────────────────────────────────────────────────────────

def run_significance_tests(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Per-task, per-model: paired Wilcoxon signed-rank test (10 seeds, XGB only for inference).
    Comparisons: KG vs TEXT, KG vs HYBRID, KG+META vs FULL
    Reports: median difference, Wilcoxon p-value, rank-biserial r (effect size in [-1,1]).
    Note: LR results are effectively deterministic on the fixed split (std ≈ 0 throughout);
    LR rows are included for completeness but should not be used for inferential claims.
    """
    print("\n" + "="*70)
    print("SECTION: SIGNIFICANCE TESTS [M3]")
    print("="*70)

    comparisons = [
        # Always-available comparisons (no-text runs)
        ("KG",     "META",    "KG vs META"),
        ("KG",     "NETFEAT", "KG vs NETFEAT"),
        ("KG",     "KG+META", "KG vs KG+META"),
        ("META",   "KG+META", "META vs KG+META"),
        # Text-run-only comparisons
        ("KG",     "TEXT",    "KG vs TEXT"),
        ("KG",     "HYBRID",  "KG vs HYBRID"),
        ("KG+META","FULL",    "KG+META vs FULL"),
    ]

    primary_metric = {
        "renewed_4y":           "AUC",
        "is_triadic":           "AUC",
        "forward_citations_cat":"MacroF1",
    }

    rows = []
    for task in df_res["task"].unique():
        met = primary_metric.get(task, "AUC")
        sub = df_res[df_res["task"] == task]

        # Only XGB — LR is deterministic on the fixed split (std ≈ 0),
        # so its seed rows are not independent replicates.
        for model in ["XGB"]:
            sub_m = sub[sub["model"] == model]

            for rep_a, rep_b, label in comparisons:
                scores_a = sub_m[sub_m["representation"] == rep_a] \
                    .sort_values("seed")[met].values
                scores_b = sub_m[sub_m["representation"] == rep_b] \
                    .sort_values("seed")[met].values

                if len(scores_a) < 2 or len(scores_b) < 2:
                    continue
                if len(scores_a) != len(scores_b):
                    min_n = min(len(scores_a), len(scores_b))
                    scores_a, scores_b = scores_a[:min_n], scores_b[:min_n]

                diff = scores_a - scores_b
                med_diff = float(np.median(diff))

                try:
                    stat, pval = wilcoxon(scores_a, scores_b,
                                         alternative="two-sided",
                                         zero_method="wilcox")
                    # Rank-biserial r: use nonzero-difference count (matches
                    # scipy's internal n after zero_method="wilcox" drops ties)
                    nonzero_n = int(np.sum(diff != 0))
                    if nonzero_n > 0:
                        total_ranks = nonzero_n * (nonzero_n + 1) / 2
                        sign = 1.0 if med_diff >= 0 else -1.0
                        r = float(sign * (1.0 - 2.0 * stat / total_ranks))
                        r = float(np.clip(r, -1.0, 1.0))
                    else:
                        r = 0.0
                except Exception:
                    stat, pval, r = np.nan, np.nan, np.nan

                rows.append({
                    "task": task,
                    "model": model,
                    "metric": met,
                    "comparison": label,
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "mean_a": float(np.mean(scores_a)),
                    "mean_b": float(np.mean(scores_b)),
                    "median_diff": med_diff,
                    "wilcoxon_stat": float(stat) if not np.isnan(stat) else np.nan,
                    "p_value": float(pval) if not np.isnan(pval) else np.nan,
                    "effect_r": r,
                    "significant_p05": (pval < 0.05) if not np.isnan(pval) else False,
                })

    df_sig = pd.DataFrame(rows)
    df_sig.to_csv(OUT_DIR / "significance_tests.csv", index=False)
    print(df_sig.to_string())
    print(f"\n[M3] Significance table saved -> {OUT_DIR / 'significance_tests.csv'}")
    return df_sig


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — COLD/WARM-START ANALYSIS  [M2]
# ──────────────────────────────────────────────────────────────────────────────

def run_cold_warm_analysis(df: pd.DataFrame,
                           cite_df: pd.DataFrame,
                           tasks: list) -> pd.DataFrame:
    """
    Stratifies test patents by citation in-degree from training patents.
    Uses clearly defined degree buckets and reports ALL buckets.
    """
    print("\n" + "="*70)
    print("SECTION: COLD/WARM-START ANALYSIS [M2]")
    print("="*70)

    year = pd.to_numeric(df["year"], errors="coerce")
    mask_tr = year.isin(TRAIN_YEARS)
    mask_te = year == TEST_YEAR
    df_tr = df[mask_tr].reset_index(drop=True)
    df_te = df[mask_te].reset_index(drop=True)

    train_ids = set(df_tr["patent_number"].astype(str))
    indeg_series = compute_train_indegree(cite_df, train_ids)

    # attach indegree to test patents
    df_te = df_te.copy()
    df_te["patent_number_str"] = df_te["patent_number"].astype(str)
    df_te["test_indegree"] = df_te["patent_number_str"].map(
        indeg_series).fillna(0).astype(int)

    print("Test-set in-degree distribution:")
    print(df_te["test_indegree"].describe())
    print(df_te["test_indegree"].value_counts().sort_index().head(20))

    # build representations for train
    X_kg_tr = build_kg_matrix(df_tr)
    X_kg_te = build_kg_matrix(df_te)
    kg_tr_s, kg_te_s = scale(X_kg_tr, X_kg_te)

    has_text = False
    try:
        X_txt_tr = build_text_matrix(df_tr)
        X_txt_te = build_text_matrix(df_te)
        txt_tr_s, txt_te_s = scale(X_txt_tr, X_txt_te)
        hyb_tr_s = np.hstack([txt_tr_s, kg_tr_s])
        hyb_te_s = np.hstack([txt_te_s, kg_te_s])
        has_text = True
    except Exception as e:
        print(f"  [WARN] Text / Hybrid unavailable for cold-start: {e}")

    rows = []

    for task in tasks:
        if task == "forward_citations_cat":
            continue  # cold-start meaningful mainly for binary tasks
        print(f"\n  Cold/warm analysis for task: {task}")
        y_tr, drop_cols = task_setup(df_tr, task)
        y_te, _         = task_setup(df_te, task)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))

        reps_tr_te = {"KG": (kg_tr_s, kg_te_s)}
        if has_text:
            reps_tr_te["TEXT"]   = (txt_tr_s, txt_te_s)
            reps_tr_te["HYBRID"] = (hyb_tr_s, hyb_te_s)

        # fit on full training set once per representation
        for rep_name, (X_tr_r, X_te_r) in reps_tr_te.items():
            model_xgb = make_xgb(RANDOM_STATE, n_classes)
            try:
                model_xgb.fit(X_tr_r, y_tr)
                y_prob = model_xgb.predict_proba(X_te_r)[:, 1]
            except Exception as e:
                print(f"    [ERROR] {rep_name}: {e}")
                continue

            # report per degree bucket
            for bi, (lo, hi) in enumerate(DEGREE_BUCKETS):
                bucket_mask = ((df_te["test_indegree"] >= lo) &
                               (df_te["test_indegree"] <= hi))
                y_b    = y_te[bucket_mask]
                yp_b   = y_prob[bucket_mask]
                n_b    = bucket_mask.sum()

                if n_b < 20 or len(np.unique(y_b)) < 2:
                    rows.append({
                        "task": task, "representation": rep_name,
                        "degree_bucket": BUCKET_LABELS[bi],
                        "n_test": int(n_b), "AUC": np.nan,
                        "PR_AUC": np.nan, "BalAcc": np.nan,
                    })
                    continue

                try:
                    auc   = roc_auc_score(y_b, yp_b)
                    prauc = average_precision_score(y_b, yp_b)
                    balacc= balanced_accuracy_score(y_b, (yp_b>=0.5).astype(int))
                except Exception:
                    auc = prauc = balacc = np.nan

                rows.append({
                    "task": task, "representation": rep_name,
                    "degree_bucket": BUCKET_LABELS[bi],
                    "n_test": int(n_b),
                    "AUC": float(auc), "PR_AUC": float(prauc),
                    "BalAcc": float(balacc),
                })
                print(f"    {rep_name} | bucket {BUCKET_LABELS[bi]:5s} "
                      f"(n={n_b:4d}): AUC={auc:.4f}")

    df_cw = pd.DataFrame(rows)
    df_cw.to_csv(OUT_DIR / "cold_warm_start.csv", index=False)
    print(f"\n[M2] Cold/warm-start table saved -> {OUT_DIR / 'cold_warm_start.csv'}")
    return df_cw


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — LEAKAGE INFLATION ABLATION  [H1]
# ──────────────────────────────────────────────────────────────────────────────

def run_leakage_ablation(df: pd.DataFrame, tasks: list) -> pd.DataFrame:
    """
    Compare temporal split vs random split AUC.
    The difference (random – temporal) quantifies leakage inflation.
    Uses only patents from 2014-2017 to keep the patent pool identical.
    """
    print("\n" + "="*70)
    print("SECTION: LEAKAGE INFLATION ABLATION [H1]")
    print("="*70)

    year = pd.to_numeric(df["year"], errors="coerce")
    df_pool = df[year.isin(TRAIN_YEARS + [TEST_YEAR])].reset_index(drop=True)
    year_pool = pd.to_numeric(df_pool["year"], errors="coerce")
    print(f"  Patent pool (2014-2017): {len(df_pool):,}")

    X_kg = build_kg_matrix(df_pool)

    rows = []

    for task in tasks:
        if task == "forward_citations_cat":
            continue
        y_all, _ = task_setup(df_pool, task)
        n_classes = len(np.unique(y_all))

        # ── Temporal split ──────────────────────────────────────────────────
        mask_tr_t = year_pool.isin(TRAIN_YEARS).values
        mask_te_t = (year_pool == TEST_YEAR).values

        X_tr_t, X_te_t = X_kg[mask_tr_t], X_kg[mask_te_t]
        y_tr_t, y_te_t = y_all[mask_tr_t], y_all[mask_te_t]
        X_tr_ts, X_te_ts = scale(X_tr_t, X_te_t)

        aucs_temporal = []
        for seed in SEEDS:
            try:
                mdl = make_xgb(seed, n_classes)
                mdl.fit(X_tr_ts, y_tr_t)
                prob = mdl.predict_proba(X_te_ts)[:, 1]
                aucs_temporal.append(roc_auc_score(y_te_t, prob))
            except Exception:
                pass

        # ── Random split (same pool, same 80/20 ratio) ──────────────────────
        aucs_random = []
        for seed in SEEDS:
            X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
                X_kg, y_all, test_size=0.2, random_state=seed,
                stratify=y_all if n_classes == 2 else None)
            X_tr_rs, X_te_rs = scale(X_tr_r, X_te_r)
            try:
                mdl = make_xgb(seed, n_classes)
                mdl.fit(X_tr_rs, y_tr_r)
                prob = mdl.predict_proba(X_te_rs)[:, 1]
                aucs_random.append(roc_auc_score(y_te_r, prob))
            except Exception:
                pass

        t_mean = float(np.mean(aucs_temporal)) if aucs_temporal else np.nan
        r_mean = float(np.mean(aucs_random))   if aucs_random   else np.nan
        inflation = r_mean - t_mean

        rows.append({
            "task": task,
            "temporal_AUC_mean": t_mean,
            "temporal_AUC_std":  float(np.std(aucs_temporal)) if aucs_temporal else np.nan,
            "random_AUC_mean":   r_mean,
            "random_AUC_std":    float(np.std(aucs_random))   if aucs_random else np.nan,
            "leakage_inflation": float(inflation),
        })
        print(f"  {task}: temporal={t_mean:.4f} | random={r_mean:.4f} "
              f"| inflation={inflation:+.4f}")

    df_lk = pd.DataFrame(rows)
    df_lk.to_csv(OUT_DIR / "leakage_ablation.csv", index=False)
    print(f"\n[H1] Leakage ablation saved -> {OUT_DIR / 'leakage_ablation.csv'}")
    return df_lk


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 — GRAPH STRUCTURE ABLATION  [H2]
# ──────────────────────────────────────────────────────────────────────────────

def run_graph_structure_ablation(df: pd.DataFrame,
                                 cite_df: pd.DataFrame,
                                 tasks: list) -> pd.DataFrame:
    """
    Citation-only structural embedding vs heterogeneous KG embedding.

    Since rerunning node2vec on the citation-only graph would require
    graph learning infra, we approximate the citation-only representation
    using citation-graph statistics (in-degree, out-degree from the training
    edges) as a simple but transparent structural baseline, and contrast
    it with the full heterogeneous KG embeddings.

    This shows whether heterogeneity (inventors, assignees, examiners)
    adds value beyond citation topology alone.
    """
    print("\n" + "="*70)
    print("SECTION: GRAPH STRUCTURE ABLATION [H2]")
    print("="*70)

    year = pd.to_numeric(df["year"], errors="coerce")
    mask_tr = year.isin(TRAIN_YEARS)
    mask_te = year == TEST_YEAR
    df_tr = df[mask_tr].reset_index(drop=True)
    df_te = df[mask_te].reset_index(drop=True)

    train_ids = set(df_tr["patent_number"].astype(str))

    def _indeg_from_full(rel_df, id_set, col_dst="dst_id"):
        return rel_df[rel_df["src_id"].isin(id_set)].groupby(col_dst).size()

    # Compute citation in/out degree within training partition
    indeg_train = _indeg_from_full(cite_df, train_ids, "dst_id")
    outdeg_train = cite_df[cite_df["dst_id"].isin(train_ids)].groupby("src_id").size()

    def _attach_cite_feats(df_sub):
        pids = df_sub["patent_number"].astype(str)
        d = pd.DataFrame({
            "cite_indeg":  pids.map(indeg_train).fillna(0).values,
            "cite_outdeg": pids.map(outdeg_train).fillna(0).values,
        })
        return d.values.astype(np.float32)

    X_citeonly_tr = _attach_cite_feats(df_tr)
    X_citeonly_te = _attach_cite_feats(df_te)

    # Also read inventor / assignee counts
    try:
        inv_df  = pd.read_csv(str(INV_FILE),  sep=",", skipinitialspace=True,
                              engine="python", on_bad_lines="warn")
        assg_df = pd.read_csv(str(ASSG_FILE), sep=",", skipinitialspace=True,
                              engine="python", on_bad_lines="warn")
        inv_df.columns  = [c.strip().strip('"') for c in inv_df.columns]
        assg_df.columns = [c.strip().strip('"') for c in assg_df.columns]
        if "src_id" not in inv_df.columns:
            inv_df.columns  = ["src_id", "dst_id"] + list(inv_df.columns[2:])
        if "src_id" not in assg_df.columns:
            assg_df.columns = ["src_id", "dst_id"] + list(assg_df.columns[2:])

        inv_deg  = inv_df[inv_df["src_id"].isin(train_ids)].groupby("src_id").size()
        assg_deg = assg_df[assg_df["src_id"].isin(train_ids)].groupby("src_id").size()

        def _attach_hetero(df_sub):
            pids = df_sub["patent_number"].astype(str)
            d = pd.DataFrame({
                "cite_indeg":  pids.map(indeg_train).fillna(0).values,
                "cite_outdeg": pids.map(outdeg_train).fillna(0).values,
                "inv_deg":     pids.map(inv_deg).fillna(0).values,
                "assg_deg":    pids.map(assg_deg).fillna(0).values,
            })
            return d.values.astype(np.float32)

        X_heterostat_tr = _attach_hetero(df_tr)
        X_heterostat_te = _attach_hetero(df_te)
        print("  HETEROSTAT includes: citation + inventor + assignee degrees")
    except Exception as e:
        print(f"  [WARN] Could not load inventor/assignee: {e}")
        X_heterostat_tr = X_citeonly_tr
        X_heterostat_te = X_citeonly_te

    # Full KG embedding (hetero node2vec)
    X_kg_tr = build_kg_matrix(df_tr)
    X_kg_te = build_kg_matrix(df_te)

    rows = []

    for task in tasks:
        if task == "forward_citations_cat":
            continue
        y_tr, _ = task_setup(df_tr, task)
        y_te, _ = task_setup(df_te, task)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))

        reps = {
            "CITEONLY-stat":  (X_citeonly_tr,   X_citeonly_te),
            "HETEROSTAT":     (X_heterostat_tr,  X_heterostat_te),
            "KG-node2vec":    (X_kg_tr,          X_kg_te),
        }

        for rep_name, (X_tr_r, X_te_r) in reps.items():
            X_tr_s, X_te_s = scale(X_tr_r, X_te_r)
            aucs = []
            for seed in SEEDS:
                try:
                    mdl = make_xgb(seed, n_classes)
                    mdl.fit(X_tr_s, y_tr)
                    prob = mdl.predict_proba(X_te_s)[:, 1]
                    aucs.append(roc_auc_score(y_te, prob))
                except Exception:
                    pass

            mean_auc = float(np.mean(aucs)) if aucs else np.nan
            std_auc  = float(np.std(aucs))  if aucs else np.nan
            rows.append({
                "task": task, "representation": rep_name,
                "AUC_mean": mean_auc, "AUC_std": std_auc,
            })
            print(f"  {task} | {rep_name:20s}: AUC={mean_auc:.4f}±{std_auc:.4f}")

    df_gs = pd.DataFrame(rows)
    df_gs.to_csv(OUT_DIR / "graph_structure_ablation.csv", index=False)
    print(f"\n[H2] Graph structure ablation -> {OUT_DIR / 'graph_structure_ablation.csv'}")
    return df_gs


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 — NETWORK FEATURE BASELINE  [H3]
# ──────────────────────────────────────────────────────────────────────────────

def run_netfeat_baseline(df: pd.DataFrame, tasks: list) -> pd.DataFrame:
    """
    Network-feature-only baseline:
      degree (in/out), team size, assignee count, inventor count, family,
      backward citations, priority count.
    Shows that KG embeddings provide value beyond simple graph statistics.
    """
    print("\n" + "="*70)
    print("SECTION: NETWORK-FEATURE BASELINE [H3]")
    print("="*70)

    year = pd.to_numeric(df["year"], errors="coerce")
    df_tr = df[year.isin(TRAIN_YEARS)].reset_index(drop=True)
    df_te = df[year == TEST_YEAR].reset_index(drop=True)

    X_nf_tr = build_netfeat_matrix(df_tr)
    X_nf_te = build_netfeat_matrix(df_te)
    X_kg_tr = build_kg_matrix(df_tr)
    X_kg_te = build_kg_matrix(df_te)

    nf_tr_s, nf_te_s = scale(X_nf_tr, X_nf_te)
    kg_tr_s, kg_te_s = scale(X_kg_tr, X_kg_te)
    kg_nf_tr = np.hstack([kg_tr_s, nf_tr_s])
    kg_nf_te = np.hstack([kg_te_s, nf_te_s])

    rows = []
    for task in tasks:
        if task == "forward_citations_cat":
            continue
        y_tr, _ = task_setup(df_tr, task)
        y_te, _ = task_setup(df_te, task)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))

        reps = {
            "NETFEAT-only": (nf_tr_s, nf_te_s),
            "KG-only":      (kg_tr_s, kg_te_s),
            "KG+NETFEAT":   (kg_nf_tr, kg_nf_te),
        }
        for rep_name, (X_tr_r, X_te_r) in reps.items():
            aucs = []
            for seed in SEEDS:
                try:
                    mdl = make_xgb(seed, n_classes)
                    mdl.fit(X_tr_r, y_tr)
                    prob = mdl.predict_proba(X_te_r)[:, 1]
                    aucs.append(roc_auc_score(y_te, prob))
                except Exception:
                    pass
            mean_auc = float(np.mean(aucs)) if aucs else np.nan
            std_auc  = float(np.std(aucs))  if aucs else np.nan
            rows.append({
                "task": task, "representation": rep_name,
                "AUC_mean": mean_auc, "AUC_std": std_auc,
            })
            print(f"  {task} | {rep_name:15s}: AUC={mean_auc:.4f}±{std_auc:.4f}")

    df_nf = pd.DataFrame(rows)
    df_nf.to_csv(OUT_DIR / "netfeat_baseline.csv", index=False)
    print(f"\n[H3] Network-feature baseline -> {OUT_DIR / 'netfeat_baseline.csv'}")
    return df_nf


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13 —  PAPER TABLE GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_paper_tables(out_dir: Path):
    """
    Reads all CSV outputs and prints LaTeX-friendly summary tables.
    """
    print("\n" + "="*70)
    print("SECTION: PAPER TABLE GENERATION")
    print("="*70)

    # ── Table 1: Main results (best model per representation) ──────────────
    main_path = out_dir / "main_results_aggregated.csv"
    if main_path.exists():
        df = pd.read_csv(main_path)

        # Pick one model family for display (XGB usually best)
        for model_choice in ["XGB", "LR"]:
            sub = df[df["model"] == model_choice]
            if sub.empty:
                continue

            print(f"\n── Main Results Table (model={model_choice}) ──")
            for task in ["renewed_4y", "is_triadic", "forward_citations_cat"]:
                t = sub[sub["task"] == task].copy()
                if t.empty:
                    continue
                met = "AUC_mean" if task != "forward_citations_cat" else "MacroF1_mean"
                prauc_col = "PR_AUC_mean" if "PR_AUC_mean" in t.columns else None
                balacc_col = "BalAcc_mean" if "BalAcc_mean" in t.columns else None
                t = t.sort_values(met, ascending=False)

                print(f"\n  Task: {task}")
                cols_display = (["representation", met]
                                + ([prauc_col] if prauc_col else [])
                                + ([balacc_col] if balacc_col else [])
                                + ["Recall_1_mean", "Precision_1_mean"])
                cols_display = [c for c in cols_display if c in t.columns]
                print(t[cols_display].to_string(index=False, float_format="{:.4f}".format))
            break  # only print once

    # ── Table 2: Significance tests ────────────────────────────────────────
    sig_path = out_dir / "significance_tests.csv"
    if sig_path.exists() and sig_path.stat().st_size > 5:
        try:
            df_sig = pd.read_csv(sig_path)
            if not df_sig.empty:
                print("\n── Significance Tests ──")
                show_cols = [c for c in ["task", "model", "comparison",
                               "mean_a", "mean_b", "median_diff",
                               "p_value", "significant_p05"] if c in df_sig.columns]
                print(df_sig[show_cols].to_string(index=False,
                                                  float_format="{:.4f}".format))
        except pd.errors.EmptyDataError:
            print("\n── Significance Tests: (no data — re-run without --no-text for TEXT comparisons) ──")

    # ── Table 3: Leakage inflation ─────────────────────────────────────────
    lk_path = out_dir / "leakage_ablation.csv"
    if lk_path.exists():
        df_lk = pd.read_csv(lk_path)
        print("\n── Leakage Inflation (random − temporal AUC) ──")
        print(df_lk.to_string(index=False, float_format="{:.4f}".format))

    # ── Table 4: Graph structure ablation ─────────────────────────────────
    gs_path = out_dir / "graph_structure_ablation.csv"
    if gs_path.exists():
        df_gs = pd.read_csv(gs_path)
        print("\n── Graph Structure Ablation ──")
        print(df_gs.to_string(index=False, float_format="{:.4f}".format))

    # ── Table 5: Cold/warm start ───────────────────────────────────────────
    cw_path = out_dir / "cold_warm_start.csv"
    if cw_path.exists():
        df_cw = pd.read_csv(cw_path)
        print("\n── Cold/Warm-Start Analysis ──")
        print(df_cw.to_string(index=False, float_format="{:.4f}".format))

    print(f"\nAll tables generated from {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text embedding loading (faster; KG/META/NETFEAT only)")
    parser.add_argument("--skip-main", action="store_true",
                        help="Skip main experiments (use existing CSV)")
    parser.add_argument("--only-tables", action="store_true",
                        help="Only regenerate paper tables from existing CSVs")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "kg_exports"),
                        help="Path to KG exports folder containing patent_full_table.parquet "
                             "and rels_*.txt files")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent.parent / "results"),
                        help="Output directory for result CSVs")
    args = parser.parse_args()

    # Set all path globals from arguments
    global BASE_KG, PARQUET_FILE, CITES_FILE, INV_FILE, ASSG_FILE, OUT_DIR
    BASE_KG      = Path(args.data_dir)
    PARQUET_FILE = BASE_KG / "patent_full_table.parquet"
    CITES_FILE   = BASE_KG / "rels_cites.txt"
    INV_FILE     = BASE_KG / "rels_invented_by.txt"
    ASSG_FILE    = BASE_KG / "rels_assigned_to.txt"
    OUT_DIR      = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only_tables:
        generate_paper_tables(OUT_DIR)
        return

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    df = load_parquet_selective(PARQUET_FILE, load_text=not args.no_text)
    df = coerce_numerics(df)

    # Derive renewed_4y if missing
    if "renewed_4y" not in df.columns or df["renewed_4y"].isna().all():
        if "num_of_maint" in df.columns:
            df["renewed_4y"] = (pd.to_numeric(df["num_of_maint"],
                                              errors="coerce").fillna(0) >= 1).astype(int)
            print("[DATA] Derived renewed_4y from num_of_maint.")

    year = pd.to_numeric(df["year"], errors="coerce")
    print(f"[DATA] Year distribution:\n{year.value_counts().sort_index()}")
    print(f"[DATA] Shape: {df.shape}")

    # ── Load relation files ────────────────────────────────────────────────
    patent_ids = set(df["patent_number"].astype(str))
    try:
        degree_df, cite_df = compute_graph_degrees(patent_ids)
    except Exception as e:
        print(f"[WARN] Could not compute graph degrees: {e}")
        # Create empty fallbacks
        degree_df = pd.DataFrame({"patent_number_str": list(patent_ids),
                                   "cite_indegree": 0.0, "cite_outdegree": 0.0,
                                   "inventor_count": 0.0, "assignee_count": 0.0})
        cite_df = pd.DataFrame(columns=["src_id", "dst_id", "rel_type"])

    # Attach degree features to main df
    df["patent_number_str"] = df["patent_number"].astype(str)
    if "patent_number" in degree_df.columns:
        degree_df = degree_df.rename(columns={"patent_number": "patent_number_str"})
    df = df.merge(degree_df, on="patent_number_str", how="left")
    for _dc in ["cite_indegree", "cite_outdegree", "inventor_count", "assignee_count"]:
        if _dc not in df.columns:
            df[_dc] = 0.0
    df[["cite_indegree", "cite_outdegree",
        "inventor_count", "assignee_count"]] = \
        df[["cite_indegree", "cite_outdegree",
            "inventor_count", "assignee_count"]].fillna(0)

    print(f"[DATA] After degree join: {df.shape}")

    # ── Define task list ──────────────────────────────────────────────────
    tasks = ["renewed_4y", "is_triadic", "forward_citations_cat"]

    # ── Run experiments ────────────────────────────────────────────────────
    if not args.skip_main:
        df_main = run_main_experiments(df, tasks)
    else:
        main_path = OUT_DIR / "main_results_all_seeds.csv"
        if main_path.exists():
            df_main = pd.read_csv(main_path)
            print(f"[SKIP] Loaded existing main results: {main_path}")
        else:
            print("[WARN] --skip-main but no existing results; running main experiments.")
            df_main = run_main_experiments(df, tasks)

    run_significance_tests(df_main)
    run_cold_warm_analysis(df, cite_df, tasks)
    run_leakage_ablation(df, tasks)
    run_graph_structure_ablation(df, cite_df, tasks)
    run_netfeat_baseline(df, tasks)
    generate_paper_tables(OUT_DIR)

    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE. All outputs in: {OUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
