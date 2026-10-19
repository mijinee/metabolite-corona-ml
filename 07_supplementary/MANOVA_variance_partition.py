"""
Two-way variance partition (polymer x QWD) with permutation tests.

Input:  Master_Processed_Conservative_Data.xlsx
Output: MANOVA_variance_partition.xlsx
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import DEFECTS, OUT_SUPPLEMENTARY, POLYMERS, PROCESSED_PATH
from utils.dedup_features import add_dedup_group

OUT_DIR  = OUT_SUPPLEMENTARY / "variance_partition"
OUT_XLSX = OUT_DIR / "MANOVA_variance_partition.xlsx"

MODES       = ["RP", "HILIC"]
REF_DEFECT  = "Ref"
REF_POLYMER = "Ref"
VALID_GAP   = ["No gap", "Missing ions", "No Gap", "Missing Ions"]
N_PERM      = 5000

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN features
OUT_DIR.mkdir(parents=True, exist_ok=True)
raw = pd.read_excel(PROCESSED_PATH, sheet_name=None)


# Data loading
def load_mode_data(mode_name):
    df  = pd.DataFrame(raw[f"{mode_name}_log2_1p"]).iloc[3:].copy()
    gap = pd.DataFrame(raw[f"{mode_name}_gap"]).iloc[3:].copy()
    rename = {"Unnamed: 0": "Defect", "Unnamed: 1": "Polymer", "Peak idx/annotation": "Method"}
    df.rename(columns=rename, inplace=True)
    gap.rename(columns=rename, inplace=True)

    feature_cols = [c for c in df.columns if c not in ["Defect", "Polymer", "Method"]]
    defect_vec   = df["Defect"].astype(str)
    polymer_vec  = df["Polymer"].astype(str)
    mask_nt  = defect_vec.isin(DEFECTS) & polymer_vec.isin(POLYMERS)
    mask_ref = (defect_vec == REF_DEFECT) & (polymer_vec == REF_POLYMER)
    sub_df   = df.loc[mask_nt | mask_ref].reset_index(drop=True)
    sub_gap  = gap.loc[mask_nt | mask_ref].reset_index(drop=True)

    X_vals  = sub_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_final = X_vals.where(sub_gap[feature_cols].isin(VALID_GAP), np.nan)

    return (X_final, [str(c) for c in feature_cols], sub_df["Defect"].astype(str),
            sub_df["Polymer"].astype(str), sub_df["Method"].astype(str).fillna("NA"))


def list_methods(mode_name):
    _, _, defects_vec, polymers_vec, methods_vec = load_mode_data(mode_name)
    ref_idx = (polymers_vec == REF_POLYMER) & (defects_vec == REF_DEFECT)
    nt_idx  = defects_vec.isin(DEFECTS) & polymers_vec.isin(POLYMERS)
    return sorted(set(methods_vec[ref_idx].unique()) & set(methods_vec[nt_idx].unique()))


def extract_method_panel(mode_name, method_label):
    """Per-replicate log2 difference from the mean plasma (Ref) profile, one method."""
    X_final, features, defects_vec, polymers_vec, methods_vec = load_mode_data(mode_name)
    ref_idx = (polymers_vec == REF_POLYMER) & (defects_vec == REF_DEFECT) & (methods_vec == method_label)
    R_mean = np.nanmean(X_final.loc[ref_idx].to_numpy(float), axis=0)

    rows_meta, rows_y = [], []
    for polymer in POLYMERS:
        for defect in DEFECTS:
            grp_idx = (polymers_vec == polymer) & (defects_vec == defect) & (methods_vec == method_label)
            if grp_idx.sum() == 0:
                continue
            Dm = X_final.loc[grp_idx].to_numpy(float)
            for rep_i in range(Dm.shape[0]):
                rows_meta.append({"Polymer": polymer, "Defect": defect, "Replicate": rep_i})
                rows_y.append(Dm[rep_i] - R_mean)
    return pd.DataFrame(rows_meta), np.vstack(rows_y), features


def build_dedup_groups(features):
    dedup_input = add_dedup_group(pd.DataFrame({"Feature": features}))
    return dedup_input.set_index("Feature")["DedupGroup"]


def collapse_dedup_columns(Y, features, dedup_group_map):
    """Median across features that share a DedupGroup (one column per metabolite)."""
    df_y = pd.DataFrame(Y, columns=[dedup_group_map.get(f, f) for f in features])
    grouped = df_y.T.groupby(level=0).apply(lambda g: np.nanmedian(g.values, axis=0))
    return np.vstack(grouped.values).T, grouped.index.tolist()


# Two-way variance partition
def cell_means(meta, Y):
    polymers = sorted(meta["Polymer"].unique())
    defects  = sorted(meta["Defect"].unique())
    Y_cell = np.full((len(polymers), len(defects), Y.shape[1]), np.nan)
    for i, p in enumerate(polymers):
        for j, d in enumerate(defects):
            idx = ((meta["Polymer"] == p) & (meta["Defect"] == d)).values
            if idx.sum() == 0:
                continue
            with np.errstate(all="ignore"):
                Y_cell[i, j, :] = np.nanmean(Y[idx], axis=0)
    return Y_cell, polymers, defects


def _marginal_means(Y_cell):
    with np.errstate(all="ignore"):
        grand_mean = np.nanmean(Y_cell.reshape(-1, Y_cell.shape[2]), axis=0)
        poly_mean  = np.nanmean(Y_cell, axis=1)
        def_mean   = np.nanmean(Y_cell, axis=0)
    return grand_mean, poly_mean, def_mean


def variance_partition(meta, Y):
    Y_cell, polymers, defects = cell_means(meta, Y)
    grand_mean, poly_mean, def_mean = _marginal_means(Y_cell)

    n_f = Y.shape[1]
    SS_poly, SS_def, SS_int, SS_noise = (np.zeros(n_f) for _ in range(4))

    for i in range(len(polymers)):
        valid = ~np.isnan(Y_cell[i, :, :]).all(axis=0)
        SS_poly += np.where(valid, (poly_mean[i] - grand_mean) ** 2, 0.0)
    for j in range(len(defects)):
        valid = ~np.isnan(Y_cell[:, j, :]).all(axis=0)
        SS_def += np.where(valid, (def_mean[j] - grand_mean) ** 2, 0.0)

    for i in range(len(polymers)):
        for j in range(len(defects)):
            valid = ~np.isnan(Y_cell[i, j, :])
            fit = poly_mean[i] + def_mean[j] - grand_mean
            SS_int += np.where(valid, (Y_cell[i, j, :] - fit) ** 2, 0.0)

    for r in range(Y.shape[0]):
        i = polymers.index(meta["Polymer"].iloc[r])
        j = defects.index(meta["Defect"].iloc[r])
        valid = ~np.isnan(Y[r])
        SS_noise += np.where(valid, (Y[r] - Y_cell[i, j, :]) ** 2, 0.0)

    SS_poly, SS_def, SS_int, SS_noise = SS_poly.sum(), SS_def.sum(), SS_int.sum(), SS_noise.sum()
    SS_total = SS_poly + SS_def + SS_int + SS_noise
    return {"SS_total": SS_total, "SS_poly": SS_poly, "SS_def": SS_def,
            "SS_int": SS_int, "SS_noise": SS_noise,
            "Frac_poly": SS_poly / SS_total, "Frac_def": SS_def / SS_total,
            "Frac_int": SS_int / SS_total, "Frac_noise": SS_noise / SS_total}


# Permutation tests
def perm_polymer(meta, Y, n):
    null = []
    for _ in range(n):
        m = meta.copy()
        m["Polymer"] = np.random.permutation(m["Polymer"].values)
        null.append(variance_partition(m, Y)["Frac_poly"])
    return np.array(null)


def perm_defect(meta, Y, n):
    null = []
    for _ in range(n):
        m = meta.copy()
        m["Defect"] = np.random.permutation(m["Defect"].values)
        null.append(variance_partition(m, Y)["Frac_def"])
    return np.array(null)


def permute_column_nan_preserving(col):
    out = col.copy()
    valid = ~np.isnan(col)
    if valid.sum() > 1:
        out[valid] = np.random.permutation(col[valid])
    return out


def perm_interaction(meta, Y, n):
    Y_cell, polymers, defects = cell_means(meta, Y)
    grand_mean, poly_mean, def_mean = _marginal_means(Y_cell)

    Y_hat = np.zeros_like(Y)
    for r in range(Y.shape[0]):
        i = polymers.index(meta["Polymer"].iloc[r])
        j = defects.index(meta["Defect"].iloc[r])
        Y_hat[r] = poly_mean[i] + def_mean[j] - grand_mean
    residuals = Y - Y_hat

    null = []
    for _ in range(n):
        Y_perm = Y_hat + np.apply_along_axis(permute_column_nan_preserving, 0, residuals)
        null.append(variance_partition(meta, Y_perm)["Frac_int"])
    return np.array(null)


def pval(null, obs):
    return (np.sum(null >= obs) + 1) / (len(null) + 1)


def run_manova(meta, Y, n_perm=N_PERM):
    vp = variance_partition(meta, Y)
    return {
        "effect_sizes": {"Polymer": vp["Frac_poly"], "Defect": vp["Frac_def"],
                         "Interaction": vp["Frac_int"], "Noise": vp["Frac_noise"]},
        "p_values": {"Polymer": pval(perm_polymer(meta, Y, n_perm), vp["Frac_poly"]),
                     "Defect": pval(perm_defect(meta, Y, n_perm), vp["Frac_def"]),
                     "Interaction": pval(perm_interaction(meta, Y, n_perm), vp["Frac_int"])},
        "SS": vp,
    }


# Run
results = {}
for mode_name in MODES:
    for method_label in list_methods(mode_name):
        meta, Y, features = extract_method_panel(mode_name, method_label)
        Y_dedup, _ = collapse_dedup_columns(Y, features, build_dedup_groups(features))

        for label, include_pristine in [("with_pristine", True), ("no_pristine", False)]:
            keep = np.ones(len(meta), bool) if include_pristine else (meta["Defect"] != "R").values
            m, y = meta[keep].reset_index(drop=True), Y_dedup[keep]
            key = f"{mode_name}_{method_label}_{label}"
            print(f"Running {key}: {y.shape[0]} samples x {y.shape[1]} features")
            results[key] = run_manova(m, y)


# Export
summary_rows = []
for key, res in results.items():
    row = {"Condition": key}
    for term in ["Polymer", "Defect", "Interaction", "Noise"]:
        row[f"{term}_fraction"] = res["effect_sizes"][term]
    for term in ["Polymer", "Defect", "Interaction"]:
        row[f"{term}_pvalue"] = res["p_values"][term]
    summary_rows.append(row)
df_summary = pd.DataFrame(summary_rows)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    for key, res in results.items():
        pd.DataFrame([res["SS"]]).to_excel(writer, sheet_name=f"{key}_SS"[:31], index=False)

print(f"Saved: {OUT_XLSX}")
print(df_summary.to_string(index=False))
