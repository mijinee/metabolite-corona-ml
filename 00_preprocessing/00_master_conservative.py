"""
Preprocessing: duplicate removal, detection filter, imputation, normalization, log2, CV filter.

Input:  01_Global_Analysis_Combined.xlsx
Output: Master_Processed_Conservative_Data.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import COMBINED_PATH, OUT_PREPROCESS, PROCESSED_PATH

PROCESSED_FILE = PROCESSED_PATH
MODES = {"HILIC": {"comb": "HILIC_comb", "gap": "HILIC_gap"},
         "RP":    {"comb": "RP_comb",    "gap": "RP_gap"}}

MZ_TOL   = 0.02
RT_TOL   = 0.3
CV_THRESH = 30.0
FC_THRESH = 1.0


def parse_mz(val):
    try:
        s = str(val)
        return float(s.split("_")[-1]) if "_" in s else float(s)
    except Exception:
        return np.nan


def build_sheet(meta, values, ids, mzs, rts):
    n = values.shape[1]
    header = pd.DataFrame([
        ["", "", "Peak idx/annotation"] + list(ids),
        ["", "", "mode_m/z"]            + list(mzs),
        ["", "", "RT (min)"]            + list(rts),
        ["Defect", "Polymer", "Method"] + [""] * n,
    ])
    body = pd.concat([meta.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    body.columns = range(body.shape[1])
    return pd.concat([header, body], ignore_index=True)


def run_pipeline(mode):
    print(f"[{mode}] Running preprocessing pipeline...")

    df_raw = pd.read_excel(COMBINED_PATH, sheet_name=MODES[mode]["comb"], header=None)
    df_gap = pd.read_excel(COMBINED_PATH, sheet_name=MODES[mode]["gap"],  header=None)

    raw_ids      = df_raw.iloc[0, 3:].astype(str).values
    raw_mz_vals  = df_raw.iloc[1, 3:].values
    mzs          = np.array([parse_mz(x) for x in raw_mz_vals])
    rts          = pd.to_numeric(df_raw.iloc[2, 3:], errors="coerce").values
    data         = df_raw.iloc[4:, 3:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    data_gap_raw = df_gap.iloc[4:, 3:].reset_index(drop=True)
    meta         = df_raw.iloc[4:, 0:3].reset_index(drop=True)
    meta.columns = ["Defect", "Polymer", "Method"]

    # Duplicate removal
    mean_int   = data.mean(axis=0, numeric_only=True).values
    drop_idx   = set()
    valid_idx  = np.where(~np.isnan(mzs))[0]
    sorted_idx = valid_idx[np.argsort(mzs[valid_idx])]

    for k in range(len(sorted_idx)):
        i = sorted_idx[k]
        if i in drop_idx:
            continue
        for l in range(k + 1, len(sorted_idx)):
            j = sorted_idx[l]
            if j in drop_idx:
                continue
            if abs(mzs[j] - mzs[i]) > MZ_TOL:
                break
            if abs(rts[j] - rts[i]) < RT_TOL:
                if mean_int[j] > mean_int[i]:
                    drop_idx.add(i)
                    break
                else:
                    drop_idx.add(j)

    mask_unique = np.ones(len(mzs), dtype=bool)
    mask_unique[list(drop_idx)] = False

    # Majority detection filter
    gap_unique = data_gap_raw.loc[:, mask_unique]
    meta["Group"] = meta["Defect"].astype(str) + "_" + meta["Polymer"].astype(str)
    groups = meta["Group"].unique()
    mask_detected = np.zeros(gap_unique.shape[1], dtype=bool)

    for g in groups:
        idx = meta[meta["Group"] == g].index
        sub = gap_unique.iloc[idx]
        thresh = 2 if len(idx) >= 3 else 1
        mask_detected |= ((sub != "Full gap").sum(axis=0) >= thresh)

    print(f"  unique={mask_unique.sum()} → detected={mask_detected.sum()}")

    # Imputation and normalization
    valid_data = data.loc[:, mask_unique].loc[:, mask_detected].copy()
    for col in valid_data.columns:
        valid = valid_data[col][valid_data[col] > 0]
        fill  = valid.min() / 2.0 if len(valid) > 0 else 1e-9
        valid_data[col] = valid_data[col].fillna(fill).replace(0, fill)

    norm_data = valid_data.div(valid_data.median(axis=1), axis=0)
    log_data  = np.log2(norm_data + 1)

    # Tiered CV filter
    ref = next((g for g in groups if "Ref" in g), None) or \
          next((g for g in groups if "C_" in g), None)

    grp_cvs, grp_means = {}, {}
    for g in groups:
        sub = norm_data.loc[meta["Group"] == g]
        mu  = sub.mean(axis=0, numeric_only=True)
        sd  = sub.std(axis=0, ddof=1, numeric_only=True)
        grp_cvs[g]   = (sd / mu) * 100
        grp_means[g] = mu

    rep_cv = pd.DataFrame(grp_cvs).mean(axis=1, numeric_only=True)

    if ref:
        fcs = [np.log2((grp_means[g] + 1e-9) / (grp_means[ref] + 1e-9)).abs()
               for g in groups if g != ref]
        max_fc = pd.concat(fcs, axis=1).max(axis=1) if fcs else pd.Series(0, index=norm_data.columns)
    else:
        max_fc = pd.Series(0, index=norm_data.columns)

    mask_tiered = (rep_cv < CV_THRESH) | ((rep_cv >= CV_THRESH) & (max_fc > FC_THRESH))

    # Assemble outputs
    final_int = norm_data.loc[:, mask_tiered].copy()
    final_log = log_data.loc[:, mask_tiered].copy()
    final_gap = gap_unique.loc[:, mask_detected].loc[:, mask_tiered].copy()

    unique_idx   = np.where(mask_unique)[0]
    detected_idx = unique_idx[mask_detected]
    final_idx    = detected_idx[mask_tiered]

    survivor_ids     = raw_ids[final_idx]
    survivor_mzs     = raw_mz_vals[final_idx]
    survivor_rts     = rts[final_idx]

    meta_cols = meta[["Defect", "Polymer", "Method"]]
    hdr = (survivor_ids, survivor_mzs, survivor_rts)
    df_int     = build_sheet(meta_cols, final_int, *hdr)
    df_log     = build_sheet(meta_cols, final_log, *hdr)
    df_gap_out = build_sheet(meta_cols, final_gap, *hdr)

    print(f"  Final features: {final_int.shape[1]}")
    return {"int_sheet": df_int, "log_sheet": df_log, "gap_sheet": df_gap_out,
            "raw_int_df": final_int, "meta_df": meta, "indices": final_idx}


def calculate_r2_cv(data_df, meta_df, mode):
    data     = data_df.astype(float)
    data_log = np.log2(data + 1e-9)
    meta     = meta_df.copy()
    meta["Key"] = (meta["Defect"].astype(str) + "_" +
                   meta["Polymer"].astype(str) + "_" +
                   meta["Method"].astype(str))

    r2s, cvs = [], []
    for key in meta["Key"].unique():
        if "Ref_Ref" in key or "nan" in key:
            continue
        idx = meta[meta["Key"] == key].index
        if len(idx) < 2:
            continue

        sub_log = data_log.iloc[idx].T
        sub_log.columns = [f"R{i}" for i in range(len(idx))]
        local_r2 = [pearsonr(sub_log[a], sub_log[b])[0] ** 2
                    for a, b in itertools.combinations(sub_log.columns, 2)]

        sub_lin  = data.iloc[idx]
        mu       = sub_lin.mean(axis=0, numeric_only=True)
        sd       = sub_lin.std(axis=0, ddof=1, numeric_only=True)
        local_cv = ((sd / mu) * 100).replace([np.inf, -np.inf], np.nan).dropna()

        parts   = key.split("_")
        sensor  = "_".join(parts[:-1])
        method  = parts[-1]

        r2s.append({"Sensor": sensor, "Method": method,
                    "Avg_R2": np.mean(local_r2), "Mode": mode})
        for v in local_cv.values:
            cvs.append({"Sensor": sensor, "Method": method, "CV": v, "Mode": mode})

    return pd.DataFrame(r2s), pd.DataFrame(cvs)


def main():
    OUT_PREPROCESS.mkdir(parents=True, exist_ok=True)
    writer = pd.ExcelWriter(PROCESSED_FILE, engine="xlsxwriter")
    all_r2, all_cvs = [], []

    for mode in ["HILIC", "RP"]:
        res = run_pipeline(mode)
        res["int_sheet"].to_excel(writer, sheet_name=f"{mode}_Intensity", index=False, header=False)
        res["log_sheet"].to_excel(writer, sheet_name=f"{mode}_log2_1p",   index=False, header=False)
        res["gap_sheet"].to_excel(writer, sheet_name=f"{mode}_gap",       index=False, header=False)

        r2, cv = calculate_r2_cv(res["raw_int_df"], res["meta_df"], mode)
        all_r2.append(r2)
        all_cvs.append(cv)

    writer.close()
    print(f"Saved: {PROCESSED_FILE}")

    final_r2  = pd.concat(all_r2,  ignore_index=True)
    final_cvs = pd.concat(all_cvs, ignore_index=True)

    final_r2.to_csv( OUT_PREPROCESS / "R2_stats.csv",  index=False)
    final_cvs.to_csv(OUT_PREPROCESS / "CV_stats.csv",  index=False)

    print("Done.")


if __name__ == "__main__":
    main()
