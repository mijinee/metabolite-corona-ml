"""
Combines the Compound Discoverer exports of the four LC-MS/MS modes into one workbook
(sheets HILIC_comb, HILIC_gap, RP_comb, RP_gap).

Input:  HILIC- (all).xlsx, HILIC+ (all).xlsx, RP- (all).xlsx, RP+ (all).xlsx
Output: 01_Global_Analysis_Combined.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import CD_EXPORTS, COMBINED_PATH, OUT_PREPROCESS

SHEET_INTENSITY = "LC-MSMS"
SHEET_GAP       = "gap status"
CHROMATOGRAPHY  = {"HILIC": ["HILIC-", "HILIC+"], "RP": ["RP-", "RP+"]}
META_COLS       = ["Defect", "Polymer", "Method"]


def read_export(path, sheet, mode):
    """Return (metadata, feature header rows, values) from one CD export sheet."""
    df = pd.read_excel(path, sheet_name=sheet)
    ids  = df.iloc[0, 3:].astype(str).values
    mzs  = [f"{mode}_{v}" for v in df.iloc[1, 3:].values]
    rts  = df.iloc[2, 3:].values
    meta = df.iloc[3:, 0:3].reset_index(drop=True)
    meta.columns = META_COLS
    vals = df.iloc[3:, 3:].reset_index(drop=True)
    return meta, (ids, mzs, rts), vals


def build_sheet(meta, ids, mzs, rts, vals):
    """Assemble one sheet in the shared 4-header-row layout."""
    n = len(ids)
    header = pd.DataFrame([
        ["", "", "Peak idx/annotation"] + list(ids),
        ["", "", "mode_m/z"]            + list(mzs),
        ["", "", "RT (min)"]            + list(rts),
        META_COLS                       + [""] * n,
    ])
    body = pd.concat([meta.reset_index(drop=True), vals.reset_index(drop=True)], axis=1)
    body.columns = range(body.shape[1])
    return pd.concat([header, body], ignore_index=True)


OUT_PREPROCESS.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(COMBINED_PATH, engine="openpyxl") as writer:
    for chrom, modes in CHROMATOGRAPHY.items():
        parts = {"comb": [], "gap": []}
        meta_ref = None
        for mode in modes:
            for kind, sheet in [("comb", SHEET_INTENSITY), ("gap", SHEET_GAP)]:
                meta, hdr, vals = read_export(CD_EXPORTS[mode], sheet, mode)
                if meta_ref is None:
                    meta_ref = meta
                elif not meta.astype(str).equals(meta_ref.astype(str)):
                    raise ValueError(f"Sample rows of {mode} ({sheet}) do not match {modes[0]}.")
                parts[kind].append((hdr, vals))

        for kind, blocks in parts.items():
            ids  = [x for hdr, _ in blocks for x in hdr[0]]
            mzs  = [x for hdr, _ in blocks for x in hdr[1]]
            rts  = [x for hdr, _ in blocks for x in hdr[2]]
            vals = pd.concat([v for _, v in blocks], axis=1, ignore_index=True)
            if len(set(ids)) != len(ids):
                print(f"  warning: duplicate feature IDs in {chrom}_{kind}")
            build_sheet(meta_ref, ids, mzs, rts, vals).to_excel(
                writer, sheet_name=f"{chrom}_{kind}", index=False, header=False)
        print(f"{chrom}: {len(meta_ref)} samples, {len(ids)} features")

print(f"Saved: {COMBINED_PATH}")
