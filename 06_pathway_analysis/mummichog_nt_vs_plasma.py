"""
Mummichog pathway analysis, nanotube corona vs plasma (per defect, per polymer, all nanotubes).

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx
Output: mummichog result folders
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DEFECTS, HILIC_LOG2FC, OUT_MUMMICHOG, POLYMERS, PRED_PATH, RP_LOG2FC
from utils.dedup_features import add_dedup_group

SAVE_PATH = OUT_MUMMICHOG

RUN_HILIC = True
RUN_RP    = True

CUTOFF = 0.05   # mummichog significance cutoff (-c)


# Locate mummichog

def _find_mummichog():
    if shutil.which("mummichog"):
        return ["mummichog"]
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    win_exe = os.path.join(scripts_dir, "mummichog.exe")
    if os.path.isfile(win_exe):
        return [win_exe]
    try:
        probe = subprocess.run([sys.executable, "-m", "mummichog", "--help"],
                               capture_output=True)
        if probe.returncode in (0, 1):
            return [sys.executable, "-m", "mummichog"]
    except FileNotFoundError:
        pass
    raise EnvironmentError(
        "\nCannot find mummichog.\n"
        "  pip install mummichog\n"
        "  then restart terminal/kernel so Scripts/ is on PATH."
    )

MUMMICHOG_CMD = _find_mummichog()
print(f"Mummichog: {' '.join(MUMMICHOG_CMD)}\n")

# Helpers

def load_and_prepare(log2fc_path, pred_path, mode):
    """Load NT_vs_Plasma sheet, join m/z/RT/Polarity, add DedupGroup."""
    print(f"Loading {mode}...")
    nt = pd.read_excel(log2fc_path, sheet_name='NT_vs_Plasma')
    pred = (pd.read_excel(pred_path, sheet_name=f'{mode}_Predictions')
              [['FeatureID', 'MZ', 'RT', 'Polarity']]
              .rename(columns={'FeatureID': 'Feature'}))
    pred['Feature'] = pred['Feature'].astype(str)
    nt['Feature'] = nt['Feature'].astype(str)
    nt = nt.merge(pred, on='Feature', how='inner')
    nt = add_dedup_group(nt)
    print(f"  {len(nt)} features -> {nt['DedupGroup'].nunique()} dedup groups  "
          f"(POS={(nt['Polarity']==1).sum()}, NEG={(nt['Polarity']==0).sum()})")
    return nt


def make_mmg_df(df, log2fc_cols, p_cols):
    """Mummichog input table (m/z, rtime, p-value, t-score)."""
    log2fc_mat = df[log2fc_cols].apply(pd.to_numeric, errors='coerce')
    p_mat      = df[p_cols].apply(pd.to_numeric, errors='coerce')

    mmg = pd.DataFrame({
        'mz':          pd.to_numeric(df['MZ'].values, errors='coerce'),
        'rtime':       pd.to_numeric(df['RT'].values, errors='coerce'),
        'p.value':     p_mat.min(axis=1).values,
        'statistic':   log2fc_mat.mean(axis=1).values,
        '_polarity':   df['Polarity'].values,
        '_DedupGroup': df['DedupGroup'].values,
    })
    n_before = len(mmg)
    mmg = mmg.dropna(subset=['mz', 'rtime', 'p.value', 'statistic'])
    n_dropped = n_before - len(mmg)
    if n_dropped:
        print(f"    dropped {n_dropped} rows with NaN (kept {len(mmg)})")
    return mmg.reset_index(drop=True)


def dedup_mmg(mmg):
    """Collapse adduct duplicates by DedupGroup: median m/z/RT/stat, min p."""
    if '_DedupGroup' not in mmg.columns:
        return mmg
    n_before = len(mmg)
    cols_out = ['mz', 'rtime', 'p.value', 'statistic']
    deduped = (mmg.groupby('_DedupGroup', sort=False)
               .agg({'mz': 'median', 'rtime': 'median',
                     'p.value': 'min', 'statistic': 'median'})
               .reset_index()[cols_out])
    n_after = len(deduped)
    if n_after < n_before:
        print(f"    dedup: {n_before} -> {n_after} "
              f"({n_before - n_after} collapsed)")
    return deduped


def split_by_polarity(mmg):
    """Split into POS and NEG subsets, keeping _DedupGroup for dedup."""
    cols = [c for c in ['mz', 'rtime', 'p.value', 'statistic', '_DedupGroup']
            if c in mmg.columns]
    pos = mmg[mmg['_polarity'] == 1][cols].reset_index(drop=True)
    neg = mmg[mmg['_polarity'] == 0][cols].reset_index(drop=True)
    return pos, neg


def save_and_run(mmg, condition_name, mode_str, base_dir):
    """Save tab-delimited input and run mummichog."""
    out_dir = os.path.join(base_dir, condition_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save input file inside the output directory
    txt_path = os.path.join(out_dir, f"{condition_name}_input.txt")

    # Mummichog 2.x requires headers: m/z, rtime, p-value, t-score
    mmg_out = mmg.rename(columns={
        'mz': 'm/z',
        'rtime': 'rtime',
        'p.value': 'p-value',
        'statistic': 't-score',
    })
    mmg_out.to_csv(txt_path, sep="\t", index=False)
    print(f"  [{condition_name}] {len(mmg_out)} features -> {mode_str}")

    # Mummichog 2.x ignores -o and writes to cwd, so we cd into out_dir
    cmd = MUMMICHOG_CMD + [
        "-f", txt_path,
        "-m", mode_str,
        "-c", str(CUTOFF),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)

    # Print last few lines of progress
    if result.stdout.strip():
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"    {line}")

    # Check for output: mummichog creates timestamped subdirectories
    new_items = [f for f in os.listdir(out_dir) if f != f"{condition_name}_input.txt"]
    if new_items:
        print(f"    Done -> {out_dir} ({len(new_items)} items)")
    elif result.returncode != 0:
        print(f"    *** FAILED (exit {result.returncode}) ***")
        if result.stderr.strip():
            print("    STDERR:", result.stderr[:800])
    else:
        print(f"    Warning: completed but no output found in {out_dir}")

    return result.returncode


def run_condition(df, log2fc_cols, p_cols, label, base_dir):
    """Full pipeline for one condition: pool -> split -> dedup -> run."""
    print(f"\n  {label}  ({len(log2fc_cols)} conditions pooled)")
    mmg_all = make_mmg_df(df, log2fc_cols, p_cols)
    mmg_pos, mmg_neg = split_by_polarity(mmg_all)
    mmg_pos = dedup_mmg(mmg_pos)
    mmg_neg = dedup_mmg(mmg_neg)
    print(f"  Final: POS={len(mmg_pos)}, NEG={len(mmg_neg)}")
    save_and_run(mmg_pos, f"{label}_POS", "positive", base_dir)
    save_and_run(mmg_neg, f"{label}_NEG", "negative", base_dir)


# Main

for mode, log2fc_path, do_run in [
    ('HILIC', HILIC_LOG2FC, RUN_HILIC),
    ('RP',    RP_LOG2FC,    RUN_RP),
]:
    if not do_run:
        continue

    print("\n" + "=" * 60)
    print(f"NT vs Plasma -- {mode}")
    print("=" * 60)

    df = load_and_prepare(log2fc_path, PRED_PATH, mode)
    base_dir = os.path.join(SAVE_PATH, mode)
    os.makedirs(base_dir, exist_ok=True)

    # Per-defect (pooled across 5 polymers)
    print(f"\n--- Per-defect (each defect pooled across polymers) ---")
    for defect in DEFECTS:
        fc_cols = [f'Log2FC_{p}_{defect}' for p in POLYMERS
                   if f'Log2FC_{p}_{defect}' in df.columns]
        p_cols  = [f'p_{p}_{defect}' for p in POLYMERS
                   if f'p_{p}_{defect}' in df.columns]
        if fc_cols:
            run_condition(df, fc_cols, p_cols, f"defect_{defect}", base_dir)

    # ALL_NT pooled (all 25 conditions)
    print(f"\n--- ALL_NT pooled (all 25 conditions) ---")
    all_fc = [f'Log2FC_{p}_{d}' for p in POLYMERS for d in DEFECTS
              if f'Log2FC_{p}_{d}' in df.columns]
    all_p  = [f'p_{p}_{d}' for p in POLYMERS for d in DEFECTS
              if f'p_{p}_{d}' in df.columns]
    if all_fc:
        run_condition(df, all_fc, all_p, "ALL_NT", base_dir)

    # Per-polymer (pooled across 5 defects)
    print(f"\n--- Per-polymer (each polymer pooled across defects) ---")
    for polymer in POLYMERS:
        fc_cols = [f'Log2FC_{polymer}_{d}' for d in DEFECTS
                   if f'Log2FC_{polymer}_{d}' in df.columns]
        p_cols  = [f'p_{polymer}_{d}' for d in DEFECTS
                   if f'p_{polymer}_{d}' in df.columns]
        if fc_cols:
            run_condition(df, fc_cols, p_cols, f"polymer_{polymer}", base_dir)

print("\n\nAll mummichog runs complete.")
print(f"Results: {SAVE_PATH}")