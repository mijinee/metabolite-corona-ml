"""
QWD effect on the corona: class-level ΔLog2FC, polymer x QWD heatmap data, enrichment scores.

Input:  HILIC_ALL_Log2FC.xlsx, RP_ALL_Log2FC.xlsx, metabolite_class_predictions.xlsx
Output: qwd_effect.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import (DEFECT_LABELS, HILIC_LOG2FC, HILIC_TRACK1, HILIC_TRACK2, OUT_FIGURES,
                    POLYMERS, PRED_PATH, QWD_DEFECTS, RP_CLASSES, RP_LOG2FC)
from utils.dedup_features import add_dedup_group, deduplicate
from utils.stats import bh_fdr, running_enrichment_score

OUT_DIR  = OUT_FIGURES / "qwd_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "qwd_effect.xlsx"

COMP_DEFECTS   = QWD_DEFECTS
DEFECT_ORDER   = ["3F", "DN", "C", "pH"]      # hydrophobic -> polar


# Helpers
def load_and_join_rt(log2fc_path, pred_path, mode, sheet='QWD_vs_Pristine'):
    df = pd.read_excel(log2fc_path, sheet_name=sheet)
    pred = pd.read_excel(pred_path, sheet_name=f'{mode}_Predictions')
    pred['FeatureID'] = pred['FeatureID'].astype(str)
    rt_lookup = pred.set_index('FeatureID')[['RT', 'MZ']].to_dict('index')
    df['Feature'] = df['Feature'].astype(str)
    df['RT'] = df['Feature'].map(lambda f: rt_lookup.get(f, {}).get('RT', np.nan))
    df['MZ'] = df['Feature'].map(lambda f: rt_lookup.get(f, {}).get('MZ', np.nan))
    return df


def dedup_with_rt(df):
    if 'DedupGroup' not in df.columns:
        df = add_dedup_group(df)
    rt_by_group = df.groupby('DedupGroup')['RT'].median()
    mz_by_group = df.groupby('DedupGroup')['MZ'].median()
    df_d = deduplicate(df, method='median')
    df_d['RT'] = df_d['Feature'].map(rt_by_group)
    df_d['MZ'] = df_d['Feature'].map(mz_by_group)
    return df, df_d


# Load data
print("Loading QWD data...")
hilic_qwd = load_and_join_rt(HILIC_LOG2FC, PRED_PATH, 'HILIC', 'QWD_vs_Pristine')
rp_qwd    = load_and_join_rt(RP_LOG2FC, PRED_PATH, 'RP', 'QWD_vs_Pristine')

print("Deduplicating...")
hilic_qwd, hilic_qwd_d = dedup_with_rt(hilic_qwd)
rp_qwd,    rp_qwd_d    = dedup_with_rt(rp_qwd)


# Class-level ΔLog2FC
print("\n" + "="*60)
print("Class-level ΔLog2FC")
print("="*60)

classlevel_rows = []
for mode_label, df, classes in [
    ('RP',    rp_qwd_d,    RP_CLASSES),
    ('HILIC', hilic_qwd_d, HILIC_TRACK1),
]:
    fc_cols = [c for c in df.columns if c.startswith('Log2FC_')]

    # Per-feature pooled QWD effect
    df['median_QWD_FC'] = df[fc_cols].median(axis=1)

    # Class median RT
    for cls_idx, cls in enumerate(classes):
        mask = df['MetaboliteClass'] == cls
        vals = df.loc[mask, 'median_QWD_FC'].dropna()
        rt_vals = df.loc[mask, 'RT'].dropna()

        if len(vals) < 3:
            continue

        # One-sample Wilcoxon signed-rank test vs zero
        try:
            stat, p_val = wilcoxon(vals, alternative='two-sided')
        except Exception:
            p_val = np.nan

        classlevel_rows.append({
            'Mode': mode_label,
            'MetaboliteClass': cls,
            'Class_Index': cls_idx,
            'Median_RT': rt_vals.median() if len(rt_vals) > 0 else np.nan,
            'n_features': len(vals),
            'median_DLog2FC': vals.median(),
            'mean_DLog2FC': vals.mean(),
            'q25': vals.quantile(0.25),
            'q75': vals.quantile(0.75),
            'q10': vals.quantile(0.10),
            'q90': vals.quantile(0.90),
            'pct_positive': 100 * (vals > 0).mean(),
            'p_value': p_val,
        })

df_classlevel = pd.DataFrame(classlevel_rows)
# BH correction within each mode
for mode in ['RP', 'HILIC']:
    mask = df_classlevel['Mode'] == mode
    df_classlevel.loc[mask, 'q_value'] = bh_fdr(
        df_classlevel.loc[mask, 'p_value'].values)
df_classlevel['significant'] = df_classlevel['q_value'] < 0.05

# Sort by median RT within each mode
df_classlevel = df_classlevel.sort_values(['Mode', 'Median_RT']).reset_index(drop=True)

print(f"  {'Mode':<6} {'Class':<28} {'n':>5} {'med_ΔFC':>9} {'IQR':>18} "
      f"{'q_val':>8} {'sig':>4}")
print(f"  {'-'*85}")
for _, r in df_classlevel.iterrows():
    sig_mark = '*' if r['significant'] else ''
    print(f"  {r['Mode']:<6} {r['MetaboliteClass']:<28} {r['n_features']:>5} "
          f"{r['median_DLog2FC']:>+9.3f} [{r['q25']:>+.3f}, {r['q75']:>+.3f}] "
          f"{r['q_value']:>8.4f} {sig_mark:>4}")


# Polymer x QWD heatmap data
print("\n" + "="*60)
print("Polymer x QWD heatmap data")
print("="*60)

bubble_rows = []
for mode_label, df, classes in [
    ('RP',    rp_qwd_d,    RP_CLASSES),
    ('HILIC', hilic_qwd_d, HILIC_TRACK1),
]:
    for cls_idx, cls in enumerate(classes):
        mask_cls = df['MetaboliteClass'] == cls
        rt_vals = df.loc[mask_cls, 'RT'].dropna()
        median_rt = rt_vals.median() if len(rt_vals) > 0 else np.nan

        # Global stats for z-score (across all defects + polymers)
        all_vals = []
        for poly in POLYMERS:
            for defect in COMP_DEFECTS:
                col = f'Log2FC_{poly}_{defect}'
                if col in df.columns:
                    all_vals.extend(df.loc[mask_cls, col].dropna().values)
        all_vals = np.array(all_vals)
        global_mean = all_vals.mean() if len(all_vals) > 0 else 0
        global_std  = all_vals.std() if len(all_vals) > 1 else 1

        for poly in POLYMERS:
            for defect in DEFECT_ORDER:
                col = f'Log2FC_{poly}_{defect}'
                if col not in df.columns:
                    continue
                vals = df.loc[mask_cls, col].dropna()
                if len(vals) < 3:
                    continue

                med = vals.median()
                z = (med - global_mean) / global_std if global_std > 0 else 0
                n_enriched = int((vals > 0).sum())

                bubble_rows.append({
                    'MetaboliteClass': cls,
                    'Number_of_Features': len(vals),
                    'Median_Log2FC': med,
                    'Enriched_Features': n_enriched,
                    'Enriched_Fraction': (vals > 0).mean(),
                    'Mode': mode_label,
                    'Polymer': poly,
                    'Defect': defect,
                    'Defect_Label': DEFECT_LABELS[defect],
                    'Median_RT': median_rt,
                    'Class_Index': cls_idx,
                    'z_score_Median_Log2FC': z,
                })

df_bubble = pd.DataFrame(bubble_rows)

# Enrichment scores, HILIC Track 2
print("\n" + "="*60)
print("Enrichment scores (Track 2, per defect)")
print("="*60)

es_curve_rows = []
es_summary_rows = []

for defect in DEFECT_ORDER:
    # Compute mean QWD effect across all 5 polymers for this defect
    defect_fc_cols = [f'Log2FC_{p}_{defect}' for p in POLYMERS
                      if f'Log2FC_{p}_{defect}' in hilic_qwd_d.columns]

    ranked = hilic_qwd_d[['Feature', 'MetaboliteClass'] + defect_fc_cols].copy()
    ranked['mean_FC'] = ranked[defect_fc_cols].mean(axis=1)
    ranked = ranked.dropna(subset=['mean_FC'])
    ranked = ranked.sort_values('mean_FC', ascending=False).reset_index(drop=True)

    for cls in HILIC_TRACK2:
        mask = (ranked['MetaboliteClass'] == cls).values
        if mask.sum() < 5:
            continue

        scores, es, es_idx = running_enrichment_score(ranked['mean_FC'].values, mask)

        for rank_i in range(len(scores)):
            es_curve_rows.append({
                'Defect': defect,
                'Defect_Label': DEFECT_LABELS[defect],
                'Class': cls,
                'Rank': rank_i + 1,
                'Rank_Fraction': (rank_i + 1) / len(scores),
                'Running_ES': scores[rank_i],
                'Log2FC_at_rank': ranked['mean_FC'].iloc[rank_i],
            })

        es_summary_rows.append({
            'Defect': defect,
            'Defect_Label': DEFECT_LABELS[defect],
            'Class': cls,
            'n_class': int(mask.sum()),
            'n_total': len(ranked),
            'ES': es,
            'ES_rank': es_idx + 1,
            'Direction': 'QWD-enriched' if es > 0 else 'QWD-depleted',
        })

    print(f"  {defect} ({DEFECT_LABELS[defect]}): {len(ranked)} ranked features")

df_es_curves  = pd.DataFrame(es_curve_rows)
df_es_summary = pd.DataFrame(es_summary_rows)

# Print ES summary
print(f"\n  {'Class':<28}", '  '.join([f'{DEFECT_LABELS[d]:>10}' for d in DEFECT_ORDER]))
print(f"  {'-'*80}")
for cls in HILIC_TRACK2:
    es_vals = []
    for d in DEFECT_ORDER:
        row = df_es_summary[(df_es_summary['Defect'] == d) & (df_es_summary['Class'] == cls)]
        es_vals.append(f"{row['ES'].values[0]:>+10.3f}" if len(row) > 0 else f"{'n/a':>10}")
    print(f"  {cls:<28} {'  '.join(es_vals)}")


# Export
print(f"\nExporting to: {OUT_PATH}")

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    df_classlevel.to_excel(writer, sheet_name='ClassLevel_DeltaFC', index=False)
    df_bubble.to_excel(writer,     sheet_name='Heatmap_PolymerxQWD', index=False)
    df_es_curves.to_excel(writer,  sheet_name='ES_RunningCurves_PerDefect', index=False)
    df_es_summary.to_excel(writer, sheet_name='ES_Summary_PerDefect', index=False)

    guide = pd.DataFrame({
        'Sheet': ['ClassLevel_DeltaFC', 'Heatmap_PolymerxQWD',
                  'ES_RunningCurves_PerDefect', 'ES_Summary_PerDefect'],
        'Content': [
            'Class-level ΔLog2FC (QWD vs pristine). Per-feature median across 20 comparisons, '
            'then class median / IQR. Wilcoxon signed-rank test vs 0, BH-FDR q within mode. '
            'RP classes + HILIC Track 1, sorted by Median_RT.',
            'Median ΔLog2FC per class x polymer x defect (RP classes + HILIC Track 1). '
            'Also enriched fraction and z-score of the median within each class. '
            'Defects ordered by logP: 3F > DN > C > pH.',
            'Running enrichment score curves per defect x Track 2 class. Features ranked by the mean '
            'ΔLog2FC across the 5 polymers for each defect.',
            'Enrichment score per defect x Track 2 class. Positive ES = enriched by the QWD; negative = depleted.',
        ],
    })
    guide.to_excel(writer, sheet_name='Guide', index=False)

print("Done.")
