"""
Collapses features that share the same annotated name (duplicate adducts/peaks).
"""

import re
import numpy as np
import pandas as pd

# Helpers

def _strip_dup_suffix(name: str) -> str:
    """Remove trailing .1, .2 etc."""
    return re.sub(r'\.\d+$', '', str(name))


def _is_numeric_id(name: str) -> bool:
    """True if the base name (suffix stripped) is purely numeric."""
    base = _strip_dup_suffix(str(name))
    return base.replace('.', '', 1).isdigit()


# Generic class-prefix names (e.g. PC.4) are distinct unknowns, not duplicates
_GENERIC_PREFIXES = (
    'PC', 'PE', 'PI', 'PG', 'PS', 'PA', 'SM', 'Cer', 'CE', 'DG', 'TG', 'MG',
    'LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA',
    'FA', 'Car',
    'HexCer', 'Hex2Cer', 'Hex3Cer', 'CerPE',
)
_GENERIC_RE = re.compile(
    r'^(' + '|'.join(re.escape(p) for p in _GENERIC_PREFIXES) + r')\.\d+$',
    re.IGNORECASE
)


def _is_generic_prefix(name: str) -> bool:
    """True if feature name is a class prefix + numeric index (e.g. PC.4, FA.11)."""
    return bool(_GENERIC_RE.match(str(name)))


# Core functions

def add_dedup_group(df: pd.DataFrame, feature_col: str = 'Feature') -> pd.DataFrame:
    """Add a 'DedupGroup' column (features sharing a base annotated name)."""
    df = df.copy()

    def _assign_group(feat):
        feat_str = str(feat)
        if _is_numeric_id(feat_str) or _is_generic_prefix(feat_str):
            return feat_str  # unique — no collapsing
        return _strip_dup_suffix(feat_str)

    df['DedupGroup'] = df[feature_col].apply(_assign_group)
    df['DedupSize'] = df.groupby('DedupGroup')['DedupGroup'].transform('count')

    n_groups = df['DedupGroup'].nunique()
    n_dup_groups = (df.groupby('DedupGroup').size() > 1).sum()
    n_dup_features = df[df['DedupSize'] > 1].shape[0]
    print(f"  DedupGroup: {len(df)} features → {n_groups} groups "
          f"({n_dup_groups} groups with {n_dup_features} duplicate features)")

    return df


def deduplicate(df: pd.DataFrame,
                method: str = 'median',
                feature_col: str = 'Feature') -> pd.DataFrame:
    """Collapse each DedupGroup into one row (method: 'median' or 'ivw')."""
    if 'DedupGroup' not in df.columns:
        df = add_dedup_group(df, feature_col)

    fc_cols = [c for c in df.columns if c.startswith('Log2FC_')]
    se_cols = [c for c in df.columns if c.startswith('SE_')]
    meta_cols = [c for c in ['MetaboliteClass', 'ClassSource', 'ClassConfidence']
                 if c in df.columns]
    p_cols = [c for c in df.columns if c.startswith('p_')]
    nlp_cols = [c for c in df.columns if c.startswith('negLogP_')]
    nm_cols = [c for c in df.columns if c.startswith('nM_')]

    rows = []
    for group_name, grp in df.groupby('DedupGroup', sort=False):
        row = {}
        row[feature_col] = group_name
        row['DedupSize'] = len(grp)

        # Metadata: take from first member
        for mc in meta_cols:
            row[mc] = grp[mc].iloc[0]

        if len(grp) == 1:
            # Singleton — pass through all columns
            for c in fc_cols + se_cols + p_cols + nlp_cols + nm_cols:
                row[c] = grp[c].iloc[0]
        else:
            # Collapse
            if method == 'median':
                for c in fc_cols:
                    row[c] = grp[c].median()
                for c in se_cols:
                    # For SE, use median too (conservative)
                    row[c] = grp[c].median()
            elif method == 'ivw':
                for fc_c in fc_cols:
                    # Find matching SE column
                    se_c = fc_c.replace('Log2FC_', 'SE_')
                    if se_c in se_cols:
                        fc_vals = grp[fc_c].values.astype(float)
                        se_vals = grp[se_c].values.astype(float)
                        ok = np.isfinite(fc_vals) & np.isfinite(se_vals) & (se_vals > 0)
                        if ok.sum() == 0:
                            row[fc_c] = np.nan
                            row[se_c] = np.nan
                        elif ok.sum() == 1:
                            row[fc_c] = fc_vals[ok][0]
                            row[se_c] = se_vals[ok][0]
                        else:
                            w = 1.0 / (se_vals[ok] ** 2)
                            row[fc_c] = np.sum(w * fc_vals[ok]) / np.sum(w)
                            row[se_c] = 1.0 / np.sqrt(np.sum(w))
                    else:
                        row[fc_c] = grp[fc_c].median()

            # Recompute p and negLogP from collapsed FC/SE
            for fc_c in fc_cols:
                se_c = fc_c.replace('Log2FC_', 'SE_')
                p_c = fc_c.replace('Log2FC_', 'p_')
                nlp_c = fc_c.replace('Log2FC_', 'negLogP_')
                nm_c = fc_c.replace('Log2FC_', 'nM_')

                fc_val = row.get(fc_c, np.nan)
                se_val = row.get(se_c, np.nan)

                if np.isfinite(fc_val) and np.isfinite(se_val) and se_val > 0:
                    from scipy.stats import norm
                    z = fc_val / se_val
                    p = 2 * norm.sf(abs(z))
                    row[p_c] = p
                    row[nlp_c] = -np.log10(p + 1e-300)
                else:
                    row[p_c] = np.nan
                    row[nlp_c] = np.nan

                # nMethods: take max across group
                if nm_c in nm_cols:
                    row[nm_c] = grp[nm_c].max()

        rows.append(row)

    result = pd.DataFrame(rows)

    # Reorder columns to match original
    col_order = [feature_col] + meta_cols + ['DedupSize']
    for c in df.columns:
        if c in result.columns and c not in col_order:
            col_order.append(c)
    col_order = [c for c in col_order if c in result.columns]
    result = result[col_order]

    print(f"  Deduplicated: {len(df)} → {len(result)} rows (method={method})")
    return result


# Sensitivity comparison

def sensitivity_comparison(df: pd.DataFrame,
                           classes: list,
                           conditions: list = None,
                           feature_col: str = 'Feature') -> pd.DataFrame:
    """Class-level enrichment with full vs deduplicated features."""
    if 'DedupGroup' not in df.columns:
        df = add_dedup_group(df, feature_col)

    df_dedup = deduplicate(df, method='median', feature_col=feature_col)

    rows = []
    for cls in classes:
        for label, d in [('All features', df), ('Deduplicated', df_dedup)]:
            sub = d[d['MetaboliteClass'] == cls]

            row = {'Class': cls, 'Version': label, 'n': len(sub)}

            # ALL_NT summary
            if 'Log2FC_ALL_NT' in sub.columns:
                vals = sub['Log2FC_ALL_NT'].dropna()
                row['median_Log2FC_ALL_NT'] = vals.median() if len(vals) > 0 else np.nan
                row['pct_positive_ALL_NT'] = (
                    100 * (vals > 0).mean() if len(vals) > 0 else np.nan)

            # Per-condition consistency
            if conditions:
                n_pos = 0
                n_tested = 0
                for cond in conditions:
                    fc_col = f'Log2FC_{cond}'
                    if fc_col in sub.columns:
                        cond_vals = sub[fc_col].dropna()
                        if len(cond_vals) >= 3:
                            n_tested += 1
                            if cond_vals.median() > 0:
                                n_pos += 1
                row['n_conditions_tested'] = n_tested
                row['n_conditions_positive'] = n_pos
                row['pct_conditions_positive'] = (
                    100 * n_pos / n_tested if n_tested > 0 else np.nan)

            rows.append(row)

    result = pd.DataFrame(rows)

    # Add direction agreement column
    summary_rows = []
    for cls in classes:
        all_row = result[(result['Class'] == cls) & (result['Version'] == 'All features')]
        ded_row = result[(result['Class'] == cls) & (result['Version'] == 'Deduplicated')]
        if len(all_row) > 0 and len(ded_row) > 0:
            med_all = all_row['median_Log2FC_ALL_NT'].values[0]
            med_ded = ded_row['median_Log2FC_ALL_NT'].values[0]
            same_dir = (np.sign(med_all) == np.sign(med_ded))
            summary_rows.append({
                'Class': cls,
                'n_all': int(all_row['n'].values[0]),
                'n_dedup': int(ded_row['n'].values[0]),
                'n_collapsed': int(all_row['n'].values[0]) - int(ded_row['n'].values[0]),
                'median_all': med_all,
                'median_dedup': med_ded,
                'same_direction': same_dir,
                'pct_pos_all': all_row['pct_positive_ALL_NT'].values[0],
                'pct_pos_dedup': ded_row['pct_positive_ALL_NT'].values[0],
            })

    summary = pd.DataFrame(summary_rows)
    return result, summary


# Run on a Log2FC Excel file

def run_sensitivity(log2fc_path: str,
                    sheet: str = 'NT_vs_Plasma',
                    classes: list = None,
                    mode: str = 'HILIC') -> tuple:
    """
    Load a Log2FC file and run the full sensitivity comparison.

    Returns (full_comparison_df, summary_df)
    """
    df = pd.read_excel(log2fc_path, sheet_name=sheet)

    if classes is None:
        classes = [c for c in df['MetaboliteClass'].dropna().unique()
                   if c != 'Low Confidence']

    # Detect condition columns
    fc_cols = [c for c in df.columns if c.startswith('Log2FC_')
               and c != 'Log2FC_ALL_NT']
    conditions = [c.replace('Log2FC_', '') for c in fc_cols]

    print(f"\n{'='*60}")
    print(f"  Sensitivity analysis: {mode} ({sheet})")
    print(f"{'='*60}")

    df = add_dedup_group(df)
    full, summary = sensitivity_comparison(df, classes, conditions)

    print(f"\n  Summary:")
    print(f"  {'Class':<28} {'n_all':>6} {'n_ded':>6} {'Δ':>4} "
          f"{'med_all':>8} {'med_ded':>8} {'same?':>6}")
    print(f"  {'-'*72}")
    for _, r in summary.iterrows():
        marker = '✓' if r['same_direction'] else '✗ DIFF'
        print(f"  {r['Class']:<28} {r['n_all']:>6} {r['n_dedup']:>6} "
              f"{r['n_collapsed']:>4} {r['median_all']:>+8.3f} "
              f"{r['median_dedup']:>+8.3f} {marker:>6}")

    return full, summary


if __name__ == '__main__':
    # Demo: run on HILIC and RP
    for mode, path in [('HILIC', 'HILIC_ALL_Log2FC.xlsx'),
                       ('RP', 'RP_ALL_Log2FC.xlsx')]:
        try:
            full, summary = run_sensitivity(path, mode=mode)
        except FileNotFoundError:
            print(f"  {path} not found, skipping")
