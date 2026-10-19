"""
Inter-batch reproducibility of the (AT)15-pH corona.

Input:  01_Global_Analysis_Combined.xlsx, repeated-run exports
Output: CV_results.xlsx, Analysis_summary.xlsx
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import COMBINED_PATH, OUT_BATCH, REPRO_RUN_DIR

OLD_PATH = COMBINED_PATH
NEW_DIR  = REPRO_RUN_DIR
OUT_DIR  = OUT_BATCH
OUT_DIR.mkdir(parents=True, exist_ok=True)

MZ_TOL         = 0.02   # Da
RT_TOL         = 0.3    # min
MIN_VALID_REPS = 2      # majority rule threshold
VALID_GAP      = {'No gap', 'Missing ions', 'No Gap', 'Missing Ions'}
MODES          = ['RP', 'HILIC']
NEW_FILES      = {
    'RP':    ['RP+.xlsx',    'RP-.xlsx'],
    'HILIC': ['HILIC+.xlsx', 'HILIC-.xlsx']
}

print('Loading old workbook...')
old_raw = pd.read_excel(OLD_PATH, sheet_name=None, header=None)
print(f'Old sheets: {list(old_raw.keys())}')

# Helper functions
def parse_mz(val):
    try:
        s = str(val)
        return float(s.split('_')[-1]) if '_' in s else float(s)
    except Exception:
        return np.nan


def remove_duplicates(data, mzs, rts):
    """Keep higher-intensity feature within m/z±MZ_TOL & RT±RT_TOL window."""
    mean_int     = data.mean(axis=0, skipna=True).values
    drop_indices = set()
    valid_idx    = np.where(~np.isnan(mzs))[0]
    sorted_idx   = valid_idx[np.argsort(mzs[valid_idx])]
    for k in range(len(sorted_idx)):
        i = sorted_idx[k]
        if i in drop_indices: continue
        for l in range(k + 1, len(sorted_idx)):
            j = sorted_idx[l]
            if j in drop_indices: continue
            if abs(mzs[j] - mzs[i]) > MZ_TOL: break
            if abs(rts[j] - rts[i]) < RT_TOL:
                if mean_int[j] > mean_int[i]: drop_indices.add(i); break
                else: drop_indices.add(j)
    mask = np.ones(len(mzs), dtype=bool)
    mask[list(drop_indices)] = False
    return mask


def majority_filter(gap_data, groups):
    """Keep features detected in >= 2/3 replicates in at least one group."""
    mask = np.zeros(gap_data.shape[1], dtype=bool)
    for idx in groups:
        sub  = gap_data.iloc[idx]
        det  = sub.isin(VALID_GAP).sum(axis=0)
        thresh = 2 if len(idx) >= 3 else 1
        mask |= (det >= thresh).values
    return mask


def gap_to_nan(data, gap):
    return data.where(gap.isin(VALID_GAP).values, np.nan)


def match_features(old_mzs, old_rts, new_mzs, new_rts):
    old_idx, new_idx = [], []
    for i, (mz_o, rt_o) in enumerate(zip(old_mzs, old_rts)):
        if np.isnan(mz_o) or np.isnan(rt_o): continue
        best_j, best_diff = None, np.inf
        for j, (mz_n, rt_n) in enumerate(zip(new_mzs, new_rts)):
            if np.isnan(mz_n) or np.isnan(rt_n): continue
            if abs(mz_n - mz_o) <= MZ_TOL and abs(rt_n - rt_o) <= RT_TOL:
                if abs(mz_n - mz_o) < best_diff:
                    best_diff = abs(mz_n - mz_o); best_j = j
        if best_j is not None:
            old_idx.append(i); new_idx.append(best_j)
    return old_idx, new_idx

# Stage 1: Load raw data
def load_old(mode_name):
    df  = old_raw[f'{mode_name}_comb']
    gap = old_raw[f'{mode_name}_gap']
    feat_ids = df.iloc[0, 3:].values.astype(str)
    mzs      = np.array([parse_mz(v) for v in df.iloc[1, 3:].values])
    rts      = pd.to_numeric(df.iloc[2, 3:], errors='coerce').values
    data     = df.iloc[4:, 3:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce')
    gap_data = gap.iloc[4:, 3:].reset_index(drop=True)
    meta     = df.iloc[4:, 0:3].reset_index(drop=True)
    meta.columns = ['Defect', 'Polymer', 'Method']

    mask_atph = (meta['Defect'] == 'pH')  & (meta['Polymer'] == 'AT')
    mask_ref  = (meta['Defect'] == 'Ref') & (meta['Polymer'] == 'Ref')
    mask_rows = mask_atph | mask_ref
    sub_data  = data[mask_rows.values].reset_index(drop=True)
    sub_gap   = gap_data[mask_rows.values].reset_index(drop=True)
    sub_meta  = meta[mask_rows.values].reset_index(drop=True)

    mask_u = remove_duplicates(sub_data, mzs, rts)
    sub_data_u = sub_data.iloc[:, mask_u].copy()
    sub_gap_u  = sub_gap.iloc[:, mask_u].copy()

    atph_idx = list(np.where((sub_meta['Defect'] == 'pH')  & (sub_meta['Polymer'] == 'AT'))[0])
    ref_idx  = list(np.where((sub_meta['Defect'] == 'Ref') & (sub_meta['Polymer'] == 'Ref'))[0])
    mask_d   = majority_filter(sub_gap_u, [g for g in [atph_idx, ref_idx] if g])

    X = gap_to_nan(sub_data_u.iloc[:, mask_d].copy(), sub_gap_u.iloc[:, mask_d].copy())
    X.columns = feat_ids[mask_u][mask_d]
    mzs_f = mzs[mask_u][mask_d]; rts_f = rts[mask_u][mask_d]

    atph_m = sub_meta[sub_meta['Defect'] == 'pH'][['Method']].copy().reset_index(drop=True)
    atph_m.insert(0, 'Run', 'Run1'); atph_m.insert(1, 'Batch', 'Batch1')
    ref_m  = sub_meta[sub_meta['Defect'] == 'Ref'][['Method']].copy().reset_index(drop=True)
    ref_m.insert(0, 'Run', 'Run1'); ref_m.insert(1, 'Batch', 'Ref1')
    meta_out = pd.concat([atph_m, ref_m], ignore_index=True)
    atph_rows = (sub_meta['Defect'] == 'pH').values
    ref_rows  = (sub_meta['Defect'] == 'Ref').values
    X_out = pd.concat([X[atph_rows].reset_index(drop=True),
                       X[ref_rows].reset_index(drop=True)], ignore_index=True)

    print(f'OLD {mode_name}: AT+pH n={atph_rows.sum()}, Ref n={ref_rows.sum()}, features={mask_d.sum()}')
    return meta_out, X_out, mzs_f, rts_f, X_out.columns.values


def load_new(mode_name):
    all_meta, all_X, all_gap_list = [], [], []
    all_mzs, all_rts, all_ids = [], [], []
    for fname in NEW_FILES[mode_name]:
        df  = pd.read_excel(NEW_DIR / fname, sheet_name='LC-MSMS',    header=None)
        gap = pd.read_excel(NEW_DIR / fname, sheet_name='gap status', header=None)
        feat_ids = df.iloc[0, 2:].values.astype(str)
        mzs      = pd.to_numeric(df.iloc[1, 2:], errors='coerce').values
        rts      = pd.to_numeric(df.iloc[2, 2:], errors='coerce').values
        data     = df.iloc[3:, 2:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce')
        gap_data = gap.iloc[3:, 2:].reset_index(drop=True)
        meta     = df.iloc[3:, 0:2].reset_index(drop=True)
        meta.columns = ['Method', 'Batch']
        all_meta.append(meta); all_X.append(data); all_gap_list.append(gap_data)
        all_mzs.append(mzs); all_rts.append(rts); all_ids.append(feat_ids)

    meta_c = all_meta[0].reset_index(drop=True)
    data_c = pd.concat([x.reset_index(drop=True) for x in all_X],        axis=1)
    gap_c  = pd.concat([x.reset_index(drop=True) for x in all_gap_list], axis=1)
    mzs_c  = np.concatenate(all_mzs); rts_c = np.concatenate(all_rts)
    ids_c  = np.concatenate(all_ids)

    mask_u = remove_duplicates(data_c, mzs_c, rts_c)
    data_u = data_c.iloc[:, mask_u].copy(); gap_u = gap_c.iloc[:, mask_u].copy()
    mzs_u  = mzs_c[mask_u]; rts_u = rts_c[mask_u]; ids_u = ids_c[mask_u]

    numeric_b = pd.to_numeric(meta_c['Batch'], errors='coerce')
    atph_idx  = list(np.where(numeric_b.notna())[0])
    ref_idx   = list(np.where(meta_c['Batch'].astype(str).str.strip() == 'Ref')[0])
    mask_d    = majority_filter(gap_u, [g for g in [atph_idx, ref_idx] if g])

    X = gap_to_nan(data_u.iloc[:, mask_d].copy(), gap_u.iloc[:, mask_d].copy())
    X.columns = ids_u[mask_d]

    def remap(b):
        b = str(b).strip()
        if b == 'Ref': return 'Ref2'
        if b == '1':   return 'Batch2'
        if b == '2':   return 'Batch3'
        return b

    meta_c['Batch'] = meta_c['Batch'].apply(remap)
    meta_c['Run']   = 'Run2'
    meta_c = meta_c[['Run', 'Batch', 'Method']].reset_index(drop=True)

    # Drop QC
    qc_mask = meta_c['Batch'] == 'QC'
    meta_c  = meta_c[~qc_mask].reset_index(drop=True)
    X       = X[~qc_mask.values].reset_index(drop=True)

    n_atph = meta_c['Batch'].isin(['Batch2','Batch3']).sum()
    n_ref  = (meta_c['Batch'] == 'Ref2').sum()
    print(f'NEW {mode_name}: AT+pH n={n_atph}, Ref n={n_ref}, features={mask_d.sum()}')
    return meta_c, X, mzs_u[mask_d], rts_u[mask_d], ids_u[mask_d]


consolidated_raw = {}
match_info = {}

for mode in MODES:
    print(f"\n{'='*40}\n  {mode}\n{'='*40}")
    old_meta, old_X, old_mzs, old_rts, old_ids = load_old(mode)
    new_meta, new_X, new_mzs, new_rts, new_ids = load_new(mode)
    m_old, m_new = match_features(old_mzs, old_rts, new_mzs, new_rts)
    print(f'Matched features: {len(m_old)}')
    shared_ids = old_ids[m_old]; shared_mzs = old_mzs[m_old]; shared_rts = old_rts[m_old]
    old_X_s = old_X.iloc[:, m_old].copy(); new_X_s = new_X.iloc[:, m_new].copy()
    old_X_s.columns = new_X_s.columns = shared_ids
    meta_all = pd.concat([old_meta, new_meta], ignore_index=True)
    X_all    = pd.concat([old_X_s.reset_index(drop=True),
                          new_X_s.reset_index(drop=True)], ignore_index=True)
    consolidated_raw[mode] = {'meta': meta_all, 'X': X_all,
                               'mzs': shared_mzs, 'rts': shared_rts, 'ids': shared_ids}
    match_info[mode] = len(m_old)

print('\nStage 1 done.')

# Export raw
out_raw = OUT_DIR / 'RunConsolidated_AT-pH_raw.xlsx'
with pd.ExcelWriter(out_raw, engine='openpyxl') as writer:
    for mode in MODES:
        meta = consolidated_raw[mode]['meta']
        X    = consolidated_raw[mode]['X']
        ids  = consolidated_raw[mode]['ids']
        mzs  = consolidated_raw[mode]['mzs']
        rts  = consolidated_raw[mode]['rts']
        mz_row  = pd.DataFrame([['m/z','','',''] + list(mzs)],
                                columns=['Run','Batch','Method','_'] + list(ids)).drop(columns=['_'])
        rt_row  = pd.DataFrame([['RT', '','',''] + list(rts)],
                                columns=['Run','Batch','Method','_'] + list(ids)).drop(columns=['_'])
        data_df = pd.concat([meta, X.reset_index(drop=True)], axis=1)
        pd.concat([mz_row, rt_row, data_df], ignore_index=True).to_excel(
            writer, sheet_name=mode, index=False)
    pd.DataFrame([
        ['Date',                     str(date.today())],
        ['Description',              'AT-pH corona + Plasma (Ref), Run1 (old) + Run2 (new)'],
        ['Feature matching',         f'm/z ±{MZ_TOL} Da, RT ±{RT_TOL} min'],
        ['Duplicate removal',        'Keep higher mean intensity within tolerance window'],
        ['Majority detection',       '>= 2/3 replicates detected in at least one group'],
        ['Gap handling',             'Full gap → NaN'],
        ['Batch mapping',            'Old AT+pH=Batch1, New Batch1=Batch2, New Batch2=Batch3'],
        ['Ref mapping',              'Old Ref=Ref1, New Ref=Ref2'],
        ['QC',                       'Excluded'],
        ['N matched features RP',    str(match_info['RP'])],
        ['N matched features HILIC', str(match_info['HILIC'])],
    ], columns=['Key', 'Value']).to_excel(writer, sheet_name='Metadata', index=False)
print(f'Saved: {out_raw.name}')

# Stage 2: Normalization (plasma, then per-sample median)

def normalize(meta, X):
    X_norm = X.copy().astype(float)
    meta   = meta.reset_index(drop=True)
    for (run, method), _ in meta.groupby(['Run', 'Method']):
        ref_mask  = (meta['Run'] == run) & (meta['Batch'].isin(['Ref1','Ref2'])) & \
                    (meta['Method'].astype(str).str.strip() == str(method).strip())
        atph_mask = (meta['Run'] == run) & (meta['Batch'].isin(['Batch1','Batch2','Batch3'])) & \
                    (meta['Method'].astype(str).str.strip() == str(method).strip())
        if ref_mask.sum() == 0 or atph_mask.sum() == 0: continue
        ref_mean = X_norm.loc[ref_mask.values].mean(axis=0).replace(0, np.nan)
        X_norm.loc[atph_mask.values] = X_norm.loc[atph_mask.values].div(ref_mean, axis=1)
    atph_all  = meta['Batch'].isin(['Batch1','Batch2','Batch3'])
    meta_atph = meta[atph_all.values].reset_index(drop=True)
    X_atph    = X_norm.loc[atph_all.values].reset_index(drop=True)
    row_med   = X_atph.median(axis=1).replace(0, np.nan)
    return meta_atph, X_atph.div(row_med, axis=0)


consolidated_norm = {}
for mode in MODES:
    print(f'Normalizing {mode}...')
    m, X = normalize(consolidated_raw[mode]['meta'], consolidated_raw[mode]['X'])
    consolidated_norm[mode] = {'meta': m, 'X': X}
    print(f'  {mode}: {len(m)} AT+pH samples, {X.shape[1]} features')
print('\nStage 2 done.')

# Export normalized
out_norm = OUT_DIR / 'RunConsolidated_AT-pH_normalized.xlsx'
with pd.ExcelWriter(out_norm, engine='openpyxl') as writer:
    for mode in MODES:
        meta = consolidated_norm[mode]['meta']
        X    = consolidated_norm[mode]['X']
        ids  = consolidated_raw[mode]['ids']
        mzs  = consolidated_raw[mode]['mzs']
        rts  = consolidated_raw[mode]['rts']
        mz_row  = pd.DataFrame([['m/z','','',''] + list(mzs)],
                                columns=['Run','Batch','Method','_'] + list(ids)).drop(columns=['_'])
        rt_row  = pd.DataFrame([['RT', '','',''] + list(rts)],
                                columns=['Run','Batch','Method','_'] + list(ids)).drop(columns=['_'])
        data_df = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        pd.concat([mz_row, rt_row, data_df], ignore_index=True).to_excel(
            writer, sheet_name=mode, index=False)
    pd.DataFrame([
        ['Date',                  str(date.today())],
        ['Normalization Step 1',  'AT+pH / mean(Ref) per (Run, Method) — plasma normalization'],
        ['Normalization Step 2',  'Sample / median(matched features) per sample'],
        ['Output',                'AT+pH only (Ref excluded after normalization)'],
        ['N matched features RP', str(match_info['RP'])],
        ['N matched features HILIC', str(match_info['HILIC'])],
    ], columns=['Key', 'Value']).to_excel(writer, sheet_name='Metadata', index=False)
print(f'Saved: {out_norm.name}')

# Stage 3: Inter-batch CV

def cv_intra(meta, X):
    records = []
    for (batch, method), idx in meta.groupby(['Batch','Method']).groups.items():
        grp = X.iloc[list(idx)]
        vf  = grp.notna().sum(axis=0) >= MIN_VALID_REPS
        grp = grp.loc[:, vf]
        if grp.empty: continue
        cv  = (grp.std(axis=0, ddof=1) / grp.mean(axis=0).abs()).where(grp.mean(axis=0) != 0) * 100
        records.append({'Batch': str(batch), 'Method': str(method),
                        'Median_CV%': float(np.nanmedian(cv)), 'N_features': int(vf.sum())})
    return pd.DataFrame(records)


def cv_inter(meta, X):
    batch_means = {}
    for (batch, method), idx in meta.groupby(['Batch','Method']).groups.items():
        grp = X.iloc[list(idx)]
        vf  = grp.notna().sum(axis=0) >= MIN_VALID_REPS
        batch_means[(batch, method)] = grp.loc[:, vf].mean(axis=0)
    records = []
    for method in meta['Method'].unique():
        means_df = pd.DataFrame({b: v for (b,m), v in batch_means.items() if m == method})
        if means_df.shape[1] < 2: continue
        means_df.columns = range(means_df.shape[1])
        means_df = means_df.dropna(how='any')
        if means_df.empty: continue
        cv = (means_df.std(axis=1, ddof=1) / means_df.mean(axis=1).replace(0, np.nan)) * 100
        records.append({'Method': str(method), 'Median_CV%': float(np.nanmedian(cv)),
                        'N_features': len(means_df)})
    if not records:
        return pd.DataFrame(columns=['Method','Median_CV%','N_features'])
    return pd.DataFrame(records).set_index('Method')


cv_results = {}
for mode in MODES:
    print(f'\n{mode}:')
    meta = consolidated_norm[mode]['meta']
    X    = consolidated_norm[mode]['X'].copy()
    X.columns = range(X.shape[1])
    inter = cv_inter(meta, X)
    cv_results[mode] = {'intra': cv_intra(meta, X), 'inter': inter}
    print(f'  Inter-batch CV:\n{inter.to_string()}')

with pd.ExcelWriter(OUT_DIR / 'CV_results.xlsx', engine='openpyxl') as writer:
    for mode in MODES:
        cv_results[mode]['intra'].to_excel(writer, sheet_name=f'{mode}_intra', index=False)
        cv_results[mode]['inter'].to_excel(writer, sheet_name=f'{mode}_inter')
print('Saved: CV_results.xlsx')

# Stage 4: RLE

rle_records = {}
for mode in MODES:
    meta = consolidated_raw[mode]['meta'].reset_index(drop=True)
    X    = consolidated_raw[mode]['X'].copy()
    X.columns = range(X.shape[1])

    # AT+pH only
    atph_mask = meta['Batch'].isin(['Batch1','Batch2','Batch3'])
    meta_atph = meta[atph_mask].reset_index(drop=True)
    X_atph    = X[atph_mask.values].reset_index(drop=True)

    # Global median across AT+pH samples only
    global_median = X_atph.median(axis=0).replace(0, np.nan)
    X_rle = np.log2(X_atph.div(global_median, axis=1))

    # Group by (Batch, Method)
    rows = []
    for (batch, method), idx in meta_atph.groupby(['Batch','Method']).groups.items():
        vals = X_rle.iloc[list(idx)].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0: continue
        rows.append({'Batch': batch, 'Method': str(method).strip(),
                     'Median_RLE': float(np.median(vals)),
                     'IQR_low':    float(np.percentile(vals, 25)),
                     'IQR_high':   float(np.percentile(vals, 75)),
                     'N_features': len(vals)})
    rle_records[mode] = pd.DataFrame(rows)

# Stage 5: PCA

pca_records = {}
for mode in MODES:
    meta = consolidated_raw[mode]['meta'].reset_index(drop=True)
    X    = consolidated_raw[mode]['X'].copy()
    X.columns = range(X.shape[1])

    global_median = X.median(axis=0).replace(0, np.nan)
    X_log2 = np.log2(X.div(global_median, axis=1))

    min_valid = len(meta) // 2
    keep  = X_log2.notna().sum(axis=0) >= min_valid
    X_pca = X_log2.loc[:, keep].apply(lambda col: col.fillna(col.median()), axis=0)

    pca_m  = PCA(n_components=2)
    coords = pca_m.fit_transform(StandardScaler().fit_transform(X_pca))
    pc1_var = pca_m.explained_variance_ratio_[0] * 100
    pc2_var = pca_m.explained_variance_ratio_[1] * 100

    # Export coords
    df_c = meta.copy()
    df_c['PC1'] = coords[:, 0]; df_c['PC2'] = coords[:, 1]
    df_c['PC1_var'] = round(pc1_var, 1); df_c['PC2_var'] = round(pc2_var, 1)
    df_c['N_features'] = int(keep.sum())
    pca_records[mode] = df_c

# Export summary
out_summary = OUT_DIR / 'Analysis_summary.xlsx'
with pd.ExcelWriter(out_summary, engine='openpyxl') as writer:
    for mode in MODES:
        cv_results[mode]['intra'].to_excel(writer, sheet_name=f'{mode}_CV_intra',   index=False)
        cv_results[mode]['inter'].to_excel(writer, sheet_name=f'{mode}_CV_inter')
        rle_records[mode].to_excel(writer,         sheet_name=f'{mode}_RLE',        index=False)
        pca_records[mode].to_excel(writer,         sheet_name=f'{mode}_PCA',        index=False)
print(f'Saved: {out_summary.name}')
