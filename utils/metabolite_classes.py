"""
Class labels, manual curation and class merging shared by the classification scripts.
"""

import re

import pandas as pd


def strip_dup_suffix(name: str) -> str:
    return re.sub(r"\.\d+$", "", str(name))


# Rule-based classifier

def classify_metabolite(name: str) -> str:
    n = strip_dup_suffix(str(name)).lower()

    if any(x in n for x in ["lyso", "lpa(", "lpc(", "lpe(", "lpg(", "lpi(", "lps("]):
        return "Lysophospholipids"
    if re.match(r"^lp[cepgais]", n):
        return "Lysophospholipids"
    if any(x in n for x in ["pc(", "pe(", "ps(", "pg(", "pi(", "pa(", "glycerophospho", "phosphatidyl"]):
        return "Phospholipids"
    if any(x in n for x in ["ceramide", "sphing", "sm(", "cer(", "ganglioside"]):
        return "Sphingolipids"
    if any(x in n for x in ["triacylglycerol", "diacylglycerol", "monoacylglycerol",
                              "tg(", "dg(", "mg(", "triglyceride", "diglyceride"]):
        return "Acylglycerols"
    if any(x in n for x in ["stearamide", "palmitamide", "oleamide", "margaroyl-ea", "ethanolamine"]):
        return "Fatty Amides"
    if "amide" in n and not any(ex in n for ex in ["nicotin", "carboxamide", "sulfonamide"]):
        return "Fatty Amides"
    if "carnitine" in n:
        return "Acylcarnitines"

    fatty_markers = [
        "fa(", "eicosa", "octadeca", "decenoic", "linole", "hexadecanoic", "octadecanoic",
        "docosanoic", "tetracosanoic", "heneicosylic", "lignoceric", "heptacosane",
        "tricosenoic", "docosatetraenoic", "hexacosanoic", "oxohexacosanoic", "hyenic acid",
        "3-oxohexacos", "4,7,10,13-docosa",
    ]
    if any(x in n for x in fatty_markers) or re.match(r"^fa$", n.strip()):
        return "Fatty Acids"

    if any(x in n for x in ["pregn", "androst", "cholest", "sterol", "cortisol", "cheno",
                              "bile", "cholic", "testosterone", "cortistatin", "estradiol",
                              "estrone", "norethindrone", "dehydroepiandrosterone", "dhea",
                              "aldosterone", "ursodeoxycholic"]):
        return "Steroids & Bile Acids"
    if any(x in n for x in ["adenine", "guanine", "cytosine", "thymine", "uracil", "xanthine",
                              "inosine", "adenosine", "guanosine", "nucleotide", "nucleoside",
                              "camp", "gmp", "ump", "imp", "uridine", "thymidine", "cytidine"]):
        return "Nucleotides & Bases"
    if any(x in n for x in ["glucose", "fructose", "galactose", "mannose", "ribose", "inositol",
                              "mannitol", "arabitol", "saccharide", "lactose", "sucrose",
                              "glucuron", "glucoside", "pyranose", "furanose", "deoxyribose",
                              "xylose", "allopyranose", "glucopyranosyl"]):
        return "Sugars & Polyols"

    amino_acids = [
        "alanine", "arginine", "asparagine", "aspartic", "cysteine", "glutamine", "glutamic",
        "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine",
        "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine",
        "taurine", "peptide", "glutathione", "hippuric", "hippurate", "betaine",
        "sarcosine", "homocysteine", "ornithine", "citrulline", "hydroxyproline",
    ]
    if any(x in n for x in amino_acids):
        return "Amino Acids & Peptides"
    if any(x in n for x in ["indole", "indolin", "tryptamine", "serotonin", "melatonin",
                              "kynurenine", "kynurenic", "indoleacet"]):
        return "Indoles & Trp Metabolites"
    if any(x in n for x in ["lactate", "pyruvate", "succinate", "citrate", "malate", "fumarate",
                              "glutarate", "kynurenate", "oxalate", "tartaric", "pantothen",
                              "pantoate", "shikimic", "quinic", "gluconic"]):
        return "Organic Acids"
    if "acid" in n and not any(ex in n for ex in ["nucleic", "amino"]):
        return "Organic Acids"
    if any(x in n for x in ["sulfate", "sulfonic", "sulfonate", "sulfooxy"]):
        return "Sulfates & Sulfonates"
    if any(x in n for x in ["terpene", "terpen", "farnes", "geranyl", "linalool", "limonene",
                              "carotene", "retinol", "squalene", "menthol", "camphor", "thymol",
                              "methylfarnesoate", "albaflavenol", "sesqui", "diterpene", "monoterpene"]):
        return "Terpenoids"
    if any(x in n for x in ["alkaloid", "cocaine", "morphine", "quinine", "cuscohygrine",
                              "evocarpine", "berberine", "piperine", "nicotine", "atropine",
                              "hygrine", "strychnine", "colchicine"]):
        return "Alkaloids"
    if any(x in n for x in ["flavone", "flavonol", "flavanone", "isoflavone", "anthocyanin",
                              "quercetin", "kaempferol", "rutin", "naringenin", "resveratrol",
                              "catechin", "epicatechin", "curcumin", "coumarin", "naphthoquinone",
                              "bisphenol", "hydroxynaphtho", "xanthoxin"]):
        return "Polyphenols & Flavonoids"
    if any(x in n for x in ["vitamin", "riboflavin", "thiamine", "cobalamin", "pyridoxine",
                              "folate", "biotin", "tocopherol", "ascorbic", "nicotinamide",
                              "niacin", "coenzyme", "nad(", "fad", "coa"]):
        return "Vitamins & Cofactors"

    drugs_explicit = ["rivoglitazone", "budipine", "terbinafine", "fusaricate",
                      "diphenhydramine", "caffeine", "cotinine", "salirasib",
                      "imicyafos", "nilvadipine", "carbamazepine", "thiotetromycin"]
    drug_suffixes  = ["dipine", "azepine", "oxacin", "mycin", "cillin", "azole", "caine"]
    if any(x in n for x in drugs_explicit) or any(n.endswith(s) for s in drug_suffixes):
        return "Drugs & Xenobiotics"
    if any(x in n for x in ["benzyl", "phenyl", "benzo", "naphthal"]):
        return "Benzenoids"

    return "Other/Unknown"


# Class merging

MERGE_MAP_HILIC = {
    "Acylglycerols": "Complex Lipids",
    "Sphingolipids": "Complex Lipids",
    "Steroids & Bile Acids": "Complex Lipids",
    "Indoles & Trp Metabolites": "Amino Acids & Peptides",
    "Polar Metabolites": "Amino Acids & Peptides",
    "Sugars & Polyols": "Organic Acids",
    "Nucleotides & Bases": "Secondary & Xenobiotic",
    "Terpenoids": "Secondary & Xenobiotic",
    "Alkaloids": "Secondary & Xenobiotic",
    "Polyphenols & Flavonoids": "Secondary & Xenobiotic",
    "Vitamins & Cofactors": "Secondary & Xenobiotic",
    "Secondary Metabolites": "Secondary & Xenobiotic",
    "Drugs & Xenobiotics": "Secondary & Xenobiotic",
    "Fatty Acids": "Fatty Acids & Amides",
    "Fatty Amides": "Fatty Acids & Amides",
    "Sulfates & Sulfonates": "Organic Acids",
}
MERGE_MAP_RP = {
    "Acylglycerols": "Complex Lipids",
    "Sphingolipids": "Complex Lipids",
    "Steroids & Bile Acids": "Complex Lipids",
    "Vitamins & Cofactors": "Complex Lipids",
}
MERGE_MAPS = {"HILIC": MERGE_MAP_HILIC, "RP": MERGE_MAP_RP}


# Corrections to known misclassified ground-truth labels (HILIC)

_OA_TO_FA = {
    "Icosadienoicacid": "Fatty Acids", "Docosahexaenoicacid": "Fatty Acids",
    "Margaric acid": "Fatty Acids", "Palmitoleicacid": "Fatty Acids",
    "Myristicacid": "Fatty Acids", "Nonadecanoicacid": "Fatty Acids",
    "Hexadecanedioicacid;Thapsicacid": "Fatty Acids",
    "[FAmethyl(14:0)]12-methyl-tetradecanoicacid": "Fatty Acids",
    "[FAdioxo_hydrox]6_9-dioxo-11R_15S-dihydroxy-13E-prostenoicacid": "Fatty Acids",
    "[FAoxo(4:0)]11-oxo-5Z_9_12E_14E-prostatetraenoicacid": "Fatty Acids",
    "Ambrettolic acid": "Fatty Acids", "Gallicynoic acid A": "Fatty Acids",
    "3-Acetoxystearic acid": "Fatty Acids", "2-hydroxy palmitic acid": "Fatty Acids",
    "2-hydroxy-heptadecanoic acid": "Fatty Acids", "Japanic acid": "Fatty Acids",
    "Nonadecandioic acid": "Fatty Acids", "11-Methyl-2Z,5Z-dodecadienoic acid": "Fatty Acids",
    "18S-hydroxyneodihydroprotolichesterinic acid": "Fatty Acids",
    "3-C-Carboxy-2-decyl-2,4-dideoxypentaric acid": "Fatty Acids",
    "3-carboxy-4-methyl-5-(1-hydroxypropyl)-2-furanpropionic acid": "Fatty Acids",
    "3-carboxy-4-methyl-5-pentyl-2-furanpropanoic acid": "Fatty Acids",
    "3-carboxy-4-methyl-5-propyl-2-furanpropanoic acid": "Fatty Acids",
    "3-hydroxy-dodecanedioic acid": "Fatty Acids", "2-hydroxy-decanedioic acid": "Fatty Acids",
    "3-Hydroxyoctanoic acid": "Fatty Acids", "2-hydroxy pelargonic acid": "Fatty Acids",
    "2-hydroxycaproicacid": "Fatty Acids", "Sebacicacid": "Fatty Acids",
    "Azelaicacid": "Fatty Acids", "Heptylic acid": "Fatty Acids",
}
_OA_TO_STEROID = {
    "[SThydrox]3alpha_6alpha-Dihydroxy-5beta-cholan-24-oicAcid": "Steroids & Bile Acids",
}
_OA_TO_XENO = {
    "paracetamol sulfate": "Drugs & Xenobiotics", "2-methoxyacetaminophen sulfate": "Drugs & Xenobiotics",
    "eugenol sulfate": "Drugs & Xenobiotics", "guaiacol sulfate": "Drugs & Xenobiotics",
    "tyramine sulfate": "Drugs & Xenobiotics", "Indoxylsulfate": "Drugs & Xenobiotics",
    "8-Anilinonaphthalene-1-sulfonic acid": "Drugs & Xenobiotics",
    "p-Toluenesulfonic acid": "Drugs & Xenobiotics", "Trifluoroacetic acid": "Drugs & Xenobiotics",
    "Homovanillicacidsulfate": "Drugs & Xenobiotics", "4-vinylguaiacol sulfate": "Drugs & Xenobiotics",
    "4-vinylphenol sulfate": "Drugs & Xenobiotics", "Caffeic acid 3-sulfate": "Drugs & Xenobiotics",
    "Vanillic acid 4-sulfate": "Drugs & Xenobiotics", "4-phenolsulfonic acid": "Drugs & Xenobiotics",
    "Ascorbicacid-2-sulfate": "Drugs & Xenobiotics", "Diethyl sulfate": "Drugs & Xenobiotics",
}
_AA_TO_OTHER = {
    "D-(+)-Glucose": "Sugars & Polyols", "D-(-)-Mannitol": "Sugars & Polyols",
    "D-Arabitol": "Sugars & Polyols", "Erythritol": "Sugars & Polyols",
    "β-D-Mannopyranose": "Sugars & Polyols", "1,5-Anhydro-D-glucitol": "Sugars & Polyols",
    "Hept-2-ulose": "Sugars & Polyols", "1D-chiro-inositol": "Sugars & Polyols",
    "6-Deoxy-2,3-di-O-methyl-D-allopyranose": "Sugars & Polyols",
    "6-Deoxy-3-O-methyl-β-D-galactopyranose": "Sugars & Polyols",
    "D-Arabinonate": "Organic Acids", "4-(Hydroxymethyl)-1,2,5-pentanetriol": "Sugars & Polyols",
    "5D-5-O-Methyl-2_3_5/4_6-pentahydroxycyclohexanone": "Sugars & Polyols",
    "D-(-)-Erythrose;L-Threose": "Sugars & Polyols", "2-Carboxy-D-arabinitol": "Sugars & Polyols",
    "Adenine": "Nucleotides & Bases", "1-Methyladenosine": "Nucleotides & Bases",
    "1-Methylguanine": "Nucleotides & Bases", "1-Methylxanthine": "Nucleotides & Bases",
    "Uracil;2_4-Dihydroxypyrimidine": "Nucleotides & Bases", "Uridine": "Nucleotides & Bases",
    "Xanthine": "Nucleotides & Bases", "Dihydrouridine": "Nucleotides & Bases",
    "cIMP": "Nucleotides & Bases", "Dihydrothymine": "Nucleotides & Bases",
    "5-Acetylamino-6-amino-3-methyluracil": "Nucleotides & Bases",
    "5-Hydroxyisourate": "Nucleotides & Bases",
    "1-Pyrroline-4-hydroxy-2-carboxylate": "Nucleotides & Bases",
    "Skatole": "Benzenoids", "Indoline": "Benzenoids",
    "N-oleoyl L-serine": "Fatty Amides", "N-palmitoyl serine": "Fatty Amides",
    "N-stearoyl serine": "Fatty Amides", "(+)-exo-5-Hydroxycamphor": "Terpenoids",
    "Dipropylcyclopentylxanthine": "Drugs & Xenobiotics",
}
_FA_TO_LYSO = {
    "1-linoleoyl-sn-glycero-3-phosphoethanolamine": "Lysophospholipids",
    "2-arachidonyl-sn-glycero-3-phosphoethanolamine": "Lysophospholipids",
    "sn-glycero-3-Phosphoethanolamine": "Lysophospholipids",
}
_CL_TO_OTHER = {
    "[FAamino_oxo(6:0/2:0)]2-amino-3-oxo-hexanedioicacid": "Organic Acids",
    "(4S)-4-[(2E)-2-Octenoyloxy]-4-(trimethylammonio)butanoate": "Acylcarnitines",
    "Triacetin": "Organic Acids",
    "DIACETIN MONOPROPANOATE": "Organic Acids",
}
_PL_TO_LYSO = {
    "1-Linoleoylglycerophosphocholine": "Lysophospholipids",
    "1-Oleoylglycerophosphocholine": "Lysophospholipids",
    "1-Palmitoylglycerophosphocholine": "Lysophospholipids",
    "PA(18:0/0:0)": "Lysophospholipids",
    "PE(16:0/0:0)": "Lysophospholipids",
    "PE(17:0/0:0)": "Lysophospholipids",
    "PE(18:0/0:0)": "Lysophospholipids",
    "PE(18:1/0:0)": "Lysophospholipids",
    "PE(18:2/0:0)": "Lysophospholipids",
    "PE(20:4/0:0)": "Lysophospholipids",
    "PE(22:6/0:0)": "Lysophospholipids",
    "PE(O-16:0/0:0)": "Lysophospholipids",
    "PE(O-18:0/0:0)": "Lysophospholipids",
    "PE(O-18:1/0:0)": "Lysophospholipids",
    "PE(P-18:0/0:0)": "Lysophospholipids",
}
ANNOTATION_CORRECTIONS = {
    "HILIC": {**_OA_TO_FA, **_OA_TO_STEROID, **_OA_TO_XENO, **_AA_TO_OTHER,
              **_FA_TO_LYSO, **_CL_TO_OTHER, **_PL_TO_LYSO},
    "RP": {},
}


# Helpers

def _find_col(df, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    norm_map = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in cols}
    for c in candidates:
        key = re.sub(r"\s+", " ", str(c)).strip().lower()
        if key in norm_map:
            return norm_map[key]
    for c in cols:
        if "compound" in str(c).lower() and "name" in str(c).lower():
            return c
    return None


def load_curation(curation_path) -> dict:
    """Manually curated classes for 'Other/Unknown' annotated features, per mode."""
    try:
        df = pd.read_excel(curation_path, sheet_name="Manual_Curation")
    except FileNotFoundError:
        print(f"Curation file not found: {curation_path}")
        return {"HILIC": {}, "RP": {}}
    name_col  = _find_col(df, ["Compound Name"])
    class_col = _find_col(df, ["Suggested Class ▼ (fill in)", "Suggested Class (fill in)", "Suggested Class"])
    mode_col  = _find_col(df, ["Mode(s)", "Modes", "Mode"])
    if name_col is None or class_col is None or mode_col is None:
        raise KeyError(f"Could not find required curation columns. Found: {list(df.columns)}")
    result = {}
    for mode in ["HILIC", "RP"]:
        sub = df[(df[mode_col] == mode) & (df[class_col] != "Cannot Classify")].copy()
        sub[name_col] = sub[name_col].astype(str).str.strip()
        result[mode] = dict(zip(sub[name_col], sub[class_col]))
    print(f"Curation loaded: HILIC={len(result['HILIC'])}, RP={len(result['RP'])}")
    return result


def apply_merges(class_series: pd.Series, mode: str) -> pd.Series:
    merge_map = MERGE_MAPS.get(mode, {})
    counts_before = class_series.value_counts()
    print(f"\nClass merging ({mode}):")
    for original, target in merge_map.items():
        n = counts_before.get(original, 0)
        if n > 0:
            print(f"  {original:<30} {n:>4} -> {target}")
    return class_series.map(lambda x: merge_map.get(x, x))


def apply_annotation_corrections(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    corrections = ANNOTATION_CORRECTIONS.get(mode, {})
    if not corrections:
        return df
    n = 0
    for idx, row in df.iterrows():
        base = strip_dup_suffix(str(row["FeatureID"]))
        if base in corrections and row["RuleClass"] != corrections[base]:
            df.at[idx, "RuleClass"] = corrections[base]
            n += 1
    print(f"\nAnnotation corrections ({mode}): {n} features corrected")
    return df


def build_feature_table(sheet: pd.DataFrame, mode: str, curation_labels: dict) -> pd.DataFrame:
    """
    Feature table (FeatureID, MZ, RT, Polarity, MassDefect, IsAnnotated, RuleClass)
    from a '<mode>_gap' sheet of the processed workbook.
    """
    mz_row = rt_row = None
    for i in range(min(8, len(sheet))):
        cell = str(sheet.iloc[i, 2]).lower()
        if "m/z" in cell and mz_row is None:
            mz_row = i
        elif ("rt" in cell or "retention" in cell) and rt_row is None:
            rt_row = i
        if mz_row is not None and rt_row is not None:
            break
    if mz_row is None or rt_row is None:
        raise ValueError("Could not locate 'm/z' and 'RT' header rows.")

    import numpy as np
    mz_strings  = sheet.iloc[mz_row, 3:].astype(str).values
    rt_vals     = pd.to_numeric(pd.Series(sheet.iloc[rt_row, 3:].values), errors="coerce").values
    feature_ids = sheet.columns.tolist()[3:]

    mz_vals, polarities = [], []
    for s in mz_strings:
        try:
            mz_vals.append(float(s.split("_")[-1]))
            polarities.append(1 if "+" in s else 0)
        except Exception:
            mz_vals.append(np.nan)
            polarities.append(0)

    df = pd.DataFrame({"FeatureID": [str(f) for f in feature_ids],
                       "MZ": mz_vals, "RT": rt_vals, "Polarity": polarities})
    df["MassDefect"] = df["MZ"] - np.floor(df["MZ"].fillna(0))
    df["IsAnnotated"] = ~df["FeatureID"].apply(lambda x: x.replace(".", "", 1).isdigit())

    manual = (curation_labels or {}).get(mode, {})
    df["RuleClass"] = df["FeatureID"].apply(
        lambda fid: manual.get(strip_dup_suffix(str(fid)), classify_metabolite(str(fid))))
    return apply_annotation_corrections(df, mode)
