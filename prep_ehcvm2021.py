"""Préparation du fichier d'estimation EASI à partir de l'EHCVM 2021-2022 (Côte d'Ivoire), ménages urbains.

Utilisation : python3 prep_ehcvm2021.py <dossier_csv_ehcvm> <fichier_sortie.csv> [avec_repas]

Étapes
1. Dépenses annuelles par produit (ehcvm_conso, tous modes d'acquisition) regroupées en cinq groupes,
   avec la même nomenclature que l'ENV 2008. Par défaut, alimentation consommée à domicile (codes 1-177) ;
   option « avec_repas » : repas et boissons pris hors du ménage (codes 191-197) ajoutés aux « autres aliments ».
2. Valeurs unitaires (FCFA/kg) tirées du dernier achat (s07b : valeur / quantité), quantités converties
   en kg avec la table officielle des unités non standard (ehcvm_nsu, médiane par produit-unité-taille).
   Valeurs hors de l'intervalle [1er ; 99e centile] du produit écartées.
3. Prix composite de groupe par ménage = moyenne géométrique des valeurs unitaires pondérée par la part
   de chaque produit dans la dépense du groupe (Urzúa, 2010). Prix manquant -> médiane de la grappe,
   puis de la région, puis nationale. Une variante « prix de grappe » (médiane de la grappe) est aussi
   produite.
4. Caractéristiques du ménage (ehcvm_welfare, ehcvm_individu) ; pauvre = consommation par tête sous le
   seuil national ; riche = dernier quintile national (pondéré) de consommation par tête.
"""
import sys
import numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
REPAS = len(sys.argv) > 3 and sys.argv[3] == "avec_repas"   # inclure les repas pris hors du ménage
rd = lambda f, **k: pd.read_csv(f"{D}/{f}.csv", low_memory=False, encoding="latin-1", **k)

# ---------- nomenclature : produit -> groupe (1 amidon, 2 huiles, 3 fruits et légumes, 4 viande et laitiers, 5 autres)
G = {}
def setg(codes, g):
    for c in codes: G[c] = g
setg(list(range(1, 23)) + [166, 168, 169, 174, 91, 109, 110, 111, 112] + list(range(123, 134)), 1)
setg([61, 62, 63, 64, 67, 68, 69, 70] + list(range(113, 123)), 2)
setg(list(range(71, 91)) + list(range(92, 99)) + list(range(100, 108)) + [140, 141, 142, 154], 3)
setg(list(range(27, 36)) + list(range(37, 47)) + list(range(52, 61)) + [170, 171, 172, 173, 175], 4)
setg([23, 24, 25, 26, 36, 47, 48, 49, 50, 51, 99, 108] + list(range(134, 140)) + list(range(143, 154))
     + list(range(155, 166)) + [167, 176, 177] + (list(range(191, 198)) if REPAS else []), 5)

# ---------- 1. dépenses
c = rd("ehcvm_conso_civ2021")
c = c[c.milieu == 1]
c = c[c.codpr.isin(G.keys())].copy()
c["g"] = c.codpr.map(G)
dep = c.groupby(["hhid", "g"]).depan.sum().unstack(fill_value=0.0)
dep.columns = [f"dep_g{g}" for g in dep.columns]
dep["x"] = dep.sum(axis=1)
dep = dep[dep.x > 0]
for g in range(1, 6):
    dep[f"budget_g{g}"] = dep[f"dep_g{g}"] / dep["x"]
depi = c.groupby(["hhid", "codpr"]).depan.sum()      # item expenditure, for within-group weights

# ---------- 2. valeurs unitaires
n = rd("ehcvm_nsu_civ2021")
conv = n.groupby(["produitID", "uniteID", "tailleID"]).poids.median()
conv2 = n.groupby(["produitID", "uniteID"]).poids.median()
b = rd("s07b_me_civ2021")
b = b[(b.s07bq02 == 1) & b.s07bq07a.gt(0) & b.s07bq08.gt(0)].copy()
b["s07bq07c"] = b.s07bq07c.fillna(0)
k3 = pd.MultiIndex.from_arrays([b.s07bq01, b.s07bq07b, b.s07bq07c])
g3 = conv.reindex(k3).to_numpy()
g2 = conv2.reindex(pd.MultiIndex.from_arrays([b.s07bq01, b.s07bq07b])).to_numpy()
b["grams"] = np.where(np.isnan(g3), g2, g3)
b["kg"] = b.s07bq07a * b.grams / 1000
b = b[b.kg > 0]
b["uv"] = b.s07bq08 / b.kg
b["hhid"] = b.grappe * 100 + b.menage
q = b.groupby("s07bq01").uv.quantile([0.01, 0.99]).unstack()
b = b.join(q, on="s07bq01")
b = b[(b.uv >= b[0.01]) & (b.uv <= b[0.99])]
b = b[b.s07bq01.isin(G.keys())]
b["g"] = b.s07bq01.map(G)
b["w_item"] = depi.reindex(pd.MultiIndex.from_arrays([b.hhid, b.s07bq01])).to_numpy()
b = b[b.w_item > 0]
b["lw"] = np.log(b.uv) * b.w_item
agg = b.groupby(["hhid", "g"]).agg(lw=("lw", "sum"), w=("w_item", "sum"))
lp = (agg.lw / agg.w).unstack()
lp.columns = [f"prix{g}" for g in lp.columns]

# ---------- 4. caractéristiques
w = rd("ehcvm_welfare_civ2021")
ind = rd("ehcvm_individu_civ2021")
age65 = ind[ind.age >= 65].groupby("hhid").size()
w = w.set_index("hhid")
w["age65"] = age65.reindex(w.index).fillna(0)
# quintile national pondéré (poids individuels = hhweight * hhsize)
ww = (w.hhweight * w.hhsize).to_numpy(); order = np.argsort(w.pcexp.to_numpy())
cum = np.cumsum(ww[order]) / ww.sum()
q80 = w.pcexp.to_numpy()[order][np.searchsorted(cum, 0.8)]

d = dep.join(lp, how="left").join(w[["grappe", "region", "vague", "month", "hhweight", "hhsize", "hgender", "hage",
                                     "hreligion", "hethnie", "pcexp", "zref", "dali", "dtot", "age65"]], how="inner")
d = d[d.hhsize > 0]
# imputation : grappe -> région -> national ; variante prix de grappe
nimp = {}
miss_hh = d[[f"prix{g}" for g in range(1, 6)]].isna().any(axis=1)
any_miss = miss_hh.mean() * 100
for gname in [f"prix{g}" for g in range(1, 6)]:
    miss = d[gname].isna()
    nimp[gname] = round(miss.mean() * 100, 1)
    med_c = d.groupby("grappe")[gname].transform("median")
    med_r = d.groupby("region")[gname].transform("median")
    d[f"{gname}_grappe"] = med_c.fillna(med_r).fillna(d[gname].median())
    d[gname] = d[gname].fillna(med_c).fillna(med_r).fillna(d[gname].median())
print("prix imputés (%) :", nimp, "| ménages avec au moins un prix imputé :",
      round(any_miss, 1))

out = pd.DataFrame({
    "obs": d.index,
    **{f"budget_g{g}": d[f"budget_g{g}"] for g in range(1, 6)},
    **{f"prix{g}": d[f"prix{g}"] for g in range(1, 6)},
    "log_y": np.log(d.x),
    "tailmen": d.hhsize, "sexe": (d.hgender == 1).astype(int), "etnie": d.hethnie.fillna(-1).astype(int),
    "relig": d.hreligion.fillna(-1).astype(int), "agem40": d.hage - 40, "age65": d.age65,
    "pauvre": (d.pcexp < d.zref).astype(int), "riche": (d.pcexp >= q80).astype(int),
    "vague2": (d.vague == 2).astype(int), "prix_impute": miss_hh.astype(int), "grappe": d.grappe, "wgt": d.hhweight,
    **{f"prix{g}_grappe": d[f"prix{g}_grappe"] for g in range(1, 6)},
    "alim_share_dtot": d.dali / d.dtot,
})
out.to_csv(OUT, index=False)
print("ménages urbains :", len(out), "| pauvres :", int(out.pauvre.sum()), "| riches :", int(out.riche.sum()),
      "| grappes :", out.grappe.nunique())
print("parts moyennes :", out[[f"budget_g{g}" for g in range(1, 6)]].mean().round(3).to_dict())
