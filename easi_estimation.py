"""Système de demande EASI approximé (Lewbel et Pendakur, 2009) — ménages urbains de Côte d'Ivoire.

Article : « Élasticités de la demande alimentaire en milieu urbain ivoirien », Oupoh Oupoh Emmanuel.
Deux sources : ENV 2008 (INS) et EHCVM 2021-2022 (INS/ANStat, UEMOA, Banque mondiale).

Estimation par moindres carrés généralisés réalisables itérés (SUR), avec homogénéité, additivité
et symétrie de Slutsky imposées ; paramètres de prix et de dépense différenciés selon le niveau de vie ;
écarts-types des élasticités par bootstrap sur les ménages (graine 2014).

Utilisation :
    ENV 2008  : python3 easi_estimation.py env2008 <fichier_pauvre.txt> <fichier_riche.txt> <dossier_sortie> [variante] [nb_bootstrap]
    EHCVM 2021: python3 easi_estimation.py csv <fichier_prepare.csv> <dossier_sortie> [variante] [nb_bootstrap]
    variante : main (défaut) | drop_imputed | weighted | prix_grappe (EHCVM seulement) ; nb_bootstrap : 199 par défaut
Pour l'ENV 2008, les deux fichiers d'entrée sont les fichiers d'estimation tabulés (identiques sauf la colonne
statp, qui marque les ménages pauvres dans le premier et les ménages riches dans le second). Pour l'EHCVM,
le fichier est produit par prep_ehcvm2021.py.

Sorties : results_<variante>.json (tous les paramètres, tests et élasticités), tableau1/2/3_*.csv,
figure1_parts_*.csv et, pour la variante main, engel_rows.csv (données de la figure des courbes d'Engel).
Dépendances : numpy, pandas, scipy.
"""
import sys, json
import numpy as np, pandas as pd

MODE = sys.argv[1]
if MODE == "env2008":
    fp, fr, outdir = sys.argv[2], sys.argv[3], sys.argv[4]
    rest = sys.argv[5:]
    d = pd.read_csv(fp, sep="\t")
    d["pauvre"] = d["statp"].astype(float)
    d["riche"] = pd.read_csv(fr, sep="\t")["statp"].astype(float)
elif MODE == "csv":
    fcsv, outdir = sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]
    d = pd.read_csv(fcsv)
else:
    sys.exit("mode inconnu : env2008 ou csv")
variant = rest[0] if len(rest) > 0 else "main"
NBOOT = int(rest[1]) if len(rest) > 1 else 199
rng = np.random.default_rng(2014)
J = 5
S = [f"budget_g{i}" for i in range(1, 6)]
P = [f"prix{i}" for i in range(1, 6)]
LAB = ["Amidon", "Huiles", "Fruits et légumes", "Viande et laitiers", "Autres aliments"]

# ---------- data preparation ----------
if MODE == "env2008":
    # prix codés 0 ou hors [1er ; 99e centile] -> médiane de l'échantillon (pas d'identifiant de grappe)
    imp = pd.DataFrame(False, index=d.index, columns=P)
    for p in P:
        v = d[p].copy()
        nz = v[v != 0]
        lo, hi = nz.quantile(0.01), nz.quantile(0.99)
        bad = (v == 0) | (v < lo) | (v > hi)
        imp[p] = bad
        d[p] = v.where(~bad, nz[(nz >= lo) & (nz <= hi)].median())
    d["any_imp"] = imp.any(axis=1)
    prep_stats = {"n": int(len(d)), "imputed_pct": (imp.mean() * 100).round(1).to_dict(),
                  "any_imputed_pct": round(d["any_imp"].mean() * 100, 1)}
else:
    # EHCVM : prix déjà nettoyés et imputés à la médiane de grappe par prep_ehcvm2021.py
    d["any_imp"] = d["prix_impute"].astype(bool)
    prep_stats = {"n": int(len(d)), "any_imputed_pct": round(d["any_imp"].mean() * 100, 1)}
    if variant == "prix_grappe":
        for p in P:
            d[p] = d[f"{p}_grappe"]
    # catégories rares (< 30 ménages) de religion et d'ethnie regroupées
    for v in ["relig", "etnie"]:
        cnt = d[v].value_counts()
        d[v] = d[v].where(d[v].map(cnt) >= 30, -9)
if variant == "drop_imputed":
    d = d[~d["any_imp"]].reset_index(drop=True)
W = d["wgt"].to_numpy() / d["wgt"].mean() if variant == "weighted" else np.ones(len(d))

N = len(d)
w = d[S].to_numpy()
lp = d[P].to_numpy()
wbar = w.mean(axis=0)
ytil = d["log_y"].to_numpy() - lp @ wbar
y0 = np.median(ytil)
yc = ytil - y0
lpn = lp[:, :4] - lp[:, [4]]          # homogeneity: prices relative to good 5

tail_c = d["tailmen"].to_numpy() - 4.0
age10 = d["agem40"].to_numpy() / 10.0
zcont = {"tailmen": tail_c, "sexe": d["sexe"].to_numpy().astype(float), "agem40_10": age10,
         "age65": d["age65"].to_numpy().astype(float), "pauvre": d["pauvre"].to_numpy(),
         "riche": d["riche"].to_numpy()}
relig = pd.get_dummies(d["relig"].astype(int), prefix="relig", drop_first=False).astype(float)
etnie = pd.get_dummies(d["etnie"].astype(int), prefix="etnie", drop_first=False).astype(float)
relig = relig.drop(columns=relig.sum().idxmax()); etnie = etnie.drop(columns=etnie.sum().idxmax())
if "vague2" in d.columns:   # EHCVM : effet fixe de vague (saisonnalité, flambée des prix de 2022)
    etnie["vague2"] = d["vague2"].astype(float).to_numpy()

def unrestricted_X(yc, z, rel, eth):
    cols = [np.ones_like(yc)] + [yc ** r for r in range(1, 6)]
    names = ["cst"] + [f"y{r}" for r in range(1, 6)]
    for k, v in z.items():
        cols.append(v); names.append(k)
    for c in rel.columns: cols.append(rel[c].to_numpy()); names.append(c)
    for c in eth.columns: cols.append(eth[c].to_numpy()); names.append(c)
    for k, v in z.items():
        cols.append(v * yc); names.append(f"y*{k}")
    return np.column_stack(cols), names

Xu, unames = unrestricted_X(yc, zcont, relig, etnie)
Ku = Xu.shape[1]
pairs = [(k, l) for k in range(4) for l in range(k, 4)]
pz_names = ["base", "pauvre", "riche"]
pz_vals = [np.ones(N), zcont["pauvre"], zcont["riche"]]
Kp = len(pairs) * len(pz_vals)
K = 4 * Ku + Kp

def build_X(Xu, lpn, pzv):
    n = Xu.shape[0]
    X = np.zeros((4 * n, K))
    for j in range(4):
        X[j * n:(j + 1) * n, j * Ku:(j + 1) * Ku] = Xu
    c = 4 * Ku
    for t, zt in enumerate(pzv):
        for (k, l) in pairs:
            X[k * n:(k + 1) * n, c] += zt * lpn[:, l]
            if l != k:
                X[l * n:(l + 1) * n, c] += zt * lpn[:, k]
            c += 1
    return X

def fgls(X, Y, n, wt, iters=3):
    sw = np.sqrt(np.tile(wt, 4))
    Xw, Yw = X * sw[:, None], Y * sw
    b = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
    for _ in range(iters):
        e = (Y - X @ b).reshape(4, n)
        Sig = (e * wt) @ e.T / wt.sum()
        L = np.linalg.cholesky(np.linalg.inv(Sig)).T  # Sig^-1 = L'L
        Xt = np.einsum("ij,jnk->ink", L, Xw.reshape(4, n, -1)).reshape(4 * n, -1)
        Yt = (L @ Yw.reshape(4, n)).reshape(-1)
        XtX = Xt.T @ Xt
        b = np.linalg.solve(XtX, Xt.T @ Yt)
    V = np.linalg.inv(XtX)
    return b, V, Sig

def unpack(b):
    bu = b[:4 * Ku].reshape(4, Ku)
    A = []
    c = 4 * Ku
    for t in range(len(pz_vals)):
        M = np.zeros((4, 4))
        for (k, l) in pairs:
            M[k, l] = M[l, k] = b[c]; c += 1
        A.append(M)
    return bu, A

def full5(G4):
    G = np.zeros((5, 5)); G[:4, :4] = G4
    G[:4, 4] = -G4.sum(axis=1); G[4, :4] = -G4.sum(axis=0); G[4, 4] = G4.sum()
    return G

ref_groups = {"intermediaires": (zcont["pauvre"] == 0) & (zcont["riche"] == 0),
              "pauvres": zcont["pauvre"] == 1, "riches": zcont["riche"] == 1}

def elasticities(b, wdata, ycv, masks, zc):
    bu, A = unpack(b)
    out = {}
    for g, m in masks.items():
        wg = wdata[m].mean(axis=0)
        ym = np.median(ycv[m])
        zr = {"tailmen": np.median(zc["tailmen"][m]), "sexe": 1.0, "agem40_10": np.median(zc["agem40_10"][m]),
              "age65": 0.0, "pauvre": 1.0 if g == "pauvres" else 0.0, "riche": 1.0 if g == "riches" else 0.0}
        G4 = A[0] + zr["pauvre"] * A[1] + zr["riche"] * A[2]
        G = full5(G4)
        # dw/dy
        dwdy4 = np.zeros(4)
        for r in range(1, 6):
            dwdy4 += r * bu[:, unames.index(f"y{r}")] * ym ** (r - 1)
        for k, v in zr.items():
            dwdy4 += bu[:, unames.index(f"y*{k}")] * v
        dwdy = np.append(dwdy4, -dwdy4.sum())
        eta = 1 + dwdy / wg
        H = G / wg[:, None] + wg[None, :] - np.eye(5)
        M = H - np.outer(eta, wg)
        Sl = G + np.outer(wg, wg) - np.diag(wg)
        out[g] = {"w": wg, "G": G, "eta": eta, "H": H, "M": M, "maxeig": np.linalg.eigvalsh(Sl).max()}
    return out

Y = w[:, :4].T.reshape(-1)
X = build_X(Xu, lpn, pz_vals)
b, V, Sig = fgls(X, Y, N, W)
el = elasticities(b, w, yc, ref_groups, zcont)

# ---------- Wald tests on Engel curve shape ----------
def wald(idx):
    R = np.zeros((len(idx), K)); R[np.arange(len(idx)), idx] = 1
    rb = R @ b
    stat = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
    from math import exp
    return stat, len(idx)
from scipy import stats as st
tests = {}
for j in range(4):
    for name, terms in [("cubique_et_plus(y3-y5)", ["y3", "y4", "y5"]), ("non_lineaire(y2-y5)", ["y2", "y3", "y4", "y5"])]:
        idx = [j * Ku + unames.index(t) for t in terms]
        s, q = wald(idx)
        tests[f"{LAB[j]}|{name}"] = {"chi2": round(s, 2), "df": q, "p": float(st.chi2.sf(s, q))}
idx = [j * Ku + unames.index(t) for j in range(4) for t in ["y3", "y4", "y5"]]
s, q = wald(idx); tests["joint|y3-y5"] = {"chi2": round(s, 2), "df": q, "p": float(st.chi2.sf(s, q))}
# poverty interactions with prices jointly zero?
idx = list(range(4 * Ku + len(pairs), K)); s, q = wald(idx)
tests["prix x pauvrete (20 coef)"] = {"chi2": round(s, 2), "df": q, "p": float(st.chi2.sf(s, q))}

# ---------- concavity at household level ----------
bu, A = unpack(b)
fitted4 = (X @ b).reshape(4, N).T
fitted = np.column_stack([fitted4, 1 - fitted4.sum(axis=1)])
conc = 0
for i in range(N):
    G4 = A[0] + zcont["pauvre"][i] * A[1] + zcont["riche"][i] * A[2]
    wi = fitted[i]
    Sl = full5(G4) + np.outer(wi, wi) - np.diag(wi)
    conc += np.linalg.eigvalsh(Sl).max() <= 1e-10
conc_pct = conc / N * 100

# ---------- bootstrap ----------
boot = {g: {"eta": [], "Hown": [], "Mown": [], "H": [], "M": []} for g in ref_groups}
for rep in range(NBOOT):
    ii = rng.integers(0, N, N)
    Xub = Xu[ii]; lpnb = lpn[ii]; pzb = [v[ii] for v in pz_vals]
    Xb = build_X(Xub, lpnb, pzb)
    Yb = w[ii, :4].T.reshape(-1)
    try:
        bb, _, _ = fgls(Xb, Yb, N, W[ii], iters=2)
    except np.linalg.LinAlgError:
        continue
    masks_b = {g: m[ii] for g, m in ref_groups.items()}
    zcb = {k: v[ii] for k, v in zcont.items()}
    eb = elasticities(bb, w[ii], yc[ii], masks_b, zcb)
    for g in ref_groups:
        boot[g]["eta"].append(eb[g]["eta"]); boot[g]["H"].append(eb[g]["H"]); boot[g]["M"].append(eb[g]["M"])

res = {"variant": variant, "prep": prep_stats, "N_used": N, "wbar": wbar.round(4).tolist(), "tests": tests,
       "concavity_pct": round(conc_pct, 1), "nboot": NBOOT, "groups": {}}
for g in ref_groups:
    e = el[g]
    se = lambda key: np.std(np.array(boot[g][key]), axis=0, ddof=1) if len(boot[g][key]) > 1 else None
    res["groups"][g] = {
        "n": int(ref_groups[g].sum()), "w": e["w"].round(4).tolist(),
        "eta": e["eta"].round(3).tolist(), "eta_se": se("eta").round(3).tolist(),
        "G": e["G"].round(4).tolist(), "H": e["H"].round(3).tolist(), "H_se": se("H").round(3).tolist(),
        "M": e["M"].round(3).tolist(), "M_se": se("M").round(3).tolist(), "slutsky_maxeig": float(e["maxeig"])}

# demographic coefficients (level) with SE
coef = {}
sev = np.sqrt(np.diag(V))
for j in range(4):
    for nm in ["y1", "y2", "y3", "y4", "y5", "tailmen", "sexe", "agem40_10", "age65", "pauvre", "riche",
               "y*tailmen", "y*age65", "y*pauvre", "y*riche"]:
        k = j * Ku + unames.index(nm)
        coef[f"{LAB[j]}|{nm}"] = [round(float(b[k]), 5), round(float(sev[k]), 5)]
res["coef"] = coef
# price parameter tables
for t, nmz in enumerate(pz_names):
    res[f"A_{nmz}"] = unpack(b)[1][t].round(4).tolist()
    se_t = np.zeros((4, 4)); c = 4 * Ku + t * len(pairs)
    for (k, l) in pairs:
        se_t[k, l] = se_t[l, k] = sev[c]; c += 1
    res[f"A_{nmz}_se"] = se_t.round(4).tolist()

# ---------- Engel curves: fitted vs observed by percentile of log x ----------
if variant == "main":
    lx = d["log_y"].to_numpy()
    pct = pd.qcut(lx, 20, labels=False)
    rows = []
    lpn_mean = lpn.mean(axis=0)
    for q in range(20):
        m = pct == q
        ycq = np.median(yc[m]); pv = float(np.round(zcont["pauvre"][m].mean())); rv = float(np.round(zcont["riche"][m].mean()))
        zr = {"tailmen": 0.0, "sexe": 1.0, "agem40_10": np.median(age10), "age65": 0.0, "pauvre": pv, "riche": rv}
        xrow = [1] + [ycq ** r for r in range(1, 6)] + list(zr.values()) + [0] * (relig.shape[1] + etnie.shape[1]) + [v * ycq for v in zr.values()]
        xrow = np.array(xrow, float)
        G4 = A[0] + pv * A[1] + rv * A[2]
        wf4 = bu @ xrow + G4 @ lpn_mean
        wf = np.append(wf4, 1 - wf4.sum())
        wo = w[m].mean(axis=0)
        wfit = fitted[m].mean(axis=0)
        for j in range(5):
            rows.append({"vingtile": q + 1, "groupe": LAB[j], "modele_ref": round(float(wf[j]), 4),
                         "ajuste": round(float(wfit[j]), 4), "observe": round(float(wo[j]), 4)})
    pd.DataFrame(rows).to_csv(f"{outdir}/engel_rows.csv", index=False)

# ---------- tables of the article (CSV) ----------
G3 = ["pauvres", "intermediaires", "riches"]
fmt = lambda v, s: f"{v:.3f} ({s:.3f})"
t1 = pd.DataFrame({"Groupe": LAB,
                   **{f"Marshallienne {g}": [fmt(res['groups'][g]['M'][i][i], res['groups'][g]['M_se'][i][i]) for i in range(5)] for g in G3},
                   **{f"Hicksienne {g}": [round(res['groups'][g]['H'][i][i], 3) for i in range(5)] for g in ["pauvres", "riches"]}})
t2 = pd.DataFrame({"Groupe": LAB, **{g: [fmt(res['groups'][g]['eta'][i], res['groups'][g]['eta_se'][i]) for i in range(5)] for g in G3}})
t3 = pd.DataFrame({"Demande de": LAB[1:], **{g: [fmt(res['groups'][g]['M'][i][0], res['groups'][g]['M_se'][i][0]) for i in range(1, 5)] for g in G3}})
t1.to_csv(f"{outdir}/tableau1_elasticites_prix_{variant}.csv", index=False)
t2.to_csv(f"{outdir}/tableau2_elasticites_depense_{variant}.csv", index=False)
t3.to_csv(f"{outdir}/tableau3_effet_prix_amidon_{variant}.csv", index=False)
# figure 1: weighted mean budget shares by living-standard group
wt_s = d["wgt"].to_numpy()
fig1 = []
for g, m in [("pauvres", ref_groups["pauvres"]), ("ensemble", np.ones(N, bool)), ("riches", ref_groups["riches"])]:
    sh = (w[m] * wt_s[m, None]).sum(axis=0) / wt_s[m].sum()
    fig1 += [{"menage": g, "groupe": LAB[j], "part": round(float(sh[j]), 4)} for j in range(5)]
pd.DataFrame(fig1).to_csv(f"{outdir}/figure1_parts_{variant}.csv", index=False)

with open(f"{outdir}/results_{variant}.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["variant", "N_used", "prep", "concavity_pct"]}, ensure_ascii=False))
np.set_printoptions(precision=3, suppress=True)
for g in ref_groups:
    e = res["groups"][g]
    print("==", g, "n", e["n"], "w", e["w"])
    print(" eta", e["eta"], "+-", e["eta_se"])
    print(" Marsh own", np.diag(np.array(e["M"])), "+-", np.diag(np.array(e["M_se"])))
    print(" Hicks own", np.diag(np.array(e["H"])), "+-", np.diag(np.array(e["H_se"])))
    print(" Slutsky max eig", e["slutsky_maxeig"])
for k, v in tests.items(): print(k, v)
