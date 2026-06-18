"""Generate notebooks/xgb_eda2.ipynb — second EDA iteration."""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "id": uid(),
            "metadata": {}, "outputs": [], "source": source}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# XGB EDA 2 — Iteratie op basis van individuele feature diagnose

## Context

`xgb_eda.ipynb` liet features cumulatief toevoegen → ruis-features (magnitude stats, target encoding,
redshift transforms) annuleerden de winst van goede features.

### Diagnose: individueel getest (3-fold, 300 trees)

| Groep | Delta |
|---|---|
| Galactische coördinaten | **+0.00096** ✅ |
| Kleurindices (alle 10) | **+0.00039** ✅ |
| SED slope/curvature | **+0.00020** ✅ |
| Redshift transforms | -0.00005 ❌ |
| Magnitude stats | -0.00039 ❌ |
| Target encoding | -0.00014 ❌ |

### Strategie dit notebook
1. Individueel testen (geen cumulatie tot het eind)
2. Dieper in galactische features (fysisch sterkste signaal)
3. Selectief in kleurindices (niet alle 10, maar welke helpen echt?)
4. Nieuwe kandidaten: flux-ratios, stellaire locus, galactische × kleur interacties
5. Minimale feature-set: alleen wat het CV-verschil significant verbetert
6. Volledige 5-fold validatie op de finale set
"""))

# ── Section 0: Setup ──────────────────────────────────────────────────────────
cells.append(md("## 0 – Setup"))

cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 4)
sns.set_style('whitegrid')

CLASS_MAP   = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
CLASS_NAMES = ['GALAXY', 'QSO', 'STAR']
TARGET = 'class'
bands  = ['u', 'g', 'r', 'i', 'z']
"""))

cells.append(code("""\
train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')
train.columns = train.columns.str.lower()
test.columns  = test.columns.str.lower()

train[TARGET] = train[TARGET].map(CLASS_MAP)

CAT_COLS = ['spectral_type', 'galaxy_population']
train_enc = pd.get_dummies(train, columns=CAT_COLS, drop_first=True)
test_enc  = pd.get_dummies(test,  columns=CAT_COLS, drop_first=True)

BASE_FEATURES = [c for c in train_enc.columns if c not in ['id', TARGET]]
print(f'Base features ({len(BASE_FEATURES)}): {BASE_FEATURES}')
"""))

cells.append(code("""\
import torch
device = 'gpu' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

QUICK_PARAMS = dict(
    objective='multi:softprob', num_class=3,
    tree_method='hist', device=device,
    n_estimators=300, learning_rate=0.1,
    max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, random_state=42, enable_categorical=True,
)

def quick_cv(extra_features, df=None, n_folds=3):
    \"\"\"Test BASE_FEATURES + extra_features with quick 3-fold CV.\"\"\"
    if df is None:
        df = train_enc
    feats = BASE_FEATURES + extra_features
    y = df[TARGET]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros((len(y), 3))
    for tr_idx, val_idx in skf.split(df[feats], y):
        Xtr, Xvl = df[feats].iloc[tr_idx], df[feats].iloc[val_idx]
        ytr = y.iloc[tr_idx]
        w = compute_sample_weight('balanced', ytr)
        m = XGBClassifier(**QUICK_PARAMS)
        m.fit(Xtr, ytr, sample_weight=w, verbose=False)
        oof[val_idx] = m.predict_proba(Xvl)
    return balanced_accuracy_score(y, np.argmax(oof, axis=1))

BASELINE = quick_cv([])
print(f'Baseline quick CV: {BASELINE:.5f}')
results = {}
"""))

# ── Section 1: Galactic Coordinates Deep Dive ─────────────────────────────────
cells.append(md("""\
## 1 – Galactische Coördinaten (Beste Feature, +0.00096)

Galactische breedte `b` meet de hoekafstand tot het vlak van de Melkweg.
- **STARs** concentreren langs het galactische vlak (|b| < 30°)
- **GALAXYs & QSOs** worden vermeden bij lage |b| (extinctiezone + sterrenvelden)

We testen varianten om het beste signaal te isoleren.
"""))

cells.append(code("""\
def radec_to_galactic(ra_deg, dec_deg):
    ra, dec     = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    ra_gp       = np.deg2rad(192.8595)
    dec_gp      = np.deg2rad(27.1284)
    l_0         = np.deg2rad(122.9320)
    sin_b = (np.sin(dec) * np.sin(dec_gp) +
             np.cos(dec) * np.cos(dec_gp) * np.cos(ra - ra_gp))
    b = np.arcsin(np.clip(sin_b, -1, 1))
    x = np.cos(dec) * np.sin(ra - ra_gp)
    y = (np.sin(dec) * np.cos(dec_gp) -
         np.cos(dec) * np.sin(dec_gp) * np.cos(ra - ra_gp))
    l = l_0 - np.arctan2(x, y)
    return np.rad2deg(l % (2 * np.pi)), np.rad2deg(b)

l_tr, b_tr = radec_to_galactic(train['alpha'].values, train['delta'].values)
l_te, b_te = radec_to_galactic(test['alpha'].values,  test['delta'].values)

# Galactic b distribution per class
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, vals) in zip(axes, [
    ('galactic_b', b_tr),
    ('|galactic_b|', np.abs(b_tr)),
    ('sin(galactic_b)', np.sin(np.deg2rad(b_tr))),
]):
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = train[TARGET] == cls_id
        ax.hist(vals[mask], bins=60, alpha=0.5, label=cls_name, density=True)
    ax.set_title(title)
    ax.legend(fontsize=8)
plt.suptitle('Galactic Latitude Variants per Class')
plt.tight_layout()
plt.show()

# Spearman correlation vs target
for name, vals in [('galactic_b', b_tr), ('abs_galactic_b', np.abs(b_tr)),
                   ('sin_b', np.sin(np.deg2rad(b_tr))), ('galactic_l', l_tr)]:
    r, p = spearmanr(vals, train[TARGET].values)
    print(f'  {name:25s}: Spearman |r| = {abs(r):.4f}  p={p:.2e}')
"""))

cells.append(code("""\
# Test galactic variants individually
galactic_variants = {}

# 1a: raw galactic_b + galactic_l
train_enc['galactic_l'] = l_tr
train_enc['galactic_b'] = b_tr
s = quick_cv(['galactic_l', 'galactic_b'])
galactic_variants['l+b'] = s
print(f'galactic_l + galactic_b     : {s:.5f}  ({s-BASELINE:+.5f})')

# 1b: |galactic_b| alone (STAR signal is symmetric above/below plane)
train_enc['abs_galactic_b'] = np.abs(b_tr)
s = quick_cv(['abs_galactic_b'])
galactic_variants['|b|'] = s
print(f'|galactic_b|                : {s:.5f}  ({s-BASELINE:+.5f})')

# 1c: sin(b) — smooth and bounded
train_enc['sin_galactic_b'] = np.sin(np.deg2rad(b_tr))
s = quick_cv(['sin_galactic_b'])
galactic_variants['sin(b)'] = s
print(f'sin(galactic_b)             : {s:.5f}  ({s-BASELINE:+.5f})')

# 1d: b + |b| together
s = quick_cv(['galactic_b', 'abs_galactic_b'])
galactic_variants['b+|b|'] = s
print(f'galactic_b + |galactic_b|   : {s:.5f}  ({s-BASELINE:+.5f})')

# 1e: b + l + |b|
s = quick_cv(['galactic_l', 'galactic_b', 'abs_galactic_b'])
galactic_variants['l+b+|b|'] = s
print(f'l + b + |b|                 : {s:.5f}  ({s-BASELINE:+.5f})')

# 1f: |b| + near-plane flag
train_enc['is_galactic_plane'] = (np.abs(b_tr) < 20).astype(int)
s = quick_cv(['abs_galactic_b', 'is_galactic_plane'])
galactic_variants['|b|+plane_flag'] = s
print(f'|b| + is_galactic_plane<20° : {s:.5f}  ({s-BASELINE:+.5f})')

best_galactic = max(galactic_variants, key=galactic_variants.get)
best_galactic_score = galactic_variants[best_galactic]
print(f'\\nBeste galactic variant: {best_galactic}  ({best_galactic_score:.5f})')
"""))

cells.append(code("""\
# Store test set galactic features
test_enc['galactic_l']        = l_te
test_enc['galactic_b']        = b_te
test_enc['abs_galactic_b']    = np.abs(b_te)
test_enc['sin_galactic_b']    = np.sin(np.deg2rad(b_te))
test_enc['is_galactic_plane'] = (np.abs(b_te) < 20).astype(int)

CONFIRMED_GALACTIC = ['galactic_b', 'abs_galactic_b', 'galactic_l', 'is_galactic_plane']
results['galactic'] = best_galactic_score
print(f'Galactic features opgeslagen. Beste delta: {best_galactic_score - BASELINE:+.5f}')
"""))

# ── Section 2: Color Indices – Which Ones Actually Help? ───────────────────────
cells.append(md("""\
## 2 – Kleurindices: Selectief i.p.v. Alle 10

Alle 10 kleurindices gaven +0.00039. Maar zijn ALLE 10 nuttig, of slechts enkele?
XGB kan kleurverschillen in 2 splits leren — expliciete features helpen alleen als ze
het model vrijspelen voor diepere interacties.

Spearman-rangcorrelatie met het target geeft een snelle screening.
"""))

cells.append(code("""\
# Spearman screening of all 10 color indices
color_spearman = {}
for i, b1 in enumerate(bands):
    for j, b2 in enumerate(bands):
        if i < j:
            col = f'{b1}_{b2}'
            diff = train[b1].values - train[b2].values
            r, _ = spearmanr(diff, train[TARGET].values)
            color_spearman[col] = abs(r)

color_rank = sorted(color_spearman.items(), key=lambda x: -x[1])
print('Kleurindices gerangschikt op |Spearman r| met target:')
for col, r in color_rank:
    print(f'  {col:6s}: {r:.4f}')
"""))

cells.append(code("""\
# Add all color indices
color_all = []
for i, b1 in enumerate(bands):
    for j, b2 in enumerate(bands):
        if i < j:
            col = f'{b1}_{b2}'
            train_enc[col] = train[b1] - train[b2]
            test_enc[col]  = test[b1]  - test[b2]
            color_all.append(col)

# Test sets: all 10, top-5, top-3, top-1
top1   = [color_rank[0][0]]
top3   = [c for c,_ in color_rank[:3]]
top5   = [c for c,_ in color_rank[:5]]

s_all  = quick_cv(color_all)
s_top5 = quick_cv(top5)
s_top3 = quick_cv(top3)
s_top1 = quick_cv(top1)

print(f'Alle 10 kleurindices: {s_all:.5f}  ({s_all-BASELINE:+.5f})')
print(f'Top-5 kleurindices  : {s_top5:.5f}  ({s_top5-BASELINE:+.5f})')
print(f'Top-3 kleurindices  : {s_top3:.5f}  ({s_top3-BASELINE:+.5f})')
print(f'Top-1 kleurindex    : {s_top1:.5f}  ({s_top1-BASELINE:+.5f})')
print(f'Top-3: {top3}')

best_color_feats = top5 if s_top5 >= s_all else color_all
results['color'] = max(s_all, s_top5, s_top3)
"""))

cells.append(code("""\
# Per-index individual test: which single color index helps most?
print('Individuele kleurindex scores:')
individual_color_scores = {}
for col in color_all:
    s = quick_cv([col])
    individual_color_scores[col] = s
    print(f'  {col:6s}: {s:.5f}  ({s-BASELINE:+.5f})')
"""))

# ── Section 3: Galactic × Color Interactions ──────────────────────────────────
cells.append(md("""\
## 3 – Galactische × Kleur Interacties

STARs in het galactische vlak hebben andere kleuren dan GALAXYs op dezelfde locatie.
XGB kan deze interactie niet in één split leren — een product-feature geeft een directe numerieke koppeling.
"""))

cells.append(code("""\
# Test: galactic_b × top color indices
top_color = color_rank[0][0]   # sterkste kleurindex
second_color = color_rank[1][0]

train_enc[f'gal_b_x_{top_color}']    = b_tr * train_enc[top_color]
train_enc[f'gal_b_x_{second_color}'] = b_tr * train_enc[second_color]
test_enc[f'gal_b_x_{top_color}']     = b_te * test_enc[top_color]
test_enc[f'gal_b_x_{second_color}']  = b_te * test_enc[second_color]

interaction_feats = [f'gal_b_x_{top_color}', f'gal_b_x_{second_color}']

# Test interacties alleen
s_inter = quick_cv(interaction_feats)
print(f'Galactic×color (alleen)          : {s_inter:.5f}  ({s_inter-BASELINE:+.5f})')

# Test galactisch + kleur + interactie
best_galactic_feats = ['galactic_b', 'abs_galactic_b', 'galactic_l']
combined = best_galactic_feats + best_color_feats + interaction_feats
s_combined = quick_cv(combined)
print(f'Galactic + Color + Interactie    : {s_combined:.5f}  ({s_combined-BASELINE:+.5f})')

results['galactic+color+interaction'] = s_combined
"""))

# ── Section 4: New Candidates ─────────────────────────────────────────────────
cells.append(md("""\
## 4 – Nieuwe Kandidaat Features

Op basis van de diagnose zijn redshift transforms en magnitude stats niet nuttig als groep.
Maar sommige SPECIFIEKE sub-features kunnen nog helpen.
Testen we individueel, zonder de slechte sub-features mee te nemen.
"""))

cells.append(md("### 4.1 – Flux Ratios (niet-lineaire band-vergelijking)"))

cells.append(code("""\
# Magnitudes → flux (f = 10^(-0.4*m), schaalinvariant)
# Flux ratio = 10^(-0.4*(m1-m2)) — dit is een NIET-LINEAIRE transformatie van kleurindex
# XGB kan dit NIET leren met 2 simpele splits — het vereist veel splits om te benaderen

eps = 1e-10
for i, b1 in enumerate(bands):
    for j, b2 in enumerate(bands):
        if i < j:
            col = f'flux_{b1}_{b2}'
            train_enc[col] = 10 ** (-0.4 * (train[b1] - train[b2]))
            test_enc[col]  = 10 ** (-0.4 * (test[b1]  - test[b2]))

flux_cols = [f'flux_{b1}_{b2}' for i,b1 in enumerate(bands)
             for j,b2 in enumerate(bands) if i < j]

# Clip extremes (magnitude differences can blow up in flux space)
for col in flux_cols:
    lo = train_enc[col].quantile(0.001)
    hi = train_enc[col].quantile(0.999)
    train_enc[col] = train_enc[col].clip(lo, hi)

# Spearman screening of flux ratios
flux_spearman = {}
for col in flux_cols:
    r, _ = spearmanr(train_enc[col].values, train[TARGET].values)
    flux_spearman[col] = abs(r)

flux_rank = sorted(flux_spearman.items(), key=lambda x: -x[1])
print('Top flux ratios per |Spearman r|:')
for col, r in flux_rank[:5]:
    print(f'  {col:15s}: {r:.4f}')

top_flux = [c for c,_ in flux_rank[:5]]
s_flux = quick_cv(top_flux)
print(f'\\nTop-5 flux ratios: {s_flux:.5f}  ({s_flux-BASELINE:+.5f})')
results['flux_ratios'] = s_flux
"""))

cells.append(md("### 4.2 – Stellaire Locus: Afstand tot de Hoofdrij in Kleurruimte"))

cells.append(code("""\
# In de (g-r, r-i) kleurruimte volgen sterren een smalle band (de 'stellar locus').
# Galactische sterren zitten OP de locus; QSOs en verre sterrenstelsels ERNAAST.
# We berekenen de loodlijnafstand tot de gemiddelde sterrenlocus.

# Schat de locus: neem de mediaan per klasse STAR
star_mask = train[TARGET] == 2
gr_stars  = (train.loc[star_mask, 'g'] - train.loc[star_mask, 'r']).median()
ri_stars  = (train.loc[star_mask, 'r'] - train.loc[star_mask, 'i']).median()
iz_stars  = (train.loc[star_mask, 'i'] - train.loc[star_mask, 'z']).median()

# Locus-richting schatten via PCA op STAR kleurpunten
from sklearn.decomposition import PCA
star_colors = train.loc[star_mask, ['g','r','i','z']].copy()
star_gr = (star_colors['g'] - star_colors['r']).values
star_ri = (star_colors['r'] - star_colors['i']).values
star_iz = (star_colors['i'] - star_colors['z']).values

pca = PCA(n_components=1)
star_pts = np.column_stack([star_gr, star_ri, star_iz])
pca.fit(star_pts)

# Project ALL objects onto the locus direction
all_gr = (train['g'] - train['r']).values
all_ri = (train['r'] - train['i']).values
all_iz = (train['i'] - train['z']).values

all_pts = np.column_stack([all_gr, all_ri, all_iz])
locus_proj    = pca.transform(all_pts)[:, 0]          # along-locus coordinate
locus_deviation = np.linalg.norm(
    all_pts - pca.inverse_transform(locus_proj.reshape(-1,1)), axis=1
)                                                       # perpendicular distance

train_enc['locus_proj']      = locus_proj
train_enc['locus_deviation'] = locus_deviation

# Test set
te_gr = (test['g'] - test['r']).values
te_ri = (test['r'] - test['i']).values
te_iz = (test['i'] - test['z']).values
te_pts = np.column_stack([te_gr, te_ri, te_iz])
te_locus_proj = pca.transform(te_pts)[:, 0]
te_locus_dev  = np.linalg.norm(te_pts - pca.inverse_transform(te_locus_proj.reshape(-1,1)), axis=1)
test_enc['locus_proj']      = te_locus_proj
test_enc['locus_deviation'] = te_locus_dev

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cls_id, cls_name in enumerate(CLASS_NAMES):
    mask = train[TARGET] == cls_id
    axes[0].hist(locus_proj[mask], bins=60, alpha=0.5, label=cls_name, density=True)
    axes[1].hist(locus_deviation[mask], bins=60, alpha=0.5, label=cls_name, density=True)
axes[0].set_title('Along-locus projection')
axes[1].set_title('Perpendicular deviation from stellar locus')
axes[0].legend()
plt.tight_layout()
plt.show()

s_locus = quick_cv(['locus_proj', 'locus_deviation'])
print(f'Stellaire locus features: {s_locus:.5f}  ({s_locus-BASELINE:+.5f})')
results['locus'] = s_locus
"""))

cells.append(md("### 4.3 – is_negative_redshift (Enige Redshift-variant die helpt?)"))

cells.append(code("""\
# De groep 'redshift transforms' als geheel hurts (-0.00005)
# Maar is_negative_redshift is een puur binaire vlag — wellicht isoleerbaar nuttig

train_enc['is_neg_z'] = (train['redshift'] < 0).astype(int)
test_enc['is_neg_z']  = (test['redshift'] < 0).astype(int)

s_neg = quick_cv(['is_neg_z'])
print(f'is_negative_redshift alleen : {s_neg:.5f}  ({s_neg-BASELINE:+.5f})')

# Redshift kwadraat (alleen)
train_enc['redshift_sq'] = train['redshift'] ** 2
test_enc['redshift_sq']  = test['redshift'] ** 2

s_rsq = quick_cv(['redshift_sq'])
print(f'redshift_sq alleen          : {s_rsq:.5f}  ({s_rsq-BASELINE:+.5f})')

# log1p_redshift (alleen, handig bij sterk scheve distributie)
train_enc['log1p_redshift'] = np.log1p(train['redshift'].clip(0))
test_enc['log1p_redshift']  = np.log1p(test['redshift'].clip(0))

s_logz = quick_cv(['log1p_redshift'])
print(f'log1p_redshift alleen       : {s_logz:.5f}  ({s_logz-BASELINE:+.5f})')

# redshift × galactic_b (sterren in galactische vlak bij lage redshift)
train_enc['z_x_gal_b'] = train['redshift'] * b_tr
test_enc['z_x_gal_b']  = test['redshift']  * b_te

s_zb = quick_cv(['z_x_gal_b'])
print(f'redshift × galactic_b alleen: {s_zb:.5f}  ({s_zb-BASELINE:+.5f})')
"""))

cells.append(md("### 4.4 – Galactic Extinction Proxy"))

cells.append(code("""\
# Galactische extinctieschatting (stofkolom is hoger bij laag |b|)
# Approximate: E(B-V) ∝ 1/sin(|b|)  (Cosec-wet)
# Dit verschilt per hemelgebied, maar geeft XGB een extra extinctiehoogte-proxy

train_enc['csc_galactic_b'] = 1.0 / (np.abs(np.sin(np.deg2rad(b_tr))) + 0.05)
test_enc['csc_galactic_b']  = 1.0 / (np.abs(np.sin(np.deg2rad(b_te)))  + 0.05)

# Clip extreme values at low galactic latitudes
clip_val = train_enc['csc_galactic_b'].quantile(0.999)
train_enc['csc_galactic_b'] = train_enc['csc_galactic_b'].clip(upper=clip_val)
test_enc['csc_galactic_b']  = test_enc['csc_galactic_b'].clip(upper=clip_val)

fig, ax = plt.subplots(figsize=(8, 3))
for cls_id, cls_name in enumerate(CLASS_NAMES):
    mask = train[TARGET] == cls_id
    ax.hist(train_enc.loc[mask, 'csc_galactic_b'], bins=60, alpha=0.5, label=cls_name, density=True)
ax.set_title('Extinction proxy (csc |galactic_b|) per class')
ax.legend()
plt.tight_layout()
plt.show()

s_csc = quick_cv(['csc_galactic_b'])
print(f'Extinctieschatting csc(b): {s_csc:.5f}  ({s_csc-BASELINE:+.5f})')
results['csc_b'] = s_csc
"""))

cells.append(md("### 4.5 – SED-features: Asymmetrie Rood vs Blauw"))

cells.append(code("""\
# SED slope was nuttig (+0.00020). Maar SED slope + curvature + res_std waren cumulatief.
# Test: is de ASYMMETRIE (blauw vs rood kant van SED) informatiever?

# Blauw-helft gemiddelde: u, g
# Rood-helft gemiddelde: r, i, z
train_enc['sed_blue_mean'] = train[['u','g']].mean(axis=1)
train_enc['sed_red_mean']  = train[['r','i','z']].mean(axis=1)
train_enc['sed_asym']      = train_enc['sed_blue_mean'] - train_enc['sed_red_mean']
test_enc['sed_blue_mean']  = test[['u','g']].mean(axis=1)
test_enc['sed_red_mean']   = test[['r','i','z']].mean(axis=1)
test_enc['sed_asym']       = test_enc['sed_blue_mean'] - test_enc['sed_red_mean']

# SED slope (vectorized)
WAVELENGTHS = np.array([354.3, 477.0, 623.1, 762.5, 913.4])
log_wl = np.log10(WAVELENGTHS)
x = log_wl - log_wl.mean()
mags_tr = train[bands].values
mags_te = test[bands].values

train_enc['sed_slope'] = (mags_tr @ x) / (x @ x)
test_enc['sed_slope']  = (mags_te @ x) / (x @ x)

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for cls_id, cls_name in enumerate(CLASS_NAMES):
    mask = train[TARGET] == cls_id
    axes[0].hist(train_enc.loc[mask,'sed_asym'], bins=60, alpha=0.5, label=cls_name, density=True)
    axes[1].hist(train_enc.loc[mask,'sed_slope'], bins=60, alpha=0.5, label=cls_name, density=True)
axes[0].set_title('SED asymmetrie (blauw - rood helft)')
axes[1].set_title('SED slope (lineaire fit)')
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

s_asym  = quick_cv(['sed_asym'])
s_slope = quick_cv(['sed_slope'])
s_both  = quick_cv(['sed_asym', 'sed_slope'])
print(f'sed_asym           : {s_asym:.5f}  ({s_asym-BASELINE:+.5f})')
print(f'sed_slope          : {s_slope:.5f}  ({s_slope-BASELINE:+.5f})')
print(f'sed_asym + slope   : {s_both:.5f}  ({s_both-BASELINE:+.5f})')
results['sed'] = max(s_asym, s_slope, s_both)
"""))

# ── Section 5: Ablation on All Candidates ────────────────────────────────────
cells.append(md("""\
## 5 – Greedy Feature Selectie

Test alle kandidaten samen, dan verwijder overbodige features.
Begin met de bevestigde winnaars en voeg toe wat statistisch helpt (delta > 0).
"""))

cells.append(code("""\
# Summary of all individual results
print('=== OVERZICHT INDIVIDUELE FEATURE-TESTS ===')
print(f'Baseline: {BASELINE:.5f}')
print()
for group, score in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {group:35s}: {score:.5f}  ({score-BASELINE:+.5f})')
"""))

cells.append(code("""\
# Start met bewezen best galactic set
current_feats = ['galactic_b', 'abs_galactic_b', 'galactic_l', 'is_galactic_plane']
current_score = quick_cv(current_feats)
print(f'Start (galactic): {current_score:.5f}')

# Candidates to try adding one by one
candidates = {
    'top5_color':    top5,
    'locus_features': ['locus_proj', 'locus_deviation'],
    'top_flux':      [c for c,_ in flux_rank[:3]],
    'sed_features':  ['sed_slope', 'sed_asym'],
    'is_neg_z':      ['is_neg_z'],
    'redshift_sq':   ['redshift_sq'],
    'csc_b':         ['csc_galactic_b'],
    'z_x_gal_b':     ['z_x_gal_b'],
    f'gal_b_x_color': interaction_feats,
}

greedy_results = []
for name, feats in candidates.items():
    test_set = current_feats + feats
    s = quick_cv(test_set)
    delta = s - current_score
    greedy_results.append((name, feats, s, delta))
    print(f'  + {name:30s}: {s:.5f}  ({delta:+.5f})')

# Sort by delta, accept positives
greedy_results.sort(key=lambda x: -x[3])
print()
for name, feats, score, delta in greedy_results:
    if delta > 0:
        print(f'ACCEPT {name}: delta={delta:+.5f}')
    else:
        print(f'REJECT {name}: delta={delta:+.5f}')
"""))

cells.append(code("""\
# Build the minimal best feature set greedily
BEST_FEATURES = ['galactic_b', 'abs_galactic_b', 'galactic_l', 'is_galactic_plane']
best_score = quick_cv(BEST_FEATURES)
print(f'Start: {best_score:.5f}  features={BEST_FEATURES}')

for name, feats, _, _ in greedy_results:
    candidate = BEST_FEATURES + feats
    s = quick_cv(candidate)
    if s > best_score:
        BEST_FEATURES = candidate
        best_score = s
        print(f'  ACCEPT {name:30s}: {s:.5f}  (+{s - best_score:.5f})')
    # else skip silently

print()
print(f'Finale feature-set: {BEST_FEATURES}')
print(f'Finale quick CV score: {best_score:.5f}  (delta vs baseline: {best_score - BASELINE:+.5f})')
"""))

# ── Section 6: Full 5-fold Validation ────────────────────────────────────────
cells.append(md("""\
## 6 – Volledige 5-fold Validatie

Definitieve meting met productie-parameters (zelfde als model_baseline.ipynb):
10 000 bomen, lr=0.01, early stopping via eval_set.
"""))

cells.append(code("""\
FULL_PARAMS = dict(
    objective='multi:softprob', num_class=3,
    tree_method='hist', device=device,
    n_estimators=10000, learning_rate=0.01,
    max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, random_state=42, enable_categorical=True,
)

ALL_FEATURES = BASE_FEATURES + BEST_FEATURES
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_all  = train_enc[ALL_FEATURES]
y_all  = train_enc[TARGET]

oof5 = np.zeros((len(y_all), 3))
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf5.split(X_all, y_all)):
    Xtr, Xvl = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    ytr, yvl = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    w = compute_sample_weight('balanced', ytr)
    m = XGBClassifier(**FULL_PARAMS)
    m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xvl, yvl)], verbose=0)
    oof5[val_idx] = m.predict_proba(Xvl)
    s = balanced_accuracy_score(yvl, np.argmax(oof5[val_idx], axis=1))
    fold_scores.append(s)
    print(f'  Fold {fold+1}: {s:.5f}')

full_oof_score = balanced_accuracy_score(y_all, np.argmax(oof5, axis=1))
print(f'\\n5-fold OOF (full model): {full_oof_score:.5f}')
print(f'Baseline OOF            : 0.96552')
print(f'Delta                   : {full_oof_score - 0.96552:+.5f}')
"""))

cells.append(code("""\
# Feature importance from final fold model
import matplotlib.pyplot as plt
final_model = XGBClassifier(**{**FULL_PARAMS, 'n_estimators': 300, 'learning_rate': 0.1})
final_model.fit(X_all, y_all, sample_weight=compute_sample_weight('balanced', y_all), verbose=False)

imp = pd.Series(final_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
imp.head(25).plot(kind='barh', ax=ax)
ax.invert_yaxis()
ax.set_xlabel('XGB Feature Importance (gain)')
ax.set_title('Top-25 Features — XGB Gain Importance')
plt.tight_layout()
plt.show()

print('\\nTop-20 features:')
print(imp.head(20).round(5))
"""))

# ── Section 7: Summary ────────────────────────────────────────────────────────
cells.append(md("## 7 – Samenvatting & Code voor model_baseline.ipynb"))

cells.append(code("""\
print('=== SAMENVATTING ===')
print(f'Baseline XGB OOF (model_baseline.ipynb) : 0.96552')
print(f'Nieuw OOF (5-fold, volle model)          : {full_oof_score:.5f}')
print(f'Delta                                    : {full_oof_score - 0.96552:+.5f}')
print()
print('Opgenomen features:')
for f in BEST_FEATURES:
    print(f'  - {f}')
print()
print('Afgewezen features (hurten individueel):')
print('  - log1p_redshift, redshift_sq (XGB leert dit zelf)')
print('  - mag_skew, mag_kurtosis, brightest_band (ruis)')
print('  - target_encoding spectral_type (multicollineariteit)')
"""))

cells.append(code("""\
# Ready-to-paste code for model_baseline.ipynb
snippet = '''
# ── Feature Engineering (from xgb_eda2.ipynb) ────────────────────────────────
import numpy as np

def add_new_features(df_enc, df_raw):
    \"\"\"
    df_enc  : encoded DataFrame (result of pd.get_dummies)
    df_raw  : original raw DataFrame (with alpha, delta, u, g, r, i, z columns)
    \"\"\"
    # 1. Galactische coördinaten
    def radec_to_galactic(ra_deg, dec_deg):
        ra, dec     = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
        ra_gp, dec_gp = np.deg2rad(192.8595), np.deg2rad(27.1284)
        l_0 = np.deg2rad(122.9320)
        sin_b = (np.sin(dec)*np.sin(dec_gp) +
                 np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp))
        b = np.arcsin(np.clip(sin_b, -1, 1))
        x = np.cos(dec)*np.sin(ra - ra_gp)
        y = (np.sin(dec)*np.cos(dec_gp) -
             np.cos(dec)*np.sin(dec_gp)*np.cos(ra - ra_gp))
        l = l_0 - np.arctan2(x, y)
        return np.rad2deg(l % (2*np.pi)), np.rad2deg(b)

    l, b = radec_to_galactic(df_raw[\\'alpha\\'].values, df_raw[\\'delta\\'].values)
    df_enc[\\'galactic_l\\']        = l
    df_enc[\\'galactic_b\\']        = b
    df_enc[\\'abs_galactic_b\\']    = np.abs(b)
    df_enc[\\'is_galactic_plane\\'] = (np.abs(b) < 20).astype(int)

    # 2. Geselecteerde kleurindices (top Spearman met target)
    # (Vul hier de BEST_FEATURES in op basis van de greedy selectie)
    for b1, b2 in [(\\'g\\',\\'z\\'), (\\'g\\',\\'i\\'), (\\'g\\',\\'r\\'), (\\'u\\',\\'z\\'), (\\'u\\',\\'r\\')]:
        df_enc[f\\'{b1}_{b2}\\'] = df_raw[b1] - df_raw[b2]

    return df_enc

# Gebruik:
# train_enc = add_new_features(train_enc, train)
# test_enc  = add_new_features(test_enc,  test)
'''
print(snippet)
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

import pathlib
out = pathlib.Path(__file__).parent / 'xgb_eda2.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Written: {out}  ({out.stat().st_size/1024:.1f} KB)')
