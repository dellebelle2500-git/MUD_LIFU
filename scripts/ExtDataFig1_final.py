"""
ExtDataFig1_clustering.py — how the connectomic subtypes were arrived at, how
stable they are, and what separates them.

    a  scree plot of the patient connectomes, with the elbow at PC8
    b  cumulative variance, marking the 73.5% carried by eight components
    c  Ward dendrogram on Euclidean distances between the first eight components
    d  the same tree computed on cosine distances with average linkage
    e  PCA(8) -> Ward(k = 3) -> t-SNE, the procedure used for the reported subtypes
    f  the same partition recovered across principal-component counts and t-SNE
       perplexity
    g  co-clustering stability under repeated resampling of the connectivity
       features, with the PCA basis held fixed
    h  the twelve network pairs that separate a subtype from the rest, as
       control-referenced Z scores, with patient and network-pair dendrograms

Panel f shows only whether the reported partition is recovered at each setting;
the embedding geometry itself is not the point, so the panels are small.

Panel h uses the category definitions and ordering of the original volcano
scripts (fig_volcano_cat_C1/C2/C3.py), so the pairs shown are exactly those
plotted in Fig. 4e-g.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import (linkage, fcluster, dendrogram,
                                     set_link_color_palette)
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import CLUSTER_COLORS, CLUSTER_NAMES, apply_base_style, FONT, fs, nature_figsize, save_nature_figure

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
bl = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
nl = pd.read_csv(f'{P}/normal_subject_feature_matrix_clean.csv', index_col=0)
for d in (bl, nl):
    d.columns = d.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
cl = pd.read_csv(f'{P}/subject_cluster_new.csv', index_col=0)
ids = np.array([str(i).replace('.mat', '').replace('sub_', '') for i in bl.index])
ref = cl.loc[bl.index, 'Cluster'].astype(int).values
feat = sorted(set(bl.columns) & set(nl.columns))
X = bl[feat].values
REFSETS = sorted([set(ids[ref == c]) for c in (1, 2, 3)], key=lambda s: sorted(s))


def matches_reference(lab):
    got = sorted([set(ids[lab == c]) for c in set(lab)], key=lambda s: sorted(s))
    return got == REFSETS


# --- category definitions, matching the original volcano scripts exactly ---
CATEGORY_MAP = {
    'Reward': ['NAc'], 'Default': ['DefaultMode'], 'Relay': ['Thalamus'],
    'Memory-Emotion': ['Hippocampus', 'Amygdala', 'Brain-Stem', 'PaHC'],
    'Execution': ['FrontoParietal'], 'Automaticity': ['Pallidum', 'Putamen'],
    'Compulsion': ['Caudate'], 'Attention': ['DorsalAttention'],
    'Regulation': ['FOrb', 'IFG'], 'Salience': ['Salience'],
}
CATS = list(CATEGORY_MAP.keys())
SHORT = {'Memory-Emotion': 'Mem', 'Automaticity': 'Auto', 'Compulsion': 'Comp',
         'Execution': 'Exec', 'Regulation': 'Reg', 'Attention': 'Att',
         'Default': 'DMN', 'Salience': 'Sal', 'Reward': 'Rew', 'Relay': 'Relay'}


def get_cat(name):
    for cat, kws in CATEGORY_MAP.items():
        for k in kws:
            if k.lower() in name.lower():
                return cat
    return 'Other'


def aggregate(df):
    """collapse the 990 edges to the 55 category pairs, one row per subject"""
    rows = []
    for _, row in df.iterrows():
        acc = {c: {c2: [] for c2 in CATS} for c in CATS}
        for f, v in row.items():
            parts = f.split('_vs_')
            if len(parts) != 2:
                continue
            a, b = get_cat(parts[0]), get_cat(parts[1])
            if a in CATS and b in CATS:
                acc[a][b].append(v)
                if a != b:
                    acc[b][a].append(v)
        fr = {}
        for i in range(len(CATS)):
            for j in range(i, len(CATS)):
                vals = acc[CATS[i]][CATS[j]]
                fr[f'{CATS[i]}_vs_{CATS[j]}'] = np.mean(vals) if vals else 0
        rows.append(fr)
    return pd.DataFrame(rows, index=df.index)


def hedges_g(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) /
                 (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp * (1 - 3 / (4 * (n1 + n2 - 2) - 1))


FIG_W, FIG_H = 7.5, 7.05
fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


for L, (lx, ly) in {'a': (0.42, 0.14), 'b': (4.12, 0.14),
                    'c': (0.48, 1.70), 'd': (4.18, 1.70),
                    'e': (0.60, 3.20), 'f': (4.18, 3.20),
                    'g': (0.40, 4.74), 'h': (3.52, 4.74)}.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------ a, b
pca_full = PCA().fit(X)
ev = pca_full.explained_variance_ratio_
cum = np.cumsum(ev)
K = 8

ax_a = fig.add_axes(rect(0.62, 0.28, 2.75, 1.05))
ax_a.plot(np.arange(1, len(ev) + 1), ev * 100, 'o-', color='#2C6FA6', ms=3.4, lw=1.2)
ax_a.scatter([K], [ev[K - 1] * 100], s=55, facecolor='none', edgecolor='#C0392B',
             lw=1.3, zorder=6)
ax_a.annotate(f'elbow, PC{K}', xy=(K, ev[K - 1] * 100),
              xytext=(K + 1.6, ev[K - 1] * 100 + 3.4), fontsize=FONT['annotation'],
              color='#C0392B', arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.8))
ax_a.set_xlabel('principal component', fontsize=FONT['axis_label'])
ax_a.set_ylabel('explained variance (%)', fontsize=FONT['axis_label'])
ax_a.tick_params(labelsize=FONT['tick'])
ax_a.spines[['top', 'right']].set_visible(False)

ax_b = fig.add_axes(rect(4.35, 0.28, 2.75, 1.05))
ax_b.plot(np.arange(1, len(cum) + 1), cum * 100, 's-', color='#C0392B', ms=3.2, lw=1.2)
ax_b.axhline(cum[K - 1] * 100, color='#999', ls='--', lw=0.8)
ax_b.axvline(K, color='#999', ls='--', lw=0.8)
ax_b.annotate(f'{cum[K-1]*100:.1f}% at PC{K}', xy=(K, cum[K - 1] * 100),
              xytext=(K + 0.6, cum[K - 1] * 100 - 16), fontsize=FONT['annotation'],
              color='#333')
ax_b.set_xlabel('principal component', fontsize=FONT['axis_label'])
ax_b.set_ylabel('cumulative variance (%)', fontsize=FONT['axis_label'])
ax_b.tick_params(labelsize=FONT['tick'])
ax_b.spines[['top', 'right']].set_visible(False)

S8 = PCA(n_components=8, random_state=42).fit_transform(X)
Z_ward = linkage(S8, 'ward')
Z_cos = linkage(pdist(S8, 'cosine'), 'average')


def draw_tree(ax, Z, title):
    lab = fcluster(Z, 3, criterion='maxclust')
    mapping = {c: np.bincount(ref[lab == c]).argmax() for c in set(lab)}
    order = dendrogram(Z, no_plot=True)['leaves']
    seen, pal = [], []
    for i in order:
        c = mapping[lab[i]]
        if c not in seen:
            seen.append(c)
            pal.append(CLUSTER_COLORS[c])
    set_link_color_palette(pal)
    thr = (Z[-3, 2] + Z[-2, 2]) / 2
    dn = dendrogram(Z, no_labels=True, ax=ax, color_threshold=thr,
                    above_threshold_color='#9AA0A6')
    ax.axhline(thr, ls='--', color='#777', lw=0.8)
    # a colour bar under the leaves stands in for the (withheld) patient numbers
    y0 = -0.045 * ax.get_ylim()[1]
    for k, i in enumerate(dn['leaves']):
        ax.add_patch(plt.Rectangle((k * 10 + 1.5, y0), 7, 0.030 * ax.get_ylim()[1],
                                   color=CLUSTER_COLORS[ref[i]], clip_on=False))
    ax.set_xticks([])
    ax.set_ylabel('merge distance', fontsize=FONT['axis_label'])
    ax.tick_params(axis='y', labelsize=FONT['tick'], length=2)
    ax.tick_params(axis='x', length=0, pad=1)
    ax.set_title(title, fontsize=FONT['panel_title'], loc='left', pad=4)
    ax.spines[['top', 'right']].set_visible(False)
    set_link_color_palette(None)
    return [ids[i] for i in dn['leaves']]


# ------------------------------------------------------------------ c, d
ax_c = fig.add_axes(rect(0.72, 1.84, 2.62, 1.10))
o_w = draw_tree(ax_c, Z_ward, 'Euclidean distance, Ward linkage')
ax_d = fig.add_axes(rect(4.42, 1.84, 2.62, 1.10))
o_c = draw_tree(ax_d, Z_cos, 'cosine distance, average linkage')
print('  c Ward order  :', ' '.join(o_w))
print('  d cosine order:', ' '.join(o_c))

# ------------------------------------------------------------------ e, f
emb_c = TSNE(n_components=2, perplexity=3, random_state=0, init='pca',
             learning_rate='auto').fit_transform(S8)


def scatter_subtypes(ax, emb, s=30):
    for c in (1, 2, 3):
        m = ref == c
        ax.scatter(emb[m, 0], emb[m, 1], s=s, color=CLUSTER_COLORS[c], alpha=0.92,
                   edgecolors='white', linewidths=0.5, zorder=5)


lab_e = fcluster(Z_ward, 3, criterion='maxclust')
ax_e = fig.add_axes(rect(0.85, 3.40, 2.35, 0.96))
scatter_subtypes(ax_e, emb_c)
ax_e.set_xticks([]); ax_e.set_yticks([])
ax_e.set_xlabel('t-SNE 1', fontsize=FONT['axis_label'])
ax_e.set_ylabel('t-SNE 2', fontsize=FONT['axis_label'])
ax_e.set_title('PCA(8) \u2192 Ward \u2192 t-SNE\nclustering in component space',
               fontsize=FONT['panel_title'], color='#333', loc='left', pad=4)
ax_e.spines[['top', 'right']].set_visible(False)
print(f'  e clustering on components reproduces the partition: {matches_reference(lab_e)}')

PCS = (8, 10)
PERPS = (2, 3, 4, 5)
fx, fy, fw, fh = rect(4.42, 3.40, 2.72, 0.96)
cw, ch = fw / len(PERPS), fh / len(PCS)
n_ok = 0
for i, k in enumerate(PCS):
    Sk = PCA(n_components=k, random_state=42).fit_transform(X)
    for j, perp in enumerate(PERPS):
        e = TSNE(n_components=2, perplexity=perp, random_state=0, init='pca',
                 learning_rate='auto').fit_transform(Sk)
        lab = fcluster(linkage(e, 'ward'), 3, criterion='maxclust')
        ok = matches_reference(lab)
        n_ok += ok
        ax = fig.add_axes([fx + j * cw, fy + (len(PCS) - 1 - i) * ch + ch * 0.06,
                           cw * 0.86, ch * 0.62])
        if ok:
            ax.set_facecolor('#EAF1F8')
        scatter_subtypes(ax, e, s=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#8FA9C4' if ok else '#DDE0E3')
            sp.set_linewidth(1.0 if ok else 0.6)
        ax.set_title(f'{k} PCs, perp {perp}', fontsize=FONT['micro'],
                     color='#2C6FA6' if ok else '#8A8F94', pad=2)
        print(f'  f PC{k}/perp{perp}: {"recovered" if ok else "differs"}')
fig.text((fx + fw / 2), fy - 0.012, f'partition recovered in {n_ok} of '
         f'{len(PCS) * len(PERPS)} settings (shaded)',
         ha='center', va='top', fontsize=FONT['stat_inset'], color='#555')

# ------------------------------------------------------------------ g
# co-clustering stability: resample 80% of the edges, keep the PCA basis fixed
N_RESAMPLES = 1000
EDGE_FRACTION = 0.80
SEED = 20260814

pca8 = PCA(n_components=8, random_state=42).fit(X)
loadings = pca8.components_
Xc = X - pca8.mean_
rng = np.random.default_rng(SEED)
n_sub, n_edge = X.shape
n_keep = int(round(n_edge * EDGE_FRACTION))

cons_w = np.zeros((n_sub, n_sub))
cons_c = np.zeros((n_sub, n_sub))
exact_w = exact_c = 0
for _ in range(N_RESAMPLES):
    idx = rng.choice(n_edge, size=n_keep, replace=False)
    # scores reconstructed in the full-data basis from the retained edges only
    Sf = Xc[:, idx] @ loadings[:, idx].T
    lw = fcluster(linkage(Sf, 'ward'), 3, criterion='maxclust')
    lc = fcluster(linkage(pdist(Sf, 'cosine'), 'average'), 3, criterion='maxclust')
    cons_w += (lw[:, None] == lw[None, :])
    cons_c += (lc[:, None] == lc[None, :])
    exact_w += matches_reference(lw)
    exact_c += matches_reference(lc)
consensus = 0.5 * (cons_w + cons_c) / N_RESAMPLES

within = [consensus[i, j] for i in range(n_sub) for j in range(i + 1, n_sub)
          if ref[i] == ref[j]]
between = [consensus[i, j] for i in range(n_sub) for j in range(i + 1, n_sub)
           if ref[i] != ref[j]]
print(f'  g within-subtype {np.mean(within):.3f} | between {np.mean(between):.3f}')
print(f'  g exact recovery: Ward {exact_w/N_RESAMPLES:.3f} | cosine {exact_c/N_RESAMPLES:.3f}')

order_w = dendrogram(Z_ward, no_plot=True)['leaves']
M_cons = consensus[np.ix_(order_w, order_w)]
ax_g = fig.add_axes(rect(0.62, 4.92, 2.30, 1.72))
im_g = ax_g.imshow(M_cons, vmin=0, vmax=1, cmap='viridis', aspect='equal')
ax_g.set_xticks([]); ax_g.set_yticks([])
for k, i in enumerate(order_w):
    ax_g.add_patch(plt.Rectangle((k - 0.5, n_sub - 0.5), 1, 0.55,
                                 color=CLUSTER_COLORS[ref[i]], clip_on=False))
    ax_g.add_patch(plt.Rectangle((-1.05, k - 0.5), 0.55, 1,
                                 color=CLUSTER_COLORS[ref[i]], clip_on=False))
grp = ref[order_w]
start_i = 0
for i in range(1, n_sub + 1):
    if i == n_sub or grp[i] != grp[start_i]:
        if i < n_sub:
            ax_g.axvline(i - 0.5, color='white', lw=1.1)
            ax_g.axhline(i - 0.5, color='white', lw=1.1)
        ax_g.text((start_i + i - 1) / 2, -1.15, f'C{grp[start_i]}',
                  ha='center', va='bottom', fontsize=FONT['stat_inset'],
                  fontweight='bold', color=CLUSTER_COLORS[grp[start_i]])
        start_i = i
ax_g.tick_params(length=1.2, pad=1)
ax_g.set_xlabel('patient', fontsize=FONT['axis_label'])
ax_g.set_ylabel('patient', fontsize=FONT['axis_label'])
cax_g = fig.add_axes(rect(3.02, 5.22, 0.09, 1.06))
cb_g = fig.colorbar(im_g, cax=cax_g)
cb_g.set_label('co-clustering probability', fontsize=FONT['stat_inset'])
cb_g.ax.tick_params(labelsize=FONT['micro'], length=1.5)

# ------------------------------------------------------------------ h
A = aggregate(bl)
N = aggregate(nl)
sig = set()
for cid in (1, 2, 3):
    n_c = 0
    for cp in A.columns:
        v1, v2 = A.loc[ref == cid, cp].values, A.loc[ref != cid, cp].values
        if mannwhitneyu(v1, v2, alternative='two-sided')[1] < 0.05:
            sig.add(cp)
            n_c += 1
    print(f'  h C{cid}: {n_c} significant pairs')
sel = sorted(sig)
pair_lab = [f"{SHORT[c.split('_vs_')[0]]}\u2013{SHORT[c.split('_vs_')[1]]}" for c in sel]
Zmat = np.zeros((len(ids), len(sel)))
for j, cp in enumerate(sel):
    ctrl = N[cp].values
    Zmat[:, j] = (A[cp].values - ctrl.mean()) / ctrl.std(ddof=1)
print(f'  h {len(sel)} unique pairs: {", ".join(pair_lab)}')

Z_col = linkage(Zmat.T, 'average')
ax_ht = fig.add_axes(rect(4.92, 4.92, 2.10, 0.32))
lab3 = fcluster(Z_ward, 3, criterion='maxclust')
mapping = {c: np.bincount(ref[lab3 == c]).argmax() for c in set(lab3)}
o0 = dendrogram(Z_ward, no_plot=True)['leaves']
seen, pal = [], []
for i in o0:
    c = mapping[lab3[i]]
    if c not in seen:
        seen.append(c)
        pal.append(CLUSTER_COLORS[c])
set_link_color_palette(pal)
dnt = dendrogram(Z_ward, ax=ax_ht, color_threshold=(Z_ward[-3, 2] + Z_ward[-2, 2]) / 2,
                 above_threshold_color='#9AA0A6', no_labels=True)
ax_ht.axis('off')
op = dnt['leaves']
set_link_color_palette(None)

ax_hl = fig.add_axes(rect(3.52, 5.28, 0.42, 1.30))
set_link_color_palette(['#9AA0A6'])
dnl = dendrogram(Z_col, ax=ax_hl, orientation='left', color_threshold=0,
                 above_threshold_color='#9AA0A6', no_labels=True)
ax_hl.axis('off')
oc = dnl['leaves']
set_link_color_palette(None)

ax_h = fig.add_axes(rect(4.92, 5.28, 2.10, 1.30))
M = Zmat[np.ix_(op, oc)].T
v = np.percentile(np.abs(Zmat), 96)
im = ax_h.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
ax_h.set_xticks([])
ax_h.set_yticks(range(len(oc)))
ax_h.set_yticklabels([pair_lab[j] for j in oc], fontsize=FONT['micro'])
for k, i in enumerate(op):
    ax_h.add_patch(plt.Rectangle((k - 0.5, len(oc) - 0.5), 1, 0.5,
                                 color=CLUSTER_COLORS[ref[i]], clip_on=False))
prev = ref[op[0]]
for k, i in enumerate(op[1:], 1):
    if ref[i] != prev:
        ax_h.axvline(k - 0.5, color='#111', lw=1.3)
        prev = ref[i]
ax_h.set_xlabel('patient', fontsize=FONT['axis_label'], labelpad=8)
ax_h.tick_params(length=1.5, pad=1)
cax = fig.add_axes(rect(7.14, 5.50, 0.09, 0.86))
cb = fig.colorbar(im, cax=cax)
cb.set_label('$Z$ vs controls', fontsize=FONT['stat_inset'])
cb.ax.tick_params(labelsize=FONT['micro'], length=1.5)

fig.legend(handles=[Line2D([0], [0], marker='o', color='w', markersize=5,
                           markerfacecolor=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
                    for c in (1, 2, 3)],
           loc='center', bbox_to_anchor=(0.5, 1 - (4.62 / FIG_H)), ncol=3,
           fontsize=FONT['legend'], frameon=False)


save_nature_figure(
    fig, OUTPUT_DIR / 'ExtDataFig1.pdf',
    OUTPUT_DIR / 'ExtDataFig1_preview.png', preview_dpi=200,
    submission_jpg_path=OUTPUT_DIR / 'ExtDataFig1.jpg', submission_jpg_dpi=300
)

plt.close()
print('saved ExtDataFig1.pdf / ExtDataFig1_preview.png')
