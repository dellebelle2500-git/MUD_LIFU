"""
Figure4_composite_v2.py — assembles Figure 4 as a single vector PDF.

    a    PHATE embedding of baseline RS-FC, coloured by subtype
    b    patient-by-patient cosine similarity in PC1–5 space
    c    within- versus between-subtype cosine distance in PC1–8 space
    d    subtype-specific control-referenced between-network chord diagrams
    e–g  subtype-specific baseline markers, one-vs-rest volcano plots

Subtypes use the shared Okabe-Ito palette; networks use the shared Tableau
palette and canonical anatomical order from fig_style.
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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
import networkx as nx
import phate
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import (NETWORK_ORDER, NET_COLORS, net_color, short,
                       CLUSTER_COLORS, CLUSTER_NAMES, apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_W, FIG_H = 7.5, 6.85


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


BOX = {
    'a':  rect(0.55, 0.32, 1.85, 1.48),
    'b1': rect(3.05, 0.32, 1.75, 1.48),
    'b2': rect(5.42, 0.32, 1.90, 1.48),
    'c':  rect(0.15, 2.45, 7.20, 1.95),
    'd':  rect(0.55, 5.02, 1.95, 1.45),
    'e':  rect(2.95, 5.02, 1.95, 1.45),
    'f':  rect(5.35, 5.02, 1.95, 1.45),
}
LETTER = {'a': (0.30, 0.17), 'b': (2.80, 0.17), 'c': (5.20, 0.17),
          'd': (0.30, 2.25), 'e': (0.32, 4.86), 'f': (2.72, 4.86),
          'g': (5.12, 4.86)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------- data ---
nl = pd.read_csv(f'{P}/normal_subject_feature_matrix_clean.csv', index_col=0)
bl = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
for df in (nl, bl):
    df.columns = df.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
    df.index = df.index.astype(str)
feat = sorted(set(nl.columns) & set(bl.columns))
nl, bl = nl[feat], bl[feat]

cl = pd.read_csv(f'{P}/subject_cluster_new.csv', index_col=0)
cl.index = cl.index.astype(str)
subs = [s for s in bl.index if s in cl.index]
bl = bl.loc[subs]
labels = cl.loc[subs, 'Cluster'].astype(int).values

CMAP_KW = {'Reward': ['NAc'], 'Default': ['DefaultMode'], 'Relay': ['Thalamus'],
           'Memory-Emotion': ['Hippocampus', 'Amygdala', 'Brain-Stem', 'PaHC'],
           'Execution': ['FrontoParietal'], 'Automaticity': ['Pallidum', 'Putamen'],
           'Compulsion': ['Caudate'], 'Attention': ['DorsalAttention'],
           'Regulation': ['FOrb', 'IFG'], 'Salience': ['Salience']}


def getcat(n):
    for c, ks in CMAP_KW.items():
        for k in ks:
            if k.lower() in n.lower():
                return c
    return 'Other'


CATS = NETWORK_ORDER


def agg_cat(df):
    rows = []
    for _, row in df.iterrows():
        acc = {f'{a}|{b}': [] for i, a in enumerate(CATS) for b in CATS[i:]}
        for f_, v in row.items():
            pp = f_.split('_vs_')
            if len(pp) != 2:
                continue
            c1, c2 = getcat(pp[0]), getcat(pp[1])
            if 'Other' in (c1, c2):
                continue
            i, j = sorted((CATS.index(c1), CATS.index(c2)))
            acc[f'{CATS[i]}|{CATS[j]}'].append(v)
        rows.append({k: (np.mean(v) if v else np.nan) for k, v in acc.items()})
    return pd.DataFrame(rows, index=df.index)


bl_cat, nl_cat = agg_cat(bl), agg_cat(nl)
pairs = list(bl_cat.columns)

# ------------------------------------------------------------------ panel a
ax_a = fig.add_axes(BOX['a'])
op = phate.PHATE(n_components=2, knn=2, decay=40, t='auto', random_state=42,
                 verbose=0)
emb = op.fit_transform(bl.values)
for c in (1, 2, 3):
    m = labels == c
    ax_a.scatter(-emb[m, 0], -emb[m, 1], s=28, color=CLUSTER_COLORS[c], alpha=0.9,
                 edgecolors='white', linewidths=0.6, zorder=5,
                 label=f'{CLUSTER_NAMES[c]} (n = {int(m.sum())})')
ax_a.set_xlabel('PHATE 1', fontsize=FONT['axis_label'])
ax_a.set_ylabel('PHATE 2', fontsize=FONT['axis_label'])
ax_a.tick_params(labelsize=FONT['micro'], length=2, pad=1)
ax_a.legend(fontsize=fs('stat_inset', -0.8), frameon=False, loc='upper right',
            bbox_to_anchor=(1.02, 1.02), handletextpad=0.25, borderpad=0.2,
            labelspacing=0.18, markerscale=0.7)
ax_a.spines[['top', 'right']].set_visible(False)

# --------------------------------------------------------------- panels b,c
pca8 = PCA(n_components=8, random_state=42)
S = pca8.fit_transform(bl.values)
expl = pca8.explained_variance_ratio_.sum()
expl5 = pca8.explained_variance_ratio_[:5].sum()
prof = {c: S[labels == c].mean(axis=0) for c in (1, 2, 3)}


def cosdist(u, v):
    return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


rng = np.random.default_rng(42)
comb = [(1, 2), (1, 3), (2, 3)]
# screen blend of the two subtype colours: 1-(1-a)(1-b)
BLEND = {(1, 2): '#D5A6B2', (1, 3): '#D5C273', (2, 3): '#00C9D5'}
ax_b1 = fig.add_axes(BOX['b1'])
K_VIS = 5
S_vis = S[:, :K_VIS]
order_vis = np.argsort(labels, kind='stable')
lord_vis = labels[order_vis]
Nv = S_vis[order_vis] / np.linalg.norm(S_vis[order_vis], axis=1, keepdims=True)
COS15 = Nv @ Nv.T
SIM_CMAP = LinearSegmentedColormap.from_list(
    'simrb', ['#2C6FAD', '#6EA6D0', '#FFFFFF', '#D98C86', '#B4403A'])
ax_b1.imshow(COS15, cmap=SIM_CMAP, vmin=-1, vmax=1)
for e in np.where(np.diff(lord_vis))[0] + 0.5:
    ax_b1.axhline(e, color='#111', lw=0.9)
    ax_b1.axvline(e, color='#111', lw=0.9)
for j, c in enumerate(lord_vis):
    ax_b1.add_patch(mpatches.Rectangle((-1.5, j - 0.5), 0.85, 1.0,
                                       color=CLUSTER_COLORS[c], clip_on=False))
    ax_b1.add_patch(mpatches.Rectangle((j - 0.5, 14.65), 1.0, 0.85,
                                       color=CLUSTER_COLORS[c], clip_on=False))
ax_b1.set_xticks([]); ax_b1.set_yticks([])
for sp in ax_b1.spines.values():
    sp.set_visible(False)
ax_b1.set_title(f'patient similarity (PC1\u20135, {expl5*100:.0f}% of variance)',
                fontsize=FONT['panel_title'], color='#333', loc='left', pad=4)
cb_b1 = fig.colorbar(ax_b1.images[0], ax=ax_b1, fraction=0.045, pad=0.03)
cb_b1.set_ticks([-1, 0, 1])
cb_b1.set_label('cosine similarity', fontsize=FONT['micro_small'])
cb_b1.ax.tick_params(labelsize=FONT['micro_small'], length=2)
cb_b1.outline.set_visible(False)

_prof5 = np.array([S_vis[labels == c].mean(axis=0) for c in (1, 2, 3)])
_u5 = _prof5 / np.linalg.norm(_prof5, axis=1, keepdims=True)
_cen5 = _u5 @ _u5.T
_w = [COS15[i, j] for i in range(15) for j in range(i + 1, 15)
      if lord_vis[i] == lord_vis[j]]
_b = [COS15[i, j] for i in range(15) for j in range(i + 1, 15)
      if lord_vis[i] != lord_vis[j]]
print(f'  b PC1-{K_VIS} ({expl5*100:.1f}% var): within {np.mean(_w):+.3f}, '
      f'between {np.mean(_b):+.3f}, MWU p = '
      f'{mannwhitneyu(_w, _b, alternative="greater")[1]:.2e}')
print(f'    centroid cosine  C1-C2 {_cen5[0,1]:+.3f}  C1-C3 {_cen5[0,2]:+.3f}  '
      f'C2-C3 {_cen5[1,2]:+.3f}')

ax_b2 = fig.add_axes(BOX['b2'])
within_by = {c: [] for c in (1, 2, 3)}
between_all = []
for i in range(len(subs)):
    for j in range(i + 1, len(subs)):
        d = cosdist(S[i], S[j])
        if labels[i] == labels[j]:
            within_by[labels[i]].append(d)
        else:
            between_all.append(d)
all_within = [d for v in within_by.values() for d in v]
p_wb = mannwhitneyu(all_within, between_all, alternative='two-sided')[1]

for k, (vals, colr) in enumerate([(all_within, '#7A7A7A'), (between_all, '#C4C8CC')]):
    v = np.asarray(vals, float)
    ax_b2.boxplot([v], positions=[k], widths=0.48, patch_artist=True,
                  showfliers=False, zorder=2,
                  medianprops=dict(color='#222', lw=1.0),
                  whiskerprops=dict(color='#999', lw=0.7),
                  capprops=dict(color='#999', lw=0.7),
                  boxprops=dict(facecolor=colr, alpha=0.45, edgecolor=colr, lw=1.0))
for c in (1, 2, 3):
    v = np.asarray(within_by[c], float)
    ax_b2.scatter(rng.normal(0, 0.075, len(v)), v, s=8, color=CLUSTER_COLORS[c],
                  alpha=0.9, edgecolors='white', linewidths=0.25, zorder=5)
v = np.asarray(between_all, float)
ax_b2.scatter(1 + rng.normal(0, 0.075, len(v)), v, s=8, color='#9AA0A6',
              alpha=0.75, edgecolors='white', linewidths=0.25, zorder=5)

_ymax = max(max(all_within), max(between_all))
_ymin = min(min(all_within), min(between_all))
_span = _ymax - _ymin
_bar = _ymax + _span * 0.08
ax_b2.plot([0, 0, 1, 1], [_bar, _bar + _span * 0.03, _bar + _span * 0.03, _bar],
           color='#555', lw=0.8, zorder=6)
ax_b2.text(0.5, _bar + _span * 0.05,
           f'$p$ = {p_wb:.1e}'.replace('e-0', r'$\times$10$^{-') + '}$',
           ha='center', va='bottom', fontsize=FONT['stat_inset'], color='#333')
ax_b2.set_xticks([0, 1])
ax_b2.set_xticklabels(['within\nsubtype', 'between\nsubtypes'], fontsize=FONT['micro'])
ax_b2.set_xlim(-0.6, 1.6)
ax_b2.set_ylim(top=_ymax + _span * 0.24)
ax_b2.set_ylabel('subject-pair cosine distance', fontsize=FONT['axis_label'])
ax_b2.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_b2.tick_params(axis='x', length=2, pad=1)
ax_b2.set_title('subject-pair distance', fontsize=FONT['panel_title'],
                color='#333', loc='left', pad=4)
ax_b2.spines[['top', 'right']].set_visible(False)
print(f'  c within {np.mean(all_within):.2f} (n={len(all_within)}) vs '
      f'between {np.mean(between_all):.2f} (n={len(between_all)}), MWU p = {p_wb:.2e}')

# ------------------------------------------------------------------ panel d
# chord diagram: arc length = sum |Z| over the pairs a network takes part in
# (equal to the total ribbon width it carries); ribbons = between-network Z.
ncm, ncs = nl_cat.mean(), nl_cat.std(ddof=1).replace(0, 1e-9)
between = [c for c in bl_cat.columns if c.split('|')[0] != c.split('|')[1]]

N_CAT = len(CATS)
GAP = np.radians(2.6)
R_OUT, R_IN = 1.00, 0.94
MIN_FRAC = 0.22
Z_CAP = 3.0


def build_arcs(weights):
    """arc length proportional to the given weight, clockwise from 12 o'clock"""
    w = np.array([max(weights[c], 0.0) for c in CATS], float)
    w = np.maximum(w, w.mean() * MIN_FRAC)
    spans = w / w.sum() * (2 * np.pi - N_CAT * GAP)
    arc0, cur = {}, np.pi / 2
    for c, sp in zip(CATS, spans):
        cur -= sp
        arc0[c] = cur
        cur -= GAP
    return arc0, dict(zip(CATS, spans))


def ribbon(ax, th1a, th1b, th2a, th2b, color, alpha, zo=2):
    p1a = np.array([R_IN * np.cos(th1a), R_IN * np.sin(th1a)])
    p1b = np.array([R_IN * np.cos(th1b), R_IN * np.sin(th1b)])
    p2a = np.array([R_IN * np.cos(th2a), R_IN * np.sin(th2a)])
    p2b = np.array([R_IN * np.cos(th2b), R_IN * np.sin(th2b)])
    ctrl = np.array([0.0, 0.0])
    verts = [p1b, ctrl, p2a, p2b, ctrl, p1a, p1a]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color,
                                    edgecolor='none', alpha=alpha, zorder=zo))


dx, dy, dw, dh = BOX['c']
cell = dw / 3
for k, cid in enumerate((1, 2, 3)):
    axc = fig.add_axes([dx + k * cell + cell * 0.04, dy, cell * 0.92, dh])
    axc.set_aspect('equal'); axc.axis('off')
    axc.set_xlim(-1.30, 1.30); axc.set_ylim(-1.32, 1.30)
    Z = (bl_cat[labels == cid].mean() - ncm) / ncs
    tot = {c: sum(abs(Z[cp]) for cp in between if c in cp.split('|')) for c in CATS}
    arc0, span_of = build_arcs(tot)
    used = {c: 0.0 for c in CATS}
    for cp in sorted(between, key=lambda q: abs(Z[q])):
        a, b = cp.split('|')
        z = Z[cp]
        if pd.isna(z) or tot[a] == 0 or tot[b] == 0:
            continue
        mag = min(abs(z), Z_CAP) / Z_CAP
        wa = span_of[a] * abs(z) / tot[a]
        wb = span_of[b] * abs(z) / tot[b]
        th1a = arc0[a] + used[a]; th2a = arc0[b] + used[b]
        used[a] += wa; used[b] += wb
        ribbon(axc, th1a, th1a + wa, th2a, th2a + wb,
               '#C0392B' if z > 0 else '#2C6FA6', 0.10 + 0.55 * mag, 2 + int(4 * mag))
    for c in CATS:
        th = np.linspace(arc0[c], arc0[c] + span_of[c], 60)
        axc.plot(R_OUT * np.cos(th), R_OUT * np.sin(th), color=net_color(c),
                 lw=3.4, solid_capstyle='butt', zorder=8)
        m = arc0[c] + span_of[c] / 2
        rot = np.degrees(m); ha = 'left'
        if 90 < rot % 360 < 270:
            rot -= 180; ha = 'right'
        axc.text(1.07 * np.cos(m), 1.07 * np.sin(m), short(c),
                 fontsize=FONT['micro_small'], color=net_color(c), fontweight='bold',
                 ha=ha, va='center', rotation=rot, rotation_mode='anchor')
    n_hyper = int((Z[between] > 0).sum()); n_hypo = int((Z[between] < 0).sum())
    axc.set_title(f'{CLUSTER_NAMES[cid]} (n = {int((labels == cid).sum())})\n'
                  f'{n_hyper} hyper / {n_hypo} hypo pairs',
                  fontsize=FONT['panel_title'], color=CLUSTER_COLORS[cid],
                  fontweight='bold', pad=4)
    print(f'  d C{cid}: {n_hyper} hyper / {n_hypo} hypo, arc {min(tot.values()):.1f}-{max(tot.values()):.1f}')

fig.legend(handles=[
    Line2D([0], [0], color='#C0392B', lw=3, alpha=0.7, label='$Z$ > 0 (hyper-connectivity)'),
    Line2D([0], [0], color='#2C6FA6', lw=3, alpha=0.7, label='$Z$ < 0 (hypo-connectivity)')],
    loc='center', bbox_to_anchor=(0.5, 1 - (4.64 / FIG_H)), ncol=2,
    fontsize=FONT['stat_inset'], frameon=False)

# ---------------------------------------------------------------- panels e-g
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) /
                 (n1 + n2 - 2))
    return 0.0 if sp == 0 else (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


for key, cid in [('d', 1), ('e', 2), ('f', 3)]:
    axv = fig.add_axes(BOX[key])
    inn = bl_cat[labels == cid]
    out = bl_cat[labels != cid]
    gs, ps, names = [], [], []
    for cp in pairs:
        a_, b_ = inn[cp].dropna().values, out[cp].dropna().values
        if len(a_) < 2 or len(b_) < 2:
            continue
        gs.append(hedges_g(a_, b_))
        ps.append(mannwhitneyu(a_, b_, alternative='two-sided')[1])
        names.append(cp)
    gs, ps = np.array(gs), np.array(ps)
    logp = -np.log10(ps)
    sig = ps < 0.05
    axv.scatter(gs[~sig], logp[~sig], s=5, color='#C4C8CC', alpha=0.7,
                edgecolors='none')
    axv.scatter(gs[sig], logp[sig], s=16, color=CLUSTER_COLORS[cid], alpha=0.9,
                edgecolors='white', linewidths=0.4, zorder=5)
    axv.axhline(-np.log10(0.05), color='#999', ls=':', lw=0.7)
    axv.axvline(0, color='#CCC', lw=0.6)
    # Place significant labels on left/right rails with leader lines.
    # This keeps 5-6 labels legible without changing the data positions.
    axv.margins(x=0.42, y=0.25)
    if np.any(sig):
        xmin, xmax = axv.get_xlim(); ymin, ymax = axv.get_ylim()
        xspan, yspan = xmax - xmin, ymax - ymin
        for side in (-1, 1):
            idxs = [ii for ii in np.where(sig)[0] if (gs[ii] < 0) == (side < 0)]
            idxs = sorted(idxs, key=lambda ii: logp[ii], reverse=True)
            if not idxs:
                continue
            top = ymax - 0.06 * yspan
            sep = 0.095 * yspan
            rail_x = xmin + 0.03 * xspan if side < 0 else xmax - 0.03 * xspan
            for jj, ii in enumerate(idxs):
                yy = top - jj * sep
                a, b = names[ii].split('|')
                txt = f'{short(a)}\u2013{short(b)}'
                axv.annotate(txt, xy=(gs[ii], logp[ii]), xytext=(rail_x, yy),
                             textcoords='data', ha='left' if side < 0 else 'right', va='center',
                             fontsize=FONT['micro_small'], color=CLUSTER_COLORS[cid],
                             fontweight='bold',
                             arrowprops=dict(arrowstyle='-', color='#B8BDC2', lw=0.45,
                                             shrinkA=1, shrinkB=2))
    axv.set_xlabel("Hedges' $g$ (one vs rest)", fontsize=FONT['axis_label'])
    if key == 'd':
        axv.set_ylabel('$-\\log_{10}$ $p$', fontsize=FONT['axis_label'])
    axv.set_title(f'{CLUSTER_NAMES[cid]}  ({int(sig.sum())}/{len(gs)} pairs)',
                  fontsize=FONT['panel_title'], color=CLUSTER_COLORS[cid],
                  fontweight='bold', loc='left', pad=4)
    axv.tick_params(labelsize=FONT['micro'], length=2, pad=1)
    axv.spines[['top', 'right']].set_visible(False)
    print(f'  {key} C{cid}: {int(sig.sum())} significant pairs')


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure4.pdf',
    OUTPUT_DIR / 'Figure4_preview.png', preview_dpi=200
)

plt.close()
print('saved Figure4.pdf / Figure4_preview.png')
print(f'  c: PCA8 explains {expl*100:.1f}%, all-within {np.mean(all_within):.2f} vs '
      f'between {np.mean(between_all):.2f}, MWU p = {p_wb:.2e}')
