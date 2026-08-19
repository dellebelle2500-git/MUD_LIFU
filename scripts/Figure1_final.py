"""
Figure1_composite_v2.py — assembles Figure 1 as a single vector PDF.

Layout follows MUD_main_Figs_v7.pptx (7.5 x 10.83 in):

    a  whole-brain paired t-test montage      (raster, left blank)
    b  hierarchical clustering of node        (re-implemented with scipy so it
       connectivity profiles                   lives inside this figure)
    c  ROI-level circular connectome
    d  eigenvector centrality   (Hedges' g)
    e  participation coefficient (Hedges' g)
    f  craving VAS
    g  HAM-D / HAM-A
    h  Kaplan-Meier relapse-free survival

Every panel is drawn with fig_style so that network order, colour and type
sizes match across the figure. Panels d and e are computed directly in this
script (no fig1_de_panels dependency). Text is embedded as text (fonttype 42),
so the PDF can be edited downstream; the space reserved for panel a is where
the fMRI montage is dropped in.
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
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram  # dendrogram used for leaf order only
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, wilcoxon
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import (NETWORK_ORDER, NET_COLORS, net_color, short,
                       apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
RASTER = ROOT / 'rasters'
FIG_W, FIG_H = 7.5, 7.05

def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]

# Nature final-size layout: 180 mm wide, 169.2 mm high.
BOX = {
    'a':  rect(0.07, 0.20, 3.84, 2.00),
    'b':  rect(4.00, 0.32, 3.43, 1.88),
    'c':  rect(0.18, 2.34, 4.90, 3.16),
    'd':  rect(5.38, 2.42, 1.92, 1.34),
    'e':  rect(5.38, 4.10, 1.92, 1.34),
    'f':  rect(0.42, 5.88, 2.42, 0.94),
    'g1': rect(3.32, 5.88, 1.34, 0.94),
    'g2': rect(4.92, 5.88, 1.34, 0.94),
    'h':  rect(6.52, 5.88, 0.92, 0.94),
}
LETTER = {'a': (0.10, 0.10), 'b': (4.02, 0.18), 'c': (0.10, 2.22),
          'd': (5.20, 2.30), 'e': (5.20, 3.98), 'f': (0.18, 5.73),
          'g': (3.20, 5.73), 'h': (6.40, 5.73)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))

for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'], fontweight='bold',
             va='top', ha='left')

# ---------------------------------------------------------------- shared data
dn = pd.read_csv(f'{P}/normal_subject_feature_matrix_clean.csv', index_col=0)
dp = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
for df in (dn, dp):
    df.columns = df.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
common = sorted(set(dn.columns) & set(dp.columns))
dn, dp = dn[common], dp[common]

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

rois = set()
for col in common:
    p = col.split('_vs_')
    if len(p) == 2:
        rois.add(p[0]); rois.add(p[1])
roi_list = sorted(rois)
roi_cat = {r: getcat(r) for r in roi_list}

def shorten(roi):
    r = roi.replace('networks.', '').replace('atlas.', '').replace('CerebraA_', '')
    r = (r.replace('DefaultMode.', 'DMN.').replace('Salience.', 'Sal.')
           .replace('DorsalAttention.', 'DAN.').replace('FrontoParietal.', 'CEN.'))
    r = r.split(' (')[0]
    for a, b in [('Hippocampus', 'Hipp'), ('Amygdala', 'Amyg'), ('Thalamus', 'Thal'),
                 ('Caudate', 'Caud'), ('Putamen', 'Put'), ('Pallidum', 'Pall'),
                 ('Brain-Stem', 'BStem'), ('Cerebellum', 'Cereb')]:
        r = r.replace(a, b)
    return r.strip()

# ------------------------------------------------------ panels d/e helpers ----
# Compute both graph metrics at ROI level first, then average within the ten
# network categories. This preserves the original Figure 1 d/e analysis logic
# while keeping the composite script self-contained.
CATS = NETWORK_ORDER


def _hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) +
                  (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (0.0 if sp == 0 else
            (np.mean(a) - np.mean(b)) / sp *
            (1 - 3 / (4 * (n1 + n2) - 9)))


def _bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    order_ = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order_[rank]
        prev = min(prev, ps[i] * n / (rank + 1))
        q[i] = prev
    return q


def _perm_p(g1, g2, n_perms=10000):
    n1 = len(g1)
    pool = np.concatenate((g1, g2)).copy()
    obs = abs(np.mean(g1) - np.mean(g2))
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perms):
        rng.shuffle(pool)
        if abs(np.mean(pool[:n1]) - np.mean(pool[n1:])) >= obs:
            count += 1
    return (count + 1) / (n_perms + 1)


def ec_table(dn_, dp_, roi_list_, roi_cat_):
    """ROI-level eigenvector centrality -> category means -> Hedges' g."""
    def ec(row):
        G = nx.Graph()
        G.add_nodes_from(roi_list_)
        for feat, w in row.items():
            if w > 0:
                p = feat.split('_vs_')
                if len(p) == 2:
                    G.add_edge(p[0], p[1], weight=w)
        try:
            return nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except Exception:
            return nx.degree_centrality(G)

    en = pd.DataFrame([ec(r) for _, r in dn_.iterrows()])
    ep = pd.DataFrame([ec(r) for _, r in dp_.iterrows()])

    def agg(d):
        return pd.DataFrame({
            c: d[[r for r in roi_list_ if roi_cat_[r] == c]].mean(axis=1)
            for c in CATS
        })

    cn, cp = agg(en), agg(ep)
    rows_ = [
        {'net': c,
         'g': _hedges_g(cp[c].values, cn[c].values),
         'p': _perm_p(cp[c].values, cn[c].values)}
        for c in CATS
    ]
    tab = pd.DataFrame(rows_)
    tab['q'] = _bh(tab['p'].values)
    return tab


def pc_table(dn_, dp_, roi_list_, roi_cat_):
    """ROI-level participation coefficient -> category means -> Hedges' g."""
    def pc_row(row):
        by_mod = {r: {c: 0.0 for c in CATS} for r in roi_list_}
        total = {r: 0.0 for r in roi_list_}
        for feat, val in row.items():
            if val <= 0:
                continue
            p = feat.split('_vs_')
            if len(p) != 2:
                continue
            u, v = p
            cu, cv = roi_cat_.get(u, 'Other'), roi_cat_.get(v, 'Other')
            if 'Other' in (cu, cv):
                continue
            by_mod[u][cv] += val
            total[u] += val
            by_mod[v][cu] += val
            total[v] += val
        return {
            r: (0.0 if total[r] == 0 else
                1.0 - sum((by_mod[r][c] / total[r]) ** 2 for c in CATS))
            for r in roi_list_
        }

    def agg(df):
        vals = pd.DataFrame([pc_row(r) for _, r in df.iterrows()],
                            index=df.index, columns=roi_list_)
        return pd.DataFrame({
            c: vals[[r for r in roi_list_ if roi_cat_[r] == c]].mean(axis=1)
            for c in CATS
        })

    cn, cp = agg(dn_), agg(dp_)
    rows_ = []
    for c in CATS:
        _, p = mannwhitneyu(cp[c].values, cn[c].values, alternative='two-sided')
        rows_.append({'net': c,
                      'g': _hedges_g(cp[c].values, cn[c].values),
                      'p': p})
    tab = pd.DataFrame(rows_)
    tab['q'] = _bh(tab['p'].values)
    return tab


def draw_effect_panel(ax, tab, title, xlim=None, show_ylabels=True):
    """Horizontal Hedges' g bars in canonical network order."""
    y = np.arange(len(CATS))[::-1]
    tab = tab.set_index('net').loc[CATS].reset_index()
    ax.barh(y, tab['g'], color=[net_color(c) for c in tab['net']],
            height=0.68, edgecolor='none')
    for yi, (_, r) in zip(y, tab.iterrows()):
        star = '**' if r['q'] < 0.05 else ('*' if r['p'] < 0.05 else '')
        if star:
            off = -0.05 if r['g'] < 0 else 0.05
            ax.text(r['g'] + off, yi, star, va='center',
                    ha='right' if r['g'] < 0 else 'left',
                    fontsize=FONT['annotation'], color='#333')
    ax.axvline(0, color='#333', lw=0.9)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([short(c) for c in CATS], fontsize=FONT['tick'])
        for lab in ax.get_yticklabels():
            if lab.get_text() == 'Rew':
                lab.set_fontweight('bold')
                lab.set_color(net_color('Reward'))
    else:
        ax.set_yticklabels([])
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=FONT['panel_title'], loc='left', pad=6)
    ax.set_xlabel("Hedges' $g$", fontsize=FONT['axis_label'])
    ax.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------- panel a
# Whole-brain two-sample t-test montage, dropped in as a raster.
ax_a = fig.add_axes(BOX['a'])
ax_a.set_xticks([]); ax_a.set_yticks([])
for s in ax_a.spines.values():
    s.set_visible(False)
_mont = plt.imread(f'{RASTER}/Fig1a.png')
ax_a.imshow(_mont, interpolation='lanczos', aspect='equal')

# ------------------------------------------------------------------- panel b
# Node x node connectivity-profile similarity in patients, Ward-linkage order.
prof = np.zeros((len(roi_list), len(roi_list)))
idx = {r: i for i, r in enumerate(roi_list)}
for col in common:
    a, b = col.split('_vs_')
    v = dp[col].mean()
    prof[idx[a], idx[b]] = v
    prof[idx[b], idx[a]] = v
np.fill_diagonal(prof, 1.0)
sim = np.corrcoef(prof)
dist = 1 - sim
np.fill_diagonal(dist, 0.0)
Z = linkage(squareform(dist, checks=False), method='ward')
dend = dendrogram(Z, no_plot=True)
order = dend['leaves']

bx, by, bw, bh = BOX['b']
ax_b = fig.add_axes([bx, by, bw * 0.86, bh * 0.90])
im = ax_b.imshow(sim[np.ix_(order, order)], cmap='RdBu_r', vmin=-1, vmax=1,
                 aspect='auto')
ax_b.set_xticks([]); ax_b.set_yticks([])
for sp in ax_b.spines.values():
    sp.set_linewidth(0.6); sp.set_color('#999')

# category colour strip on the left edge
for k, roi_i in enumerate(order):
    c = roi_cat[roi_list[roi_i]]
    ax_b.add_patch(mpatches.Rectangle((-2.4, k - 0.5), 2.0, 1.0,
                                      color=net_color(c) if c != 'Other' else '#DDD',
                                      clip_on=False, lw=0))
ax_b.set_xlim(-2.5, len(order) - 0.5)

# 45-degree category labels above the matrix, one per contiguous run
runs, prev, start_i = [], None, 0
for k, roi_i in enumerate(order):
    c = roi_cat[roi_list[roi_i]]
    if c != prev:
        if prev is not None:
            runs.append((prev, start_i, k - 1))
        prev, start_i = c, k
runs.append((prev, start_i, len(order) - 1))
for c, i0, i1 in runs:
    if c == 'Other' or (i1 - i0) < 1:
        continue
    ax_b.text((i0 + i1) / 2, -3.0, short(c), rotation=45, rotation_mode='anchor',
              ha='left', va='bottom', fontsize=fs('stat_inset', -0.5),
              fontweight='bold', color=net_color(c), clip_on=False)
    ax_b.add_patch(mpatches.Rectangle((i0 - 0.5, -2.2), (i1 - i0) + 1.0, 1.5,
                                      color=net_color(c), clip_on=False, lw=0))

cax = fig.add_axes([bx + bw * 0.885, by + bh * 0.20, 0.012, bh * 0.44])
cb = fig.colorbar(im, cax=cax)
cb.set_label('profile similarity ($r$)', fontsize=FONT['stat_inset'])
cb.ax.tick_params(labelsize=fs('stat_inset', -0.5), width=0.6, length=2)
cb.outline.set_linewidth(0.5)

# ------------------------------------------------------------------- panel c
ax_c = fig.add_axes(BOX['c'])
ax_c.set_aspect('equal'); ax_c.axis('off')

rows = []
for col in common:
    pv, nv = dp[col].values, dn[col].values
    u, p = mannwhitneyu(pv, nv, alternative='two-sided')
    n1, n2 = len(pv), len(nv)
    sp = np.sqrt(((n1 - 1) * np.var(pv, ddof=1) + (n2 - 1) * np.var(nv, ddof=1)) / (n1 + n2 - 2))
    g = 0.0 if sp == 0 else (np.mean(pv) - np.mean(nv)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
    rows.append({'edge': col, 'p': p, 'g': g})
edges = pd.DataFrame(rows)
sig = edges[edges['p'] < 0.05]

df_roi = pd.DataFrame({'ROI': roi_list})
df_roi['Category'] = df_roi['ROI'].map(roi_cat)
df_roi['Order'] = df_roi['Category'].apply(
    lambda x: NETWORK_ORDER.index(x) if x in NETWORK_ORDER else 99)
df_roi = df_roi.sort_values(['Order', 'ROI']).reset_index(drop=True)
node_pos = {r: i for i, r in enumerate(df_roi['ROI'])}
N = len(df_roi)
radius = 10.0
ang = np.array([np.pi / 2 - i * 2 * np.pi / N for i in range(N)])
nx_, ny_ = radius * np.cos(ang), radius * np.sin(ang)

deg = {r: 0 for r in roi_list}
for _, r in sig.iterrows():
    a, b = r['edge'].split('_vs_')
    deg[a] += 1; deg[b] += 1

for _, r in sig.iterrows():
    a, b = r['edge'].split('_vs_')
    i, j = node_pos[a], node_pos[b]
    col = '#C0504D' if r['g'] > 0 else '#4A7EBB'
    lw = 0.18 + min(abs(r['g']), 2.0) * 0.5
    alpha = 0.14 + min(abs(r['g']), 2.0) * 0.16
    verts = np.array([[nx_[i], ny_[i]], [0, 0], [nx_[j], ny_[j]]])
    t = np.linspace(0, 1, 40)[:, None]
    curve = (1 - t) ** 2 * verts[0] + 2 * (1 - t) * t * verts[1] * 0.35 + t ** 2 * verts[2]
    ax_c.plot(curve[:, 0], curve[:, 1], color=col, lw=lw, alpha=alpha, zorder=2,
              solid_capstyle='round')

for i, r in df_roi.iterrows():
    c = net_color(r['Category'])
    sz = 8 + deg[r['ROI']] * 1.5
    ax_c.scatter(nx_[i], ny_[i], s=sz, color=c, zorder=10,
                 edgecolors='white', linewidths=0.7)
    a_deg = np.degrees(ang[i]) % 360
    if 90 < a_deg < 270:
        ha, rot = 'right', a_deg - 180
    else:
        ha, rot = 'left', a_deg
    ax_c.text(nx_[i] * 1.06, ny_[i] * 1.06, shorten(r['ROI']), fontsize=FONT['micro_small'],
              ha=ha, va='center', rotation=rot, rotation_mode='anchor',
              color=c, zorder=11)

start = 0
for cat in NETWORK_ORDER:
    n = int((df_roi['Category'] == cat).sum())
    if n == 0:
        continue
    th1 = np.degrees(ang[start]) + (360 / N) / 2
    th2 = np.degrees(ang[start + n - 1]) - (360 / N) / 2
    if th2 > th1:
        th1, th2 = th2, th1
    ax_c.add_patch(mpatches.Wedge((0, 0), radius * 1.30, th2, th1, width=0.35,
                                  color=net_color(cat), alpha=0.55, zorder=1))
    mid = np.radians((th1 + th2) / 2)
    ax_c.text(radius * 1.40 * np.cos(mid), radius * 1.40 * np.sin(mid), short(cat),
              ha='center', va='center', fontsize=FONT['stat_inset'],
              fontweight='bold', color=net_color(cat))
    start += n

ax_c.set_xlim(-radius * 1.55, radius * 1.55)
ax_c.set_ylim(-radius * 1.55, radius * 1.55)
ax_c.legend(handles=[
    Line2D([0], [0], color='#C0504D', lw=1.6,
           label=f"patient > control ({int((sig['g'] > 0).sum())})"),
    Line2D([0], [0], color='#4A7EBB', lw=1.6,
           label=f"patient < control ({int((sig['g'] < 0).sum())})")],
    loc='lower center', bbox_to_anchor=(0.5, -0.055), ncol=2,
    fontsize=FONT['stat_inset'], frameon=False)

# ----------------------------------------------------------------- panels d,e
t_ec = ec_table(dn, dp, roi_list, roi_cat)
t_pc = pc_table(dn, dp, roi_list, roi_cat)
xlim = (min(t_ec['g'].min(), t_pc['g'].min()) - 0.42,
        max(t_ec['g'].max(), t_pc['g'].max()) + 0.42)
ax_d = fig.add_axes(BOX['d'])
draw_effect_panel(ax_d, t_ec, 'Eigenvector centrality', xlim=xlim)
ax_d.title.set_fontsize(fs('annotation', 0.5)); ax_d.title.set_color('#333')
ax_e = fig.add_axes(BOX['e'])
draw_effect_panel(ax_e, t_pc, 'Participation coefficient', xlim=xlim)
ax_e.title.set_fontsize(fs('annotation', 0.5)); ax_e.title.set_color('#333')
ax_d.set_xlabel('')
ax_d.set_xticklabels([])
ax_e.set_xlabel("Hedges' $g$ (patient − control)", fontsize=FONT['annotation'])

# ------------------------------------------------------------------- panel f
# 14 measurement columns arranged in 5 timepoint blocks, as in fig_vas_total.py
GROUPS = [2, 7, 2, 2, 1]
GAP = 1.8
SHORT = ["BL\npre", "BL\npost", "Pre-\nreg", "Post-\nreg", "1st", "2nd", "3rd",
         "4th", "5th", "7d\npre", "7d\npost", "3M\npre", "3M\npost", "6M\npre"]
BLOCKS = ["Baseline", "Sonication day\n(LIFU)", "Day 7", "3\nmonths", "6\nmonths"]

vas = pd.read_csv(f'{P}/subject_VAS_final.csv')
num = vas.select_dtypes(include=[np.number])
V = num.iloc[:, :14].values.astype(float)

xpos, cur = [], 0.0
for gi, n in enumerate(GROUPS):
    for k in range(n):
        xpos.append(cur + k)
    cur += n - 1 + GAP
xpos = np.array(xpos)

ax_f = fig.add_axes(BOX['f'])
for row in V:
    ax_f.plot(xpos, row, color='#CDD1D5', lw=0.45, alpha=0.85, zorder=1)
mV = np.nanmean(V, axis=0)
sV = np.nanstd(V, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(V), axis=0))
ax_f.errorbar(xpos, mV, yerr=sV, color='#1A1A1A', lw=1.4, marker='o', ms=2.8,
              capsize=1.8, zorder=5)
ax_f.set_xticks(xpos)
ax_f.set_xticklabels(SHORT, fontsize=FONT['micro_small'])
ax_f.tick_params(axis='x', pad=1, length=2)
start = 0
for n, lab in zip(GROUPS, BLOCKS):
    mid = xpos[start:start + n].mean()
    ax_f.text(mid, -0.27, lab, transform=ax_f.get_xaxis_transform(),
              ha='center', va='top', fontsize=FONT['micro'], fontweight='bold', color='#444')
    start += n
# paired Wilcoxon on the pre-cue (uncued) columns, as in fig_vas_total.py
def _sig(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
IDX = {'BL': 0, 'PREREG': 2, 'D7': 9, 'M3': 11, 'M6': 13}
def _wil(i, j):
    a, b = V[:, i], V[:, j]
    ok = ~np.isnan(a) & ~np.isnan(b)
    try:
        return wilcoxon(a[ok], b[ok])[1]
    except Exception:
        return np.nan
ytop = np.nanmax(V) * 1.02
step = np.nanmax(V) * 0.115
for k, (i, j) in enumerate([
        (IDX['PREREG'], IDX['D7']),
        (IDX['PREREG'], IDX['M3']),
        (IDX['PREREG'], IDX['M6'])]):
    p = _wil(i, j)
    if np.isnan(p):
        continue
    yy = ytop + step * k
    ax_f.plot([xpos[i], xpos[i], xpos[j], xpos[j]],
              [yy - step * 0.16, yy, yy, yy - step * 0.16],
              color='#444', lw=0.7, clip_on=False)
    ax_f.text((xpos[i] + xpos[j]) / 2, yy + step * 0.04, _sig(p), ha='center',
              va='bottom', fontsize=FONT['micro'], color='#333', clip_on=False)
ax_f.set_ylabel('Craving VAS', fontsize=FONT['axis_label'])
ax_f.set_ylim(min(-0.4, np.nanmin(V) - 0.4), ytop + step * 2.6)
ax_f.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------- panel g
# HAM-D / HAM-A at fixed column indices (as in fig_clinical_scales.py). The
# name -> subject map lives in an external file so no identifiers sit in code.
HAM_X = ['Baseline', '7d', '30d', '90d', '180d']

# Clinician-rated scales come from the de-identified export, which carries the
# same 15 patients and the same values as the original clinical workbook; the
# name-to-subject map is therefore no longer needed.
_ham = pd.read_csv(f'{P}/clinical_scales_deident.csv').sort_values('subject')
hamd = _ham[[f'HAMD_{t}' for t in HAM_X]].to_numpy(float)
hama = _ham[[f'HAMA_{t}' for t in HAM_X]].to_numpy(float)

xg = np.arange(len(HAM_X))

def _sig_star(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

for key, M, colr, nm in [('g1', hamd, '#3B6FA0', 'HAM-D (depression)'),
                         ('g2', hama, '#B4544E', 'HAM-A (anxiety)')]:
    axg = fig.add_axes(BOX[key])
    n_ok = np.sum(~np.isnan(M), axis=0)
    mm = np.nanmean(M, axis=0)
    ss = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n_ok)
    ci = 1.96 * ss
    axg.fill_between(xg, mm - ci, mm + ci, color=colr, alpha=0.20, lw=0)
    axg.plot(xg, mm, color=colr, lw=1.5, marker='o', ms=3.0, zorder=5)
    top = np.nanmax(mm + ci)
    for t in range(1, M.shape[1]):
        a, b = M[:, 0], M[:, t]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 3:
            continue
        try:
            p = wilcoxon(a[ok], b[ok])[1]
        except Exception:
            continue
        st = _sig_star(p)
        if st:
            axg.text(t, mm[t] + ci[t] + top * 0.06, st, ha='center', va='bottom',
                     fontsize=FONT['annotation'], color=colr, fontweight='bold')
    axg.set_xticks(xg)
    axg.set_xticklabels(HAM_X, fontsize=FONT['micro'], rotation=35, ha='right')
    axg.tick_params(axis='x', pad=1, length=2)
    axg.set_ylim(0, top * 1.42)
    axg.set_title(nm, fontsize=FONT['annotation'], color='#333', loc='left', pad=4)
    axg.set_ylabel('Score', fontsize=FONT['axis_label'])
    axg.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------- panel h
treat = {'sub_04': '2025.05.30', 'sub_07': '2025.07.25', 'sub_09': '2025.08.22',
         'sub_14': '2025.10.16', 'sub_17': '2025.12.03'}
rel_days = [77, 105, 113, 77, 74]
n_tot, follow = 15, 180
ax_h = fig.add_axes(BOX['h'])
times = sorted(rel_days)
surv, at_risk = [1.0], n_tot
xs, ys = [0], [1.0]
for t in times:
    surv_prev = ys[-1]
    s = surv_prev * (1 - 1 / at_risk)
    xs += [t, t]; ys += [surv_prev, s]
    at_risk -= 1
xs.append(follow); ys.append(ys[-1])
ax_h.step(xs, ys, where='post', color='#222', lw=1.5)
ax_h.set_ylim(0, 1.02); ax_h.set_xlim(0, follow)
ax_h.set_xticks([0, 90, 180])
ax_h.set_xticklabels(['0', '3M', '6M'], fontsize=FONT['tick'])
ax_h.set_xlabel('Time since LIFU', fontsize=FONT['axis_label'])
ax_h.set_ylabel('Relapse-free', fontsize=FONT['axis_label'])
ax_h.spines[['top', 'right']].set_visible(False)
ax_h.text(0.95, 0.10, f'{ys[-1]*100:.0f}% at 6M\n5/15 relapsed',
          transform=ax_h.transAxes, ha='right', va='bottom',
          fontsize=fs('stat_inset', -0.5), color='#333')


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure1.pdf',
    OUTPUT_DIR / 'Figure1_preview.png', preview_dpi=200
)

plt.close()
print('saved Figure1.pdf / Figure1_preview.png')
print(f"  panel c: {len(sig)}/{len(edges)} edges p<0.05 "
      f"({int((sig['g'] > 0).sum())} up, {int((sig['g'] < 0).sum())} down)")
print(f"  panel d: EC Reward g={float(t_ec[t_ec.net=='Reward'].g.iloc[0]):+.3f}")
print(f"  panel e: PC Reward g={float(t_pc[t_pc.net=='Reward'].g.iloc[0]):+.3f}")
