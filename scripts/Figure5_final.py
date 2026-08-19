"""
Figure5_composite.py — assembles Figure 5 as a single vector PDF.

Layout follows MUD_main_Figs_v7.pptx (7.5 x 10.83 in):

    a  label transfer with the underlying class probabilities (hybrid flow)
    b  baseline RS-FC of the nine Reward-crossing pairs, by relapse status
    c  NAc coupling index after the spiraling reorientation
    d  within-subject permutation test (ROI level, spiraling-corrected)
    e  baseline cue-evoked co-reactivity, non-relapse vs relapse
    f  change in co-reactivity at day 7, non-relapse vs relapse
    g  coupling trajectory (left) and baseline strip (right)
    h  delta-R for Reward-crossing vs non-Reward pairs, by relapse status

Panel logic follows the original scripts:
    a  fig_hybrid_flow_v4.py                 b  fig_rsfc_baseline_strip_relapse.py
    c  fig_rsfc_baseline_strip_flipcomp_comparison.py (Strategy C panel)
    d  fig_rsfc_perm_comparison.py (spiraling-corrected panel)
    e  fig_coact_baseline_heatmap_relapse_NR_R.py
    f  fig_heatmap_relapse_upper.py
    g  fig_baseline_predictor_relapse.py (panels swapped)
    h  fig_strip_relapse.py
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import (NETWORK_ORDER, net_color, short, CLUSTER_COLORS,
                       CLUSTER_NAMES, apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_W, FIG_H = 7.5, 7.05
C_NON, C_REL = '#B0BEC5', '#78909C'      # outcome palette, as agreed


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


BOX = {
    'a': rect(0.56, 0.25, 6.55, 2.05),
    'b': rect(0.55, 2.63, 2.75, 1.18),
    'c': rect(3.85, 2.63, 1.05, 1.28),
    'd': rect(5.42, 2.63, 1.85, 1.28),
    'e': rect(0.55, 4.18, 2.95, 1.03),
    'f': rect(4.20, 4.18, 2.95, 1.03),
    'g': rect(0.55, 5.55, 3.00, 0.95),
    'h': rect(4.20, 5.55, 3.00, 0.95),
}
LETTER = {'a': (0.30, 0.11), 'b': (0.30, 2.49), 'c': (3.62, 2.49),
          'd': (5.18, 2.49), 'e': (0.30, 4.04), 'f': (3.95, 4.04),
          'g': (0.30, 5.41), 'h': (3.95, 5.41)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# --------------------------------------------------------------- shared data
nl = pd.read_csv(f'{P}/normal_subject_feature_matrix_clean.csv', index_col=0)
bl = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
p7 = pd.read_csv(f'{P}/post_7d_subject_feature_matrix_new.csv', index_col=0)
p3 = pd.read_csv(f'{P}/post_3M_subject_feature_matrix_new.csv', index_col=0)
for df in (nl, bl, p7, p3):
    df.columns = df.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
    df.index = df.index.astype(str)
cl = pd.read_csv(f'{P}/subject_cluster_new.csv', index_col=0)
cl.index = cl.index.astype(str)

RELAPSERS = {'sub_04', 'sub_07', 'sub_09', 'sub_14', 'sub_17'}
subs = [s for s in bl.index if s in cl.index]
feat = sorted(set(bl.columns) & set(nl.columns))
bl, p7, p3 = bl.loc[subs, feat], p7.loc[subs, feat], p3.loc[subs, feat]
nl_f = nl[feat]
labels = cl.loc[subs, 'Cluster'].astype(int).values
is_rel = np.array([s.replace('.mat', '') in RELAPSERS for s in subs])

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
rois = set()
for col in feat:
    pp = col.split('_vs_')
    if len(pp) == 2:
        rois.add(pp[0].strip()); rois.add(pp[1].strip())
roi_list = sorted(rois)
roi_cat = {r: getcat(r) for r in roi_list}


def net_pair_matrix(df):
    """subject x (network pair) mean connectivity."""
    out = {}
    for i, a in enumerate(CATS):
        for b in CATS[i:]:
            cols = []
            for col in feat:
                x, y = col.split('_vs_')
                ca, cb = roi_cat.get(x.strip(), 'Other'), roi_cat.get(y.strip(), 'Other')
                if {ca, cb} == {a, b} or (a == b and ca == cb == a):
                    cols.append(col)
            if cols:
                out[f'{a}|{b}'] = df[cols].mean(axis=1).values
    return pd.DataFrame(out, index=df.index)


bl_np = net_pair_matrix(bl)
REW_PAIRS = [c for c in bl_np.columns if 'Reward' in c.split('|') and
             c.split('|')[0] != c.split('|')[1]]

# ------------------------------------------------------------------ panel a
ax_a = fig.add_axes(BOX['a']); ax_a.axis('off')
ref_X = pd.concat([bl, nl_f], axis=0)
ref_y = np.array([f"C{cl.loc[s, 'Cluster']}" for s in subs] + ['Normal'] * len(nl_f))
pca = PCA(n_components=7, random_state=42)
ref_pca = pca.fit_transform(ref_X.values)
knn = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(ref_pca, ref_y)
classes = list(knn.classes_)
PR = {'col0': knn.predict_proba(ref_pca[:len(subs)]),
      'col1': knn.predict_proba(pca.transform(p7.values)),
      'col2': knn.predict_proba(pca.transform(p3.values))}
pred = {k: np.array(classes)[PR[k].argmax(1)] for k in ('col1', 'col2')}

patients = [{'name': s.replace('.mat', '').replace('sub_', ''), 'i': i,
             'col0': f"C{cl.loc[s, 'Cluster']}", 'col1': pred['col1'][i],
             'col2': pred['col2'][i],
             'col3': 'Relapse' if is_rel[i] else 'No relapse', 'rel': bool(is_rel[i])}
            for i, s in enumerate(subs)]

FLOW_COL = {'C1': CLUSTER_COLORS[1], 'C2': CLUSTER_COLORS[2], 'C3': CLUSTER_COLORS[3],
            'Normal': '#4DB6AC', 'No relapse': C_NON, 'Relapse': C_REL}
columns = ['col0', 'col1', 'col2', 'col3']
col_x = [0.09, 0.38, 0.66, 0.92]
col_titles = ['Baseline', 'Day 7\n(transferred)', '3 months\n(transferred)', 'Outcome']
col_order = {'col0': ['C1', 'C2', 'C3'], 'col1': ['C1', 'C2', 'C3', 'Normal'],
             'col2': ['C1', 'C2', 'C3', 'Normal'], 'col3': ['No relapse', 'Relapse']}
unit_h, gap, node_w = 0.052, 0.042, 0.050


def layout(col):
    groups = defaultdict(list)
    for p in patients:
        groups[p[col]].append(p['name'])
    active = [c for c in col_order[col] if groups[c]]
    total = sum(len(groups[c]) * unit_h for c in active) + (len(active) - 1) * gap
    y = 0.5 + total / 2
    out = {}
    for cat in col_order[col]:
        names = groups[cat]
        if not names:
            continue
        h = len(names) * unit_h
        out[cat] = {'ybot': y - h, 'ytop': y, 'n': len(names),
                    'patient_ys': {nm: y - (j + 0.5) * unit_h for j, nm in enumerate(names)}}
        y = y - h - gap
    return out


L = {c: layout(c) for c in columns}
ax_a.set_xlim(-0.02, 1.06); ax_a.set_ylim(-0.02, 1.06)


def band(x1, y1, x2, y2, color, alpha):
    t = np.linspace(0, 1, 50); sm = 3 * t ** 2 - 2 * t ** 3
    xs = x1 + node_w / 2 + (x2 - x1 - node_w) * t
    half = unit_h * 0.38
    yu = (y1 + half) + ((y2 + half) - (y1 + half)) * sm
    ylo = (y1 - half) + ((y2 - half) - (y1 - half)) * sm
    ax_a.add_patch(plt.Polygon(list(zip(xs, yu)) + list(zip(xs[::-1], ylo[::-1])),
                               facecolor=color, alpha=alpha, edgecolor='none', zorder=2))


for p in patients:
    base_c = FLOW_COL[p['col0']]
    for ci in range(3):
        cs, ct = columns[ci], columns[ci + 1]
        band(col_x[ci], L[cs][p[cs]]['patient_ys'][p['name']],
             col_x[ci + 1], L[ct][p[ct]]['patient_ys'][p['name']],
             base_c, 0.42 if p['rel'] else 0.18)

for ci, col in enumerate(columns):
    x = col_x[ci]
    for cat, info in L[col].items():
        c = FLOW_COL.get(cat, '#999')
        for nm, yc in info['patient_ys'].items():
            pat = next(q for q in patients if q['name'] == nm)
            slot_h = unit_h * 0.86
            y0 = yc - slot_h / 2
            if col in PR:
                pr = PR[col][pat['i']]; x0 = x - node_w / 2
                for k, cls in enumerate(classes):
                    w = pr[k] * node_w
                    if w <= 0:
                        continue
                    ax_a.add_patch(plt.Rectangle((x0, y0), w, slot_h,
                                                 facecolor=FLOW_COL[cls],
                                                 edgecolor='none', zorder=6))
                    x0 += w
            else:
                ax_a.add_patch(plt.Rectangle((x - node_w / 2, y0), node_w, slot_h,
                                             facecolor=c, edgecolor='none', zorder=6))
        label = cat if ci in (0, 3) else (f'{cat}-like' if cat != 'Normal' else 'Normal-like')
        ax_a.text(x, info['ytop'] + 0.012, f'{label} (n = {info["n"]})', ha='center',
                  va='bottom', fontsize=FONT['stat_inset'], fontweight='bold',
                  color=c, zorder=9)
for ci, t in enumerate(col_titles):
    ax_a.text(col_x[ci], 1.03, t, ha='center', va='bottom',
              fontsize=FONT['panel_title'], fontweight='bold', color='#333')
ax_a.legend(handles=[mpatches.Patch(facecolor=CLUSTER_COLORS[1], label=CLUSTER_NAMES[1]),
                     mpatches.Patch(facecolor=CLUSTER_COLORS[2], label=CLUSTER_NAMES[2]),
                     mpatches.Patch(facecolor=CLUSTER_COLORS[3], label=CLUSTER_NAMES[3]),
                     mpatches.Patch(facecolor='#4DB6AC', label='Normal-like'),
                     mpatches.Patch(facecolor=C_REL, label='Relapse'),
                     mpatches.Patch(facecolor=C_NON, label='No relapse')],
            loc='lower center', bbox_to_anchor=(0.5, -0.055), ncol=6,
            fontsize=FONT['stat_inset'], frameon=False, handlelength=1.1)

# ------------------------------------------------------------------ panel b
ax_b = fig.add_axes(BOX['b'])
stats_b = []
for cp in REW_PAIRS:
    v = bl_np[cp].values
    p = stats.mannwhitneyu(v[~is_rel], v[is_rel], alternative='two-sided')[1]
    stats_b.append({'pair': cp, 'p': p})
def _partner(cp):
    a, b = cp.split('|')
    return b if a == 'Reward' else a


stats_b.sort(key=lambda d: NETWORK_ORDER.index(_partner(d['pair'])))
rng = np.random.default_rng(1)
for k, st in enumerate(stats_b):
    v = bl_np[st['pair']].values
    for msk, colr, off in [(~is_rel, C_NON, -0.17), (is_rel, C_REL, 0.17)]:
        ax_b.boxplot([v[msk]], positions=[k + off], widths=0.24, patch_artist=True,
                     showfliers=False, zorder=3,
                     medianprops=dict(color='#333', lw=0.9),
                     whiskerprops=dict(color='#AAA', lw=0.6),
                     capprops=dict(color='#AAA', lw=0.6),
                     boxprops=dict(facecolor=colr, alpha=0.45, edgecolor=colr, lw=0.8))
        xx = k + off + rng.normal(0, 0.04, msk.sum())
        ax_b.scatter(xx, v[msk], s=8, color=colr, alpha=0.95,
                     edgecolors='white', linewidths=0.3, zorder=5)
    ax_b.text(k, ax_b.get_ylim()[0], '', ha='center')
ax_b.axhline(0, color='#CCC', lw=0.6, ls=':')
ax_b.set_xlim(-0.6, len(stats_b) - 0.4)
ax_b.set_xticks(range(len(stats_b)))
ax_b.set_xticklabels([short(_partner(s['pair'])) for s in stats_b],
                     fontsize=FONT['micro'], rotation=40, ha='right')
for _t in ax_b.get_xticklabels():
    if _t.get_text() == 'Comp':
        _t.set_color(net_color('Compulsion')); _t.set_fontweight('bold')
ax_b.set_ylabel('baseline FC (Fisher $z$)', fontsize=FONT['axis_label'])
ax_b.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_b.tick_params(axis='x', length=2, pad=1)
ax_b.set_title('Reward-crossing pairs (network order)', fontsize=FONT['panel_title'],
               color='#333', loc='left', pad=4)
ax_b.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NON,
                            markersize=4, label='non-relapse'),
                     Line2D([0], [0], marker='o', color='w', markerfacecolor=C_REL,
                            markersize=4, label='relapse')],
            fontsize=fs('stat_inset', -0.5), frameon=False, loc='upper right',
            handletextpad=0.2, borderpad=0.1)
ax_b.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel c
ax_c = fig.add_axes(BOX['c'])
idx_A = bl_np[REW_PAIRS].mean(axis=1).values
flip = np.array([-1 if 'Compulsion' in cp.split('|') else 1 for cp in REW_PAIRS])
idx_C = (bl_np[REW_PAIRS].values * flip).mean(axis=1)
p_A = stats.mannwhitneyu(idx_A[~is_rel], idx_A[is_rel], alternative='two-sided')[1]
p_C = stats.mannwhitneyu(idx_C[~is_rel], idx_C[is_rel], alternative='two-sided')[1]
for k, (msk, colr, lab) in enumerate([(~is_rel, C_NON, f'non-relapse\n(n = {int((~is_rel).sum())})'),
                                      (is_rel, C_REL, f'relapse\n(n = {int(is_rel.sum())})')]):
    ax_c.boxplot([idx_C[msk]], positions=[k], widths=0.44, patch_artist=True,
                 showfliers=False, zorder=3,
                 medianprops=dict(color='#333', lw=1.0),
                 whiskerprops=dict(color='#AAA', lw=0.6),
                 capprops=dict(color='#AAA', lw=0.6),
                 boxprops=dict(facecolor=colr, alpha=0.45, edgecolor=colr, lw=0.9))
    ax_c.scatter(k + rng.normal(0, 0.06, msk.sum()), idx_C[msk], s=12, color=colr,
                 alpha=0.95, edgecolors='white', linewidths=0.35, zorder=5)
ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(['non-\nrelapse', 'relapse'],
                                              fontsize=FONT['micro'])
ax_c.set_xlim(-0.55, 1.55)
ax_c.set_ylabel('NAc coupling index', fontsize=FONT['axis_label'])
ax_c.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_c.tick_params(axis='x', length=2, pad=1)
ax_c.set_title(f'spiraling-corrected\nMWU $p$ = {p_C:.4f}', fontsize=FONT['panel_title'],
               color='#333', loc='left', pad=4)
ax_c.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel d
ax_d = fig.add_axes(BOX['d'])
nac_rois = [r for r in roi_list if roi_cat[r] == 'Reward']
comp_rois = set(r for r in roi_list if roi_cat[r] == 'Compulsion')


def crossing_index(sub_i, seed_rois, do_flip):
    vals = []
    for col in feat:
        a, b = [x.strip() for x in col.split('_vs_')]
        in_a, in_b = a in seed_rois, b in seed_rois
        if in_a == in_b:
            continue
        other = b if in_a else a
        v = bl.iloc[sub_i][col]
        if do_flip and other in comp_rois:
            v = -v
        vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


# vectorised: build an (n_subj x n_roi x n_roi) tensor once, then index seeds
n_roi = len(roi_list)
r_idx = {r: i for i, r in enumerate(roi_list)}
T = np.zeros((len(subs), n_roi, n_roi))
comp_mask = np.array([roi_cat[r] == 'Compulsion' for r in roi_list])
for col in feat:
    a, b = [x.strip() for x in col.split('_vs_')]
    ia, ib = r_idx[a], r_idx[b]
    v = bl[col].values
    T[:, ia, ib] = v
    T[:, ib, ia] = v

def crossing_from_mask(seed_mask, do_flip=True):
    """mean of seed x non-seed edges, caudate partners sign-flipped."""
    other = ~seed_mask
    sub = T[:, seed_mask][:, :, other]                    # (n_subj, n_seed, n_other)
    if do_flip:
        sgn = np.where(comp_mask[other], -1.0, 1.0)
        sub = sub * sgn[None, None, :]
    return sub.reshape(len(subs), -1).mean(axis=1)

nac_mask = np.array([roi_cat[r] == 'Reward' for r in roi_list])
obs = crossing_from_mask(nac_mask)

N_PERM = 10000
rng_p = np.random.default_rng(42)
pool_idx = np.where(~nac_mask)[0]
null = np.zeros((len(subs), N_PERM))
for pi in range(N_PERM):
    m = np.zeros(n_roi, bool)
    m[rng_p.choice(pool_idx, int(nac_mask.sum()), replace=False)] = True
    null[:, pi] = crossing_from_mask(m)
z = (obs - null.mean(axis=1)) / null.std(axis=1, ddof=1)
p_d = stats.mannwhitneyu(z[~is_rel], z[is_rel], alternative='two-sided')[1]
order = np.argsort(z)
# shade the band occupied by relapsers; the two groups do not overlap, so the
# shading also marks where the separation lies
z_r_max, z_nr_min = z[is_rel].max(), z[~is_rel].min()
ax_d.axhspan(z.min() - 0.35, z_r_max, color=C_REL, alpha=0.13, lw=0, zorder=1)
for rank, i in enumerate(order):
    colr = C_REL if is_rel[i] else C_NON
    ax_d.scatter(rank, z[i], s=16, color=colr, alpha=0.95, edgecolors='white',
                 linewidths=0.4, zorder=5)
p_d = stats.mannwhitneyu(z[~is_rel], z[is_rel], alternative='two-sided')[1]
ax_d.text(0.97, 0.05,
          f'complete separation\nrelapse $z$ \u2264 {z_r_max:.2f} < '
          f'{z_nr_min:.2f} \u2264 non-relapse\nMWU $p$ = {p_d:.4f}',
          transform=ax_d.transAxes, va='bottom', ha='right',
          fontsize=fs('stat_inset', -0.5), color='#333', linespacing=1.4,
          bbox=dict(boxstyle='square,pad=0.35', facecolor='white',
                    edgecolor='#D5D8DC', lw=0.6, alpha=0.92))
ax_d.set_ylim(z.min() - 0.35, z.max() + 0.45)
ax_d.set_xticks([])
ax_d.set_ylabel('permutation $z$', fontsize=FONT['axis_label'])
ax_d.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_d.set_title('within-subject permutation', fontsize=FONT['panel_title'],
               color='#333', loc='left', pad=4)
ax_d.spines[['top', 'right']].set_visible(False)

# --------------------------------------------------------- FDCR (panels e-h)
fd = pd.read_csv(f'{P}/FDCR_Final_45ROIs_Mean_Features.csv')
roi_cols_f = [c for c in fd.columns if c.startswith('Cat')]
CAT_NAME = {'Cat01': 'Default', 'Cat02': 'Memory-Emotion', 'Cat03': 'Reward',
            'Cat04': 'Relay', 'Cat05': 'Compulsion', 'Cat06': 'Automaticity',
            'Cat07': 'Attention', 'Cat08': 'Salience', 'Cat09': 'Execution',
            'Cat10': 'Regulation'}
ORDER_KEYS = [k for n in NETWORK_ORDER for k in CAT_NAME if CAT_NAME[k] == n]
rew_pos = [i for i, k in enumerate(ORDER_KEYS) if CAT_NAME[k] == 'Reward'][0]


def coact(tp, subset):
    sub = fd[(fd['Time'] == tp) & (fd['Subject'].isin(subset))].sort_values('Subject')
    nd = np.column_stack([sub[[c for c in roi_cols_f if c.startswith(k)]].mean(axis=1).values
                          for k in ORDER_KEYS])
    return np.corrcoef(nd.T)


all_subj = sorted(fd['Subject'].unique())
rel_subj = [s for s in all_subj if s in RELAPSERS]
non_subj = [s for s in all_subj if s not in RELAPSERS]

for key, tp_pair, ttl in [('e', None, 'baseline co-reactivity'),
                          ('f', ('post7d', 'Pre'), '$\\Delta R$ (day 7 \u2212 baseline)')]:
    bx, by, bw, bh = BOX[key]
    for k, (grp, lab) in enumerate([(non_subj, f'non-relapse (n = {len(non_subj)})'),
                                    (rel_subj, f'relapse (n = {len(rel_subj)})')]):
        axh = fig.add_axes([bx + k * (bw / 2 + 0.012), by, bw / 2 * 0.88, bh])
        M = coact('Pre', grp) if key == 'e' else (coact(tp_pair[0], grp) - coact(tp_pair[1], grp))
        vlim = 1 if key == 'e' else np.nanmax(np.abs(M))
        im = axh.imshow(M, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axh.set_xticks(range(10)); axh.set_yticks(range(10))
        axh.set_xticklabels([short(CAT_NAME[k_]) for k_ in ORDER_KEYS],
                            fontsize=FONT['micro_small'], rotation=60, ha='right')
        axh.set_yticklabels([short(CAT_NAME[k_]) for k_ in ORDER_KEYS] if k == 0 else [],
                            fontsize=FONT['micro_small'])
        for t_ in axh.get_xticklabels():
            if t_.get_text() == 'Rew':
                t_.set_color(net_color('Reward')); t_.set_fontweight('bold')
        for t_ in axh.get_yticklabels():
            if t_.get_text() == 'Rew':
                t_.set_color(net_color('Reward')); t_.set_fontweight('bold')
        axh.add_patch(plt.Rectangle((rew_pos - 0.5, -0.5), 1, 10, fill=False,
                                    edgecolor=net_color('Reward'), lw=0.9, zorder=5))
        axh.add_patch(plt.Rectangle((-0.5, rew_pos - 0.5), 10, 1, fill=False,
                                    edgecolor=net_color('Reward'), lw=0.9, zorder=5))
        axh.tick_params(length=1.5, pad=0.5)
        axh.set_title(lab, fontsize=FONT['stat_inset'], color='#333', pad=3)
        rowmean = np.nanmean(np.delete(M[rew_pos], rew_pos))
        axh.text(0.5, -0.52, f'Reward row mean = {rowmean:+.2f}',
                 transform=axh.transAxes, ha='center', va='top',
                 fontsize=fs('stat_inset', -0.6), color='#555')
    cax = fig.add_axes([bx + bw - 0.004, by + bh * 0.25, 0.008, bh * 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs('stat_inset', -1), width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    fig.text(bx + bw / 2, by + bh + 0.020, ttl, ha='center', va='bottom',
             fontsize=FONT['panel_title'], color='#333')

# ------------------------------------------------------------------ panel g
gx, gy, gw, gh = BOX['g']
# left: coupling trajectory (Reward-crossing solid, non-Reward dashed)
ax_g1 = fig.add_axes([gx, gy, gw / 2 * 0.88, gh])
tps = [('Pre', 'Baseline'), ('post7d', 'Day 7'), ('post3m', '3 months')]
for grp, colr, lab in [(non_subj, C_NON, 'non-relapse'), (rel_subj, C_REL, 'relapse')]:
    rew_tr, non_tr = [], []
    for tp, _ in tps:
        M = coact(tp, grp)
        rew_tr.append(np.mean([M[rew_pos, j] for j in range(10) if j != rew_pos]))
        non_tr.append(np.mean([M[i, j] for i in range(10) for j in range(i + 1, 10)
                               if rew_pos not in (i, j)]))
    ax_g1.plot(range(3), rew_tr, color=colr, lw=1.6, marker='o', ms=3.2, label=f'{lab}, Reward')
    ax_g1.plot(range(3), non_tr, color=colr, lw=1.2, ls='--', marker='s', ms=2.8,
               alpha=0.8, label=f'{lab}, non-Reward')
ax_g1.axhline(0, color='#CCC', lw=0.6, ls=':')
ax_g1.margins(y=0.18)
ax_g1.set_xticks(range(3)); ax_g1.set_xticklabels([t[1] for t in tps], fontsize=FONT['micro'])
ax_g1.set_ylabel('co-reactivity $r$', fontsize=FONT['axis_label'])
ax_g1.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_g1.tick_params(axis='x', length=2, pad=1)
ax_g1.set_title('coupling trajectory', fontsize=FONT['panel_title'], color='#333',
                loc='left', pad=4)
ax_g1.legend(fontsize=fs('stat_inset', -1.4), frameon=False, loc='upper center',
             bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.1,
             labelspacing=0.12, columnspacing=0.6, borderpad=0.05)
ax_g1.spines[['top', 'right']].set_visible(False)

# right: baseline Reward vs non-Reward co-reactivity by outcome
ax_g2 = fig.add_axes([gx + gw / 2 + 0.020, gy, gw / 2 * 0.88, gh])
for k, (grp, colr, lab) in enumerate([(non_subj, C_NON, 'non-relapse'),
                                      (rel_subj, C_REL, 'relapse')]):
    M = coact('Pre', grp)
    rew_v = np.array([M[rew_pos, j] for j in range(10) if j != rew_pos])
    non_v = np.array([M[i, j] for i in range(10) for j in range(i + 1, 10)
                      if rew_pos not in (i, j)])
    p_g = stats.mannwhitneyu(rew_v, non_v, alternative='two-sided')[1]
    for kk, (v, mark) in enumerate([(non_v, 'NR'), (rew_v, 'Rew')]):
        xc = k * 2 + kk
        cc = '#C4C8CC' if kk == 0 else colr
        ax_g2.boxplot([v], positions=[xc], widths=0.5, patch_artist=True,
                      showfliers=False, zorder=3,
                      medianprops=dict(color='#333', lw=0.9),
                      whiskerprops=dict(color='#AAA', lw=0.6),
                      capprops=dict(color='#AAA', lw=0.6),
                      boxprops=dict(facecolor=cc, alpha=0.45, edgecolor=cc, lw=0.8))
        ax_g2.scatter(xc + rng.normal(0, 0.07, len(v)), v, s=6, color=cc,
                      alpha=0.95, edgecolors='none', zorder=5)
    ax_g2.text((k * 2 + 0.5) / 4 + 0.02, 0.96, f'$p$ = {p_g:.1e}',
               transform=ax_g2.transAxes, ha='center', va='top',
               fontsize=fs('stat_inset', -0.8), color='#444')
ax_g2.set_xlim(-0.6, 3.6)
ax_g2.set_xticks([0, 1, 2, 3])
ax_g2.set_xticklabels(['non-\nRew', 'Rew', 'non-\nRew', 'Rew'], fontsize=FONT['micro_small'])
ax_g2.set_ylabel('baseline $r$', fontsize=FONT['axis_label'])
ax_g2.set_ylim(-0.6, 1.18)
ax_g2.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_g2.tick_params(axis='x', length=2, pad=1)
ax_g2.set_title('baseline coupling', fontsize=FONT['panel_title'], color='#333',
                loc='left', pad=4)
ax_g2.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel h
hx, hy, hw, hh = BOX['h']
for k, (tp, lab) in enumerate([('post7d', '$\\Delta R$ (day 7 \u2212 baseline)'),
                               ('post3m', '$\\Delta R$ (3 months \u2212 baseline)')]):
    axh = fig.add_axes([hx + k * (hw / 2 + 0.014), hy, hw / 2 * 0.86, hh])
    for g, (grp, colr, lab2) in enumerate([(non_subj, C_NON, 'non-relapse'),
                                           (rel_subj, C_REL, 'relapse')]):
        D = coact(tp, grp) - coact('Pre', grp)
        rew_v = np.array([D[rew_pos, j] for j in range(10) if j != rew_pos])
        non_v = np.array([D[i, j] for i in range(10) for j in range(i + 1, 10)
                          if rew_pos not in (i, j)])
        p_h = stats.mannwhitneyu(rew_v, non_v, alternative='two-sided')[1]
        for kk, v in enumerate([non_v, rew_v]):
            xc = g * 2 + kk
            cc = '#C4C8CC' if kk == 0 else colr
            axh.boxplot([v], positions=[xc], widths=0.5, patch_artist=True,
                        showfliers=False, zorder=3,
                        medianprops=dict(color='#333', lw=0.9),
                        whiskerprops=dict(color='#AAA', lw=0.6),
                        capprops=dict(color='#AAA', lw=0.6),
                        boxprops=dict(facecolor=cc, alpha=0.45, edgecolor=cc, lw=0.8))
            axh.scatter(xc + rng.normal(0, 0.07, len(v)), v, s=6, color=cc,
                        alpha=0.95, edgecolors='none', zorder=5)
        axh.text((g * 2 + 0.5) / 4 + 0.02, 0.02, f'$p$ = {p_h:.1e}',
                 transform=axh.transAxes, ha='center', va='bottom',
                 fontsize=fs('stat_inset', -0.8), color='#444')
    axh.axhline(0, color='#CCC', lw=0.6, ls=':')
    axh.set_xlim(-0.6, 3.6)
    axh.set_xticks([0, 1, 2, 3])
    axh.set_xticklabels(['non-\nRew', 'Rew', 'non-\nRew', 'Rew'],
                        fontsize=FONT['micro_small'])
    axh.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
    axh.tick_params(axis='x', length=2, pad=1)
    if k == 0:
        axh.set_ylabel('$\\Delta R$', fontsize=FONT['axis_label'])
    axh.set_title(lab, fontsize=FONT['panel_title'], color='#333', loc='left', pad=4)
    axh.spines[['top', 'right']].set_visible(False)


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure5.pdf',
    OUTPUT_DIR / 'Figure5_preview.png', preview_dpi=200
)

plt.close()
print('saved Figure5.pdf / Figure5_preview.png')
print(f'  c: Strategy A p = {p_A:.4f}  |  Strategy C (spiraling) p = {p_C:.4f}')
print(f'  d: MWU p = {p_d:.4f}, complete separation: relapse z <= {z_r_max:.3f} < {z_nr_min:.3f} <= non-relapse')
c2i = classes.index('C2')
print('  a: day-7 C2 probability — ' + ', '.join(
    f"{p['name']}{'*' if p['rel'] else ''} {PR['col1'][p['i']][c2i]:.2f}" for p in patients))
