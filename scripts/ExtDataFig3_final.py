"""
SupplFig3_relapse_rsfc.py — supporting analyses for Fig. 5b-d.

    a, b  network-level resting-state connectivity at baseline, shown separately
          for non-relapsers and relapsers, with the Reward row and column outlined
    c     the baseline NAc coupling index computed two ways: as the plain mean of
          the nine Reward-crossing pairs and after reorienting the
          Reward-Compulsion pair
    d     the same contrast carried through the permutation test, showing the
          per-patient z scores obtained with and without the reorientation

Panels c and d make explicit what Fig. 5c,d assert: the group difference appears
only once the caudate edge is reoriented.
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import apply_base_style, FONT, NET_COLORS, NETWORK_ORDER, fs, nature_figsize, save_nature_figure

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
C_NON, C_REL = '#B0BEC5', '#78909C'

bl = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
bl.columns = bl.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
subs = [str(s) for s in bl.index]
ids = [s.replace('.mat', '') for s in subs]
RELAPSE = {'sub_04', 'sub_07', 'sub_09', 'sub_14', 'sub_17'}
is_rel = np.array([i in RELAPSE for i in ids])
edge_cols = list(bl.columns)


def network_of(roi):
    if 'NAc' in roi: return 'Reward'
    if 'DefaultMode' in roi: return 'Default'
    if 'Salience' in roi: return 'Salience'
    if 'DorsalAttention' in roi: return 'Attention'
    if 'FrontoParietal' in roi: return 'Execution'
    if 'IFG' in roi or 'FOrb' in roi: return 'Regulation'
    if 'PaHC' in roi or 'Hippocampus' in roi or 'Amygdala' in roi: return 'Memory-Emotion'
    if 'Thalamus' in roi or 'Brain-Stem' in roi: return 'Relay'
    if 'Caudate' in roi: return 'Compulsion'
    if 'Putamen' in roi or 'Pallidum' in roi: return 'Automaticity'
    return 'Other'


NETS = list(NETWORK_ORDER)
SHORT = {'Default': 'DMN', 'Salience': 'Sal', 'Attention': 'Att', 'Regulation': 'Reg',
         'Execution': 'Exec', 'Memory-Emotion': 'Mem', 'Reward': 'Rew',
         'Compulsion': 'Comp', 'Automaticity': 'Auto', 'Relay': 'Relay'}

# group edges by the network pair they connect
pair_edges = {}
for c in edge_cols:
    a, b = [x.strip() for x in c.split('_vs_')]
    na, nb = network_of(a), network_of(b)
    if 'Other' in (na, nb):
        continue
    key = tuple(sorted((na, nb), key=NETS.index))
    pair_edges.setdefault(key, []).append(c)


def network_matrix(mask):
    M = np.full((10, 10), np.nan)
    for (na, nb), cols in pair_edges.items():
        v = bl.loc[np.array(subs)[mask], cols].values.mean()
        i, j = NETS.index(na), NETS.index(nb)
        M[i, j] = M[j, i] = v
    return M


def coupling_index(mask_subject_rows, flip):
    """mean of the nine Reward-crossing network pairs, optionally reorienting
    the Reward-Compulsion pair"""
    out = []
    for s in np.array(subs)[mask_subject_rows]:
        vals = []
        for (na, nb), cols in pair_edges.items():
            if 'Reward' not in (na, nb) or na == nb:
                continue
            v = bl.loc[s, cols].values.mean()
            other = nb if na == 'Reward' else na
            if flip and other == 'Compulsion':
                v = -v
            vals.append(v)
        out.append(np.mean(vals))
    return np.array(out)


FIG_W, FIG_H = 7.5, 6.6
fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


for L, (lx, ly) in {'a': (0.48, 0.25), 'b': (4.20, 0.25),
                    'c': (0.48, 3.62), 'd': (4.20, 3.62)}.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------ a, b
def draw_matrix(ax, M, title):
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('#E8E8E8')
    im = ax.imshow(M, cmap=cmap, vmin=-0.35, vmax=0.35, aspect='equal')
    for i in range(10):
        for j in range(10):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f'{M[i, j]:+.2f}'.replace('+0.', '.').replace('-0.', '\u2212.'),
                    ha='center', va='center', fontsize=fs('micro', -0.6),
                    color='white' if abs(M[i, j]) > 0.22 else '#333')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    lab = [SHORT[n] for n in NETS]
    ax.set_xticklabels(lab, rotation=45, ha='right', fontsize=FONT['micro'])
    ax.set_yticklabels(lab, fontsize=FONT['micro'])
    for tk, n in zip(ax.get_xticklabels(), NETS): tk.set_color(NET_COLORS[n])
    for tk, n in zip(ax.get_yticklabels(), NETS): tk.set_color(NET_COLORS[n])
    r = NETS.index('Reward')
    for pos in (r - 0.5, r + 0.5):
        ax.axhline(pos, color='#111', lw=1.1)
        ax.axvline(pos, color='#111', lw=1.1)
    ax.tick_params(length=1.5, pad=1)
    ax.set_title(title, fontsize=FONT['panel_title'], pad=4)
    return im

ax_a = fig.add_axes(rect(0.70, 0.48, 2.70, 2.70))
im = draw_matrix(ax_a, network_matrix(~is_rel), f'Non-relapse (n = {(~is_rel).sum()})')
ax_b = fig.add_axes(rect(4.45, 0.48, 2.70, 2.70))
draw_matrix(ax_b, network_matrix(is_rel), f'Relapse (n = {is_rel.sum()})')
cax = fig.add_axes(rect(7.20, 1.05, 0.10, 1.55))
cb = fig.colorbar(im, cax=cax)
cb.set_label('connectivity (Fisher $z$)', fontsize=FONT['stat_inset'])
cb.ax.tick_params(labelsize=FONT['micro'], length=1.5)

# ------------------------------------------------------------------ c
ax_c = fig.add_axes(rect(0.72, 3.85, 2.55, 2.20))
rng = np.random.default_rng(0)
xs = []
for k, flip in enumerate((False, True)):
    for g, (mask, colr, lab) in enumerate([(~is_rel, C_NON, 'non-relapse'),
                                           (is_rel, C_REL, 'relapse')]):
        v = coupling_index(mask, flip)
        x = k * 1.4 + g * 0.42
        ax_c.scatter(x + rng.normal(0, 0.05, len(v)), v, s=22, color=colr,
                     edgecolors='white', linewidths=0.4, zorder=4)
        ax_c.plot([x - 0.16, x + 0.16], [v.mean()] * 2, color='#111', lw=1.4, zorder=5)
        xs.append(x)
    p = stats.mannwhitneyu(coupling_index(~is_rel, flip),
                           coupling_index(is_rel, flip), alternative='two-sided')[1]
    ax_c.text(k * 1.4 + 0.21, ax_c.get_ylim()[1], f'$p$ = {p:.3f}',
              ha='center', va='bottom', fontsize=FONT['stat_inset'],
              fontweight='bold' if p < 0.05 else 'normal')
    print(f"  c  {'reoriented' if flip else 'plain mean'}: p = {p:.4f}")
ax_c.set_xticks([0.21, 1.61])
ax_c.set_xticklabels(['plain mean', 'reoriented'], fontsize=FONT['tick'])
ax_c.set_ylabel('NAc coupling index', fontsize=FONT['axis_label'])
ax_c.tick_params(axis='y', labelsize=FONT['tick'], length=2)
ax_c.tick_params(axis='x', length=0, pad=3)
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.set_title('Baseline coupling index', fontsize=FONT['panel_title'], loc='left', pad=4)

# ------------------------------------------------------------------ d
N_PERM = 10000
roi_all = sorted({r.strip() for c in edge_cols for r in c.split('_vs_')})
nac_idx = [i for i, r in enumerate(roi_all) if 'NAc' in r]
pool_idx = [i for i, r in enumerate(roi_all) if 'NAc' not in r]
caud = np.array(['Caudate' in r for r in roi_all])
n_roi = len(roi_all)
ridx = {r: i for i, r in enumerate(roi_all)}

# dense subject x roi x roi array, so the permutation is pure numpy
A = np.full((len(subs), n_roi, n_roi), np.nan)
vals = bl.loc[subs].values
for k, c in enumerate(edge_cols):
    a, b = [x.strip() for x in c.split('_vs_')]
    i, j = ridx[a], ridx[b]
    A[:, i, j] = vals[:, k]
    A[:, j, i] = vals[:, k]


def crossing_idx(seed, flip):
    """mean edge value between a seed set and all other regions"""
    mask = np.zeros(n_roi, bool)
    mask[list(seed)] = True
    sub = A[:, mask][:, :, ~mask]                 # subjects x seed x others
    if flip:
        w = np.where(caud[~mask], -1.0, 1.0)
        sub = sub * w[None, None, :]
    return np.nanmean(sub.reshape(len(subs), -1), axis=1)


ax_d = fig.add_axes(rect(4.45, 3.85, 2.55, 2.20))
for k, flip in enumerate((False, True)):
    obs = crossing_idx(nac_idx, flip)
    rngp = np.random.default_rng(42)
    null = np.zeros((len(subs), N_PERM))
    for pi in range(N_PERM):
        pseudo = rngp.choice(pool_idx, len(nac_idx), replace=False)
        null[:, pi] = crossing_idx(pseudo, flip)
    z = (obs - null.mean(1)) / null.std(1, ddof=1)
    p = stats.mannwhitneyu(z[~is_rel], z[is_rel], alternative='two-sided')[1]
    for g, (mask, colr) in enumerate([(~is_rel, C_NON), (is_rel, C_REL)]):
        x = k * 1.4 + g * 0.42
        ax_d.scatter(x + rng.normal(0, 0.05, mask.sum()), z[mask], s=22, color=colr,
                     edgecolors='white', linewidths=0.4, zorder=4)
        ax_d.plot([x - 0.16, x + 0.16], [z[mask].mean()] * 2, color='#111', lw=1.4, zorder=5)
    ax_d.text(k * 1.4 + 0.21, ax_d.get_ylim()[1], f'$p$ = {p:.4f}',
              ha='center', va='bottom', fontsize=FONT['stat_inset'],
              fontweight='bold' if p < 0.05 else 'normal')
    print(f"  d  {'reoriented' if flip else 'plain'}: MWU p = {p:.4f}")
ax_d.axhline(0, color='#CCC', lw=0.7, ls=':')
ax_d.set_xticks([0.21, 1.61])
ax_d.set_xticklabels(['plain mean', 'reoriented'], fontsize=FONT['tick'])
ax_d.set_ylabel('permutation $z$', fontsize=FONT['axis_label'])
ax_d.tick_params(axis='y', labelsize=FONT['tick'], length=2)
ax_d.tick_params(axis='x', length=0, pad=3)
ax_d.spines[['top', 'right']].set_visible(False)
ax_d.set_title(f'Permutation test ({N_PERM:,} iterations)',
               fontsize=FONT['panel_title'], loc='left', pad=4)

fig.legend(handles=[Line2D([0], [0], marker='o', color='w', markersize=5,
                           markerfacecolor=C_NON, label='non-relapse (n = 10)'),
                    Line2D([0], [0], marker='o', color='w', markersize=5,
                           markerfacecolor=C_REL, label='relapse (n = 5)')],
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2,
           fontsize=FONT['legend'], frameon=False)


save_nature_figure(
    fig, OUTPUT_DIR / 'ExtDataFig3.pdf',
    OUTPUT_DIR / 'ExtDataFig3_preview.png', preview_dpi=200,
    submission_jpg_path=OUTPUT_DIR / 'ExtDataFig3.jpg', submission_jpg_dpi=300
)

plt.close()
print('saved ExtDataFig3.pdf / ExtDataFig3_preview.png')
