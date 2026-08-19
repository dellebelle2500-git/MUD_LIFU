"""
Figure3_composite.py — assembles Figure 3 as a single vector PDF.

Layout follows MUD_main_Figs_v7.pptx (7.5 x 10.83 in):

    a  voxelwise FDCR cue reactivity, six right-hemisphere main-effect regions
    b  the same six regions rendered on the MNI template (raster, left blank)
    c  ROI co-reactivity matrices at Pre, Post-7d and Post-3m
    d  network-level co-reactivity change: Pre-vs-Post scatter and Reward
       versus non-Reward delta-R strips

Panel logic follows the original scripts:
    a  fig_fdcr_voxel_trajectory.py
    c  fig_coactivation_matrix_3tp_upper.py
    d  fig_coact_scatter_cat.py  +  fig_reward_contrast_strip.py

Text is embedded as text (fonttype 42); the space reserved for panel b is where
the brain renders are dropped in.
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
from scipy.stats import mannwhitneyu, wilcoxon
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_rasters import place_montage
from helpers.fig_style import (NETWORK_ORDER, net_color, short, apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_W, FIG_H = 7.5, 6.65


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


BOX = {
    'a':  rect(0.52, 0.28, 6.85, 1.22),
    'b':  rect(0.15, 1.72, 7.35, 1.02),
    'c':  rect(0.92, 3.14, 6.20, 1.58),
    'd1': rect(0.50, 5.10, 3.30, 1.30),
    'd2': rect(4.02, 5.10, 3.30, 1.30),
}
LETTER = {'a': (0.22, 0.14), 'b': (0.22, 1.58), 'c': (0.22, 3.00),
          'd': (0.22, 4.96)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------ panel a
raw = pd.read_excel(f'{P}/FDCR_main_effect_activation.xlsx', sheet_name='Sheet2',
                    header=None)
REGIONS = [('dlPFC_R', 'dlPFC (R)', 1), ('IFC_R', 'IFC (R)', 6),
           ('Anterior_PFC_R', 'aPFC (R)', 11), ('OFC_R', 'OFC (R)', 16),
           ('PPC_R', 'PPC (R)', 21), ('MTG_R', 'MTG (R)', 26)]
C_TRAJ = '#333333'                       # common trajectory colour for all six regions

ax_x, ax_y, ax_w, ax_h = BOX['a']
cell_w = ax_w / 6
tp_x = np.arange(3)
ylo, yhi = [], []
data = {}
for key, disp, c0 in REGIONS:
    M = raw.iloc[1:16, c0:c0 + 3].apply(pd.to_numeric, errors='coerce').values.astype(float)
    data[key] = M
    ylo.append(np.nanmin(M)); yhi.append(np.nanmax(M))
YL = (min(ylo) - 0.05, max(yhi) + 0.05)

for k, (key, disp, _) in enumerate(REGIONS):
    M = data[key]
    pw = cell_w * 0.90                      # panel width  (figure fraction)
    ph = pw * (FIG_W / FIG_H)               # same physical size -> square plane
    axa = fig.add_axes([ax_x + k * cell_w, ax_y + ax_h - ph, pw, ph])
    colr = C_TRAJ
    for s in range(M.shape[0]):
        axa.plot(tp_x, M[s], color=colr, alpha=0.13, lw=0.5, zorder=2)
    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
    axa.errorbar(tp_x, mean, yerr=sem, color=colr, lw=1.6, marker='o', ms=3.2,
                 markeredgecolor='white', markeredgewidth=0.7, capsize=2, zorder=6)
    # No statistics are printed in this panel. The omnibus test that selected
    # these six regions was run voxelwise in CONN, and its F and p belong to
    # Supplementary Table 9; the values recoverable here are from ROI-level
    # extracts and do not match the voxelwise result. The orthogonal polynomial
    # decomposition is still used, but only to classify trajectory shape.
    axa.set_ylim(*YL)
    axa.set_xticks(tp_x); axa.set_xticklabels(['BL', '7d', '3M'], fontsize=FONT['micro'])
    axa.tick_params(axis='x', length=2, pad=1)
    axa.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
    axa.set_title(disp, fontsize=FONT['panel_title'], fontweight='bold', color=colr, pad=3)
    if k == 0:
        axa.set_ylabel('Cue reactivity ($\\beta$)', fontsize=FONT['axis_label'])
    else:
        axa.set_yticklabels([])
    axa.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel b
# The same six regions on the MNI template, MRIcroGL renders dropped in.
place_montage(fig, BOX['b'],
              ['Fig3b_dlPFC', 'Fig3b_IFC', 'Fig3b_antPFC',
               'Fig3b_OFC', 'Fig3b_PPC', 'Fig3b_MTG'],
              labels=[lab for _, lab, _ in REGIONS],
              fig_size=(FIG_W, FIG_H), font=FONT['annotation'])

# ---------------------------------------------------------------- FDCR data
fd = pd.read_csv(f'{P}/FDCR_Final_45ROIs_Mean_Features.csv')
roi_cols = [c for c in fd.columns if c.startswith('Cat')]
CAT_KEYS = [f'Cat{i:02d}' for i in range(1, 11)]
CAT_NAME = {'Cat01': 'Default', 'Cat02': 'Memory-Emotion', 'Cat03': 'Reward',
            'Cat04': 'Relay', 'Cat05': 'Compulsion', 'Cat06': 'Automaticity',
            'Cat07': 'Attention', 'Cat08': 'Salience', 'Cat09': 'Execution',
            'Cat10': 'Regulation'}
ORDER_KEYS = [k for n in NETWORK_ORDER for k in CAT_KEYS if CAT_NAME[k] == n]


def cat_matrix(tp):
    sub = fd[fd['Time'] == tp].sort_values('Subject')
    return np.column_stack([sub[[c for c in roi_cols if c.startswith(k)]].mean(axis=1).values
                            for k in ORDER_KEYS])


def roi_matrix(tp):
    sub = fd[fd['Time'] == tp].sort_values('Subject')
    ordered = [c for k in ORDER_KEYS for c in roi_cols if c.startswith(k)]
    return sub[ordered].values, ordered


TPS = [('Pre', 'Baseline'), ('post7d', 'Day 7'), ('post3m', '3 months')]

# ------------------------------------------------------------------ panel c
cx, cy, cw, ch = BOX['c']
sub_cw = cw / 3 * 0.86
_, ordered_rois = roi_matrix('Pre')
roi_cat_idx = [next(i for i, k in enumerate(ORDER_KEYS) if c.startswith(k))
               for c in ordered_rois]
for k, (tp, lab) in enumerate(TPS):
    R, _ = roi_matrix(tp)
    C = np.corrcoef(R.T)
    axc = fig.add_axes([cx + k * (cw / 3), cy, sub_cw, ch])
    im = axc.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#999')
    # category colour strips
    nR = len(ordered_rois)
    for i, ci in enumerate(roi_cat_idx):
        col = net_color(CAT_NAME[ORDER_KEYS[ci]])
        axc.add_patch(plt.Rectangle((i - 0.5, -2.6), 1, 2.0, color=col,
                                    clip_on=False, lw=0))
        axc.add_patch(plt.Rectangle((-2.6, i - 0.5), 2.0, 1, color=col,
                                    clip_on=False, lw=0))
    axc.set_xlim(-2.7, nR - 0.5); axc.set_ylim(nR - 0.5, -2.7)
    if k == 0:
        runs, prev, start_i = [], None, 0
        for i, ci in enumerate(roi_cat_idx):
            if ci != prev:
                if prev is not None:
                    runs.append((prev, start_i, i - 1))
                prev, start_i = ci, i
        runs.append((prev, start_i, len(roi_cat_idx) - 1))
        for ci, i0, i1 in runs:
            nm = CAT_NAME[ORDER_KEYS[ci]]
            axc.text(-3.6, (i0 + i1) / 2, short(nm), ha='right', va='center',
                     fontsize=FONT['stat_inset'], fontweight='bold',
                     color=net_color(nm), clip_on=False)
    axc.set_title(lab, fontsize=FONT['panel_title'], color='#333', pad=4)
cax = fig.add_axes([cx + cw - 0.010, cy + ch * 0.22, 0.010, ch * 0.5])
cb = fig.colorbar(im, cax=cax)
cb.set_label('co-reactivity ($r$)', fontsize=FONT['stat_inset'])
cb.ax.tick_params(labelsize=fs('stat_inset', -0.5), width=0.5, length=2)
cb.outline.set_linewidth(0.5)

# ---------------------------------------------------------------- panel d
rew_i = ORDER_KEYS.index(next(k for k in ORDER_KEYS if CAT_NAME[k] == 'Reward'))


def pair_corr(tp):
    M = cat_matrix(tp)
    R = np.corrcoef(M.T)
    return {(i, j): R[i, j] for i in range(10) for j in range(i + 1, 10)}


pre_c = pair_corr('Pre')
keys = list(pre_c.keys())
is_rew = np.array([rew_i in k for k in keys])

# d1 — Pre vs Post scatter
ax_d1 = fig.add_axes(BOX['d1'])
d1w = BOX['d1'][2] / 2 * 0.86
for k, (tp, lab) in enumerate([('post7d', 'Baseline vs day 7'), ('post3m', 'Baseline vs 3 months')]):
    axd = fig.add_axes([BOX['d1'][0] + k * (BOX['d1'][2] / 2), BOX['d1'][1],
                        d1w, BOX['d1'][3]])
    post_c = pair_corr(tp)
    x = np.array([pre_c[kk] for kk in keys])
    y = np.array([post_c[kk] for kk in keys])
    axd.plot([-0.3, 1.0], [-0.3, 1.0], ls='--', color='#BBB', lw=0.7, zorder=1)
    axd.scatter(x[~is_rew], y[~is_rew], s=9, color='#B7C4CF', alpha=0.65,
                edgecolors='none', zorder=3)
    axd.scatter(x[is_rew], y[is_rew], s=18, color=net_color('Reward'), alpha=0.9,
                edgecolors='white', linewidths=0.5, zorder=5)
    dR_r = (y[is_rew] - x[is_rew]).mean()
    dR_n = (y[~is_rew] - x[~is_rew]).mean()
    p = mannwhitneyu(y[is_rew] - x[is_rew], y[~is_rew] - x[~is_rew],
                     alternative='two-sided')[1]
    axd.text(0.03, 0.97, f'$\\Delta R$ Reward = {dR_r:+.2f}\n'
                         f'$\\Delta R$ non-Reward = {dR_n:+.2f}\n'
                         f'MWU $p$ = {p:.1e}',
             transform=axd.transAxes, va='top', ha='left',
             fontsize=fs('stat_inset', -0.4), color='#444',
             bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                       edgecolor='#D5D8DC', lw=0.6, alpha=0.92))
    axd.set_xlim(-0.3, 1.0); axd.set_ylim(-0.3, 1.0)
    axd.set_xlabel('Baseline ($r$)', fontsize=FONT['axis_label'])
    if k == 0:
        axd.set_ylabel('Post ($r$)', fontsize=FONT['axis_label'])
    axd.set_title(lab, fontsize=FONT['panel_title'], color='#333', pad=3)
    axd.tick_params(labelsize=FONT['micro'], length=2, pad=1)
    axd.spines[['top', 'right']].set_visible(False)
ax_d1.axis('off')

# d2 — delta-R strips
ax_d2 = fig.add_axes(BOX['d2'])
d2w = BOX['d2'][2] / 2 * 0.86
rng = np.random.default_rng(0)
for k, (tp, lab) in enumerate([('post7d', '$\\Delta R$ (day 7 \u2212 baseline)'),
                               ('post3m', '$\\Delta R$ (3 months \u2212 baseline)')]):
    axd = fig.add_axes([BOX['d2'][0] + k * (BOX['d2'][2] / 2), BOX['d2'][1],
                        d2w, BOX['d2'][3]])
    post_c = pair_corr(tp)
    dR = np.array([post_c[kk] - pre_c[kk] for kk in keys])
    for grp, mask, colr, xc in [('Non-Reward', ~is_rew, '#B7C4CF', 0),
                                ('Reward', is_rew, net_color('Reward'), 1)]:
        v = dR[mask]
        jitter = rng.normal(0, 0.055, len(v))
        axd.scatter(xc + jitter, v, s=12 if xc else 8,
                    color=colr, alpha=0.85 if xc else 0.6,
                    edgecolors='white' if xc else 'none', linewidths=0.4,
                    zorder=5 if xc else 3)
        axd.plot([xc - 0.22, xc + 0.22], [v.mean()] * 2, color=colr, lw=1.6,
                 ls='--', zorder=6)
    p = mannwhitneyu(dR[is_rew], dR[~is_rew], alternative='two-sided')[1]
    axd.text(0.03, 0.03, f'MWU $p$ = {p:.1e}', transform=axd.transAxes,
             va='bottom', ha='left', fontsize=fs('stat_inset', -0.4),
             color='#444')
    axd.axhline(0, color='#BBB', lw=0.6, ls=':')
    axd.set_xticks([0, 1])
    axd.set_xticklabels([f'non-Reward\n(n = {int((~is_rew).sum())})',
                         f'Reward\n(n = {int(is_rew.sum())})'],
                        fontsize=FONT['micro'])
    axd.set_xlim(-0.5, 1.5)
    axd.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
    axd.tick_params(axis='x', length=2, pad=1)
    if k == 0:
        axd.set_ylabel('$\\Delta R$', fontsize=FONT['axis_label'])
    axd.set_title(lab, fontsize=FONT['panel_title'], color='#333', pad=3)
    axd.spines[['top', 'right']].set_visible(False)
ax_d2.axis('off')


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure3.pdf',
    OUTPUT_DIR / 'Figure3_preview.png', preview_dpi=200
)

plt.close()

print('saved Figure3.pdf / Figure3_preview.png')
for tp, lab in [('post7d', 'Post-7d'), ('post3m', 'Post-3m')]:
    post_c = pair_corr(tp)
    dR = np.array([post_c[kk] - pre_c[kk] for kk in keys])
    p = mannwhitneyu(dR[is_rew], dR[~is_rew], alternative='two-sided')[1]
    print(f'  {lab}: Reward dR={dR[is_rew].mean():+.3f}, '
          f'non-Reward dR={dR[~is_rew].mean():+.3f}, MWU p={p:.2e}')
for tp, lab in TPS:
    M = cat_matrix(tp); R = np.corrcoef(M.T)
    rew = [R[i, j] for i in range(10) for j in range(i + 1, 10) if rew_i in (i, j)]
    non = [R[i, j] for i in range(10) for j in range(i + 1, 10) if rew_i not in (i, j)]
    print(f'  {lab}: Reward-crossing r={np.mean(rew):+.3f}, non-Reward r={np.mean(non):+.3f}')
