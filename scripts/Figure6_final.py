"""
Figure6_composite.py — assembles Figure 6 as a single vector PDF.

Layout follows MUD_main_Figs_v7.pptx (7.5 x 10.83 in):

    a  baseline cue-evoked co-reactivity by subtype (network level)
    b  Reward-crossing versus non-Reward coupling across the three timepoints
    c  tonic-phasic plane at baseline and day 7, by subtype

Panel logic follows the original scripts:
    a  fig_coact_baseline_heatmap_cluster.py
    b  fig_baseline_predictor_cluster.py (trajectory panel)
    c  fig_tonic_phasic_2tp.py

The plane in c tests the Reward-crossing pairs against the remaining pairs
separately on each axis (Mann-Whitney U), so that a separation confined to the
cue-evoked axis is visible as such. Because the 45 network pairs derive from ten
networks and are not independent, these tests are reported as exploratory.
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
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_style import (NETWORK_ORDER, net_color, short, CLUSTER_COLORS,
                       CLUSTER_NAMES, apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_W, FIG_H = 7.5, 7.05


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


BOX = {
    'a': rect(0.55, 0.38, 4.75, 1.45),
    'b': rect(5.95, 0.38, 1.35, 1.45),
    'c': rect(0.60, 2.20, 6.70, 4.05),
}
LETTER = {'a': (0.34, 0.22), 'b': (5.72, 0.22), 'c': (0.36, 2.04)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------- data ---
fd = pd.read_csv(f'{P}/FDCR_Final_45ROIs_Mean_Features.csv')
roi_cols = [c for c in fd.columns if c.startswith('Cat')]
CAT_NAME = {'Cat01': 'Default', 'Cat02': 'Memory-Emotion', 'Cat03': 'Reward',
            'Cat04': 'Relay', 'Cat05': 'Compulsion', 'Cat06': 'Automaticity',
            'Cat07': 'Attention', 'Cat08': 'Salience', 'Cat09': 'Execution',
            'Cat10': 'Regulation'}
ORDER_KEYS = [k for n in NETWORK_ORDER for k in CAT_NAME if CAT_NAME[k] == n]
REW_POS = [i for i, k in enumerate(ORDER_KEYS) if CAT_NAME[k] == 'Reward'][0]

subj_cluster = {r['Subject']: int(r['Cluster'])
                for _, r in fd[fd['Time'] == 'Pre'].iterrows()}


def coact(tp, cid):
    ids = [s for s, c in subj_cluster.items() if c == cid]
    sub = fd[(fd['Time'] == tp) & (fd['Subject'].isin(ids))].sort_values('Subject')
    nd = np.column_stack([sub[[c for c in roi_cols if c.startswith(k)]].mean(axis=1).values
                          for k in ORDER_KEYS])
    return np.corrcoef(nd.T)


def rew_non(M):
    rew = np.array([M[REW_POS, j] for j in range(10) if j != REW_POS])
    non = np.array([M[i, j] for i in range(10) for j in range(i + 1, 10)
                    if REW_POS not in (i, j)])
    return rew, non


# ------------------------------------------------------------------ panel a
ax_, ay, aw, ah = BOX['a']
cell = aw / 3
for k, cid in enumerate((1, 2, 3)):
    axh = fig.add_axes([ax_ + k * cell, ay, cell * 0.86, ah])
    M = coact('Pre', cid)
    im = axh.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    axh.set_xticks(range(10)); axh.set_yticks(range(10))
    axh.set_xticklabels([short(CAT_NAME[q]) for q in ORDER_KEYS],
                        fontsize=FONT['micro_small'], rotation=60, ha='right')
    axh.set_yticklabels([short(CAT_NAME[q]) for q in ORDER_KEYS]
                        if k == 0 else [], fontsize=FONT['micro_small'])
    for t_ in list(axh.get_xticklabels()) + list(axh.get_yticklabels()):
        if t_.get_text() == 'Rew':
            t_.set_color(net_color('Reward')); t_.set_fontweight('bold')
    axh.add_patch(plt.Rectangle((REW_POS - 0.5, -0.5), 1, 10, fill=False,
                                edgecolor=net_color('Reward'), lw=0.9, zorder=5))
    axh.add_patch(plt.Rectangle((-0.5, REW_POS - 0.5), 10, 1, fill=False,
                                edgecolor=net_color('Reward'), lw=0.9, zorder=5))
    axh.tick_params(length=1.5, pad=0.5)
    rew, _ = rew_non(M)
    n_c = sum(1 for s, c in subj_cluster.items() if c == cid)
    axh.set_title(f'{CLUSTER_NAMES[cid]} (n = {n_c})\nReward row mean $r$ = {rew.mean():+.2f}',
                  fontsize=FONT['panel_title'], color=CLUSTER_COLORS[cid],
                  fontweight='bold', pad=4)
    print(f'  a C{cid}: Reward-row mean r = {rew.mean():+.3f}')
cax = fig.add_axes([ax_ + aw - 0.020, ay + ah * 0.28, 0.007, ah * 0.44])
cb = fig.colorbar(im, cax=cax)
cb.set_label('$r$', fontsize=FONT['stat_inset'], labelpad=1)
cb.ax.tick_params(labelsize=fs('stat_inset', -0.5), width=0.5, length=2)
cb.outline.set_linewidth(0.5)

# ------------------------------------------------------------------ panel b
ax_b = fig.add_axes(BOX['b'])
TPS = [('Pre', 'BL'), ('post7d', '7d'), ('post3m', '3M')]
for cid in (1, 2, 3):
    rew_tr, non_tr = [], []
    for tp, _ in TPS:
        M = coact(tp, cid)
        r_, n_ = rew_non(M)
        rew_tr.append(r_.mean()); non_tr.append(n_.mean())
    ax_b.plot(range(3), rew_tr, color=CLUSTER_COLORS[cid], lw=1.6, marker='o',
              ms=3.0, zorder=5)
    ax_b.plot(range(3), non_tr, color=CLUSTER_COLORS[cid], lw=1.0, ls='--',
              marker='s', ms=2.4, alpha=0.55, zorder=4)
    print(f'  b C{cid}: Reward {rew_tr[0]:+.2f} -> {rew_tr[1]:+.2f} -> {rew_tr[2]:+.2f}')
ax_b.axhline(0, color='#CCC', lw=0.6, ls=':')
ax_b.set_xticks(range(3))
ax_b.set_xticklabels([t[1] for t in TPS], fontsize=FONT['micro'])
ax_b.set_ylabel('co-reactivity $r$', fontsize=FONT['axis_label'])
ax_b.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
ax_b.tick_params(axis='x', length=2, pad=1)
ax_b.set_title('coupling trajectory', fontsize=FONT['panel_title'], color='#333',
               loc='left', pad=4)
ax_b.legend(handles=[Line2D([0], [0], color='#555', lw=1.6, marker='o', ms=3,
                            label='Reward-crossing'),
                     Line2D([0], [0], color='#555', lw=1.0, ls='--', marker='s',
                            ms=2.4, alpha=0.6, label='non-Reward')],
            fontsize=fs('stat_inset', -0.5), frameon=False, loc='lower left',
            handlelength=1.4, labelspacing=0.15, borderpad=0.1)
ax_b.margins(y=0.22)
ax_b.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel c
rs = {'Pre': pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv'),
      'post7d': pd.read_csv(f'{P}/post_7d_subject_feature_matrix_new.csv')}
edge_cols = rs['Pre'].columns[1:]
rsfc_subjects = [s.replace('.mat', '') for s in rs['Pre'].iloc[:, 0].values]

rois = set()
for col in edge_cols:
    a, b = col.split('_vs_'); rois.add(a.strip()); rois.add(b.strip())
roi_list = sorted(rois)
r_idx = {r: i for i, r in enumerate(roi_list)}


def assign_net(roi):
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


roi_net = {r: assign_net(r) for r in roi_list}
edge_pairs = [(r_idx[c.split('_vs_')[0].strip()], r_idx[c.split('_vs_')[1].strip()])
              for c in edge_cols]
net_idx = {n: [i for i, r in enumerate(roi_list) if roi_net[r] == n]
           for n in NETWORK_ORDER}
pair_edges = {}
for i, ni in enumerate(NETWORK_ORDER):
    for nj in NETWORK_ORDER[i + 1:]:
        pair_edges[frozenset([ni, nj])] = [
            e for e, (a, b) in enumerate(edge_pairs)
            if (a in net_idx[ni] and b in net_idx[nj]) or (b in net_idx[ni] and a in net_idx[nj])]
cl_rows = {cid: [i for i, s in enumerate(rsfc_subjects) if subj_cluster.get(s) == cid]
           for cid in (1, 2, 3)}


def rsfc_pair(cid, tp):
    X = rs[tp][edge_cols].values
    return {k: (np.mean([X[s][e] for s in cl_rows[cid] for e in ee]) if ee else np.nan)
            for k, ee in pair_edges.items()}


def fdcr_pair(cid, tp):
    M = coact(tp, cid)
    out = {}
    for i, ni in enumerate(NETWORK_ORDER):
        for j in range(i + 1, 10):
            out[frozenset([ni, NETWORK_ORDER[j]])] = M[i, j]
    return out


cx, cy, cw, ch = BOX['c']
row_h = ch / 2
summary = {}
for ri, (tp, tp_lab) in enumerate([('Pre', 'Baseline'), ('post7d', 'Day 7')]):
    for ci, cid in enumerate((1, 2, 3)):
        axp = fig.add_axes([cx + ci * (cw / 3), cy + (1 - ri) * row_h + row_h * 0.10,
                            cw / 3 * 0.84, row_h * 0.76])
        R, F = rsfc_pair(cid, tp), fdcr_pair(cid, tp)
        axp.axhline(0, color='#DDD', lw=0.6); axp.axvline(0, color='#DDD', lw=0.6)
        rew_x, rew_y, non_x, non_y = [], [], [], []
        for key in R:
            if key not in F or np.isnan(R[key]):
                continue
            x, y = R[key], F[key]
            if 'Reward' in list(key):
                axp.scatter(x, y, s=26, color=CLUSTER_COLORS[cid], alpha=0.92,
                            edgecolors='white', linewidths=0.5, zorder=5)
                rew_x.append(x); rew_y.append(y)
            else:
                axp.scatter(x, y, s=10, color='#C8CDD2', alpha=0.55,
                            edgecolors='none', zorder=3)
                non_x.append(x); non_y.append(y)
        rew_a, non_a = np.asarray(rew_y), np.asarray(non_y)
        p_phasic = mannwhitneyu(rew_a, non_a, alternative='two-sided')[1]
        p_tonic = mannwhitneyu(np.asarray(rew_x), np.asarray(non_x),
                               alternative='two-sided')[1]

        def _fmt(p):
            return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'

        hot = p_phasic < 0.05
        axp.text(0.96, 0.05,
                 f'MWU, Reward vs rest\ntonic $p$ = {_fmt(p_tonic)}\n'
                 f'phasic $p$ = {_fmt(p_phasic)}',
                 transform=axp.transAxes, va='bottom', ha='right',
                 fontsize=fs('stat_inset', -0.5), color='#222' if hot else '#8A8F94',
                 linespacing=1.35, multialignment='left',
                 bbox=dict(boxstyle='square,pad=0.40', facecolor='white',
                           edgecolor='#111111' if hot else '#D5D8DC',
                           linewidth=0.9 if hot else 0.7, alpha=0.94))
        axp.set_xlim(-0.46, 0.52)
        # headroom so markers sitting at r = 1 are not clipped, but the axis is
        # only ticked up to the ceiling of a Pearson correlation
        axp.set_ylim(-0.85, 1.25)
        axp.set_yticks([-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
        axp.tick_params(labelsize=FONT['micro'], length=2, pad=1)
        if ri == 0:
            n_c = sum(1 for s, c in subj_cluster.items() if c == cid)
            axp.set_title(f'{CLUSTER_NAMES[cid]} (n = {n_c})',
                          fontsize=FONT['panel_title'], color=CLUSTER_COLORS[cid],
                          fontweight='bold', pad=5)
        else:
            axp.set_xlabel('RS-FC connectivity (Fisher $z$)   [tonic]',
                           fontsize=FONT['axis_label'])
        if ci == 0:
            axp.set_ylabel(f'{tp_lab}\nFDCR co-reactivity $r$   [phasic]',
                           fontsize=FONT['axis_label'])
        else:
            axp.set_yticklabels([])
        axp.spines[['top', 'right']].set_visible(False)
        summary[(tp, cid)] = (rew_a.mean(), p_tonic, p_phasic)

fig.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#666', markersize=4.5,
           markeredgecolor='white', label='Reward-crossing pair'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C8CDD2', markersize=3.5,
           label='non-Reward pair')],
    loc='center', bbox_to_anchor=(0.5, 1 - (6.63 / FIG_H)), ncol=2,
    fontsize=FONT['stat_inset'], frameon=False)


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure6.pdf',
    OUTPUT_DIR / 'Figure6_preview.png', preview_dpi=200
)

plt.close()
print('saved Figure6.pdf / Figure6_preview.png')
print('  c  Reward-crossing pairs vs the rest (Mann-Whitney U per axis):')
for tp, lab in [('Pre', 'Baseline'), ('post7d', 'Day 7')]:
    for cid in (1, 2, 3):
        m, pt, pp = summary[(tp, cid)]
        print(f'    {lab:<8} C{cid}: phasic mean={m:+.2f}  tonic p={pt:.3f}  phasic p={pp:.2e}')
