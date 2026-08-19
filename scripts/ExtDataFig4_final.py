"""
SupplFig4_subtype_coreactivity.py — supporting analyses for Fig. 6a,b.

    a-c   change in cue-evoked co-reactivity at day 7 relative to baseline, by subtype
    d     baseline Reward-crossing versus non-Reward co-reactivity, by subtype,
          with each Reward-crossing pair labelled
    e, f  the corresponding change at day 7 and at three months
    g-l   pre versus post co-reactivity for every network pair, by subtype and
          timepoint, against the line of no change

Figure 6a shows the baseline matrices and 6b their trajectories; this figure
gives the change matrices behind those trajectories and the pair-level spread
that the subtype means summarise.
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

from helpers.fig_style import (apply_base_style, FONT, NET_COLORS, NETWORK_ORDER,
                       CLUSTER_COLORS, CLUSTER_NAMES, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
fdcr = pd.read_csv(f'{P}/FDCR_Final_45ROIs_Mean_Features.csv')
roi_cols = [c for c in fdcr.columns if c.startswith('Cat')]

CATMAP = {'Cat01': 'Default', 'Cat02': 'Memory-Emotion', 'Cat03': 'Reward', 'Cat04': 'Relay',
          'Cat05': 'Compulsion', 'Cat06': 'Automaticity', 'Cat07': 'Attention',
          'Cat08': 'Salience', 'Cat09': 'Execution', 'Cat10': 'Regulation'}
NETS = list(NETWORK_ORDER)
ORDER = [next(c for c, n in CATMAP.items() if n == net) for net in NETS]
SHORT = {'Default': 'DMN', 'Salience': 'Sal', 'Attention': 'Att', 'Regulation': 'Reg',
         'Execution': 'Exec', 'Memory-Emotion': 'Mem', 'Reward': 'Rew',
         'Compulsion': 'Comp', 'Automaticity': 'Auto', 'Relay': 'Relay'}
REW = NETS.index('Reward')

subj_cluster = {r['Subject']: int(r['Cluster'])
                for _, r in fdcr[fdcr['Time'] == 'Pre'].iterrows()}
cl_ids = {c: [s for s, v in subj_cluster.items() if v == c] for c in (1, 2, 3)}
print('  cluster sizes:', {c: len(v) for c, v in cl_ids.items()})


def coact(ids, tp):
    sub = fdcr[(fdcr['Time'] == tp) & (fdcr['Subject'].isin(ids))].sort_values('Subject')
    nd = {c: sub[[x for x in roi_cols if x.startswith(c)]].mean(axis=1).values for c in ORDER}
    R = np.full((10, 10), np.nan)
    for i, ci in enumerate(ORDER):
        for j, cj in enumerate(ORDER):
            if i != j:
                R[i, j] = np.corrcoef(nd[ci], nd[cj])[0, 1]
    return R


def split_pairs(R):
    """upper-triangle values, separated into Reward-crossing and the rest"""
    rew, rew_lab, non = [], [], []
    for i in range(10):
        for j in range(i + 1, 10):
            if REW in (i, j):
                rew.append(R[i, j])
                rew_lab.append(SHORT[NETS[j if i == REW else i]])
            else:
                non.append(R[i, j])
    return np.array(rew), rew_lab, np.array(non)


FIG_W, FIG_H = 7.5, 7.05
fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


for L, (lx, ly) in {'a': (0.40, 0.14), 'b': (2.72, 0.14), 'c': (5.04, 0.14),
                    'd': (0.50, 2.36), 'e': (3.20, 2.36), 'f': (5.16, 2.36),
                    'g': (0.58, 3.86), 'h': (3.00, 3.86), 'i': (5.42, 3.86),
                    'j': (0.58, 5.31), 'k': (3.00, 5.31), 'l': (5.42, 5.31)}.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')


def draw_delta(ax, D, title, vabs=1.6):
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad('#E8E8E8')
    im = ax.imshow(D, cmap=cmap, vmin=-vabs, vmax=vabs, aspect='equal')
    for i in range(10):
        for j in range(10):
            if np.isnan(D[i, j]):
                continue
            ax.text(j, i, f'{D[i, j]:+.1f}'.replace('+0.', '.').replace('-0.', '\u2212.'),
                    ha='center', va='center', fontsize=fs('micro', -1.2),
                    color='white' if abs(D[i, j]) > 0.75 else '#333')
    lab = [SHORT[n] for n in NETS]
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(lab, rotation=45, ha='right', fontsize=fs('micro', -0.8))
    ax.set_yticklabels(lab, fontsize=fs('micro', -0.8))
    for tk, n in zip(ax.get_xticklabels(), NETS): tk.set_color(NET_COLORS[n])
    for tk, n in zip(ax.get_yticklabels(), NETS): tk.set_color(NET_COLORS[n])
    for pos in (REW - 0.5, REW + 0.5):
        ax.axhline(pos, color='#111', lw=1.0)
        ax.axvline(pos, color='#111', lw=1.0)
    ax.tick_params(length=1.2, pad=0.8)
    ax.set_title(title, fontsize=fs('panel_title', -0.5), pad=3)
    return im


# ---------------------------------------------------------------- a-f
for k, cl in enumerate((1, 2, 3)):
    ax = fig.add_axes(rect(0.62 + k * 2.32, 0.30, 2.00, 1.62))
    D = coact(cl_ids[cl], 'post7d') - coact(cl_ids[cl], 'Pre')
    im = draw_delta(ax, D, f'{CLUSTER_NAMES[cl]}  (day 7 \u2212 baseline)')
    m = np.nanmean([D[REW, j] for j in range(10) if j != REW])
    print(f'  Post-7d \u2212 Pre  {CLUSTER_NAMES[cl]:<20} Reward-row mean \u0394R = {m:+.3f}')
cax = fig.add_axes(rect(7.24, 0.48, 0.07, 1.10))
cb = fig.colorbar(im, cax=cax); cb.set_label('\u0394 co-reactivity', fontsize=FONT['stat_inset'])
cb.ax.tick_params(labelsize=FONT['micro'], length=1.5)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- g
ax_g = fig.add_axes(rect(0.72, 2.55, 2.15, 1.08))
for k, cl in enumerate((1, 2, 3)):
    rew, rew_lab, non = split_pairs(coact(cl_ids[cl], 'Pre'))
    xn, xr = k * 1.15, k * 1.15 + 0.42
    ax_g.scatter(xn + rng.normal(0, 0.055, len(non)), non, s=12, color='#C9CDD1',
                 alpha=0.75, edgecolors='white', linewidths=0.25, zorder=3)
    ax_g.hlines(non.mean(), xn - 0.15, xn + 0.15, color='#6B7278', lw=1.2, zorder=4)
    jit = np.linspace(-0.10, 0.10, len(rew))
    ax_g.scatter(xr + jit, rew, s=17, color=CLUSTER_COLORS[cl], alpha=0.9,
                 edgecolors='white', linewidths=0.35, zorder=5)
    ax_g.hlines(rew.mean(), xr - 0.15, xr + 0.15, color=CLUSTER_COLORS[cl], lw=1.4, zorder=6)
    for x, y, t in zip(xr + jit, rew, rew_lab):
        ax_g.annotate(t, (x, y), fontsize=fs('micro', -1.6), color=CLUSTER_COLORS[cl],
                      xytext=(2.5, 1.5), textcoords='offset points', alpha=0.85)
    p = stats.mannwhitneyu(rew, non, alternative='two-sided')[1]
    ax_g.text((xn + xr) / 2, 1.02, f'{p:.0e}' if p < 0.001 else f'{p:.2f}',
              transform=ax_g.get_xaxis_transform(), ha='center', va='bottom',
              fontsize=fs('stat_inset', -0.5), fontweight='bold' if p < 0.05 else 'normal',
              color=CLUSTER_COLORS[cl] if p < 0.05 else '#8A8F94')
    print(f'  Baseline         {CLUSTER_NAMES[cl]:<20} Rew vs non-Rew p = {p:.4g}')
ax_g.axhline(0, color='#DDD', lw=0.6, ls=':')
ax_g.set_xticks([k * 1.15 + 0.21 for k in range(3)])
ax_g.set_xticklabels(['C1', 'C2', 'C3'], fontsize=FONT['tick'])
ax_g.set_ylabel('co-reactivity ($r$)', fontsize=FONT['axis_label'])
ax_g.tick_params(axis='y', labelsize=FONT['tick'], length=2)
ax_g.tick_params(axis='x', length=0, pad=2)
ax_g.spines[['top', 'right']].set_visible(False)
ax_g.set_title('Baseline, each point one network pair',
               fontsize=FONT['panel_title'], loc='left', pad=8)

# ---------------------------------------------------------------- h
for pi, (tp, ttl) in enumerate([('post7d', 'Change at day 7'),
                                ('post3m', 'Change at 3 months')]):
    ax_h = fig.add_axes(rect(3.42 + pi * 1.95, 2.55, 1.72, 1.08))
    for k, cl in enumerate((1, 2, 3)):
        r1, _, n1 = split_pairs(coact(cl_ids[cl], 'Pre'))
        r2, _, n2 = split_pairs(coact(cl_ids[cl], tp))
        rew, non = r2 - r1, n2 - n1
        xn, xr = k * 1.05, k * 1.05 + 0.38
        ax_h.scatter(xn + rng.normal(0, 0.05, len(non)), non, s=11, color='#C9CDD1',
                     alpha=0.75, edgecolors='white', linewidths=0.25, zorder=3)
        ax_h.hlines(non.mean(), xn - 0.14, xn + 0.14, color='#6B7278', lw=1.2, zorder=4)
        ax_h.scatter(xr + rng.normal(0, 0.04, len(rew)), rew, s=16, color=CLUSTER_COLORS[cl],
                     alpha=0.9, edgecolors='white', linewidths=0.35, zorder=5)
        ax_h.hlines(rew.mean(), xr - 0.14, xr + 0.14, color=CLUSTER_COLORS[cl], lw=1.4, zorder=6)
        p = stats.mannwhitneyu(rew, non, alternative='two-sided')[1]
        ax_h.text((xn + xr) / 2, 1.02, f'{p:.0e}' if p < 0.001 else f'{p:.2f}',
                  transform=ax_h.get_xaxis_transform(), ha='center', va='bottom',
                  fontsize=fs('stat_inset', -0.5),
                  fontweight='bold' if p < 0.05 else 'normal',
                  color=CLUSTER_COLORS[cl] if p < 0.05 else '#8A8F94')
        print(f'  {ttl:<24} {CLUSTER_NAMES[cl]:<20} Rew vs non-Rew p = {p:.4g}')
    ax_h.axhline(0, color='#DDD', lw=0.6, ls=':')
    ax_h.set_ylim(-1.9, 1.9)
    ax_h.set_xticks([k * 1.05 + 0.19 for k in range(3)])
    ax_h.set_xticklabels(['C1', 'C2', 'C3'], fontsize=FONT['tick'])
    ax_h.set_ylabel('\u0394 co-reactivity' if pi == 0 else '', fontsize=FONT['axis_label'])
    if pi == 1:
        ax_h.set_yticklabels([])
    ax_h.tick_params(axis='y', labelsize=FONT['tick'], length=2)
    ax_h.tick_params(axis='x', length=0, pad=2)
    ax_h.spines[['top', 'right']].set_visible(False)
    ax_h.set_title(ttl, fontsize=FONT['panel_title'], loc='left', pad=8)

# ---------------------------------------------------------------- g-l
for row, (tp, rlab) in enumerate([('post7d', 'Baseline \u2192 day 7'),
                                  ('post3m', 'Baseline \u2192 3 months')]):
    for k, cl in enumerate((1, 2, 3)):
        ax = fig.add_axes(rect(0.78 + k * 2.40, 4.04 + row * 1.45, 1.84, 1.15))
        pre, post = coact(cl_ids[cl], 'Pre'), coact(cl_ids[cl], tp)
        ax.plot([-0.6, 1.05], [-0.6, 1.05], ls='--', color='#BBB', lw=0.7, zorder=1)
        rew_d, non_d = [], []
        for i in range(10):
            for j in range(i + 1, 10):
                x, y = pre[i, j], post[i, j]
                if REW in (i, j):
                    rew_d.append(y - x)
                    ax.scatter(x, y, s=15, color=CLUSTER_COLORS[cl], alpha=0.9,
                               edgecolors='white', linewidths=0.3, zorder=5)
                else:
                    non_d.append(y - x)
                    ax.scatter(x, y, s=8, color='#C9CDD1', alpha=0.6,
                               edgecolors='white', linewidths=0.2, zorder=3)
        p = stats.mannwhitneyu(rew_d, non_d, alternative='two-sided')[1]
        ptxt = f'{p:.0e}' if p < 0.001 else f'{p:.2f}'
        ax.text(0.04, 0.04,
                f'\u0394Rew {np.mean(rew_d):+.2f}\n\u0394non {np.mean(non_d):+.2f}\n$p$ = {ptxt}',
                transform=ax.transAxes, fontsize=fs('stat_inset', -1.0), va='bottom', ha='left',
                linespacing=1.25,
                bbox=dict(boxstyle='square,pad=0.22', facecolor='white',
                          edgecolor='#D9DCDF', linewidth=0.5, alpha=0.92))
        ax.set_xlim(-0.55, 1.03); ax.set_ylim(-0.55, 1.03)
        ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.tick_params(labelsize=FONT['micro'], length=1.5, pad=1)
        if k == 0:
            ax.set_ylabel(f'{rlab}\npost $r$', fontsize=fs('axis_label', -0.5))
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel('baseline $r$', fontsize=fs('axis_label', -0.5))
        else:
            ax.set_xticklabels([])
        if row == 0:
            ax.set_title(CLUSTER_NAMES[cl], fontsize=fs('panel_title', -0.5),
                         color=CLUSTER_COLORS[cl], pad=3)
        ax.spines[['top', 'right']].set_visible(False)
        print(f'  {rlab:<16} {CLUSTER_NAMES[cl]:<20} \u0394Rew {np.mean(rew_d):+.3f} '
              f'\u0394non {np.mean(non_d):+.3f}  p = {p:.4g}')

fig.legend(handles=[Line2D([0], [0], marker='o', color='w', markersize=5,
                           markerfacecolor=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
                    for c in (1, 2, 3)]
                   + [Line2D([0], [0], marker='o', color='w', markersize=5,
                             markerfacecolor='#C9CDD1', label='non-Reward pairs'),
                      Line2D([0], [0], ls='--', color='#BBB', lw=0.8, label='no change')],
           loc='lower center', bbox_to_anchor=(0.5, -0.030), ncol=5,
           fontsize=FONT['legend'], frameon=False)


save_nature_figure(
    fig, OUTPUT_DIR / 'ExtDataFig4.pdf',
    OUTPUT_DIR / 'ExtDataFig4_preview.png', preview_dpi=200,
    submission_jpg_path=OUTPUT_DIR / 'ExtDataFig4.jpg', submission_jpg_dpi=300
)

plt.close()
print('saved ExtDataFig4.pdf / ExtDataFig4_preview.png')
