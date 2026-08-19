"""
SupplFig_12ROI.py — NAc seed-to-voxel trajectories for all twelve regions that
showed a significant whole-group main effect of time, split by outcome.

Figure 2e in the main text shows the four posterior-medial nodes; this figure
shows the full set, ordered by the trajectory shape assigned to each region by
the orthogonal polynomial contrasts (monotonic decline, monotonic rise, then
dip-and-rebound), with the quadratic x group contrast reported for each.
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

from helpers.fig_style import apply_base_style, FONT, fs, nature_figsize, save_nature_figure

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
C_ALL, C_REL, C_NON = '#333333', '#78909C', '#B0BEC5'

d = pd.read_excel(f'{P}/seedtovoxel_ANOVA_main_effect_longitudinal_final.xlsx')
COLS = {
    'vmPFC': ['pre', 'post7d', 'post3m'],
    'AG.R': ['pre.1', 'post7d.1', 'post3m.1'],
    'dACC': ['pre.2', 'post7d.2', 'post3m.2'],
    'Putamen L': ['pre.3', 'post7d.3', 'post3m.3'],
    'Putamen R': ['pre.4', 'post7d.4', 'post3m.4'],
    'Hippocampus R': ['pre.5', 'post7d.5', 'post3m.5'],
    'PHC.L': ['pre.6', 'post7d.6', 'post3m.6'],
    'SMA': ['pre.7', 'post7d.7', 'post3m.7'],
    'dmPFC': ['pre.8', 'post7d.8', 'post3m.8'],
    'PCC': ['pre.9', 'post7d.9', 'post3m.9'],
    'Precuneus': ['pre.10', 'post7d.10', 'post3m.10'],
    'MTG.R': ['pre.11', 'post7d.11', 'post3m.11'],
}
RELAPSERS = {'sub_04', 'sub_07', 'sub_09', 'sub_14', 'sub_17'}
SUBJ = d.iloc[:, 0].astype(str).values
IS_REL = np.array([s in RELAPSERS for s in SUBJ])

# orthogonal polynomial contrasts for three equally spaced levels
LIN = np.array([-1.0, 0.0, 1.0])
QUA = np.array([1.0, -2.0, 1.0])


def contrast_scores(M, c):
    return M @ c / np.sqrt((c ** 2).sum())


rows = []
for name, cols in COLS.items():
    M = d[cols].values.astype(float)
    lin, qua = contrast_scores(M, LIN), contrast_scores(M, QUA)
    p_lin = stats.wilcoxon(lin)[1]
    p_qua = stats.wilcoxon(qua)[1]
    is_rel = IS_REL
    p_qxg = stats.mannwhitneyu(qua[is_rel], qua[~is_rel], alternative='two-sided')[1]
    shape = ('dip-and-rebound' if p_qua < 0.05 and qua.mean() > 0 else
             'peak-and-return' if p_qua < 0.05 else
             'monotonic decline' if lin.mean() < 0 else 'monotonic rise')
    rows.append(dict(name=name, M=M, is_rel=is_rel, shape=shape,
                     p_lin=p_lin, p_qua=p_qua, p_qxg=p_qxg,
                     lin_mean=lin.mean(), qua_mean=qua.mean()))

ORDER = {'monotonic decline': 0, 'monotonic rise': 1,
         'dip-and-rebound': 2, 'peak-and-return': 3}
rows.sort(key=lambda r: (ORDER[r['shape']], r['p_qxg']))

FIG_W, FIG_H = 7.5, 4.9
fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
NC, NR = 4, 3
L, R, T, B = 0.62, 0.22, 0.62, 0.55
cw = (FIG_W - L - R) / NC
ch = (FIG_H - T - B) / NR
x = np.arange(3)

for k, r in enumerate(rows):
    i, j = divmod(k, NC)
    ax = fig.add_axes([(L + j * cw) / FIG_W,
                       1 - (T + i * ch + ch * 0.80) / FIG_H,
                       cw * 0.76 / FIG_W, ch * 0.62 / FIG_H])
    M, is_rel = r['M'], r['is_rel']
    for msk, colr, lw, ms in [(np.ones(len(M), bool), C_ALL, 1.5, 3.0),
                              (is_rel, C_REL, 1.1, 2.4),
                              (~is_rel, C_NON, 1.1, 2.4)]:
        m = M[msk].mean(0)
        se = M[msk].std(0, ddof=1) / np.sqrt(msk.sum())
        ax.errorbar(x, m, yerr=se, color=colr, lw=lw, marker='o', ms=ms,
                    capsize=1.6, elinewidth=0.7, zorder=5 if colr == C_ALL else 4)
    ax.axhline(0, color='#DDD', lw=0.6, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(['BL', '7d', '3M'], fontsize=FONT['micro'])
    ax.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
    ax.tick_params(axis='x', length=2, pad=1)
    if j == 0:
        ax.set_ylabel('NAc seed FC ($r$)', fontsize=FONT['axis_label'])
    hot = r['p_qxg'] < 0.05
    ax.set_title(f"{r['name']}\n{r['shape']}", fontsize=FONT['panel_title'],
                 color='#222', pad=3)
    ax.text(0.98, 0.97, f"quad \u00d7 group $p$ = {r['p_qxg']:.3f}",
            transform=ax.transAxes, va='top', ha='right',
            fontsize=fs('stat_inset', -0.6), color='#222' if hot else '#8A8F94',
            bbox=dict(boxstyle='square,pad=0.24', facecolor='white',
                      edgecolor='#111111' if hot else '#D9DCDF',
                      linewidth=0.9 if hot else 0.5, alpha=0.94))
    ax.margins(y=0.28)
    if hot:
        for s in ax.spines.values():
            s.set_color('#111111'); s.set_linewidth(1.1)
    ax.spines[['top', 'right']].set_visible(False)
    print(f"  {r['name']:<15}{r['shape']:<18}lin p={r['p_lin']:.4f}  "
          f"quad p={r['p_qua']:.4f}  quad x group p={r['p_qxg']:.3f}")

fig.legend(handles=[Line2D([0], [0], color=C_ALL, lw=1.5, marker='o', ms=3,
                           label='all (n = 15)'),
                    Line2D([0], [0], color=C_REL, lw=1.1, marker='o', ms=2.4,
                           label='relapse (n = 5)'),
                    Line2D([0], [0], color=C_NON, lw=1.1, marker='o', ms=2.4,
                           label='non-relapse (n = 10)')],
           loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=3,
           fontsize=FONT['legend'], frameon=False)


save_nature_figure(
    fig, OUTPUT_DIR / 'ExtDataFig2.pdf',
    OUTPUT_DIR / 'ExtDataFig2_preview.png', preview_dpi=200,
    submission_jpg_path=OUTPUT_DIR / 'ExtDataFig2.jpg', submission_jpg_dpi=300
)

plt.close()
print('saved ExtDataFig2.pdf / ExtDataFig2_preview.png')
