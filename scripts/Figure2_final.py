"""
Figure2_composite.py — assembles Figure 2 as a single vector PDF.

Layout follows MUD_main_Figs_v7.pptx (7.5 x 10.83 in):

    a  longitudinal trajectory of pathological deviation
    b  treatment-response scatter (990 edges, Post-7d and Post-3M)
    c  category-level treatment response, hive plot (exploratory)
    d  RS-FC trajectories of the five Friedman-significant network pairs
    e  NAc seed-to-voxel trajectories of the posterior-medial system, by outcome
    f  NAc seed-to-voxel main-effect regions (raster, left blank)

Panel logic is taken from the original scripts:
    a  fig_projection_slope_total_left.py
    b  fig_treatment_response_scatter.py
    c  fig_hive_all15_cat_exploratory.py
    d  friedman_five_cat_pairs_trajectory.py
    e  fig_seedtovoxel_12ROIs_by_relapse.py

Text is embedded as text (fonttype 42) so the PDF stays editable; the space
reserved for panel f is where the brain renders are dropped in.
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
from matplotlib.path import Path
from matplotlib.lines import Line2D
from scipy import stats
import pingouin as pg
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

from helpers.fig_rasters import place_montage
from helpers.fig_style import (NETWORK_ORDER, net_color, short, apply_base_style, FONT, fs, nature_figsize, save_nature_figure)

apply_base_style()

P = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_W, FIG_H = 7.5, 7.00


def rect(left, top, w, h):
    return [left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H]


BOX = {
    'a':  rect(0.62, 0.30, 2.05, 1.28),
    'b1': rect(3.12, 0.30, 1.95, 1.28),
    'b2': rect(5.25, 0.30, 1.95, 1.28),
    'c':  rect(0.12, 1.82, 4.25, 2.30),
    'd':  rect(4.55, 1.95, 2.78, 2.05),
    'e':  rect(0.42, 4.35, 6.85, 1.05),
    'f':  rect(0.35, 5.72, 6.95, 1.08),
}
LETTER = {'a': (0.22, 0.16), 'b': (3.02, 0.16), 'c': (0.18, 1.68),
          'd': (4.40, 1.91), 'e': (0.18, 4.20), 'f': (0.18, 5.57)}

fig = plt.figure(figsize=nature_figsize(FIG_W, FIG_H))
for L, (lx, ly) in LETTER.items():
    fig.text(lx / FIG_W, 1 - ly / FIG_H, L, fontsize=FONT['letter'],
             fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------ data ----
nl = pd.read_csv(f'{P}/normal_subject_feature_matrix_clean.csv', index_col=0)
bl = pd.read_csv(f'{P}/baseline_subject_feature_matrix_new.csv', index_col=0)
p7 = pd.read_csv(f'{P}/post_7d_subject_feature_matrix_new.csv', index_col=0)
p3 = pd.read_csv(f'{P}/post_3M_subject_feature_matrix_new.csv', index_col=0)
for df in (nl, bl, p7, p3):
    df.columns = df.columns.str.replace('CerebrA_NAc', 'CerebraA_NAc', regex=False)
feat = sorted(set(nl.columns) & set(bl.columns) & set(p7.columns) & set(p3.columns))
nl, bl, p7, p3 = nl[feat], bl[feat], p7[feat], p3[feat]

mu, sd = nl.values.mean(axis=0), nl.values.std(axis=0, ddof=1)
sd[sd == 0] = 1e-9
Zb = (bl.values - mu) / sd
Z7 = (p7.values - mu) / sd
Z3 = (p3.values - mu) / sd

# ------------------------------------------------------------------ panel a
# each subject's baseline Z-vector is their pathological axis; later timepoints
# are projected onto it (fig_projection_slope_total_left.py)
proj = []
for i in range(Zb.shape[0]):
    v = Zb[i]
    mag = np.linalg.norm(v)
    u = v / mag
    proj.append([mag, float(np.dot(Z7[i], u)), float(np.dot(Z3[i], u))])
proj = np.array(proj)

ax_a = fig.add_axes(BOX['a'])
xa = np.arange(3)
for row in proj:
    ax_a.plot(xa, row, color='#C9CDD2', lw=0.5, alpha=0.9, zorder=1)
m = proj.mean(axis=0)
se = proj.std(axis=0, ddof=1) / np.sqrt(proj.shape[0])
ax_a.errorbar(xa, m, yerr=se, color='#1A1A1A', lw=1.5, marker='s', ms=3.4,
              capsize=2, zorder=5)
ax_a.axhline(0, color='#8FB8DE', ls=':', lw=0.8)
fr_a = friedmanchisquare(proj[:, 0], proj[:, 1], proj[:, 2])[1]
ax_a.set_xticks(xa)
ax_a.set_xticklabels(['BL', '7d', '3M'], fontsize=FONT['micro'])
ax_a.set_ylabel('Projected deviation\n(distance from control)', fontsize=FONT['axis_label'])
ax_a.set_title(f'Friedman $p$ = {fr_a:.1e}', fontsize=FONT['panel_title'],
               color='#333', loc='left', pad=4)
ax_a.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel b
zb_mean = Zb.mean(axis=0)
for key, Zp, lab in [('b1', Z7, 'Day 7'), ('b2', Z3, '3 months')]:
    axb = fig.add_axes(BOX[key])
    x = zb_mean
    y = (Zp - Zb).mean(axis=0)
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
    axb.scatter(x, y, s=2.0, c='#3A3A3A', alpha=0.30, edgecolors='none', zorder=3)
    sl, ic, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(-lim, lim, 50)
    axb.plot(xs, sl * xs + ic, color='#C0504D', lw=1.3, zorder=5)
    axb.axhline(0, color='#999', lw=0.6); axb.axvline(0, color='#999', lw=0.6)
    axb.set_xlim(-lim, lim); axb.set_ylim(-lim, lim)
    axb.set_xlabel('Baseline abnormality ($Z$)', fontsize=FONT['axis_label'])
    if key == 'b1':
        axb.set_ylabel('Treatment response ($\\Delta Z$)', fontsize=FONT['axis_label'])
    axb.set_title(f'{lab}   slope = {sl:.2f}, $r$ = {r:.2f}',
                  fontsize=FONT['panel_title'], color='#333', loc='left', pad=4)
    axb.tick_params(labelsize=FONT['tick'])
    axb.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel c
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
            p_ = f_.split('_vs_')
            if len(p_) != 2:
                continue
            c1, c2 = getcat(p_[0]), getcat(p_[1])
            if 'Other' in (c1, c2):
                continue
            i, j = sorted((CATS.index(c1), CATS.index(c2)))
            acc[f'{CATS[i]}|{CATS[j]}'].append(v)
        rows.append({k: (np.mean(v) if v else np.nan) for k, v in acc.items()})
    return pd.DataFrame(rows, index=df.index)


cb_, c7_, c3_, cn_ = agg_cat(bl), agg_cat(p7), agg_cat(p3), agg_cat(nl)
pairs = list(cb_.columns)
mu_c, sd_c = cn_.mean(), cn_.std(ddof=1).replace(0, 1e-9)
Zc = {k: ((d - mu_c) / sd_c) for k, d in [('Baseline', cb_), ('Post-7d', c7_), ('Post-3M', c3_)]}

ax_c = fig.add_axes(BOX['c'])
ax_c.set_aspect('equal'); ax_c.axis('off')

# geometry and styling from fig_hive_all15_cat_exploratory.py
R_ZERO, SCALE, Z_MAX = 0.48, 0.20, 2.0
ANG = {'Baseline': np.radians(210), 'Post-7d': np.radians(90), 'Post-3M': np.radians(330)}
AXIS_LABEL = {'Baseline': 'Baseline', 'Post-7d': 'Day 7', 'Post-3M': '3 months'}
BLACK, DGREY, FAINT = '#111111', '#555555', '#CCCCCC'


def z_to_xy(z, angle):
    r = R_ZERO + np.clip(z, -Z_MAX, Z_MAX) * SCALE
    return r * np.cos(angle), r * np.sin(angle)


def safe_wilcoxon(pre, post):
    dd = post - pre; dd = dd[dd != 0]
    if len(dd) < 3:
        return 1.0
    try:
        return wilcoxon(dd)[1]
    except Exception:
        return 1.0


def hedges_g_paired(pre_vals, post_vals):
    diff = post_vals - pre_vals; n = len(diff)
    if n < 2:
        return 0.0
    j = 1 - (3 / (4 * (n - 1) - 1)); sd = np.std(diff, ddof=1)
    return 0.0 if sd == 0 else (np.mean(diff) / sd) * j


def draw_convex_curve(ax, p1, p2, color, alpha, lw):
    x1, y1 = p1; x2, y2 = p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    edge_len = np.hypot(x2 - x1, y2 - y1)
    nx_, ny_ = -(y2 - y1), (x2 - x1)
    nlen = np.hypot(nx_, ny_)
    if nlen < 1e-9:
        cx, cy = mx, my
    else:
        nx_, ny_ = nx_ / nlen, ny_ / nlen
        mid_dist = np.hypot(mx, my)
        if mid_dist > 1e-9 and (nx_ * mx / mid_dist + ny_ * my / mid_dist) < 0:
            nx_, ny_ = -nx_, -ny_
        push = edge_len * 0.28
        cx, cy = mx + push * nx_, my + push * ny_
    ax.add_patch(mpatches.PathPatch(
        Path([(x1, y1), (cx, cy), (x2, y2)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
        facecolor='none', edgecolor=color, lw=lw, alpha=alpha, capstyle='round'))


def seg_style(fr_sig, rm_sig, w_sig):
    """Colour encodes the omnibus test that the pair passed, thickness the
    per-segment Wilcoxon post-hoc. Friedman takes precedence over RM-ANOVA,
    being the more conservative of the two."""
    if fr_sig:
        return (BLACK, 2.0, 0.85) if w_sig else (BLACK, 0.9, 0.80)
    if rm_sig:
        return (DGREY, 2.0, 0.70) if w_sig else (DGREY, 0.9, 0.62)
    return FAINT, 0.5, 0.28


ncm, ncs = cn_.mean(), cn_.std(ddof=1).replace(0, 1e-9)
bl_z = (cb_.mean() - ncm) / ncs
p7_z = (c7_.mean() - ncm) / ncs
p3_z = (c3_.mean() - ncm) / ncs

r_min, r_max = R_ZERO - Z_MAX * SCALE, R_ZERO + Z_MAX * SCALE
for tp, angle in ANG.items():
    ax_c.plot([r_min * np.cos(angle), r_max * np.cos(angle)],
              [r_min * np.sin(angle), r_max * np.sin(angle)],
              color='#BBBBBB', lw=1.0, zorder=1)
    x0, y0 = R_ZERO * np.cos(angle), R_ZERO * np.sin(angle)
    perp = angle + np.pi / 2; tk = 0.025
    ax_c.plot([x0 - tk * np.cos(perp), x0 + tk * np.cos(perp)],
              [y0 - tk * np.sin(perp), y0 + tk * np.sin(perp)],
              color='#2CA02C', lw=1.8, alpha=0.75, zorder=3)
    for zt in [-2, 0, 2]:
        rt = R_ZERO + zt * SCALE
        xt, yt = rt * np.cos(angle), rt * np.sin(angle)
        off = 0.07
        ax_c.text(xt + off * np.cos(perp), yt + off * np.sin(perp),
                  f'{zt:+d}' if zt else '0', fontsize=FONT['micro_small'],
                  color='#888', ha='center', va='center')
    ax_c.text((r_max + 0.11) * np.cos(angle), (r_max + 0.11) * np.sin(angle),
              AXIS_LABEL[tp],
              fontsize=FONT['annotation'], fontweight='bold', ha='center',
              va='center', color='#222')

lab_black, lab_grey = [], []
for cp in pairs:
    pre = cb_[cp].values; po7 = c7_[cp].values; po3 = c3_[cp].values
    try:
        fp = friedmanchisquare(pre, po7, po3)[1]
    except Exception:
        fp = 1.0
    long_rm = pd.DataFrame({'y': np.concatenate([pre, po7, po3]),
                            'time': np.repeat(['Pre', '7d', '3M'], len(pre)),
                            'subj': np.tile(np.arange(len(pre)), 3)})
    try:
        aov = pg.rm_anova(dv='y', within='time', subject='subj', data=long_rm,
                          detailed=True)
        rp = float(aov.iloc[0]['p_unc'])
    except Exception:
        rp = 1.0
    fr_sig, rm_sig = fp < 0.05, rp < 0.05
    pos_bl = z_to_xy(bl_z[cp], ANG['Baseline'])
    pos_7d = z_to_xy(p7_z[cp], ANG['Post-7d'])
    pos_3m = z_to_xy(p3_z[cp], ANG['Post-3M'])
    for pa, pb, pw in [(pos_bl, pos_7d, safe_wilcoxon(pre, po7)),
                       (pos_7d, pos_3m, safe_wilcoxon(po7, po3)),
                       (pos_bl, pos_3m, safe_wilcoxon(pre, po3))]:
        col, lw, al = seg_style(fr_sig, rm_sig, pw < 0.05)
        draw_convex_curve(ax_c, pa, pb, col, al, lw)
    if fr_sig:
        lab_black.append({'cp': cp, 'anchor': pos_3m, 'z': p3_z[cp],
                          'g': hedges_g_paired(pre, po3)})
    elif rm_sig:
        lab_grey.append({'cp': cp, 'anchor': pos_bl, 'z': bl_z[cp],
                         'g': hedges_g_paired(pre, po3)})

for group, xa, xt, ha_, colr in [(lab_black, 0.60, 0.63, 'left', BLACK),
                                 (lab_grey, -0.60, -0.63, 'right', DGREY)]:
    if not group:
        continue
    group.sort(key=lambda t: t['z'])
    ys_ = np.linspace(0.60, 0.10, len(group))
    for i, lc in enumerate(group):
        a, b = lc['cp'].split('|')
        arrow = '\u2191' if lc['g'] > 0 else '\u2193'
        px, py = lc['anchor']
        ax_c.plot([px, xa], [py, ys_[i]], color='grey', lw=0.4, alpha=0.45, zorder=4)
        ax_c.text(xt, ys_[i], f'{short(a)} \u2194 {short(b)} {arrow}',
                  fontsize=FONT['micro'], fontweight='bold', color=colr,
                  va='center', ha=ha_, zorder=7)

ax_c.legend(handles=[
    Line2D([0], [0], marker='|', color='#2CA02C', markersize=6, lw=0,
           markeredgewidth=1.8, alpha=0.75, label='$Z$ = 0 (control mean)'),
    Line2D([0], [0], color=BLACK, lw=2.0, alpha=0.85, label='Friedman + Wilcoxon'),
    Line2D([0], [0], color=BLACK, lw=0.9, alpha=0.80, label='Friedman only'),
    Line2D([0], [0], color=DGREY, lw=2.0, alpha=0.70, label='RM-ANOVA + Wilcoxon'),
    Line2D([0], [0], color=DGREY, lw=0.9, alpha=0.62, label='RM-ANOVA only'),
    Line2D([0], [0], color=FAINT, lw=0.8, alpha=0.4, label='n.s.')],
    loc='lower center', bbox_to_anchor=(0.46, -0.045), ncol=2,
    fontsize=FONT['stat_inset'], frameon=False, handlelength=1.5,
    labelspacing=0.28, columnspacing=0.8)
ax_c.set_xlim(-1.32, 1.12); ax_c.set_ylim(-0.95, 1.05)
sig_pairs = lab_black

# ------------------------------------------------------------------ panel d
FIVE = [('Reward', 'Default'), ('Reward', 'Reward'), ('Memory-Emotion', 'Salience'),
        ('Automaticity', 'Compulsion'), ('Memory-Emotion', 'Automaticity')]
bx, by, bw, bh = BOX['d']
NC, NR_ = 3, 2
cell_w, cell_h = bw / NC, bh / NR_
sub_w, sub_h = cell_w * 0.80, cell_h * 0.60
for k, (n1, n2) in enumerate(FIVE):
    r_, c_ = divmod(k, NC)
    i, j = sorted((CATS.index(n1), CATS.index(n2)))
    key = f'{CATS[i]}|{CATS[j]}'
    axd = fig.add_axes([bx + c_ * cell_w, by + (NR_ - 1 - r_) * cell_h + cell_h * 0.10,
                        sub_w, sub_h])
    vals = np.column_stack([cb_[key].values, c7_[key].values, c3_[key].values])
    hc = cn_[key].values
    xd = np.arange(3)
    mm = vals.mean(axis=0)
    ss = vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
    axd.errorbar(xd, mm, yerr=ss, color='#C0504D', lw=1.2, marker='o', ms=2.4,
                 capsize=1.5, zorder=5)
    hse = hc.std(ddof=1) / np.sqrt(len(hc))
    axd.axhspan(hc.mean() - hse, hc.mean() + hse, color='#7FBF7F', alpha=0.20, zorder=0)
    fp = friedmanchisquare(vals[:, 0], vals[:, 1], vals[:, 2])[1]
    axd.set_title(f'{short(n1)} \u2194 {short(n2)}\n$p$ = {fp:.3f}',
                  fontsize=FONT['micro'], color='#333', pad=2)
    axd.set_xticks(xd); axd.set_xticklabels(['BL', '7d', '3M'], fontsize=FONT['micro_small'])
    axd.tick_params(axis='y', labelsize=FONT['micro_small'], length=2, pad=1)
    axd.tick_params(axis='x', length=2, pad=1)
    if c_ == 0:
        axd.set_ylabel('FC ($z$)', fontsize=FONT['annotation'])
    axd.spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------------ panel e
# 12-ROI ANOVA table; posterior-medial row only (fig_seedtovoxel_12ROIs_by_relapse.py)
XL = f'{P}/seedtovoxel_ANOVA_main_effect_longitudinal_final.xlsx'
d_sv = pd.read_excel(XL)
COLS = {
    'vmPFC': ['pre', 'post7d', 'post3m'],            'AG_r': ['pre.1', 'post7d.1', 'post3m.1'],
    'ACC': ['pre.2', 'post7d.2', 'post3m.2'],        'Put_l': ['pre.3', 'post7d.3', 'post3m.3'],
    'Put_r': ['pre.4', 'post7d.4', 'post3m.4'],      'Hippo_r': ['pre.5', 'post7d.5', 'post3m.5'],
    'paraHippo_l': ['pre.6', 'post7d.6', 'post3m.6'], 'SMA': ['pre.7', 'post7d.7', 'post3m.7'],
    'dmPFC': ['pre.8', 'post7d.8', 'post3m.8'],      'PCC': ['pre.9', 'post7d.9', 'post3m.9'],
    'precuneus': ['pre.10', 'post7d.10', 'post3m.10'], 'MTG_R': ['pre.11', 'post7d.11', 'post3m.11'],
}
PM = [('Angular gyrus', 'AG_r'), ('PCC', 'PCC'), ('Precuneus', 'precuneus'),
      ('Parahippocampal L', 'paraHippo_l')]
subj_sv = d_sv.iloc[:, 0].astype(str).values
relapsers = {'sub_04', 'sub_07', 'sub_09', 'sub_14', 'sub_17'}
rel_mask = np.array([sv_id in relapsers for sv_id in subj_sv])

# shared y-limits across the four PM panels, as in the original
lo, hi = [], []
for _, k in PM:
    M = d_sv[COLS[k]].values.astype(float)
    for msk in [None, rel_mask, ~rel_mask]:
        MM = M if msk is None else M[msk]
        mm = np.nanmean(MM, axis=0)
        ss = np.nanstd(MM, axis=0, ddof=1) / np.sqrt(len(MM))
        lo.append((mm - ss).min()); hi.append((mm + ss).max())
pad = 0.10 * (max(hi) - min(lo))
YL = (min(lo) - pad, max(hi) + pad + 0.22 * (max(hi) - min(lo)))

C_ALL, C_REL, C_NON = '#333333', '#78909C', '#B0BEC5'
ex, ey, ew, eh = BOX['e']
sub_ew = ew / 4 * 0.80
xe = np.arange(3)
for k, (lab, key) in enumerate(PM):
    M = d_sv[COLS[key]].values.astype(float)
    axe = fig.add_axes([ex + k * (ew / 4), ey, sub_ew, sub_ew * 0.75 * (FIG_W / FIG_H)])
    for msk, colr, mk, nm in [(~rel_mask, C_NON, '^', 'Non-relapse (n = 10)'),
                              (rel_mask, C_REL, 's', 'Relapse (n = 5)'),
                              (None, C_ALL, 'o', 'All (n = 15)')]:
        MM = M if msk is None else M[msk]
        mm = np.nanmean(MM, axis=0)
        ss = np.nanstd(MM, axis=0, ddof=1) / np.sqrt(len(MM))
        axe.errorbar(xe, mm, yerr=ss, color=colr, lw=1.3, marker=mk, ms=2.8,
                     capsize=1.6, zorder=5, label=nm, mec='white', mew=0.5)
    # quadratic contrast (Pre - 2*7d + 3M) isolates the curvature of the
    # dip-and-rebound, independent of any baseline offset between groups
    quad = M[:, 0] - 2 * M[:, 1] + M[:, 2]
    try:
        p_q = stats.mannwhitneyu(quad[rel_mask], quad[~rel_mask],
                                 alternative='two-sided')[1]
    except Exception:
        p_q = np.nan
    hot = (not np.isnan(p_q)) and p_q < 0.05
    axe.text(0.04, 0.97, f'quad \u00d7 group\n$p$ = {p_q:.3f}',
             transform=axe.transAxes, va='top', ha='left', fontsize=FONT['stat_inset'],
             color='#111111' if hot else '#8A8F94',
             bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                       edgecolor='#111111' if hot else '#D5D8DC', lw=0.8))
    axe.axhline(0, color='#BBB', lw=0.6, ls=':')
    axe.set_ylim(*YL)
    axe.set_xticks(xe); axe.set_xticklabels(['BL', '7d', '3M'], fontsize=FONT['micro'])
    axe.tick_params(axis='y', labelsize=FONT['micro'], length=2, pad=1)
    axe.tick_params(axis='x', length=2, pad=1)
    axe.set_title(lab, fontsize=FONT['panel_title'], color='#333', pad=3)
    if k == 0:
        axe.set_ylabel('NAc seed FC ($r$)', fontsize=FONT['axis_label'])
    else:
        axe.set_yticklabels([])
    axe.spines[['top', 'right']].set_visible(False)
fig.legend(handles=[Line2D([0], [0], color=C_ALL, lw=1.6, marker='o', ms=3.2,
                           mec='white', label='All (n = 15)'),
                    Line2D([0], [0], color=C_REL, lw=1.6, marker='s', ms=3.2,
                           mec='white', label='Relapse (n = 5)'),
                    Line2D([0], [0], color=C_NON, lw=1.6, marker='^', ms=3.2,
                           mec='white', label='Non-relapse (n = 10)')],
           loc='center', bbox_to_anchor=(0.5, 1 - (5.56 / FIG_H)), ncol=3,
           fontsize=FONT['stat_inset'], frameon=False)

# ------------------------------------------------------------------ panel f
# NAc seed-to-voxel main-effect regions, MRIcroGL renders dropped in.
place_montage(fig, BOX['f'],
              ['Fig2f_AG_Rt', 'Fig2f_PCC', 'Fig2f_Precuneus', 'Fig2f_PHC_Lt'],
              labels=['AG (R)', 'PCC', 'Precuneus', 'PHC (L)'],
              fig_size=(FIG_W, FIG_H), font=FONT['annotation'])


save_nature_figure(
    fig, OUTPUT_DIR / 'Figure2.pdf',
    OUTPUT_DIR / 'Figure2_preview.png', preview_dpi=200
)

plt.close()
print('saved Figure2.pdf / Figure2_preview.png')
print(f'  a: Friedman p = {fr_a:.2e}')
print(f'  c: {len(sig_pairs)} Friedman-significant category pairs')
