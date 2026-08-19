"""
fig_style.py — shared visual grammar for the MUD LIFU main figures.

Import from the repository root, for example:

    from helpers.fig_style import (NETWORK_ORDER, NET_COLORS, NET_SHORT,
                                   CLUSTER_COLORS, CLUSTER_NAMES, OUTCOME_COLORS,
                                   apply_base_style, net_color, short)

Three things are fixed here so that panels can be read against one another:

1. NETWORK_ORDER — anatomical grouping, identical for linear axes (heatmaps,
   strips) and circular layouts (radar, connectome):
       cortical      DMN, Salience, Attention, Regulation, Execution
       limbic        Memory-Emotion
       subcortical   Reward, Compulsion, Automaticity, Relay
   Reward, Compulsion and Automaticity are the striatal nodes and stay adjacent;
   Relay (thalamus) closes the loop. Memory-Emotion sits at the boundary, as it
   spans hippocampus, amygdala, parahippocampal cortex and brainstem.

2. NET_COLORS — Tableau-10, the palette already used by most heatmap/strip
   scripts; ten hues remain distinguishable at panel scale.

3. CLUSTER_COLORS — Okabe-Ito, the colour-vision-deficiency-safe palette used in
   the R/ggplot2 community. C1 is pushed towards vermillion and C3 towards
   bluish green so the two do not collapse onto a red-green axis, and neither
   collides with the Tableau reward hue.
"""

import matplotlib as mpl

# ---------------------------------------------------------------- order ----
NETWORK_ORDER = [
    'Default',          # cortical
    'Salience',
    'Attention',
    'Regulation',
    'Execution',
    'Memory-Emotion',   # limbic
    'Reward',           # subcortical / striatal
    'Compulsion',
    'Automaticity',
    'Relay',
]

NETWORK_BLOCKS = {
    'Cortical':    ['Default', 'Salience', 'Attention', 'Regulation', 'Execution'],
    'Limbic':      ['Memory-Emotion'],
    'Subcortical': ['Reward', 'Compulsion', 'Automaticity', 'Relay'],
}

# Aliases seen across the original scripts -> canonical name
NET_ALIASES = {
    'Default Mode': 'Default', 'DefaultMode': 'Default', 'DMN': 'Default',
    'Memory/Emotion': 'Memory-Emotion', 'Mem-Emo': 'Memory-Emotion',
    'Mem': 'Memory-Emotion', 'MemEmo': 'Memory-Emotion',
    'Auto': 'Automaticity', 'Comp': 'Compulsion', 'Att': 'Attention',
    'Sal': 'Salience', 'Exec': 'Execution', 'Reg': 'Regulation',
    'Rew': 'Reward',
}

def canon(name):
    """Map any spelling used in the original scripts to the canonical name."""
    n = str(name).strip()
    return NET_ALIASES.get(n, n)

# --------------------------------------------------------------- colours ----
# Tableau-10
NET_COLORS = {
    'Default':        '#4E79A7',
    'Salience':       '#F28E2B',
    'Attention':      '#76B7B2',
    'Regulation':     '#59A14F',
    'Execution':      '#EDC948',
    'Memory-Emotion': '#B07AA1',
    'Reward':         '#E15759',
    'Compulsion':     '#FF9DA7',
    'Automaticity':   '#9C755F',
    'Relay':          '#BAB0AC',
}

# Okabe-Ito (colour-vision-deficiency safe)
CLUSTER_COLORS = {
    1: '#D55E00',   # vermillion
    2: '#0072B2',   # blue
    3: '#009E73',   # bluish green
    'C1': '#D55E00', 'C2': '#0072B2', 'C3': '#009E73',
    'Normal': '#999999',
}

# Subtype names describe the coupling partner that most distinguishes each
# subtype from the other two (one-vs-rest, ROI level):
#   C1  salience regions bound to attention, executive and regulatory systems,
#       and detached from hippocampus/amygdala
#   C2  default-mode and executive regions bound to hippocampus/amygdala,
#       and separated from salience
#   C3  salience and attention regions bound directly to hippocampus
CLUSTER_NAMES = {
    1: 'C1 Salience-Bound',
    2: 'C2 Self-Referential',
    3: 'C3 Memory-Anchored',
}

# Neutral, deliberately low-salience so outcome never competes with subtype
OUTCOME_COLORS = {
    'No relapse': '#B0BEC5',
    'Relapse':    '#78909C',
}

# Diverging map for connectivity / co-reactivity matrices
DIVERGING_CMAP = 'RdBu_r'

# ------------------------------------------------------------ short names ----
NET_SHORT = {
    'Default': 'DMN', 'Salience': 'Sal', 'Attention': 'Att',
    'Regulation': 'Reg', 'Execution': 'Exec', 'Memory-Emotion': 'Mem-Emo',
    'Reward': 'Rew', 'Compulsion': 'Comp', 'Automaticity': 'Auto',
    'Relay': 'Relay',
}

def short(name):
    return NET_SHORT.get(canon(name), str(name))

def net_color(name):
    return NET_COLORS.get(canon(name), '#BBBBBB')

def order_index(name):
    c = canon(name)
    return NETWORK_ORDER.index(c) if c in NETWORK_ORDER else len(NETWORK_ORDER)

def sort_networks(names):
    return sorted(names, key=order_index)

# -------------------------------------------------------------- typography --
# Nature figure guidance (final artwork):
# - sans-serif type throughout, preferably Helvetica/Arial
# - panel letters: 8 pt, bold, upright, lowercase
# - all other lettering: 5-7 pt at final reproduction size
# - standard double-column width: 180 mm; maximum depth: 170 mm

from matplotlib import font_manager

MM_PER_INCH = 25.4
NATURE_WIDTH_MM = 180.0
NATURE_MAX_HEIGHT_MM = 170.0
NATURE_WIDTH_IN = NATURE_WIDTH_MM / MM_PER_INCH

FONT = {
    'letter':       8.0,
    'panel_title':  7.0,
    'axis_label':   6.5,
    'tick':         5.5,
    'annotation':   5.5,
    'legend':       5.5,
    'stat_inset':   5.2,
    'micro':        5.0,
    'micro_small':  5.0,
}


def nature_pt(value):
    """Clamp non-panel lettering to Nature's 5-7 pt final-size range."""
    return min(7.0, max(5.0, float(value)))


def fs(key, delta=0.0):
    """Return a font size while enforcing the 5-7 pt rule (except panel letters)."""
    if key == 'letter':
        return 8.0
    return nature_pt(FONT[key] + delta)


def _font_available(name):
    try:
        font_manager.findfont(font_manager.FontProperties(family=name),
                              fallback_to_default=False)
        return True
    except Exception:
        return False


# Do not distribute proprietary font files with the repository.  Use Arial or
# Helvetica when installed; otherwise use a metrically similar/open sans-serif.
_FONT_CANDIDATES = ('Arial', 'Helvetica', 'Nimbus Sans', 'Arimo',
                    'Liberation Sans', 'DejaVu Sans')
FONT_FAMILY = next((f for f in _FONT_CANDIDATES if _font_available(f)),
                   'DejaVu Sans')


def apply_base_style():
    """Nature-facing defaults at final reproduction size."""
    mpl.rcParams.update({
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'savefig.facecolor':  'white',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          False,
        'font.family':        'sans-serif',
        'font.sans-serif':    [FONT_FAMILY] + [f for f in _FONT_CANDIDATES if f != FONT_FAMILY],
        'font.size':          FONT['tick'],
        'axes.titlesize':     FONT['panel_title'],
        'axes.labelsize':     FONT['axis_label'],
        'xtick.labelsize':    FONT['tick'],
        'ytick.labelsize':    FONT['tick'],
        'legend.fontsize':    FONT['legend'],
        'axes.linewidth':     0.7,
        'xtick.major.width':  0.7,
        'ytick.major.width':  0.7,
        'lines.linewidth':    1.4,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'svg.fonttype':       'none',
        'mathtext.fontset':   'custom',
        'mathtext.rm':        FONT_FAMILY,
        'mathtext.it':        f'{FONT_FAMILY}:italic',
        'mathtext.bf':        f'{FONT_FAMILY}:bold',
    })


def nature_figsize(layout_width=7.5, layout_height=7.0):
    """Return an exact 180-mm-wide figure size preserving logical aspect ratio.

    ``layout_width`` and ``layout_height`` are dimensionless layout coordinates.
    A width of 7.5 maps to 180 mm.  The function raises if the implied height
    exceeds Nature's 170-mm page depth.
    """
    height_mm = NATURE_WIDTH_MM * float(layout_height) / float(layout_width)
    if height_mm > NATURE_MAX_HEIGHT_MM + 1e-9:
        raise ValueError(
            f'Figure height {height_mm:.1f} mm exceeds Nature maximum '
            f'{NATURE_MAX_HEIGHT_MM:.0f} mm.'
        )
    return NATURE_WIDTH_IN, height_mm / MM_PER_INCH


def save_nature_figure(fig, pdf_path, preview_path=None, preview_dpi=200,
                       submission_jpg_path=None, submission_jpg_dpi=300):
    """Save final-size artwork without changing its physical canvas.

    PDF preserves vector text/line art for main figures and archival use.
    ``submission_jpg_path`` is optional and is used by Extended Data scripts
    to produce the RGB 300-dpi JPEG requested by Nature for online-only
    Extended Data artwork.  Never use ``bbox_inches='tight'`` here because
    it changes the final physical size and therefore the effective font size.
    """
    fig.savefig(pdf_path, dpi=300)
    if preview_path is not None:
        fig.savefig(preview_path, dpi=preview_dpi)
    if submission_jpg_path is not None:
        fig.savefig(submission_jpg_path, dpi=submission_jpg_dpi, format='jpeg',
                    pil_kwargs={'quality': 95, 'subsampling': 0})


# Panel widths (inches) at the 180-mm Nature double-column width.
PANEL_W = {'full': NATURE_WIDTH_IN, 'half': 3.46, 'third': 2.24, 'two_third': 4.72}
