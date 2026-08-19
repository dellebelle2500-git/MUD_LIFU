"""Raster placement for final MUD figures.

Uses unannotated MRI renders and redraws one scaled yellow ROI ring from
``rasters/circles.json``.  Paths are repository-relative for GitHub use.
"""
from pathlib import Path
import json
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RASTER_DIR = ROOT / 'rasters'
RING_COLOUR = '#F5C400'
RING_LW = 1.0
GAP_IN = 0.06
_circles = None


def _rings():
    global _circles
    if _circles is None:
        with open(RASTER_DIR / 'circles.json', encoding='utf-8') as fh:
            _circles = json.load(fh)
    return _circles


def load_render(name):
    im = np.asarray(Image.open(RASTER_DIR / f'{name}.png').convert('RGB')).copy()
    near_white = (im >= 238).all(axis=2)
    im[near_white] = 255
    return im


def _scaled_ring(name, im):
    c = _rings().get(name)
    if c is None:
        return None
    h, w = im.shape[:2]
    sx = w / float(c.get('W', w))
    sy = h / float(c.get('H', h))
    sr = (sx * sy) ** 0.5
    return c['cx'] * sx, c['cy'] * sy, c['r'] * sr


def place_montage(fig, box, names, labels=None, fig_size=None,
                  ring=True, label_pad=0.055, font=5.5, label_colour='#333'):
    if fig_size is None:
        fw, fh = fig.get_size_inches()
    else:
        fw, fh = fig_size
    bx, by, bw, bh = box
    box_w_in, box_h_in = bw * fw, bh * fh
    imgs = [load_render(n) for n in names]
    ratios = [im.shape[1] / im.shape[0] for im in imgs]
    h_in = box_h_in
    span = lambda h: sum(r * h for r in ratios) + GAP_IN * (len(imgs) - 1)
    if span(h_in) > box_w_in:
        h_in = (box_w_in - GAP_IN * (len(imgs) - 1)) / sum(ratios)
    x_in = bx * fw + (box_w_in - span(h_in)) / 2
    y_in = by * fh + (box_h_in - h_in) / 2
    axes = []
    for k, (name, im, r) in enumerate(zip(names, imgs, ratios)):
        w_in = r * h_in
        ax = fig.add_axes([x_in / fw, y_in / fh, w_in / fw, h_in / fh])
        ax.imshow(im, interpolation='lanczos', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ring:
            rr = _scaled_ring(name, im)
            if rr is not None:
                cx, cy, rad = rr
                ax.add_patch(Circle((cx, cy), rad, fill=False,
                                    edgecolor=RING_COLOUR,
                                    linewidth=RING_LW, zorder=5, clip_on=True))
        if labels:
            ax.text(0.5, -label_pad, labels[k], transform=ax.transAxes,
                    ha='center', va='top', fontsize=font, color=label_colour)
        axes.append(ax)
        x_in += w_in + GAP_IN
    return axes
