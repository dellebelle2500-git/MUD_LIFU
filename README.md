# MUD LIFU figure build

Reproducible code for the main and Extended Data figures in the methamphetamine-use-disorder / nucleus-accumbens LIFU study.

## Nature-facing artwork rules used here

- Multi-panel artwork is built at a fixed **180 mm width** with the original aspect ratio and a maximum height of **170 mm** in this repository.
- Panel labels are **8 pt, bold, upright, lower-case**.
- All other vector lettering is constrained to **5-7 pt at final output size**.
- A single sans-serif family is used throughout. The code selects **Arial**, then **Helvetica** when installed; open sans-serif fallbacks are used only when those fonts are unavailable. Font files are not distributed with the repository.
- Figure-level titles/captions (for example, `Figure 1. ...`) are intentionally **not embedded in the artwork**. Figure legends belong in the manuscript text; only panel titles/labels that are part of the data display are drawn here.
- PDF output keeps text and line art as vector objects. The save function deliberately does not use `bbox_inches='tight'`, because changing the canvas after plotting would change the effective final font size.
- Extended Data scripts additionally export **RGB JPEG at 300 dpi** for submission. PDF copies are retained for vector archival/review.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PHATE

`Figure4_final.py` uses the **official PyPI PHATE package**:

```python
import phate
op = phate.PHATE(
    n_components=2,
    knn=2,
    decay=40,
    t='auto',
    random_state=42,
    verbose=0,
)
```

There is intentionally **no local `phate.py`** in this repository. `requirements.txt` pins `phate==2.0.0` so a normal `pip install -r requirements.txt` installs the official package and its dependencies.

`Figure2_final.py` likewise uses the official `pingouin` package for repeated-measures ANOVA.

## Build

From the repository root:

```bash
python build_all.py
python preflight_figures.py
```

`build_all.py` creates:

- `outputs/Figure1.pdf` through `outputs/Figure6.pdf`
- `outputs/ExtDataFig1.jpg` through `outputs/ExtDataFig4.jpg` (300 dpi RGB submission versions)
- vector PDF copies of all Extended Data figures
- `outputs/MUD_main_Figs_NatureReady.pdf` and `outputs/MUD_ExtData_Figs_NatureReady.pdf` as convenience review composites

The individual figure files, rather than the convenience composites, are the intended submission units.

## Repository layout

```text
.
├── build_all.py
├── preflight_figures.py
├── requirements.txt
├── data/                 # derived tabular inputs used by the scripts
├── rasters/              # MRI/statistical-map raster inputs and circles.json
├── helpers/
│   ├── fig_style.py      # shared typography, colours and final-size output
│   └── fig_rasters.py    # no-circle raster placement + scaled ROI rings
├── scripts/
│   ├── Figure1_final.py ... Figure6_final.py
│   └── ExtDataFig1_final.py ... ExtDataFig4_final.py
└── outputs/              # generated; ignored by git except .gitkeep
```

## Font reproducibility

The figure source does not bundle proprietary fonts. On a workstation with Arial or Helvetica installed, that family will be selected automatically for all figures. On systems without either, the helper selects the first available Helvetica-like/open sans-serif fallback and reports the resulting embedded font in PDF preflight. Do not copy or commit proprietary font files into the repository.

## Raster inputs

The MRI/statistical-map inputs in `rasters/` are the supplied **no-circle** versions. Yellow ROI rings are reconstructed once from `circles.json`, with coordinates scaled to each current raster size. The code does not artificially upsample the raster source images.
