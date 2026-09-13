"""
Figure for Claim 2 — group attribution overlaid on MNI anatomy.

Rows    = attribution methods
Columns = the six (modality, plane) streams
Overlay = group map averaged over correctly classified AD subjects
Backdrop= CN-group mean image (same MNI slice, so anatomy and overlay align)
Contours= a-priori AD ROIs, so a reader can see where the anatomy actually is

Run test_attribution_claim2_lowmem.py and build_roi_masks.py first.

Orientation: the .npy slices come straight out of nibabel array order, which is
usually NOT how a radiologist expects to see them. Set DISPLAY per plane
(rot90 count, then optional left-right flip) until the brains look upright, then
leave it fixed. Nothing else in the analysis depends on this — it is display
only.
"""

import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ATTR_DIR = "./attrib"
MASK_DIR = "./roi_masks"
OUT_DIR  = "./figures"; os.makedirs(OUT_DIR, exist_ok=True)

MODS   = ["mri", "pet"]
PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
STREAMS = [(m, p) for m in MODS for p in PLANES]

METHODS = ["occlusion", "layercam", "gradcam", "gradcampp"]
METHOD_LABEL = {"occlusion": "Occlusion", "layercam": "Layer-CAM",
                "gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++"}

# ROI outlines to draw. Keep it short or the figure gets busy.
CONTOUR_ROIS = {
    "hippocampus": "#00e5ff",
    "post_cing":   "#7CFC00",
    "precuneus":   "#ffd000",
}
DRAW_CONTOURS = True

# display orientation per plane: (number of 90deg rotations, flip left-right)
DISPLAY = {"axial": (1, False), "coronal": (1, False), "sagittal": (1, False)}

PCT_LO, PCT_HI = 60, 99.5    # overlay shown from the 60th to the 99.5th percentile
                             # raise PCT_LO to show only the strongest attribution
ALPHA_MAX = 0.75
DPI = 300

# transparent -> warm colormap so anatomy shows through weak attribution
CMAP = LinearSegmentedColormap.from_list(
    "attr", [(0, 0, 0, 0), (0.55, 0, 0.35, 0.55), (0.95, 0.25, 0, 0.85),
             (1, 0.95, 0.25, 1.0)])


def orient(a, plane):
    k, flip = DISPLAY[plane]
    a = np.rot90(a, k)
    return a[:, ::-1] if flip else a


def backdrop(mod, plane):
    c, _ = CONFIG[mod][plane]
    return np.load(f"{MASK_DIR}/cnmean_{mod}_{plane}_{c}_0.npy")


def brain_mask(mod, plane):
    b = backdrop(mod, plane)
    return np.abs(b) > (0.0 if mod == "mri" else 0.05)


def roi_mask(roi, mod, plane):
    c, _ = CONFIG[mod][plane]
    f = f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_0.npy"
    return (np.load(f) > 0.5) if os.path.exists(f) else None


def load_group(meth, mod, plane):
    c, _ = CONFIG[mod][plane]
    f = f"{ATTR_DIR}/groupmap_{meth}_{mod}_{plane}_{c}.npy"
    return np.load(f) if os.path.exists(f) else None


def scale(a, bm):
    """Clip to percentiles inside the brain, map to [0,1], zero outside."""
    v = a[bm]
    v = v[np.isfinite(v)]
    if v.size == 0 or np.allclose(v, 0):
        return np.zeros_like(a)
    lo, hi = np.percentile(v, [PCT_LO, PCT_HI])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1) * bm


methods = [m for m in METHODS
           if any(load_group(m, mo, pl) is not None for mo, pl in STREAMS)]
assert methods, f"no group maps found in {ATTR_DIR}"
print("methods found:", methods)

nrow, ncol = len(methods), len(STREAMS)
fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.35 * nrow),
                         constrained_layout=True)
axes = np.atleast_2d(axes)

for r, meth in enumerate(methods):
    for c_, (mod, plane) in enumerate(STREAMS):
        ax = axes[r, c_]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        bg = backdrop(mod, plane)
        bm = brain_mask(mod, plane)
        ax.imshow(orient(bg, plane), cmap="gray",
                  vmin=np.percentile(bg[bm], 1), vmax=np.percentile(bg[bm], 99))

        g = load_group(meth, mod, plane)
        if g is not None:
            ax.imshow(orient(scale(g, bm), plane), cmap=CMAP,
                      vmin=0, vmax=1, alpha=ALPHA_MAX)

        if DRAW_CONTOURS:
            for roi, col in CONTOUR_ROIS.items():
                rm = roi_mask(roi, mod, plane)
                if rm is not None and rm.sum() > 0:
                    ax.contour(orient(rm.astype(float), plane), levels=[0.5],
                               colors=[col], linewidths=0.7)

        if r == 0:
            cc, _ = CONFIG[mod][plane]
            ax.set_title(f"{mod.upper()} {plane}\n({plane[0]} = {cc} mm)", fontsize=8)
        if c_ == 0:
            ax.set_ylabel(METHOD_LABEL.get(meth, meth), fontsize=10)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axes, fraction=0.014, pad=0.01)
cb.set_label(f"attribution (percentile-scaled, {PCT_LO}–{PCT_HI})", fontsize=8)
cb.ax.tick_params(labelsize=7)

if DRAW_CONTOURS:
    handles = [plt.Line2D([0], [0], color=c_, lw=1.4, label=r_.replace("_", " "))
               for r_, c_ in CONTOUR_ROIS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

for ext in ("png", "pdf"):
    p = f"{OUT_DIR}/fig_group_attribution.{ext}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print("wrote", p)
plt.close(fig)


# ------------------------------------------------- single-method compact figure
# One row, occlusion only — the version most likely to fit a single column.
best = "occlusion" if "occlusion" in methods else methods[0]
fig, axes = plt.subplots(1, len(STREAMS), figsize=(2.2 * len(STREAMS), 2.6),
                         constrained_layout=True)
for c_, (mod, plane) in enumerate(STREAMS):
    ax = axes[c_]
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    bg, bm = backdrop(mod, plane), brain_mask(mod, plane)
    ax.imshow(orient(bg, plane), cmap="gray",
              vmin=np.percentile(bg[bm], 1), vmax=np.percentile(bg[bm], 99))
    g = load_group(best, mod, plane)
    if g is not None:
        ax.imshow(orient(scale(g, bm), plane), cmap=CMAP, vmin=0, vmax=1,
                  alpha=ALPHA_MAX)
    if DRAW_CONTOURS:
        for roi, col in CONTOUR_ROIS.items():
            rm = roi_mask(roi, mod, plane)
            if rm is not None and rm.sum() > 0:
                ax.contour(orient(rm.astype(float), plane), levels=[0.5],
                           colors=[col], linewidths=0.7)
    cc, _ = CONFIG[mod][plane]
    ax.set_title(f"{mod.upper()} {plane} ({cc} mm)", fontsize=8)

for ext in ("png", "pdf"):
    p = f"{OUT_DIR}/fig_group_attribution_{best}.{ext}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print("wrote", p)
plt.close(fig)

print("\nIf the brains are sideways or mirrored, adjust DISPLAY and rerun. "
      "It affects nothing but the picture.")