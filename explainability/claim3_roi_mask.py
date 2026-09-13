
"""
Step 1 of Claim 3 — build ROI masks in the SAME 224x224 slice space the model
sees, plus the CN-group mean image used as the ablation fill.

Mirrors your slice-extraction code exactly: mm_to_vox(affine, ...) with the same
affine, same plane indexing, same zoom to 224. Masks use order=1 zoom + 0.5
threshold so they align with the bilinear-interpolated images.

Outputs (one set, shared by all subjects, since everything is in MNI):
  {MASK_DIR}/roi_{roi}_{mod}_{plane}_{center}_{delta}.npy    uint8 mask
  {MASK_DIR}/cnmean_{mod}_{plane}_{center}_{delta}.npy       float32 CN mean image
  {MASK_DIR}/roi_areas.csv                                   pixel area per mask

CN mean is computed from TRAIN-POOL CN subjects only. Using test CN subjects
would leak the test set into the ablation baseline.
"""

import os, csv, glob, numpy as np, nibabel as nib
from scipy.ndimage import zoom

# ---------------------------------------------------------------------- paths
REF_VOL   = "../AD_dataset/FINAL_CN/002_S_0295/002_S_0295_pet_suvr.nii.gz"  # <-- set one
TRAIN_CN  = "../AD_train/preprocess/CN"          # train-pool CN .npy slices
MASK_DIR  = "./roi_masks"; os.makedirs(MASK_DIR, exist_ok=True)

SIZE   = 224
MODS   = ["mri", "pet"]
PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {                                     # must match the training script
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
STREAMS = [(m, p) for m in MODS for p in PLANES]

# ------------------------------------------------------------------ roi specs
# ("sub"|"cort", [substrings matched against atlas label names])
ATLAS_ROIS = {
    # AD targets
    "hippocampus":    ("sub",  ["Left Hippocampus", "Right Hippocampus"]),
    "amygdala":       ("sub",  ["Left Amygdala", "Right Amygdala"]),
    "parahipp":       ("cort", ["Parahippocampal Gyrus"]),
    "post_cing":      ("cort", ["Cingulate Gyrus, posterior division"]),
    "precuneus":      ("cort", ["Precuneous Cortex"]),
    "inf_parietal":   ("cort", ["Angular Gyrus",
                                "Supramarginal Gyrus, posterior division"]),
    # spared controls
    "precentral":     ("cort", ["Precentral Gyrus"]),
    "occipital_pole": ("cort", ["Occipital Pole"]),
    "lingual":        ("cort", ["Lingual Gyrus"]),
}

# exact-area disc controls: name -> MNI (x, y, z) mm centre.
# Area is matched per-slice to DISC_MATCH_ROI, so these are true size controls.
DISC_ROIS = {
    "disc_occipital": (  0, -95,   0),
    "disc_motor":     (-38, -20,  55),
    "disc_frontal":   ( 30,  45,  15),
}
DISC_MATCH_ROI = "hippocampus"

N_RANDOM_DISCS = 20      # random in-brain discs -> null distribution of ROI effect
RANDOM_SEED    = 0

ATLAS_SRC = "nilearn"    # "nilearn" (downloads) or "file"
ATLAS_FILES = {          # used when ATLAS_SRC == "file"
    "sub":  "/usr/share/fsl/data/atlases/HarvardOxford/"
            "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
    "cort": "/usr/share/fsl/data/atlases/HarvardOxford/"
            "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
}
ATLAS_LABEL_XML = {      # FSL label lists, only needed for ATLAS_SRC == "file"
    "sub":  "/usr/share/fsl/data/atlases/HarvardOxford-Subcortical.xml",
    "cort": "/usr/share/fsl/data/atlases/HarvardOxford-Cortical.xml",
}


def mm_to_vox(affine, x, y, z):
    return np.round(np.linalg.inv(affine) @ np.array([x, y, z, 1.0]))[:3].astype(int)


def resize2d(sl, order=1):
    return zoom(sl, (SIZE / sl.shape[0], SIZE / sl.shape[1]), order=order).astype("float32")


def take_slice(vol, plane, affine, coord):
    """Same indexing as your extraction script."""
    if plane == "axial":
        return vol[:, :, mm_to_vox(affine, 0, 0, coord)[2]]
    if plane == "coronal":
        return vol[:, mm_to_vox(affine, 0, coord, 0)[1], :]
    return vol[mm_to_vox(affine, coord, 0, 0)[0], :, :]


# ----------------------------------------------------------------- load atlas
ref = nib.load(REF_VOL)
print("reference grid:", ref.shape, "| voxel size", ref.header.get_zooms()[:3])


def load_atlas(kind):
    from nilearn.image import resample_to_img
    if ATLAS_SRC == "nilearn":
        from nilearn.datasets import fetch_atlas_harvard_oxford
        name = "sub-maxprob-thr25-1mm" if kind == "sub" else "cort-maxprob-thr25-1mm"
        a = fetch_atlas_harvard_oxford(name)
        img, labels = a.maps, list(a.labels)
    else:
        import xml.etree.ElementTree as ET
        img = nib.load(ATLAS_FILES[kind])
        root = ET.parse(ATLAS_LABEL_XML[kind]).getroot()
        labels = ["Background"] + [l.text for l in root.iter("label")]
    img = resample_to_img(img, ref, interpolation="nearest", force_resample=True)
    return np.asarray(img.dataobj).astype(int), labels


ATLAS = {}
for kind in {v[0] for v in ATLAS_ROIS.values()}:
    ATLAS[kind] = load_atlas(kind)
    print(f"atlas '{kind}': {len(ATLAS[kind][1])} labels, "
          f"vol shape {ATLAS[kind][0].shape}")


def roi_volume(spec):
    kind, patterns = spec
    data, labels = ATLAS[kind]
    idx = [i for i, lab in enumerate(labels)
           if any(pat.lower() in lab.lower() for pat in patterns)]
    if not idx:
        raise KeyError(f"no atlas label matched {patterns}")
    print(f"    matched: {[labels[i] for i in idx]}")
    return np.isin(data, idx)


# ------------------------------------------------------- brain mask reference
# used to constrain the random discs to in-brain locations
brain_vol = np.abs(np.asarray(ref.dataobj)) > 1e-6


# ------------------------------------------------------------------ disc util
def disc_mask(shape, cy, cx, area_px):
    """Filled circle of ~area_px pixels centred at (cy, cx)."""
    r = np.sqrt(max(area_px, 1) / np.pi)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def vox_to_pixel(vox_ij, src_shape):
    """Map an in-plane voxel index to the resized 224x224 grid."""
    return (vox_ij[0] * SIZE / src_shape[0], vox_ij[1] * SIZE / src_shape[1])


def inplane_index(plane, affine, mm_xyz):
    """In-plane (row, col) voxel index of an MNI point, per plane convention."""
    v = mm_to_vox(affine, *mm_xyz)
    if plane == "axial":     return (v[0], v[1])
    if plane == "coronal":   return (v[0], v[2])
    return (v[1], v[2])


def inplane_shape(plane, shape):
    if plane == "axial":     return (shape[0], shape[1])
    if plane == "coronal":   return (shape[0], shape[2])
    return (shape[1], shape[2])


# ------------------------------------------------------------- build the masks
rows = []
roi_slices = {}          # (roi, mod, plane, delta) -> mask

print("\nbuilding atlas ROI masks")
for roi, spec in ATLAS_ROIS.items():
    print(f"  {roi}")
    vol = roi_volume(spec)
    for mod, plane in STREAMS:
        c, d = CONFIG[mod][plane]
        for k in (-d, 0, d):
            sl = take_slice(vol.astype("float32"), plane, ref.affine, c + k)
            m  = resize2d(sl, order=1) > 0.5
            roi_slices[(roi, mod, plane, k)] = m
            np.save(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_{k}.npy", m.astype("uint8"))
            rows.append(dict(roi=roi, mod=mod, plane=plane, center=c, delta=k,
                             area_px=int(m.sum()),
                             frac=round(float(m.mean()), 5)))

print("\nbuilding area-matched disc controls")
rng = np.random.default_rng(RANDOM_SEED)
for roi, mm_xyz in DISC_ROIS.items():
    print(f"  {roi}")
    for mod, plane in STREAMS:
        c, d = CONFIG[mod][plane]
        for k in (-d, 0, d):
            area = int(roi_slices[(DISC_MATCH_ROI, mod, plane, k)].sum())
            ij   = inplane_index(plane, ref.affine, mm_xyz)
            cy, cx = vox_to_pixel(ij, inplane_shape(plane, ref.shape))
            m = disc_mask((SIZE, SIZE), cy, cx, area) if area > 0 \
                else np.zeros((SIZE, SIZE), bool)
            np.save(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_{k}.npy", m.astype("uint8"))
            rows.append(dict(roi=roi, mod=mod, plane=plane, center=c, delta=k,
                             area_px=int(m.sum()), frac=round(float(m.mean()), 5)))

print("\nbuilding random in-brain disc controls")
for r in range(N_RANDOM_DISCS):
    roi = f"rand{r:02d}"
    for mod, plane in STREAMS:
        c, d = CONFIG[mod][plane]
        for k in (-d, 0, d):
            area = int(roi_slices[(DISC_MATCH_ROI, mod, plane, k)].sum())
            bsl  = resize2d(take_slice(brain_vol.astype("float32"), plane,
                                       ref.affine, c + k), order=1) > 0.5
            ys, xs = np.nonzero(bsl)
            if area == 0 or len(ys) == 0:
                m = np.zeros((SIZE, SIZE), bool)
            else:
                t = rng.integers(len(ys))
                m = disc_mask((SIZE, SIZE), ys[t], xs[t], area) & bsl
            np.save(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_{k}.npy", m.astype("uint8"))
            rows.append(dict(roi=roi, mod=mod, plane=plane, center=c, delta=k,
                             area_px=int(m.sum()), frac=round(float(m.mean()), 5)))

with open(f"{MASK_DIR}/roi_areas.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# --------------------------------------------------- CN-group mean fill images
print("\ncomputing CN-group mean images from train pool")
cn_dirs = [d.rstrip("/") for d in sorted(glob.glob(f"{TRAIN_CN}/*/"))]
print(f"  {len(cn_dirs)} CN subjects")

for mod, plane in STREAMS:
    c, d = CONFIG[mod][plane]
    for k in (-d, 0, d):
        acc, n = None, 0
        for sd in cn_dirs:
            f = f"{sd}/{os.path.basename(sd)}_{mod}_{plane}_{c}_{k}.npy"
            if not os.path.exists(f):
                continue
            x = np.load(f).astype("float64")
            acc = x if acc is None else acc + x
            n += 1
        assert n, f"no CN slices for {mod}/{plane}/{c}/{k}"
        np.save(f"{MASK_DIR}/cnmean_{mod}_{plane}_{c}_{k}.npy",
                (acc / n).astype("float32"))
        print(f"  {mod:>3}/{plane:<9} c={c:>4} d={k:>4}  n={n}")


# ------------------------------------------------------------ area sanity dump
print("\n--- ROI area (px) per stream, centre slice only ---")
print(" " * 18 + "".join(f"{m[:3]}/{p[:3]:<9}" for m, p in STREAMS))
for roi in list(ATLAS_ROIS) + list(DISC_ROIS):
    line = f"{roi:<18}"
    for mod, plane in STREAMS:
        c, d = CONFIG[mod][plane]
        a = int(np.load(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_0.npy").sum())
        line += f"{a:<13}"
    print(line)
print(f"\nwrote masks to {MASK_DIR}/")
print("An ROI with area 0 in a stream simply is not on that slice — expected, "
      "and the ablation script reports it as such.")
