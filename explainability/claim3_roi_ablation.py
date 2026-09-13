"""
Claim 3b — ROI ablation WITHIN a single modality.

The full model is redundant across modalities, so removing a small ROI from one
of them barely moves AUC. This script silences one modality entirely, then
ablates ROIs inside the surviving one, so the effect is measured without the
other modality compensating.

Conditions:
  both        all six streams intact                     (reference)
  mri_only    PET streams silenced                       (baseline for MRI ROIs)
  pet_only    MRI streams silenced                       (baseline for PET ROIs)

For each condition, every ROI is ablated (replaced by CN-group mean) and dAUC is
computed against THAT CONDITION'S OWN baseline, not against the full model.

SILENCE_MODE
  "cnmean"  silenced modality's images replaced by the CN-group mean image.
            Streams still produce in-distribution features carrying no
            subject-specific information. Default.
  "zero"    silenced modality's pooled features multiplied by 0 (the `keep`
            path). Simpler, but off-manifold: with P_DROP=0.0 the head never
            saw a zeroed stream during training.
Run both; if they agree the result does not depend on the choice.

Requires build_roi_masks.py to have been run.
"""

import glob, os, csv, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

TEST_ROOT = "./preprocess_test"
MASK_DIR  = "./roi_masks"
MODS, CLASSES = ["mri", "pet"], ["CN", "AD"]
PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
NORM = {"mri": (-3.0, 3.0), "pet": (-3.0, 3.0)}
P_DROP, PRETRAIN = 0.0, True
NFOLDS, BS = 1, 16
N_BOOT = 2000
THR = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./test"; os.makedirs(OUT, exist_ok=True)
MEAN, STD = 0.449, 0.226

STREAMS = [(m, p) for m in MODS for p in PLANES]

# ---------------------------------------------------------------- ablation cfg
SILENCE_MODE = "zero"          # "cnmean" | "zero"

# composites: union of several atlas ROIs, ablated as one lesion
ROI_COMBOS = {
    "medial_temporal": ["hippocampus", "amygdala", "parahipp"],
    "posterior_dmn":   ["post_cing", "precuneus", "inf_parietal"],
    "ctrl_composite":  ["precentral", "occipital_pole", "lingual"],
}

# which ROIs to test in each condition. MRI slices carry medial temporal,
# PET slices carry posterior DMN — see the area table.
COND_ROIS = {
    "both":     ["hippocampus", "medial_temporal", "post_cing", "precuneus",
                 "posterior_dmn", "precentral", "occipital_pole",
                 "ctrl_composite", "disc_occipital", "disc_motor"],
    "mri_only": ["hippocampus", "amygdala", "parahipp", "medial_temporal",
                 "precentral", "occipital_pole", "ctrl_composite",
                 "disc_occipital", "disc_motor", "disc_frontal"],
    "pet_only": ["hippocampus", "post_cing", "precuneus", "inf_parietal",
                 "posterior_dmn", "precentral", "occipital_pole",
                 "ctrl_composite", "disc_occipital", "disc_motor", "disc_frontal"],
}

N_RANDOM = 20
RANDOM_ROIS = [f"rand{r:02d}" for r in range(N_RANDOM)]
RANDOM_IN_CONDS = ["mri_only", "pet_only"]      # empirical null per condition

FEATHER_SIGMA = 1.0

CONDITIONS = {            # name -> modalities that are SILENCED
    "both":     (),
    "mri_only": ("pet",),
    "pet_only": ("mri",),
}

_MASK, _CNMEAN = {}, {}


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def _raw_mask(roi, mod, plane):
    c, d = CONFIG[mod][plane]
    return np.stack([np.load(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_{k}.npy")
                     for k in (-d, 0, d)]).astype("float32")


def get_mask(roi, mod, plane):
    """(3,H,W) float alpha in [0,1]. Handles composites by union."""
    key = (roi, mod, plane)
    if key not in _MASK:
        parts = ROI_COMBOS.get(roi, [roi])
        m = np.clip(sum(_raw_mask(r, mod, plane) for r in parts), 0, 1)
        if FEATHER_SIGMA > 0:
            m = np.clip(np.stack([gaussian_filter(mi, FEATHER_SIGMA) for mi in m]), 0, 1)
        _MASK[key] = m
    return _MASK[key]


def get_cnmean(mod, plane):
    key = (mod, plane)
    if key not in _CNMEAN:
        c, d = CONFIG[mod][plane]
        _CNMEAN[key] = np.stack([np.load(f"{MASK_DIR}/cnmean_{mod}_{plane}_{c}_{k}.npy")
                                 for k in (-d, 0, d)]).astype("float32")
    return _CNMEAN[key]


def roi_area(roi, mod, plane):
    c, d = CONFIG[mod][plane]
    parts = ROI_COMBOS.get(roi, [roi])
    u = np.clip(sum(np.load(f"{MASK_DIR}/roi_{r}_{mod}_{plane}_{c}_0.npy")
                    for r in parts), 0, 1)
    return int(u.sum())


def apply_stream(x, mod, plane, roi, ablate_mods, silence):
    """RAW (3,H,W) -> silenced and/or ROI-ablated."""
    if mod in silence and SILENCE_MODE == "cnmean":
        return get_cnmean(mod, plane).copy()          # whole modality -> CN mean
    if roi is None or mod not in ablate_mods or mod in silence:
        return x
    a = get_mask(roi, mod, plane)
    return x * (1.0 - a) + get_cnmean(mod, plane) * a


def keep_vector(silence):
    """Feature-level silencing for SILENCE_MODE == 'zero'."""
    if SILENCE_MODE != "zero" or not silence:
        return None
    return [0.0 if m in silence else 1.0 for m, _ in STREAMS]


def complete(subj_dir):
    stem = f"{subj_dir}/{os.path.basename(subj_dir)}"
    return all(os.path.exists(f) for m, p in STREAMS for f in files_for(stem, m, p))


subjects, labels, dropped = [], [], 0
for y, c in enumerate(CLASSES):
    for d in sorted(glob.glob(f"{TEST_ROOT}/{c}/*/")):
        d = d.rstrip("/")
        if complete(d):
            subjects.append(d); labels.append(y)
        else:
            dropped += 1
subjects, labels = np.array(subjects), np.array(labels)
stems = np.array([f"{d}/{os.path.basename(d)}" for d in subjects])
print(f"{len(subjects)} paired test subjects", np.bincount(labels),
      f"({dropped} dropped)   SILENCE_MODE={SILENCE_MODE}")


class TestSet(Dataset):
    def __init__(self, roi=None, ablate_mods=(), silence=()):
        self.roi, self.ablate_mods, self.silence = roi, ablate_mods, silence

    def __len__(self):
        return len(stems)

    def __getitem__(self, j):
        xs = []
        for mod, plane in STREAMS:
            lo, hi = NORM[mod]
            x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
            x = apply_stream(x, mod, plane, self.roi, self.ablate_mods, self.silence)
            x = np.clip(x, lo, hi); x = (x - lo) / (hi - lo)
            xs.append((torch.from_numpy(x).float() - MEAN) / STD)
        return xs, labels[j]


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc  = nn.ModuleList([torchvision.models.vgg16_bn(weights=None).features
                                   for _ in STREAMS])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * len(STREAMS), len(CLASSES)))

    def forward(self, xs, keep=None):
        f = [self.pool(e(x)).flatten(1) for e, x in zip(self.enc, xs)]
        if keep is not None:
            f = [fi * k for fi, k in zip(f, keep)]
        return self.head(torch.cat(f, 1))


@torch.no_grad()
def predict(m, dl, keep=None):
    m.eval(); P = []
    for xs, _ in dl:
        P.append(torch.softmax(m([x.to(DEV) for x in xs], keep), 1)[:, 1].cpu())
    return torch.cat(P).numpy()


# ------------------------------------------------- paired bootstrap machinery
_BOOT_IDX = None


def boot_idx(y):
    global _BOOT_IDX
    if _BOOT_IDX is None:
        rng = np.random.default_rng(0); B = []
        while len(B) < N_BOOT:
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                B.append(i)
        _BOOT_IDX = np.array(B)
    return _BOOT_IDX


def boot_auc_ci(y, p):
    b = [roc_auc_score(y[i], p[i]) for i in boot_idx(y)]
    return np.percentile(b, [2.5, 97.5])


def paired_dauc(y, p_ref, p_abl):
    d = np.array([roc_auc_score(y[i], p_ref[i]) - roc_auc_score(y[i], p_abl[i])
                  for i in boot_idx(y)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, float((d <= 0).mean())


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------- models
tag = f"mm_fusion_p{P_DROP}" + ("_pre" if PRETRAIN else "_imnet")
models = []
for fold in range(NFOLDS):
    ck = f"./models/vgg16_{tag}_fold{fold}.pt"
    if not os.path.exists(ck):
        print(f"fold {fold}: missing {ck} — skipped"); continue
    m = Fusion(); m.load_state_dict(torch.load(ck, map_location="cpu"))
    models.append(m.to(DEV))
assert models, "no checkpoints found"


def run(roi, ablate_mods, silence):
    dl = DataLoader(TestSet(roi, ablate_mods, silence), BS, shuffle=False, num_workers=4)
    k = keep_vector(silence)
    return np.mean([predict(m, dl, k) for m in models], axis=0)


is_ad = labels == 1
rows, base_P, null_by_cond = [], {}, {}

# ------------------------------------------------------- per-condition baselines
print("\n================ condition baselines ================")
for cond, silence in CONDITIONS.items():
    P = run(None, (), silence)
    base_P[cond] = P
    a = roc_auc_score(labels, P)
    lo, hi = boot_auc_ci(labels, P)
    np.save(f"{OUT}/probs_{cond}_baseline_{SILENCE_MODE}.npy", P)
    print(f"{cond:<10} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"bal-acc {balanced_accuracy_score(labels, (P >= THR).astype(int)):.3f}")
    rows.append(dict(cond=cond, roi="baseline", area_px=0, AUC=a,
                     dAUC=0.0, dAUC_lo=0.0, dAUC_hi=0.0, p_dAUC=np.nan,
                     d_logit_AD=0.0, p_wilcoxon=np.nan, d_pAD=0.0, d_pCN=0.0))


def evaluate(cond, roi, verbose=True):
    silence = CONDITIONS[cond]
    ablate  = tuple(m for m in MODS if m not in silence)     # ablate live modality only
    P  = run(roi, ablate, silence)
    P0 = base_P[cond]
    area = sum(roi_area(roi, m_, p_) for m_, p_ in STREAMS if m_ in ablate)
    d, lo, hi, pv = paired_dauc(labels, P0, P)

    dl_ad = logit(P0[is_ad]) - logit(P[is_ad])               # per-AD-subject evidence drop
    try:
        pw = wilcoxon(dl_ad, alternative="greater").pvalue
    except ValueError:
        pw = np.nan

    row = dict(cond=cond, roi=roi, area_px=area, AUC=roc_auc_score(labels, P),
               dAUC=d, dAUC_lo=lo, dAUC_hi=hi, p_dAUC=pv,
               d_logit_AD=float(np.mean(dl_ad)), p_wilcoxon=float(pw),
               d_pAD=float((P0[is_ad] - P[is_ad]).mean()),
               d_pCN=float((P0[~is_ad] - P[~is_ad]).mean()))
    if verbose:
        flag = "  [NOT ON THESE SLICES]" if area == 0 else ""
        print(f"  {roi:<18}area {area:>6}  AUC {row['AUC']:.3f}  "
              f"dAUC {d:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"dlogit_AD {row['d_logit_AD']:+.3f} (p={pw:.4f})  "
              f"d_pAD {row['d_pAD']:+.3f}{flag}")
    np.save(f"{OUT}/probs_{cond}_{roi}_{SILENCE_MODE}.npy", P)
    return row


for cond in CONDITIONS:
    print(f"\n--- {cond}  (silenced: {CONDITIONS[cond] or 'none'}) ---")
    for roi in COND_ROIS[cond]:
        rows.append(evaluate(cond, roi))

    if cond in RANDOM_IN_CONDS:
        nd, ndl = [], []
        for roi in RANDOM_ROIS:
            r = evaluate(cond, roi, verbose=False)
            nd.append(r["dAUC"]); ndl.append(r["d_logit_AD"]); rows.append(r)
        null_by_cond[cond] = (np.array(nd), np.array(ndl))
        print(f"  [null] dAUC {np.mean(nd):+.3f} ± {np.std(nd):.3f}   "
              f"dlogit_AD {np.mean(ndl):+.3f} ± {np.std(ndl):.3f}")

with open(f"{OUT}/roi_ablation_silenced_{SILENCE_MODE}_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# -------------------------------------------------------------------- summary
print("\n================ summary ================")
print(f"{'cond':<10}{'ROI':<18}{'area':>7}{'AUC':>8}{'dAUC':>9}"
      f"{'dlogit_AD':>11}{'p_wilc':>9}{'z_null':>9}")
for r in rows:
    if r["roi"].startswith("rand"):
        continue
    z = ""
    if r["cond"] in null_by_cond and r["roi"] != "baseline":
        nd, ndl = null_by_cond[r["cond"]]
        s = ndl.std()
        z = f"{(r['d_logit_AD'] - ndl.mean()) / s:+.2f}" if s > 0 else "-"
    print(f"{r['cond']:<10}{r['roi']:<18}{r['area_px']:>7}{r['AUC']:>8.3f}"
          f"{r['dAUC']:>+9.3f}{r['d_logit_AD']:>+11.3f}"
          f"{r['p_wilcoxon'] if np.isfinite(r['p_wilcoxon']) else float('nan'):>9.4f}{z:>9}")

print(f"\nwrote {OUT}/roi_ablation_silenced_{SILENCE_MODE}_{tag}.csv")
