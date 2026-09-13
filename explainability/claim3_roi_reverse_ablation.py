"""
Claim 3c — reverse ROI ablation (sufficiency).

Forward ablation asks "is this region NECESSARY?" and the answer was no: the
model is redundant across six streams, so removing any single ROI costs little.

Reverse ablation asks "is this region SUFFICIENT?" — keep ONLY the ROI, replace
every other pixel with the CN-group mean image, and measure how much of the
model's discrimination survives. Redundancy cannot rescue the model here,
so the effects are large and directly interpretable.

Conditions (which modality is allowed to carry the kept ROI):
  both        ROI kept in MRI and PET, everything else -> CN mean
  mri_only    ROI kept in MRI only, all PET -> CN mean
  pet_only    ROI kept in PET only, all MRI -> CN mean

References:
  intact      unmodified input                      -> upper bound
  floor       EVERYTHING replaced by CN mean        -> must give AUC ~0.500
                                                      (identical input for every
                                                       subject, so it is a
                                                       hard implementation check)

Primary statistic:
  retained = (AUC_roi - 0.5) / (AUC_intact - 0.5)
             fraction of above-chance discrimination recovered from the ROI alone.

Uses the same masks as the forward script. Run build_roi_masks.py first.
"""

import glob, os, csv, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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
ROI_COMBOS = {
    "medial_temporal":  ["hippocampus", "amygdala", "parahipp"],
    "posterior_dmn":    ["post_cing", "precuneus", "inf_parietal"],
    "ctrl_composite":   ["precentral", "occipital_pole", "lingual"],
    # bigger control, closer in area to posterior_dmn on the PET slices
    "ctrl_composite_xl": ["precentral", "occipital_pole", "lingual",
                          "disc_occipital", "disc_motor", "disc_frontal"],
    # both target networks together -> how much of the model is AD-network signal
    "ad_networks":      ["hippocampus", "amygdala", "parahipp",
                         "post_cing", "precuneus", "inf_parietal"],
}

COND_ROIS = {
    "both":     ["ad_networks", "medial_temporal", "posterior_dmn",
                 "hippocampus", "ctrl_composite", "ctrl_composite_xl"],
    "mri_only": ["medial_temporal", "hippocampus", "amygdala", "parahipp",
                 "ctrl_composite", "ctrl_composite_xl",
                 "precentral", "occipital_pole",
                 "disc_occipital", "disc_motor", "disc_frontal"],
    "pet_only": ["posterior_dmn", "post_cing", "precuneus", "inf_parietal",
                 "ctrl_composite", "ctrl_composite_xl",
                 "precentral", "occipital_pole",
                 "disc_occipital", "disc_motor", "disc_frontal"],
}

N_RANDOM = 20
RANDOM_ROIS = [f"rand{r:02d}" for r in range(N_RANDOM)]
RANDOM_IN_CONDS = ["mri_only", "pet_only"]

FEATHER_SIGMA = 1.0

CONDITIONS = {            # name -> modalities allowed to KEEP the ROI
    "both":     ("mri", "pet"),
    "mri_only": ("mri",),
    "pet_only": ("pet",),
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
    """(3,H,W) alpha in [0,1], 1 INSIDE the ROI. Composites by union."""
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


def apply_stream(x, mod, plane, roi, keep_mods):
    """RAW (3,H,W).

    roi is None      -> intact (upper bound)
    roi == "_floor"  -> everything CN mean (floor)
    otherwise        -> KEEP the ROI, everything else CN mean.
                        A modality not in keep_mods becomes entirely CN mean,
                        and so does a stream where the ROI has zero area.
    """
    if roi is None:
        return x
    cm = get_cnmean(mod, plane)
    if roi == "_floor" or mod not in keep_mods:
        return cm.copy()
    a = get_mask(roi, mod, plane)          # 1 inside ROI -> keep subject pixels there
    return x * a + cm * (1.0 - a)


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
print(f"{len(subjects)} paired test subjects", np.bincount(labels), f"({dropped} dropped)")


class TestSet(Dataset):
    def __init__(self, roi=None, keep_mods=()):
        self.roi, self.keep_mods = roi, keep_mods

    def __len__(self):
        return len(stems)

    def __getitem__(self, j):
        xs = []
        for mod, plane in STREAMS:
            lo, hi = NORM[mod]
            x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
            x = apply_stream(x, mod, plane, self.roi, self.keep_mods)
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
def predict(m, dl):
    m.eval(); P = []
    for xs, _ in dl:
        P.append(torch.softmax(m([x.to(DEV) for x in xs]), 1)[:, 1].cpu())
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


def boot_retained_ci(y, p_roi, p_intact):
    """Paired CI on the retained fraction, same resamples for both."""
    r = []
    for i in boot_idx(y):
        a0 = roc_auc_score(y[i], p_intact[i])
        a1 = roc_auc_score(y[i], p_roi[i])
        if abs(a0 - 0.5) > 1e-6:
            r.append((a1 - 0.5) / (a0 - 0.5))
    return np.percentile(r, [2.5, 97.5])


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


def run(roi, keep_mods):
    dl = DataLoader(TestSet(roi, keep_mods), BS, shuffle=False, num_workers=4)
    return np.mean([predict(m, dl) for m in models], axis=0)


rows, null_by_cond = [], {}

# ------------------------------------------------------------------ references
print("\n================ references ================")
P_INTACT = run(None, ())
auc_intact = roc_auc_score(labels, P_INTACT)
lo, hi = boot_auc_ci(labels, P_INTACT)
np.save(f"{OUT}/rev_probs_intact.npy", P_INTACT)
print(f"intact   AUC {auc_intact:.3f} [{lo:.3f}, {hi:.3f}]")

P_FLOOR = run("_floor", ())
auc_floor = roc_auc_score(labels, P_FLOOR)
np.save(f"{OUT}/rev_probs_floor.npy", P_FLOOR)
print(f"floor    AUC {auc_floor:.3f}   (all CN mean; must be 0.500, "
      f"prob spread {P_FLOOR.std():.2e})")
if abs(auc_floor - 0.5) > 1e-6 or P_FLOOR.std() > 1e-6:
    print("  !! floor is not constant — the CN-mean fill is not covering "
          "every pixel. Check mask/fill alignment before trusting anything below.")

rows.append(dict(cond="-", roi="intact", area_px=0, AUC=auc_intact,
                 AUC_lo=lo, AUC_hi=hi, retained=1.0, ret_lo=1.0, ret_hi=1.0,
                 spearman_vs_intact=1.0, BAL_ACC=balanced_accuracy_score(
                     labels, (P_INTACT >= THR).astype(int)), z_null=np.nan))
rows.append(dict(cond="-", roi="floor", area_px=0, AUC=auc_floor,
                 AUC_lo=np.nan, AUC_hi=np.nan, retained=0.0, ret_lo=np.nan,
                 ret_hi=np.nan, spearman_vs_intact=np.nan,
                 BAL_ACC=balanced_accuracy_score(labels, (P_FLOOR >= THR).astype(int)),
                 z_null=np.nan))


def evaluate(cond, roi, verbose=True):
    keep_mods = CONDITIONS[cond]
    P = run(roi, keep_mods)
    area = sum(roi_area(roi, m_, p_) for m_, p_ in STREAMS if m_ in keep_mods)
    auc = roc_auc_score(labels, P)
    alo, ahi = boot_auc_ci(labels, P)
    ret = (auc - 0.5) / (auc_intact - 0.5)
    rlo, rhi = boot_retained_ci(labels, P, P_INTACT)
    rho = spearmanr(logit(P), logit(P_INTACT)).statistic

    row = dict(cond=cond, roi=roi, area_px=area, AUC=auc, AUC_lo=alo, AUC_hi=ahi,
               retained=ret, ret_lo=rlo, ret_hi=rhi, spearman_vs_intact=float(rho),
               BAL_ACC=balanced_accuracy_score(labels, (P >= THR).astype(int)),
               z_null=np.nan)
    if verbose:
        flag = "  [ROI NOT ON THESE SLICES]" if area == 0 else ""
        print(f"  {roi:<20}area {area:>6}  AUC {auc:.3f} [{alo:.3f},{ahi:.3f}]  "
              f"retained {ret:6.1%} [{rlo:5.1%},{rhi:5.1%}]  "
              f"rho {rho:+.3f}{flag}")
    np.save(f"{OUT}/rev_probs_{cond}_{roi}.npy", P)
    return row


for cond in CONDITIONS:
    print(f"\n--- {cond}  (ROI kept in: {'+'.join(CONDITIONS[cond])}) ---")
    cond_rows = [evaluate(cond, roi) for roi in COND_ROIS[cond]]

    if cond in RANDOM_IN_CONDS:
        nr = [evaluate(cond, roi, verbose=False) for roi in RANDOM_ROIS]
        nret = np.array([r["retained"] for r in nr])
        null_by_cond[cond] = nret
        print(f"  [null] retained {nret.mean():.1%} ± {nret.std():.1%}   "
              f"range [{nret.min():.1%}, {nret.max():.1%}]")
        s = nret.std()
        for r in cond_rows:
            r["z_null"] = (r["retained"] - nret.mean()) / s if s > 0 else np.nan
        cond_rows += nr
    rows += cond_rows

with open(f"{OUT}/roi_reverse_ablation_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# -------------------------------------------------------------------- summary
print("\n================ reverse ablation summary ================")
print(f"intact AUC {auc_intact:.3f}   floor AUC {auc_floor:.3f}")
print(f"\n{'cond':<10}{'ROI kept':<20}{'area':>7}{'AUC':>8}{'retained':>11}"
      f"{'95% CI':>18}{'z_null':>9}")
for r in rows:
    if r["roi"].startswith("rand"):
        continue
    ci = (f"[{r['ret_lo']:.1%},{r['ret_hi']:.1%}]"
          if np.isfinite(r.get("ret_lo", np.nan)) else "-")
    z = f"{r['z_null']:+.2f}" if np.isfinite(r.get("z_null", np.nan)) else ""
    print(f"{r['cond']:<10}{r['roi']:<20}{r['area_px']:>7}{r['AUC']:>8.3f}"
          f"{r['retained']:>10.1%}{ci:>18}{z:>9}")

print(f"\nwrote {OUT}/roi_reverse_ablation_{tag}.csv")