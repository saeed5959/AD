"""
Claim 1 — shortcut control: does the fusion model need brain tissue content,
or can it classify from brain shape / non-brain signal alone?

Modes (applied at test time to the selected modalities):
  intact      unmodified input (reference)
  silhouette  brain interior filled with a constant -> outline preserved, tissue destroyed
  exterior    brain interior zeroed -> only the region outside the mask survives
  blank       whole image constant -> model's prior / floor reference

Ablation can be applied to MRI only, PET only, or both, so you can see which
modality carries any residual shape-based signal.

NOTE: this is evaluation-only ablation. It answers "does the trained model USE
this information". It does NOT answer "is the information present at all" —
for that, retrain on ablated images (import apply_ablation into your training
script). Run this first; retrain only for the modes that stay above chance.
"""

import glob, os, csv, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_fill_holes
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             balanced_accuracy_score, f1_score, matthews_corrcoef,
                             brier_score_loss)

TEST_ROOT = "./preprocess_test"
MODS, CLASSES = ["mri", "pet"], ["CN", "AD"]
PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {                                                  # must match the training script
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
NORM = {"mri": (-3.0, 3.0), "pet": (-3.0, 3.0)}              # must match the training script
P_DROP, PRETRAIN = 0.0, True
NFOLDS, BS = 1, 16
N_BOOT = 2000
THR = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./test"; os.makedirs(OUT, exist_ok=True)
MEAN, STD = 0.449, 0.226

STREAMS = [(m, p) for m in MODS for p in PLANES]

# ---------------------------------------------------------------- ablation cfg
# (mode, which modalities to ablate).  intact/both is the reference row.
RUNS = [
    ("intact",     ()),
    ("silhouette", ("mri", "pet")),
    ("silhouette", ("mri",)),
    ("silhouette", ("pet",)),
    ("exterior",   ("mri", "pet")),
    ("blank",      ("mri", "pet")),
]

# values in RAW units (before NORM clipping/scaling), so they stay interpretable
BG_FILL  = {"mri": 0.0, "pet": 0.0}   # background / removed-region value
SIL_FILL = {"mri": 1.0, "pet": 1.0}   # constant that replaces brain tissue
MASK_EPS = {"mri": 0.0, "pet": 0.05}  # |x| > eps  ->  inside brain

MASK_FROM = "data"   # "data" = derive from the slice itself; "file" = load mask .npy


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def mask_files_for(stem, mod, plane):
    """If MASK_FROM='file', expects HD-BET masks warped to MNI and sliced with
    the same code as the images, named *_mask.npy alongside them."""
    return [f.replace(".npy", "_mask.npy") for f in files_for(stem, mod, plane)]


def brain_mask(x, mod, stem=None, plane=None):
    """x: (3,H,W) raw slices -> bool mask, True inside the brain."""
    if MASK_FROM == "file":
        return np.stack([np.load(f) for f in mask_files_for(stem, mod, plane)]) > 0.5
    m = np.abs(x) > MASK_EPS[mod]
    return np.stack([binary_fill_holes(mi) for mi in m])


def apply_ablation(x, mod, mode, ablate_mods, stem=None, plane=None):
    """x: (3,H,W) raw slices. Returns ablated copy in the same raw units."""
    if mode == "intact" or mod not in ablate_mods:
        return x
    if mode == "blank":
        return np.full_like(x, BG_FILL[mod])
    m = brain_mask(x, mod, stem, plane)
    if mode == "exterior":                       # kill the brain, keep what's outside
        return np.where(m, BG_FILL[mod], x)
    if mode == "silhouette":                     # keep only the outline
        return np.where(m, SIL_FILL[mod], BG_FILL[mod]).astype(x.dtype)
    raise ValueError(mode)


# ------------------------------------------------------------------- subjects
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
      f"({dropped} dropped: missing a modality/plane)")


class TestSet(Dataset):
    def __init__(self, mode="intact", ablate_mods=()):
        self.mode, self.ablate_mods = mode, ablate_mods

    def __len__(self):
        return len(stems)

    def __getitem__(self, j):
        xs = []
        for mod, plane in STREAMS:
            lo, hi = NORM[mod]
            x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
            x = apply_ablation(x, mod, self.mode, self.ablate_mods, stems[j], plane)
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


def boot_auc(y, p):
    rng = np.random.default_rng(0); b = []
    while len(b) < N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], p[i]))
    return np.percentile(b, [2.5, 97.5])


def report(y, p, thr=THR, name=""):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = boot_auc(y, p)
    m = dict(
        run=name, n=len(y), n_CN=int((1 - y).sum()), n_AD=int(y.sum()),
        AUC=roc_auc_score(y, p), AUC_lo=lo, AUC_hi=hi,
        AUPRC=average_precision_score(y, p),
        ACC=(pred == y).mean(),
        BAL_ACC=balanced_accuracy_score(y, pred),
        SENS=tp / (tp + fn) if tp + fn else np.nan,
        SPEC=tn / (tn + fp) if tn + fp else np.nan,
        F1=f1_score(y, pred, zero_division=0),
        MCC=matthews_corrcoef(y, pred),
        BRIER=brier_score_loss(y, p),
        TN=tn, FP=fp, FN=fn, TP=tp,
    )
    print(f"\n===== {name} =====")
    print(f"AUC       {m['AUC']:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"Accuracy  {m['ACC']:.3f}    Balanced {m['BAL_ACC']:.3f}")
    print(f"Sens      {m['SENS']:.3f}    Spec     {m['SPEC']:.3f}")
    print(f"MCC       {m['MCC']:.3f}    Brier    {m['BRIER']:.3f}")
    print(f"confusion TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return m


# ----------------------------------------------------------------------- main
tag = f"mm_fusion_p{P_DROP}" + ("_pre" if PRETRAIN else "_imnet")

models = []
for fold in range(NFOLDS):
    ck = f"./models/vgg16_{tag}_fold{fold}.pt"
    if not os.path.exists(ck):
        print(f"fold {fold}: missing {ck} — skipped"); continue
    m = Fusion(); m.load_state_dict(torch.load(ck, map_location="cpu"))
    models.append(m.to(DEV))
assert models, "no checkpoints found"

rows = []
for mode, ablate_mods in RUNS:
    name = mode if not ablate_mods else f"{mode}[{'+'.join(ablate_mods)}]"
    dl = DataLoader(TestSet(mode, ablate_mods), BS, shuffle=False, num_workers=4)
    p = np.mean([predict(m, dl) for m in models], axis=0)   # fold-averaged prob
    rows.append(report(labels, p, THR, name))
    np.save(f"{OUT}/probs_{tag}_{name.replace('|','_')}.npy", p)

with open(f"{OUT}/brain_removal_control_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n---- summary ----")
for r in rows:
    print(f"{r['run']:<26} AUC {r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}]  "
          f"bal-acc {r['BAL_ACC']:.3f}")
print(f"\nwrote {OUT}/brain_removal_control_{tag}.csv")


# ------------------------------------------------------- quick visual check
# Uncomment once to eyeball that the masks/fills are doing what you expect.
# import matplotlib.pyplot as plt
# j = 0
# fig, ax = plt.subplots(len(RUNS), len(STREAMS), figsize=(3*len(STREAMS), 3*len(RUNS)))
# for r, (mode, am) in enumerate(RUNS):
#     for c, (mod, plane) in enumerate(STREAMS):
#         x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
#         x = apply_ablation(x, mod, mode, am, stems[j], plane)
#         ax[r, c].imshow(x[1], cmap="gray"); ax[r, c].axis("off")
#         if r == 0: ax[r, c].set_title(f"{mod}/{plane}", fontsize=8)
#     ax[r, 0].set_ylabel(mode, fontsize=8)
# plt.tight_layout(); plt.savefig(f"{OUT}/ablation_examples.png", dpi=120)