import glob, os, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
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
P_DROP, PRETRAIN = 0.0, False          # must match the trained model's tag
NFOLDS, BS = 1, 16
N_BOOT = 2000
THR = 0.5                             # set from val (Youden) if you have one
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./test"; os.makedirs(OUT, exist_ok=True)
MEAN, STD = 0.449, 0.226

STREAMS = [(m, p) for m in MODS for p in PLANES]            # 6 streams, same order as training


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def complete(subj_dir):
    stem = f"{subj_dir}/{os.path.basename(subj_dir)}"
    return all(os.path.exists(f)
               for m, p in STREAMS for f in files_for(stem, m, p))


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
    def __len__(self):
        return len(stems)

    def __getitem__(self, j):
        xs = []
        for mod, plane in STREAMS:
            lo, hi = NORM[mod]
            x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
            x = np.clip(x, lo, hi); x = (x - lo) / (hi - lo)
            xs.append((torch.from_numpy(x).float() - MEAN) / STD)
        return xs, labels[j]


class Fusion(nn.Module):
    """eval-only: no stream dropout, matches the trained architecture."""
    def __init__(self):
        super().__init__()
        self.enc  = nn.ModuleList([torchvision.models.vgg16_bn(weights=None).features
                                   for _ in STREAMS])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * len(STREAMS), len(CLASSES)))

    def forward(self, xs, keep=None):
        f = [self.pool(e(x)).flatten(1) for e, x in zip(self.enc, xs)]
        if keep is not None:                       # ablation: zero out streams
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
        n=len(y), n_CN=int((1 - y).sum()), n_AD=int(y.sum()),
        AUC=roc_auc_score(y, p), AUC_lo=lo, AUC_hi=hi,
        AUPRC=average_precision_score(y, p),
        ACC=(pred == y).mean(),
        BAL_ACC=balanced_accuracy_score(y, pred),
        SENS=tp / (tp + fn) if tp + fn else np.nan,
        SPEC=tn / (tn + fp) if tn + fp else np.nan,
        PPV=tp / (tp + fp) if tp + fp else np.nan,
        NPV=tn / (tn + fn) if tn + fn else np.nan,
        F1=f1_score(y, pred, zero_division=0),
        MCC=matthews_corrcoef(y, pred),
        BRIER=brier_score_loss(y, p),
        TN=tn, FP=fp, FN=fn, TP=tp,
    )
    print(f"\n===== {name} =====")
    print(f"n={m['n']}  (CN {m['n_CN']} / AD {m['n_AD']})   threshold {thr:.2f}")
    print(f"AUC       {m['AUC']:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"AUPRC     {m['AUPRC']:.3f}")
    print(f"Accuracy  {m['ACC']:.3f}    Balanced {m['BAL_ACC']:.3f}")
    print(f"Sens/Rec  {m['SENS']:.3f}    Spec     {m['SPEC']:.3f}")
    print(f"PPV       {m['PPV']:.3f}    NPV      {m['NPV']:.3f}")
    print(f"F1        {m['F1']:.3f}    MCC      {m['MCC']:.3f}")
    print(f"Brier     {m['BRIER']:.3f}")
    print(f"confusion TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return m


dl  = DataLoader(TestSet(), BS, shuffle=False, num_workers=4)
tag = f"mm_fusion_p{P_DROP}" + ("_pre" if PRETRAIN else "_imnet")

Ps, models = [], []
for fold in range(NFOLDS):
    ck = f"./models/vgg16_{tag}_fold{fold}.pt"
    if not os.path.exists(ck):
        print(f"fold {fold}: missing {ck} — skipped"); continue
    m = Fusion(); m.load_state_dict(torch.load(ck, map_location="cpu")); m = m.to(DEV)
    p = predict(m, dl); Ps.append(p); models.append(m)
    print(f"fold {fold}  AUC {roc_auc_score(labels, p):.3f}  "
          f"acc {((p >= THR) == labels).mean():.3f}")

assert Ps, "no checkpoints found"
P = np.mean(Ps, 0)
np.savez(f"{OUT}/{tag}_test.npz", p=P, y=labels, per_fold=np.array(Ps),
         ids=np.array([os.path.basename(s) for s in subjects]))

res = report(labels, P, name=f"{tag}  (ensemble of {len(Ps)} folds)")
fa = np.array([roc_auc_score(labels, p) for p in Ps])
print(f"\nper-fold AUC spread: {fa.mean():.3f} ± {fa.std():.3f}  {np.round(fa, 3)}")

# ---- stream ablation: zero one (modality, plane) branch at a time ----
print("\n--- stream ablation (feature zeroed at inference) ---")
for i, (mod, plane) in enumerate(STREAMS):
    keep = [0.0 if k == i else 1.0 for k in range(len(STREAMS))]
    Pa = np.mean([predict(m, dl, keep) for m in models], 0)
    print(f"  without {mod}/{plane:9s}  AUC {roc_auc_score(labels, Pa):.3f}  "
          f"(full {res['AUC']:.3f})")

# ---- modality ablation: zero all three planes of one modality ----
print("\n--- modality ablation ---")
for mod in MODS:
    keep = [0.0 if m == mod else 1.0 for m, _ in STREAMS]
    Pa = np.mean([predict(m, dl, keep) for m in models], 0)
    print(f"  without {mod:9s} (all planes)  AUC {roc_auc_score(labels, Pa):.3f}  "
          f"(full {res['AUC']:.3f})")

# ---- CSV row for the paper table ----
import csv
with open(f"{OUT}/results.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model"] + list(res))
    if f.tell() == 0:
        w.writeheader()
    w.writerow({"model": tag, **{k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in res.items()}})
print(f"\nappended to {OUT}/results.csv")
