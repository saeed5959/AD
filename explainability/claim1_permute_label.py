"""
Claim 1 — label-permutation null.

Shuffles the subject->label mapping in the TRAIN POOL only, reruns the entire
pipeline unchanged, and collects the resulting AUCs into a null distribution.

Two nulls are collected:
  val_auc   validation AUC under permuted labels. Should be ~0.5. Above chance
            means a subject appears in both train and val, or a preprocessing
            statistic was computed across the split -> memorisation pays off.
  test_auc  external-test AUC against TRUE test labels (optional, EVAL_TEST).
            Should be ~0.5. Above chance means something outside the training
            labels carries the answer (e.g. selection on test data).

The reference run (perm seed None = real labels) uses the SAME reduced budget,
so the comparison is like-for-like. Do not compare a 30-epoch null against your
60-epoch headline number.

IMPORTANT: PRETRAIN must be False. Pretrained trunks were fit on true labels and
would leak them into every permutation run.
"""

import glob, os, csv, copy, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT      = "../AD_train/preprocess"
TEST_ROOT = "./preprocess_test"
MODS      = ["mri", "pet"]
CLASSES   = ["CN", "AD"]
BS, LR, SEED = 16, 1e-5, 0
NFOLDS = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
NORM = {"mri": (-3.0, 3.0), "pet": (-3.0, 3.0)}
PRE_TAG = {"mri": "mri_fusion_p0.0_pre", "pet": "pet_fusion_p0.0_pre"}

P_DROP   = 0.0
PRETRAIN = False
FREEZE   = 34

# ------------------------------------------------------------ permutation cfg
N_PERM         = 1      # number of shuffled-label runs
PERM_EPOCHS    = 30      # reduced budget; applies to the reference run too
PERM_FOLDS_RUN = 1       # folds per run (1 is enough for a null)
EVAL_TEST      = False    # also score the external test set with TRUE labels
PERM_BASE_SEED = 1000    # permutation RNG offset, independent of SEED

RUNS = [None] + list(range(N_PERM))   # None = reference run with real labels

assert not PRETRAIN, "PRETRAIN must be False for the permutation null"

OUT = "./perm"; os.makedirs(OUT, exist_ok=True); os.makedirs("./models", exist_ok=True)
STREAMS = [(m, p) for m in MODS for p in PLANES]


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def complete(subj_dir):
    stem = f"{subj_dir}/{os.path.basename(subj_dir)}"
    return all(os.path.exists(f) for m, p in STREAMS for f in files_for(stem, m, p))


def load_cohort(root):
    subs, labs, dropped = [], [], 0
    for y, c in enumerate(CLASSES):
        for d in sorted(glob.glob(f"{root}/{c}/*/")):
            d = d.rstrip("/")
            if complete(d):
                subs.append(d); labs.append(y)
            else:
                dropped += 1
    subs, labs = np.array(subs), np.array(labs)
    stm = np.array([f"{d}/{os.path.basename(d)}" for d in subs])
    return subs, labs, stm, dropped


subjects, labels, stems, dropped = load_cohort(ROOT)
print(f"{len(subjects)} paired train-pool subjects", np.bincount(labels),
      f"({dropped} dropped)")

MEAN, STD = 0.449, 0.226
aug = torchvision.transforms.RandomAffine(degrees=5, translate=(0.03, 0.03))


class FusionSlices(Dataset):
    def __init__(self, idx, train, stems, labels):
        self.idx, self.train, self.stems, self.labels = idx, train, stems, labels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        xs = []
        for mod, plane in STREAMS:
            lo, hi = NORM[mod]
            x = np.stack([np.load(f) for f in files_for(self.stems[j], mod, plane)])
            x = np.clip(x, lo, hi)
            x = (x - lo) / (hi - lo)
            x = torch.from_numpy(x).float()
            if self.train:
                x = aug(x)
            xs.append((x - MEAN) / STD)
        return xs, self.labels[j]


def trunk(mod=None, plane=None, fold=None):
    t = torchvision.models.vgg16_bn(weights="DEFAULT").features
    for prm in t[:FREEZE].parameters():
        prm.requires_grad = False
    return t


class Fusion(nn.Module):
    def __init__(self, fold, p_drop=P_DROP):
        super().__init__()
        self.enc    = nn.ModuleList([trunk(m, p, fold) for m, p in STREAMS])
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.p_drop = p_drop
        self.head   = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512 * len(STREAMS), len(CLASSES)))

    def forward(self, xs):
        f = [self.pool(e(x)).flatten(1) for e, x in zip(self.enc, xs)]
        if self.training and self.p_drop > 0:
            n, B = len(f), f[0].shape[0]
            keep = (torch.rand(B, n, device=f[0].device) > self.p_drop).float()
            keep[keep.sum(1) == 0] = 1.0
            keep = keep * (n / keep.sum(1, keepdim=True))
            f = [fi * keep[:, i:i + 1] for i, fi in enumerate(f)]
        return self.head(torch.cat(f, 1))


@torch.no_grad()
def predict(m, dl):
    m.eval(); P, Y = [], []
    for xs, y in dl:
        xs = [x.to(DEV) for x in xs]
        P.append(torch.softmax(m(xs), 1)[:, 1].cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


# ----------------------------------------------------------------- the shuffle
def permuted_labels(y, perm_seed):
    """One label per subject, so permuting this array is subject-level by
    construction: all slices/streams of a subject keep the same shuffled label."""
    if perm_seed is None:
        return y.copy()
    rng = np.random.default_rng(PERM_BASE_SEED + perm_seed)
    return rng.permutation(y)


# ---------------------------------------------------------------- test cohort
if EVAL_TEST:
    _, te_labels, te_stems, te_drop = load_cohort(TEST_ROOT)
    print(f"{len(te_stems)} test subjects", np.bincount(te_labels), f"({te_drop} dropped)")
    dl_te = DataLoader(FusionSlices(np.arange(len(te_stems)), False, te_stems, te_labels),
                       BS, shuffle=False, num_workers=4)


# ----------------------------------------------------------------------- main
def run_one(perm_seed):
    y = permuted_labels(labels, perm_seed)
    name = "real" if perm_seed is None else f"perm{perm_seed:02d}"
    frac_moved = float((y != labels).mean()) if perm_seed is not None else 0.0

    skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(subjects, y)):
        if fold >= PERM_FOLDS_RUN:
            break

        mk = lambda idx, train: DataLoader(FusionSlices(idx, train, stems, y),
                                           BS, shuffle=train, num_workers=4)
        dl_tr, dl_va = mk(tr, True), mk(va, False)

        torch.manual_seed(SEED)              # identical init across runs; only labels differ
        m = Fusion(fold).to(DEV)
        w = torch.tensor(len(tr) / (len(CLASSES) * np.bincount(y[tr])), dtype=torch.float32)
        crit  = nn.CrossEntropyLoss(weight=w.to(DEV))
        opt   = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PERM_EPOCHS)

        best_va, best_ep, best_sd = 0.0, -1, None
        for ep in tqdm(range(PERM_EPOCHS), desc=f"{name} f{fold}", leave=False):
            m.train(); tl = 0
            for xs, yb in dl_tr:
                xs, yb = [x.to(DEV) for x in xs], yb.to(DEV)
                opt.zero_grad()
                loss = crit(m(xs), yb)
                loss.backward(); opt.step()
                tl += loss.item()
            sched.step()

            P, Y = predict(m, dl_va)
            va_auc = roc_auc_score(Y, P)
            if va_auc > best_va:                       # same selection rule as the real run
                best_va, best_ep = va_auc, ep
                best_sd = copy.deepcopy(m.state_dict())

        # ---- evaluate the selected checkpoint (OUTSIDE the epoch loop)
        m.load_state_dict(best_sd)
        Pv, Yv = predict(m, dl_va)
        va_auc = roc_auc_score(Yv, Pv)

        te_auc = np.nan
        if EVAL_TEST:
            Pt, _ = predict(m, dl_te)
            te_auc = roc_auc_score(te_labels, Pt)      # TRUE test labels, never permuted
            np.save(f"{OUT}/probs_test_{name}_fold{fold}.npy", Pt)

        np.savez(f"{OUT}/{name}_fold{fold}.npz", p=Pv, y=Yv, idx=va)
        rows.append(dict(run=name, perm_seed=-1 if perm_seed is None else perm_seed,
                         fold=fold, frac_labels_moved=round(frac_moved, 3),
                         best_epoch=best_ep, val_auc=va_auc, test_auc=te_auc))
        print(f"  {name} fold {fold}  val_auc {va_auc:.3f}"
              + (f"  test_auc {te_auc:.3f}" if EVAL_TEST else "")
              + f"  (best ep {best_ep})")
        del m, best_sd; torch.cuda.empty_cache()
    return rows


all_rows = []
for perm_seed in RUNS:
    all_rows += run_one(perm_seed)
    with open(f"{OUT}/perm_null.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader(); wtr.writerows(all_rows)

# --------------------------------------------------------------- null summary
real = [r for r in all_rows if r["perm_seed"] == -1]
null = [r for r in all_rows if r["perm_seed"] >= 0]

print("\n================ label-permutation null ================")
for key in (["val_auc"] + (["test_auc"] if EVAL_TEST else [])):
    nv = np.array([r[key] for r in null], dtype=float)
    rv = float(np.mean([r[key] for r in real]))
    if len(nv) == 0:
        continue
    # one-sided permutation p-value, +1 correction
    p = (np.sum(nv >= rv) + 1) / (len(nv) + 1)
    print(f"\n{key}")
    print(f"  real          {rv:.3f}")
    print(f"  null mean     {nv.mean():.3f} ± {nv.std():.3f}")
    print(f"  null range    [{nv.min():.3f}, {nv.max():.3f}]")
    print(f"  null 95th pct {np.percentile(nv, 95):.3f}")
    print(f"  p             {p:.4f}   (n_perm={len(nv)}, floor {1/(len(nv)+1):.4f})")

print(f"\nwrote {OUT}/perm_null.csv")