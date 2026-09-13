import glob, os, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT    = "./preprocess"      # train pool ONLY — test lives in a separate folder
MODS    = ["mri", "pet"]
CLASSES = ["CN", "AD"]
BS, LR, EPOCHS, SEED = 16, 1e-5, 60, 0     # BS halved: 6 VGG trunks per sample
NFOLDS  = 5
FOLDS_RUN = 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PLANES = ["axial", "coronal", "sagittal"]
CONFIG = {                                                  # frozen from sweeps
    "mri": {"axial": (-20, 4),  "coronal": (-10, 4),  "sagittal": (-25, 4)},
    "pet": {"axial": (42, 16),  "coronal": (-40, 16), "sagittal": (-25, 16)},
}
# fixed bounds per modality — MUST match what the single-modality scripts used,
# otherwise the pretrained trunks see a different input distribution than they were fit on
NORM = {"mri": (-3.0, 3.0), "pet": (-3.0, 3.0)}

# per-modality fusion checkpoints used to init the trunks (tag from the single-modality runs)
PRE_TAG = {"mri": "mri_fusion_p0.0_pre", "pet": "pet_fusion_p0.0_pre"}

P_DROP   = 0.0       # stream dropout over the 6 (mod, plane) streams
PRETRAIN = False      # init trunks from per-modality fusion checkpoints
FREEZE   = 34
OUT = "./sweep"; os.makedirs(OUT, exist_ok=True); os.makedirs("./models", exist_ok=True)

STREAMS = [(m, p) for m in MODS for p in PLANES]            # 6 streams, fixed order


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def complete(subj_dir):
    stem = f"{subj_dir}/{os.path.basename(subj_dir)}"
    return all(os.path.exists(f)
               for m, p in STREAMS for f in files_for(stem, m, p))


subjects, labels, dropped = [], [], 0
for y, c in enumerate(CLASSES):
    for d in sorted(glob.glob(f"{ROOT}/{c}/*/")):
        d = d.rstrip("/")
        if complete(d):
            subjects.append(d); labels.append(y)
        else:
            dropped += 1
subjects, labels = np.array(subjects), np.array(labels)
stems = np.array([f"{d}/{os.path.basename(d)}" for d in subjects])
print(f"{len(subjects)} paired train-pool subjects", np.bincount(labels),
      f"({dropped} dropped: missing a modality/plane)")

MEAN, STD = 0.449, 0.226
aug = torchvision.transforms.RandomAffine(degrees=5, translate=(0.03, 0.03))


class FusionSlices(Dataset):
    """returns (list_of_6_tensors, label) — one tensor per (modality, plane) stream."""
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
                x = aug(x)                       # independent per stream
            xs.append((x - MEAN) / STD)
        return xs, self.labels[j]


def ckpt_path(mod, fold):
    return f"./models/vgg16_{PRE_TAG[mod]}_fold{fold}.pt"


_SD_CACHE = {}


def trunk(mod=None, plane=None, fold=None):
    t = torchvision.models.vgg16_bn(weights="DEFAULT").features
    if PRETRAIN and mod is not None:
        p = ckpt_path(mod, fold)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p} — run the {mod} fusion script first")
        if p not in _SD_CACHE:
            _SD_CACHE[p] = torch.load(p, map_location="cpu")
        sd = _SD_CACHE[p]
        pref = f"enc.{PLANES.index(plane)}."          # per-plane trunk inside the fusion ckpt
        fsd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if not fsd:
            raise KeyError(f"no keys with prefix '{pref}' in {os.path.basename(p)}")
        t.load_state_dict(fsd, strict=True)
        print(f"    {mod}/{plane}: loaded trunk from {os.path.basename(p)} [{pref}]")
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


tag = f"mm_fusion_p{P_DROP}" + ("_pre" if PRETRAIN else "_imnet")
skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
fold_va = []

for fold, (tr, va) in enumerate(skf.split(subjects, labels)):
    if fold >= FOLDS_RUN:
        break
    print(f"\n=== fold {fold}  train {len(tr)}  val {len(va)} ===")

    mk = lambda idx, train: DataLoader(FusionSlices(idx, train, stems, labels),
                                       BS, shuffle=train, num_workers=4)
    dl_tr, dl_va = mk(tr, True), mk(va, False)

    torch.manual_seed(SEED)
    m = Fusion(fold).to(DEV)
    w = torch.tensor(len(tr) / (len(CLASSES) * np.bincount(labels[tr])),
                     dtype=torch.float32)
    crit  = nn.CrossEntropyLoss(weight=w.to(DEV))
    opt   = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                              lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    ck, best_va = f"./models/vgg16_{tag}_fold{fold}.pt", 0
    for ep in tqdm(range(EPOCHS), desc=f"{tag} f{fold}"):
        m.train(); tl = correct = n = 0
        for xs, y in dl_tr:
            xs, y = [x.to(DEV) for x in xs], y.to(DEV)
            opt.zero_grad()
            out  = m(xs)
            loss = crit(out, y)
            loss.backward(); opt.step()
            tl += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
        sched.step()
        P, Y = predict(m, dl_va)
        va_auc = roc_auc_score(Y, P)
        if va_auc > best_va:
            best_va = va_auc; torch.save(m.state_dict(), ck)

        print(f"  loss {tl/len(dl_tr):.3f}  tr_acc {correct/n:.3f}")
        m.load_state_dict(torch.load(ck))
        Pv, Yv = predict(m, dl_va)
        np.savez(f"{OUT}/{tag}_fold{fold}.npz", p=Pv, y=Yv, idx=va)
        fold_va.append(roc_auc_score(Yv, Pv))
        print(f"  fold {fold}  n_val {len(va)}  val_auc {fold_va[-1]:.3f}")

fold_va = np.array(fold_va)
print(f"\n{tag}   val {fold_va.mean():.3f} ± {fold_va.std():.3f}",
      np.round(fold_va, 3))
