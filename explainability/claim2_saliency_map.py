"""
Claim 2 — attribution lands on AD-relevant anatomy, measured in numbers.
MEMORY-EFFICIENT VERSION.

What changed vs the first version (all fixes are structural, results identical):

1. cam_maps builds an autograd graph for ONE encoder at a time. The other five
   streams contribute constants to the concatenated feature vector, so their
   pooled features are cached under no_grad. Gradients w.r.t. the target
   activation are unchanged.
2. Layers 0..TARGET_LAYER run under no_grad; the activation is detached and
   given requires_grad_, and only layers TARGET_LAYER+1.. carry a graph. The
   224/112/56 resolution blocks — the memory-heavy ones — are never taped.
3. Occlusion builds its variant batch in chunks instead of materialising all
   729 copies of a 3x224x224 tensor (~440 MB) at once.
4. Attribution maps are written to a memmap on disk rather than held in RAM.
5. The randomisation copy is kept on CPU and moved to GPU one stage at a time.

If it still OOMs, in order: lower BS to 1, lower OCC_CHUNK, set
STORE_METHODS to fewer methods, raise OCC_STRIDE to 16.
Also useful: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Requires build_roi_masks.py (masks + CN-mean images).
"""

import glob, os, csv, copy, gc, numpy as np, torch, torch.nn as nn, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

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
NFOLDS = 1
THR = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./attrib"; os.makedirs(OUT, exist_ok=True)
MEAN, STD = 0.449, 0.226
SIZE = 224

STREAMS = [(m, p) for m in MODS for p in PLANES]

# ----------------------------------------------------------------- memory cfg
BS         = 2       # subjects per batch. 1 is safest.
OCC_CHUNK  = 32      # occlusion variants forwarded at once
CLEAR_EVERY_BATCH = True

# ------------------------------------------------------------------ attrib cfg
# vgg16_bn.features: block3 -> idx 22 (56x56) | block4 -> idx 32 (28x28)
#                    block5 -> idx 42 (14x14)
TARGET_LAYER = 32
METHODS = ["layercam", "gradcam", "gradcampp"]

RUN_OCCLUSION = True
OCC_PATCH, OCC_STRIDE = 16, 8
OCC_MAX_SUBJECTS = None      # None = all; set e.g. 30 for a quick pass

RUN_RANDOMIZATION = True
RAND_METHOD   = "layercam"
RAND_SUBJECTS = 20

AD_ROI_SET   = ["hippocampus", "amygdala", "parahipp",
                "post_cing", "precuneus", "inf_parietal"]
CTRL_ROI_SET = ["precentral", "occipital_pole", "lingual"]
ALL_ROIS     = AD_ROI_SET + CTRL_ROI_SET

_CNMEAN, _MASK = {}, {}


def files_for(stem, mod, plane):
    c, d = CONFIG[mod][plane]
    return [f"{stem}_{mod}_{plane}_{c}_{k}.npy" for k in (-d, 0, d)]


def get_cnmean(mod, plane):
    key = (mod, plane)
    if key not in _CNMEAN:
        c, d = CONFIG[mod][plane]
        _CNMEAN[key] = np.stack([np.load(f"{MASK_DIR}/cnmean_{mod}_{plane}_{c}_{k}.npy")
                                 for k in (-d, 0, d)]).astype("float32")
    return _CNMEAN[key]


def get_roi(roi, mod, plane):
    key = (roi, mod, plane)
    if key not in _MASK:
        c, _ = CONFIG[mod][plane]
        _MASK[key] = np.load(f"{MASK_DIR}/roi_{roi}_{mod}_{plane}_{c}_0.npy") > 0.5
    return _MASK[key]


def normalize(x, mod):
    lo, hi = NORM[mod]
    x = np.clip(x, lo, hi); x = (x - lo) / (hi - lo)
    return (x - MEAN) / STD


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
ids = np.array([os.path.basename(s) for s in subjects])
N = len(stems)
print(f"{N} paired test subjects", np.bincount(labels), f"({dropped} dropped)")


class TestSet(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, j):
        xs = []
        for mod, plane in STREAMS:
            x = np.stack([np.load(f) for f in files_for(stems[j], mod, plane)])
            xs.append(torch.from_numpy(normalize(x, mod)).float())
        return xs, labels[j], j


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


tag = f"mm_fusion_p{P_DROP}" + ("_pre" if PRETRAIN else "_imnet")
ck = f"./models/vgg16_{tag}_fold0.pt"
assert os.path.exists(ck), f"missing {ck}"
MODEL = Fusion(); MODEL.load_state_dict(torch.load(ck, map_location="cpu"))
MODEL = MODEL.to(DEV).eval()
for p_ in MODEL.parameters():
    p_.requires_grad_(False)          # only activations need gradients


# ============================================================= A. attribution
@torch.no_grad()
def pooled_features(model, xs):
    """All six streams' pooled features, no graph."""
    return [model.pool(e(x)).flatten(1) for e, x in zip(model.enc, xs)]


def cam_maps(model, xs, method="layercam", layer=TARGET_LAYER):
    """(6, B, 224, 224) positive attribution. One encoder taped at a time."""
    base_f = pooled_features(model, xs)
    maps = []
    for si, (e, x) in enumerate(zip(model.enc, xs)):
        with torch.no_grad():                       # early blocks: no graph
            h = x
            for i in range(layer + 1):
                h = e[i](h)
        A = h.detach().requires_grad_(True)
        del h

        h2 = A                                      # graph starts here
        for i in range(layer + 1, len(e)):
            h2 = e[i](h2)
        f_i = model.pool(h2).flatten(1)
        cat = torch.cat([f_i if k == si else base_f[k]
                         for k in range(len(STREAMS))], 1)
        out = model.head(cat)
        score = (out[:, 1] - out[:, 0]).sum()
        G, = torch.autograd.grad(score, A)

        Ad = A.detach()
        if method == "layercam":
            M = F.relu((F.relu(G) * Ad).sum(1, keepdim=True))
        elif method == "gradcam":
            M = F.relu((G.mean((2, 3), keepdim=True) * Ad).sum(1, keepdim=True))
        elif method == "gradcampp":
            g2, g3 = G ** 2, G ** 3
            den = 2 * g2 + (Ad * g3).sum((2, 3), keepdim=True)
            alpha = g2 / torch.where(den != 0, den, torch.ones_like(den))
            w = (alpha * F.relu(G)).sum((2, 3), keepdim=True)
            M = F.relu((w * Ad).sum(1, keepdim=True))
        else:
            raise ValueError(method)

        M = F.interpolate(M, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        maps.append(M[:, 0].detach().cpu().numpy())
        del A, Ad, h2, f_i, cat, out, G, M
    return maps


@torch.no_grad()
def occlusion_maps(model, xs, patch=OCC_PATCH, stride=OCC_STRIDE):
    """AD-score drop when a patch is replaced by the CN-group mean.

    Only the occluded stream's encoder is recomputed; the other five streams'
    pooled features are cached and the head is linear, so this costs one
    encoder pass per position instead of six. Variants are built in chunks.
    """
    B = xs[0].shape[0]
    base_f = pooled_features(model, xs)
    o0 = model.head(torch.cat(base_f, 1))
    base_s = (o0[:, 1] - o0[:, 0])

    pos = [(y, x) for y in range(0, SIZE - patch + 1, stride)
                  for x in range(0, SIZE - patch + 1, stride)]
    maps = []
    for si, (mod, plane) in enumerate(STREAMS):
        cm = torch.from_numpy(normalize(get_cnmean(mod, plane), mod)).float().to(DEV)
        acc = np.zeros((B, SIZE, SIZE), "float32")
        cnt = np.zeros((SIZE, SIZE), "float32")
        for y, x in pos:
            cnt[y:y + patch, x:x + patch] += 1

        for b in range(B):
            others = [base_f[k][b:b + 1] for k in range(len(STREAMS))]
            for c0 in range(0, len(pos), OCC_CHUNK):
                chunk = pos[c0:c0 + OCC_CHUNK]
                v = xs[si][b:b + 1].repeat(len(chunk), 1, 1, 1).clone()
                for k, (y, x) in enumerate(chunk):
                    v[k, :, y:y + patch, x:x + patch] = cm[:, y:y + patch, x:x + patch]
                f_occ = model.pool(model.enc[si](v)).flatten(1)
                cat = torch.cat([f_occ if k == si else others[k].expand(len(chunk), -1)
                                 for k in range(len(STREAMS))], 1)
                o = model.head(cat)
                drop = (base_s[b] - (o[:, 1] - o[:, 0])).cpu().numpy()
                for k, (y, x) in enumerate(chunk):
                    acc[b, y:y + patch, x:x + patch] += drop[k]
                del v, f_occ, cat, o
        maps.append(acc / np.maximum(cnt, 1))
        del cm
    return maps


# ========================================================== B. ROI statistics
def brain_mask_of(stem, mod, plane):
    c, _ = CONFIG[mod][plane]
    x = np.load(f"{stem}_{mod}_{plane}_{c}_0.npy")
    return np.abs(x) > (0.0 if mod == "mri" else 0.05)


def roi_stats(attr, stem, mod, plane):
    bm = brain_mask_of(stem, mod, plane)
    a = np.clip(attr, 0, None) * bm
    tot, barea = a.sum(), bm.sum()
    out = {}
    if tot <= 0 or barea == 0:
        return out, None
    for roi in ALL_ROIS:
        rm = get_roi(roi, mod, plane) & bm
        area = rm.sum()
        out[roi] = ((a[rm].sum() / tot) / (area / barea)) if area > 0 else np.nan
    u = np.zeros_like(bm)
    for roi in AD_ROI_SET:
        u |= get_roi(roi, mod, plane)
    peak = np.unravel_index(np.argmax(a), a.shape)
    return out, bool(u[peak])


# ================================================================ run part A/B
ALL_METHODS = METHODS + (["occlusion"] if RUN_OCCLUSION else [])

# maps live on disk, not in RAM
store = {m: np.lib.format.open_memmap(f"{OUT}/maps_{m}.npy", mode="w+",
                                      dtype="float16",
                                      shape=(N, len(STREAMS), SIZE, SIZE))
         for m in ALL_METHODS}


dl = DataLoader(TestSet(), BS, shuffle=False, num_workers=2)
P_all = np.zeros(N, "float32")
occ_done = 0

print("\ncomputing attributions")
for xs, y, idx in dl:
    xs = [x.to(DEV) for x in xs]
    idx = idx.numpy()
    with torch.no_grad():
        P_all[idx] = torch.softmax(MODEL(xs), 1)[:, 1].cpu().numpy()

    for meth in METHODS:
        for si, m_ in enumerate(cam_maps(MODEL, xs, meth)):
            store[meth][idx, si] = m_.astype("float16")

    if RUN_OCCLUSION and (OCC_MAX_SUBJECTS is None or occ_done < OCC_MAX_SUBJECTS):
        for si, m_ in enumerate(occlusion_maps(MODEL, xs)):
            store["occlusion"][idx, si] = m_.astype("float16")
        occ_done += len(idx)

    del xs
    if CLEAR_EVERY_BATCH:
        gc.collect(); torch.cuda.empty_cache()
    print(f"  {idx[-1] + 1}/{N}", end="\r")

print("\n\nquantifying ROIs")
correct_ad = (labels == 1) & (P_all >= THR)
print(f"  correctly classified AD: {correct_ad.sum()} / {(labels == 1).sum()}")

rows = []
group = {m: np.zeros((len(STREAMS), SIZE, SIZE), "float64") for m in ALL_METHODS}
gn = {m: 0 for m in ALL_METHODS}

for meth in ALL_METHODS:
    for j in range(N):
        if meth == "occlusion" and OCC_MAX_SUBJECTS is not None and j >= occ_done:
            continue
        for si, (mod, plane) in enumerate(STREAMS):
            a = store[meth][j, si].astype("float32")
            st, hit = roi_stats(a, stems[j], mod, plane)
            if not st:
                continue
            rows.append(dict(method=meth, subject=ids[j], label=int(labels[j]),
                             p_ad=float(P_all[j]), mod=mod, plane=plane,
                             peak_in_ad_set=hit, **st))
        if correct_ad[j]:
            group[meth] += store[meth][j].astype("float64")
            gn[meth] += 1

for meth in ALL_METHODS:
    n = max(gn[meth], 1)
    for si, (mod, plane) in enumerate(STREAMS):
        c, _ = CONFIG[mod][plane]
        np.save(f"{OUT}/groupmap_{meth}_{mod}_{plane}_{c}.npy",
                (group[meth][si] / n).astype("float32"))
    print(f"  group map [{meth}] averaged over {n} correctly classified AD subjects")

with open(f"{OUT}/roi_attribution_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n============ enrichment (AD subjects, 1.0 = chance) ============")
for meth in ALL_METHODS:
    print(f"\n--- {meth} ---")
    print(f"{'stream':<16}" + "".join(f"{r[:11]:<13}" for r in ALL_ROIS))
    for mod, plane in STREAMS:
        sel = [r for r in rows if r["method"] == meth and r["mod"] == mod
               and r["plane"] == plane and r["label"] == 1]
        if not sel:
            continue
        line = f"{mod + '/' + plane:<16}"
        for roi in ALL_ROIS:
            v = np.nanmean([r[roi] for r in sel])
            line += ("     n/a     " if not np.isfinite(v) else f"{v:<13.2f}")
        print(line)

print("\n============ pointing game: peak inside AD ROI set ============")
print(f"{'method':<12}{'stream':<16}{'AD':>8}{'CN':>8}{'chance':>9}")
chance = {}
for mod, plane in STREAMS:
    bm = brain_mask_of(stems[0], mod, plane)
    u = np.zeros_like(bm)
    for roi in AD_ROI_SET:
        u |= get_roi(roi, mod, plane)
    chance[(mod, plane)] = (u & bm).sum() / max(bm.sum(), 1)

for meth in ALL_METHODS:
    for mod, plane in STREAMS:
        sa = [r["peak_in_ad_set"] for r in rows if r["method"] == meth
              and r["mod"] == mod and r["plane"] == plane and r["label"] == 1
              and r["peak_in_ad_set"] is not None]
        sc = [r["peak_in_ad_set"] for r in rows if r["method"] == meth
              and r["mod"] == mod and r["plane"] == plane and r["label"] == 0
              and r["peak_in_ad_set"] is not None]
        if not sa:
            continue
        print(f"{meth:<12}{mod + '/' + plane:<16}{np.mean(sa):>7.1%}"
              f"{np.mean(sc):>8.1%}{chance[(mod, plane)]:>9.1%}")
print("\nRead each rate against the chance column — that is the AD-set area "
      "fraction, not 0.")


# ================================================ D. cascading randomisation
def randomize_module(mod_):
    for m in mod_.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            m.running_mean.zero_(); m.running_var.fill_(1.0)


if RUN_RANDOMIZATION:
    print("\n============ cascading weight randomisation ============")
    sub = np.where(labels == 1)[0][:RAND_SUBJECTS]
    dlr = DataLoader(torch.utils.data.Subset(TestSet(), sub.tolist()),
                     BS, shuffle=False, num_workers=2)

    def maps_for(model):
        acc = [[] for _ in STREAMS]
        for xs, _, _ in dlr:
            xs = [x.to(DEV) for x in xs]
            for si, m_ in enumerate(cam_maps(model, xs, RAND_METHOD)):
                acc[si].append(m_.astype("float32"))
            del xs
            torch.cuda.empty_cache()
        return [np.concatenate(a) for a in acc]

    ref = maps_for(MODEL)

    rnd_cpu = copy.deepcopy(MODEL).cpu()          # keep the copy off the GPU
    for p_ in rnd_cpu.parameters():
        p_.requires_grad_(False)
    stages = [("head", None), ("block5", range(34, 44)), ("block4", range(24, 34)),
              ("block3", range(14, 24)), ("block2", range(7, 14)),
              ("block1", range(0, 7))]

    print(f"{'randomised up to':<20}{'mean |Spearman| vs original':>30}")
    curve = []
    for name, rng_ in stages:
        if name == "head":
            randomize_module(rnd_cpu.head)
        else:
            for e in rnd_cpu.enc:
                for i in rng_:
                    randomize_module(e[i])
        rnd = rnd_cpu.to(DEV).eval()
        cur = maps_for(rnd)
        rnd_cpu = rnd.cpu(); del rnd
        torch.cuda.empty_cache()

        rhos = []
        for si in range(len(STREAMS)):
            for b in range(cur[si].shape[0]):
                a, c_ = ref[si][b].ravel(), cur[si][b].ravel()
                if a.std() > 0 and c_.std() > 0:
                    rhos.append(abs(spearmanr(a, c_).statistic))
        r = float(np.mean(rhos)) if rhos else np.nan
        curve.append(dict(stage=name, mean_abs_spearman=r, n=len(rhos)))
        print(f"{name:<20}{r:>30.3f}")
        del cur

    with open(f"{OUT}/randomization_{RAND_METHOD}_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "mean_abs_spearman", "n"])
        w.writeheader(); w.writerows(curve)
    print("\nA valid method decays toward 0 as more layers are randomised. "
          "Staying high means the map is driven by image structure rather than "
          "by what the model learned.")

print(f"\nwrote {OUT}/roi_attribution_{tag}.csv, maps_*.npy and group maps to {OUT}/")
