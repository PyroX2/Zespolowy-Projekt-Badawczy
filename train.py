#train.py

import os, time, random, yaml
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.models import resnet50, ResNet50_Weights

#config

cfg = dict(
    DATA_YAML   = "/users/project1/pt01254/classifier_birads_full/data.yaml",   
    OUT_DIR     = "/users/project1/pt01254/classifier_birads_full/cls_universal_yaml",  
    CACHE_ROOT  = "/users/project1/pt01254/classifier_birads_full/cache_universal_1080x1920", #cache optymalizuje kod

    CLASS_GROUPS = [
        ["B0"],               
        ["B1", "B2"],          
        ["B4", "B5", "B6"],    
        #B3 skip
    ], 

    IMG_W=1080, IMG_H=1920,   
    BATCH_SIZE=32,
    EPOCHS=50,
    IMAGES_PER_EPOCH=2240,    #0 to all
    BALANCE_RATIO=0.5,        #0.0 = naturalny wg czestosci klas w bazie, 1.0 = rowno klasy
    LR=3e-4, WEIGHT_DECAY=1e-4,
    ACCUM_STEPS=1,            
    SAVE_EVERY=5,             
    EARLY_STOP=10,            
    NUM_WORKERS=8,            
    AMP_ENABLED=True,         
    SEED=2137,                
    USE_CACHE=True,           
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CV2_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
 
    cv2.setNumThreads(0)

def load_yaml_info(DATA_YAML: str | Path):

    DATA_YAML = Path(DATA_YAML)
    with open(DATA_YAML, "r") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", DATA_YAML.parent)).expanduser()
    train_dir = root / y.get("train", "train")
    val_dir   = root / y.get("val", "val")
    names     = list(y.get("names", []))
    return root, train_dir, val_dir, names

def list_images_by_class(root_dir: str | Path):
    #slownik co do klas i klas z klas
    root = Path(root_dir)
    by_class = {}
    if not root.exists():
        return by_class
    for cdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cname = cdir.name
        files = [str(p) for p in cdir.rglob("*") if p.suffix.lower() in CV2_EXTS]
        files.sort()
        if files:
            by_class[cname] = files
    return by_class

def make_group_map(class_groups):
    mp = {}
    for gi, group in enumerate(class_groups):
        for cname in group:
            mp[cname] = gi
    return mp

def read_image_rgb_any(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Nie można wczytać: {path}")
    if im.ndim == 2:  #grayscale → RGB
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:  #BGRA → BGR → RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:               #BGR → RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def letterbox_rb(img: np.ndarray, dst_w: int, dst_h: int, pad_value=(0,0,0)) -> np.ndarray:

    h, w = img.shape[:2]
    scale = min(dst_w / w, dst_h / h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((dst_h, dst_w, 3), pad_value, dtype=img.dtype)
    canvas[:nh, :nw] = img  
    return canvas

class GroupedImageDataset(Dataset):
    def __init__(self, files_by_class: dict, class_groups: list[list[str]],
                 img_wh=(1080,1920), use_cache=True, cache_root: str | Path | None=None, augment=False):
        self.W, self.H = img_wh
        self.use_cache = use_cache
        self.cache_root = Path(cache_root) if cache_root else None
        if self.use_cache and self.cache_root:
            self.cache_root.mkdir(parents=True, exist_ok=True)

        #mapujemy klasy i liczymy
        self.cls2group = make_group_map(class_groups)  
        self.K = len(class_groups)                     

        
        self.items = []
        for cname, files in files_by_class.items():
            if cname not in self.cls2group:
                continue  #pomijamy te nie wymieniane
            gi = self.cls2group[cname]
            for p in files:
                self.items.append((p, gi))

        self.augment = augment

    def __len__(self): return len(self.items)

    def _ckey(self, path: str) -> Path:
        stem = Path(path).stem
        return self.cache_root / f"{stem}_{self.W}x{self.H}.npy"

    def _read_and_preprocess(self, path: str) -> np.ndarray:

        if self.use_cache and self.cache_root is not None:
            ck = self._ckey(path)
            if ck.exists():
                return np.load(str(ck), allow_pickle=False)
        arr = read_image_rgb_any(path)
        arr = letterbox_rb(arr, self.W, self.H, pad_value=(0,0,0))
        if self.use_cache and self.cache_root is not None:
            try: np.save(str(ck), arr)
            except Exception: pass 
        return arr

    def __getitem__(self, idx):

        tries = 4
        last_y = 0
        src_idx = idx
        for _ in range(tries):
            path, y = self.items[idx]; last_y = y
            try:
                arr = self._read_and_preprocess(path)

                if self.augment:
                    if random.random() < 0.5:
                        arr = cv2.flip(arr, 1) #flip
                    if random.random() < 0.35:
                        a = 0.9 + 0.2*random.random() #lekka zmiana kontrastu
                        b = np.random.uniform(-8, 8) #lekka zmiana jasnosci
                        arr = cv2.convertScaleAbs(arr, alpha=a, beta=b)

                arr = arr.astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

                img_t = torch.from_numpy(arr).permute(2,0,1).contiguous()
                return img_t, torch.tensor(y, dtype=torch.long), path
            except Exception:
                #probujemy 4 razy
                idx = (idx + 1) % len(self.items)

        
        arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(arr).permute(2,0,1).contiguous()
        bad = self.items[src_idx][0]
        return img_t, torch.tensor(last_y, dtype=torch.long), bad

class HybridPerEpochSampler(Sampler):

    def __init__(self, dataset: GroupedImageDataset, total_per_epoch: int,
                 balance_ratio: float = 0.5, seed: int = 42):
        self.ds   = dataset
        self.total= int(total_per_epoch)
        self.r    = float(max(0.0, min(1.0, balance_ratio)))
        self.rng  = random.Random(seed)

        self.K = dataset.K 

        self.idx_by_c = defaultdict(list)
        for i, (_, y) in enumerate(dataset.items):
            self.idx_by_c[int(y)].append(i)

        self.classes = sorted(self.idx_by_c.keys())
        sizes = np.array([len(self.idx_by_c[c]) for c in self.classes], dtype=np.float64) 

        self.probs = sizes / max(1.0, sizes.sum())

    def __len__(self): return self.total

    def __iter__(self):
        #kwestia balansu (tak by balans natural oddawal liczbe)
        if self.total <= 0 or self.K == 0:
            return iter([])

        n_bal = int(round(self.r * self.total))  
        n_nat = self.total - n_bal               
        chosen = []
        
        if n_bal > 0:
            base = n_bal // self.K
            rem  = n_bal %  self.K
            need = {c: base for c in self.classes}
            for c in self.classes[:rem]:
                need[c] += 1
            for c in self.classes:
                pool = self.idx_by_c[c]
                if not pool: continue
                picks = [pool[self.rng.randrange(len(pool))] for _ in range(need[c])]
                chosen.extend(picks)

        if n_nat > 0:
            nat = np.random.choice(self.classes, size=n_nat, replace=True, p=self.probs)
            for c in nat:
                pool = self.idx_by_c[c]
                if pool:
                    chosen.append(pool[self.rng.randrange(len(pool))])

        self.rng.shuffle(chosen)
        return iter(chosen)

@torch.no_grad()
def evaluate(model, loader, K: int, amp_enabled=True, device=torch.device("cpu")):
    #ewaluacja
    model.eval()
    total_loss, n_batches = 0.0, 0
    total, correct = 0, 0
    tp = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)

    ce = nn.CrossEntropyLoss(reduction="mean")

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=(amp_enabled and device.type=="cuda")):
            logits = model(imgs)           
            loss   = ce(logits, labels)    
        total_loss += float(loss.item()); n_batches += 1

        preds = logits.argmax(dim=1)
        total += int(labels.size(0))
        correct += int((preds == labels).sum().item())

        for c in range(K):
            tp[c] += int(((preds==c)&(labels==c)).sum().item())
            fp[c] += int(((preds==c)&(labels!=c)).sum().item())
            fn[c] += int(((preds!=c)&(labels==c)).sum().item())

    acc = correct / max(1, total)

    per_class, f1s = [], []
    for c in range(K):
        prec = tp[c] / max(1, tp[c] + fp[c])
        rec  = tp[c] / max(1, tp[c] + fn[c])
        f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        per_class.append((c, prec, rec, f1, int(tp[c]+fn[c])))  
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    avg_loss = total_loss / max(1, n_batches)
    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "per_class": per_class}


def train(CFG: dict):
    #trening
    set_seed(int(CFG["SEED"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR = Path(CFG["OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT = Path(CFG["CACHE_ROOT"]) if CFG.get("CACHE_ROOT") else None
    if CFG.get("USE_CACHE", True) and CACHE_ROOT:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    root, TRAIN_DIR, VAL_DIR, names = load_yaml_info(CFG["DATA_YAML"])
    print(f"[INFO] ROOT={root}")
    print(f"[INFO] TRAIN_DIR={TRAIN_DIR} | VAL_DIR={VAL_DIR}")
    print(f"[INFO] names (yaml)={names}")

    #Zbieramy zciezki
    train_by_class = list_images_by_class(TRAIN_DIR)
    val_by_class   = list_images_by_class(VAL_DIR)
    print(f"[INFO] train classes found: {sorted(train_by_class.keys())}")
    print(f"[INFO] val   classes found: {sorted(val_by_class.keys())}")

    K = len(CFG["CLASS_GROUPS"])
    print(f"[INFO] K={K} groups: {CFG['CLASS_GROUPS']} (pomijamy klasy spoza grup)")

    train_ds = GroupedImageDataset(
        train_by_class, CFG["CLASS_GROUPS"],
        img_wh=(CFG["IMG_W"], CFG["IMG_H"]),
        use_cache=CFG.get("USE_CACHE", True), cache_root=CACHE_ROOT,
        augment=True
    )
    val_ds = GroupedImageDataset(
        val_by_class, CFG["CLASS_GROUPS"],
        img_wh=(CFG["IMG_W"], CFG["IMG_H"]),
        use_cache=CFG.get("USE_CACHE", True), cache_root=CACHE_ROOT,
        augment=False
    )
    print(f"[INFO] train images: {len(train_ds)} | val images: {len(val_ds)}")

    if int(CFG["IMAGES_PER_EPOCH"]) > 0:
        sampler = HybridPerEpochSampler(
            train_ds,
            total_per_epoch=int(CFG["IMAGES_PER_EPOCH"]),
            balance_ratio=float(CFG["BALANCE_RATIO"]),
            seed=int(CFG["SEED"])
        )
        train_dl = DataLoader(
            train_ds, batch_size=int(CFG["BATCH_SIZE"]),
            sampler=sampler,
            num_workers=int(CFG["NUM_WORKERS"]),
            pin_memory=(device.type=="cuda"),
            prefetch_factor=2, persistent_workers=True, drop_last=False
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=int(CFG["BATCH_SIZE"]),
            shuffle=True,
            num_workers=int(CFG["NUM_WORKERS"]),
            pin_memory=(device.type=="cuda"),
            prefetch_factor=2, persistent_workers=True, drop_last=False
        )

    val_dl = DataLoader(
        val_ds, batch_size=int(CFG["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(CFG["NUM_WORKERS"]),
        pin_memory=(device.type=="cuda"),
        prefetch_factor=2, persistent_workers=True, drop_last=False
    )

    #wagi resnetu50
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, K))
    model = model.to(device).to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(CFG["LR"]), weight_decay=float(CFG["WEIGHT_DECAY"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(CFG["EPOCHS"]), 1), eta_min=1e-6
    )
    #krosujaca entropia
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    scaler = GradScaler(enabled=(CFG["AMP_ENABLED"] and device.type=="cuda"))

    best_val = float("inf")
    best_path = OUT_DIR / "resnet50_universal_best.pt"
    patience = 0

    print(f"[INFO] start TRAIN on {device} | imgs/epoch={CFG['IMAGES_PER_EPOCH']} | batch={CFG['BATCH_SIZE']} | balance_ratio={CFG['BALANCE_RATIO']}")

    for epoch in range(1, int(CFG["EPOCHS"]) + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for it, (imgs, labels, _) in enumerate(train_dl, 1):
            imgs   = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(CFG["AMP_ENABLED"] and device.type=="cuda")):
                logits = model(imgs)                         
                loss   = criterion(logits, labels) / int(CFG["ACCUM_STEPS"])  

            scaler.scale(loss).backward()
            if it % int(CFG["ACCUM_STEPS"]) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * int(CFG["ACCUM_STEPS"])
            n_seen  += 1

        val_metrics = evaluate(model, val_dl, K=K, amp_enabled=CFG["AMP_ENABLED"], device=device)
        scheduler.step()

        tr_loss = running / max(1, n_seen)
        val_loss= val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_f1  = val_metrics["macro_f1"]
        elapsed = (time.time()-t0)/60

        print(f"Ep {epoch:03d}/{CFG['EPOCHS']} | train {tr_loss:.4f} | val {val_loss:.4f} | "
              f"acc {val_acc:.3f} | F1 {val_f1:.3f} | lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f} min")

        #Zapisywanie najlepszego checkpointu wg val_loss ogolnie mozna zmienic jak ktos chce
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "class_groups": CFG["CLASS_GROUPS"],
                "img_size": (CFG["IMG_W"], CFG["IMG_H"]),
                "yaml": {
                    "data_yaml": CFG["DATA_YAML"],
                    "train": str(TRAIN_DIR),
                    "val": str(VAL_DIR),
                }
            }, best_path)
            patience = 0
            print(f"[CKPT] new best -> {best_path} (val_loss={best_val:.4f})")
        else:
            patience += 1

        if epoch % int(CFG["SAVE_EVERY"]) == 0:
            snap = OUT_DIR / f"resnet50_universal_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "class_groups": CFG["CLASS_GROUPS"],
                "img_size": (CFG["IMG_W"], CFG["IMG_H"]),
                "yaml": {
                    "data_yaml": CFG["DATA_YAML"],
                    "train": str(TRAIN_DIR),
                    "val": str(VAL_DIR),
                }
            }, snap)
            print(f"[CKPT] saved -> {snap}")

        #Early stopping 
        if patience >= int(CFG["EARLY_STOP"]):
            print(f"[STOP] early stopping (no val improvement for {CFG['EARLY_STOP']} epochs)")
            break

    #Ostatni checkpoint
    last_path = OUT_DIR / "resnet50_universal_last.pt"
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "class_groups": CFG["CLASS_GROUPS"],
        "img_size": (CFG["IMG_W"], CFG["IMG_H"]),
        "yaml": {
            "data_yaml": CFG["DATA_YAML"],
            "train": str(TRAIN_DIR),
            "val": str(VAL_DIR),
        }
    }, last_path)
    print(f"[DONE] saved last -> {last_path}")

    return {
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_val_loss": float(best_val),
    }
