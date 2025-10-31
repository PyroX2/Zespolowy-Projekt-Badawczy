#test.py

import os, re, csv, yaml, time, random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights

CFG = dict(
    DATA_YAML   = "/users/project1/pt01254/classifier_birads_full/data.yaml",
    MODELS_DIR  = "/users/project1/pt01254/classifier_birads_full/cls_universal_yaml",  #folder z modelami
    OUT_DIR     = "/users/project1/pt01254/classifier_birads_full/eval_universal",
    CACHE_ROOT  = "/users/project1/pt01254/classifier_birads_full/cache_eval_1080x1920",

    CLASS_GROUPS = [
        ["B0"], 
        ["B1","B2"], 
        ["B4","B5","B6"]
    ],  

    IMG_W=1080, IMG_H=1920,
    BATCH_SIZE=32, NUM_WORKERS=8,
    AMP_ENABLED=True,
    SEED=2137,

    VAL_SAMPLES_TOTAL=0,

    USE_CACHE=True,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CV2_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try: torch.set_float32_matmul_precision('high')
except Exception: pass
cv2.setNumThreads(0)
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])

OUT_DIR    = Path(CFG["OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path(CFG["CACHE_ROOT"]); CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def load_yaml_info(DATA_YAML: str | Path):
    DATA_YAML = Path(DATA_YAML)
    with open(DATA_YAML, "r") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", DATA_YAML.parent)).expanduser()
    train_dir = root / y.get("train", "train")
    val_dir   = root / y.get("val", "val")
    names     = list(y.get("names", []))
    return root, train_dir, val_dir, names

def list_images_by_class(root_dir: Path, names: list[str]) -> dict[str, list[str]]:
    out = {}
    for cname in names:
        cdir = root_dir / cname
        files = []
        if cdir.is_dir():
            files = [str(p) for p in cdir.rglob("*") if p.suffix.lower() in CV2_EXTS]
            files.sort()
        out[cname] = files
    return out

def make_group_map(class_groups: list[list[str]]):
    mp = {}
    for gi, group in enumerate(class_groups):
        for cname in group:
            mp[cname] = gi
    return mp

def read_image_rgb_any(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError("Nie można wczytać obrazu: " + str(path))
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def letterbox_rb(img: np.ndarray, dst_w: int, dst_h: int, pad_value=(0,0,0)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(dst_w / w, dst_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((dst_h, dst_w, 3), pad_value, dtype=img.dtype)
    canvas[:nh, :nw] = img
    return canvas

def cache_key(path: str) -> Path:
    stem = Path(path).stem
    return CACHE_ROOT / f"{stem}_{CFG['IMG_W']}x{CFG['IMG_H']}.npy"

class EvalDataset(Dataset):
    def __init__(self, items: list[tuple[str,int]], img_w: int, img_h: int, use_cache: bool):
        self.items = items
        self.W = img_w; self.H = img_h
        self.use_cache = use_cache

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        try:
            ck = cache_key(path)
            if self.use_cache and ck.exists():
                arr = np.load(str(ck), allow_pickle=False)
            else:
                img = read_image_rgb_any(path)
                arr = letterbox_rb(img, self.W, self.H, pad_value=(0,0,0))
                if self.use_cache:
                    try: np.save(str(ck), arr)
                    except Exception: pass
            arr = arr.astype(np.float32)/255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            img_t = torch.from_numpy(arr).permute(2,0,1).contiguous()
            return img_t, y, path
        except Exception:
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            arr = arr.astype(np.float32)/255.0
            arr = (arr - IMAGENET_MEAN)/IMAGENET_STD
            img_t = torch.from_numpy(arr).permute(2,0,1).contiguous()
            return img_t, y, path

def per_class_counts_from_preds(labels_np: np.ndarray, preds_np: np.ndarray, K: int):
    total = labels_np.shape[0]
    result = []
    for c in range(K):
        tp = int(((preds_np == c) & (labels_np == c)).sum())
        fp = int(((preds_np == c) & (labels_np != c)).sum())
        fn = int(((preds_np != c) & (labels_np == c)).sum())
        tn = int(total - tp - fp - fn)
        acc_c = (tp + tn) / max(1, total)
        prec  = tp / max(1, tp + fp)
        rec   = tp / max(1, tp + fn)
        f1    = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        sup   = int((labels_np == c).sum())
        result.append((tp, fp, fn, tn, acc_c, prec, rec, f1, sup))
    return result

def build_model_from_state_dict(state_dict, yaml_num_classes: int):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = model.fc.in_features

    head_type, out_features = None, None
    if "fc.weight" in state_dict and "fc.bias" in state_dict:
        head_type = "linear"
        out_features = state_dict["fc.weight"].shape[0]
    elif "fc.1.weight" in state_dict and "fc.1.bias" in state_dict:
        head_type = "sequential"
        out_features = state_dict["fc.1.weight"].shape[0]
    else:
        for k, v in state_dict.items():
            if k.startswith("fc") and k.endswith("weight") and getattr(v, "dim", lambda:0)() == 2:
                out_features = v.shape[0]; head_type = "unknown_fc"; break

    if out_features is None:
        out_features = yaml_num_classes
        head_type = "linear"

    if head_type == "sequential":
        model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, out_features))
    else:
        model.fc = nn.Linear(in_f, out_features)

    return model, out_features


def build_eval_items(val_by_class: dict[str, list[str]], class_groups: list[list[str]] | None,
                     names_yaml: list[str], total_limit: int | None, seed: int) -> tuple[list[tuple[str,int]], list[str]]:
    rng = random.Random(seed)

    if class_groups is None:
        class_names = [c for c in names_yaml if len(val_by_class.get(c, [])) > 0]
        items = []
        for ci, cname in enumerate(class_names):
            for p in val_by_class.get(cname, []):
                items.append((p, ci))
    else:
        
        cls2group = make_group_map(class_groups)
        class_names = [f"G{i}" for i in range(len(class_groups))]
        items = []
        for cname, paths in val_by_class.items():
            if cname not in cls2group:
                continue
            gi = cls2group[cname]
            for p in paths:
                items.append((p, gi))

    if total_limit and total_limit > 0 and total_limit < len(items):
        items = rng.sample(items, total_limit)

    return items, class_names

@torch.no_grad()
def evaluate_model_argmax(model: nn.Module, loader: DataLoader, amp: bool = True):
    model.eval()
    all_labels, all_preds = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        with autocast(enabled=(amp and device.type=="cuda")):
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    if not all_labels:
        return dict(accuracy=0.0, macro_precision=0.0, macro_recall=0.0, macro_f1=0.0, per_class=[])

    labels_np = np.concatenate(all_labels, 0)
    preds_np  = np.concatenate(all_preds,  0)
    K = int(np.max([labels_np.max(), preds_np.max()]) + 1) if labels_np.size else 0

    pcs = per_class_counts_from_preds(labels_np, preds_np, K)
    acc = float((labels_np == preds_np).mean()) if labels_np.size else 0.0
    precisions = [x[5] for x in pcs]; recalls = [x[6] for x in pcs]; f1s = [x[7] for x in pcs]
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec  = float(np.mean(recalls))    if recalls    else 0.0
    macro_f1   = float(np.mean(f1s))        if f1s        else 0.0
    return dict(accuracy=acc, macro_precision=macro_prec, macro_recall=macro_rec, macro_f1=macro_f1, per_class=pcs)


def run_eval(CFG: dict):
    root, TRAIN_DIR, VAL_DIR, YAML_NAMES = load_yaml_info(CFG["DATA_YAML"])
    print(f"[INFO] ROOT={root}")
    print(f"[INFO] VAL_DIR={VAL_DIR}")
    print(f"[INFO] YAML names={YAML_NAMES}")

    if CFG["CLASS_GROUPS"] is None:
        wanted = YAML_NAMES
    else:
        flat = []
        for g in CFG["CLASS_GROUPS"]:
            flat.extend(g)
        wanted = [c for c in YAML_NAMES if c in set(flat)]

    val_by_class = list_images_by_class(VAL_DIR, wanted)
    items, eval_class_names = build_eval_items(
        val_by_class,
        CFG["CLASS_GROUPS"],
        YAML_NAMES,
        CFG.get("VAL_SAMPLES_TOTAL") or 0,
        CFG["SEED"]
    )
    if not items:
        raise SystemExit("Brak obrazów do ewaluacji.")
    print(f"[INFO] Eval items: {len(items)} | classes used = {eval_class_names}")

    ds = EvalDataset(items, CFG["IMG_W"], CFG["IMG_H"], CFG["USE_CACHE"])
    dl = DataLoader(
        ds, batch_size=CFG["BATCH_SIZE"], shuffle=False,
        num_workers=CFG["NUM_WORKERS"], pin_memory=(device.type=="cuda"),
        prefetch_factor=2, persistent_workers=True, drop_last=False
    )

    MODELS_DIR = Path(CFG["MODELS_DIR"])
    ckpts = sorted([p for p in MODELS_DIR.glob("*.pt") if p.is_file()], key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit(f"Nie znaleziono modeli *.pt w {MODELS_DIR}")

    summary_csv  = OUT_DIR / "summary_per_model.csv"
    perclass_csv = OUT_DIR / "per_class_metrics.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","n_images","batch","accuracy","macro_precision","macro_recall","macro_f1","minutes"])
    with open(perclass_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","class","support","TP","FP","FN","TN","accuracy","precision","recall","f1"])

    for mp in ckpts:
        print(f"\n[MODEL] {mp.name}")
        ckpt = torch.load(str(mp), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            ckpt_names = ckpt.get("names", None)
            ckpt_groups = ckpt.get("class_groups", None)
        else:
            state_dict = ckpt
            ckpt_names = None
            ckpt_groups = None

        K_eval = len(eval_class_names)

        model, out_feats = build_model_from_state_dict(state_dict, yaml_num_classes=K_eval)
        model = model.to(device).to(memory_format=torch.channels_last)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state: missing={missing} unexpected={unexpected}")

        if out_feats != K_eval:
            print(f"[WARN] Głowica ckpt({out_feats}) != K_eval({K_eval}). Ewaluacja w przestrzeni {out_feats} klas.")

        t0 = time.time()
        metrics = evaluate_model_argmax(model, dl, amp=CFG["AMP_ENABLED"])
        mins = (time.time() - t0)/60.0

        acc, mpr, mrc, mf1 = metrics["accuracy"], metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"]
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([mp.name, len(ds), CFG["BATCH_SIZE"], f"{acc:.6f}", f"{mpr:.6f}", f"{mrc:.6f}", f"{mf1:.6f}", f"{mins:.2f}"])

        class_names_for_csv = eval_class_names if (out_feats == K_eval) else [f"class_{i}" for i in range(out_feats)]
        for ci, (tp, fp, fn, tn, acc_c, prec, rec, f1, sup) in enumerate(metrics["per_class"]):
            cname = class_names_for_csv[ci] if ci < len(class_names_for_csv) else f"class_{ci}"
            with open(perclass_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([mp.name, cname, sup, tp, fp, fn, tn,
                            f"{acc_c:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}"])

    print(f"\n[OK] Wyniki zapisane:\n - {summary_csv}\n - {perclass_csv}")


