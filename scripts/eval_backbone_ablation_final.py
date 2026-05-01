"""
Backbone Ablation Study – QMUL-SurvFace (official evaluation protocol)
=======================================================================
Exactly mirrors test_face_verification.m:
  - Loads positive_pairs_names.mat  (5320 genuine pairs)
  - Loads negative_pairs_names.mat  (5320 impostor pairs)
  - Images live in Face_Verification_Test_Set/verification_images/
  - Score = L2 DISTANCE  (smaller = more similar, matching MATLAB "<" logic)
  - Metrics: TAR@FAR={0.3,0.1,0.01,0.001}, AUC, 10-fold mean accuracy

Backbones tested:
  1. AdaFace IR-101
  2. ArcFace iResNet-50
  3. buffalo_l (InsightFace ONNX)

Usage:
  python scripts/eval_backbone_ablation_final.py \
    [--adaface_sr_ckpt experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth]
"""

import os, sys, json, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import scipy.io
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.adaface_sr import AdaFaceSR
from src.models.iresnet import iresnet50

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIRMED PATHS  (from ls output)
# data/QMUL-SurvFace/
#   Face_Verification_Test_Set/
#     positive_pairs_names.mat
#     negative_pairs_names.mat
#     verification_images/        ← all images flat here
#   Face_Verification_Evaluation/
#     test_face_verification.m    ← MATLAB protocol (distance-based, 5320+5320)
# ─────────────────────────────────────────────────────────────────────────────

QMUL_ROOT    = ROOT / "data" / "QMUL-SurvFace"
VTS_DIR      = QMUL_ROOT / "Face_Verification_Test_Set"
POS_MAT      = VTS_DIR / "positive_pairs_names.mat"
NEG_MAT      = VTS_DIR / "negative_pairs_names.mat"
IMG_DIR      = VTS_DIR / "verification_images"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD .mat PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs_mat(mat_path: Path) -> list[tuple[str, str]]:
    """
    Load positive_pairs_names.mat or negative_pairs_names.mat.
    The MATLAB variable is typically a cell array of shape (N, 2).
    Returns list of (img1_name, img2_name).
    """
    mat = scipy.io.loadmat(str(mat_path))

    # find the actual data key (not the metadata keys __header__ etc.)
    data_keys = [k for k in mat.keys() if not k.startswith("__")]
    assert data_keys, f"No data keys in {mat_path}"
    arr = mat[data_keys[0]]          # shape (N, 2) cell array → numpy object array

    pairs = []
    for row in arr:
        # each element may be a nested array; flatten to string
        def to_str(x):
            while hasattr(x, "__len__") and not isinstance(x, str):
                if len(x) == 1:
                    x = x[0]
                else:
                    x = x.flat[0]
            return str(x).strip()
        pairs.append((to_str(row[0]), to_str(row[1])))

    log.info(f"Loaded {len(pairs)} pairs from {mat_path.name}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_adaface_ir101(ckpt_path: str, device: torch.device):
    try:
        from adaface_ir101_loader import load_adaface_ir101 as _load
        model = _load(ckpt_path, device)
        log.info("AdaFace IR-101 loaded")
        return model
    except Exception as e:
        log.warning(f"adaface_ir101_loader failed ({e}), using iresnet100 fallback")
        try:
            from src.models.iresnet import iresnet100
        except ImportError:
            from src.models.iresnet import IResNet
            def iresnet100(**kw): return IResNet(num_layers=100, **kw)
        model = iresnet100(fp16=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("model.", "").replace("backbone.", ""): v
              for k, v in sd.items() if not k.startswith("head.")}
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model


def load_arcface_iresnet50(ckpt_path: str, device: torch.device):
    model = iresnet50(fp16=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    sd    = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    log.info("ArcFace iResNet-50 loaded")
    return model


def load_buffalo_l(model_dir: str, device: torch.device):
    try:
        from insightface.app import FaceAnalysis
        root = str(Path(model_dir).parent.parent)
        app  = FaceAnalysis(name="buffalo_l", root=root,
                            providers=["CUDAExecutionProvider",
                                       "CPUExecutionProvider"])
        app.prepare(ctx_id=-1 if device.type == "cuda" else -1,
                    det_size=(112, 112))
        log.info("buffalo_l loaded")
        return app
    except Exception as e:
        log.error(f"buffalo_l failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AdaFace-SR
# ─────────────────────────────────────────────────────────────────────────────

def load_adaface_sr(ckpt_path: str, device: torch.device) -> AdaFaceSR:
    model = AdaFaceSR().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    sd    = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.eval()
    log.info(f"AdaFace-SR loaded from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(
        cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0
    return (torch.from_numpy(img).permute(2, 0, 1) - MEAN) / STD


def load_image(name: str, img_dir: Path) -> np.ndarray:
    """Try name as-is, then with common extensions."""
    for candidate in [img_dir / name,
                      img_dir / (name + ".jpg"),
                      img_dir / (name + ".png"),
                      img_dir / (Path(name).stem + ".jpg")]:
        img = cv2.imread(str(candidate))
        if img is not None:
            return img
    log.warning(f"Image not found: {name}")
    return np.zeros((112, 112, 3), dtype=np.uint8)


@torch.no_grad()
def embed_pytorch(model, imgs: list, batch: int,
                  device: torch.device) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), batch):
        t = torch.stack([to_tensor(im) for im in imgs[i:i+batch]]).to(device)
        f = model(t)
        out.append(F.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_buffalo(app, imgs: list) -> np.ndarray:
    feats = []
    for img in imgs:
        faces = app.get(img)
        feats.append(faces[0].normed_embedding if faces
                     else np.zeros(512, dtype=np.float32))
    return np.array(feats)


@torch.no_grad()
def enhance_batch(sr_model: AdaFaceSR, imgs: list, input_res: int,
                  batch: int, device: torch.device) -> list:
    out = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i+batch]
        tensors = [to_tensor(
            cv2.resize(img, (input_res, input_res), interpolation=cv2.INTER_AREA)
        ) for img in chunk]
        t   = torch.stack(tensors).to(device)
        res = torch.full((len(chunk),), float(input_res), device=device)
        enh = sr_model(t, res).clamp(-1, 1)
        enh_np = ((enh.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        for e in enh_np:
            out.append(cv2.cvtColor(np.transpose(e, (1, 2, 0)),
                                    cv2.COLOR_RGB2BGR))
    return out


def bicubic_batch(imgs: list, input_res: int) -> list:
    out = []
    for img in imgs:
        lr = cv2.resize(img, (input_res, input_res), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(lr, (112, 112), interpolation=cv2.INTER_CUBIC))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OFFICIAL QMUL METRICS  (mirrors test_face_verification.m exactly)
# NOTE: scores here are L2 DISTANCES — smaller = more similar
#       so genuine pairs should have LOWER scores than impostors
#       TAR = fraction of pos pairs with score < threshold
#       FAR = fraction of neg pairs with score < threshold
# ─────────────────────────────────────────────────────────────────────────────

FAR_TARGETS = [0.3, 0.1, 0.01, 0.001]


def compute_official_metrics(pos_scores: np.ndarray,
                              neg_scores: np.ndarray) -> dict:
    """
    Exact Python translation of test_face_verification.m.
    pos_scores, neg_scores: L2 distances (lower = more similar).
    """
    initial_threshold = 1.1
    rng  = 0.9
    step = 0.001
    thresholds = np.arange(initial_threshold - rng,
                           initial_threshold + rng + step,
                           step)

    # TAR and FAR at each threshold
    FARs = np.array([np.sum(neg_scores < t) / len(neg_scores)
                     for t in thresholds])
    TARs = np.array([np.sum(pos_scores < t) / len(pos_scores)
                     for t in thresholds])

    # sort by FAR ascending (matches MATLAB sort)
    sort_idx = np.argsort(FARs)
    FARs = FARs[sort_idx]
    TARs = TARs[sort_idx]

    # TAR@FAR interpolation
    interp   = interp1d(FARs, TARs, kind="linear",
                        bounds_error=False, fill_value=(0.0, 1.0))
    tar_dict = {f: float(interp(f)) for f in FAR_TARGETS}

    # AUC  (trapz, matching MATLAB trapz(FARs,TARs))
    auc = float(np.trapz(TARs, FARs))

    # 10-fold mean accuracy
    N          = len(pos_scores)
    fold_len   = N // 10
    accuracies = []
    for i in range(10):
        s, e = i * fold_len, (i + 1) * fold_len
        test_p = pos_scores[s:e];  test_n = neg_scores[s:e]
        mask   = np.ones(N, dtype=bool); mask[s:e] = False
        train_p = pos_scores[mask];  train_n = neg_scores[mask]

        best_acc, best_thresh = 0.0, initial_threshold
        for t in thresholds:
            acc = (np.sum(train_p < t) + np.sum(train_n >= t)) / \
                  (len(train_p) + len(train_n))
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        fold_acc = (np.sum(test_p < best_thresh) +
                    np.sum(test_n >= best_thresh)) / (len(test_p) + len(test_n))
        accuracies.append(fold_acc)

    return {
        "auc":          auc * 100,        # percent, like leaderboard
        "tar_at_far":   {str(k): v * 100 for k, v in tar_dict.items()},
        "mean_accuracy":float(np.mean(accuracies)) * 100,
        "pos_mean":     float(pos_scores.mean()),
        "neg_mean":     float(neg_scores.mean()),
        "score_gap":    float(neg_scores.mean() - pos_scores.mean()),  # higher = better
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(embed_fn, sr_model, input_res: int,
                 pos_pairs: list, neg_pairs: list,
                 img_dir: Path, device: torch.device, batch: int) -> dict:
    """
    Run one (backbone, method) combination.
    Returns metrics dict from compute_official_metrics.
    """
    # collect unique image names to avoid loading duplicates
    all_names = list({n for p in pos_pairs + neg_pairs for n in p})
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    log.info(f"    Loading {len(all_names)} unique images ...")
    raw_imgs = [load_image(n, img_dir) for n in tqdm(all_names, leave=False)]

    # enhance / bicubic
    if sr_model is not None:
        log.info(f"    AdaFace-SR enhancement ({input_res}px) ...")
        enhanced = enhance_batch(sr_model, raw_imgs, input_res, batch, device)
    else:
        log.info(f"    Bicubic baseline ({input_res}px) ...")
        enhanced = bicubic_batch(raw_imgs, input_res)

    # embed all unique images once
    log.info("    Embedding ...")
    embs = embed_fn(enhanced)                          # (N, D)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    # compute L2 distances for each pair
    def pair_l2(pairs):
        dists = []
        for a, b in pairs:
            ea = embs[name_to_idx[a]]
            eb = embs[name_to_idx[b]]
            dists.append(float(np.linalg.norm(ea - eb)))
        return np.array(dists)

    pos_scores = pair_l2(pos_pairs)
    neg_scores = pair_l2(neg_pairs)

    return compute_official_metrics(pos_scores, neg_scores)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict):
    W = 82
    print("\n" + "═"*W)
    print("  BACKBONE ABLATION — QMUL-SurvFace  (official distance protocol)")
    print("═"*W)
    hdr = (f"  {'Backbone':<22} {'Method':<14} {'AUC':>6} "
           f"{'@30%':>7} {'@10%':>7} {'@1%':>7} {'@0.1%':>7} "
           f"{'MeanAcc':>8} {'Gap':>7}")
    print(hdr)
    print("  " + "─"*(W-2))
    for backbone, methods in results.items():
        for method, r in methods.items():
            t = r["tar_at_far"]
            print(f"  {backbone:<22} {method:<14}"
                  f" {r['auc']:>6.1f}"
                  f" {t.get('0.3',0):>7.1f}"
                  f" {t.get('0.1',0):>7.1f}"
                  f" {t.get('0.01',0):>7.1f}"
                  f" {t.get('0.001',0):>7.1f}"
                  f" {r['mean_accuracy']:>8.1f}"
                  f" {r['score_gap']:>7.3f}")
        print()

    print("  ── Δ AdaFace-SR over Bicubic (TAR@FAR=1%) ──\n")
    for backbone, methods in results.items():
        bic = methods.get("Bicubic",    {}).get("tar_at_far", {}).get("0.01", 0)
        ada = methods.get("AdaFace-SR", {}).get("tar_at_far", {}).get("0.01", 0)
        d   = ada - bic
        tag = "✓ improves" if d > 0 else "✗ regresses"
        print(f"  {backbone:<22}  bic={bic:.1f}%  ada={ada:.1f}%  "
              f"Δ={d:+.1f}pp  {tag}")
    print("═"*W + "\n")


def write_latex(results: dict, out_path: Path):
    rows = []
    for backbone, methods in results.items():
        bic_t1 = methods.get("Bicubic", {}).get("tar_at_far", {}).get("0.01", 0)
        for method, r in methods.items():
            t      = r["tar_at_far"]
            ada_t1 = t.get("0.01", 0)
            delta_str, row_color = "", ""
            if method == "AdaFace-SR":
                d = ada_t1 - bic_t1
                delta_str = f" (${d:+.1f}$\\,pp)"
                row_color = (r"\rowcolor{best}" if d >= 0 else r"\rowcolor{bad}") + "\n  "
            b = backbone.replace("_", r"\_")
            rows.append(
                f"  {row_color}{b} & {method}{delta_str} & "
                f"{r['auc']:.1f} & "
                f"{t.get('0.3',0):.1f} & "
                f"{t.get('0.1',0):.1f} & "
                f"{t.get('0.01',0):.1f} & "
                f"{t.get('0.001',0):.1f} & "
                f"{r['mean_accuracy']:.1f} \\\\"
            )
        rows.append(r"  \midrule")

    body = "\n".join(rows[:-1])
    tex = rf"""\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{%
  Backbone Independence of AdaFace-SR on QMUL-SurvFace
  (official evaluation protocol: L2 distance, 5320 positive + 5320 negative pairs).
  $\Delta$ = TAR@FAR$=1\%$ change over bicubic baseline.
}}
\label{{tab:backbone_ablation}}
\centering
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}llccccccc@{{}}}}
\toprule
\textbf{{FR Backbone}} & \textbf{{Method}}
  & \textbf{{AUC}} & \textbf{{@30\%}} & \textbf{{@10\%}}
  & \textbf{{@1\%}} & \textbf{{@0.1\%}} & \textbf{{MeanAcc}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    out_path.write_text(tex)
    log.info(f"LaTeX → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adaface_sr_ckpt", default=
        "experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth")
    p.add_argument("--ir101_ckpt",  default=
        "pretrained/recognition/adaface_ir101_webface4m.ckpt")
    p.add_argument("--ir50_ckpt",   default=
        "pretrained/arcface_pytorch/iresnet50.pth")
    p.add_argument("--buffalo_dir", default=
        "pretrained/recognition/models/buffalo_l")
    p.add_argument("--input_res",  type=int, default=16,
                   help="Simulated LR resolution fed to AdaFace-SR")
    p.add_argument("--output_dir", default="results/backbone_ablation")
    p.add_argument("--batch",      type=int, default=32)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load pairs ───────────────────────────────────────────────────────────
    log.info("Loading pairs from .mat files ...")
    pos_pairs = load_pairs_mat(POS_MAT)   # 5320 genuine
    neg_pairs = load_pairs_mat(NEG_MAT)   # 5320 impostor
    assert len(pos_pairs) == 5320, f"Expected 5320 pos pairs, got {len(pos_pairs)}"
    assert len(neg_pairs) == 5320, f"Expected 5320 neg pairs, got {len(neg_pairs)}"

    # ── load AdaFace-SR ──────────────────────────────────────────────────────
    sr_model = load_adaface_sr(args.adaface_sr_ckpt, device)

    # ── define backbones ─────────────────────────────────────────────────────
    backbones = {}

    if Path(args.ir101_ckpt).exists():
        m = load_adaface_ir101(args.ir101_ckpt, device)
        backbones["AdaFace IR-101"] = \
            lambda imgs, _m=m: embed_pytorch(_m, imgs, args.batch, device)

    if Path(args.ir50_ckpt).exists():
        m = load_arcface_iresnet50(args.ir50_ckpt, device)
        backbones["ArcFace iResNet-50"] = \
            lambda imgs, _m=m: embed_pytorch(_m, imgs, args.batch, device)

    if Path(args.buffalo_dir).exists():
        app = load_buffalo_l(args.buffalo_dir, device)
        if app:
            backbones["buffalo_l"] = \
                lambda imgs, _a=app: embed_buffalo(_a, imgs)

    if not backbones:
        log.error("No backbones loaded — check checkpoint paths.")
        sys.exit(1)

    # ── evaluate ─────────────────────────────────────────────────────────────
    results = {}
    for name, embed_fn in backbones.items():
        results[name] = {}
        for method, srm in [("Bicubic", None), ("AdaFace-SR", sr_model)]:
            log.info(f"[{name}]  {method} ...")
            r = evaluate_one(embed_fn, srm, args.input_res,
                             pos_pairs, neg_pairs,
                             IMG_DIR, device, args.batch)
            results[name][method] = r
            log.info(f"  AUC={r['auc']:.1f}%  "
                     f"TAR@1%={r['tar_at_far'].get('0.01',0):.1f}%  "
                     f"MeanAcc={r['mean_accuracy']:.1f}%")

    # ── save & display ────────────────────────────────────────────────────────
    json_path = out_dir / "qmul_backbone_ablation.json"
    json_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results → {json_path}")

    write_latex(results, out_dir / "table_backbone_ablation.tex")
    print_results(results)


if __name__ == "__main__":
    main()