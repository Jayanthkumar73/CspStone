"""
eval_qmul_backbones.py — FIXED VERSION
Tests AdaFace-SR v2 + Bicubic with 3 FR backbones on QMUL-SurvFace.
Fixes:
  - Model path: src/models/adaface_sr.py  (not models/)
  - IR-101 loader: src/ structure
  - Direct ONNX for buffalo_l (no RetinaFace detection crash)

Usage:
  cd /raid/home/dgxuser8/capstone1/version1
  source ../capstone1/bin/activate
  python scripts/eval_qmul_backbones.py 2>&1 | tee eval_backbone_ablation.txt
"""

import sys, torch, numpy as np, scipy.io, onnxruntime as ort
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc as sk_auc

BASE     = Path('/raid/home/dgxuser8/capstone1/version1')
QMUL_DIR = BASE / 'data/QMUL-SurvFace'
PRETRAIN = BASE / 'pretrained'
SR_CKPT  = BASE / 'experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ════════════════════════════════════════════════════════════════════════════
# QMUL pair loading
# ════════════════════════════════════════════════════════════════════════════
def load_qmul_pairs():
    pos_mat = QMUL_DIR / 'Face_Verification_Test_Set/positive_pairs_names.mat'
    neg_mat = QMUL_DIR / 'Face_Verification_Test_Set/negative_pairs_names.mat'

    def parse(mat_path):
        data = scipy.io.loadmat(str(mat_path))
        key  = [k for k in data if not k.startswith('_')][0]
        raw  = data[key]
        pairs = []
        for i in range(raw.shape[0]):
            try:
                a = str(raw[i,0][0])
                b = str(raw[i,1][0])
                if a and b:
                    pairs.append((a, b))
            except Exception:
                continue
        return pairs

    pos = parse(pos_mat)
    neg = parse(neg_mat)
    print(f"[QMUL]  {len(pos)} positive + {len(neg)} negative pairs")
    return pos, neg

def find_image(filename):
    """Search for QMUL image across known subdirectory layouts."""
    fname = Path(filename).name
    search_roots = [
        QMUL_DIR / 'surveillance_face_images',
        QMUL_DIR / 'face_images',
        QMUL_DIR / 'images',
        QMUL_DIR,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        # Direct path
        direct = root / filename
        if direct.exists():
            return direct
        # Just filename
        by_name = root / fname
        if by_name.exists():
            return by_name
        # Recursive search (slower, last resort)
        matches = list(root.rglob(fname))
        if matches:
            return matches[0]
    return None

# ════════════════════════════════════════════════════════════════════════════
# AdaFace-SR
# ════════════════════════════════════════════════════════════════════════════
def load_adaface_sr():
    sys.path.insert(0, str(BASE / 'src'))          # FIX: src/models/adaface_sr.py
    from models.adaface_sr import AdaFaceSR
    model = AdaFaceSR().to(DEVICE)
    ckpt  = torch.load(SR_CKPT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[SR]    Loaded AdaFace-SR v2  ({SR_CKPT.name})")
    return model

def apply_sr(model, img_pil):
    tf = transforms.Compose([
        transforms.Resize((112,112), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    x = tf(img_pil).unsqueeze(0).to(DEVICE)
    # Estimate resolution from image dimensions
    w, h = img_pil.size
    r = torch.tensor([min(w, h, 112)], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(x, r)
    out = (out.squeeze(0).cpu().clamp(-1,1) * 0.5 + 0.5).clamp(0,1)
    return transforms.ToPILImage()(out)

def apply_bicubic(img_pil):
    return img_pil.resize((112,112), Image.BICUBIC)

# ════════════════════════════════════════════════════════════════════════════
# BACKBONE 1: buffalo_l via direct ONNX (no RetinaFace detection crash)
# ════════════════════════════════════════════════════════════════════════════
class BuffaloLDirect:
    def __init__(self):
        path = str(PRETRAIN / 'recognition/models/buffalo_l/w600k_r50.onnx')
        self.sess = ort.InferenceSession(
            path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.inp  = self.sess.get_inputs()[0].name
        print(f"[BB1]   buffalo_l  (w600k_r50 ONNX direct)")

    def embed(self, img_pil):
        arr = np.array(img_pil.convert('RGB').resize((112,112), Image.BICUBIC),
                       dtype=np.float32)
        arr = (arr - 127.5) / 128.0          # ArcFace [-1,1] normalisation
        arr = arr.transpose(2,0,1)[None]      # NCHW
        e   = self.sess.run(None, {self.inp: arr})[0][0]
        return e / (np.linalg.norm(e) + 1e-8)

# ════════════════════════════════════════════════════════════════════════════
# BACKBONE 2: ArcFace iResNet-50
# ════════════════════════════════════════════════════════════════════════════
class IResNet50:
    def __init__(self):
        sys.path.insert(0, str(BASE / 'src'))
        try:
            from models.iresnet import iresnet50
        except ImportError:
            from backbones.iresnet import iresnet50
        self.model = iresnet50().to(DEVICE)
        ckpt  = torch.load(str(PRETRAIN / 'arcface_pytorch/iresnet50.pth'),
                           map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = {k.replace('module.','').replace('model.',''): v
                 for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())/1e6
        print(f"[BB2]   ArcFace iResNet-50  ({n:.1f}M params)")
        self.tf = transforms.Compose([
            transforms.Resize((112,112), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def embed(self, img_pil):
        x = self.tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(x)
        e = (out[0] if isinstance(out,(list,tuple)) else out).squeeze(0).cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

# ════════════════════════════════════════════════════════════════════════════
# BACKBONE 3: AdaFace IR-101
# ════════════════════════════════════════════════════════════════════════════
class AdaFaceIR101:
    def __init__(self):
        sys.path.insert(0, str(BASE / 'src'))
        try:
            from models.net import build_model
            self.model = build_model('ir_101').to(DEVICE)
        except Exception:
            try:
                from backbones.iresnet import iresnet100
                self.model = iresnet100().to(DEVICE)
            except Exception:
                from models.iresnet import iresnet100
                self.model = iresnet100().to(DEVICE)

        ckpt  = torch.load(
            str(PRETRAIN / 'recognition/adaface_ir101_webface4m.ckpt'),
            map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = {k.replace('model.','').replace('module.',''): v
                 for k, v in state.items()
                 if not k.startswith('head.')}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())/1e6
        print(f"[BB3]   AdaFace IR-101  ({n:.1f}M params)")
        self.tf = transforms.Compose([
            transforms.Resize((112,112), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def embed(self, img_pil):
        x = self.tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(x)
        e = (out[0] if isinstance(out,(list,tuple)) else out).squeeze(0).cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

# ════════════════════════════════════════════════════════════════════════════
# Verification metrics
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = sk_auc(fpr, tpr) * 100
    tar = {}
    for far in [0.30, 0.10, 0.01, 0.001]:
        idx = max(0, np.searchsorted(fpr, far, side='right') - 1)
        tar[far] = tpr[idx] * 100
    gen = scores[labels==1]
    imp = scores[labels==0]
    return roc_auc, tar, float(gen.mean()), float(imp.mean())

def run_verification(bb, sr_fn, pos_pairs, neg_pairs, bb_name, method_name):
    all_pairs = [(p,1) for p in pos_pairs] + [(p,0) for p in neg_pairs]
    scores, labels, skipped = [], [], 0
    total = len(all_pairs)

    print(f"  [{bb_name} + {method_name}]  {total} pairs ...", flush=True)

    for idx, ((fa, fb), label) in enumerate(all_pairs):
        if idx % 500 == 0:
            print(f"    {idx}/{total}", end='\r', flush=True)

        pa = find_image(fa)
        pb = find_image(fb)
        if pa is None or pb is None:
            skipped += 1
            continue

        try:
            ia = Image.open(pa).convert('RGB')
            ib = Image.open(pb).convert('RGB')
            ea = bb.embed(sr_fn(ia))
            eb = bb.embed(sr_fn(ib))
        except Exception:
            skipped += 1
            continue

        scores.append(float(np.dot(ea, eb)))
        labels.append(label)

    print(f"    done  {len(scores)} pairs  ({skipped} skipped)")

    if len(scores) < 200:
        print(f"    WARNING: too few pairs ({len(scores)}) — check image paths")
        return None

    roc_auc, tar, gen_mean, imp_mean = compute_metrics(scores, labels)
    return {
        'backbone': bb_name, 'method': method_name,
        'n': len(scores), 'skipped': skipped,
        'AUC': roc_auc,
        'TAR@30': tar[0.30], 'TAR@10': tar[0.10],
        'TAR@1':  tar[0.01], 'TAR@0.1': tar[0.001],
        'gen_mean': gen_mean, 'imp_mean': imp_mean,
        'gap': gen_mean - imp_mean,
    }

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("QMUL-SurvFace Backbone Independence Evaluation")
    print("="*65)
    print(f"Device: {DEVICE}")
    print(f"SR ckpt: {SR_CKPT}")

    pos_pairs, neg_pairs = load_qmul_pairs()

    # Load SR model
    sr_model = load_adaface_sr()
    methods  = {
        'Bicubic':    apply_bicubic,
        'AdaFace-SR': lambda img: apply_sr(sr_model, img),
    }

    # Load backbones — skip gracefully if import fails
    backbones = []

    try:
        backbones.append(('buffalo_l',       BuffaloLDirect()))
    except Exception as e:
        print(f"[WARN] buffalo_l failed: {e}")

    try:
        backbones.append(('iResNet-50',      IResNet50()))
    except Exception as e:
        print(f"[WARN] iResNet-50 failed: {e}")

    try:
        backbones.append(('AdaFace-IR101',   AdaFaceIR101()))
    except Exception as e:
        print(f"[WARN] AdaFace-IR101 failed: {e}")

    if not backbones:
        print("ERROR: no backbones loaded.")
        sys.exit(1)

    # Run all combinations
    all_results = []
    for bb_name, bb in backbones:
        for m_name, m_fn in methods.items():
            r = run_verification(bb, m_fn, pos_pairs, neg_pairs, bb_name, m_name)
            if r:
                all_results.append(r)
                print(f"  AUC={r['AUC']:.1f}%  @10%={r['TAR@10']:.1f}%  "
                      f"@1%={r['TAR@1']:.1f}%  @0.1%={r['TAR@0.1']:.1f}%  "
                      f"gap={r['gap']:.3f}")

    # ── Output table ─────────────────────────────────────────────────────────
    SEP = "="*75
    lines = [
        "", SEP,
        "BACKBONE INDEPENDENCE — QMUL-SurvFace Verification",
        "Paste into Table XVI (tab:backbone_ablation) in journal_paper_v6.tex",
        SEP,
        f"{'Backbone':<20} {'Method':<14} {'AUC':>7} {'@10%':>7} "
        f"{'@1%':>7} {'@0.1%':>7} {'Gap':>7}",
        "-"*75,
    ]
    for r in all_results:
        lines.append(
            f"{r['backbone']:<20} {r['method']:<14} "
            f"{r['AUC']:>7.1f} {r['TAR@10']:>7.1f} "
            f"{r['TAR@1']:>7.1f} {r['TAR@0.1']:>7.1f} "
            f"{r['gap']:>7.3f}"
        )
    lines.append(SEP)

    # Delta AdaFace-SR vs Bicubic per backbone
    lines.append("\nAdaFace-SR gain over Bicubic (TAR@1%):")
    bb_res = {}
    for r in all_results:
        bb_res.setdefault(r['backbone'], {})[r['method']] = r
    for bb, ms in bb_res.items():
        bic = ms.get('Bicubic', {}).get('TAR@1')
        asr = ms.get('AdaFace-SR', {}).get('TAR@1')
        if bic is not None and asr is not None:
            lines.append(f"  {bb:<20}: {asr-bic:+.2f}pp  "
                         f"({bic:.1f}% → {asr:.1f}%)")

    output = '\n'.join(lines)
    print(output)

    out = BASE / 'eval_backbone_ablation.txt'
    out.write_text(output)
    print(f"\nSaved → {out}")

if __name__ == '__main__':
    main()