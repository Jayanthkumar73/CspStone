"""
eval_ablation_scface.py — FINAL WORKING VERSION
Fixes:
  - InsightFace shape broadcast error on mugshots → use direct embedding
    via buffalo_l recognition model, bypassing RetinaFace detection
  - Correct probe dir: surveillance_cameras_all/
  - Correct model path: src/models/adaface_sr.py
  - Camera-distance mapping: cam1-3=d1, cam4-5=d2, cam6-7=d3

Usage:
  cd /raid/home/dgxuser8/capstone1/version1
  source ../capstone1/bin/activate
  python scripts/eval_ablation_scface.py 2>&1 | tee eval_ablation_results.txt
  python scripts/eval_ablation_scface.py --variant adaface_nogate
"""

import sys, argparse, torch, numpy as np, onnxruntime as ort
from pathlib import Path
from PIL import Image
from torchvision import transforms

BASE      = Path('/raid/home/dgxuser8/capstone1/version1')
PROBE_DIR = BASE / 'data/scface/SCface_database/surveillance_cameras_all'
GAL_DIR   = BASE / 'data/scface/organized/mugshot'
EXPS      = BASE / 'experiments'

# cam1-3 = distance 1 (~32px), cam4-5 = distance 2 (~24px), cam6-7 = distance 3 (~16px)
DIST_CAMS = {
    'd1': {'cam1','cam2','cam3'},
    'd2': {'cam4','cam5'},
    'd3': {'cam6','cam7'},
}

VARIANTS = {
    'adaface_nogate':   EXPS / 'adaface_nogate/checkpoints/checkpoint_best.pth',
    'adaface_nocomp':   EXPS / 'adaface_nocomp/checkpoints/checkpoint_best.pth',
    'adaface_noresenc': EXPS / 'adaface_noresenc/checkpoints/checkpoint_best.pth',
}

CONFIRMED = {
    'AdaFace-SR v2': {'16': 18.3, '24': 46.8, '32': 70.0, 'nat': 99.6},
    'PALF-Net':      {'16':  7.8, '24': 31.4, '32': 56.9, 'nat': 95.8},
    'Bicubic':       {'16': 22.7, '24': 57.5, '32': 84.0, 'nat': 99.7},
}

RESOLUTIONS = {'16': 16, '24': 24, '32': 32, 'nat': None}

# ════════════════════════════════════════════════════════════════════════════
# Direct ONNX embedding — bypasses RetinaFace detection entirely
# Uses buffalo_l w600k_r50.onnx directly on a pre-cropped 112x112 face
# This is exactly what insightface does internally after detection
# ════════════════════════════════════════════════════════════════════════════
class DirectFRModel:
    """
    Loads w600k_r50.onnx and runs inference directly on 112x112 RGB crops.
    No face detection needed — SCface images are already face-cropped.
    """
    def __init__(self):
        onnx_path = str(BASE / 'pretrained/recognition/models/buffalo_l/w600k_r50.onnx')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[FR] Loaded {Path(onnx_path).name} via ONNX  "
              f"(input: {self.input_name})")

    def get_embedding(self, img_pil):
        """
        img_pil: PIL image, will be resized to 112x112.
        Returns L2-normalised 512-dim numpy array.
        """
        # Resize to 112x112
        img = img_pil.convert('RGB').resize((112, 112), Image.BICUBIC)
        # To float32, normalise to [-1, 1] (ArcFace convention)
        arr = np.array(img, dtype=np.float32)          # HWC uint8 -> float
        arr = (arr - 127.5) / 128.0                    # [-1, 1]
        arr = arr.transpose(2, 0, 1)[np.newaxis]       # NCHW
        emb = self.sess.run(None, {self.input_name: arr})[0][0]  # (512,)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)


# ════════════════════════════════════════════════════════════════════════════
# Pair loading
# ════════════════════════════════════════════════════════════════════════════
def build_gallery():
    index = {}
    for p in GAL_DIR.glob('*.jpg'):
        subj = p.stem.split('_')[0].zfill(3)
        # Prefer frontal; only overwrite if we don't have frontal yet
        if subj not in index or 'frontal' in p.stem:
            index[subj] = p
    return index

def load_pairs(camera):
    cams    = DIST_CAMS[camera]
    gallery = build_gallery()
    pairs   = []
    no_gal  = 0
    for p in sorted(PROBE_DIR.glob('*.jpg')):
        # filename: 001_cam1_1.jpg → subj=001, cam=cam1
        parts = p.stem.split('_')        # ['001', 'cam1', '1']
        subj  = parts[0].zfill(3)
        cam   = parts[1] if len(parts) > 1 else ''
        if cam not in cams:
            continue
        if subj in gallery:
            pairs.append((p, gallery[subj], subj))
        else:
            no_gal += 1
    if no_gal:
        print(f"  [warn] {no_gal} probes had no gallery match")
    n_subj = len(set(s for _,_,s in pairs))
    print(f"[SCface] {camera} ({cams}): {len(pairs)} pairs, {n_subj} subjects")
    return pairs

# ════════════════════════════════════════════════════════════════════════════
# SR models
# ════════════════════════════════════════════════════════════════════════════
def load_sr(ckpt_path, device):
    sys.path.insert(0, str(BASE / 'src'))
    from models.adaface_sr import AdaFaceSR
    model = AdaFaceSR().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded SR checkpoint: {ckpt_path.name}")
    return model

def apply_sr(model, img_pil, res, device):
    tf = transforms.Compose([
        transforms.Resize((112,112), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    x = tf(img_pil).unsqueeze(0).to(device)
    r = torch.tensor([res or 112], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x, r)
    out = (out.squeeze(0).cpu().clamp(-1,1) * 0.5 + 0.5).clamp(0,1)
    return transforms.ToPILImage()(out)

def apply_bicubic(img_pil, res):
    if res is None:
        return img_pil.resize((112,112), Image.BICUBIC)
    return img_pil.resize((res,res), Image.BICUBIC).resize((112,112), Image.BICUBIC)

# ════════════════════════════════════════════════════════════════════════════
# Rank-1 evaluation
# ════════════════════════════════════════════════════════════════════════════
def rank1(fr, pairs, enhance_fn, res, tag=''):
    # Build gallery embeddings
    gal_embs = {}
    for _, gal_path, sid in pairs:
        if sid in gal_embs:
            continue
        gal_pil = Image.open(gal_path).convert('RGB')
        gal_embs[sid] = fr.get_embedding(gal_pil)

    n_subj = len(set(s for _,_,s in pairs))
    print(f"  Gallery: {len(gal_embs)}/{n_subj} embedded", flush=True)

    correct = total = errors = 0
    for i, (probe_path, _, true_sid) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  [{tag}] {i}/{len(pairs)}", end='\r', flush=True)
        try:
            img = Image.open(probe_path).convert('RGB')
            enh = enhance_fn(img, res)
            if enh.size != (112,112):
                enh = enh.resize((112,112), Image.BICUBIC)
            pe = fr.get_embedding(enh)
        except Exception as ex:
            errors += 1
            continue

        best_sid = max(gal_embs, key=lambda s: float(np.dot(pe, gal_embs[s])))
        if best_sid == true_sid:
            correct += 1
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"  [{tag}] {correct}/{total}  ({errors} errors)  →  {acc:.1f}%")
    return acc

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all',
                    choices=['all'] + list(VARIANTS.keys()))
    ap.add_argument('--camera', default='d3', choices=['d1','d2','d3'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Camera: {args.camera}")

    # Path check
    n_probes = len(list(PROBE_DIR.glob('*.jpg')))
    n_gal    = len(list(GAL_DIR.glob('*.jpg')))
    print(f"  Probes : {PROBE_DIR.name}  ({n_probes} files)")
    print(f"  Gallery: {GAL_DIR.name}  ({n_gal} files)")
    if n_probes == 0 or n_gal == 0:
        print("ERROR: probe or gallery directory is empty. Check paths.")
        sys.exit(1)

    fr    = DirectFRModel()
    pairs = load_pairs(args.camera)
    if not pairs:
        print("ERROR: 0 pairs. Check DIST_CAMS mapping.")
        sys.exit(1)

    to_run = (list(VARIANTS.items()) if args.variant == 'all'
              else [(args.variant, VARIANTS[args.variant])])

    results = {}

    # Bicubic
    print("\n── Bicubic baseline ──")
    bic = {}
    for rn, rv in RESOLUTIONS.items():
        bic[rn] = rank1(fr, pairs, apply_bicubic, rv, f'Bic-{rn}')
    results['Bicubic'] = bic

    # SR variants
    for vname, ckpt in to_run:
        print(f"\n── {vname} ──")
        if not ckpt.exists():
            print(f"  SKIP: {ckpt} not found")
            results[vname] = {k: 'N/A' for k in RESOLUTIONS}
            continue
        model = load_sr(ckpt, device)
        vres  = {}
        for rn, rv in RESOLUTIONS.items():
            vres[rn] = rank1(fr, pairs,
                             lambda img,r,m=model,d=device: apply_sr(m,img,r,d),
                             rv, f'{vname[:12]}-{rn}')
        results[vname] = vres

    # Output table
    SEP = "=" * 70
    lines = [
        "", SEP,
        f"ABLATION TABLE — SCface {args.camera.upper()} Rank-1 (%)",
        "Paste into Table XV (tab:ablation) in journal_paper_v6.tex",
        SEP,
        f"{'Method':<26} {'16px':>7} {'24px':>7} {'32px':>7} {'Nat':>7}",
        "-"*70, "[Confirmed — already in paper]",
    ]
    for m, r in CONFIRMED.items():
        lines.append(f"{m:<26}"
                     + "".join(f"  {r[k]:>5.1f}" for k in ['16','24','32','nat']))
    lines += ["-"*70, "[New ablation results]"]
    bic16 = results.get('Bicubic', {}).get('16')
    for m, r in results.items():
        row = f"{m:<26}"
        for k in ['16','24','32','nat']:
            v = r.get(k)
            row += f"  {v:>5.1f}" if isinstance(v, float) else f"  {'N/A':>5}"
        lines.append(row)
    lines.append(SEP)
    if bic16:
        lines.append("\nΔ vs Bicubic at 16px:")
        for m, r in results.items():
            if m == 'Bicubic': continue
            v = r.get('16')
            if isinstance(v, float):
                lines.append(f"  {m:<26}: {v - bic16:+.1f}pp")

    output = '\n'.join(lines)
    print(output)
    out = BASE / 'eval_ablation_results.txt'
    out.write_text(output)
    print(f"\nSaved → {out}")

if __name__ == '__main__':
    main()