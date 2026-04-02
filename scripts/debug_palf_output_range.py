"""
debug_palf_output_range.py
--------------------------
Checks the actual output range of PALFNet and how the eval scripts
handled it, so we can render the qualitative figures correctly.

Run:
    python scripts/debug_palf_output_range.py \
        --palfnet_ckpt experiments/palfnet_v1/checkpoints/checkpoint_best.pth \
        --scface_root  data/scface
"""
import sys, argparse
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.palfnet import PALFNet


def find_probe(scface_root):
    d3 = Path(scface_root) / "organized" / "surveillance" / "d3"
    for p in sorted(d3.glob("*.jpg")) + sorted(d3.glob("*.JPG")):
        img = cv2.imread(str(p))
        if img is not None:
            return img, str(p)
    raise FileNotFoundError(f"No probe in {d3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--palfnet_ckpt", default=
        "experiments/palfnet_v1/checkpoints/checkpoint_best.pth")
    parser.add_argument("--scface_root", default="data/scface")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = PALFNet().to(device)
    ck = torch.load(args.palfnet_ckpt, map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck.get("state_dict", ck)),
                          strict=False)
    model.eval()

    probe_bgr, probe_path = find_probe(args.scface_root)
    print(f"Probe: {probe_path}")

    small = cv2.resize(probe_bgr, (16,16), interpolation=cv2.INTER_AREA)
    bic   = cv2.resize(small, (112,112), interpolation=cv2.INTER_CUBIC)

    # Test both input normalisations
    for name, t in [
        ("[0,1] input  ",
         torch.from_numpy(cv2.cvtColor(bic, cv2.COLOR_BGR2RGB)
                          ).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0),
        ("[-1,1] input ",
         (torch.from_numpy(cv2.cvtColor(bic, cv2.COLOR_BGR2RGB)
                           ).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
          - 0.5) / 0.5),
    ]:
        pose = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            out = model(t, pose)
        if isinstance(out, (tuple,list)): out = out[0]
        v = out.cpu().numpy()
        print(f"  Input {name}: out.min={v.min():.4f}  out.max={v.max():.4f}  "
              f"out.mean={v.mean():.4f}")

        # Save both renderings
        for render_name, render in [
            ("clamp_0_1",  (v.squeeze().transpose(1,2,0).clip(0,1)*255).astype(np.uint8)),
            ("denorm_-1_1", ((v.squeeze().transpose(1,2,0)*0.5+0.5).clip(0,1)*255).astype(np.uint8)),
        ]:
            bgr = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            fname = f"results/figures/palf_debug_{name.strip()}_{render_name}.png"
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(fname, bgr)
            print(f"    Saved: {fname}")


if __name__ == "__main__":
    main()