import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import model
from src.models.adaface_sr import AdaFaceSR

def create_gate_heatmaps(checkpoint_path, scface_gallery_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading AdaFace-SR from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_config = {
        "num_in_ch": config.get("num_in_ch", 3),
        "num_out_ch": config.get("num_out_ch", 3),
        "num_feat": config.get("num_feat", 64),
        "num_block": config.get("num_block", 23),
        "num_grow_ch": config.get("num_grow_ch", 32),
        "gate_channels": config.get("gate_channels", 32),
    }
    model = AdaFaceSR(model_config)
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device).eval()
    
    # 2. Get a few sample gallery images
    exts = ('.jpg', '.png', '.jpeg')
    files = [f for f in os.listdir(scface_gallery_dir) if f.lower().endswith(exts)]
    sample_files = sorted(files)[:3]
    
    resolutions = [16, 32, 112] # 16px, 32px, and Native
    
    for filename in sample_files:
        path = os.path.join(scface_gallery_dir, filename)
        img_hr = cv2.imread(path)
        if img_hr is None:
            continue
            
        fig, axes = plt.subplots(len(resolutions), 4, figsize=(16, 4 * len(resolutions)))
        
        for idx, res in enumerate(resolutions):
            # Formulate LR input
            if res < 112:
                lr_bgr = cv2.resize(img_hr, (res, res), interpolation=cv2.INTER_AREA)
            else:
                lr_bgr = cv2.resize(img_hr, (112, 112), interpolation=cv2.INTER_AREA)
                
            # Resize up to target (112)
            lr_112 = cv2.resize(lr_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            # Prepare tensor
            lr_rgb = cv2.cvtColor(lr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            lr_t = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            lr_size_t = torch.tensor([float(res)], device=device)
            
            # Forward pass
            with torch.no_grad():
                sr_t, alpha_t = model(lr_t, lr_size_t)
                
            sr_t = sr_t.clamp(0, 1)
            sr_rgb_out = sr_t[0].permute(1, 2, 0).cpu().numpy()
            
            # Extract Alpha Map
            alpha_map = alpha_t[0, 0].cpu().numpy()
            
            # Compute Nuance Stats
            mean_a = np.mean(alpha_map)
            max_a = np.max(alpha_map)
            p99_a = np.percentile(alpha_map, 99)
            
            # Plot 1: Original LR resized to 112
            axes[idx, 0].imshow(lr_rgb)
            axes[idx, 0].set_title(f"Input ({res}px)", fontsize=12)
            axes[idx, 0].axis("off")
            
            # Plot 2: SR Output
            axes[idx, 1].imshow(sr_rgb_out)
            axes[idx, 1].set_title("AdaFace-SR Output", fontsize=12)
            axes[idx, 1].axis("off")
            
            # Plot 3: Alpha Heatmap
            im = axes[idx, 2].imshow(alpha_map, cmap='jet', vmin=0, vmax=1)
            axes[idx, 2].set_title("Gate Activation (Heatmap)", fontsize=12)
            axes[idx, 2].axis("off")
            fig.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)
            
            # Plot 4: Statistics details
            axes[idx, 3].axis('off')
            stats_text = (
                f"Resolution: {res}px\n\n"
                f"Mean Gate α: {mean_a:.4f}\n"
                f"Max Gate α: {max_a:.4f}\n"
                f"99th %ile α: {p99_a:.4f}\n\n"
                f"Status: " + ("CLOSED" if p99_a < 0.05 else "ACTIVE")
            )
            axes[idx, 3].text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        out_name = os.path.basename(filename).replace('.', '_gate.')
        out_filepath = os.path.join(out_dir, f"{out_name}.png")
        plt.tight_layout()
        plt.savefig(out_filepath, dpi=200)
        plt.close()
        print(f"✅ Saved detailed heatmap analysis to {out_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AdaFace-SR best checkpoint")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Directory with high-res gallery images")
    parser.add_argument("--out_dir", type=str, default="results/evaluation/gate_heatmaps")
    args = parser.parse_args()
    
    create_gate_heatmaps(args.checkpoint, args.gallery_dir, args.out_dir)