"""Pose estimation wrapper for PALF-Net (used by multiple scripts)."""

import os, cv2, torch, numpy as np


class PoseEstimator:
    """6DRepNet wrapper — predicts (yaw, pitch, roll) from a BGR face image."""

    def __init__(self, pretrained_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()
            self._mode = "package"
        except Exception:
            if pretrained_dir:
                self._manual_load(pretrained_dir)

    def _manual_load(self, pretrained_dir):
        w = os.path.join(pretrained_dir, "pose", "6DRepNet_300W_LP_AFLW2000.pth")
        if not os.path.isfile(w):
            return
        try:
            from sixdrepnet.model import SixDRepNet as M
            self.model = M(backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False)
            sd = torch.load(w, map_location=self.device)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            self.model.load_state_dict(sd, strict=False)
            self.model.to(self.device).eval()
            self._mode = "manual"
        except Exception:
            self.model = None

    def predict(self, bgr_img):
        if self.model is None:
            return None, None, None
        try:
            if self._mode == "package":
                y, p, r = self.model.predict(bgr_img)
                if isinstance(y, (list, np.ndarray)):
                    y, p, r = float(y[0]), float(p[0]), float(r[0])
                return float(y), float(p), float(r)
            else:
                rgb = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (224, 224))
                t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    out = self.model(t.to(self.device))
                e = out.cpu().numpy().flatten()
                return float(e[0]), float(e[1]), float(e[2])
        except:
            return None, None, None
