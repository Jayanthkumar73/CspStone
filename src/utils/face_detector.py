"""Face detection wrapper using InsightFace RetinaFace."""

import cv2, numpy as np


class FaceDetector:
    def __init__(self, pretrained_dir=None):
        self.app = None
        try:
            from insightface.app import FaceAnalysis
            root = pretrained_dir + "/recognition" if pretrained_dir else None
            kwargs = {"root": root} if root else {}
            self.app = FaceAnalysis(name="buffalo_l", **kwargs,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            pass

    def detect_and_crop(self, bgr_img, size=112):
        if self.app is None:
            return None
        faces = self.app.get(bgr_img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1+x2)//2, (y1+y2)//2
        s = int(max(w, h) * 1.3)
        x1 = max(0, cx - s//2)
        y1 = max(0, cy - s//2)
        x2 = min(bgr_img.shape[1], x1+s)
        y2 = min(bgr_img.shape[0], y1+s)
        crop = bgr_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size))

    def get_embedding(self, bgr_img):
        """Get 512-d face embedding from InsightFace."""
        if self.app is None:
            return None
        faces = self.app.get(bgr_img)
        if not faces:
            return None
        return faces[0].embedding
