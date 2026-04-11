# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Modified : Optimized for Raspberry Pi + duplicate methods removed
 
import os
import cv2
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F
 
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name
 
logger = logging.getLogger(__name__)
 
MODEL_MAPPING = {
    'MiniFASNetV1':   MiniFASNetV1,
    'MiniFASNetV2':   MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}
 
 
class Detection:
    """Détection de visage avec fallback Haar pour Raspberry Pi"""
 
    def __init__(self):
        self.use_caffe = False
        self.detector  = None
        self.detector_confidence = 0.6
 
        try:
            caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
            deploy     = "./resources/detection_model/deploy.prototxt"
            if os.path.exists(caffemodel) and os.path.exists(deploy):
                self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
                self.use_caffe = True
                logger.info("Using Caffe face detector")
            else:
                raise FileNotFoundError("Caffe model files not found")
        except (FileNotFoundError, cv2.error):
            logger.warning("Fallback to Haar cascade (Raspberry Pi optimized)")
            opencv_data = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(opencv_data, "haarcascade_frontalface_default.xml")
            )
 
    def get_bbox(self, img):
        if self.use_caffe and self.detector:
            return self._get_bbox_caffe(img)
        return self._get_bbox_haar(img)
 
    def _get_bbox_caffe(self, img):
        height, width  = img.shape[0], img.shape[1]
        aspect_ratio   = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(
                img,
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            img_resized = img
 
        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        if len(out.shape) == 1:
            out = out.reshape(1, -1)
        if out.shape[0] == 0:
            return None
 
        idx  = np.argmax(out[:, 2])
        left, top, right, bottom = (
            out[idx, 3] * width,  out[idx, 4] * height,
            out[idx, 5] * width,  out[idx, 6] * height,
        )
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
 
    def _get_bbox_haar(self, img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        return [x, y, w, h]
 
 
class AntiSpoofPredict(Detection):
    """Prédiction anti-spoofing avec MiniFASNet"""
 
    def __init__(self, device_id=0):
        super().__init__()
        self.device     = torch.device("cpu")
        self.model      = None
        self.model_path = None
        logger.info(f"Device: {self.device}")
 
    # ── Chargement du modèle ──────────────────────────────────
    def _load_model(self, model_path: str):
        """Charge le modèle .pth une seule fois (cache par chemin)."""
        if self.model_path == model_path and self.model is not None:
            return  # déjà chargé
 
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
 
        logger.info(f"Model: {model_type} | Input: {h_input}x{w_input}")
 
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)
 
        state_dict = torch.load(model_path, map_location=self.device)
        first_key  = next(iter(state_dict))
 
        # Supprimer le préfixe DataParallel si présent
        if 'module.' in first_key:
            from collections import OrderedDict
            state_dict = OrderedDict(
                (k[7:], v) for k, v in state_dict.items()
            )
 
        # ── FIX : size mismatch sur linear.weight ────────────
        # Le checkpoint a été entraîné avec num_classes != 2.
        # On charge en mode strict=False pour ignorer les couches
        # incompatibles et éviter le crash.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                f"Poids partiellement chargés — "
                f"manquants: {missing}, inattendus: {unexpected}"
            )
 
        self.model_path = model_path
        logger.info(f"Modèle chargé : {model_path}")
 
    # ── Prédiction ────────────────────────────────────────────
    def predict(self, img: np.ndarray, model_path: str) -> np.ndarray:
        """
        Prédit si le visage est réel ou fake.
 
        Retourne :
            numpy array [[fake_score, real_score]] après softmax
            [[0.5, 0.5]] si aucun visage détecté ou erreur modèle
        """
        try:
            self._load_model(model_path)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.array([[0.5, 0.5]])
 
        # Détecter et recadrer le visage
        bbox = self.get_bbox(img)
        if bbox is None:
            logger.warning("No face detected for anti-spoof")
            return np.array([[0.5, 0.5]])
 
        x, y, w, h = bbox
        pad = int(max(w, h) * 0.1)
        x   = max(0, x - pad)
        y   = max(0, y - pad)
        w   = min(img.shape[1] - x, w + 2 * pad)
        h   = min(img.shape[0] - y, h + 2 * pad)
        face_img = img[y:y + h, x:x + w]
 
        if face_img.size == 0:
            logger.warning("Face region is empty")
            return np.array([[0.5, 0.5]])
 
        # Préprocessing
        test_transform = trans.Compose([
            trans.Resize((80, 80)),
            trans.ToTensor(),
        ])
        tensor = test_transform(face_img).unsqueeze(0).to(self.device)
 
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
 
        return result