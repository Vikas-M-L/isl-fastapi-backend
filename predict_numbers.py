import base64
import copy
import itertools
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow import keras


# Load the combined model once at import time
# Expected layout: first 9 outputs correspond to digits '1'..'9'.
_MODEL = keras.models.load_model("models/model.h5")
_MP_HANDS = mp.solutions.hands
_NUMBER_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']


def _calc_landmarks(img: np.ndarray, lms) -> list[list[int]]:
    h, w = img.shape[0], img.shape[1]
    pts: list[list[int]] = []
    for p in lms.landmark:
        x = min(int(p.x * w), w - 1)
        y = min(int(p.y * h), h - 1)
        pts.append([x, y])
    return pts


def _preprocess(pts: list[list[int]]) -> list[float]:
    pts = copy.deepcopy(pts)
    bx, by = pts[0]
    for i in range(len(pts)):
        pts[i][0] -= bx
        pts[i][1] -= by
    flat = list(itertools.chain.from_iterable(pts))
    m = max(map(abs, flat)) or 1.0
    return [float(v) / m for v in flat]


def _decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    
    if not b64 or len(b64) < 10:
        return None
    
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return None
    
    if len(img_bytes) == 0:
        return None
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def predict_number_from_base64(data_url: str) -> Dict[str, Any]:
    """Predict a digit (1..9) from a base64-encoded frame.

    Returns a dict with keys: success, sign, confidence, message (optional).
    """
    # Validate input
    if not data_url or not isinstance(data_url, str):
        return {
            "success": False,
            "message": "Missing or invalid image field",
            "sign": None,
            "confidence": 0.0,
        }
    if "," not in data_url:
        return {
            "success": False,
            "message": "Invalid data URL format (expected 'data:image/...;base64,...')",
            "sign": None,
            "confidence": 0.0,
        }
    
    try:
        img = _decode_data_url(data_url)
        if img is None:
            return {
                "success": False,
                "message": "Invalid image data (imdecode failed)",
                "sign": None,
                "confidence": 0.0,
            }

        with _MP_HANDS.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                return {
                    "success": False,
                    "message": "No hand detected",
                    "sign": None,
                    "confidence": 0.0,
                }

            pts = _calc_landmarks(img, res.multi_hand_landmarks[0])
            feats = _preprocess(pts)
            df = pd.DataFrame(feats).T
            pred = _MODEL.predict(df, verbose=0)[0]

            num_scores = pred[:9]
            if num_scores.size == 0:
                return {
                    "success": False,
                    "message": "Model output too short for numbers",
                    "sign": None,
                    "confidence": 0.0,
                }
            ni = int(np.argmax(num_scores))
            return {
                "success": True,
                "sign": _NUMBER_LABELS[ni],
                "confidence": float(num_scores[ni]),
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {e}",
            "sign": None,
            "confidence": 0.0,
        }
