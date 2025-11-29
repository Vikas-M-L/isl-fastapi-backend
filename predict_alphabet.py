import base64
import copy
import itertools
import string
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow import keras

# Load the combined model once at import time
_MODEL = keras.models.load_model("models/model.h5")
_MP_HANDS = mp.solutions.hands
_ALPHABET = list(string.ascii_uppercase)

# Define which ISL letters require TWO HANDS
# Based on standard ISL alphabet conventions
TWO_HAND_LETTERS = {
    'C', 'G', 'H', 'N', 'O', 'P', 'Q', 'R', 'X'
}

# Optional: Define letters that can be done with either one or two hands
FLEXIBLE_LETTERS = {
    'M', 'T', 'W'
}

def _calc_landmarks(img: np.ndarray, lms) -> List[List[int]]:
    """Extract landmark coordinates from MediaPipe hand landmarks"""
    h, w = img.shape[0], img.shape[1]
    pts: List[List[int]] = []
    for p in lms.landmark:
        x = min(int(p.x * w), w - 1)
        y = min(int(p.y * h), h - 1)
        pts.append([x, y])
    return pts

def _preprocess(pts: List[List[int]]) -> List[float]:
    """Normalize landmarks: translation and scale invariant"""
    pts = copy.deepcopy(pts)
    bx, by = pts[0]  # Wrist as origin
    for i in range(len(pts)):
        pts[i][0] -= bx
        pts[i][1] -= by
    flat = list(itertools.chain.from_iterable(pts))
    m = max(map(abs, flat)) or 1.0
    return [float(v) / m for v in flat]

def _decode_data_url(data_url: str) -> Optional[np.ndarray]:
    """Decode a base64 data URL into a BGR image (np.ndarray)"""
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

def predict_alphabet_from_base64(data_url: str) -> Dict[str, Any]:
    """
    Predict an alphabet letter from a base64-encoded frame.
    Supports BOTH single-hand and two-hand ISL letters.
    
    Returns:
        {
            "success": bool,
            "sign": str or None,
            "confidence": float,
            "hand_count": int,
            "requires_two_hands": bool,
            "message": str (optional)
        }
    """
    # Input validation
    if not data_url or not isinstance(data_url, str):
        return {
            "success": False,
            "message": "Missing or invalid image field",
            "sign": None,
            "confidence": 0.0,
            "hand_count": 0,
            "requires_two_hands": False
        }
    
    if "," not in data_url:
        return {
            "success": False,
            "message": "Invalid data URL format",
            "sign": None,
            "confidence": 0.0,
            "hand_count": 0,
            "requires_two_hands": False
        }
    
    try:
        img = _decode_data_url(data_url)
        if img is None:
            return {
                "success": False,
                "message": "Invalid image data",
                "sign": None,
                "confidence": 0.0,
                "hand_count": 0,
                "requires_two_hands": False
            }

        # Use MediaPipe to detect up to 2 hands
        with _MP_HANDS.Hands(
            model_complexity=0,
            max_num_hands=2,  # ← DETECT UP TO 2 HANDS
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
                    "hand_count": 0,
                    "requires_two_hands": False
                }
            
            hand_count = len(res.multi_hand_landmarks)
            
            # CASE 1: ONE HAND DETECTED
            if hand_count == 1:
                pts = _calc_landmarks(img, res.multi_hand_landmarks[0])
                feats = _preprocess(pts)
                df = pd.DataFrame(feats).T
                pred = _MODEL.predict(df, verbose=0)[0]
                
                letter_scores = pred[9:]  # Skip digits
                if letter_scores.size == 0:
                    return {
                        "success": False,
                        "message": "Model output too short",
                        "sign": None,
                        "confidence": 0.0,
                        "hand_count": 1,
                        "requires_two_hands": False
                    }
                
                li = int(np.argmax(letter_scores))
                predicted_letter = _ALPHABET[li]
                confidence = float(letter_scores[li])
                
                # Check if this letter REQUIRES two hands
                requires_two = predicted_letter in TWO_HAND_LETTERS
                
                return {
                    "success": True,
                    "sign": predicted_letter,
                    "confidence": confidence,
                    "hand_count": 1,
                    "requires_two_hands": requires_two,
                    "warning": f"Letter '{predicted_letter}' typically requires TWO hands in ISL" if requires_two else None
                }
            
            # CASE 2: TWO HANDS DETECTED
            elif hand_count == 2:
                left_hand = None
                right_hand = None
                
                # Identify left and right hands
                for idx, hand_lms in enumerate(res.multi_hand_landmarks):
                    hand_label = res.multi_handedness[idx].classification[0].label
                    pts = _calc_landmarks(img, hand_lms)
                    feats = _preprocess(pts)
                    
                    if hand_label == "Left":
                        left_hand = feats
                    else:
                        right_hand = feats
                
                # Strategy 1: Use dominant (right) hand for prediction
                # (Your model is trained on single-hand data)
                if right_hand:
                    df = pd.DataFrame(right_hand).T
                    pred = _MODEL.predict(df, verbose=0)[0]
                    letter_scores = pred[9:]
                    li = int(np.argmax(letter_scores))
                    predicted_letter = _ALPHABET[li]
                    confidence = float(letter_scores[li])
                else:
                    # Fallback to left hand
                    df = pd.DataFrame(left_hand).T
                    pred = _MODEL.predict(df, verbose=0)[0]
                    letter_scores = pred[9:]
                    li = int(np.argmax(letter_scores))
                    predicted_letter = _ALPHABET[li]
                    confidence = float(letter_scores[li])
                
                # Check if this is a two-hand letter
                is_two_hand_letter = predicted_letter in TWO_HAND_LETTERS
                
                return {
                    "success": True,
                    "sign": predicted_letter,
                    "confidence": confidence,
                    "hand_count": 2,
                    "requires_two_hands": is_two_hand_letter,
                    "note": f"Two-handed sign detected. Letter '{predicted_letter}' {'requires' if is_two_hand_letter else 'can use'} two hands."
                }
            
            # CASE 3: More than 2 hands (error)
            else:
                return {
                    "success": False,
                    "message": f"Unexpected hand count: {hand_count}",
                    "sign": None,
                    "confidence": 0.0,
                    "hand_count": hand_count,
                    "requires_two_hands": False
                }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "sign": None,
            "confidence": 0.0,
            "hand_count": 0,
            "requires_two_hands": False
        }

# ========================================
# BONUS: Utility function for batch processing
# ========================================

def predict_batch(data_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Predict multiple frames in sequence.
    Useful for processing video frames efficiently.
    """
    results = []
    for data_url in data_urls:
        result = predict_alphabet_from_base64(data_url)
        results.append(result)
    return results

# ========================================
# BONUS: Get info about a specific letter
# ========================================

def get_letter_info(letter: str) -> Dict[str, Any]:
    """
    Get information about ISL letter requirements.
    """
    letter = letter.upper()
    if letter not in _ALPHABET:
        return {
            "valid": False,
            "message": f"'{letter}' is not a valid alphabet letter"
        }
    
    return {
        "valid": True,
        "letter": letter,
        "requires_two_hands": letter in TWO_HAND_LETTERS,
        "flexible": letter in FLEXIBLE_LETTERS,
        "description": f"ISL letter '{letter}' {'requires TWO hands' if letter in TWO_HAND_LETTERS else 'uses ONE hand'}"
    }
