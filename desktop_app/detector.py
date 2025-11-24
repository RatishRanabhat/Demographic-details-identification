"""Model wrapper for age and gender detection using MTCNN and OpenCV DNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONSTANTS_DIR = _PROJECT_ROOT / "constants"

AGE_PROTO = str(_CONSTANTS_DIR / "age_deploy.prototxt")
AGE_MODEL = str(_CONSTANTS_DIR / "age_net.caffemodel")
GENDER_PROTO = str(_CONSTANTS_DIR / "gender_deploy.prototxt")
GENDER_MODEL = str(_CONSTANTS_DIR / "gender_net.caffemodel")

MODEL_MEAN_VALUES: Tuple[float, float, float] = (
    78.4263377603,
    87.7689143744,
    114.895847746,
)

AGE_LABELS: Tuple[str, ...] = (
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
)

GENDER_LABELS: Tuple[str, ...] = ("Male", "Female")


@dataclass(frozen=True)
class DetectionResult:
    """Structured output for a single face detection."""

    bbox: Tuple[int, int, int, int]
    gender: str
    age: str
    gender_confidence: float
    age_confidence: float


class AgeGenderDetector:
    """Performs face detection along with age and gender estimation."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._mtcnn = MTCNN(keep_all=True, device=self.device)
        self._age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        self._gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Run detection and annotation on a single frame.

        Args:
            frame: BGR frame captured from a video source.

        Returns:
            A tuple with the annotated frame and a list of detection results.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self._mtcnn.detect(img_rgb)
        results: List[DetectionResult] = []

        if boxes is None:
            return frame, results

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            padding = 20
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(frame.shape[1] - 1, x2 + padding)
            y2p = min(frame.shape[0] - 1, y2 + padding)

            face = frame[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(
                face,
                scalefactor=1.0,
                size=(227, 227),
                mean=MODEL_MEAN_VALUES,
                swapRB=False,
                crop=False,
            )

            self._gender_net.setInput(blob)
            gender_preds = self._gender_net.forward()
            gender_idx = int(gender_preds[0].argmax())
            gender_confidence = float(gender_preds[0][gender_idx])
            gender = GENDER_LABELS[gender_idx]

            self._age_net.setInput(blob)
            age_preds = self._age_net.forward()
            age_idx = int(age_preds[0].argmax())
            age_confidence = float(age_preds[0][age_idx])
            age = AGE_LABELS[age_idx]

            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            label = f"{gender} ({gender_confidence:.2f}), {age} ({age_confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x1p, y1p - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            results.append(
                DetectionResult(
                    bbox=(x1p, y1p, x2p, y2p),
                    gender=gender,
                    age=age,
                    gender_confidence=gender_confidence,
                    age_confidence=age_confidence,
                )
            )

        return frame, results

