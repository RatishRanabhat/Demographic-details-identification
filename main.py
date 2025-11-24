import argparse
import os
import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np

# ---------------------- Constants ---------------------- #
AGE_PROTO  = "constants/age_deploy.prototxt"
AGE_MODEL  = "constants/age_net.caffemodel"
GENDER_PROTO = "constants/gender_deploy.prototxt"
GENDER_MODEL = "constants/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGELIST  = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERLIST= ['Male', 'Female']

# ---------------------- Argument Parsing ---------------------- #
parser = argparse.ArgumentParser(description='Age and Gender prediction on video using MTCNN + OpenCV DNN')
parser.add_argument('--input', required=True, help='Path to input video or "0" for webcam')
parser.add_argument('--output', help='Path to save output video', default="output_mtcnn.avi")
args = parser.parse_args()

# ---------------------- Load Models ---------------------- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

age_net    = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# ---------------------- Open Video ---------------------- #
if args.input == "0":
    cap = cv2.VideoCapture(0)  # webcam
else:
    if not os.path.exists(args.input):
        raise ValueError(f"Input video not found: {args.input}")
    cap = cv2.VideoCapture(args.input)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

# ---------------------- Process Video Frames ---------------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            padding = 20
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(frame.shape[1]-1, x2 + padding)
            y2p = min(frame.shape[0]-1, y2 + padding)

            face = frame[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False, crop=False)

            # Gender prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDERLIST[gender_preds[0].argmax()]

            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGELIST[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1p, y1p-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Write frame to output video
    out.write(frame)

    # Show frame in real-time
    cv2.imshow("Age/Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to {args.output}")
