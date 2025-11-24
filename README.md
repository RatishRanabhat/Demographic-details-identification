Demographic Details Identification
==================================

This project provides two ways to analyse faces in video streams:

- `main.py`: a command-line script that runs age and gender detection on a prerecorded video or a live webcam feed.
- `desktop_app`: a Tkinter desktop application that allows a user to upload a video, view annotated frames, and review live statistics in a dashboard.

Both solutions rely on MTCNN for face detection and pre-trained Caffe networks for age and gender classification.

Project Structure
-----------------

- `main.py` — CLI workflow for processing videos or camera streams.
- `desktop_app/` — Tkinter application package (`python -m desktop_app.app`).
- `constants/` — Caffe model definitions (`*.prototxt`) and weights (`*.caffemodel`).
- `requirements.txt` — locked dependency list generated from the project virtual environment.

Prerequisites
-------------

- Python 3.10 or later.
- System packages required by OpenCV (for Ubuntu: `sudo apt install libgl1`).
- A functional webcam (optional) if you want to run live capture.

Setup
-----

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Command-Line Usage
------------------

Process a stored video file:

```bash
source .venv/bin/activate
python main.py --input myvideo.mp4 --output processed.avi
```

Use a webcam by passing `--input 0`. Press `q` in the preview window to stop recording early. If the environment does not support GUI display (Wayland without Qt), append `--no-display` to disable the preview window.

Desktop Application
-------------------

Launch the Tkinter dashboard:

```bash
source .venv/bin/activate
python -m desktop_app.app
```

Workflow:

1. Click **Choose video…** and select a local video file.
2. The application displays annotated frames, including gender and age predictions.
3. The right-hand dashboard tracks:
   - Frames processed
   - Faces detected
   - Gender distribution
   - Detection counts per age group
4. Use **Stop** to cancel processing before the video ends.

The window opens maximised where supported; otherwise, the size can be adjusted manually.

Model Files
-----------

Ensure the following files remain in the `constants/` directory:

- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

Do not rename or relocate these assets unless you update both `main.py` and `desktop_app/detector.py`.

Troubleshooting
---------------

- **Missing modules**: run `pip install -r requirements.txt` inside the virtual environment.
- **Qt platform errors**: install `qtwayland5` or run without display using `--no-display`.
- **No detections**: verify that the input video contains clear faces and adequate lighting.

License
-------

The original model weights are subject to their respective licences. Review them before distributing applications built on top of this project.

