# Pose-Face-Avatar-Windows-
Real-time human pose detection and face animation
## Requirements

OS: Windows 10/11

Python: 3.10/3.11 (recommended). 3.13 not supported by PyTorch/Ultralytics as of now.

Hardware: CPU works. NVIDIA GPU optional.

## Setup (Windows, CMD)
Create and activate venv
```cmd
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat
```
Upgrade pip
```cmd
python -m pip install --upgrade pip
```
Install dependencies (CPU)
```cmd
pip install ultralytics opencv-python mediapipe
```

## Run
Realtime pose overlay
```cmd
python pose_realtime.py --source 0
```
Drive a simple cartoon body
```cmd
python pose_avatar.py --source 0
```
Cartoon face (offline, no external model)
```cmd
python face_avatar_nomodel.py --source 0 --mirror
```
Cartoon face with BlendShapes (needs model download)
```cmd
python face_avatar.py --source 0 --mirror
```
