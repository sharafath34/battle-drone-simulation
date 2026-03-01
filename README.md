# Battle Drone

Computer-vision drone simulation built with Python, OpenCV, and YOLOv8.  
The project runs on a video feed and overlays tactical information such as detected targets, radar/minimap, threat labels, and mission actions.

## Project Files

- `battle_drone.py`: Base version with person detection, auto-lock, radar, target marking, and medical assist mode.
- `battle_drone_updated.py`: Extended version with simulated soldier vitals, threat levels, QR enemy detection, and surveillance target cycling.
- `test.mov`: Input video stream used by both scripts.
- `yolov8n.pt`: YOLO model weights.
- `requirements.txt`: Python dependencies list (needs cleanup; see install steps below).

## Features

### Base Script (`battle_drone.py`)

- Real-time person detection using YOLOv8.
- Auto-lock to closest detected target.
- On-screen enemy boxes and labels.
- Mini radar map.
- Medical drone deployment indicator.
- Keyboard controls for mission actions.

### Updated Script (`battle_drone_updated.py`)

- All core detection/radar behavior from the base concept.
- Simulated soldier vitals:
  - Temperature
  - Heart rate
  - Health %
- Dynamic threat level (`LOW`, `MEDIUM`, `HIGH`) with color-coded overlays.
- QR-based enemy detection using `pyzbar`.
- Surveillance mode to cycle highlighted targets.

## Requirements

- Python 3.9+
- Windows/Linux/macOS
- Webcam or video file input (this repo uses `test.mov`)

Install dependencies:

```bash
pip install ultralytics opencv-python numpy torch torchvision pyzbar
```

Note for Windows users:

- `pyzbar` may require the ZBar runtime library installed on your system.

## Run

Run base version:

```bash
python battle_drone.py
```

Run updated version:

```bash
python battle_drone_updated.py
```

## Controls

### `battle_drone.py`

- `A`: Activate medical assistance
- `M`: Mark target
- `ESC`: Quit

### `battle_drone_updated.py`

- `A`: Activate medical drone
- `S`: Switch surveillance target
- `ESC`: Quit

## Configuration

You can tune these constants in the scripts:

- `VIDEO_PATH`: Source video path.
- `CONF`: Detection confidence threshold.
- `MAP_SIZE`: Radar map size.
- `PROC_W`, `PROC_H` (updated script): Processing resolution.

## Notes

- If the video reaches the end, playback restarts automatically.
- `requirements.txt` currently includes package names and raw pip command lines. Consider keeping only package entries for cleaner installs.

## Disclaimer

This repository is a simulation/demo project for computer-vision workflows.  
It is not a real autonomous combat system.
