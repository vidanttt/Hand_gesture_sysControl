# MY Hand Gesture System Control

A Python-based desktop application for controlling system volume and monitor brightness using hand gestures, inspired by sci-fi HUDs (Heads-Up Displays).

## Features

- **Real-time Hand Tracking:** Uses Mediapipe and OpenCV to detect and track left and right hands via webcam.
- **Volume Control:** Adjust system volume with gestures from your left hand. Detection is made more robust by identifying a black ring on the ring finger.
- **Brightness Control:** Change monitor brightness with right-hand gestures using DDC/CI monitor control.
- **Sci-fi HUD UI:** Modern, animated interface built with PyQt5, featuring a futuristic HUD overlay and live feedback for both volume and brightness.
- **Visual Feedback:** Neon overlays, animated meters, and status panels show hand detection and system changes.
- **Customizable:** Easily modify gesture logic, UI theme, and detection parameters.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/vidanttt/Hand_gesture_sysControl.git
    cd Hand_gesture_sysControl
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Typical requirements include:
    - opencv-python
    - mediapipe
    - numpy
    - pycaw
    - monitorcontrol
    - PyQt5

3. **(Optional) Download the HUD template image:**
    Place your HUD image in `Iron Man HUD Template/EditingCorp Iron man HUD.png` or update the path in `script.py`.

## Usage

Run the main application:
```bash
python script.py
```
- Make sure your webcam is connected.
- Wear a black ring on your left ring finger for robust volume gesture detection.
- Use your left hand for volume and right hand for brightness.
- Both hands detected = both controls online.

## Folder Structure

- `script.py` – Main application logic, hand tracking, volume/brightness control, HUD rendering.
- `ui.py` – Separate UI testing and prototyping.
- `Iron Man HUD Template/` – Contains HUD background images.

## How It Works

- Mediapipe tracks hand landmarks and differentiates left/right hands.
- Volume and brightness are mapped to the distance between thumb and index finger.
- Robust hand detection (volume) via ring color segmentation.
- PyQt5 UI overlays the video feed with animated status panels and meters.

## Acknowledgements

- [Mediapipe](https://github.com/google/mediapipe)
- [OpenCV](https://opencv.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [pycaw](https://github.com/AndreMiras/pycaw)
- [monitorcontrol](https://github.com/newAM/monitorcontrol)

---

Feel free to customize further or add more sections (contributing, troubleshooting, etc.) as needed!
