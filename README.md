# Hand Gesture Recognition A-Z

This repository contains a collection of Python scripts for **real-time hand gesture recognition** using **OpenCV** and **MediaPipe**. Each script is designed to detect and classify a specific hand gesture (A-Z, Savasana, etc.) via webcam, providing instant visual feedback and gesture accuracy.

---

## Features

- 🖐️ **Real-time hand detection and tracking** using your webcam  
- 🔤 **Gesture classification** for multiple gestures (A-Z, Savasana, etc.)  
- 📊 **Visual feedback**: gesture name and accuracy displayed on video feed  
- 🛠️ **Easily extensible**: add or modify gesture logic in individual scripts

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd "a to z/a to z"
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # Windows (Command Prompt)
   venv\Scripts\activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```sh
   pip install opencv-python mediapipe
   ```

---

## Usage

1. **Run a gesture recognition script:**
   ```sh
   python A_.py
   ```
   Replace `A_.py` with any other script (e.g., `B.py`, `C.py`, `savasana.py`) to detect different gestures.

2. **Instructions:**
   - The webcam window will open and start detecting hand gestures.
   - The detected gesture and accuracy will be displayed on the video feed.
   - Press `q` to exit the application.

---

## Project Structure

- `A_.py`, `B.py`, `C.py`, ... : Scripts for recognizing individual gestures
- `savasana.py` : Script for Savasana pose recognition
- `README.md` : Project documentation
- `run.txt` : (Optional) Run instructions or notes
- `venv/` : (Optional) Python virtual environment

---

## Customization

- To add or modify gestures, edit the corresponding script or create a new one following the existing structure.
- Each script uses MediaPipe hand landmarks and custom logic for gesture classification.

---

## License

This project is for educational and research purposes.

---

**Feel free to fork, modify, and contribute!**

