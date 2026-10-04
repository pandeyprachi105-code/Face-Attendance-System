# Face Attendance System

A Python project that marks attendance automatically by recognising faces through a webcam.
Built with **OpenCV** (Haar Cascade face detection + LBPH face recognition).

## Abstract
Manual attendance is slow and allows proxy attendance. This system detects a person's face
in real time, identifies them using a trained model, and records their name and time in a
CSV file — once per day per person.

## Features
- Register new people from the webcam (40 face samples each)
- Train the recogniser with one command
- Real-time recognition with a confirmation check (5 consecutive frames) to avoid false marks
- "Unknown" faces are rejected using a confidence threshold
- Attendance saved as `attendance/attendance_YYYY-MM-DD.csv`, no duplicates

## Technologies
Python 3.8+, OpenCV (opencv-contrib-python), NumPy, CSV

## Project Structure
```
face_attendance_system/
├── common.py         # settings + helper functions
├── register.py       # capture face samples
├── train.py          # train LBPH model
├── attendance.py     # live recognition + mark attendance
├── requirements.txt
├── dataset/          # (auto) face images, one folder per ID
├── model/            # (auto) trainer.yml
├── attendance/       # (auto) daily CSV files
└── students.csv      # (auto) ID -> name list
```

## How to Run
```bash
pip install -r requirements.txt

python register.py --id 1 --name "Rahul Sharma"   # repeat for each person
python register.py --id 2 --name "Priya Singh"
python train.py                                    # run again after adding people
python attendance.py                               # press q to quit
```

## How It Works
1. **Detection** – Haar Cascade finds face boxes in each webcam frame.
2. **Pre-processing** – face is cropped, resized to 200x200, histogram-equalised.
3. **Recognition** – LBPH compares the face texture pattern to the trained model and gives
   a label + confidence (lower = better match).
4. **Marking** – if confidence < 65 for 5 consecutive frames, attendance is written to CSV.

## Tuning (in common.py)
- `CONFIDENCE_THRESHOLD` – lower = stricter, higher = more lenient
- `CONFIRM_FRAMES` – how many frames to confirm before marking

## Limitations
- LBPH is sensitive to very different lighting and side-on faces
- A printed photo could fool it (no liveness / anti-spoofing check)

## Future Scope
- Deep-learning recognition (`face_recognition`, ArcFace, or OpenCV's SFace model)
- Blink / liveness detection to stop photo spoofing
- Web dashboard (Flask/Django) and database (SQLite/MySQL)
- Email/SMS report to teachers or parents
