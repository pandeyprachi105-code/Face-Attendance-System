"""Shared settings and helper functions used by all scripts."""
import csv
import os

import cv2

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trainer.yml")
STUDENTS_CSV = "students.csv"
ATTENDANCE_DIR = "attendance"

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 65   # LBPH: LOWER = better match. Raise it to be more lenient.
CONFIRM_FRAMES = 5          # face must be recognised in N consecutive frames

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(gray):
    """Return a list of (x, y, w, h) face boxes found in a grayscale image."""
    return _cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                     minSize=(80, 80))


def preprocess(gray, box):
    """Crop the face, resize it and equalise lighting."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


def load_students():
    """Return {id(str): name}."""
    if not os.path.exists(STUDENTS_CSV):
        return {}
    with open(STUDENTS_CSV, newline="") as f:
        return {row["id"]: row["name"] for row in csv.DictReader(f)}


def save_students(students):
    with open(STUDENTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
        for sid, name in sorted(students.items(), key=lambda kv: int(kv[0])):
            writer.writerow([sid, name])
