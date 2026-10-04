"""Step 3 - Live face recognition that marks attendance in a daily CSV file.

Usage:  python attendance.py      (press 'q' to quit)
"""
import csv
import os
from datetime import date, datetime

import cv2

from common import (ATTENDANCE_DIR, CONFIDENCE_THRESHOLD, CONFIRM_FRAMES,
                    MODEL_PATH, detect_faces, load_students, preprocess)


def today_file():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.today().isoformat()}.csv")


def already_marked(path):
    """IDs already present in today's file."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def mark(path, sid, name):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "name", "date", "time"])
        now = datetime.now()
        writer.writerow([sid, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No trained model. Run register.py and train.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    students = load_students()

    path = today_file()
    marked = already_marked(path)
    streak = {}                                   # id -> consecutive frames seen

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam.")
    print("Attendance started. Press 'q' to quit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen = set()

        for box in detect_faces(gray):
            x, y, w, h = box
            label, conf = recognizer.predict(preprocess(gray, box))
            sid = str(label)

            if conf < CONFIDENCE_THRESHOLD and sid in students:
                name = students[sid]
                seen.add(sid)
                streak[sid] = streak.get(sid, 0) + 1

                if sid in marked:
                    text, color = f"{name} - Present", (0, 200, 0)
                else:
                    text, color = f"{name} ({streak[sid]}/{CONFIRM_FRAMES})", (0, 255, 255)
                    if streak[sid] >= CONFIRM_FRAMES:
                        mark(path, sid, name)
                        marked.add(sid)
                        print(f"Marked present: {name}")
            else:
                text, color = "Unknown", (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        for sid in streak:                        # reset people not seen this frame
            if sid not in seen:
                streak[sid] = 0

        cv2.putText(frame, f"Present today: {len(marked)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Face Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Attendance saved in {path}")


if __name__ == "__main__":
    main()
