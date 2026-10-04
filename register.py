"""Step 1 - Register a person by capturing face samples from the webcam.

Usage:  python register.py --id 1 --name "Rahul Sharma"
"""
import argparse
import os
import time

import cv2

from common import (DATASET_DIR, detect_faces, load_students, preprocess,
                    save_students)

SAMPLES = 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Unique numeric ID")
    parser.add_argument("--name", required=True, help="Person's name")
    args = parser.parse_args()

    folder = os.path.join(DATASET_DIR, str(args.id))
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam.")

    count = 0
    print("Look at the camera. Move your head slightly. Press 'q' to stop.")
    while count < SAMPLES:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        if len(faces) == 1:                      # save only when exactly one face
            box = faces[0]
            count += 1
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"),
                        preprocess(gray, box))
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            time.sleep(0.15)                     # small gap for varied samples

        cv2.putText(frame, f"Samples: {count}/{SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count:
        students = load_students()
        students[str(args.id)] = args.name
        save_students(students)
        print(f"Saved {count} samples for {args.name} (ID {args.id}).")
        print("Now run: python train.py")
    else:
        print("No face captured. Try again with better light.")


if __name__ == "__main__":
    main()
