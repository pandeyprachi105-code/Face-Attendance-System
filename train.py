"""Step 2 - Train the LBPH face recogniser on the registered images.

Usage:  python train.py
"""
import os

import cv2
import numpy as np

from common import DATASET_DIR, MODEL_DIR, MODEL_PATH


def main():
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit("No dataset found. Run register.py first.")

    images, labels = [], []
    for folder in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, folder)
        if not (os.path.isdir(path) and folder.isdigit()):
            continue
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(int(folder))

    if not images:
        raise SystemExit("Dataset is empty. Run register.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)
    print(f"Trained on {len(images)} images of {len(set(labels))} people.")
    print(f"Model saved to {MODEL_PATH}. Now run: python attendance.py")


if __name__ == "__main__":
    main()
