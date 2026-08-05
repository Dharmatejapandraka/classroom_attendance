"""
face_utils.py
Face detection (Haar Cascade) + recognition (LBPH) utilities.
Chosen deliberately over dlib/face_recognition because LBPH ships inside
opencv-contrib-python and needs no system-level compilation, which makes
deployment on Streamlit Community Cloud painless.
"""

import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
TRAINER_PATH = os.path.join(DATA_DIR, "trainer.yml")

FACE_SIZE = (200, 200)
# LBPH: LOWER confidence value = better match. Tune if you get false positives/negatives.
CONFIDENCE_THRESHOLD = 75

_cascade = None


def get_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(cascade_path)
    return _cascade


def pil_to_gray(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def detect_largest_face(gray_img):
    cascade = get_cascade()
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_crop = gray_img[y:y + h, x:x + w]
    face_resized = cv2.resize(face_crop, FACE_SIZE)
    return face_resized, (x, y, w, h)


def save_face_samples(member_id, pil_images):
    """Detects a face in each provided PIL image and saves crops to disk. Returns count saved."""
    member_dir = os.path.join(FACES_DIR, str(member_id))
    os.makedirs(member_dir, exist_ok=True)
    saved = 0
    for pil_img in pil_images:
        gray = pil_to_gray(pil_img)
        result = detect_largest_face(gray)
        if result is None:
            continue
        face_resized, _ = result
        path = os.path.join(member_dir, f"sample_{saved}.jpg")
        cv2.imwrite(path, face_resized)
        saved += 1
    return saved

def find_matching_member(pil_images):
    """Checks the captured photos against the already-trained model.
    Returns the member_id of a match if this face is already registered, else None."""
    if not os.path.exists(TRAINER_PATH):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    for pil_img in pil_images:
        gray = pil_to_gray(pil_img)
        result = detect_largest_face(gray)
        if result is None:
            continue
        face_resized, _ = result
        label, confidence = recognizer.predict(face_resized)
        if confidence < CONFIDENCE_THRESHOLD:
            return label
    return None


def train_model():
    """(Re)trains the LBPH recognizer on all stored face samples."""
    if not os.path.exists(FACES_DIR):
        return False, "No face data available yet."

    faces = []
    labels = []
    for member_id_str in os.listdir(FACES_DIR):
        member_dir = os.path.join(FACES_DIR, member_id_str)
        if not os.path.isdir(member_dir):
            continue
        try:
            member_id = int(member_id_str)
        except ValueError:
            continue
        for filename in os.listdir(member_dir):
            img_path = os.path.join(member_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(member_id)

    if len(faces) == 0:
        # No samples left (e.g. everyone deleted) - remove stale trainer file.
        if os.path.exists(TRAINER_PATH):
            os.remove(TRAINER_PATH)
        return False, "No valid face samples found."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    os.makedirs(DATA_DIR, exist_ok=True)
    recognizer.write(TRAINER_PATH)
    return True, f"Model trained on {len(faces)} sample(s) across {len(set(labels))} member(s)."


def recognize_from_image(pil_image):
    """
    Detects all faces in an image and attempts to recognize each one.
    Returns (results, error_message). results is a list of dicts:
        {member_id, confidence, bbox, matched}
    """
    if not os.path.exists(TRAINER_PATH):
        return None, "No trained model found yet. Please register at least one face first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    gray = pil_to_gray(pil_image)
    cascade = get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return [], None

    results = []
    for (x, y, w, h) in faces:
        face_crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
        label, confidence = recognizer.predict(face_crop)
        results.append({
            "member_id": label,
            "confidence": confidence,
            "bbox": (int(x), int(y), int(w), int(h)),
            "matched": confidence < CONFIDENCE_THRESHOLD,
        })
    return results, None
