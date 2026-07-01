"""
preprocess.py

Stage 2 of the pipeline: turns a raw face image into a model-ready input.

Steps:
1. Detect the face in the image (OpenCV Haar Cascade)
2. Crop just the face region
3. Resize to 224x224 (the input size EfficientNet-B0 expects)
4. Convert BGR -> RGB (OpenCV loads images backwards compared to PyTorch)
5. Normalize pixel values from 0-255 to 0.0-1.0

This same function gets reused in two places later:
- training.py, to preprocess every image in the dataset before feeding the model
- app.py, to preprocess whatever image a user uploads to the live Gradio demo
"""

import cv2
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

TARGET_SIZE = (224, 224)


def detect_and_crop_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y:y + h, x:x + w]


def preprocess_image(image_bgr):
    """
    Full pipeline for raw images: detect face, crop, resize, convert, normalize.
    Use this for images where the face is somewhere inside a larger scene.
    Returns None if no face detected.
    """
    face = detect_and_crop_face(image_bgr)
    if face is None:
        return None
    face_resized = cv2.resize(face, TARGET_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    return face_normalized


def preprocess_precropped_face(image_bgr):
    """
    For datasets already cropped to just the face -- skips detection entirely.
    Use this for this project's dataset (1000_videos) which arrives pre-cropped.
    """
    face_resized = cv2.resize(image_bgr, TARGET_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    return face_normalized


if __name__ == "__main__":
    import sys
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpeg"
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load {test_image_path} -- check the file exists.")
        sys.exit(1)
    result = preprocess_precropped_face(image)
    print("Preprocessed successfully.")
    print("Output shape:", result.shape)
    print("Output dtype:", result.dtype)
    print("Min pixel value:", result.min())
    print("Max pixel value:", result.max())