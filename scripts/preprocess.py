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

# Load OpenCV's built-in pretrained face detector once, at import time,
# so we don't reload it from disk every time we process an image.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

TARGET_SIZE = (224, 224)  # height, width expected by EfficientNet-B0


def detect_and_crop_face(image_bgr):
    """
    Finds the largest face in an image and crops it out.

    Args:
        image_bgr: a NumPy array as returned by cv2.imread() (BGR order)

    Returns:
        A cropped BGR face image, or None if no face was found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns a list of (x, y, width, height) boxes,
    # one per face it thinks it found in the image.
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image size is reduced at each scan
        minNeighbors=5,    # higher = fewer false positives, may miss some faces
        minSize=(60, 60),  # ignore detections smaller than 60x60 pixels
    )

    if len(faces) == 0:
        return None

    # If multiple faces were detected, keep only the largest one
    # (by area) -- this is almost always the main subject of the photo.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face_crop = image_bgr[y:y + h, x:x + w]
    return face_crop


def preprocess_image(image_bgr):
    """
    Full preprocessing pipeline: detect, crop, resize, convert, normalize.

    Args:
        image_bgr: a NumPy array as returned by cv2.imread() (BGR order)

    Returns:
        A (224, 224, 3) float32 NumPy array with values in [0.0, 1.0],
        in RGB order, ready to be converted into a model input tensor.
        Returns None if no face was detected.
    """
    face = detect_and_crop_face(image_bgr)
    if face is None:
        return None

    face_resized = cv2.resize(face, TARGET_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0

    return face_normalized


if __name__ == "__main__":
    # Quick manual test: run this file directly to preprocess image.jpeg
    # and confirm the pipeline works before we plug it into anything else.
    import sys

    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpeg"

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load {test_image_path} -- check the file exists.")
        sys.exit(1)

    result = preprocess_image(image)

    if result is None:
        print("No face detected in the image.")
    else:
        print("Face detected and preprocessed successfully.")
        print("Output shape:", result.shape)
        print("Output dtype:", result.dtype)
        print("Min pixel value:", result.min())
        print("Max pixel value:", result.max())

        # Save a visual copy so you can SEE what got cropped --
        # convert back to 0-255 BGR for cv2.imwrite to save correctly.
        preview = (result * 255).astype(np.uint8)
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        cv2.imwrite("face_crop_preview.jpeg", preview_bgr)
        print("Saved cropped face as face_crop_preview.jpeg -- open it to check.")
