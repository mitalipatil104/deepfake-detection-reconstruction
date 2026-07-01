"""
batch_preprocess.py

Stage 3 of the pipeline: runs preprocess_image() across an entire folder
of images instead of one at a time, and saves the results as ready-to-use
JPEGs in a parallel processed/ folder.

This turns your raw dataset:
    1000_videos/test/real/*.jpg
    1000_videos/test/fake/*.jpg

into a training-ready dataset:
    dataset/processed/real/*.jpg   (cropped, resized, face-only)
    dataset/processed/fake/*.jpg

We keep the real/fake folder split exactly as-is, since that split IS the
label your model will learn from later (PyTorch's ImageFolder reads the
folder name as the class).

Run it with:
    python scripts/preprocess_dataset.py
"""

import os
import cv2
from preprocess import preprocess_precropped_face  # reuses Day-stage-2 function, unchanged

# Where the raw downloaded dataset lives
RAW_DIR = "1000_videos/test"

# Where we'll save the cropped, resized, ready-for-training versions
OUTPUT_DIR = "dataset/processed"

# Valid image extensions to look for -- skips anything else (e.g. stray .txt files)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def process_folder(label):
    """
    Processes every image in 1000_videos/test/<label>/ and saves the
    cropped, normalized result into dataset/processed/<label>/.

    Args:
        label: either "real" or "fake" -- matches the folder name exactly.

    Returns:
        (success_count, fail_count) -- how many images processed cleanly
        vs. how many had no detectable face.
    """
    input_folder = os.path.join(RAW_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_folder, exist_ok=True)

    filenames = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    success_count = 0
    fail_count = 0
    failed_files = []

    print(f"\nProcessing '{label}': {len(filenames)} images found")

    for i, filename in enumerate(filenames, start=1):
        input_path = os.path.join(input_folder, filename)

        image = cv2.imread(input_path)
        if image is None:
            fail_count += 1
            failed_files.append(filename)
            continue

        result = preprocess_precropped_face(image)

        if result is None:
            # No face detected -- skip this image, log it, move on.
            # This is expected to happen occasionally; it is not a bug.
            fail_count += 1
            failed_files.append(filename)
            continue

        # Convert back from [0,1] float RGB to [0,255] uint8 BGR
        # so cv2.imwrite saves a normal, viewable JPEG.
        output_image = (result * 255).astype("uint8")
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, output_image_bgr)
        success_count += 1

        # Print progress every 50 images so the terminal isn't silent
        # for long stretches during a big batch.
        if i % 50 == 0 or i == len(filenames):
            print(f"  {i}/{len(filenames)} processed...")

    print(f"'{label}' done: {success_count} succeeded, {fail_count} failed (no face detected)")

    if failed_files:
        log_path = os.path.join(OUTPUT_DIR, f"{label}_failed.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(failed_files))
        print(f"  Failed filenames logged to {log_path}")

    return success_count, fail_count


if __name__ == "__main__":
    print("Starting batch preprocessing...")

    real_success, real_fail = process_folder("real")
    fake_success, fake_fail = process_folder("fake")

    print("\n--- Summary ---")
    print(f"Real: {real_success} processed, {real_fail} skipped")
    print(f"Fake: {fake_success} processed, {fake_fail} skipped")
    print(f"Total training-ready images: {real_success + fake_success}")
    print(f"\nOutput saved to: {OUTPUT_DIR}/real/ and {OUTPUT_DIR}/fake/")
