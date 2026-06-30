# Deepfake Detection and Reconstruction

A computer vision project that detects whether a face image is real or AI-generated (deepfake), built end-to-end: data preprocessing, a fine-tuned classifier, and a deployed live demo.

**Live demo:** _(coming soon — will link to Hugging Face Spaces)_

## Status

🚧 In progress. Currently on: dataset preprocessing.

## Problem

Deepfakes — AI-generated face swaps and synthetic faces — are increasingly hard to distinguish from real photos by eye. This project builds a binary classifier that flags real vs. fake faces, using subtle pixel-level and frequency-domain artifacts that GAN-based generation methods tend to leave behind.

## Approach

1. **Dataset** — subset of [FaceForensics++](https://github.com/ondyari/FaceForensics), real and manipulated face video frames
2. **Preprocessing** — face detection and cropping (OpenCV), resize to 224×224, normalize to [0,1]
3. **Model** — EfficientNet-B0, pretrained on ImageNet, fine-tuned as a binary classifier (real/fake)
4. **Training** — fine-tuned on GPU (Google Colab), evaluated with accuracy, precision, recall, and AUC
5. **Demo** — Gradio web app: upload an image, get a real/fake prediction with confidence score
6. **Deployment** — hosted live on Hugging Face Spaces

## Project structure

## Setup

```bash
git clone https://github.com/mitalipatil104/deepfake-detection-reconstruction.git
cd deepfake-detection-reconstruction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Results

_(to be filled in after training — accuracy, precision/recall, confusion matrix)_

## Tech stack

Python, OpenCV, PyTorch, timm (EfficientNet-B0), Gradio, Hugging Face Spaces

## Author

Mitali — built as a portfolio project to demonstrate end-to-end ML competency, from image fundamentals through a deployed detection pipeline.