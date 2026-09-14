"""
ISL Interpreter - Setup Script
==============================
Downloads required model files and installs dependencies.
Run this once after cloning the repository.
"""

import os, sys, subprocess, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, 'Code')

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(CODE_DIR, "hand_landmarker.task")

REQUIREMENTS = [
    "flask",
    "opencv-python",
    "numpy",
    "mediapipe",
    "scikit-learn",
    "pyttsx3",
]


def install_packages():
    print("Installing Python packages...")
    for pkg in REQUIREMENTS:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            print(f"  ✓ {pkg}")
        except subprocess.CalledProcessError:
            print(f"  ✗ {pkg} (FAILED)")


def download_model():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already exists ({size_mb:.1f} MB): {MODEL_PATH}")
        return

    print(f"Downloading hand_landmarker.task ({7.8:.1f} MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  Manually download from: {MODEL_URL}")
        print(f"  Save to: {MODEL_PATH}")


def main():
    print("=" * 50)
    print("  ISL Sign Language Interpreter - Setup")
    print("=" * 50)

    install_packages()
    print()
    download_model()

    print()
    print("=" * 50)
    print("  Setup complete!")
    print()
    print("  To start the interpreter:")
    print("    python Code/app.py")
    print()
    print("  To collect training data:")
    print("    python Code/collect_data.py")
    print()
    print("  To train custom classifier:")
    print("    python Code/train_classifier.py")
    print("=" * 50)


if __name__ == '__main__':
    main()
