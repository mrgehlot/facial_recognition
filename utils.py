import requests
import os

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def download_file(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_path} successfully.")
    else:
        print(f"{file_path} already exists.")

def ensure_models_downloaded():
    detector_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    detector_path = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
    download_file(detector_url, detector_path)
    return detector_path
