import os
import chromadb
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras_facenet import FaceNet
import cv2
from utils import ensure_models_downloaded

client = chromadb.PersistentClient()
try:
    face_database_collection = client.get_collection(name="face_database")
    client.delete_collection(name="face_database")
    face_database_collection = client.create_collection(name="face_database")
except:
    face_database_collection = client.create_collection(name="face_database")


def store_encodings_into_database(image_file, detector, embedder):
    person_id = image_file.split("/")[-1]
    image = mp.Image.create_from_file(image_file)
    detection_result = detector.detect(image)
    
    image_np = cv2.imread(image_file)
    
    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            # Crop the face from the image
            face_img = image_np[bbox.origin_y:bbox.origin_y+bbox.height, bbox.origin_x:bbox.origin_x+bbox.width]
            
            if face_img.size > 0:
                # Get embeddings using FaceNet
                embeddings = embedder.embeddings([face_img])
                if len(embeddings) > 0:
                    face_encodings = embeddings[0].tolist()
                    face_database_collection.add(
                        ids=[person_id],
                        embeddings=[face_encodings],
                        metadatas=[{"person_id": person_id}],
                    )
            else:
                print(f"Warning: Bounding box for {person_id} resulted in an empty image slice.")

if __name__ == "__main__":
    detector_path = ensure_models_downloaded()
    base_options = python.BaseOptions(model_asset_path=detector_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    
    embedder = FaceNet()
    
    data_dir = 'face_images'
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Please create it and add face images.")
    else:
        for image_file in os.listdir(data_dir):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(data_dir, image_file)
                print(f"Processing {image_path}...")
                store_encodings_into_database(image_path, detector, embedder)
            else:
                print(f"Skipping non-image file: {image_file}")
