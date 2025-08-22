import streamlit as st
import chromadb
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras_facenet import FaceNet
import cv2
import numpy as np
from utils import ensure_models_downloaded

# Initialize models
detector_path = ensure_models_downloaded()
base_options = python.BaseOptions(model_asset_path=detector_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
embedder = FaceNet()

chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_collection(name='face_database')
st.set_page_config(layout='wide')
st.title("Face Data Vectorstore")

# File uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    image_np = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect faces
    detection_result = detector.detect(mp_image)
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            
            # Crop the face from the original BGR image for FaceNet
            face_img = image_np[bbox.origin_y:bbox.origin_y+bbox.height, bbox.origin_x:bbox.origin_x+bbox.width]

            if face_img.size > 0:
                # Get embeddings using FaceNet
                embeddings = embedder.embeddings([face_img])
                if len(embeddings) > 0:
                    face_encoding = embeddings[0].tolist()

                    # Query ChromaDB
                    response = collection.query(query_embeddings=[face_encoding], n_results=2)
                    st.write("Query Result:")
                    st.write(response)
                else:
                    st.error("Could not generate embedding for the detected face.")
            else:
                st.error("Detected face bounding box is empty.")
    else:
        st.error("No faces found in the image.")