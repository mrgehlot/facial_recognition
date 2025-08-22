import streamlit as st
import streamlit_webrtc as webrtc
import chromadb
import cv2
import av
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras_facenet import FaceNet
from utils import ensure_models_downloaded
import numpy as np

# These models will be loaded once and are globally available.
detector_path = ensure_models_downloaded()
base_options = python.BaseOptions(model_asset_path=detector_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
embedder = FaceNet()

chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_collection(name='face_database')

class VideoProcessor:
    def __init__(self):
        self.frame_counter = 0
        self.recognition_interval = 10  # Process every 10th frame
        self.last_known_faces = []

    def process_frame(self, image_array):
        """
        Detects and recognizes faces in a single frame.
        This is the computationally expensive part.
        """
        # Convert the BGR image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect faces
        detection_result = detector.detect(mp_image)
        
        current_faces = []
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                top = max(0, bbox.origin_y)
                left = max(0, bbox.origin_x)
                bottom = min(image_array.shape[0], bbox.origin_y + bbox.height)
                right = min(image_array.shape[1], bbox.origin_x + bbox.width)

                face_img = image_array[top:bottom, left:right]
                
                if face_img.size > 0:
                    embeddings = embedder.embeddings([face_img])
                    if len(embeddings) > 0:
                        face_encoding = embeddings[0]
                        db_result = collection.query(query_embeddings=[face_encoding.tolist()], n_results=1)
                        
                        if db_result['ids'] and db_result['ids'][0]:
                            filename_stem = db_result['ids'][0][0].split('.')[0]
                            name = filename_stem.rsplit('_', 1)[0]
                            distance = db_result['distances'][0][0]

                            current_faces.append(((left, top, right, bottom), name, distance))

        return current_faces

    def draw_faces(self, image_array, faces):
        """
        Draws bounding boxes and names on the frame.
        """
        for (left, top, right, bottom), name, distance in faces:
            if distance < 1.0:
                cv2.rectangle(image_array, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image_array, f"{name} ({distance:.2f})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(image_array, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image_array, "Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        return image_array

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_array = frame.to_ndarray(format="bgr24")
        
        # Perform expensive recognition only periodically.
        if self.frame_counter % self.recognition_interval == 0:
            start_time = time.time()
            self.last_known_faces = self.process_frame(image_array)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Recognition time: {execution_time:.2f} seconds")
        
        self.frame_counter += 1
        
        # Draw the last known faces on the current frame.
        processed_image = self.draw_faces(image_array, self.last_known_faces)
        
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

st.title("Simple Facial Recognizer")
# st.subheader('Check if you exists in Aubergine Facial Database')

webrtc_streamer = webrtc.webrtc_streamer(
    key="face-recognition", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)