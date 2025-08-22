import streamlit as st
import streamlit_webrtc as webrtc
import cv2
import os
import time
from av import VideoFrame

# Directory to save the images
if "face_images" not in os.listdir("."):
    os.mkdir("face_images")

st.title("Create Facial Dataset")
st.write("Enter the name of the person and capture an image from your webcam.")

person_name = st.text_input("Enter Person's Name")


class VideoProcessor:
    def __init__(self):
        self.frame_to_capture = None

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # We will be capturing the frame to save it.
        self.frame_to_capture = img.copy()

        return frame


ctx = webrtc.webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if st.button("Capture Image", key="capture_image"):
    if person_name:
        if ctx.video_processor and ctx.video_processor.frame_to_capture is not None:
            image_to_save = ctx.video_processor.frame_to_capture

            # Create a unique filename
            filename = f"face_images/{person_name}_{int(time.time())}.jpg"

            # Save the image
            cv2.imwrite(filename, image_to_save)

            st.success(f"Image saved as {filename}")
            st.image(image_to_save, channels="BGR", caption="Captured Image")
        else:
            st.warning(
                "Could not capture image. Please make sure your webcam is active."
            )
    else:
        st.warning("Please enter a name for the person.")
