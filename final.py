import streamlit as st
import face_recognition
import cv2
from datetime import datetime
import os

def setup_face_recognition(sample_image_path):
    """
    Load and encode a sample face image for face recognition.
    """
    try:
        sample_image = face_recognition.load_image_file(sample_image_path)
        face_locations = face_recognition.face_locations(sample_image, model="hog")
        if not face_locations:
            raise Exception("No face found in the sample image.")
        face_encodings = face_recognition.face_encodings(sample_image, face_locations)
        if not face_encodings:
            raise Exception("Could not compute face encoding.")
        return face_encodings[0]
    except Exception as e:
        raise Exception(f"Error processing sample image: {e}")

def run_face_recognition(sample_face_encoding):
    """
    Run real-time face recognition using webcam and display results on Streamlit.
    """
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not access the webcam.")
    
    # Streamlit placeholders
    video_placeholder = st.empty()  # For displaying video
    status_placeholder = st.empty()  # For displaying status
    date_placeholder = st.empty()  # For date and time
    accepted_time_placeholder = st.empty()  # For accepted time

    # Variables for frame processing
    process_every_n_frames = 3
    frame_count = 0
    last_accepted_time = None

    # Button to stop the webcam
    stop_button = st.button("Stop Webcam")

    while not stop_button:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture frame from the webcam.")
            break

        # Only process every nth frame for performance
        if frame_count % process_every_n_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            status = "No Face Detected"

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces([sample_face_encoding], face_encoding, tolerance=0.6)
                    if matches[0]:
                        status = "Accepted"
                        last_accepted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        break

            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                color = (0, 255, 0) if status == "Accepted" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Update Streamlit placeholders
            status_placeholder.text(f"Status: {status}")
            date_placeholder.text(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if last_accepted_time:
                accepted_time_placeholder.text(f"Last Accepted Time: {last_accepted_time}")

        # Encode the frame as JPEG for Streamlit
        _, buffer = cv2.imencode('.jpg', frame)
        video_bytes = buffer.tobytes()
        video_placeholder.image(video_bytes, channels="BGR", use_container_width=True)

        frame_count += 1

    # Release webcam after stopping
    video_capture.release()
    st.write("Webcam stopped.")
    if last_accepted_time:
        st.write(f"Final Last Accepted Time: {last_accepted_time}")

def main():
    st.title("Real-Time Face Recognition")
    st.write("This application uses a sample image for real-time face recognition.")

    # Load the sample image
    sample_image_path = "test_img.jpg"
    if not os.path.exists(sample_image_path):
        st.error(f"Sample image not found at '{sample_image_path}'. Please ensure the image is in the working directory.")
        return

    try:
        sample_face_encoding = setup_face_recognition(sample_image_path)
    except Exception as e:
        st.error(str(e))
        return

    st.write("Starting the webcam...")
    try:
        run_face_recognition(sample_face_encoding)
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
