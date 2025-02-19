import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
# Suhas was here

def setup_face_recognition(sample_image_path, name="Unknown", person_id="N/A"):
    """
    Load and process a predefined sample image for face recognition.
    Returns the face encoding of the sample image and associated person info.
    """
    try:
        # Load the sample image
        sample_image = face_recognition.load_image_file(sample_image_path)
        
        # Detect faces in the sample image
        face_locations = face_recognition.face_locations(sample_image, model="hog")
        if not face_locations:
            raise Exception("No face found in the sample image. Please use a valid image.")

        # Get the face encodings for the detected face(s)
        face_encodings = face_recognition.face_encodings(sample_image, face_locations)
        if not face_encodings:
            raise Exception("Could not compute face encoding for the sample image.")

        # Return the encoding and associated person information
        return face_encodings[0], name, person_id  # Return encoding and additional data
    except Exception as e:
        raise Exception(f"Error loading sample image: {str(e)}")

def process_webcam_frame(frame, sample_face_encoding, name, person_id):
    """
    Process a single frame from the webcam and compare it with the sample face encoding.
    """
    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    status = "No Face Detected"
    color = (255, 0, 0)  # Red for no match
    displayed_data = "No data found"  # Default message when no match is found

    if face_locations:
        # Get face encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare the detected face with the sample face
            matches = face_recognition.compare_faces([sample_face_encoding], face_encoding, tolerance=0.6)
            if matches[0]:
                status = "MATCH: Accepted"
                color = (0, 255, 0)  # Green for match
                displayed_data = f"Name: {name}\nID: {person_id}\nStatus: {status}"
            else:
                status = "NO MATCH: Rejected"
                color = (255, 0, 0)  # Red for no match
                displayed_data = "No data found"

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Annotate the status
            cv2.putText(frame, status, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display global status on the frame
    cv2.putText(frame, "Status: " + status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), displayed_data

def main():
    st.title("Web Based Face Detection System")
    st.write("This application performs real-time face recognition using a preloaded sample image and webcam.")

    # Preload the sample image and user info
    sample_image_path = "dron.jpg"
    name = "Dron"  # Set the name of the person associated with the sample face
    person_id = "12345"  # Set the ID associated with the person

    if not os.path.exists(sample_image_path):
        st.error(f"Sample image not found at '{sample_image_path}'. Please add the image and restart.")
        st.stop()

    try:
        # Load the sample face encoding and person info
        sample_face_encoding, name, person_id = setup_face_recognition(sample_image_path, name, person_id)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Webcam feed
    start_webcam = st.button("Start Webcam")

    # Use Streamlit's empty container for dynamic updates
    status_container = st.empty()
    info_container = st.empty()

    if start_webcam:
        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("Could not access the webcam. Please check your device.")
            st.stop()

        frame_placeholder = st.empty()
        stop_webcam = st.button("Stop Webcam")

        while not stop_webcam:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture a frame from the webcam.")
                break

            # Process the frame for face recognition
            processed_frame, displayed_data = process_webcam_frame(frame, sample_face_encoding, name, person_id)

            # Display the processed frame
            frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)

            # Update the status dynamically based on face match
            status_container.text(f"Status: {displayed_data.splitlines()[0]}")
            if "No data found" in displayed_data:
                info_container.empty()  # Clear previous data
            else:
                info_container.text(f"Name: {displayed_data.splitlines()[1]}")
                info_container.text(f"ID: {displayed_data.splitlines()[2]}")

        # Release webcam resources
        video_capture.release()
        st.success("Webcam stopped.")

if __name__ == "__main__":
    main()
