import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

def setup_face_recognition(sample_image_path):
    """
    Set up face recognition with a sample image
    Returns the face encoding of the sample image
    """
    try:
        # Load sample image
        sample_image = face_recognition.load_image_file(sample_image_path)
        
        # Convert to RGB if needed
        if len(sample_image.shape) == 2:  # If grayscale
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2RGB)
        elif sample_image.shape[2] == 4:  # If RGBA
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGBA2RGB)
            
        # Detect faces in sample image
        face_locations = face_recognition.face_locations(sample_image, model="hog")
        
        if not face_locations:
            raise Exception("No face found in sample image")
            
        # Get face encoding
        face_encodings = face_recognition.face_encodings(sample_image, face_locations)
        
        if not face_encodings:
            raise Exception("Could not compute face encoding for sample image")
            
        return face_encodings[0]
        
    except Exception as e:
        raise Exception(f"Error processing sample image: {str(e)}")

def run_face_recognition(sample_face_encoding):
    """
    Run real-time face recognition using webcam
    """
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    
    # Set frame size and fps for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    if not video_capture.isOpened():
        raise Exception("Could not access webcam")

    print("Face recognition system started. Press 'q' to quit.")
    
    # Initialize variables for frame processing
    process_every_n_frames = 3
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Only process every nth frame to improve performance
        if frame_count % process_every_n_frames == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings in current frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            # Initialize status
            status = "No Face Detected"
            color = (0, 0, 255)  # Red
            
            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        # Compare face with sample face
                        matches = face_recognition.compare_faces([sample_face_encoding], 
                                                              face_encoding, 
                                                              tolerance=0.6)
                        
                        if matches[0]:
                            status = "ACCEPTED"
                            color = (0, 255, 0)  # Green
                        else:
                            status = "REJECTED"
                            color = (0, 0, 255)  # Red
                            
                        # Draw rectangles around faces
                        for top, right, bottom, left in face_locations:
                            # Draw rectangle around face
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            
                            # Draw status below face
                            cv2.putText(frame, status, 
                                      (left, bottom + 30), 
                                      cv2.FONT_HERSHEY_DUPLEX, 
                                      0.7, color, 1)
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
            
            # Draw global status on frame
            cv2.putText(frame, f"Status: {status}", 
                      (10, 30), 
                      cv2.FONT_HERSHEY_DUPLEX, 
                      0.7, color, 1)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        frame_count += 1

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    try:
        # Replace with path to your sample image
        sample_image_path = "test_img.jpg"
        
        if not os.path.exists(sample_image_path):
            raise Exception(f"Sample image not found at {sample_image_path}")
        
        print("Initializing face recognition system...")
        sample_face_encoding = setup_face_recognition(sample_image_path)
        
        print("Starting webcam...")
        run_face_recognition(sample_face_encoding)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()