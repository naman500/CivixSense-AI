import cv2
import time

def test_webcam():
    print("Testing webcam connection...")
    print("OpenCV version:", cv2.__version__)
    
    # Try to open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    print("Webcam opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam")
        cap.release()
        return
    
    print("Successfully read frame from webcam")
    print("Frame shape:", frame.shape)
    
    # Display the frame
    cv2.imshow('Webcam Test', frame)
    cv2.waitKey(2000)  # Wait for 2 seconds
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam() 