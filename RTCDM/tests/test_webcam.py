import cv2

# Try to open the default camera (usually webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera")
else:
    print("Successfully opened camera")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame")
        
        # Save the frame to confirm it worked
        cv2.imwrite('test_frame.jpg', frame)
        print("Saved test frame to test_frame.jpg")
    else:
        print("Failed to read a frame")
    
    # Release the camera
    cap.release() 