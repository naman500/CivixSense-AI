import cv2
import time
import socket

def test_ip_connection(ip, port=8080, timeout=2):
    """Test basic TCP connection to an IP:port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Network error: {e}")
        return False

def test_camera_url(url, name="Test", timeout_seconds=10):
    """Test if a camera URL works with OpenCV"""
    print(f"\nTesting camera: {name}")
    print(f"URL: {url}")
    
    # Try to open the camera
    start_time = time.time()
    cap = cv2.VideoCapture(url)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ Failed to open camera")
        cap.release()
        return False
    
    print("✅ Camera opened successfully")
    
    # Try to read a frame with timeout
    frame_read = False
    
    while time.time() - start_time < timeout_seconds:
        ret, frame = cap.read()
        if ret:
            print("✅ Successfully read a frame")
            
            # Save frame to file for verification
            filename = f"test_{name.replace(' ', '_')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Saved frame to {filename}")
            
            frame_read = True
            break
        
        print("⏳ Waiting for frame...")
        time.sleep(1)
    
    if not frame_read:
        print("❌ Failed to read any frames within timeout period")
    
    # Release the camera
    cap.release()
    return frame_read

# Test local webcam
print("==== TESTING LOCAL WEBCAM ====")
test_camera_url(0, "Local Webcam")

# Test network connectivity to the camera
print("\n==== TESTING NETWORK CONNECTIVITY ====")
camera_ip = "192.168.1.28"  # Change this to your camera's IP
camera_port = 8080  # Change this to your camera's port

if test_ip_connection(camera_ip, camera_port):
    print(f"✅ TCP connection to {camera_ip}:{camera_port} successful")
else:
    print(f"❌ TCP connection to {camera_ip}:{camera_port} failed")
    print("   This suggests a network issue, wrong IP, or camera is not running")

# Test different camera URL formats
print("\n==== TESTING DIFFERENT URL FORMATS ====")
camera_base = f"http://{camera_ip}:{camera_port}"

urls_to_test = [
    {"url": camera_base, "name": "Base URL"},
    {"url": f"{camera_base}/video", "name": "Video Path"},
    {"url": f"{camera_base}/shot.jpg", "name": "MJPEG Still"},
    {"url": f"{camera_base}/videostream.cgi", "name": "Stream CGI"},
    {"url": f"rtsp://{camera_ip}:{camera_port}/h264_ulaw.sdp", "name": "RTSP H.264"}
]

for test in urls_to_test:
    test_camera_url(test["url"], test["name"])

print("\nSuggested troubleshooting steps:")
print("1. Verify the camera's IP address and port")
print("2. Ensure the camera is powered on and connected to your network")
print("3. Try accessing the camera directly through a web browser")
print("4. Check if any firewall is blocking the connection")
print("5. If using IP Webcam app, ensure the app is running and screen is on") 