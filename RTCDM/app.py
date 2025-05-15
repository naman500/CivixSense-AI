from flask import Flask, Response, request
from camera_manager import camera_manager
from dashboard.dashboard import gen_frames
import cv2

app = Flask(__name__)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Video feed endpoint with support for static and optimized frames"""
    try:
        # Check if camera exists
        if not camera_manager.camera_exists(camera_id):
            return 'Camera not found', 404
            
        # Get camera stream
        stream = camera_manager.get_stream(camera_id)
        if not stream:
            return 'Stream not available', 503
            
        # Handle static frame request
        if request.args.get('static') == 'true':
            frame = stream.get_static_frame()
            if frame is None:
                return 'Failed to capture frame', 503
                
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
            
        # Regular video stream
        return Response(gen_frames(stream),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
                      
    except Exception as e:
        print(f"Error in video feed: {str(e)}")
        return 'Internal server error', 500 