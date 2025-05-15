import cv2
import time
import numpy as np
import requests
from urllib.parse import urlparse
import socket
import threading
from typing import Tuple, Optional, Dict, Any

class StreamTester:
    def __init__(self):
        self.timeout = 5  # seconds
        self.max_retries = 3
        self.frame_timeout = 2  # seconds
        self.test_duration = 5  # seconds
        
    def test_stream(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Test a video stream URL (RTSP or HTTP)
        
        Args:
            url: Stream URL to test
            
        Returns:
            Tuple of (success, info_dict)
        """
        # Parse URL
        parsed_url = urlparse(url)
        protocol = parsed_url.scheme.lower()
        
        if protocol == 'rtsp':
            return self._test_rtsp(url)
        elif protocol in ['http', 'https']:
            return self._test_http(url)
        else:
            return False, {'error': f'Unsupported protocol: {protocol}'}
    
    def _test_rtsp(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Test RTSP stream"""
        try:
            # Try to open the stream
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                return False, {'error': 'Failed to open RTSP stream'}
            
            # Set timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame reading
            start_time = time.time()
            frames_read = 0
            last_frame_time = start_time
            
            while time.time() - start_time < self.test_duration:
                ret, frame = cap.read()
                if not ret:
                    if time.time() - last_frame_time > self.frame_timeout:
                        return False, {'error': 'Timeout reading frames'}
                    continue
                
                frames_read += 1
                last_frame_time = time.time()
                
                # Get stream info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Calculate actual FPS
                elapsed_time = time.time() - start_time
                actual_fps = frames_read / elapsed_time if elapsed_time > 0 else 0
                
                # Check frame quality
                if frame is not None and frame.size > 0:
                    # Calculate frame statistics
                    mean_brightness = np.mean(frame)
                    std_brightness = np.std(frame)
                    
                    # Check for frozen frame
                    if frames_read > 1:
                        frame_diff = np.mean(np.abs(frame - self.last_frame))
                        if frame_diff < 1.0:  # Threshold for frozen frame
                            return False, {'error': 'Stream appears to be frozen'}
                    
                    self.last_frame = frame.copy()
            
            # Cleanup
            cap.release()
            
            # Return success with stream info
            return True, {
                'width': width,
                'height': height,
                'fps': fps,
                'actual_fps': actual_fps,
                'frames_read': frames_read,
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness
            }
            
        except Exception as e:
            return False, {'error': f'RTSP test failed: {str(e)}'}
    
    def _test_http(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Test HTTP stream"""
        try:
            # First check if URL is accessible
            response = requests.head(url, timeout=self.timeout)
            if response.status_code != 200:
                return False, {'error': f'HTTP status code: {response.status_code}'}
            
            # Try to open the stream
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                return False, {'error': 'Failed to open HTTP stream'}
            
            # Set timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame reading
            start_time = time.time()
            frames_read = 0
            last_frame_time = start_time
            
            while time.time() - start_time < self.test_duration:
                ret, frame = cap.read()
                if not ret:
                    if time.time() - last_frame_time > self.frame_timeout:
                        return False, {'error': 'Timeout reading frames'}
                    continue
                
                frames_read += 1
                last_frame_time = time.time()
                
                # Get stream info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Calculate actual FPS
                elapsed_time = time.time() - start_time
                actual_fps = frames_read / elapsed_time if elapsed_time > 0 else 0
                
                # Check frame quality
                if frame is not None and frame.size > 0:
                    # Calculate frame statistics
                    mean_brightness = np.mean(frame)
                    std_brightness = np.std(frame)
                    
                    # Check for frozen frame
                    if frames_read > 1:
                        frame_diff = np.mean(np.abs(frame - self.last_frame))
                        if frame_diff < 1.0:  # Threshold for frozen frame
                            return False, {'error': 'Stream appears to be frozen'}
                    
                    self.last_frame = frame.copy()
            
            # Cleanup
            cap.release()
            
            # Return success with stream info
            return True, {
                'width': width,
                'height': height,
                'fps': fps,
                'actual_fps': actual_fps,
                'frames_read': frames_read,
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness
            }
            
        except requests.exceptions.RequestException as e:
            return False, {'error': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return False, {'error': f'HTTP test failed: {str(e)}'}
    
    def test_multiple_urls(self, urls: list) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """
        Test multiple stream URLs in parallel
        
        Args:
            urls: List of stream URLs to test
            
        Returns:
            Dictionary mapping URLs to their test results
        """
        results = {}
        threads = []
        
        def test_url(url):
            results[url] = self.test_stream(url)
        
        # Create and start threads
        for url in urls:
            thread = threading.Thread(target=test_url, args=(url,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def get_best_url(self, urls: list) -> Optional[str]:
        """
        Test multiple URLs and return the best performing one
        
        Args:
            urls: List of stream URLs to test
            
        Returns:
            Best performing URL or None if all fail
        """
        results = self.test_multiple_urls(urls)
        
        # Filter successful results
        successful = {url: info for url, (success, info) in results.items() if success}
        
        
        if not successful:
            return None
        
        # Score each successful stream
        scores = {}
        for url, info in successful.items():
            score = 0
            # Higher FPS is better
            score += info.get('actual_fps', 0) * 2
            # Higher resolution is better
            score += (info.get('width', 0) * info.get('height', 0)) / 1000000
            # More frames read is better
            score += info.get('frames_read', 0) / 10
            # Higher brightness standard deviation is better (more dynamic content)
            score += info.get('std_brightness', 0) / 10
            
            scores[url] = score
        
        # Return URL with highest score
        return max(scores.items(), key=lambda x: x[1])[0] 