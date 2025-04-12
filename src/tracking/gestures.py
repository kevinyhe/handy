import numpy as np
import cv2
import concurrent.futures
from threading import Lock

class GestureDetector:
    """Detects and classifies hand gestures from finger positions."""
    
    
    
    def __init__(self):
        """Initialize the gesture detector."""
        # Gesture configuration parameters
        self.config = {
            'left_click': {
                'threshold': 0.2777777778,
                'fingers': ['thumb', 'index'] 
            },
            'right_click': {
                'threshold': 0.1944444444, 
                'fingers': ['thumb', 'middle']
            },
            'move': {
                'threshold': 0.1944444444,
                'fingers': ['index', 'middle', 'avg_middle']
            }
        }
        
        # Debug/visualization settings
        self.draw_debug = True
        # Create thread pool for parallel gesture detection
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.config))
        
        # Create a lock for thread-safe operations
        self.lock = Lock()
        
        # Last detected gestures cache
        self.last_gestures = {}
        
    def detect_gestures(self, finger_positions, palm_size):
        """Detect gestures based on finger positions.
        
        Args:
            finger_positions (dict): Dictionary mapping finger names to (x, y) positions
                                   {'thumb': (x, y), 'index': (x, y), ...}
        
        Returns:
            dict: Detected gestures with confidence scores
                 {'left_click': 0.95, 'right_click': 0.0, ...}
        """
        # Skip if no finger positions
        if not finger_positions:
            return {}
        
        # Create futures for all gesture detections to run in parallel
        futures = []
        futures.append(self.executor.submit(self.left_click, finger_positions, palm_size))
        futures.append(self.executor.submit(self.right_click, finger_positions, palm_size))
        futures.append(self.executor.submit(self.move_gesture, finger_positions, palm_size))
        
        # Collect results from completed futures
        gestures = {}
        for future in concurrent.futures.as_completed(futures, timeout=0.05):
            try:
                # Each future returns a dict of detected gestures
                result = future.result()
                gestures.update(result)
            except concurrent.futures.TimeoutError:
                # Skip if detection is taking too long
                pass
            except Exception as e:
                print(f"Error in gesture detection: {e}")
                
        # Update last gestures cache
        with self.lock:
            self.last_gestures = gestures
            
        return gestures
        
    def left_click(self, finger_positions, palm_size):
        """Detect left_click gesture (thumb and index finger close).
        
        Args:
            finger_positions (dict): Dictionary of finger positions
        
        Returns:
            dict: {'left_click': confidence} or {} if not detected
        """
        result = {}
        
        # Get config for left_click
        config = self.config['left_click']
        fingers = config['fingers']
        threshold = config['threshold']
        
        # Check if required fingers are present
        if all(finger in finger_positions for finger in fingers):
            # Calculate distance between thumb and index finger
            thumb_pos = finger_positions['thumb']
            index_pos = finger_positions['index']
            
            distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + 
                              (thumb_pos[1] - index_pos[1])**2)
            
            true_threshold = threshold * palm_size
            # Calculate confidence (1.0 when distance is 0, 0.0 when distance is >= threshold)
            if distance < (true_threshold):
                confidence = 1.0 - (distance / true_threshold)
                result['left_click'] = confidence
                
        return result
    
    def right_click(self, finger_positions, palm_size):
        """Detect right_click gesture (thumb and middle finger close).
        
        Args:
            finger_positions (dict): Dictionary of finger positions
        
        Returns:
            dict: {'right_click': confidence} or {} if not detected
        """
        result = {}
        
        # Get config for right_click
        config = self.config['right_click']
        fingers = config['fingers']
        threshold = config['threshold']
        
        # Check if required fingers are present
        if all(finger in finger_positions for finger in fingers):
            # Calculate distance between thumb and middle finger
            thumb_pos = finger_positions['thumb']
            middle_pos = finger_positions['middle']
            
            distance = np.sqrt((thumb_pos[0] - middle_pos[0])**2 + 
                              (thumb_pos[1] - middle_pos[1])**2)
            
            true_threshold = threshold * palm_size
            # Calculate confidence (1.0 when distance is 0, 0.0 when distance is >= threshold)
            if distance < (true_threshold):
                confidence = 1.0 - (distance / true_threshold)
                result['right_click'] = confidence
                
        return result
    
    def move_gesture(self, finger_positions, palm_size):
        """Detect move fingers gesture.
        
        Args:
            finger_positions (dict): Dictionary of finger positions
        
        Returns:
            dict: {'move': confidence} or {} if not detected
        """
        result = {}
        
        # Get config for move
        config = self.config['move']
        fingers = config['fingers']
        threshold = config['threshold']
        
        ## Check if index and middle finger are touching
        if all(finger in finger_positions for finger in fingers):
            # Calculate distance between thumb and middle finger
            index_pos = finger_positions['index']
            first_middle_pos = finger_positions['middle']
            second_middle_pos = finger_positions['avg_middle']
            
            middle_pos = (abs((first_middle_pos[0] + second_middle_pos[0]) // 2), abs((first_middle_pos[1] + second_middle_pos[1]) // 2))
            
            distance = np.sqrt((index_pos[0] - middle_pos[0])**2 + 
                              (index_pos[1] - middle_pos[1])**2)
            
            true_threshold = threshold * palm_size
            # Calculate confidence (1.0 when distance is 0, 0.0 when distance is >= threshold)
            if distance < (true_threshold):
                confidence = 1.0 - (distance / true_threshold)
                result['move'] = confidence        
                
        return result
    
    def draw_gesture_feedback(self, frame, finger_positions, gestures):
        """Draw visual feedback for detected gestures.
        
        Args:
            frame: The video frame to draw on
            finger_positions (dict): Dictionary of finger positions
            gestures (dict): Dictionary of detected gestures with confidence scores
        
        Returns:
            frame: Frame with gesture visualization
        """
        if not self.draw_debug:
            return frame
        
        # Create a copy to avoid modifying the original frame
        feedback_frame = frame.copy()
            
        # Draw left click visualization
        if 'left_click' in gestures and gestures['left_click'] > 0.3:
            thumb_pos = finger_positions['thumb']
            index_pos = finger_positions['index']
            
            # Draw line between fingers
            cv2.line(feedback_frame, thumb_pos, index_pos, (0, 255, 255), 2)
            
            # Draw midpoint circle
            midpoint = ((thumb_pos[0] + index_pos[0])//2, 
                       (thumb_pos[1] + index_pos[1])//2)
            radius = int(20 * gestures['left_click'])  # Size based on confidence
            cv2.circle(feedback_frame, midpoint, radius, (0, 255, 255), -1)
            
            # Add text label
            cv2.putText(feedback_frame, "Left Click", 
                       (midpoint[0] - 50, midpoint[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # Draw right click visualization
        if 'right_click' in gestures and gestures['right_click'] > 0.3:
            thumb_pos = finger_positions['thumb']
            middle_pos = finger_positions['middle']
            
            # Draw line between fingers
            cv2.line(feedback_frame, thumb_pos, middle_pos, (255, 0, 255), 2)
            
            # Draw midpoint circle
            midpoint = ((thumb_pos[0] + middle_pos[0])//2, 
                       (thumb_pos[1] + middle_pos[1])//2)
            radius = int(20 * gestures['right_click'])  # Size based on confidence
            cv2.circle(feedback_frame, midpoint, radius, (255, 0, 255), -1)
            
            # Add text label
            cv2.putText(feedback_frame, "Right Click", 
                       (midpoint[0] - 50, midpoint[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw three fingers visualization
        if 'move' in gestures and gestures['move'] > 0.3:
            index_pos = finger_positions['index']
            middle_pos = finger_positions['middle']
            
            # Draw connecting lines
            cv2.line(feedback_frame, index_pos, middle_pos, (0, 255, 0), 2)
            
            # Calculate center point
            center_x = (index_pos[0] + middle_pos[0]) // 2
            center_y = (index_pos[1] + middle_pos[1]) // 2
            
            # Add text label
            cv2.putText(feedback_frame, "Move", 
                       (center_x - 60, center_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return feedback_frame
    
    def shutdown(self):
        """Properly shut down the thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)