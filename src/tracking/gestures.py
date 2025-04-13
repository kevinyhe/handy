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
                'threshold': 0.2244444444,
                'fingers': ['thumb', 'index'] 
            },
            'right_click': {
                'threshold': 0.2244444444, 
                'fingers': ['thumb', 'middle']
            },
            'move': {
                'threshold': 0.2877777778,
                'fingers': ['index', 'middle', 'dip_middle', 'pip_middle', 'dip_index', 'pip_index']
            },
            'drag': {
                'threshold': 0.20,
                'fingers': ['ring', 'pinky', 'palm_center', 'mcp_ring', 'mcp_pinky']
            },
            'scroll': {
                'threshold': 0.2477777778,
                'fingers': ['middle', 'ring', 'dip_middle', 'pip_middle', 'dip_ring', 'pip_ring', 'wrist', 'palm_center'],
                'base_speed': 20.0,        # Base scroll speed
                'max_speed': 80.0,         # Maximum scroll speed
                'straightness_factor': 3.3  # How much straightness affects speed
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
            first_index_pos = finger_positions['index']
            second_index_pos = finger_positions['dip_index']
            third_index_pos = finger_positions['pip_index']
            
            first_middle_pos = finger_positions['middle']
            second_middle_pos = finger_positions['dip_middle']
            third_middle_pos = finger_positions['pip_middle']

            # Prevent division by zero in slope calculations
            # Top index finger segment
            dx_top_index = first_index_pos[0] - second_index_pos[0]
            dy_top_index = first_index_pos[1] - second_index_pos[1]
            top_index_slope = np.arctan2(abs(dy_top_index), abs(dx_top_index)) if abs(dx_top_index) > 0.001 else np.pi/2
            
            # Bottom index finger segment
            dx_bot_index = third_index_pos[0] - second_index_pos[0]
            dy_bot_index = third_index_pos[1] - second_index_pos[1]
            bot_index_slope = np.arctan2(abs(dy_bot_index), abs(dx_bot_index)) if abs(dx_bot_index) > 0.001 else np.pi/2
            
            # Top middle finger segment
            dx_top_middle = first_middle_pos[0] - second_middle_pos[0]
            dy_top_middle = first_middle_pos[1] - second_middle_pos[1]
            top_middle_slope = np.arctan2(abs(dy_top_middle), abs(dx_top_middle)) if abs(dx_top_middle) > 0.001 else np.pi/2
            
            # Bottom middle finger segment
            dx_bot_middle = third_middle_pos[0] - second_middle_pos[0]
            dy_bot_middle = third_middle_pos[1] - second_middle_pos[1]
            bot_middle_slope = np.arctan2(abs(dy_bot_middle), abs(dx_bot_middle)) if abs(dx_bot_middle) > 0.001 else np.pi/2
            
            # Check if fingers are roughly parallel
            if abs(top_index_slope - top_middle_slope) > 0.25 or abs(bot_index_slope - bot_middle_slope) > 0.25:
                # print("move fingers slope too much lol")
                return result
            
            # Integer division for OpenCV coordinates
            middle_pos = (int((first_middle_pos[0] + second_middle_pos[0]) / 2), 
                        int((first_middle_pos[1] + second_middle_pos[1]) / 2))
            
            distance = np.sqrt((first_index_pos[0] - middle_pos[0])**2 + 
                            (first_index_pos[1] - middle_pos[1])**2)
            
            true_threshold = threshold * palm_size
            # Calculate confidence (1.0 when distance is 0, 0.0 when distance is >= threshold)
            if distance < (true_threshold):
                confidence = 1.0 - (distance / true_threshold)
                result['move'] = confidence        
                
        return result
    
    def drag_gesture(self, finger_positions, palm_size):
        """Detect if ring and pinky fingers are clenched for drag gesture.
        
        Args:
            finger_positions (dict): Dictionary of finger joint positions
            palm_size (float): Size of palm for scaling thresholds
            
        Returns:
            dict: {'drag': confidence} or {} if not detected
        """
        result = {}
        
        # Get config for drag gesture
        config = self.config['drag']
        required_points = config['fingers']
        threshold = config['threshold']
        
        # Check if all required points are available
        if not all(point in finger_positions for point in required_points):
            return result
        
        # Get positions
        ring_tip = finger_positions['ring']
        pinky_tip = finger_positions['pinky']
        palm_center = finger_positions['palm_center']
        mcp_ring = finger_positions['mcp_ring']
        mcp_pinky = finger_positions['mcp_pinky']
        
        # Calculate direction vectors from knuckle to fingertip
        # This is more reliable than just comparing y-coordinates
        ring_vector = (ring_tip[0] - mcp_ring[0], ring_tip[1] - mcp_ring[1])
        pinky_vector = (pinky_tip[0] - mcp_pinky[0], pinky_tip[1] - mcp_pinky[1])
        
        # Check if fingers are curled inward by checking if they're closer to the palm
        # than they would be if extended
        ring_distance = ((ring_tip[0] - palm_center[0])**2 + (ring_tip[1] - palm_center[1])**2)**0.5
        pinky_distance = ((pinky_tip[0] - palm_center[0])**2 + (pinky_tip[1] - palm_center[1])**2)**0.5
        
        # Expected extended distance (from knuckle to palm center plus a bit more)
        ring_extended_distance = ((mcp_ring[0] - palm_center[0])**2 + (mcp_ring[1] - palm_center[1])**2)**0.5 * 2
        pinky_extended_distance = ((mcp_pinky[0] - palm_center[0])**2 + (mcp_pinky[1] - palm_center[1])**2)**0.5 * 2
        
        # If distances are significantly less than extended, fingers are curled
        ring_curl_factor = ring_distance / ring_extended_distance
        pinky_curl_factor = pinky_distance / pinky_extended_distance
        
        # Scale threshold based on palm size, similar to other gestures
        curl_threshold = threshold * 2  # The threshold from config needs scaling for curl factor
        
        # Fingers are considered clenched if curl factor is below threshold
        ring_clenched = ring_curl_factor < curl_threshold
        pinky_clenched = pinky_curl_factor < curl_threshold
        
        # Also check if fingers are pointing downward
        ring_pointing_down = ring_vector[1] > 0
        pinky_pointing_down = pinky_vector[1] > 0
        both_pointing_down = ring_pointing_down and pinky_pointing_down
        
        print(f"Ring curl: {ring_curl_factor:.2f}, Pinky curl: {pinky_curl_factor:.2f}")
        
        # Detect drag if either both fingers are clenched or both are pointing down
        if (ring_clenched and pinky_clenched) or both_pointing_down:
            # Calculate confidence based on how clenched the fingers are
            if ring_clenched and pinky_clenched:
                confidence = 1.0 - ((ring_curl_factor + pinky_curl_factor) / 2 / curl_threshold)
            else:
                # If using pointing down detection, use a fixed confidence
                confidence = 0.7
                
            result['drag'] = min(1.0, max(0.0, confidence))
            print(f"Ring and pinky clenched detected: {result['drag']:.2f}")
        
        return result
    
    def scroll_gesture(self, finger_positions, palm_size):
        """Detect scroll gesture when middle and ring fingers are touching.
        
        Args:
            finger_positions (dict): Dictionary of finger joint positions
            palm_size (float): Size of palm for scaling thresholds
            
        Returns:
            dict: {'scroll': confidence, 'direction': value} or {} if not detected
        """
        result = {}
        
        # Get config for scroll gesture
        config = self.config['scroll']
        required_points = ['middle', 'ring', 'dip_middle', 'pip_middle', 'dip_ring', 'pip_ring', 'wrist', 'palm_center']
        threshold = config['threshold']
        
        # Get scroll speed settings from config
        base_speed = config.get('base_speed', 1.0)
        max_speed = config.get('max_speed', 3.0)
        straightness_factor = config.get('straightness_factor', 2.0)
        
        # Check if all required points are available
        if not all(point in finger_positions for point in required_points):
            return result
        
        # Get finger positions
        first_middle_pos = finger_positions['middle']
        second_middle_pos = finger_positions['dip_middle']
        third_middle_pos = finger_positions['pip_middle']
        
        first_ring_pos = finger_positions['ring']
        second_ring_pos = finger_positions['dip_ring']
        third_ring_pos = finger_positions['pip_ring']
        
        wrist_pos = finger_positions['wrist']
        palm_center = finger_positions['palm_center']

        # Calculate finger direction vectors
        middle_vector = (first_middle_pos[0] - third_middle_pos[0], first_middle_pos[1] - third_middle_pos[1])
        ring_vector = (first_ring_pos[0] - third_ring_pos[0], first_ring_pos[1] - third_ring_pos[1])
        
        # In image coordinates, negative y is up, positive y is down
        # Determine if fingers are pointing up or down
        middle_pointing_up = middle_vector[1] < 0
        ring_pointing_up = ring_vector[1] < 0
        
        # Debug output for finger direction
        direction_str = "up" if middle_pointing_up else "down"
        print(f"Middle pointing {direction_str}, Ring pointing {'up' if ring_pointing_up else 'down'}")
        
        # Require both fingers to point in the same direction
        if middle_pointing_up != ring_pointing_up:
            print("Scroll gesture: Fingers not pointing in same direction")
            return result

        # Prevent division by zero in slope calculations
        # Top middle finger segment
        dx_top_middle = first_middle_pos[0] - second_middle_pos[0]
        dy_top_middle = first_middle_pos[1] - second_middle_pos[1]
        top_middle_slope = np.arctan2(abs(dy_top_middle), abs(dx_top_middle)) if abs(dx_top_middle) > 0.001 else np.pi/2
        
        # Bottom middle finger segment
        dx_bot_middle = third_middle_pos[0] - second_middle_pos[0]
        dy_bot_middle = third_middle_pos[1] - second_middle_pos[1]
        bot_middle_slope = np.arctan2(abs(dy_bot_middle), abs(dx_bot_middle)) if abs(dx_bot_middle) > 0.001 else np.pi/2
        
        # Top ring finger segment
        dx_top_ring = first_ring_pos[0] - second_ring_pos[0]
        dy_top_ring = first_ring_pos[1] - second_ring_pos[1]
        top_ring_slope = np.arctan2(abs(dy_top_ring), abs(dx_top_ring)) if abs(dx_top_ring) > 0.001 else np.pi/2
        
        # Bottom ring finger segment
        dx_bot_ring = third_ring_pos[0] - second_ring_pos[0]
        dy_bot_ring = third_ring_pos[1] - second_ring_pos[1]
        bot_ring_slope = np.arctan2(abs(dy_bot_ring), abs(dx_bot_ring)) if abs(dx_bot_ring) > 0.001 else np.pi/2
        
        # Check if fingers are roughly parallel
        if abs(top_middle_slope - top_ring_slope) > 0.25 or abs(bot_middle_slope - bot_ring_slope) > 0.25:
            print("Scroll gesture: Fingers not parallel enough")
            return result
        
        # Calculate finger straightness 
        # Compare the angles between top and bottom segments of each finger
        # Perfect straightness would have identical slopes for top and bottom segments
        middle_straightness = 1.0 - min(abs(top_middle_slope - bot_middle_slope), np.pi/4) / (np.pi/4)
        ring_straightness = 1.0 - min(abs(top_ring_slope - bot_ring_slope), np.pi/4) / (np.pi/4)
        
        # Average straightness of both fingers (0.0 = bent, 1.0 = perfectly straight)
        avg_straightness = (middle_straightness + ring_straightness) / 2.0
        
        print(f"Finger straightness: middle={middle_straightness:.2f}, ring={ring_straightness:.2f}, avg={avg_straightness:.2f}")
        
        # Integer division for OpenCV coordinates
        ring_pos = (int((first_ring_pos[0] + second_ring_pos[0]) / 2), 
                int((first_ring_pos[1] + second_ring_pos[1]) / 2))
        
        # Calculate distance between middle and ring fingertips
        distance = np.sqrt((first_middle_pos[0] - first_ring_pos[0])**2 + 
                        (first_middle_pos[1] - first_ring_pos[1])**2)
        
        true_threshold = threshold * palm_size
        print(f"Finger distance: {distance:.2f}, Threshold: {true_threshold:.2f}")
        
        # Calculate confidence (1.0 when distance is 0, 0.0 when distance is >= threshold)
        if distance < (true_threshold):
                confidence = 1.0 - (distance / true_threshold)
                result['scroll'] = confidence
                
                # Calculate scroll speed based on finger straightness
                # Straighter fingers = faster scrolling
                # Apply non-linear scaling using the straightness factor
                speed_multiplier = base_speed + (avg_straightness ** straightness_factor) * (max_speed - base_speed)
                
                # Determine direction based on finger orientation
                if middle_pointing_up:
                    # Fingers pointing up = scroll up = negative direction
                    direction = -speed_multiplier
                else:
                    # Fingers pointing down = scroll down = positive direction
                    direction = speed_multiplier
                
                # Store the direction (will be handled by mouse_controller)
                result['scroll_direction'] = direction
                
                print(f"Middle-Ring scroll detected: confidence={confidence:.2f}, direction={direction:.2f}, speed={speed_multiplier:.2f}")
                        
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
        if 'left_click' in gestures and gestures['left_click'] > 0:
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
        if 'right_click' in gestures and gestures['right_click'] > 0:
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
        
        # Draw move gesture visualization
        if 'move' in gestures and gestures['move'] > 0:
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
        
        # Draw ring-pinky drag visualization
        if 'drag' in gestures and gestures['drag'] > 0:
            if 'ring' in finger_positions and 'pinky' in finger_positions:
                ring_pos = finger_positions['ring']
                pinky_pos = finger_positions['pinky']
                
                # Draw connecting lines
                cv2.line(feedback_frame, ring_pos, pinky_pos, (255, 165, 0), 2)  # Orange
                
                # Calculate center point
                center_x = (ring_pos[0] + pinky_pos[0]) // 2
                center_y = (ring_pos[1] + pinky_pos[1]) // 2
                
                # Add text label
                cv2.putText(feedback_frame, "Drag", 
                        (center_x - 60, center_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
        if 'scroll' in gestures:
            # Get the center of the palm for visualization
            if 'palm_center' in finger_positions:
                palm_center = finger_positions['palm_center']
                
                # Draw circle at palm center
                radius = int(30 * gestures['scroll'])  # Size based on confidence
                cv2.circle(feedback_frame, palm_center, radius, (0, 128, 255), -1)  # Orange-ish color
                
                # Draw arrow to indicate scroll direction
                if 'scroll_direction' in gestures:
                    direction = gestures['scroll_direction']
                    arrow_length = int(50 * abs(direction))
                    arrow_length = min(150, max(30, arrow_length))  # Limit length
                    
                    # Get arrow endpoint based on direction
                    if direction > 0:  # Scroll down
                        arrow_end = (palm_center[0], palm_center[1] + arrow_length)
                        cv2.arrowedLine(feedback_frame, palm_center, arrow_end, (0, 128, 255), 3)
                    else:  # Scroll up
                        arrow_end = (palm_center[0], palm_center[1] - arrow_length)
                        cv2.arrowedLine(feedback_frame, palm_center, arrow_end, (0, 128, 255), 3)
                
                # Add text label
                cv2.putText(feedback_frame, "Scroll", 
                        (palm_center[0] - 30, palm_center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
                        
        return feedback_frame
    
    def detect_gestures(self, finger_positions, palm_size):
        """Detect gestures based on finger positions.
        
        Args:
            finger_positions (dict): Dictionary mapping finger names to (x, y) positions
                                {'thumb': (x, y), 'index': (x, y), ...}
            palm_size (float): Size of palm for scaling thresholds
        
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
        futures.append(self.executor.submit(self.drag_gesture, finger_positions, palm_size))
        futures.append(self.executor.submit(self.scroll_gesture, finger_positions, palm_size))
        
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
    
    def shutdown(self):
        """Properly shut down the thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)