import cv2
import mediapipe as mp

class HandTracker:
    """Handles hand detection and tracking using MediaPipe."""
    
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, frame, draw=True):
        """Detect hands in a frame and optionally draw landmarks."""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    # Draw the hand landmarks and connections
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Draw a highlight around the hand
                    self._highlight_hand(frame, hand_landmarks)
        
        return frame, results
    
    def get_palm_size(self, frame, hand_landmarks):
        """Calculate the palm size based on hand landmarks."""
        h, w, c = frame.shape
        
        # Get palm landmark coordinates (landmark 0 is wrist)
        wrist = hand_landmarks.landmark[0]
        thumb_cmc = hand_landmarks.landmark[1]
        index_finger_mcp = hand_landmarks.landmark[5]   
        
        pinky_mcp = hand_landmarks.landmark[17]

        palm_width = abs(((thumb_cmc.x - pinky_mcp.x) ** 2 + (thumb_cmc.y - pinky_mcp.y) ** 2) ** 0.5 * w)
        palm_height = abs(((wrist.x - index_finger_mcp.x) ** 2 + (wrist.y - index_finger_mcp.y) ** 2) ** 0.5 * h)

        palm_size = (palm_width + palm_height) / 2  # Average of width and height
        return palm_size
    
    def _highlight_hand(self, frame, hand_landmarks):
        """Draw a highlight around the detected hand."""
        h, w, c = frame.shape
        
        # Get bounding box coordinates
        x_min, x_max = w, 0
        y_min, y_max = h, 0
        
        # Find bounding box of hand
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
        
        # Add padding to bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw rectangle around hand
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add semi-transparent highlight
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
        alpha = 0.2  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def get_fingertips(self, hand_landmarks, frame_shape):
        """Get positions of all fingertips and key hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            dict: Dictionary of finger positions with finger names as keys
        """
        h, w = frame_shape[:2]
        
        # Landmark indices for all important points
        landmark_ids = {
            # Fingertips
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20,
            
            # Intermediate joints (DIP = Distal Interphalangeal)
            'dip_thumb': 3,
            'dip_index': 7,
            'dip_middle': 11,
            'dip_ring': 15,
            'dip_pinky': 19,
            
            # Middle joints (PIP = Proximal Interphalangeal)
            'pip_thumb': 2,
            'pip_index': 6,
            'pip_middle': 10,
            'pip_ring': 14,
            'pip_pinky': 18,
            
            # Knuckles (MCP = Metacarpophalangeal)
            'mcp_thumb': 1,
            'mcp_index': 5,
            'mcp_middle': 9,
            'mcp_ring': 13,
            'mcp_pinky': 17,
            
            # Hand reference points
            'wrist': 0,
            'palm_center': 9,  # Middle finger MCP joint (good palm center reference)
            
            # Additional references used in gestures.py
            'avg_index': 6,    # Alternative name for pip_index for compatibility
            'avg_middle': 10,  # Alternative name for pip_middle for compatibility
            'avg_ring': 14,    # Alternative name for pip_ring for compatibility
            'avg_pinky': 18,   # Alternative name for pip_pinky for compatibility
            
            'bottom_index': 5, # Alternative name for mcp_index for compatibility
            'bottom_middle': 9, # Alternative name for mcp_middle for compatibility
            'bottom_ring': 13, # Alternative name for mcp_ring for compatibility
            'bottom_pinky': 17, # Alternative name for mcp_pinky for compatibility
        }
        
        finger_positions = {}
        for point_name, landmark_id in landmark_ids.items():
            landmark = hand_landmarks.landmark[landmark_id]
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * w), int(landmark.y * h)
            finger_positions[point_name] = (x, y)
        
        return finger_positions

    def draw_fingertips(self, frame, fingertips):
        """Draw colored circles at fingertip positions.
        
        Args:
            frame: Frame to draw on
            fingertips: Dictionary of fingertip positions returned by get_fingertips()
            
        Returns:
            frame: Frame with fingertips highlighted
        """
        # Define colors for each finger (BGR format)
        colors = {
            'thumb': (255, 0, 0),    # Blue
            'index': (0, 255, 0),    # Green
            'middle': (0, 0, 255),   # Red
            'ring': (255, 0, 255),   # Magenta
            'pinky': (0, 255, 255)   # Yellow
        }
        
        # Draw circles at fingertip positions
        for finger_name, position in fingertips.items():
            if finger_name not in colors:
                continue
            # Draw filled circle
            cv2.circle(frame, position, 8, colors[finger_name], -1)
            # Draw circle outline
            cv2.circle(frame, position, 8, (255, 255, 255), 2)
            
            # # Add label text
            # cv2.putText(frame, finger_name, 
            #         (position[0]-10, position[1]-10), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            #         colors[finger_name], 2)
                    
        return frame

    def find_hands(self, frame, draw=True, track_fingertips=True):
        """Detect hands in a frame and optionally draw landmarks.
        
        Args:
            frame: Frame to process
            draw: Whether to draw hand landmarks
            track_fingertips: Whether to track and highlight fingertips
            
        Returns:
            frame: Processed frame
            results: Hand detection results
            fingertips_list: List of dictionaries containing fingertip positions for each hand
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = self.hands.process(rgb_frame)
        
        fingertips_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    # Draw the hand landmarks and connections
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Draw a highlight around the hand
                    self._highlight_hand(frame, hand_landmarks)
                
                if track_fingertips:
                    # Get fingertip positions
                    fingertips = self.get_fingertips(hand_landmarks, frame.shape)
                    fingertips_list.append(fingertips)
                    
                    # Draw fingertips
                    frame = self.draw_fingertips(frame, fingertips)
        
        return frame, results, fingertips_list