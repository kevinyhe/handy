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
        """Get positions of all fingertips.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            dict: Dictionary of fingertip positions with finger names as keys
        """
        h, w = frame_shape[:2]
        
        # Fingertip landmark indices
        fingertip_ids = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'avg_middle': 11,
            'ring': 16,
            'pinky': 20
        }
        
        fingertips = {}
        for finger_name, landmark_id in fingertip_ids.items():
            landmark = hand_landmarks.landmark[landmark_id]
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * w), int(landmark.y * h)
            fingertips[finger_name] = (x, y)
            
        return fingertips

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
            'avg_middle': (125, 0, 255),   # Red
            'ring': (255, 0, 255),   # Magenta
            'pinky': (0, 255, 255)   # Yellow
        }
        
        # Draw circles at fingertip positions
        for finger_name, position in fingertips.items():
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