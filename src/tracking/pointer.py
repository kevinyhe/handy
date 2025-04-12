import numpy as np
import time

class Pointer:
    """Represents a single fingertip as a pointer."""
    
    def __init__(self, finger_name):
        """Initialize the finger pointer.
        
        Args:
            finger_name (str): Name of the finger ('thumb', 'index', etc.)
        """
        self.finger_name = finger_name
        self.position = (0, 0)  # Current x, y position
        self.prev_positions = []  # Store recent positions for smoothing
        self.is_active = False  # Whether this finger is currently active as a pointer
        self.history_size = 5  # Number of positions to keep for smoothing
        self.last_update_time = 0
        self.velocity = (0, 0)  # x, y velocity components
        
    def update_position(self, position):
        """Update the finger position.
        
        Args:
            position (tuple): New (x, y) position
        """
        # Calculate velocity
        current_time = time.time()
        if self.last_update_time > 0:
            dt = current_time - self.last_update_time
            if dt > 0:
                dx = position[0] - self.position[0]
                dy = position[1] - self.position[1]
                self.velocity = (dx/dt, dy/dt)
        
        self.last_update_time = current_time
        
        # Update position history
        self.prev_positions.append(self.position)
        if len(self.prev_positions) > self.history_size:
            self.prev_positions.pop(0)
        
        self.position = position
    
    def get_smoothed_position(self):
        """Get the smoothed position based on recent position history.
        
        Returns:
            tuple: Smoothed (x, y) position
        """
        if not self.prev_positions:
            return self.position
        
        # Calculate exponentially weighted average with more weight to recent positions
        positions = self.prev_positions + [self.position]
        weights = [0.7 ** (len(positions) - i) for i in range(len(positions))]
        weight_sum = sum(weights)
        
        x_sum = sum(p[0] * w for p, w in zip(positions, weights))
        y_sum = sum(p[1] * w for p, w in zip(positions, weights))
        
        return (int(x_sum / weight_sum), int(y_sum / weight_sum))
    
    def get_velocity_magnitude(self):
        """Get the magnitude of the velocity vector.
        
        Returns:
            float: Velocity magnitude in pixels per second
        """
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    def is_moving(self, threshold=100):
        """Check if the finger is moving significantly.
        
        Args:
            threshold (float): Velocity threshold in pixels per second
            
        Returns:
            bool: True if finger is moving above threshold
        """
        return self.get_velocity_magnitude() > threshold


class PointerTracker:
    """Tracks multiple fingertips as pointers."""
    
    def __init__(self):
        """Initialize the finger pointer tracker."""
        # Create finger pointers for each finger
        self.fingers = {
            'thumb': Pointer('thumb'),
            'index': Pointer('index'),
            'middle': Pointer('middle'),
            'ring': Pointer('ring'),
            'pinky': Pointer('pinky')
        }
        
        # By default, use index finger as primary pointer
        self.primary_pointer = 'index'
        self.pointer_active = False
        
        # Screen dimensions for coordinate mapping
        self.screen_dimensions = (1920, 1080)  # Default, should be set to actual screen size
        
        # Frame dimensions for coordinate mapping
        self.frame_dimensions = (640, 480)  # Default, should be set to actual frame size
        
    def update_from_fingertips(self, fingertips):
        """Update finger pointers from detected fingertips.
        
        Args:
            fingertips (dict): Dictionary of fingertip positions
                               {finger_name: (x, y), ...}
        """
        for finger_name, position in fingertips.items():
            if finger_name in self.fingers:
                self.fingers[finger_name].update_position(position)
    
    def set_primary_pointer(self, finger_name):
        """Set which finger to use as the primary pointer.
        
        Args:
            finger_name (str): Name of the finger to use as primary pointer
        """
        if finger_name in self.fingers:
            self.primary_pointer = finger_name
    
    def get_primary_pointer_position(self):
        """Get the position of the primary pointer.
        
        Returns:
            tuple: (x, y) position of the primary pointer
        """
        return self.fingers[self.primary_pointer].get_smoothed_position()
    
    def is_pointer_active(self):
        """Check if the pointer is currently active.
        
        Returns:
            bool: True if pointer is active
        """
        return self.pointer_active
    
    def set_pointer_active(self, active):
        """Set whether the pointer is active.
        
        Args:
            active (bool): Whether the pointer should be active
        """
        self.pointer_active = active
    
    def map_to_screen_coordinates(self, position, region=None):
        """Map a position from frame coordinates to screen coordinates.
        
        Args:
            position (tuple): (x, y) position in frame coordinates
            region (tuple, optional): Region in frame to map from (x, y, width, height).
                                     If None, use the entire frame.
                                    
        Returns:
            tuple: (x, y) position in screen coordinates
        """
        x, y = position
        screen_width, screen_height = self.screen_dimensions
        
        if region:
            region_x, region_y, region_width, region_height = region
            
            # Check if position is in region
            if (region_x <= x <= region_x + region_width and 
                region_y <= y <= region_y + region_height):
                
                # Map from region to screen
                x_mapped = (x - region_x) / region_width * screen_width
                y_mapped = (y - region_y) / region_height * screen_height
                
                return (int(x_mapped), int(y_mapped))
            else:
                # Position outside region
                return None
        else:
            # Map from entire frame
            frame_width, frame_height = self.frame_dimensions
            x_mapped = x / frame_width * screen_width
            y_mapped = y / frame_height * screen_height
            
            return (int(x_mapped), int(y_mapped))
    
    def set_dimensions(self, frame_dimensions, screen_dimensions=None):
        """Set the dimensions for coordinate mapping.
        
        Args:
            frame_dimensions (tuple): (width, height) of camera frame
            screen_dimensions (tuple, optional): (width, height) of screen.
                                               If None, don't change current value.
        """
        self.frame_dimensions = frame_dimensions
        if screen_dimensions:
            self.screen_dimensions = screen_dimensions
    
    def detect_gestures(self):
        """Detect common hand gestures based on finger positions.
        
        Returns:
            dict: Detected gestures {gesture_name: confidence, ...}
        """
        gestures = {}
        
        # Detect pinch (thumb and index finger close together)
        thumb_pos = self.fingers['thumb'].position
        index_pos = self.fingers['index'].position
        distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + 
                          (thumb_pos[1] - index_pos[1])**2)
        
        if distance < 50:  # Threshold for pinch detection
            gestures['pinch'] = 1.0 - (distance / 50.0)  # Confidence
        
        # Additional gestures can be implemented here
        
        return gestures