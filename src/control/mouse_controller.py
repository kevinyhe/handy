import pyautogui
import time
import numpy as np

class MouseController:
    """Controls the mouse based on detected gestures with relative movement."""
    
    def __init__(self, screen_dimensions=(2560, 1440), smoothing=0.4, sensitivity=6.0):
        """Initialize the mouse controller.
        
        Args:
            screen_dimensions (tuple): Screen dimensions (width, height)
            smoothing (float): Smoothing factor for mouse movement (0-1)
            sensitivity (float): Multiplier for mouse movement speed
        """
        self.screen_width, self.screen_height = screen_dimensions
        self.smoothing = smoothing
        self.sensitivity = sensitivity
        
        # Get actual screen size from pyautogui
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Set PyAutoGUI settings
        pyautogui.FAILSAFE = True  # Move mouse to screen corner to abort
        pyautogui.PAUSE = 0.0      # Minimum delay between commands
        
        # Mouse state tracking
        self.last_raw_position = None
        self.is_moving = False
        self.left_button_down = False
        self.right_button_down = False
        self.last_click_time = 0
        self.click_cooldown = 0.05  # Seconds between allowed clicks
        self.inactivity_time = 0   # Time since last movement
        self.inactivity_threshold = 1.0  # Seconds of inactivity before resetting tracking
        self.last_cursor_position = None  # Store last known cursor position
        
        print(f"MouseController initialized with screen size: {self.screen_width}x{self.screen_height}")
            
    def update_mouse(self, finger_tracker, gestures, palm_size=None):
        """Update mouse position and actions based on gestures with relative movement.
        
        Args:
            finger_tracker (PointerTracker): The finger tracker
            gestures (dict): Detected gestures with confidence scores
            palm_size (float, optional): Size of the palm for scaling sensitivity
        """
        # Get pointer position
        pointer_pos = finger_tracker.get_primary_pointer_position()
        if pointer_pos is None:
            return
        
        # Get raw position
        raw_x, raw_y = pointer_pos
        current_time = time.time()
        
        # Check if move gesture is active
        move_active = 'move' in gestures
        
        # Check if drag is active (ring and pinky clenched) during a click
        drag_active = ('drag' in gestures and 
                    (self.left_button_down or self.right_button_down or 
                    'left_click' in gestures or 'right_click' in gestures))
        
        # Should move if either regular move is active or drag is active
        should_move = move_active or drag_active
        
        # Debug the gestures dict to see what's going on
        print(f"Gestures: {gestures}, Should move: {should_move}")
        
        # Scale sensitivity based on palm size if provided
        effective_sensitivity = self.sensitivity
        if palm_size is not None and palm_size > 0:
            # Base calibration value - adjust this based on your typical tracking distance
            base_palm_size = 110  # This should be the palm size at your "normal" distance
            
            # Calculate scaling factor based on palm size
            # Smaller palm size (hand further away) = higher sensitivity
            # Larger palm size (hand closer) = lower sensitivity
            scaling_factor = base_palm_size / palm_size
            
            # Apply bounds to prevent extreme sensitivity values
            scaling_factor = max(0.5, min(scaling_factor, 2.0))
            
            # Apply scaling to sensitivity
            effective_sensitivity = self.sensitivity * scaling_factor
            
            print(f"Palm size: {palm_size}, Scaling: {scaling_factor:.2f}, Effective sensitivity: {effective_sensitivity:.2f}")
        
        # Initialize tracking if this is the first detection
        if self.last_raw_position is None:
            self.last_raw_position = (raw_x, raw_y)
            self.last_delta = (0, 0)
            self.position_history = [(raw_x, raw_y)] * 5
            print("Initialized tracking")
            return
        
        # Add to position history for smoothing
        self.position_history.append((raw_x, raw_y))
        if len(self.position_history) > 5:
            self.position_history.pop(0)
        
        # Calculate movement delta (how much the hand has moved)
        delta_x = (raw_x - self.last_raw_position[0]) * -effective_sensitivity
        delta_y = (raw_y - self.last_raw_position[1]) * -effective_sensitivity
        
        # Debug print to see what's happening
        print(f"Delta: ({delta_x:.2f}, {delta_y:.2f}), Move active: {should_move}")
        
        # Apply dead zone 
        dead_zone = 0.8
        if abs(delta_x) < dead_zone:
            delta_x = 0
        else:
            delta_x = (abs(delta_x) - dead_zone) * (delta_x / abs(delta_x))
        
        if abs(delta_y) < dead_zone:
            delta_y = 0
        else:
            delta_y = (abs(delta_y) - dead_zone) * (delta_y / abs(delta_y))
        
        # Move if should_move is true AND we have delta values
        if should_move and (abs(delta_x) > 0 or abs(delta_y) > 0):
            # Invert movement direction (camera is mirrored)
            delta_x = -delta_x
            delta_y = -delta_y
            
            # Apply smoothing
            if hasattr(self, 'last_delta'):
                smooth_delta_x = delta_x * 0.7 + self.last_delta[0] * 0.3
                smooth_delta_y = delta_y * 0.7 + self.last_delta[1] * 0.3
            else:
                smooth_delta_x, smooth_delta_y = delta_x, delta_y
            
            # Store for next frame
            self.last_delta = (delta_x, delta_y)
            
            # Actually move the mouse - use larger values to ensure movement is visible
            movement_x = smooth_delta_x * 1.5
            movement_y = smooth_delta_y * 1.5
            
            print(f"Moving mouse: ({movement_x:.2f}, {movement_y:.2f})")
            pyautogui.moveRel(movement_x, movement_y, duration=0)
            self.is_moving = True
            self.last_active_time = current_time
        else:
            if self.is_moving and not should_move:
                self.is_moving = False
                print("Move/drag gesture ended")
        
        # Update last position for next frame's delta calculation
        self.last_raw_position = (raw_x, raw_y)

        # Handle mouse button actions
        if current_time - self.last_click_time > self.click_cooldown:
            # Left click - press and hold
            if 'left_click' in gestures:
                if not self.left_button_down:
                    pyautogui.mouseDown(button='left')
                    self.left_button_down = True
                    self.last_click_time = current_time
                    print("Left mouse button down")
            elif self.left_button_down:
                pyautogui.mouseUp(button='left')
                self.left_button_down = False
                print("Left mouse button up")
            
            # Right click - press and hold
            if 'right_click' in gestures:
                if not self.right_button_down:
                    pyautogui.mouseDown(button='right')
                    self.right_button_down = True
                    self.last_click_time = current_time
                    print("Right mouse button down")
            elif self.right_button_down:
                pyautogui.mouseUp(button='right')
                self.right_button_down = False
                print("Right mouse button up")
                
        if 'scroll' in gestures:  # Threshold for activation
            if 'scroll_direction' in gestures:
                direction = gestures['scroll_direction']
                
                # Scale the scroll amount based on the direction value
                # Adjust the multiplier to control scroll sensitivity
                scroll_amount = int(direction * 3)  # Adjust multiplier as needed
                
                # Apply minimum threshold to avoid tiny scrolls
                if abs(scroll_amount) > 0.5:
                    # Scroll (positive = down, negative = up)
                    pyautogui.scroll(-scroll_amount)  # Invert for natural scrolling
                    print(f"Scrolling: {scroll_amount}")