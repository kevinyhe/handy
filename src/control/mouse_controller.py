import pyautogui
import time
import numpy as np

class MouseController:
    """Controls the mouse based on detected gestures with relative movement."""
    
    def __init__(self, screen_dimensions=(2560, 1440), smoothing=0.2, sensitivity=4.0):
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
        self.click_cooldown = 0.3  # Seconds between allowed clicks
        self.inactivity_time = 0   # Time since last movement
        self.inactivity_threshold = 1.0  # Seconds of inactivity before resetting tracking
        self.last_cursor_position = None  # Store last known cursor position
        
        print(f"MouseController initialized with screen size: {self.screen_width}x{self.screen_height}")
        
    def update_mouse(self, finger_tracker, gestures):
        """Update mouse position and actions based on gestures with relative movement.
        
        Args:
            finger_tracker (PointerTracker): The finger pointer tracker
            gestures (dict): Detected gestures with confidence scores
        """
        # Only process if pointer is active
        if not finger_tracker.is_pointer_active():
            # Save cursor position before resetting tracking
            if self.is_moving:
                self.last_cursor_position = pyautogui.position()
                self.is_moving = False
                print(f"Saved cursor position: {self.last_cursor_position}")
            
            # Reset position history when pointer becomes inactive to prevent teleportation
            self.last_raw_position = None
            return
        
        # Get pointer position
        pointer_pos = finger_tracker.get_primary_pointer_position()
        if pointer_pos is None:
            return
        
        # Get raw position
        raw_x, raw_y = pointer_pos
        
        # Prevent teleportation on first detection or after inactivity
        if self.last_raw_position is None:
            # Initialize tracking from current hand position without moving the cursor
            self.last_raw_position = (raw_x, raw_y)
            
            # Initialize position history to prevent jumps
            if not hasattr(self, 'position_history') or self.position_history is None:
                self.position_history = [(raw_x, raw_y)] * 5
            
            print("Initializing position tracking")
            
            # Reset movement flags
            self.last_delta = (0, 0)
            
            # Track activation time
            self.last_active_time = time.time()
            return
        
        # Process movement if 'move' gesture is active
        if 'move' in gestures:
            # Reset inactivity timer
            self.last_active_time = time.time()
            
            # Add current position to history
            self.position_history.append((raw_x, raw_y))
            if len(self.position_history) > 5:  # Keep last 5 positions
                self.position_history.pop(0)
            
            # Calculate smoothed delta using weighted average
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # More weight to recent positions
            smooth_x = sum(pos[0] * w for pos, w in zip(self.position_history, weights))
            smooth_y = sum(pos[1] * w for pos, w in zip(self.position_history, weights))
            
            # Calculate movement based on smoothed position - RELATIVE MOVEMENT ONLY
            delta_x = (smooth_x - self.last_raw_position[0]) * self.sensitivity
            delta_y = (smooth_y - self.last_raw_position[1]) * self.sensitivity
            
            # Invert the movement direction to fix inverted controls
            delta_x = -delta_x  # Invert horizontal movement
            delta_y = -delta_y  # Invert vertical movement
            
            # Apply dead zone to reduce jitter
            dead_zone = 0.8
            if abs(delta_x) < dead_zone:
                delta_x = 0
            else:
                # Apply non-linear response for better control
                delta_x = (abs(delta_x) - dead_zone) * (delta_x / abs(delta_x))
                
            if abs(delta_y) < dead_zone:
                delta_y = 0
            else:
                delta_y = (abs(delta_y) - dead_zone) * (delta_y / abs(delta_y))
                    
            # Move mouse by the delta amount
            if abs(delta_x) > 0 or abs(delta_y) > 0:
                # Apply additional smoothing to the actual movement
                if hasattr(self, 'last_delta'):
                    # Blend previous and current delta for smoother transitions
                    smooth_delta_x = delta_x * 0.7 + self.last_delta[0] * 0.3
                    smooth_delta_y = delta_y * 0.7 + self.last_delta[1] * 0.3
                else:
                    smooth_delta_x, smooth_delta_y = delta_x, delta_y
                
                # Store current delta for next frame
                self.last_delta = (delta_x, delta_y)
                
                # moveRel moves relative to current position - ALWAYS USE RELATIVE MOVEMENT
                pyautogui.moveRel(smooth_delta_x, smooth_delta_y, duration=0)
                self.is_moving = True
            else:
                self.is_moving = False
        else:
            # When not moving, save the current cursor position
            if self.is_moving:
                self.last_cursor_position = pyautogui.position()
                self.is_moving = False
                print(f"Saved cursor position: {self.last_cursor_position}")
        
        # Update last position with raw (not smoothed) position
        self.last_raw_position = (raw_x, raw_y)

                
        # Handle mouse button actions
        current_time = time.time()
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