"""Module containing all adjustable application settings."""
import json
import os

class Settings:
    """Manages application settings and provides save/load functionality."""
        
    def __init__(self, config_path="settings.json"):
        """Initialize settings with default values."""
        self.config_path = config_path
        
        # Default settings
        self.defaults = {
            # Hand tracking
            "hand_detection_confidence": 0.7,
            "hand_tracking_confidence": 0.5,
            
            # Gesture thresholds
            "left_click_threshold": 0.2244444444,
            "right_click_threshold": 0.2244444444,
            "move_threshold": 0.2877777778,
            "drag_threshold": 0.20,
            "scroll_threshold": 0.2477777778,
            
            # Scroll settings
            "scroll_base_speed": 10.0,
            "scroll_max_speed": 30.0,
            "scroll_straightness_factor": 2.0,
            
            # Mouse control
            "mouse_sensitivity": 6.0,
            "mouse_smoothing": 0.4,
            "base_palm_size": 110.0,
            "dead_zone": 0.8,
            
            # Display
            "show_debug_info": True,
            "show_gestures": True,
            "show_fps": True,
        }
        
        # Current settings (will be populated from file or defaults)
        self.current = self.defaults.copy()
        
        # Load settings from file if it exists
        self.load()
    
    def load(self):
        """Load settings from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    # Update current settings with loaded values
                    self.current.update(loaded)
                print(f"Settings loaded from {self.config_path}")
            else:
                print(f"Settings file {self.config_path} not found, using defaults")
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save(self):
        """Save current settings to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.current, f, indent=4)
            print(f"Settings saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get(self, key, default=None):
        """Get a setting value by key."""
        return self.current.get(key, default)
    
    def set(self, key, value):
        """Set a setting value by key."""
        self.current[key] = value
    
    def reset(self):
        """Reset settings to defaults."""
        self.current = self.defaults.copy()
        print("Settings reset to defaults")