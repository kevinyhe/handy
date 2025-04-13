"""GUI menu for adjusting application settings using PyQt5."""
from PyQt5.QtWidgets import (QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QSlider, QCheckBox, QPushButton, QFrame, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal

class SettingsMenu(QDialog):
    """Settings menu dialog for adjusting application parameters."""
    
    # Signal for when settings change
    settings_changed = pyqtSignal()
    
    def __init__(self, settings):
        """Initialize the settings menu.
        
        Args:
            settings: The Settings object to modify
        """
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Hand Gesture Control Settings")
        self.setMinimumWidth(500)
        
        self.sliders = {}
        self.checkboxes = {}
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        self.tracking_tab = self.create_tracking_tab()
        self.gesture_tab = self.create_gesture_tab()
        self.mouse_tab = self.create_mouse_tab()
        self.display_tab = self.create_display_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.tracking_tab, "Hand Tracking")
        self.tabs.addTab(self.gesture_tab, "Gestures")
        self.tabs.addTab(self.mouse_tab, "Mouse Control")
        self.tabs.addTab(self.display_tab, "Display")
        
        main_layout.addWidget(self.tabs)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Save button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(save_button)
        
        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        buttons_layout.addWidget(apply_button)
        
        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_settings)
        buttons_layout.addWidget(reset_button)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        buttons_layout.addWidget(cancel_button)
        
        main_layout.addLayout(buttons_layout)
    
    def create_tracking_tab(self):
        """Create the hand tracking settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Hand detection confidence
        self.add_slider(layout, "Hand Detection Confidence", 
                       "hand_detection_confidence", 0.1, 1.0, 0.05, 
                       "Higher values require more confidence before detecting a hand")
        
        # Hand tracking confidence
        self.add_slider(layout, "Hand Tracking Confidence", 
                       "hand_tracking_confidence", 0.1, 1.0, 0.05, 
                       "Higher values require more confidence to maintain tracking")
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        return tab
    
    def create_gesture_tab(self):
        """Create the gesture settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Click gestures group
        click_group = QGroupBox("Click Gestures")
        click_layout = QVBoxLayout(click_group)
        
        # Left click threshold
        self.add_slider(click_layout, "Left Click Threshold", 
                       "left_click_threshold", 0.1, 0.5, 0.01, 
                       "Smaller values make left click more sensitive")
        
        # Right click threshold
        self.add_slider(click_layout, "Right Click Threshold", 
                       "right_click_threshold", 0.1, 0.5, 0.01, 
                       "Smaller values make right click more sensitive")
        
        layout.addWidget(click_group)
        
        # Movement gestures group
        move_group = QGroupBox("Movement Gestures")
        move_layout = QVBoxLayout(move_group)
        
        # Move threshold
        self.add_slider(move_layout, "Move Gesture Threshold", 
                       "move_threshold", 0.1, 0.5, 0.01, 
                       "Smaller values make move gesture more sensitive")
        
        # Drag threshold
        self.add_slider(move_layout, "Drag Gesture Threshold", 
                       "drag_threshold", 0.1, 0.5, 0.01, 
                       "Smaller values make drag gesture more sensitive")
        
        # Drag curl multiplier
        self.add_slider(move_layout, "Drag Curl Multiplier", 
                       "drag_curl_multiplier", 1.0, 4.0, 0.1, 
                       "Smaller values require more finger curling for drag")
        
        scroll_group = QGroupBox("Scroll Gesture")
        scroll_layout = QVBoxLayout(scroll_group)

        # Scroll base speed
        self.add_slider(scroll_layout, "Scroll Base Speed", 
                        "scroll_base_speed", 0.5, 3.0, 0.1, 
                        "Base speed for scrolling")

        # Scroll max speed
        self.add_slider(scroll_layout, "Scroll Max Speed", 
                        "scroll_max_speed", 1.0, 6.0, 0.1, 
                        "Maximum speed for scrolling with straight fingers")

        # Scroll straightness factor
        self.add_slider(scroll_layout, "Straightness Impact", 
                        "scroll_straightness_factor", 0.5, 4.0, 0.1, 
                        "How much finger straightness affects scroll speed")

        layout.addWidget(scroll_group)
        
        layout.addWidget(move_group)
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        return tab
    
    def create_mouse_tab(self):
        """Create the mouse control settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Mouse sensitivity
        self.add_slider(layout, "Mouse Sensitivity", 
                       "mouse_sensitivity", 1.0, 15.0, 0.5, 
                       "Higher values make cursor move faster")
        
        # Mouse smoothing
        self.add_slider(layout, "Mouse Smoothing", 
                       "mouse_smoothing", 0.0, 1.0, 0.05, 
                       "Higher values make cursor movement smoother but less responsive")
        
        # Base palm size
        self.add_slider(layout, "Base Palm Size", 
                       "base_palm_size", 50.0, 200.0, 5.0, 
                       "Reference palm size for distance calibration")
        
        # Dead zone
        self.add_slider(layout, "Dead Zone", 
                       "dead_zone", 0.1, 2.0, 0.1, 
                       "Minimum movement required before cursor moves")
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        return tab
    
    def create_display_tab(self):
        """Create the display settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Show debug info
        self.add_checkbox(layout, "Show Debug Info", 
                        "show_debug_info", 
                        "Display tracking and gesture information on screen")
        
        # Show gestures
        self.add_checkbox(layout, "Show Gesture Indicators", 
                        "show_gestures", 
                        "Highlight detected gestures on screen")
        
        # Show FPS
        self.add_checkbox(layout, "Show FPS", 
                        "show_fps", 
                        "Display frames per second counter")
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        return tab
    
    def add_slider(self, parent_layout, label_text, setting_key, min_val, max_val, step, help_text=""):
        """Add a slider for a numeric setting."""
        # Create frame for this control
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        layout = QVBoxLayout(frame)
        
        # Add main label
        label = QLabel(label_text)
        layout.addWidget(label)
        
        # Add help text if provided
        if help_text:
            help_label = QLabel(help_text)
            help_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(help_label)
        
        # Create slider layout
        slider_layout = QHBoxLayout()
        
        # Get current value
        current_value = self.settings.get(setting_key)
        
        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(int(current_value * 100))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int(step * 100))
        
        # Create value label
        value_label = QLabel(f"{current_value:.2f}")
        value_label.setMinimumWidth(50)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Connect slider value change to update function
        def update_value():
            val = slider.value() / 100.0
            value_label.setText(f"{val:.2f}")
            self.settings.set(setting_key, val)
            
        slider.valueChanged.connect(update_value)
        
        # Add to slider layout
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        # Add slider layout to main layout
        layout.addLayout(slider_layout)
        
        # Add to parent layout
        parent_layout.addWidget(frame)
        
        # Store reference to slider
        self.sliders[setting_key] = (slider, value_label)
    
    def add_checkbox(self, parent_layout, label_text, setting_key, help_text=""):
        """Add a checkbox for a boolean setting."""
        # Create frame for this control
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        layout = QVBoxLayout(frame)
        
        # Get current value
        current_value = self.settings.get(setting_key)
        
        # Create checkbox
        checkbox = QCheckBox(label_text)
        checkbox.setChecked(current_value)
        layout.addWidget(checkbox)
        
        # Add help text if provided
        if help_text:
            help_label = QLabel(help_text)
            help_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(help_label)
        
        # Connect checkbox state change to update function
        def update_value(state):
            self.settings.set(setting_key, state == Qt.Checked)
            
        checkbox.stateChanged.connect(update_value)
        
        # Add to parent layout
        parent_layout.addWidget(frame)
        
        # Store reference to checkbox
        self.checkboxes[setting_key] = checkbox
    
    def save_settings(self):
        """Save current settings to file and apply them."""
        self.settings.save()
        self.apply_settings()
        self.close()
    
    def apply_settings(self):
        """Apply the current settings."""
        self.settings_changed.emit()
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        self.settings.reset()
        
        # Update all UI elements
        for key, (slider, value_label) in self.sliders.items():
            new_value = self.settings.get(key)
            slider.setValue(int(new_value * 100))
            value_label.setText(f"{new_value:.2f}")
        
        for key, checkbox in self.checkboxes.items():
            new_value = self.settings.get(key)
            checkbox.setChecked(new_value)