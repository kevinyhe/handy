import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

import threading
from queue import Queue, Empty
import os

from src.vision.camera import CameraManager
from src.tracking.hand_tracker import HandTracker
from src.tracking.pointer import PointerTracker
from src.tracking.gestures import GestureDetector
from src.control.mouse_controller import MouseController
from src.settings.settings import Settings
from src.settings.menu import SettingsMenu 

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        
        # Initialize settings
        self.settings = Settings()
        
        # Initialize camera and hand tracker
        self.camera_manager = CameraManager()
        self.hand_tracker = HandTracker(
            min_detection_confidence=self.settings.get('hand_detection_confidence'),
            min_tracking_confidence=self.settings.get('hand_tracking_confidence')
        )
        
        # Initialize finger pointer tracker
        self.finger_tracker = PointerTracker()
        
        # Initialize gesture detector
        self.gesture_detector = GestureDetector()
        
        # Apply initial settings to gesture detector
        self.gesture_detector.config['left_click']['threshold'] = self.settings.get('left_click_threshold')
        self.gesture_detector.config['right_click']['threshold'] = self.settings.get('right_click_threshold')
        self.gesture_detector.config['move']['threshold'] = self.settings.get('move_threshold')
        self.gesture_detector.config['drag']['threshold'] = self.settings.get('drag_threshold')
        self.gesture_detector.config['scroll']['threshold'] = self.settings.get('scroll_threshold')
        
        # Initialize mouse controller with settings
        self.mouse_controller = MouseController(
            sensitivity=self.settings.get('mouse_sensitivity'),
            smoothing=self.settings.get('mouse_smoothing')
        )
        self.mouse_controller.dead_zone = self.settings.get('dead_zone')
        
        # Initialize settings menu
        self.settings_menu = SettingsMenu(self.settings)
        
        # Set up UI
        self.setup_ui()
        
        # Set up timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Add frame processing queue
        self.frame_queue = Queue(maxsize=1)
        self.processed_frame_queue = Queue(maxsize=1)
        self.processing_thread = None
        self.is_processing = False
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Hand Tracking App")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)

        # Controls layout definition
        controls_layout = QHBoxLayout()

        self.status_label = QLabel("Status: No hand detected")
        controls_layout.addWidget(self.status_label)

        # Camera display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.image_label)
        
        # Controls

        # Camera selection
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        self.camera_combo = QComboBox()
        
        # Populate camera list
        available_cameras = self.camera_manager.get_available_cameras()
        for cam_id in available_cameras:
            self.camera_combo.addItem(f"Camera {cam_id}", cam_id)
            
        if not available_cameras:
            self.camera_combo.addItem("No cameras found", -1)
            
            
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        controls_layout.addLayout(camera_layout)
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.start_stop_button)
        
        main_layout.addLayout(controls_layout)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        controls_layout.addWidget(self.settings_button)
        
        main_layout.addLayout(controls_layout)
        
    def open_settings(self):
        """Open the settings menu dialog."""
        # Create settings menu dialog
        settings_dialog = self.settings_menu
        
        # Connect the settings_changed signal to update components
        settings_dialog.settings_changed.connect(self.apply_settings)
        
        # Show the dialog
        settings_dialog.exec_()

    def apply_settings(self):
        """Apply current settings to all components."""
        # Update hand tracker
        self.hand_tracker.min_detection_confidence = self.settings.get("hand_detection_confidence")
        self.hand_tracker.min_tracking_confidence = self.settings.get("hand_tracking_confidence")
        
        # Update gesture detector thresholds
        self.gesture_detector.config['left_click']['threshold'] = self.settings.get('left_click_threshold')
        self.gesture_detector.config['right_click']['threshold'] = self.settings.get('right_click_threshold')
        self.gesture_detector.config['move']['threshold'] = self.settings.get('move_threshold')
        self.gesture_detector.config['drag']['threshold'] = self.settings.get('drag_threshold')
        
        # Update scroll settings
        self.gesture_detector.config['scroll']['threshold'] = self.settings.get('scroll_threshold')
        self.gesture_detector.config['scroll']['base_speed'] = self.settings.get('scroll_base_speed')
        self.gesture_detector.config['scroll']['max_speed'] = self.settings.get('scroll_max_speed')
        self.gesture_detector.config['scroll']['straightness_factor'] = self.settings.get('scroll_straightness_factor')
        
        # Update mouse controller
        self.mouse_controller.sensitivity = self.settings.get('mouse_sensitivity')
        self.mouse_controller.smoothing = self.settings.get('mouse_smoothing')
        self.mouse_controller.dead_zone = self.settings.get('dead_zone')
    
    print("Settings applied")
            
    def change_camera(self, index):
        """Change the active camera."""
        if self.timer.isActive():
            self.toggle_camera()  # Stop current camera
            self.toggle_camera()  # Start new camera
                    
    def toggle_camera(self):
        """Start or stop the camera feed."""
        if self.timer.isActive():
            # Stop camera
            self.timer.stop()
            
            # Stop processing thread
            self.is_processing = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
                
            self.camera_manager.release()
            self.start_stop_button.setText("Start")
        else:
            # Update status
            self.status_label.setText("Starting, please wait...")
            QApplication.processEvents()  # Force UI update
            
            # Start camera
            camera_id = self.camera_combo.currentData()
            if camera_id >= 0:
                self.camera_manager.camera_id = camera_id
                if self.camera_manager.initialize():
                    # Start processing thread
                    self.is_processing = True
                    self.processing_thread = threading.Thread(target=self.process_frames)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    
                    # Start the timer
                    self.timer.start(30)  # ~33 FPS
                    self.start_stop_button.setText("Stop")
                    self.status_label.setText("Camera active")
                else:
                    self.status_label.setText("Failed to initialize camera")
            else:
                self.status_label.setText("No camera selected")
    
    def process_frames(self):
        """Process frames in a separate thread."""
        while self.is_processing:
            try:
                # Get frame from queue with a timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame with hand tracker
                processed_frame, results, fingertips_list = self.hand_tracker.find_hands(frame, track_fingertips=True)
                
                # Process gestures if hands detected
                if fingertips_list:
                    # Update tracker with the first hand's fingertips
                    fingertips = fingertips_list[0]
                    self.finger_tracker.update_from_fingertips(fingertips)
                    
                    palm_size = self.hand_tracker.get_palm_size(frame, results.multi_hand_landmarks[0])
                    
                    # print palm size
                    print(f"Palm size: {palm_size:.2f} pixels")

                    # Detect gestures with dedicated detector
                    gestures = self.gesture_detector.detect_gestures(fingertips, palm_size)

                    # Update mouse with gestures and palm size
                    self.mouse_controller.update_mouse(self.finger_tracker, gestures, palm_size)

                    # Set pointer active if 'move' gesture or drag is detected
                    if 'move' in gestures or ('drag' in gestures and 
                                            ('left_click' in gestures or 'right_click' in gestures)):
                        self.finger_tracker.set_pointer_active(True)
                    else:
                        self.finger_tracker.set_pointer_active(False)
                    
                    # Add gesture visualization to frame
                    processed_frame = self.gesture_detector.draw_gesture_feedback(
                        processed_frame, fingertips, gestures)
                    
                    # Store gestures for UI updates
                    self.current_gestures = gestures
                else:
                    self.current_gestures = {}
                    self.finger_tracker.set_pointer_active(False)  # No hand detected, disable pointer
                
                # Put processed frame in output queue, replacing any existing frame
                if not self.processed_frame_queue.empty():
                    try:
                        self.processed_frame_queue.get_nowait()
                    except Empty:
                        pass
                self.processed_frame_queue.put(processed_frame)
                
            except Empty:
                # No new frame available, continue
                pass
            except Exception as e:
                print(f"Error in frame processing thread: {e}")
    
    @pyqtSlot()
    def update_frame(self):
        """Update the video frame in the UI."""
        frame = self.camera_manager.get_frame()
        
        if frame is not None:
            # Get frame dimensions
            h, w, ch = frame.shape
            self.finger_tracker.set_dimensions((w, h))
            
            # Put frame in processing queue, replacing any existing frame
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    # Remove old frame
                    self.frame_queue.get_nowait()
                    # Add new frame
                    self.frame_queue.put(frame)
                except Empty:
                    pass
            
            # Check if we have a processed frame to display
            try:
                if not self.processed_frame_queue.empty():
                    display_frame = self.processed_frame_queue.get_nowait()
                else:
                    # If no processed frame is available, use the raw frame
                    display_frame = frame
            except Empty:
                display_frame = frame
            
            # Convert frame to QImage for display
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Show frame
            self.image_label.setPixmap(pixmap)
            
            # Update UI based on current gestures if available
            if hasattr(self, 'current_gestures'):
                # Collect all active gestures for status display
                active_gestures = []
                
                if 'left_click' in self.current_gestures:
                    active_gestures.append(f"Left click ({self.current_gestures['left_click']:.2f})")
                
                if 'right_click' in self.current_gestures:
                    active_gestures.append(f"Right click ({self.current_gestures['right_click']:.2f})")
                    
                if 'move' in self.current_gestures:
                    active_gestures.append(f"Move ({self.current_gestures['move']:.2f})")
                
                if active_gestures:
                    self.status_label.setText(f"Status: {', '.join(active_gestures)}")
                elif self.current_gestures != {}:
                    self.status_label.setText("Status: Hand detected")
                else:
                    self.status_label.setText("Status: No hand detected")
        
    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        self.timer.stop()
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if hasattr(self, 'gesture_detector'):
            self.gesture_detector.shutdown()  # Clean shutdown of thread pool
        self.camera_manager.release()
        event.accept()
            
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()