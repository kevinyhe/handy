import cv2

class CameraManager:
    """Manages camera access and frame acquisition."""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
    def initialize(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Try to set reasonable camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return self.cap.isOpened()
    
    def get_frame(self):
        """Get the next frame from the camera."""
        if not self.is_initialized():
            return None
        
        # # Skip any buffered frames to get the most recent one
        # for _ in range(2):  # Skip up to 2 buffered frames
        #     self.cap.grab()
        
        # Read the final frame
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Flip horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        return frame
    
    def is_initialized(self):
        """Check if the camera is initialized."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_available_cameras(self):
        """Get a list of available camera indices."""
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                available_cameras.append(i)
                temp_cap.release()
        return available_cameras