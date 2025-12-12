import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Optional
from wrapper.camera_wrapper.base import CameraWrapper


class RealSenseWrapper(CameraWrapper):
    """
    Intel RealSense Camera Wrapper implementation.
    Reads the Color stream by default to comply with the single-array return of the base class.
    """
    def __init__(self, width: int = 256, height: int = 256, fps: int = 30):
        super().__init__()

        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = None
        self.config = None
        self.profile = None
        
        self._is_active = False

    def open(self) -> None:
        """Initialize and start the RealSense pipeline."""
        if self._is_active:
            print("[RealSenseWrapper] Camera is already active.")
            return

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            self.config.enable_stream(
                rs.stream.color, 
                self.width, 
                self.height, 
                rs.format.bgr8, 
                self.fps
            )
            
            self.profile = self.pipeline.start(self.config)
            self._is_active = True
            
            print("[RealSenseWrapper] Warming up camera...")
            for _ in range(10):
                self.pipeline.wait_for_frames()
                
            print(f"[RealSenseWrapper] Camera opened: {self.width}x{self.height} @ {self.fps}fps")

        except Exception as e:
            print(f"[RealSenseWrapper] Error opening camera: {e}")
            self._is_active = False
            if self.pipeline:
                self.pipeline = None

    def close(self) -> None:
        """Stop the pipeline and release resources."""
        if self._is_active and self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"[RealSenseWrapper] Error stopping pipeline: {e}")
            finally:
                self._is_active = False
                self.pipeline = None
                print("[RealSenseWrapper] Camera closed.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Wait for a coherent pair of frames and return the color frame.
        """
        if not self._is_active or not self.pipeline:
            return False, None

        try:
            frames = self.pipeline.wait_for_frames()
            
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return False, None

            image = np.asanyarray(color_frame.get_data())
            
            return True, image

        except RuntimeError as e:
            print(f"[RealSenseWrapper] Runtime error during read: {e}")
            return False, None
        except Exception as e:
            print(f"[RealSenseWrapper] Unexpected error: {e}")
            return False, None

    def is_opened(self) -> bool:
        """Check if the pipeline is active."""
        return self._is_active


if __name__ == "__main__":
    cam = RealSenseWrapper(width=640, height=480, fps=30)
    
    try:
        cam.open()

        if not cam.is_opened():
            print("Failed to open camera. Exiting.")
            exit(1)

        print("Press 'q' to quit the video stream.")
        
        while True:
            ret, frame = cam.read()

            if ret and frame is not None:
                cv2.imshow('RealSense Camera Test', frame)
            else:
                print("Failed to grab frame.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    finally:
        cam.close()
        cv2.destroyAllWindows()