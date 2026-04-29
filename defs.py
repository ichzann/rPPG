
import cv2
from IPython.display import display, Image, clear_output
import time


def play_avi(path):
    video_path = "1020_USBVideo_after.avi"
    cap = cv2.VideoCapture(path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert the OpenCV frame (BGR) to a JPEG format that the notebook can read
            ret, buffer = cv2.imencode('.jpg', frame)
            
            # Display the image inline
            display(Image(data=buffer.tobytes()))
            
            # Clear the previous frame immediately to create the illusion of video
            clear_output(wait=True)
            
            # Optional: Add a tiny sleep to control playback speed (e.g., 0.03 = ~30fps)
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        # Allows you to stop playback gracefully by interrupting the notebook kernel (the Stop button)
        print("Video stopped.")

    finally:
        cap.release()