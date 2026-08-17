# 取影片一偵偵 存為圖片
import cv2
import os

def capture_video_frames(video_path, output_folder, interval=1):
    """
    Capture frames from a video file and save them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder where the captured images will be saved.
    - interval: Interval in seconds between captured frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


capture_video_frames("videoplayback2.mp4", "img", interval=1)# 取影片一偵偵 存為圖片