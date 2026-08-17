#使用 模型對連續影像進行描述
import cv2
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np

class VideoCaptioning:
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_caption(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 圖片預處理
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # 生成描述
        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_length=32, num_beams=8, early_stopping=True)

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def caption_video(self, video_path, interval_seconds=1.0):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                raise ValueError("Could not determine FPS of video.")
            
            frame_interval = int(fps * interval_seconds)
            print(f"Video FPS: {fps:.2f}, Capturing 1 frame every {interval_seconds} seconds (~{frame_interval} frames)")

            frame_count = 0
            captions = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    caption = self.generate_caption(frame)
                    captions.append((frame_count, caption))
                    print(f"Frame {frame_count}: {caption}")

                frame_count += 1

            cap.release()
            return captions
    
if __name__ == "__main__":
    video_path = "videoplayback2.mp4"
    vc = VideoCaptioning()
    vc.caption_video(video_path, interval_seconds=1.0)