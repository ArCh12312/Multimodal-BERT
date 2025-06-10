import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from glob import glob

class ResNet50FeatureExtractor:
    def __init__(self, data_path, output_path, frame_rate=5):
        self.data_path = data_path
        self.output_path = output_path
        self.frame_rate = frame_rate
        os.makedirs(output_path, exist_ok=True)

        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])  # Remove final classification layer
        self.resnet50.eval()

        # Define an FC layer to ensure 2048 feature dimension
        self.fc_layer = nn.Linear(2048, 768)  # Ensures compatibility with BERT
        self.fc_layer.eval()

        # Define preprocessing transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_frames(self, video_path):
        """Extracts frames from a video at the specified frame rate."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_rate == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def extract_features(self, video_path):
        """Extracts features from video frames using ResNet50 and FC layer."""
        frames = self.extract_frames(video_path)
        features = []
        
        for frame in frames:
            frame = self.transform(frame)  # Preprocess frame
            frame = frame.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                feature = self.resnet50(frame)  # Extract ResNet50 features
                feature = feature.view(feature.size(0), -1)  # Flatten
                feature = self.fc_layer(feature)  # Apply FC layer
            features.append(feature.squeeze().numpy())
        
        return np.array(features)
    
    def process_videos(self):
        """Processes all videos in the dataset and saves extracted features."""
        video_files = glob(os.path.join(self.data_path, "*.avi"))  # Adjust file extension if needed
        
        for video in video_files:
            video_name = os.path.basename(video).split('.')[0]
            print(f"Processing {video_name}...")
            features = self.extract_features(video)
            np.save(os.path.join(self.output_path, f"{video_name}.npy"), features)
        
        print("Feature extraction complete!")

def main():
    data_path = "C:/Users/aryan/Documents/Study/Research/IEMOCAP_full_release/Session1/dialog/avi/DivX"  # Change this to your dataset path
    output_path = "Video_output"  # Change this to where you want to save features
    extractor = ResNet50FeatureExtractor(data_path, output_path)
    extractor.process_videos()

if __name__ == "__main__":
    main()