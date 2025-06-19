import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from glob import glob

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from glob import glob

class ResNet50FeatureExtractor:
    def __init__(self, data_path, output_path, frame_rate=5, max_frames=1024):
        self.data_path = data_path
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        os.makedirs(output_path, exist_ok=True)

        # Device configuration: use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pre-trained ResNet50 model and remove last classification layer
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()
        self.resnet50.to(self.device)  # Move model to device

        # FC layer to map 2048-dim feature to 768-dim feature
        self.fc_layer = nn.Linear(2048, 768)
        self.fc_layer.eval()
        self.fc_layer.to(self.device)  # Move FC layer to device

        # Preprocessing transform pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
        """Extracts uniformly sampled features from video and maps them to 768-dim."""
        raw_frames = self.extract_frames(video_path)
        total_frames = len(raw_frames)

        if total_frames == 0:
            print(f"No frames extracted from {video_path}.")
            return np.zeros((self.max_frames, 768), dtype=np.float32)

        if total_frames >= self.max_frames:
            # Uniformly sample 256 frames
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            sampled_frames = [raw_frames[i] for i in indices]
            print(f"Sampled {self.max_frames} frames from {total_frames} total frames.")
        else:
            # Use all frames and pad
            sampled_frames = raw_frames
            print(f"Padded {video_path}: {total_frames} frames to {self.max_frames} frames.")

        extracted_features = []

        for frame in sampled_frames:
            frame = self.transform(frame)
            frame = frame.unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.resnet50(frame)
                feature = feature.view(feature.size(0), -1)
                feature = self.fc_layer(feature)

            extracted_features.append(feature.squeeze().cpu().numpy())

        features_array = np.array(extracted_features)

        # If padding is needed
        if features_array.shape[0] < self.max_frames:
            padding = np.zeros((self.max_frames - features_array.shape[0], 768), dtype=features_array.dtype)
            features_array = np.vstack((features_array, padding))

        return features_array


    def process_videos(self):
        """Processes all .avi videos in the dataset directory."""
        video_files = glob(os.path.join(self.data_path, "*.avi"))
        
        if not video_files:
            print(f"No .avi video files found in {self.data_path}. Please check the path and extension.")
            return

        for video in video_files:
            video_name = os.path.basename(video).split('.')[0]
            print(f"Processing {video_name}...")
            try:
                features = self.extract_features(video)
                np.save(os.path.join(self.output_path, f"{video_name}.npy"), features)
                print(f"Saved features for {video_name} to {self.output_path}/{video_name}.npy")
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
        
        print("Feature extraction complete!")


def main():
    base_data_path = "C:/Users/aryan/Documents/Study/Research/IEMOCAP_full_release"
    base_output_path = "Video_output"
    max_frames = 256

    for session_num in range(1, 6):
        data_path = f"{base_data_path}/Session{session_num}/dialog/avi/DivX"
        output_path = f"{base_output_path}/Session{session_num}"
        
        extractor = ResNet50FeatureExtractor(data_path, output_path, max_frames=max_frames)
        extractor.process_videos()

if __name__ == "__main__":
    main()