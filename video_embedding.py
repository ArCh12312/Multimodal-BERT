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
        self.max_frames = max_frames # New parameter for maximum frames
        os.makedirs(output_path, exist_ok=True)

        # Load pre-trained ResNet50 model
        # Using ResNet50_Weights.IMAGENET1K_V1 for consistency with common practices
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get features
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1]) 
        self.resnet50.eval() # Set model to evaluation mode

        # Define an FC layer to ensure the feature dimension is 768
        # ResNet50's last pooling layer outputs 2048 features, so we map it to 768
        self.fc_layer = nn.Linear(2048, 768) 
        self.fc_layer.eval() # Set FC layer to evaluation mode

        # Define preprocessing transformation for input frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # Convert OpenCV BGR image to PIL image
            transforms.Resize((224, 224)), # Resize image to 224x224, as expected by ResNet
            transforms.ToTensor(), # Convert PIL image to PyTorch tensor (HWC to CHW, values to [0,1])
            # Normalize with ImageNet's mean and standard deviation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_frames(self, video_path):
        """Extracts frames from a video at the specified frame rate."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read() # Read a frame
            if not ret: # If no more frames, break the loop
                break
            if frame_count % self.frame_rate == 0: # Check if current frame is at the desired rate
                frames.append(frame) # Add frame to list
            frame_count += 1
            
        cap.release() # Release the video capture object
        return frames
    
    def extract_features(self, video_path):
        """
        Extracts features from video frames using ResNet50 and FC layer,
        and then truncates or zero-pads the features to max_frames.
        """
        raw_frames = self.extract_frames(video_path)
        extracted_features = []
        
        for frame in raw_frames:
            # Preprocess frame
            frame = self.transform(frame) 
            # Add batch dimension (1, C, H, W) for model input
            frame = frame.unsqueeze(0) 
            with torch.no_grad(): # Disable gradient calculation for inference
                feature = self.resnet50(frame) # Extract ResNet50 features
                feature = feature.view(feature.size(0), -1) # Flatten the feature tensor
                feature = self.fc_layer(feature) # Apply FC layer to get 768-dim feature
            # Append numpy array of features, remove batch dim with .squeeze()
            extracted_features.append(feature.squeeze().numpy())
        
        # Convert list of features to a NumPy array
        features_array = np.array(extracted_features)

        # --- Apply Truncation or Zero-Padding ---
        num_current_frames = features_array.shape[0]

        if num_current_frames > self.max_frames:
            # Truncate if there are more frames than max_frames
            final_features = features_array[:self.max_frames]
            print(f"Truncated {video_path}: {num_current_frames} frames to {self.max_frames} frames.")
        elif num_current_frames < self.max_frames:
            # Zero-pad if there are fewer frames than max_frames
            padding_needed = self.max_frames - num_current_frames
            # Create a padding vector of zeros with the feature dimension (768)
            padding_vector = np.zeros((padding_needed, 768), dtype=features_array.dtype)
            final_features = np.vstack((features_array, padding_vector)) # Vertically stack features and padding
            print(f"Padded {video_path}: {num_current_frames} frames to {self.max_frames} frames.")
        else:
            # No change needed if already at max_frames
            final_features = features_array
            print(f"Processed {video_path}: {num_current_frames} frames (no truncation/padding).")

        return final_features
    
    def process_videos(self):
        """Processes all videos in the dataset and saves extracted features."""
        # Adjust file extension if needed, currently set to .avi
        video_files = glob(os.path.join(self.data_path, "*.avi")) 
        
        if not video_files:
            print(f"No .avi video files found in {self.data_path}. Please check the path and extension.")
            return

        for video in video_files:
            video_name = os.path.basename(video).split('.')[0] # Get video name without extension
            print(f"Processing {video_name}...")
            try:
                features = self.extract_features(video)
                # Save the extracted features as a NumPy .npy file
                np.save(os.path.join(self.output_path, f"{video_name}.npy"), features)
                print(f"Saved features for {video_name} to {self.output_path}/{video_name}.npy")
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
        
        print("Feature extraction complete!")

def main():
    # Set your dataset path here. Make sure it points to a directory containing .avi files.
    data_path = "C:/Users/aryan/Documents/Study/Research/IEMOCAP_full_release/Session1/dialog/avi/DivX" 
    # Set the output directory where you want to save the extracted features.
    output_path = "Video_output" 
    # Initialize the extractor with the specified paths and the new max_frames parameter
    extractor = ResNet50FeatureExtractor(data_path, output_path, max_frames=1024)
    extractor.process_videos()

if __name__ == "__main__":
    main()