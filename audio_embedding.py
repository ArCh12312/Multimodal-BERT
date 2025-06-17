import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from glob import glob
from torchvggish import vggish, vggish_input

class VGGishFeatureExtractor:
    def __init__(self, output_dim=768):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.model = vggish()
        self.model.to(self.device)
        self.model.eval()
        self.projection_layer = nn.Linear(128, output_dim)
        self.projection_layer.to(self.device) # Move projection layer to device
        self.projection_layer.eval()
        
    def extract_features(self, audio_path, target_length=256):
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16 kHz mono
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        waveform_np = waveform.squeeze(0).cpu().numpy()

        input_batch = vggish_input.waveform_to_examples(waveform_np, sample_rate=16000)
        input_tensor = input_batch.float().to(self.device)

        with torch.no_grad():
            embeddings = self.model(input_tensor)
            embeddings = self.projection_layer(embeddings)

        embeddings = embeddings.cpu().numpy()  # Shape: [num_frames, 768]

        # Pad or truncate to fixed length
        num_frames = embeddings.shape[0]
        if num_frames < target_length:
            padding = np.zeros((target_length - num_frames, embeddings.shape[1]), dtype=np.float32)
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = embeddings[:target_length, :]

        return embeddings  # Shape: [target_length, 768]
    
def main():
    data_path = "C:/Users/aryan/Documents/Study/Research/IEMOCAP_full_release/Session1/dialog/wav"  # Change to dataset path
    output_path = "Audio_Output"  # Change output path
    os.makedirs(output_path, exist_ok=True)
    
    extractor = VGGishFeatureExtractor()
    
    audio_files = glob(os.path.join(data_path, "*.wav"))  # Adjust file extension if needed
    
    for i, audio in enumerate(audio_files):
        audio_name = os.path.basename(audio).split('.')[0]
        print(f"Processing {audio_name}...")
        features = extractor.extract_features(audio)

        # Print the shape of the features for the first processed audio file
        if i == 0:
            print(f"Shape of features for {audio_name}: {features.shape}")

        np.save(os.path.join(output_path, f"{audio_name}.npy"), features)
    
    print("Audio feature extraction complete!")

if __name__ == "__main__":
    main()
