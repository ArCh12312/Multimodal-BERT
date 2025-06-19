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
        
    def extract_features(self, audio_path, target_length=128):
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16 kHz mono
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform_np = waveform.squeeze(0).cpu().numpy()

        input_batch = vggish_input.waveform_to_examples(waveform_np, sample_rate=16000)
        input_tensor = input_batch.float().to(self.device)

        with torch.no_grad():
            embeddings = self.model(input_tensor)  # [num_frames, 128]
            embeddings = self.projection_layer(embeddings)  # [num_frames, 768]

        embeddings = embeddings.cpu().numpy()

        num_frames = embeddings.shape[0]

        if num_frames == 0:
            print(f"Warning: No features extracted from {audio_path}. Returning zeros.")
            return np.zeros((target_length, 768), dtype=np.float32)

        if num_frames >= target_length:
            indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
            sampled_embeddings = embeddings[indices]
            print(f"Sampled {target_length} features from {num_frames} frames.")
        else:
            padding = np.zeros((target_length - num_frames, 768), dtype=np.float32)
            sampled_embeddings = np.vstack([embeddings, padding])
            print(f"Padded {num_frames} features to {target_length}.")

        return sampled_embeddings
    
def main():
    base_data_path = "C:/Users/aryan/Documents/Study/Research/IEMOCAP_full_release"
    base_output_path = "Audio_Output"
    
    extractor = VGGishFeatureExtractor()
    
    for session_num in range(1, 6):
        data_path = os.path.join(base_data_path, f"Session{session_num}", "dialog", "wav")
        output_path = os.path.join(base_output_path, f"Session{session_num}")
        os.makedirs(output_path, exist_ok=True)
        
        audio_files = glob(os.path.join(data_path, "*.wav"))
        
        for i, audio in enumerate(audio_files):
            audio_name = os.path.basename(audio).split('.')[0]
            print(f"Processing {audio_name} in Session{session_num}...")
            features = extractor.extract_features(audio)

            if i == 0:
                print(f"Shape of features for {audio_name}: {features.shape}")

            np.save(os.path.join(output_path, f"{audio_name}.npy"), features)
    
    print("Audio feature extraction for all sessions complete!")

if __name__ == "__main__":
    main()
