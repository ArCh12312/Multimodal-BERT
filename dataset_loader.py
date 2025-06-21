import os
import pandas as pd

class IEMOCAPLoader:
    def __init__(self, transcript_dir, audio_dir, video_dir, label_file):
        self.transcript_dir = transcript_dir
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.label_file = label_file
        self.label_df = self._load_scene_emotions()
        self.dataset = []

    def _load_scene_emotions(self):
        df = pd.read_csv(self.label_file)
        return df

    def load_dataset(self):
        self.dataset = []  # Clear existing dataset if called again

        for fname in os.listdir(self.transcript_dir):
            if not fname.endswith(".txt"):
                continue

            scene_id = os.path.splitext(fname)[0]
            transcript_path = os.path.join(self.transcript_dir, fname)
            audio_path = os.path.join(self.audio_dir, scene_id + ".npy")
            video_path = os.path.join(self.video_dir, scene_id + ".npy")

            if not (os.path.exists(audio_path) and os.path.exists(video_path)):
                print(f"Missing audio or video for {scene_id}, skipping.")
                continue

            # Read and clean transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                stripped_lines = [
                    line.split(":", 1)[1].strip() for line in f if ":" in line
                ]
                transcript = "\n".join(stripped_lines)


            # Get emotion distribution for the scene
            emotion_row = self.label_df[self.label_df["scene_id"] == scene_id]
            if emotion_row.empty:
                print(f"No label for {scene_id}, skipping.")
                continue

            # Convert the row to a dictionary of emotion counts (excluding scene_id)
            emotion_counts = emotion_row.drop(columns=["scene_id"]).iloc[0].to_dict()
            
            self.dataset.append({
                "scene_id": scene_id,
                "transcript": transcript,
                "audio_path": audio_path,
                "video_path": video_path,
                "emotion_counts": emotion_counts  # Dictionary of emotion: count
            })

        return self.dataset

    def get_dataset(self):
        if not self.dataset:
            return self.load_dataset()
        return self.dataset
