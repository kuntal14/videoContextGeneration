from pathlib import Path
import subprocess
import shutil
import os

AUDIO_FOLDER = "audio"
AUDIO_PATH = f"{AUDIO_FOLDER}/transcript_16k.wav"
CSV_PATH = f"{AUDIO_FOLDER}/transcript_16k_word_ts.csv"
SRT_PATH = f"{AUDIO_FOLDER}/transcript_16k_word_ts.srt"

def demux_audio(video_path):
    # check if the audio folder exists, if so delete it
    if os.path.exists(AUDIO_FOLDER):
        shutil.rmtree(AUDIO_FOLDER)
    
    # add the audio folder
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    # demux using subprocess
    try:
        """Demux the audio from the video file."""
        subprocess.run(["ffmpeg", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(AUDIO_PATH)])
        print(f"Audio demuxed and saved to {AUDIO_PATH}")
    except Exception as e:
        print(f"Error demuxing audio: {e}")

if __name__ == "__main__":
    video_path = Path.home()/ "Downloads" / "demo.mov"
    demux_audio(video_path)
