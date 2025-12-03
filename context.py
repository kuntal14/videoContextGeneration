from time import time
import shutil
import frameExtractor as fe 
import transcription as tr
import os
import subprocess
from pathlib import Path

keyframes_folder_path = fe.keyframes_folder_path

def get_frames():
    # check if the folder exists, if so delete it
    if os.path.exists('keyframes'):
        shutil.rmtree(keyframes_folder_path)

    # # extract keyframes
    fe.execute()

def get_transcript(video_path):
    # check if the folder exists, if so delete it
    if os.path.exists('audio'):
        shutil.rmtree('audio')

    # transcribe audio
    tr.execute(video_path)    

if __name__ == "__main__":
    video_path = Path.home()/ "Downloads" / "demo.mov"
    t0 = time()
    # get_frames()
    # get_transcript(video_path)
    # 46.88424015045166 -> time for synchronous exec
    subprocess.run(["venv_311/bin/python3", "transcription.py"])
    subprocess.run(["venv_311/bin/python3", "frameExtractor.py"])
    t1 = time()
    print(t1 - t0)


