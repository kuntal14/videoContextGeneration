from time import time
import shutil
import frameExtractor as fe 
import transcription as tr
import caption_images as ci
import os
from pathlib import Path

keyframes_folder_path = fe.frames_folder_path

def get_frames():
    # extract keyframes
    fe.execute()

def get_transcript(video_path):
    # check if the folder exists, if so delete it
    if os.path.exists('audio'):
        shutil.rmtree('audio')

    # transcribe audio
    tr.execute(video_path)    

def caption_images(video_file, caption_path):
    ci.execute(video_file, caption_path)

if __name__ == "__main__":
    video_file = "demo.mp4"
    video_path = Path.home()/ "Downloads" / "demo.mp4"
    t0 = time()
    get_frames()
    get_transcript(video_path)
    t1 = time()
    print(t1 - t0)
    # make a folder for the image caption
    caption_path = Path.home()/"context"/f"video_file"/"images_caption"
    if not os.path.exists(caption_path):
        os.makedirs(caption_path)
    caption_images(video_file, caption_path)
    
        



