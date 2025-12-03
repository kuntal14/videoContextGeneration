import shutil
import frameExtractor as fe 
import os

keyframes_folder_path = fe.keyframes_folder_path

def get_frames():
    # check if the folder exists, if so delete it
    if os.path.exists('keyframes'):
        shutil.rmtree(keyframes_folder_path)

    # # extract keyframes
    fe.execute()

if __name__ == "__main__":
    get_frames()

