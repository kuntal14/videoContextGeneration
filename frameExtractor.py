import subprocess
import json
import os
from pathlib import Path
import cv2

# Video Path
video_file = "demo.mov"
keyframes_folder_path = 'keyframes'
video_path = "/Users/kuntalsuman/Downloads/demo.mov"
keyframes_path = f'keyframes/{video_file}_keyframes.json'
all_frames_path = 'keyframes/all_frames.json'
images_path = 'keyframes/images'

# Verify the file exists
def verify_video_path(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Video file: {video_path}")
    print(f"File size: {os.path.getsize(video_path):,} bytes")

# this will give you the keyframes and their data
def extract_keyframe_offsets(video_path):
    """
    Extract keyframe information including byte offsets from a video file.
    
    Returns:
        list: List of dictionaries containing keyframe metadata
    """
    # Use ffprobe to get packet information
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',  # Select first video stream
        '-show_entries', 'packet=pts_time,pos,flags,size',  # Get timestamp, position, flags, and size
        '-of', 'json',  # Output as JSON
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        keyframes = []
        frame_num = 0
        
        for packet in data.get('packets', []):
            # Check if this is a keyframe (I-frame)
            # The 'K' flag indicates a keyframe
            if 'K' in packet.get('flags', ''):
                keyframe_info = {
                    'frame_number': frame_num,
                    'byte_offset': int(packet.get('pos', -1)),
                    'pts_time': float(packet.get('pts_time', 0)),
                    'packet_size': int(packet.get('size', 0))
                }
                keyframes.append(keyframe_info)
            frame_num += 1
        
        return keyframes
    
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        raise

# check if there is a keyframes folder, if not, the create one

def save_keyframes(keyframes):
    if not os.path.exists('keyframes'):
        os.makedirs('keyframes')

        # check if it has the keyframes file for this video file
        if os.path.exists(keyframes_path):
            print(f"Keyframes file already exists for {video_file}")
            with open(keyframes_path, 'w') as f:
                json.dump(keyframes, f)
        else:
            with open(keyframes_path, 'w') as f:
                json.dump(keyframes, f)
    else:
        if os.path.exists(f'keyframes/{video_file}_keyframes.json'):
            print(f"Keyframes file already exists for {video_file}")
            with open(keyframes_path, 'w') as f:
                json.dump(keyframes, f)
        else:
            with open(keyframes_path, 'w') as f:
                json.dump(keyframes, f)

# we need 2 more intermediate frames in between the keyframes for more context
def get_all_frames(keyframes):
    all_frames = []
    for i in range(len(keyframes)):
        if i == 0:
            all_frames.append(f"{keyframes[i]['pts_time']:.2f}")
        elif i == len(keyframes) - 1:
            # find the duration of the video
            t0 = keyframes[i]['pts_time']
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            t1 = float(subprocess.check_output(cmd).decode('utf-8')) - 0.1
            t_inter_1 = t0 + (t1 - t0) / 3
            t_inter_2 = t0 + (2 * (t1 - t0)) / 3
            t_inter_2 = t_inter_2
            all_frames.append(f"{t0:.2f}")
            all_frames.append(f"{t_inter_1:.2f}")
            all_frames.append(f"{t_inter_2:.2f}")
            all_frames.append(f"{t1:.2f}")
        else:
            t0 = keyframes[i]['pts_time']
            t1 = keyframes[i+1]['pts_time']
            t_inter_1 = t0 + (t1 - t0) / 3
            t_inter_2 = t0 + (2 * (t1 - t0)) / 3
            all_frames.append(f"{t0:.2f}")
            all_frames.append(f"{t_inter_1:.2f}")
            all_frames.append(f"{(t_inter_2):.2f}")
    # save it on the file named all_frames.json
    print('about to add all_frames')
    with open(all_frames_path, 'w') as f:
        json.dump(all_frames, f)
        print("saved all frames")

# make images folder
def make_images_folder():
    if not os.path.exists('keyframes/images'):
        os.makedirs('keyframes/images')
        print("made the images folder")
    else:
        print("images folder already exists")

# extract images
def extract_frames():
    # Open the video file once
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    with open(all_frames_path, 'r') as f:
        all_frames = json.load(f)
        
        for time_str in all_frames:
            try:
                # Convert time string to float seconds, then to milliseconds
                time_sec = float(time_str)
                time_msec = time_sec * 1000.0
                
                # Seek to the specific time
                cap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
                
                # Read the frame
                ret, frame = cap.read()
                
                if ret:
                    output_filename = f'{images_path}/{time_str}.jpg'
                    cv2.imwrite(output_filename, frame)
                    # print(f"Saved {output_filename}") # Optional logging
                else:
                    print(f"Warning: Could not read frame at {time_str}s")
            except ValueError as e:
                print(f"Error processing time {time_str}: {e}")

    cap.release()
    print("Frame extraction complete.")


def execute():
    verify_video_path(video_path)
    keyframes = extract_keyframe_offsets(video_path)
    save_keyframes(keyframes)
    get_all_frames(keyframes)
    make_images_folder()
    extract_frames()

if __name__ == "__main__":
    execute()