import ollama
import json
import os
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import global_context_builder as gcb

# Define paths
HOME = Path.home()
CONTEXT_FOLDER_PATH = f'{HOME}/context'

# --- NEW: Load transcript once into memory ---
def load_transcript_cache(video_file):
    """Loads the entire transcript into a list of dicts for fast access."""
    csv_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/audio/transcript_16k_word_ts.csv'
    if not os.path.exists(csv_path):
        return None
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None

def get_transcript_from_cache(transcript_data, timestamp, window=3.0):
    """Fast lookup from memory instead of disk."""
    if not transcript_data:
        return "no transcription available"

    words = []
    # Optimization: This is still linear search but generally fast enough for memory.
    # For very long videos, you could optimize further with bisect, but this is fine for now.
    for r in transcript_data:
        try:
            start = float(r['start_sec'])
            if abs(start - timestamp) <= window:
                words.append(r['word'])
        except ValueError:
            continue
    
    return " ".join(words) if words else "silence"

def format_global_context_for_prompt(global_context):
    prompt = "### GLOBAL VIDEO CONTEXT ###\n"
    prompt += f"Summary: {global_context.get('summary', 'N/A')}\n"
    prompt += f"Video Style: {global_context.get('narrative_style', 'unknown')}\n"
    
    people = global_context.get('entities', {}).get('people', [])
    if people:
        prompt += "Key People:\n"
        for p in people:
            prompt += f"- {p.get('name')} ({p.get('role')}): {p.get('description')}\n"
    
    prompt += "###########################\n"
    return prompt

# --- CHANGED: Accept transcript_cache as an argument ---
def process_single_image(image_file, video_file, caption_path, global_context_text, transcript_cache):
    images_folder_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'
    image_path = os.path.abspath(os.path.join(images_folder_path, image_file))
    
    if not os.path.exists(image_path):
        return f"❌ Missing: {image_file}"
    
    try:
        timestamp = float(Path(image_file).stem)
        # Use the memory cache function
        transcript = get_transcript_from_cache(transcript_cache, timestamp)
        
        prompt = f"""You are analyzing a video frame with accompanying audio context.
### GLOBAL VIDEO CONTEXT (Use for identification)
{global_context_text}

### LOCAL CONTEXT
Timestamp: {timestamp:.2f}s
Audio/Transcript: "{transcript}"

TASK: Provide a cohesive description of this moment.
INSTRUCTIONS:
1. VISUAL: Describe the image in detail (scenery, objects, atmosphere).
2. AUDIO: If the transcript shows someone speaking, identify them using Global Context.
3. SYNTHESIS: If the speaker is NOT in the frame, describe them as "speaking off-camera" or "narrating over the scene".
4. IDENTITY: Do not just say "a man"; use names from Global Context if they match the transcript or appearance.
5. DONT's : do not requote the trasncript that is being sent to you.

Return ONLY this JSON:
{{
  "description": "A paragraph describing the visual scene AND how the audio/transcript relates to it ",
  "entities": ["names of visible people", "identified off-camera speakers", "key objects"],
  "actions": ["visual actions", "speech or narration"]
}}
"""
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Instantiate a client explicitly (good practice for threading)
        client = ollama.Client()
        
        resp = client.chat(
            model="gemma3:4b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_bytes]
            }],
            format="json",
            options={"temperature": 0.1, "num_ctx": 4096}
        )

        content = resp["message"]["content"]
        # Parse JSON string to dictionary before modifying
        content_dict = json.loads(content) if isinstance(content, str) else content
        content_dict["transcript"] = transcript
        save_caption(image_file, caption_path, content_dict)
        return f"✅ {image_file}"

    except Exception as e:
        error_entry = {
            "description": f"Processing error: {str(e)}",
            "entities": [],
            "actions": [],
            "error": True
        }
        save_caption(image_file, caption_path, json.dumps(error_entry))
        return f"❌ {image_file}: {str(e)}"

def save_caption(image_file, caption_path, caption):
    os.makedirs(caption_path, exist_ok=True)
    file_path = f'{caption_path}/{image_file}.json'
    try:
        data = json.loads(caption) if isinstance(caption, str) else caption
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"raw": str(caption), "error": "json_parse_error"}, f, indent=4)

def process_video_images(video_file, caption_path, global_context, max_workers=2):
    images_folder_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'
    
    if not os.path.exists(images_folder_path):
        print(f"Images folder not found: {images_folder_path}")
        return

    images = [f for f in os.listdir(images_folder_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: float(Path(x).stem))

    if not images:
        print("No images to process.")
        return

    global_context_text = format_global_context_for_prompt(global_context)
    
    # --- NEW: Load transcript ONCE before the loop ---
    print("Loading transcript into memory...")
    transcript_cache = load_transcript_cache(video_file)

    print(f"\n🚀 Processing {len(images)} images with {max_workers} workers...")
    
    completed = 0
    total = len(images)
    lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Pass the transcript_cache to the workers
        futures = {
            executor.submit(process_single_image, img, video_file, caption_path, global_context_text, transcript_cache): img 
            for img in images
        }
        
        for future in as_completed(futures):
            res = future.result()
            with lock:
                completed += 1
                print(f"[{completed}/{total}] {res}")

def execute(video_file, max_workers=4):
    global_context = gcb.load_global_context(video_file)
    if not global_context:
        print("⚠️ Building missing global context...")
        global_context = gcb.execute(video_file)
    
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption'
    process_video_images(video_file, caption_path, global_context, max_workers)

    # global_context_text = format_global_context_for_prompt(global_context)
    # transcript_cache = load_transcript_cache(video_file)
    # process_single_image("16.27.jpg", video_file, caption_path, global_context_text, transcript_cache)
    
    return caption_path

if __name__ == "__main__":
    import sys
    # Default to 4 workers if not specified
    max_workers_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    # HARDCODED FOR DEMO (You can change this back)
    target_video = "demo.mp4" 
    
    execute(target_video, max_workers=max_workers_arg)