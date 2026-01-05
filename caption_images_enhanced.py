import ollama
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import global_context_builder as gcb

# Define paths
HOME = Path.home()
CONTEXT_FOLDER_PATH = f'{HOME}/context'

def format_global_context_for_prompt(global_context):
    """Format global context into a concise prompt section"""
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

def process_single_image(image_file, video_file, caption_path, global_context_text):
    """Process a single image with global context awareness"""
    images_folder_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'
    image_path = os.path.abspath(os.path.join(images_folder_path, image_file))
    
    if not os.path.exists(image_path):
        return f"❌ Missing: {image_file}"
    
    try:
        timestamp = float(Path(image_file).stem)
        transcript = get_transcript(video_file, image_file)
        
        # Identity-focused instructions
        prompt = f""" this ahead is a general overview of what is happening in the video, you can use this to ground your description of the image <video context>{global_context_text}</video context>
Current Timestamp: {timestamp:.2f}s
Transcript Context: "{transcript}"

TASK: Describe this frame.
RULES:
1. If there is a person on the screen,Identify the person on screen using the Global Context.
2. If the person is identified and the transcript shows them speaking, they are NOT a narrator. They are the SPEAKER on screen.
3. Be concise.
4. If it is talking about a place then be mindful of that.
5. just describe what the image is showing.

Return ONLY this JSON:
{{
  "description": "Rich paragraph of what's happening, completely describe what the image given to you is showing",
  "entities": ["names of people/objects"],
  "actions": ["what is happening"]
}}
"""
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        resp = ollama.chat(
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
        save_caption(image_file, caption_path, content)
        return f"✅ {image_file}"

    except Exception as e:
        # Fallback minimal entry
        error_entry = {
            "description": f"Processing error: {str(e)}",
            "entities": [],
            "actions": [],
            "error": True
        }
        save_caption(image_file, caption_path, json.dumps(error_entry))
        return f"❌ {image_file}: {str(e)}"

def process_video_images(video_file, caption_path, global_context, max_workers=4):
    """Process video images with global context using ThreadPoolExecutor"""
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
    
    print(f"\n🚀 Processing {len(images)} images with {max_workers} workers...")
    
    # Track progress
    completed = 0
    total = len(images)
    lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, img, video_file, caption_path, global_context_text): img for img in images}
        
        for future in as_completed(futures):
            res = future.result()
            with lock:
                completed += 1
                print(f"[{completed}/{total}] {res}")

def get_transcript(video_file, image_file):
    """Get transcript words near this frame's timestamp"""
    csv_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/audio/transcript_16k_word_ts.csv'
    if not os.path.exists(csv_path): return "no transcription available"

    try:
        ts = float(Path(image_file).stem)
        import csv
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))
        
        words = []
        window = 3.0 # Shorter window for precision
        for r in rows:
            if abs(float(r['start_sec']) - ts) <= window:
                words.append(r['word'])
        
        return " ".join(words) if words else "silence"
    except:
        return "error loading transcript"

def save_caption(image_file, caption_path, caption):
    """Save the generated caption to a JSON file"""
    os.makedirs(caption_path, exist_ok=True)
    file_path = f'{caption_path}/{image_file}.json'
    
    try:
        if isinstance(caption, str):
            data = json.loads(caption)
        else:
            data = caption
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except:
        # Save raw if JSON fails
        with open(file_path, 'w') as f:
            json.dump({"raw": str(caption), "error": "json_parse_error"}, f, indent=4)

def execute(video_file, max_workers=4):
    """Main execution function"""
    global_context = gcb.load_global_context(video_file)
    if not global_context:
        print("⚠️ Building missing global context...")
        global_context = gcb.execute(video_file)
    
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption'
    # process_video_images(video_file, caption_path, global_context, max_workers)
    global_context_text = format_global_context_for_prompt(global_context)
    process_single_image("16.27.jpg", video_file, caption_path, global_context_text)
    
    return caption_path

if __name__ == "__main__":
    import sys
    video_file = "demo.mp4"
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    execute(video_file, max_workers=max_workers)