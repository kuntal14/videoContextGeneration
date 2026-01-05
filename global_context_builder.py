import json
import os
import csv
import base64
from pathlib import Path
import ollama

HOME = Path.home()
CONTEXT_FOLDER_PATH = f'{HOME}/context'

def encode_image(image_path):
    """Encode image to base64 for Ollama"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_full_transcript(video_file):
    """Get complete transcript for the video from WhisperX results"""
    csv_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/audio/transcript_16k_word_ts.csv'
    
    if not os.path.exists(csv_path):
        return "No transcript available"
    
    try:
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))
        
        transcript_lines = []
        current_line = []
        last_ts = 0
        
        for r in rows:
            ts = float(r['start_sec'])
            word = r['word']
            
            # Group words every 10 seconds for readability
            if ts - last_ts > 10 and current_line:
                transcript_lines.append(f"[{int(last_ts)}s] {' '.join(current_line)}")
                current_line = []
                last_ts = ts
            
            current_line.append(word)
        
        if current_line:
            transcript_lines.append(f"[{int(last_ts)}s] {' '.join(current_line)}")
        
        return "\n".join(transcript_lines)
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return "Error loading transcript."

def sample_keyframes(video_file, max_frames=8):
    """Sample keyframes evenly across the video for global context pass 1"""
    images_folder_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'
    
    if not os.path.exists(images_folder_path):
        print(f"Images folder not found: {images_folder_path}")
        return []
    
    images = [f for f in os.listdir(images_folder_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: float(Path(x).stem))
    
    if len(images) <= max_frames:
        sampled = images
    else:
        step = len(images) / (max_frames - 1)
        indices = [int(i * step) for i in range(max_frames - 1)]
        indices.append(len(images) - 1)
        sampled = [images[i] for i in sorted(list(set(indices)))]
    
    return [(float(Path(img).stem), os.path.join(images_folder_path, img)) 
            for img in sampled]

def _get_frame_description(img_path, timestamp):
    """Get a brief description of a single frame using Gemma 3:4b vision"""
    prompt = f"Describe this video frame at {timestamp:.2f}s precisely. Identify people, visible text, and the setting."
    
    try:
        resp = ollama.chat(
            model="gemma3:4b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_path]
            }],
            options={"temperature": 0.1}
        )
        return resp['message']['content'].strip()
    except Exception as e:
        print(f"Error describing frame {timestamp}: {e}")
        return f"Frame at {timestamp}s: analysis failed."

def build_global_context(video_file):
    """
    Two-pass context builder for Gemma 3:4b:
    1. Direct visual analysis of sampled frames.
    2. Textual synthesis of frame descriptions + full transcript.
    """
    print(f"\n🌍 Phase 1: Analyzing key visual frames for {video_file}...")
    sampled_frames = sample_keyframes(video_file, max_frames=8)
    
    frame_descriptions = []
    for ts, img_path in sampled_frames:
        print(f"  🔍 Analyzing frame at {ts:.2f}s...")
        desc = _get_frame_description(img_path, ts)
        frame_descriptions.append(f"- At {ts:.2f}s: {desc}")
    
    all_frame_desc = "\n".join(frame_descriptions)
    transcript = get_full_transcript(video_file)
    
    print(f"\n🌍 Phase 2: Synthesizing global context for {video_file}...")
    return _synthesize_context(all_frame_desc, transcript)

def _synthesize_context(frame_descriptions, transcript):
    """Synthesize visual and audio data into a robust global context JSON"""
    prompt = f"""You are an expert video indexer. Combine the visual frame descriptions and the audio transcript to build a detailed Global Context.

VISUAL DESCRIPTIONS:
{frame_descriptions}

AUDIO TRANSCRIPT:
{transcript}

TASK:
Identify the primary speaker(s), their names (look for self-intros or text overlays), roles, the video's style, and key events. 
CRITICAL: If someone says "I am [Name]" or text shows a name, use it. Do NOT use generic terms like "narrator" if the person is on screen.

Return ONLY a JSON object:
{{
  "summary": "2-3 sentence overview",
  "entities": {{
    "people": [
      {{
        "name": "Full name",
        "role": "speaker/subject",
        "description": "appearance/identity",
        "appearance_timestamps": [list of floats]
      }}
    ],
    "locations": ["places shown"],
    "objects": ["key items"]
  }},
  "narrative_style": "interview/vlog/presentation/etc",
  "speaker_map": {{
    "start_time-end_time": "Speaker Name"
  }},
  "key_moments": [
    {{
      "timestamp": float,
      "description": "what happened"
    }}
  ]
}}
"""
    try:
        resp = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.2}
        )
        return json.loads(resp['message']['content'])
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return {{
            "summary": "Video context synthesis failed",
            "entities": {{"people": [], "locations": [], "objects": []}},
            "narrative_style": "unknown",
            "speaker_map": {{}},
            "key_moments": []
        }}

def save_global_context(video_file, context):
    """Save global context to the designated folder"""
    output_path = Path(CONTEXT_FOLDER_PATH) / video_file / "global_context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(context, f, indent=4)
    print(f"✅ Global context saved: {output_path}")

def load_global_context(video_file):
    """Load existing context if available"""
    path = Path(CONTEXT_FOLDER_PATH) / video_file / "global_context.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def execute(video_file, use_claude=False, force_rebuild=False, **kwargs):
    """Entry point for the pipeline"""
    if not force_rebuild:
        existing = load_global_context(video_file)
        if existing:
            return existing
            
    context = build_global_context(video_file)
    save_global_context(video_file, context)
    return context

if __name__ == "__main__":
    v = "demo.mp4"
    res = execute(v, force_rebuild=True)
    print(json.dumps(res, indent=2))