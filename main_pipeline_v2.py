"""
Enhanced Video Captioning Pipeline with Global Context (Gemma 3:4b)

This pipeline implements a two-pass approach:
1. Pass 1: Build global video context using Gemma 3:4b (Vision + Text synthesis)
2. Pass 2: Caption individual frames with global context and identity awareness
"""

from time import time
import shutil
import os
from pathlib import Path

# Core modules
import frameExtractor as fe 
import transcription as tr
import global_context_builder as gcb
import caption_images_enhanced as ci_enhanced

# You can still use the old one if needed
# import caption_images as ci_old

keyframes_folder_path = fe.frames_folder_path

def get_frames(video_path):
    """Extract keyframes and intermediate frames from video"""
    print("\n" + "="*60)
    print("PHASE 1: EXTRACTING FRAMES")
    print("="*60)
    fe.execute(video_path)
    print("✅ Frames extracted")

def get_transcript(video_path):
    """Transcribe audio using WhisperX"""
    print("\n" + "="*60)
    print("PHASE 2: TRANSCRIBING AUDIO")
    print("="*60)
    
    # Check if the folder exists, if so delete it
    audio_folder = 'audio'
    if os.path.exists(audio_folder):
        print(f"Removing existing {audio_folder} folder...")
        shutil.rmtree(audio_folder)

    # Transcribe audio
    tr.execute(video_path)
    print("✅ Audio transcribed")

def build_global_context(video_file, force_rebuild=False):
    """Build global video context (Pass 1)"""
    print("\n" + "="*60)
    print("PHASE 3: BUILDING GLOBAL CONTEXT")
    print("="*60)
    
    context = gcb.execute(video_file, force_rebuild=force_rebuild)
    print("✅ Global context built")
    return context

def caption_frames(video_file, max_workers=4):
    """Caption frames with global context (Pass 2) - Multithreaded"""
    print("\n" + "="*60)
    print(f"PHASE 4: CAPTIONING FRAMES ({max_workers} workers)")
    print("="*60)
    
    caption_path = ci_enhanced.execute(video_file, max_workers=max_workers)
    print("✅ All frames captioned")
    return caption_path

def generate_summary_report(video_file, caption_path):
    """Generate a summary report of the captioning results"""
    import json
    
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Load global context
    global_context = gcb.load_global_context(video_file)
    
    # Count captions
    caption_files = [f for f in os.listdir(caption_path) if f.endswith('.json')]
    
    print(f"\n📊 Results:")
    print(f"   Video: {video_file}")
    print(f"   Frames captioned: {len(caption_files)}")
    print(f"   Summary: {global_context.get('summary', 'N/A')}")
    
    # Sample some captions
    print(f"\n📝 Sample captions:")
    for i, cf in enumerate(sorted(caption_files)[:3]):
        try:
            with open(os.path.join(caption_path, cf), 'r') as f:
                caption = json.load(f)
            
            timestamp = Path(cf).stem.replace('.json', '')
            desc = caption.get('description', 'N/A')[:100] + "..."
            entities = caption.get('entities', [])
            
            print(f"\n   [{timestamp}s]")
            print(f"   Identity: {', '.join(entities) if entities else 'Unknown'}")
            print(f"   Description: {desc}")
        except:
            continue
    
    print("\n✅ Pipeline complete!")

def main(video_file, video_path, skip_extraction=False, force_rebuild_context=False, max_workers=4):
    """
    Main pipeline orchestrator
    """
    
    print("="*60)
    print("VIDEO CAPTIONING PIPELINE V2 (GEMMA ENHANCED)")
    print("="*60)
    print(f"Video: {video_file}")
    print(f"Workers: {max_workers} threads")
    print("="*60)
    
    start_time = time()
    
    try:
        # Phase 1 & 2: Extract frames and transcribe
        if not skip_extraction:
            get_frames(str(video_path))
            get_transcript(str(video_path))
        else:
            print("\n⏭️  Skipping frame extraction/transcription")
        
        # Phase 3: Build global context
        build_global_context(video_file, force_rebuild=force_rebuild_context)
        
        # Phase 4: Caption frames with context
        caption_path = caption_frames(video_file, max_workers=max_workers)
        
        # Reports
        generate_summary_report(video_file, caption_path)
        
        total_time = time() - start_time
        print(f"\n⏱️  Total pipeline time: {total_time:.2f}s")
        
        return caption_path
        
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configuration
    video_file = "demo.mp4"
    video_path = Path.home() / "Downloads" / video_file
    
    # Run the pipeline
    caption_path = main(
        video_file, 
        video_path, 
        skip_extraction=True,  # Set to False for first run
        force_rebuild_context=True,
        max_workers=4
    )