"""
Debug helper for captioning issues
Helps diagnose why Ollama might be returning empty JSON
"""

import json
import os
from pathlib import Path

HOME = Path.home()
CONTEXT_FOLDER_PATH = f'{HOME}/context'

def check_caption_folder(video_file):
    """Check caption folder for issues"""
    
    print(f"\n{'='*60}")
    print("CAPTION FOLDER DIAGNOSTICS")
    print(f"{'='*60}\n")
    
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption'
    
    if not os.path.exists(caption_path):
        print(f"⚠️ Caption folder doesn't exist yet: {caption_path}")
        return
    
    caption_files = [f for f in os.listdir(caption_path) if f.endswith('.json')]
    
    print(f"Total caption files: {len(caption_files)}")
    
    # Check for empty captions
    empty_count = 0
    error_count = 0
    valid_count = 0
    raw_content_count = 0
    
    issues = []
    
    for cf in caption_files:
        with open(os.path.join(caption_path, cf), 'r') as f:
            try:
                data = json.load(f)
                
                # Check if empty
                if not data or data == {}:
                    empty_count += 1
                    issues.append((cf, "Empty JSON"))
                    continue
                
                # Check if it has 'raw_content' (means JSON parsing failed in captioning)
                if 'raw_content' in data:
                    raw_content_count += 1
                    error_count += 1
                    issues.append((cf, "Has raw_content wrapper - JSON parsing failed"))
                    continue
                
                # Check if it has explicit error field
                if 'error' in data:
                    error_count += 1
                    issues.append((cf, f"Error: {data['error']}"))
                    continue
                
                # Validate against expected schema
                required_fields = ['scene', 'objects', 'people', 'actions', 'emotion', 'lighting']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    error_count += 1
                    issues.append((cf, f"Missing fields: {missing_fields}"))
                    continue
                
                # Check if people have proper structure
                if 'people' in data and isinstance(data['people'], list):
                    for person in data['people']:
                        # Check if person is missing 'type' field (should have name/type)
                        if isinstance(person, dict) and 'type' not in person and 'name' not in person:
                            # This is suspicious but not necessarily invalid
                            pass
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                error_count += 1
                issues.append((cf, f"Invalid JSON: {str(e)[:50]}"))
            except Exception as e:
                error_count += 1
                issues.append((cf, f"Error reading: {str(e)[:50]}"))
    
    print(f"  Valid captions: {valid_count}")
    print(f"  Empty captions: {empty_count}")
    print(f"  Error captions: {error_count}")
    
    if empty_count > 0:
        print(f"\n⚠️ Found {empty_count} empty captions!")
        print("   This suggests Ollama is struggling with the prompt complexity.")
        print("   Recommendations:")
        print("   1. Reduce num_ctx from 8192 to 4096")
        print("   2. Simplify global context (fewer people/objects)")
        print("   3. Update Ollama: ollama pull gemma3:4b")

def check_image_filenames(video_file):
    """Check for filename parsing issues"""
    
    print(f"\n{'='*60}")
    print("IMAGE FILENAME DIAGNOSTICS")
    print(f"{'='*60}\n")
    
    images_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'
    
    if not os.path.exists(images_path):
        print(f"❌ Images folder not found: {images_path}")
        return
    
    images = [f for f in os.listdir(images_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Total images: {len(images)}")
    
    # Check for parsing issues
    invalid_count = 0
    valid_timestamps = []
    
    for img in images:
        try:
            ts = float(Path(img).stem)
            valid_timestamps.append((ts, img))
        except ValueError:
            print(f"  ⚠️ Invalid timestamp filename: {img}")
            invalid_count += 1
    
    if invalid_count == 0:
        print("✅ All filenames have valid timestamps")
        
        # Show timestamp range
        valid_timestamps.sort()
        print(f"   Range: {valid_timestamps[0][0]:.2f}s to {valid_timestamps[-1][0]:.2f}s")
        print(f"   Sample files: {[img for _, img in valid_timestamps[:3]]}")
    else:
        print(f"❌ Found {invalid_count} invalid filenames")

def fix_raw_content_captions(video_file, dry_run=True):
    """
    Fix captions that have raw_content wrapper
    
    Args:
        video_file: Name of video file
        dry_run: If True, only shows what would be fixed without modifying files
    """
    print(f"\n{'='*60}")
    print("FIXING RAW_CONTENT CAPTIONS")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print(f"{'='*60}\n")
    
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption'
    
    if not os.path.exists(caption_path):
        print(f"❌ Caption folder not found: {caption_path}")
        return
    
    caption_files = [f for f in os.listdir(caption_path) if f.endswith('.json')]
    
    fixed_count = 0
    failed_count = 0
    
    for cf in caption_files:
        file_path = os.path.join(caption_path, cf)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it has raw_content
        if 'raw_content' not in data:
            continue
        
        print(f"\n📄 Processing: {cf}")
        
        try:
            # Extract the raw content
            raw_content = data['raw_content']
            
            # Try to parse it as JSON
            parsed = json.loads(raw_content)
            
            print(f"   ✅ Successfully parsed raw_content")
            print(f"   Fields: {list(parsed.keys())}")
            
            if not dry_run:
                # Save the fixed version
                with open(file_path, 'w') as f:
                    json.dump(parsed, f, indent=4)
                print(f"   💾 Saved fixed version")
            else:
                print(f"   🔍 Would save: {list(parsed.keys())}")
            
            fixed_count += 1
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse: {str(e)[:100]}")
            print(f"   Raw content preview: {raw_content[:200]}...")
            failed_count += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Failed: {failed_count}")
    
    if dry_run and fixed_count > 0:
        print(f"\n💡 To apply fixes, run:")
        print(f"   fix_raw_content_captions('{video_file}', dry_run=False)")

def inspect_caption(video_file, frame_name):
    """Inspect a specific caption file in detail"""
    
    print(f"\n{'='*60}")
    print(f"INSPECTING: {frame_name}")
    print(f"{'='*60}\n")
    
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption/{frame_name}.json'
    
    if not os.path.exists(caption_path):
        print(f"❌ Caption file not found: {caption_path}")
        return
    
    with open(caption_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} bytes\n")
    
    # Try to parse as JSON
    try:
        data = json.loads(content)
        print("✅ Valid JSON file\n")
        
        print("Top-level keys:")
        for key in data.keys():
            value = data[key]
            value_type = type(value).__name__
            
            if isinstance(value, (list, dict)):
                length = len(value)
                print(f"  - {key}: {value_type} (length: {length})")
            else:
                preview = str(value)[:50]
                print(f"  - {key}: {value_type} = {preview}...")
        
        # Check for issues
        issues = []
        
        if 'raw_content' in data:
            issues.append("❌ Has 'raw_content' wrapper (JSON parsing failed during captioning)")
            print(f"\nRaw content preview:")
            print(f"{data['raw_content'][:300]}...")
            
            # Try to parse the raw content
            try:
                parsed = json.loads(data['raw_content'])
                print(f"\n✅ Raw content IS valid JSON with keys: {list(parsed.keys())}")
                print("💡 This can be auto-fixed with fix_raw_content_captions()")
            except:
                print(f"\n❌ Raw content is NOT valid JSON")
        
        if 'error' in data:
            issues.append(f"❌ Has error field: {data['error']}")
        
        required_fields = ['scene', 'objects', 'people', 'actions', 'emotion', 'lighting']
        missing = [f for f in required_fields if f not in data]
        if missing:
            issues.append(f"❌ Missing required fields: {missing}")
        
        if issues:
            print(f"\n⚠️  Issues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"\n✅ No issues detected")
        
    except json.JSONDecodeError as e:
        print(f"❌ INVALID JSON FILE\n")
        print(f"Error: {e}\n")
        print(f"File content preview (first 500 chars):")
        print(content[:500])
        print("...")

if __name__ == "__main__":
    import sys
    
    video_file = "demo.mp4"
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "inspect" and len(sys.argv) > 2:
            frame_name = sys.argv[2]
            inspect_caption(video_file, frame_name)
        
        elif command == "fix":
            dry_run = "--live" not in sys.argv
            fix_raw_content_captions(video_file, dry_run=dry_run)
        
        elif command == "test" and len(sys.argv) > 2:
            frame_name = sys.argv[2]
            test_single_frame(video_file, frame_name)
        
        else:
            print("Usage:")
            print("  python debug_captioning.py                    # Run full diagnostics")
            print("  python debug_captioning.py fix                # Dry-run fix raw_content")
            print("  python debug_captioning.py fix --live         # Actually fix files")
    
    else:
        # Run full diagnostics
        print("="*60)
        print("VIDEO CAPTIONING DIAGNOSTICS")
        print("="*60)
        
        check_image_filenames(video_file)
        check_caption_folder(video_file)
        