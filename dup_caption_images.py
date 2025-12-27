import ollama
import json
import os
from pathlib import Path

# Load schema
with open("image_caption_schema.json", "r") as f:
    schema = json.load(f)

# Define paths
HOME = Path.home()
CONTEXT_FOLDER_PATH = f'{HOME}/context'

def process_video_images(video_file):
    images_folder_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images'

    # Check if images folder exists
    if not os.path.exists(images_folder_path):
        print(f"Images folder not found: {images_folder_path}")
        return

    # Get list of images
    images = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort() # Ensure consistent order

    if not images:
        print("No images found in the folder.")
        return

    # Process just the first image as requested
    # TODO: Iterate over all images if needed in the future
    image_file = images[0]
    image_path = os.path.join(images_folder_path, image_file)
    print(f"Processing image: {image_path}")

    prompt = f"Analyze this image and provide a structured description in JSON format strictly adhering to the following schema:\n{json.dumps(schema, indent=2)}"

    try:
        resp = ollama.chat(
            model="gemma3:4b",  # Using a vision-capable model.
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ],
            format="json",
        )

        print("\n=== Answer ===")
        print(resp["message"]["content"])
        return resp["message"]["content"]

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def execute(video_file):
    return process_video_images(video_file)

if __name__ == "__main__":
    execute("demo.mov")
