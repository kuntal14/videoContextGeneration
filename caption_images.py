import ollama
import json
import os
from pathlib import Path

# Load schema
with open("image_caption_schema.json", "r") as f:
    schema = json.load(f)

# Define paths
HOME = Path.home()
video_file = "demo.mov"
context_folder_path = f'{HOME}/context'
images_folder_path = f'{context_folder_path}/{video_file}/images'

def call_ollama():
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
    image_file = images[0]
    image_path = os.path.join(images_folder_path, image_file)
    print(f"Processing image: {image_path}")

    prompt = f"Analyze this image and provide a structured description in JSON format strictly adhering to the following schema:\n{json.dumps(schema, indent=2)}"

    try:
        resp = ollama.chat(
            model="gemma3:4b",  # Using a vision-capable model. gemma3:4b might not support images. 
            # If the user specifically wants gemma3:4b and it supports vision, we can revert. 
            # But usually llama3.2-vision or llava is used. 
            # Wait, the user had "gemma3:4b" in the original file. I should probably stick to it or ask.
            # However, gemma2/3 are text models usually? 
            # Let's check if I should change the model. 
            # The user said "schema is how i want the responce from the ollama response".
            # I will stick to the user's model if possible, but if it fails on image, it's an issue.
            # Actually, let's use the model from the file but add a comment or fallback?
            # No, I should probably use a vision model. 
            # But I will use the one in the file for now, assuming the user knows what they are doing or has a custom model.
            # Wait, the original file had "gemma3:4b". 
            # I will use "llava" or "llama3.2-vision" if I can, but I don't know what they have pulled.
            # I'll stick to "gemma3:4b" as placeholder but I suspect it might not work with images if it's not a VLM.
            # Actually, let's look at the user request again. "make the prompt...".
            # I will use "gemma3:4b" as it was there.
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

    except Exception as e:
        print(f"Error calling Ollama: {e}")

if __name__ == "__main__":
    call_ollama()
