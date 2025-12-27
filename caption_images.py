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

def process_video_images(video_file, caption_path):
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

    # process all the images
    for image_file in images:
        image_path = os.path.abspath(os.path.join(images_folder_path, image_file))
        if not os.path.exists(image_path):
            print(f"❌ Image file does not exist: {image_path}")
            return
        print(f"Processing image: {image_path}")
        
        prompt = (
            "Analyze the provided image in detail. "
            "Fill in the following JSON template based on the image content:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Ensure every field has a value. Predict every value to the best of your ability. No field should be empty strictly"
            "Return ONLY the populated JSON object."
        )

        try:
            resp = ollama.chat(
                model="gemma3:4b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path]
                    }
                ],
                format="json",
                options={
                    "temperature": 0.2,
                    "num_ctx": 4096
                }
            )

            content = resp["message"]["content"]
            print("\n=== Answer ===")
            print(content)
            
            if content.strip() == "{}":
                print("⚠️ Warning: Ollama returned an empty JSON object. Check model/image.")
            
            # Save the result
            newEntry(image_file, caption_path, content)

        except Exception as e:
            print(f"Error calling Ollama: {e}")


def newEntry(image_file, caption_path, caption):
    # Ensure directory exists
    os.makedirs(caption_path, exist_ok=True)
    
    file_path = f'{caption_path}/{image_file}.json'
    
    # Convert string response to JSON object if necessary
    if isinstance(caption, str):
        try:
            caption_data = json.loads(caption)
        except json.JSONDecodeError:
            print("⚠️ Warning: Ollama returned invalid JSON. Saving as raw string.")
            caption_data = {"raw_content": caption}
    else:
        caption_data = caption

    # Write as formatted JSON
    with open(file_path, 'w') as f:
        json.dump(caption_data, f, indent=4)
    
    print(f"✅ Caption saved to {file_path}")

def execute(video_file):
    caption_path = f'{CONTEXT_FOLDER_PATH}/{video_file}/images_caption'
    return process_video_images(video_file, caption_path)

if __name__ == "__main__":
    execute("demo.mp4")
