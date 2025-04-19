from PIL import Image
import os
import sys

def reduce_resolution(input_path, output_path, scale_factor=0.5):
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert to RGB if image has transparency (RGBA or P mode)
            if resized_img.mode in ("RGBA", "P"):
                resized_img = resized_img.convert("RGB")

            resized_img.save(output_path)
            print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resize.py <input_image> <output_image> [scale_factor]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    scale_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    reduce_resolution(input_image, output_image, scale_factor)
