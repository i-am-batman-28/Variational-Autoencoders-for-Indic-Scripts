#!/usr/bin/env python3
"""
Create a grid of sample images for visual inspection
This will help determine if the dataset is suitable
"""
import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL/Pillow not installed. Install with: pip install Pillow")

if not HAS_PIL:
    exit(1)

vowels_dir = Path("vowels")

print("Creating sample image grid for visual inspection...")

# Create a grid showing samples from each directory
grid_size = (4, 3)  # 4 columns, 3 rows (for 12 directories)
sample_per_dir = 1
img_size = 28
padding = 5
label_height = 20

# Calculate grid dimensions
grid_width = grid_size[0] * (img_size + padding) + padding
grid_height = grid_size[1] * (img_size + padding + label_height) + padding

# Create output image
output_img = Image.new('RGB', (grid_width, grid_height), color='white')
draw = ImageDraw.Draw(output_img)

dirs = sorted([d for d in os.listdir(vowels_dir) if os.path.isdir(vowels_dir / d) and d.isdigit()], key=int)

print(f"\nFound {len(dirs)} directories")
print("Creating grid with one sample from each directory...")

for idx, dir_name in enumerate(dirs):
    row = idx // grid_size[0]
    col = idx % grid_size[0]
    
    # Get a sample image
    files = sorted([f for f in os.listdir(vowels_dir / dir_name) if f.endswith('.jpg')])
    if files:
        sample_file = files[0]  # First image
        img_path = vowels_dir / dir_name / sample_file
        
        try:
            img = Image.open(img_path)
            # Convert to RGB if grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate position
            x = padding + col * (img_size + padding)
            y = padding + row * (img_size + padding + label_height)
            
            # Resize if needed (should already be 28x28)
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            
            # Paste image
            output_img.paste(img, (x, y))
            
            # Add label
            label_text = f"Dir {dir_name}"
            try:
                # Try to use a small font
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (img_size - text_width) // 2
            text_y = y + img_size + 2
            
            draw.text((text_x, text_y), label_text, fill='black', font=font)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Save the grid
output_path = "vowels_sample_grid.png"
output_img.save(output_path)
print(f"\n✓ Sample grid saved to: {output_path}")
print("Open this file to see samples from all 12 directories at once")

# Also create a larger grid with multiple samples per directory
print("\nCreating detailed grid with 3 samples per directory...")

grid_cols = 12  # One column per directory
grid_rows = 3   # 3 samples per directory
img_size_large = 56  # Larger for better visibility

grid_width_large = grid_cols * (img_size_large + padding) + padding
grid_height_large = grid_rows * (img_size_large + padding + label_height) + padding

output_img_large = Image.new('RGB', (grid_width_large, grid_height_large), color='white')
draw_large = ImageDraw.Draw(output_img_large)

for col_idx, dir_name in enumerate(dirs):
    files = sorted([f for f in os.listdir(vowels_dir / dir_name) if f.endswith('.jpg')])
    
    # Sample from beginning, middle, end
    sample_indices = [0, len(files)//2, len(files)-1] if len(files) > 2 else [0] * 3
    
    for row_idx, file_idx in enumerate(sample_indices[:3]):
        if file_idx < len(files):
            sample_file = files[file_idx]
            img_path = vowels_dir / dir_name / sample_file
            
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                x = padding + col_idx * (img_size_large + padding)
                y = padding + row_idx * (img_size_large + padding + label_height)
                
                if img.size != (img_size_large, img_size_large):
                    img = img.resize((img_size_large, img_size_large), Image.Resampling.LANCZOS)
                
                output_img_large.paste(img, (x, y))
                
                # Add label only on first row
                if row_idx == 0:
                    label_text = f"{dir_name}"
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                    except:
                        font = ImageFont.load_default()
                    
                    text_bbox = draw_large.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_x = x + (img_size_large - text_width) // 2
                    text_y = y + img_size_large + 2
                    
                    draw_large.text((text_x, text_y), label_text, fill='black', font=font)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

output_path_large = "vowels_detailed_grid.png"
output_img_large.save(output_path_large)
print(f"✓ Detailed grid saved to: {output_path_large}")
print("This shows 3 samples from each of the 12 directories")

print("\n" + "="*60)
print("NEXT: Open the grid images to visually inspect:")
print("  - Are characters handwritten or printed?")
print("  - What script? (Telugu or Devanagari)")
print("  - Quality and variation across samples")
print("="*60)
