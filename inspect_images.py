#!/usr/bin/env python3
"""
Visual inspection of sample images from the vowels dataset
"""
import os
from pathlib import Path
import subprocess
import base64

vowels_dir = Path("vowels")

print("=" * 70)
print("VISUAL INSPECTION OF VOWELS DATASET")
print("=" * 70)

# Function to get image info and create ASCII preview
def inspect_image(img_path):
    """Get detailed info about an image"""
    result = subprocess.run(['file', str(img_path)], capture_output=True, text=True)
    file_info = result.stdout.strip()
    
    # Try to get more details using identify (ImageMagick) if available
    try:
        result = subprocess.run(['identify', str(img_path)], capture_output=True, text=True)
        identify_info = result.stdout.strip()
        return file_info, identify_info
    except:
        return file_info, None

# Sample images from different directories
print("\nSAMPLING IMAGES FROM DIFFERENT DIRECTORIES:")
print("-" * 70)

dirs_to_check = ['1', '3', '6', '9', '12']
for dir_name in dirs_to_check:
    files = sorted([f for f in os.listdir(vowels_dir / dir_name) if f.endswith('.jpg')])
    # Sample from beginning, middle, and end
    sample_indices = [0, len(files)//2, len(files)-1]
    sample_files = [files[i] for i in sample_indices if i < len(files)]
    
    print(f"\n📁 Directory {dir_name} (Alphabet {dir_name}):")
    for f in sample_files[:3]:
        img_path = vowels_dir / dir_name / f
        file_info, identify_info = inspect_image(img_path)
        
        # Extract key info
        size_info = "28x28" if "28x28" in file_info else "unknown"
        color_info = "grayscale" if "components 1" in file_info or "Gray" in file_info else "color"
        
        print(f"   📷 {f}")
        print(f"      Size: {size_info}, Color: {color_info}")
        
        # Try to get file size
        file_size = os.path.getsize(img_path)
        print(f"      File size: {file_size} bytes")

print("\n" + "=" * 70)
print("CHECKLIST FOR DATASET EVALUATION:")
print("=" * 70)
print("\n1. VISUAL INSPECTION NEEDED:")
print("   [ ] Open a few images from each directory to see:")
print("       - Are they handwritten or printed?")
print("       - What script? (Telugu or Devanagari)")
print("       - Quality: clear, well-centered, consistent?")
print("       - Variation: different writing styles/samples?")
print("\n2. ALPHABET MAPPING:")
print("   [ ] Verify directories 1-12 correspond to:")
print("       a, aa, i, ii, u, uu, e, ee, ai, o, oo, aha")
print("\n3. SAMPLE SELECTION:")
print("   [ ] Choose 30 diverse samples per alphabet")
print("       - Different writers/styles if handwritten")
print("       - Good quality, well-formed characters")
print("\n4. DATASET SUITABILITY:")
print("   [ ] Check if this matches your project requirements")
print("   [ ] Determine if you need to look for alternatives")

print("\n" + "=" * 70)
print("TO VIEW IMAGES:")
print("=" * 70)
print("Run this command to open sample images:")
print("  open vowels/1/001_01.jpg")
print("  open vowels/6/001_01.jpg")
print("  open vowels/12/001_01.jpg")
print("\nOr use Python with PIL/Pillow:")
print("  from PIL import Image; Image.open('vowels/1/001_01.jpg').show()")
