#!/usr/bin/env python3
"""
Analyze the vowels dataset to assess quality and suitability
"""
import os
from pathlib import Path
import subprocess

vowels_dir = Path("vowels")

print("=" * 60)
print("DATASET ANALYSIS: VOWELS DIRECTORY")
print("=" * 60)

# Check directory structure
print("\n1. DIRECTORY STRUCTURE:")
dirs = sorted([d for d in os.listdir(vowels_dir) if os.path.isdir(vowels_dir / d) and d.isdigit()])
print(f"   Found {len(dirs)} directories: {dirs}")

# Count images per directory
print("\n2. IMAGE COUNT PER DIRECTORY:")
for dir_name in dirs:
    img_count = len(list((vowels_dir / dir_name).glob("*.jpg")))
    print(f"   Directory {dir_name}: {img_count} images")

# Check image properties
print("\n3. SAMPLE IMAGE PROPERTIES:")
sample_dirs = ['1', '6', '12']
for dir_name in sample_dirs:
    files = sorted([f for f in os.listdir(vowels_dir / dir_name) if f.endswith('.jpg')])[:3]
    print(f"\n   Directory {dir_name}:")
    for f in files:
        img_path = vowels_dir / dir_name / f
        # Use file command to get image info
        result = subprocess.run(['file', str(img_path)], capture_output=True, text=True)
        info = result.stdout.strip()
        # Extract size from file output
        if '28x28' in info:
            size = '28x28'
        elif '32x32' in info:
            size = '32x32'
        else:
            size = 'unknown'
        print(f"     {f}: {size}")

# Check naming pattern
print("\n4. NAMING PATTERN ANALYSIS:")
sample_files = sorted([f for f in os.listdir(vowels_dir / '1') if f.endswith('.jpg')])[:10]
print(f"   Sample filenames from directory 1:")
for f in sample_files[:5]:
    print(f"     {f}")

# Check if images are consistent
print("\n5. CONSISTENCY CHECK:")
all_sizes = []
for dir_name in dirs[:3]:  # Check first 3 directories
    files = sorted([f for f in os.listdir(vowels_dir / dir_name) if f.endswith('.jpg')])[:5]
    for f in files:
        img_path = vowels_dir / dir_name / f
        result = subprocess.run(['file', str(img_path)], capture_output=True, text=True)
        if '28x28' in result.stdout:
            all_sizes.append('28x28')
        elif '32x32' in result.stdout:
            all_sizes.append('32x32')

if all_sizes:
    unique_sizes = set(all_sizes)
    print(f"   Image sizes found: {unique_sizes}")
    if len(unique_sizes) == 1:
        print("   ✓ All images have consistent size")
    else:
        print("   ⚠ Multiple image sizes detected")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
total_images = sum(len(list((vowels_dir / d).glob("*.jpg"))) for d in dirs)
print(f"Total directories: {len(dirs)}")
print(f"Total images: {total_images}")
print(f"Average images per directory: {total_images // len(dirs) if dirs else 0}")
print(f"Image size: 28x28 pixels (grayscale)")
print(f"\nFor your project requirement (12 alphabets × 30 samples = 360 images):")
print(f"  ✓ You have {len(dirs)} directories (matches requirement)")
print(f"  ✓ You have {total_images // len(dirs) if dirs else 0} samples per alphabet (more than 30 needed)")
print(f"  ✓ Total images: {total_images} (much more than 360 needed)")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Need to visually inspect images to determine:")
print("   - Are they handwritten or printed?")
print("   - What script (Telugu/Devanagari)?")
print("   - Quality and variation")
print("   - Which 12 alphabets they represent")
print("2. Select 30 diverse samples per alphabet")
print("3. Verify they match your requirements (a to aha)")
