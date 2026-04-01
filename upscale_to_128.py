#!/usr/bin/env python3
"""
Upscale all vowel images to 128x128 and save in vowels_scaled directory
Maintains the same directory structure as original
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def upscale_image(input_path, output_path, target_size=128, method='lanczos'):
    """
    Upscale an image to target size using Lanczos resampling
    """
    try:
        img = Image.open(input_path)
        
        # Convert to RGB if grayscale (for consistency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize using Lanczos (best quality for upscaling)
        upscaled = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Save as PNG for better quality
        upscaled.save(output_path, 'PNG', quality=95)
        return True
    except Exception as e:
        print(f"\nError processing {input_path}: {e}")
        return False

def upscale_vowels_dataset(source_dir='vowels', output_dir='vowels_scaled', target_size=128):
    """
    Upscale entire vowels dataset to target size
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        return
    
    # Get all numbered directories
    dirs = sorted([d for d in os.listdir(source_path) 
                   if os.path.isdir(source_path / d) and d.isdigit()], key=int)
    
    if not dirs:
        print(f"Error: No numbered directories found in '{source_dir}'")
        return
    
    print("=" * 70)
    print("UPSCALING VOWELS DATASET TO 128x128")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Method: Lanczos (high quality)")
    print(f"Directories to process: {len(dirs)}")
    print("=" * 70)
    
    total_images = 0
    total_successful = 0
    total_failed = 0
    
    # Process each directory
    for dir_name in dirs:
        input_dir = source_path / dir_name
        output_subdir = output_path / dir_name
        
        # Create output directory
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Get all JPG files
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
        
        if not image_files:
            print(f"\n⚠️  No images found in directory {dir_name}")
            continue
        
        total_images += len(image_files)
        successful = 0
        failed = 0
        
        print(f"\n📁 Processing directory {dir_name} ({len(image_files)} images)...")
        
        # Process with progress bar
        for filename in tqdm(image_files, desc=f"  Dir {dir_name}", leave=False):
            input_file = input_dir / filename
            # Change extension to PNG for better quality
            output_filename = filename.replace('.jpg', '.png')
            output_file = output_subdir / output_filename
            
            if upscale_image(input_file, output_file, target_size):
                successful += 1
            else:
                failed += 1
        
        total_successful += successful
        total_failed += failed
        
        print(f"  ✓ Successfully upscaled: {successful} images")
        if failed > 0:
            print(f"  ✗ Failed: {failed} images")
    
    # Summary
    print("\n" + "=" * 70)
    print("UPSCALING COMPLETE")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print(f"✓ Successfully upscaled: {total_successful} images")
    print(f"✗ Failed: {total_failed} images")
    print(f"\nOutput directory: {output_dir}")
    print("Structure maintained: vowels_scaled/1/, vowels_scaled/2/, etc.")
    print("=" * 70)
    
    # Verify output
    print("\nVerifying output...")
    for dir_name in dirs[:3]:  # Check first 3 directories
        output_subdir = output_path / dir_name
        if output_subdir.exists():
            png_count = len(list(output_subdir.glob("*.png")))
            print(f"  {output_dir}/{dir_name}: {png_count} PNG images")
    
    print("\n✓ All done! Your upscaled dataset is ready in 'vowels_scaled/'")

if __name__ == "__main__":
    # Check if tqdm is available, if not, use simple progress
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
    
    upscale_vowels_dataset(
        source_dir='vowels',
        output_dir='vowels_scaled',
        target_size=128
    )
