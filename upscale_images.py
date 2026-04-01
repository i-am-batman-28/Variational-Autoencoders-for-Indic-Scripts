#!/usr/bin/env python3
"""
Upscale images from 28x28 to larger sizes (64x64 or 128x128)
Multiple upscaling methods available for comparison
"""

import os
from pathlib import Path
from PIL import Image
import argparse

def upscale_image(input_path, output_path, target_size, method='lanczos'):
    """
    Upscale an image to target size
    
    Methods:
    - 'lanczos': High-quality resampling (recommended)
    - 'bicubic': Good quality, smooth
    - 'nearest': Fast but pixelated
    - 'bilinear': Fast, moderate quality
    """
    try:
        img = Image.open(input_path)
        
        # Convert to RGB if grayscale (for consistency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize using specified method
        resample_methods = {
            'lanczos': Image.Resampling.LANCZOS,  # Best quality, recommended
            'bicubic': Image.Resampling.BICUBIC,   # Good quality
            'bilinear': Image.Resampling.BILINEAR,  # Moderate quality
            'nearest': Image.Resampling.NEAREST     # Fast but pixelated
        }
        
        resample = resample_methods.get(method.lower(), Image.Resampling.LANCZOS)
        upscaled = img.resize((target_size, target_size), resample)
        
        # Save the upscaled image
        upscaled.save(output_path, 'PNG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def upscale_directory(input_dir, output_dir, target_size, method='lanczos', sample_count=None):
    """
    Upscale all images in a directory
    
    Args:
        input_dir: Source directory
        output_dir: Destination directory
        target_size: Target size (e.g., 64 or 128)
        method: Upscaling method
        sample_count: If specified, only process first N images (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all JPG files
    image_files = sorted([f for f in os.listdir(input_path) if f.endswith('.jpg')])
    
    if sample_count:
        image_files = image_files[:sample_count]
        print(f"Processing first {sample_count} images for testing...")
    
    total = len(image_files)
    successful = 0
    failed = 0
    
    print(f"\nUpscaling {total} images from {input_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Method: {method}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    for idx, filename in enumerate(image_files, 1):
        input_file = input_path / filename
        # Change extension to PNG for better quality
        output_filename = filename.replace('.jpg', '.png')
        output_file = output_path / output_filename
        
        if upscale_image(input_file, output_file, target_size, method):
            successful += 1
        else:
            failed += 1
        
        if idx % 50 == 0:
            print(f"Progress: {idx}/{total} images processed...")
    
    print("-" * 60)
    print(f"✓ Successfully upscaled: {successful} images")
    if failed > 0:
        print(f"✗ Failed: {failed} images")
    
    return successful, failed

def upscale_vowels_dataset(vowels_dir='vowels', target_size=64, method='lanczos', 
                           output_suffix='_upscaled', sample_test=False):
    """
    Upscale the entire vowels dataset
    
    Args:
        vowels_dir: Path to vowels directory
        target_size: Target size (64 or 128)
        method: Upscaling method
        output_suffix: Suffix for output directory
        sample_test: If True, only process 5 images per directory for testing
    """
    vowels_path = Path(vowels_dir)
    
    if not vowels_path.exists():
        print(f"Error: Directory '{vowels_dir}' not found!")
        return
    
    # Get all numbered directories
    dirs = sorted([d for d in os.listdir(vowels_path) 
                   if os.path.isdir(vowels_path / d) and d.isdigit()], key=int)
    
    print("=" * 70)
    print("VOWELS DATASET UPSCALING")
    print("=" * 70)
    print(f"Source: {vowels_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Method: {method}")
    print(f"Directories to process: {len(dirs)}")
    print("=" * 70)
    
    total_successful = 0
    total_failed = 0
    
    for dir_name in dirs:
        input_dir = vowels_path / dir_name
        output_dir = vowels_path / f"{dir_name}{output_suffix}_{target_size}"
        
        print(f"\n📁 Processing directory: {dir_name}")
        
        if sample_test:
            success, failed = upscale_directory(input_dir, output_dir, target_size, 
                                               method, sample_count=5)
        else:
            success, failed = upscale_directory(input_dir, output_dir, target_size, method)
        
        total_successful += success
        total_failed += failed
    
    print("\n" + "=" * 70)
    print("UPSCALING COMPLETE")
    print("=" * 70)
    print(f"Total successful: {total_successful} images")
    print(f"Total failed: {total_failed} images")
    print(f"\nUpscaled images saved in directories with suffix: {output_suffix}_{target_size}")
    print("=" * 70)

def compare_methods(input_dir, output_dir, target_size=64):
    """
    Create comparison images using different upscaling methods
    Useful for choosing the best method
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get a sample image
    image_files = sorted([f for f in os.listdir(input_path) if f.endswith('.jpg')])
    if not image_files:
        print("No images found!")
        return
    
    sample_file = image_files[0]
    input_file = input_path / sample_file
    
    methods = ['lanczos', 'bicubic', 'bilinear', 'nearest']
    
    print(f"\nCreating comparison for: {sample_file}")
    print(f"Original size: 28x28 → Target size: {target_size}x{target_size}")
    
    for method in methods:
        output_file = output_path / f"comparison_{method}_{target_size}.png"
        upscale_image(input_file, output_file, target_size, method)
        print(f"  ✓ Created: {output_file.name}")
    
    print(f"\nCompare the images in: {output_dir}")
    print("LANCZOS is usually best for character images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upscale vowel dataset images')
    parser.add_argument('--size', type=int, default=64, choices=[64, 128],
                       help='Target size (64 or 128, default: 64)')
    parser.add_argument('--method', type=str, default='lanczos',
                       choices=['lanczos', 'bicubic', 'bilinear', 'nearest'],
                       help='Upscaling method (default: lanczos)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: only process 5 images per directory')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison images with different methods')
    parser.add_argument('--input-dir', type=str, default='vowels',
                       help='Input directory (default: vowels)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Create comparison for first directory
        vowels_path = Path(args.input_dir)
        dirs = sorted([d for d in os.listdir(vowels_path) 
                       if os.path.isdir(vowels_path / d) and d.isdigit()], key=int)
        if dirs:
            compare_dir = Path('upscale_comparison')
            compare_methods(vowels_path / dirs[0], compare_dir, args.size)
            print(f"\n✓ Comparison images created in: {compare_dir}")
            print("Open these images to see which method looks best!")
    else:
        upscale_vowels_dataset(
            vowels_dir=args.input_dir,
            target_size=args.size,
            method=args.method,
            sample_test=args.test
        )
