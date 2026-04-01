#!/usr/bin/env python3
"""
Visual Quality Checker - Creates sample grids for visual inspection
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_quality_grid(dataset_dir='dataset', output_file='quality_samples.png'):
    """Create a grid showing samples from different categories"""
    
    dataset_path = Path(dataset_dir)
    
    # Create a grid: 4 columns (sizes) × 3 rows (fonts + alphabet samples)
    grid_cols = 4
    grid_rows = 6  # 2 fonts × 3 sample rows
    img_size = 128
    padding = 5
    label_height = 20
    
    grid_width = grid_cols * (img_size + padding) + padding
    grid_height = grid_rows * (img_size + padding + label_height) + padding
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    fonts = ['akshar', 'mangal']
    sizes = ['10pt', '14pt', '18pt', '22pt']
    alphabets = ['a', 'aa', 'i']
    
    row = 0
    for font in fonts:
        for alphabet in alphabets:
            for col, size in enumerate(sizes):
                # Find a sample image
                img_path = dataset_path / font / alphabet / size
                if img_path.exists():
                    png_files = list(img_path.glob('*.png'))
                    if png_files:
                        sample_file = png_files[0]
                        
                        try:
                            img = Image.open(sample_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize if needed
                            if img.size != (img_size, img_size):
                                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                            
                            # Calculate position
                            x = padding + col * (img_size + padding)
                            y = padding + row * (img_size + padding + label_height)
                            
                            # Paste image
                            grid_img.paste(img, (x, y))
                            
                            # Add label
                            label = f"{font[0]}/{alphabet}/{size}"
                            try:
                                font_obj = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
                            except:
                                font_obj = ImageFont.load_default()
                            
                            text_bbox = draw.textbbox((0, 0), label, font=font_obj)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_x = x + (img_size - text_width) // 2
                            text_y = y + img_size + 2
                            
                            draw.text((text_x, text_y), label, fill='black', font=font_obj)
                            
                        except Exception as e:
                            print(f"Error processing {sample_file}: {e}")
            
            row += 1
    
    grid_img.save(output_file)
    print(f"✅ Quality sample grid saved to: {output_file}")
    return output_file

def analyze_sample_quality(dataset_dir='dataset'):
    """Analyze and report on sample image quality"""
    
    dataset_path = Path(dataset_dir)
    
    print("\n" + "=" * 70)
    print("DETAILED QUALITY ANALYSIS")
    print("=" * 70)
    
    # Analyze samples from each size
    for size in ['10pt', '14pt', '18pt', '22pt']:
        print(f"\n📏 Size: {size}")
        print("-" * 70)
        
        samples_analyzed = 0
        sharpness_scores = []
        contrast_scores = []
        brightness_scores = []
        
        for font_dir in dataset_path.iterdir():
            if not font_dir.is_dir():
                continue
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                size_dir = alphabet_dir / size
                if not size_dir.exists():
                    continue
                
                # Sample first image from this category
                png_files = list(size_dir.glob('*.png'))
                if png_files:
                    try:
                        img = Image.open(png_files[0])
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img_array = np.array(img)
                        
                        # Calculate metrics
                        gray = np.mean(img_array, axis=2)
                        laplacian = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
                        sharpness = np.var(laplacian)
                        contrast = np.std(img_array)
                        brightness = np.mean(img_array)
                        
                        sharpness_scores.append(sharpness)
                        contrast_scores.append(contrast)
                        brightness_scores.append(brightness)
                        samples_analyzed += 1
                        
                    except Exception:
                        pass
        
        if samples_analyzed > 0:
            avg_sharpness = np.mean(sharpness_scores)
            avg_contrast = np.mean(contrast_scores)
            avg_brightness = np.mean(brightness_scores)
            
            print(f"   Samples: {samples_analyzed}")
            print(f"   Sharpness: {avg_sharpness:.2f} {'✅' if avg_sharpness > 50 else '⚠️'}")
            print(f"   Contrast: {avg_contrast:.2f} {'✅' if 30 < avg_contrast < 150 else '⚠️'}")
            print(f"   Brightness: {avg_brightness:.2f} {'✅' if 200 < avg_brightness < 250 else '⚠️'}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Create visual grid
    create_quality_grid()
    
    # Detailed analysis
    analyze_sample_quality()
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Open quality_samples.png to visually inspect samples")
    print("   2. Check quality_report.json for detailed metrics")
    print("   3. Review any ⚠️ warnings in the test results")
    print("   4. If contrast is low, consider adjusting rendering parameters")
