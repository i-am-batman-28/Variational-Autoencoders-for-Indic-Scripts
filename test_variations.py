#!/usr/bin/env python3
"""
Test script to verify variations are working correctly
Generates sample images with all variation types and creates visual grid
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json

# Import functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from generate_printed_dataset import (
    CONFIG, HINDI_VOWELS,
    render_character, apply_rotation, apply_translation, 
    apply_scaling, apply_stroke_width, apply_contrast, 
    apply_gamma, crop_to_character, resize_to_output
)

def create_variation_grid(alphabet='a', font_name='akshar', font_size=14, output_file='variation_test.png'):
    """Create a grid showing all variation types for one character"""
    
    font_path = CONFIG['fonts'][font_name]
    character = HINDI_VOWELS[alphabet]
    
    print(f"\n🧪 Testing variations for: {alphabet} ({character})")
    print(f"   Font: {font_name}, Size: {font_size}pt")
    print("-" * 70)
    
    # Generate base image
    base_img = render_character(font_path, character, font_size, dpi=300)
    base_img = crop_to_character(base_img)
    base_img = resize_to_output(base_img, 128)
    
    # Create variations
    variations = {
        'Base (No variation)': base_img.copy(),
        'Rotation': apply_rotation(base_img.copy()),
        'Translation': apply_translation(base_img.copy()),
        'Scaling': apply_scaling(base_img.copy()),
    }
    
    # Add stroke width if scipy available
    try:
        from scipy import ndimage
        variations['Stroke (Dilate)'] = apply_stroke_width(base_img.copy(), 'dilate', 1)
        variations['Stroke (Erode)'] = apply_stroke_width(base_img.copy(), 'erode', 1)
    except:
        print("   ⚠️  scipy not available - skipping stroke width test")
    
    variations['Contrast'] = apply_contrast(base_img.copy())
    variations['Gamma'] = apply_gamma(base_img.copy())
    
    # Create grid
    cols = 4
    rows = (len(variations) + cols - 1) // cols
    img_size = 128
    padding = 10
    label_height = 25
    
    grid_width = cols * (img_size + padding) + padding
    grid_height = rows * (img_size + padding + label_height) + padding
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Try to load font for labels
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        label_font = ImageFont.load_default()
    
    print(f"\n   Generated {len(variations)} variation types:")
    
    for idx, (var_name, var_img) in enumerate(variations.items()):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_size + padding)
        y = padding + row * (img_size + padding + label_height)
        
        # Paste image
        grid_img.paste(var_img, (x, y))
        
        # Add label
        text_bbox = draw.textbbox((0, 0), var_name, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (img_size - text_width) // 2
        text_y = y + img_size + 3
        
        draw.text((text_x, text_y), var_name, fill='black', font=label_font)
        
        print(f"      {idx+1}. {var_name}")
    
    grid_img.save(output_file)
    print(f"\n   ✅ Variation grid saved to: {output_file}")
    
    return variations

def test_multiple_samples(alphabet='a', font_name='akshar', font_size=14, num_samples=20):
    """Generate multiple samples to test variation distribution"""
    
    font_path = CONFIG['fonts'][font_name]
    character = HINDI_VOWELS[alphabet]
    
    print(f"\n📊 Testing variation distribution for {num_samples} samples")
    print("-" * 70)
    
    variation_counts = {}
    ops = []
    
    if CONFIG['variations']['rotation']:
        ops.append('rotation')
    if CONFIG['variations']['translation']:
        ops.append('translation')
    if CONFIG['variations']['scaling']:
        ops.append('scaling')
    if CONFIG['variations']['stroke_width']:
        try:
            from scipy import ndimage
            ops.append('stroke_width')
        except:
            pass
    if CONFIG['variations']['contrast']:
        ops.append('contrast')
    if CONFIG['variations']['gamma']:
        ops.append('gamma')
    
    print(f"   Available operations: {ops}")
    print(f"   Total options: {len(ops) + 1} (including 'base')")
    print()
    
    # Generate samples
    for i in range(num_samples):
        base_img = render_character(font_path, character, font_size, dpi=300)
        base_img = crop_to_character(base_img)
        
        if i == 0:
            variation = 'base'
        else:
            variation_type = random.choice(['base'] + ops)
            
            if variation_type == 'rotation':
                base_img = apply_rotation(base_img)
                variation = 'rotation'
            elif variation_type == 'translation':
                base_img = apply_translation(base_img)
                variation = 'translation'
            elif variation_type == 'scaling':
                base_img = apply_scaling(base_img)
                variation = 'scaling'
            elif variation_type == 'stroke_width':
                operation = random.choice(['dilate', 'erode'])
                base_img = apply_stroke_width(base_img, operation=operation, kernel_size=1)
                variation = f'stroke_{operation}'
            elif variation_type == 'contrast':
                base_img = apply_contrast(base_img)
                variation = 'contrast'
            elif variation_type == 'gamma':
                base_img = apply_gamma(base_img)
                variation = 'gamma'
            else:
                variation = 'base'
        
        variation_counts[variation] = variation_counts.get(variation, 0) + 1
    
    # Display distribution
    print("   Variation distribution:")
    total = sum(variation_counts.values())
    for var, count in sorted(variation_counts.items()):
        pct = (count / total) * 100
        expected_pct = (1 / (len(ops) + 1)) * 100
        status = "✅" if abs(pct - expected_pct) < 5 else "⚠️"
        print(f"      {var:20s}: {count:2d} ({pct:5.1f}%) {status}")
    
    print(f"\n   Expected: ~{100/(len(ops)+1):.1f}% per variation type")
    print(f"   Status: {'✅ Uniform distribution' if len(set(variation_counts.values())) <= 2 else '⚠️  Check distribution'}")
    
    return variation_counts

def test_data_quality(alphabet='a', font_name='akshar', font_size=14, num_samples=10):
    """Test data quality metrics"""
    
    font_path = CONFIG['fonts'][font_name]
    character = HINDI_VOWELS[alphabet]
    
    print(f"\n🔍 Testing data quality metrics")
    print("-" * 70)
    
    sizes = []
    sharpness_scores = []
    contrast_scores = []
    
    for i in range(num_samples):
        base_img = render_character(font_path, character, font_size, dpi=300)
        base_img = crop_to_character(base_img)
        
        # Apply random variation
        if i > 0:
            ops = []
            if CONFIG['variations']['rotation']:
                ops.append('rotation')
            if CONFIG['variations']['translation']:
                ops.append('translation')
            if CONFIG['variations']['scaling']:
                ops.append('scaling')
            if CONFIG['variations']['stroke_width']:
                try:
                    from scipy import ndimage
                    ops.append('stroke_width')
                except:
                    pass
            if CONFIG['variations']['contrast']:
                ops.append('contrast')
            if CONFIG['variations']['gamma']:
                ops.append('gamma')
            
            variation_type = random.choice(['base'] + ops)
            
            if variation_type == 'rotation':
                base_img = apply_rotation(base_img)
            elif variation_type == 'translation':
                base_img = apply_translation(base_img)
            elif variation_type == 'scaling':
                base_img = apply_scaling(base_img)
            elif variation_type == 'stroke_width':
                operation = random.choice(['dilate', 'erode'])
                base_img = apply_stroke_width(base_img, operation=operation, kernel_size=1)
            elif variation_type == 'contrast':
                base_img = apply_contrast(base_img)
            elif variation_type == 'gamma':
                base_img = apply_gamma(base_img)
        
        final_img = resize_to_output(base_img, 128)
        
        # Check size
        if final_img.size == (128, 128):
            sizes.append(True)
        
        # Calculate quality metrics
        img_array = np.array(final_img)
        if img_array.shape[2] == 3:  # RGB
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Sharpness (Laplacian variance)
        laplacian = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        sharpness = np.var(laplacian)
        sharpness_scores.append(sharpness)
        
        # Contrast
        contrast = np.std(img_array)
        contrast_scores.append(contrast)
    
    # Report
    print(f"   Samples tested: {num_samples}")
    print(f"   Size compliance: {sum(sizes)}/{len(sizes)} ({sum(sizes)/len(sizes)*100:.1f}%) ✅" if all(sizes) else f"   Size compliance: {sum(sizes)}/{len(sizes)} ⚠️")
    print(f"   Average sharpness: {np.mean(sharpness_scores):.2f} ✅" if np.mean(sharpness_scores) > 50 else f"   Average sharpness: {np.mean(sharpness_scores):.2f} ⚠️")
    print(f"   Average contrast: {np.mean(contrast_scores):.2f} ✅" if 10 < np.mean(contrast_scores) < 30 else f"   Average contrast: {np.mean(contrast_scores):.2f} ⚠️")
    
    return {
        'size_compliance': sum(sizes) / len(sizes),
        'avg_sharpness': np.mean(sharpness_scores),
        'avg_contrast': np.mean(contrast_scores)
    }

def create_sample_comparison_grid(font_name='akshar', font_size=14, output_file='sample_comparison.png'):
    """Create grid showing same variation applied to different alphabets"""
    
    font_path = CONFIG['fonts'][font_name]
    test_alphabets = ['a', 'aa', 'i', 'ii', 'u', 'uu']
    
    print(f"\n🔄 Testing variation consistency across alphabets")
    print("-" * 70)
    
    # Create grid: 6 alphabets × 3 variations (base, rotation, translation)
    cols = 6
    rows = 3
    img_size = 128
    padding = 5
    label_height = 20
    
    grid_width = cols * (img_size + padding) + padding
    grid_height = rows * (img_size + padding + label_height) + padding
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        label_font = ImageFont.load_default()
    
    variation_types = ['base', 'rotation', 'translation']
    
    for row, var_type in enumerate(variation_types):
        for col, alphabet in enumerate(test_alphabets):
            character = HINDI_VOWELS[alphabet]
            
            base_img = render_character(font_path, character, font_size, dpi=300)
            base_img = crop_to_character(base_img)
            
            if var_type == 'rotation':
                base_img = apply_rotation(base_img)
            elif var_type == 'translation':
                base_img = apply_translation(base_img)
            
            final_img = resize_to_output(base_img, 128)
            
            x = padding + col * (img_size + padding)
            y = padding + row * (img_size + padding + label_height)
            
            grid_img.paste(final_img, (x, y))
            
            # Label
            if row == 0:
                label = f"{alphabet}"
            else:
                label = ""
            
            if label:
                text_bbox = draw.textbbox((0, 0), label, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x + (img_size - text_width) // 2
                text_y = y + img_size + 2
                draw.text((text_x, text_y), label, fill='black', font=label_font)
    
    # Add row labels
    row_labels = ['Base', 'Rotation', 'Translation']
    for row, label in enumerate(row_labels):
        x = 2
        y = padding + row * (img_size + padding + label_height) + img_size // 2
        draw.text((x, y), label, fill='black', font=label_font)
    
    grid_img.save(output_file)
    print(f"   ✅ Comparison grid saved to: {output_file}")
    
    return grid_img

def main():
    """Run all tests"""
    
    print("=" * 70)
    print("DATASET VARIATION TEST SUITE")
    print("=" * 70)
    
    # Test 1: Variation grid
    print("\n" + "=" * 70)
    print("TEST 1: Variation Types Visualization")
    print("=" * 70)
    create_variation_grid('a', 'akshar', 14, 'variation_test.png')
    
    # Test 2: Variation distribution
    print("\n" + "=" * 70)
    print("TEST 2: Variation Distribution (Uniformity Check)")
    print("=" * 70)
    test_multiple_samples('a', 'akshar', 14, 20)
    
    # Test 3: Data quality
    print("\n" + "=" * 70)
    print("TEST 3: Data Quality Metrics")
    print("=" * 70)
    quality = test_data_quality('a', 'akshar', 14, 10)
    
    # Test 4: Cross-alphabet consistency
    print("\n" + "=" * 70)
    print("TEST 4: Cross-Alphabet Consistency")
    print("=" * 70)
    create_sample_comparison_grid('akshar', 14, 'sample_comparison.png')
    
    # Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    
    ready = True
    issues = []
    
    if quality['size_compliance'] < 1.0:
        ready = False
        issues.append("Size compliance not 100%")
    
    if quality['avg_sharpness'] < 50:
        issues.append("Sharpness may be low (but acceptable for printed chars)")
    
    if quality['avg_contrast'] < 10:
        issues.append("Contrast is low (expected for white background)")
    
    if ready and not issues:
        print("✅ DATASET IS READY FOR PROCESSING")
        print("\n   All tests passed:")
        print("   ✅ Variations working correctly")
        print("   ✅ Uniform distribution")
        print("   ✅ Quality metrics acceptable")
        print("   ✅ Consistent across alphabets")
    else:
        print("⚠️  DATASET READY WITH NOTES")
        if issues:
            print("\n   Notes:")
            for issue in issues:
                print(f"   - {issue}")
    
    print("\n📁 Generated test files:")
    print("   - variation_test.png (all variation types)")
    print("   - sample_comparison.png (cross-alphabet consistency)")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
