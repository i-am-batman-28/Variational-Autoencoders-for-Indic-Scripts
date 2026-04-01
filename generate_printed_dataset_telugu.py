#!/usr/bin/env python3
"""
Production-grade dataset generator for printed Telugu characters
Fonts: Pothana2000, Lohit Telugu Regular
Alphabets: 13 vowels (a to aha)
Output: dataset_Rohit
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import random
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed. Stroke width variation will be disabled.")
    print("   Install with: pip install scipy")

# Configuration
CONFIG = {
    'fonts': {
        'pothana2000': '/Users/karthiksarma/Library/Fonts/Pothana2000 Pothana2000.ttf',
        'lohit_telugu': '/Users/karthiksarma/Library/Fonts/Lohit Telugu Regular.ttf'
    },
    'alphabets': ['a', 'aa', 'i', 'ii', 'u', 'uu', 'ri', 'e', 'ai', 'o', 'au', 'am', 'aha'],
    'font_sizes': {
        'small': 10,      # 20 samples per size
        'medium': 14,     # 20 samples per size
        'large': 18,      # 20 samples per size
        'xlarge': 22      # 20 samples per size
    },
    'samples_per_size': 20,  # 20 samples per font size
    'samples_per_alphabet': 80,  # 4 sizes × 20 = 80 per alphabet per font
    'output_size': 128,
    'render_dpi': 300,  # Fixed DPI - variation removed for consistency
    'output_dir': 'dataset_Rohit',
    'variations': {
        'rotation': True,        # ±3° - simulates printing misalignment
        'translation': True,     # ±3 px - increased for larger character size
        'scaling': True,         # 0.94–1.06 - slightly wider range for larger characters
        'stroke_width': True,    # erosion/dilation (kernel_size=2) - more visible on larger chars
        'contrast': True,        # 0.95–1.05 - ink density variation
        'gamma': True            # 0.95–1.05 - brightness shift (better than noise)
    }
}

# Telugu vowel characters (Telugu Unicode)
# 13 vowels in same order as Hindi script (a to aha)
TELUGU_VOWELS = {
    'a': '\u0C05',      # అ (A)
    'aa': '\u0C06',     # ఆ (AA)
    'i': '\u0C07',      # ఇ (I)
    'ii': '\u0C08',     # ఈ (II)
    'u': '\u0C09',      # ఉ (U)
    'uu': '\u0C0A',     # ఊ (UU)
    'ri': '\u0C0B',     # ఋ (RI)
    'e': '\u0C0E',      # ఎ (E)
    'ai': '\u0C10',     # ఐ (AI)
    'o': '\u0C12',      # ఒ (O)
    'au': '\u0C14',     # ఔ (AU)
    'am': '\u0C05\u0C02',  # అం (AM) - అ + anusvara
    'aha': '\u0C05\u0C03'  # అః (AHA) - అ + visarga
}

def verify_fonts():
    """Verify that fonts are available"""
    missing = []
    for font_name, font_path in CONFIG['fonts'].items():
        if not os.path.exists(font_path):
            missing.append(f"{font_name}: {font_path}")
    
    if missing:
        print("⚠️  Missing fonts:")
        for m in missing:
            print(f"   {m}")
        return False
    else:
        print("✅ All fonts found")
        return True

def render_character(font_path, character, font_size, dpi=300, scale_factor=3.0):
    """
    Render a character at high resolution with larger scale
    scale_factor: Multiplier to make character larger (default 3.0 for bigger characters)
    """
    # Calculate base size
    base_font_size = int(font_size * dpi / 72)
    
    # Apply scale factor to make character larger
    scaled_font_size = int(base_font_size * scale_factor)
    
    # Create canvas - even tighter fit (1.10x for minimal background)
    canvas_size = int(scaled_font_size * 1.10)
    img = Image.new('RGB', (canvas_size, canvas_size), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, scaled_font_size)
    except:
        # Fallback to default font if custom font fails
        font = ImageFont.load_default()
        print(f"⚠️  Could not load font: {font_path}, using default")
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (canvas_size - text_width) // 2 - bbox[0]
    y = (canvas_size - text_height) // 2 - bbox[1]
    
    # Draw the character
    draw.text((x, y), character, font=font, fill='black')
    
    return img

def apply_rotation(img, angle_range=(-3, 3)):
    """Apply slight rotation"""
    angle = random.uniform(angle_range[0], angle_range[1])
    return img.rotate(angle, fillcolor='white', resample=Image.Resampling.BICUBIC)

def apply_translation(img, max_offset=3):
    """Apply slight translation - increased for larger character size"""
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    
    new_img = Image.new('RGB', img.size, color='white')
    new_img.paste(img, (offset_x, offset_y))
    return new_img

def apply_scaling(img, scale_range=(0.94, 1.06)):
    """Apply slight scaling - slightly wider range for larger characters"""
    scale = random.uniform(scale_range[0], scale_range[1])
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    scaled = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center on original canvas
    new_img = Image.new('RGB', img.size, color='white')
    paste_x = (img.size[0] - scaled.size[0]) // 2
    paste_y = (img.size[1] - scaled.size[1]) // 2
    new_img.paste(scaled, (paste_x, paste_y))
    return new_img

def apply_stroke_width(img, operation='dilate', kernel_size=2):
    """
    Apply stroke thickening (dilation) or thinning (erosion)
    Simulates printer/ink variation in stroke width
    Increased kernel_size for larger characters (more visible effect)
    """
    if not HAS_SCIPY:
        return img  # Return unchanged if scipy not available
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        gray = img.convert('L')
    else:
        gray = img
    
    img_array = np.array(gray)
    
    # Create small kernel for morphological operations
    kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), dtype=np.uint8)
    
    if operation == 'dilate':
        # Thicken strokes (dilation)
        processed = ndimage.binary_dilation(img_array < 128, structure=kernel)
    else:  # 'erode'
        # Thin strokes (erosion)
        processed = ndimage.binary_erosion(img_array < 128, structure=kernel)
    
    # Convert back to image
    result_array = np.where(processed, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_array, mode='L')
    
    # Convert back to RGB if original was RGB
    if img.mode == 'RGB':
        result_img = result_img.convert('RGB')
    
    return result_img

def apply_gamma(img, gamma_range=(0.95, 1.05)):
    """
    Apply gamma correction for brightness shift
    Better than noise - preserves glyph geometry
    """
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Apply gamma correction
    img_array = 255.0 * np.power(img_array / 255.0, gamma)
    
    # Clip and convert back
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def apply_contrast(img, factor_range=(0.95, 1.05)):
    """Adjust contrast slightly"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def image_hash(img):
    """Generate hash of image for duplicate detection"""
    img_array = np.array(img)
    img_bytes = img_array.tobytes()
    return hashlib.md5(img_bytes).hexdigest()

def crop_to_character(img, padding=0):
    """Crop image to character bounds with zero padding (safe - we re-pad during resize)"""
    # Convert to grayscale for bounding box detection
    gray = img.convert('L')
    bbox = gray.getbbox()
    
    if bbox:
        # Zero padding - tight crop (safe because we re-pad during resize)
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img.size[0], bbox[2] + padding)
        y1 = min(img.size[1], bbox[3] + padding)
        
        cropped = img.crop((x0, y0, x1, y1))
        return cropped
    return img

def resize_to_output(img, target_size=128, fill_percentage=0.997):
    """
    Resize to final output size - character fills almost entire canvas
    fill_percentage: 0.997 = character fills 99.7% of canvas (~0.3% margin = ~0.4px border - visually edge-to-edge)
    """
    # Calculate target size to fill almost entire canvas
    target_char_size = int(target_size * fill_percentage)
    
    # Calculate scale factor - use MAX to fill as much as possible
    scale_w = target_char_size / img.size[0]
    scale_h = target_char_size / img.size[1]
    scale = max(scale_w, scale_h)  # Use LARGER scale - one dimension will fill 98%
    
    # Resize maintaining aspect ratio
    new_width = int(img.size[0] * scale)
    new_height = int(img.size[1] * scale)
    
    # Ensure we don't exceed target_size (safety check)
    if new_width > target_size:
        scale = target_size / img.size[0]
        new_width = target_size
        new_height = int(img.size[1] * scale)
    if new_height > target_size:
        scale = target_size / img.size[1]
        new_height = target_size
        new_width = int(img.size[0] * scale)
    
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create square canvas
    new_img = Image.new('RGB', (target_size, target_size), color='white')
    
    # Center the resized image
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_img.paste(resized, (paste_x, paste_y))
    
    return new_img

def generate_dataset():
    """Generate the complete dataset"""
    
    print("=" * 70)
    print("PRODUCTION-GRADE TELUGU DATASET GENERATION (dataset_Rohit)")
    print("=" * 70)
    
    # Verify fonts
    if not verify_fonts():
        print("\n❌ Font verification failed. Please check font paths.")
        return
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    metadata = []
    total_samples = 0
    
    # Font size configuration
    font_sizes = CONFIG['font_sizes']
    samples_per_size = CONFIG['samples_per_size']
    
    print(f"\n📊 Configuration:")
    print(f"   Script: Telugu | Output: {CONFIG['output_dir']}")
    print(f"   Font sizes: {list(font_sizes.values())}pt")
    print(f"   Samples per size: {samples_per_size}")
    print(f"   Samples per alphabet per font: {len(font_sizes) * samples_per_size}")
    print(f"   Output size: {CONFIG['output_size']}x{CONFIG['output_size']}")
    total_images = len(CONFIG['alphabets']) * len(CONFIG['fonts']) * len(font_sizes) * samples_per_size
    print(f"   Total images: {total_images}")
    print()
    
    # Process each font
    for font_name, font_path in CONFIG['fonts'].items():
        print(f"\n🔤 Processing font: {font_name.upper()}")
        print("-" * 70)
        
        font_dir = output_dir / font_name
        font_dir.mkdir(exist_ok=True)
        
        # Process each alphabet
        for alphabet in tqdm(CONFIG['alphabets'], desc=f"  {font_name}"):
            alphabet_dir = font_dir / alphabet
            alphabet_dir.mkdir(exist_ok=True)
            
            character = TELUGU_VOWELS.get(alphabet, alphabet)
            sample_count = 0
            
            # Generate samples for each font size
            for size_name, size_pt in font_sizes.items():
                
                # Create size-specific subdirectory
                size_dir = alphabet_dir / f"{size_pt}pt"
                size_dir.mkdir(exist_ok=True)
                
                # BEST STRATEGY: Exactly ONE clean base, remaining 19 with augmentations
                # Hash-based duplicate detection ensures uniqueness
                
                seen_hashes = set()
                max_attempts = 50  # Safety limit for regeneration attempts
                
                for i in range(samples_per_size):
                    attempts = 0
                    unique = False
                    
                    while not unique and attempts < max_attempts:
                        attempts += 1
                        
                        # Render base character at fixed DPI (300) with much larger scale
                        # scale_factor=20.0 makes character fill ~95-99% of 128x128 canvas
                        base_img = render_character(font_path, character, size_pt, dpi=300, scale_factor=20.0)
                        base_img = crop_to_character(base_img)
                        
                        # First sample: exactly ONE clean base (no variation)
                        if i == 0:
                            final_img = base_img
                            variation = 'base'
                            variation_list = ['base']
                        else:
                            # Remaining 19 samples: apply at least one augmentation
                            # Allow combinations for natural variation
                            var_img = base_img.copy()
                            variation_list = []
                            
                            # Build available operations
                            available_ops = []
                            if CONFIG['variations']['rotation']:
                                available_ops.append('rotation')
                            if CONFIG['variations']['translation']:
                                available_ops.append('translation')
                            if CONFIG['variations']['scaling']:
                                available_ops.append('scaling')
                            if CONFIG['variations']['stroke_width'] and HAS_SCIPY:
                                available_ops.append('stroke_width')
                            if CONFIG['variations']['contrast']:
                                available_ops.append('contrast')
                            if CONFIG['variations']['gamma']:
                                available_ops.append('gamma')
                            
                            # Apply at least one augmentation (random selection)
                            # Allow 1-3 augmentations per sample for natural combinations
                            num_augs = random.randint(1, min(3, len(available_ops)))
                            selected_ops = random.sample(available_ops, num_augs)
                            
                            # Apply selected augmentations in random order
                            random.shuffle(selected_ops)
                            
                            for op in selected_ops:
                                if op == 'rotation':
                                    var_img = apply_rotation(var_img)
                                    variation_list.append('rotation')
                                elif op == 'translation':
                                    var_img = apply_translation(var_img)
                                    variation_list.append('translation')
                                elif op == 'scaling':
                                    var_img = apply_scaling(var_img)
                                    variation_list.append('scaling')
                                elif op == 'stroke_width':
                                    operation = random.choice(['dilate', 'erode'])
                                    var_img = apply_stroke_width(var_img, operation=operation, kernel_size=2)
                                    variation_list.append(f'stroke_{operation}')
                                elif op == 'contrast':
                                    var_img = apply_contrast(var_img)
                                    variation_list.append('contrast')
                                elif op == 'gamma':
                                    var_img = apply_gamma(var_img)
                                    variation_list.append('gamma')
                            
                            final_img = var_img
                            variation = '+'.join(variation_list)
                        
                        # Resize to output size - maximize character size (99.7% fill = ~0.4px border - edge-to-edge)
                        final_img = resize_to_output(final_img, CONFIG['output_size'], fill_percentage=0.997)
                        
                        # Hash-based duplicate detection
                        img_hash = image_hash(final_img)
                        
                        if img_hash not in seen_hashes:
                            seen_hashes.add(img_hash)
                            unique = True
                        else:
                            # Duplicate detected - regenerate
                            if attempts < max_attempts:
                                continue
                            else:
                                # Safety: use this even if duplicate (shouldn't happen often)
                                unique = True
                    
                    # Save unique sample
                    sample_id = f"{alphabet}_{size_pt}pt_{i+1:02d}"
                    filename = f"{sample_id}.png"
                    filepath = size_dir / filename
                    final_img.save(filepath, 'PNG')
                    
                    # Record metadata
                    metadata.append({
                        'font': font_name,
                        'alphabet': alphabet,
                        'character': character,
                        'sample_id': sample_id,
                        'filename': str(filepath.relative_to(output_dir)),
                        'font_size_pt': size_pt,
                        'font_size_name': size_name,
                        'size_index': i + 1,
                        'dpi': 300,  # Fixed DPI for consistency
                        'variation': variation,
                        'variations_applied': variation_list,
                        'image_hash': img_hash,
                        'regeneration_attempts': attempts,
                        'output_size': CONFIG['output_size'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    sample_count += 1
                    total_samples += 1
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"✅ Total samples generated: {total_samples}")
    print(f"✅ Output directory: {CONFIG['output_dir']}")
    print(f"✅ Metadata saved: {metadata_file}")
    print("=" * 70)
    
    # Print statistics
    print("\n📊 Statistics:")
    for font_name in CONFIG['fonts'].keys():
        font_samples = sum(1 for m in metadata if m['font'] == font_name)
        print(f"   {font_name}: {font_samples} samples")
    
    samples_per_alphabet = len(font_sizes) * samples_per_size
    print(f"\n   Per size: {samples_per_size} samples")
    print(f"   Per alphabet per font: {samples_per_alphabet} samples ({len(font_sizes)} sizes × {samples_per_size})")
    print(f"   Per font: {samples_per_alphabet * len(CONFIG['alphabets'])} samples")
    print(f"   Total: {total_samples} samples")

if __name__ == "__main__":
    # Test with one sample first
    print("Testing Telugu font rendering...")
    test_char = TELUGU_VOWELS['a']
    for font_name, font_path in CONFIG['fonts'].items():
        if os.path.exists(font_path):
            test_img = render_character(font_path, test_char, 12, 300)
            test_img = crop_to_character(test_img)
            test_img = resize_to_output(test_img, 128)
            test_path = f"test_telugu_{font_name}.png"
            test_img.save(test_path)
            print(f"✅ Test image saved: {test_path}")
    
    print("\n" + "=" * 70)
    print("Fonts verified. Generating full Telugu dataset (dataset_Rohit)...")
    generate_dataset()
