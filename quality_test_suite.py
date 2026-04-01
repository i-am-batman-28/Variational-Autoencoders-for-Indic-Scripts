#!/usr/bin/env python3
"""
Production-Grade Quality Test Suite for Dataset
Comprehensive quality verification across multiple dimensions
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import statistics

class DatasetQualityTester:
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.results = {
            'summary': {},
            'image_quality': {},
            'dataset_completeness': {},
            'statistical_analysis': {},
            'visual_quality': {},
            'issues': []
        }
        
    def run_all_tests(self):
        """Run all quality tests"""
        print("=" * 70)
        print("PRODUCTION-GRADE DATASET QUALITY TEST SUITE")
        print("=" * 70)
        print()
        
        # Load metadata if available
        metadata_file = self.dataset_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
            print("⚠️  No metadata.json found. Some tests may be limited.")
        
        # Run all tests
        print("🔍 Running quality tests...\n")
        
        self.test_dataset_structure()
        self.test_image_properties()
        self.test_image_quality_metrics()
        self.test_dataset_completeness()
        self.test_statistical_distribution()
        self.test_visual_quality()
        self.test_consistency()
        
        # Generate report
        self.generate_report()
        
    def test_dataset_structure(self):
        """Test 1: Dataset Structure"""
        print("📁 Test 1: Dataset Structure")
        print("-" * 70)
        
        expected_fonts = ['akshar', 'mangal']
        expected_alphabets = ['a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ee', 'ai', 'o', 'oo', 'aha']
        expected_sizes = ['10pt', '14pt', '18pt', '22pt']
        
        structure_ok = True
        found_fonts = []
        found_alphabets = defaultdict(set)
        found_sizes = defaultdict(set)
        
        for font_dir in self.dataset_dir.iterdir():
            if font_dir.is_dir() and font_dir.name.lower() in [f.lower() for f in expected_fonts]:
                font_name = font_dir.name.lower()
                found_fonts.append(font_name)
                
                for alphabet_dir in font_dir.iterdir():
                    if alphabet_dir.is_dir():
                        alphabet = alphabet_dir.name
                        found_alphabets[font_name].add(alphabet)
                        
                        for size_dir in alphabet_dir.iterdir():
                            if size_dir.is_dir() and size_dir.name in expected_sizes:
                                found_sizes[font_name].add(size_dir.name)
        
        # Check fonts
        missing_fonts = set(expected_fonts) - set(found_fonts)
        if missing_fonts:
            self.results['issues'].append(f"Missing fonts: {missing_fonts}")
            structure_ok = False
        
        # Check alphabets
        for font in expected_fonts:
            if font in found_alphabets:
                missing_alphabets = set(expected_alphabets) - found_alphabets[font]
                if missing_alphabets:
                    self.results['issues'].append(f"Font {font}: Missing alphabets {missing_alphabets}")
                    structure_ok = False
        
        # Check sizes
        for font in expected_fonts:
            if font in found_sizes:
                missing_sizes = set(expected_sizes) - found_sizes[font]
                if missing_sizes:
                    self.results['issues'].append(f"Font {font}: Missing sizes {missing_sizes}")
                    structure_ok = False
        
        self.results['dataset_completeness']['structure'] = {
            'fonts_found': found_fonts,
            'alphabets_per_font': {f: len(found_alphabets.get(f, set())) for f in expected_fonts},
            'sizes_per_font': {f: len(found_sizes.get(f, set())) for f in expected_fonts},
            'status': 'PASS' if structure_ok else 'FAIL'
        }
        
        status = "✅ PASS" if structure_ok else "❌ FAIL"
        print(f"   Structure: {status}")
        print(f"   Fonts found: {found_fonts}")
        print()
        
    def test_image_properties(self):
        """Test 2: Image Properties (Size, Format, Mode)"""
        print("🖼️  Test 2: Image Properties")
        print("-" * 70)
        
        image_stats = {
            'total_images': 0,
            'correct_size': 0,
            'correct_format': 0,
            'correct_mode': 0,
            'size_issues': [],
            'format_issues': [],
            'mode_issues': []
        }
        
        expected_size = (128, 128)
        expected_format = 'PNG'
        expected_mode = 'RGB'
        
        # Sample images from each category
        sample_count = 0
        max_samples = 100  # Sample up to 100 images for speed
        
        for font_dir in self.dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                for size_dir in alphabet_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    for img_file in size_dir.glob('*.png'):
                        if sample_count >= max_samples:
                            break
                        
                        try:
                            img = Image.open(img_file)
                            image_stats['total_images'] += 1
                            sample_count += 1
                            
                            # Check size
                            if img.size == expected_size:
                                image_stats['correct_size'] += 1
                            else:
                                image_stats['size_issues'].append(f"{img_file.name}: {img.size}")
                            
                            # Check format
                            if img.format == expected_format:
                                image_stats['correct_format'] += 1
                            else:
                                image_stats['format_issues'].append(f"{img_file.name}: {img.format}")
                            
                            # Check mode
                            if img.mode == expected_mode:
                                image_stats['correct_mode'] += 1
                            else:
                                image_stats['mode_issues'].append(f"{img_file.name}: {img.mode}")
                                
                        except Exception as e:
                            self.results['issues'].append(f"Error reading {img_file}: {e}")
                        
                        if sample_count >= max_samples:
                            break
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        
        # Calculate percentages
        if image_stats['total_images'] > 0:
            size_pct = (image_stats['correct_size'] / image_stats['total_images']) * 100
            format_pct = (image_stats['correct_format'] / image_stats['total_images']) * 100
            mode_pct = (image_stats['correct_mode'] / image_stats['total_images']) * 100
        else:
            size_pct = format_pct = mode_pct = 0
        
        self.results['image_quality']['properties'] = {
            'samples_tested': image_stats['total_images'],
            'size_compliance': f"{size_pct:.1f}%",
            'format_compliance': f"{format_pct:.1f}%",
            'mode_compliance': f"{mode_pct:.1f}%",
            'issues': len(image_stats['size_issues']) + len(image_stats['format_issues']) + len(image_stats['mode_issues'])
        }
        
        print(f"   Images tested: {image_stats['total_images']}")
        print(f"   Size (128x128): {size_pct:.1f}% ✅" if size_pct == 100 else f"   Size (128x128): {size_pct:.1f}% ⚠️")
        print(f"   Format (PNG): {format_pct:.1f}% ✅" if format_pct == 100 else f"   Format (PNG): {format_pct:.1f}% ⚠️")
        print(f"   Mode (RGB): {mode_pct:.1f}% ✅" if mode_pct == 100 else f"   Mode (RGB): {mode_pct:.1f}% ⚠️")
        print()
        
    def test_image_quality_metrics(self):
        """Test 3: Image Quality Metrics (Sharpness, Contrast, Brightness)"""
        print("📊 Test 3: Image Quality Metrics")
        print("-" * 70)
        
        quality_metrics = {
            'sharpness_scores': [],
            'contrast_scores': [],
            'brightness_scores': [],
            'centering_scores': []
        }
        
        sample_count = 0
        max_samples = 50
        
        for font_dir in self.dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                for size_dir in alphabet_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    for img_file in size_dir.glob('*.png'):
                        if sample_count >= max_samples:
                            break
                        
                        try:
                            img = Image.open(img_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Convert to numpy array
                            img_array = np.array(img)
                            
                            # Sharpness (Laplacian variance)
                            gray = np.mean(img_array, axis=2)
                            laplacian = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
                            sharpness = np.var(laplacian)
                            quality_metrics['sharpness_scores'].append(sharpness)
                            
                            # Contrast (standard deviation)
                            contrast = np.std(img_array)
                            quality_metrics['contrast_scores'].append(contrast)
                            
                            # Brightness (mean)
                            brightness = np.mean(img_array)
                            quality_metrics['brightness_scores'].append(brightness)
                            
                            # Centering (check if character is centered)
                            # Assume white background, character is darker
                            threshold = 200
                            non_white = np.sum(img_array < threshold, axis=2) > 0
                            if np.any(non_white):
                                y_coords, x_coords = np.where(non_white)
                                center_x, center_y = 64, 64  # Expected center
                                actual_center_x = np.mean(x_coords)
                                actual_center_y = np.mean(y_coords)
                                offset = np.sqrt((center_x - actual_center_x)**2 + (center_y - actual_center_y)**2)
                                centering_score = 1.0 / (1.0 + offset / 10.0)  # Normalize
                                quality_metrics['centering_scores'].append(centering_score)
                            
                            sample_count += 1
                            
                        except Exception as e:
                            pass
                        
                        if sample_count >= max_samples:
                            break
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        
        # Calculate statistics
        metrics_summary = {}
        for metric_name, scores in quality_metrics.items():
            if scores:
                metrics_summary[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                }
        
        self.results['image_quality']['metrics'] = metrics_summary
        
        print(f"   Samples analyzed: {sample_count}")
        if quality_metrics['sharpness_scores']:
            avg_sharpness = np.mean(quality_metrics['sharpness_scores'])
            print(f"   Average Sharpness: {avg_sharpness:.2f} {'✅' if avg_sharpness > 50 else '⚠️'}")
        if quality_metrics['contrast_scores']:
            avg_contrast = np.mean(quality_metrics['contrast_scores'])
            print(f"   Average Contrast: {avg_contrast:.2f} {'✅' if 30 < avg_contrast < 150 else '⚠️'}")
        if quality_metrics['brightness_scores']:
            avg_brightness = np.mean(quality_metrics['brightness_scores'])
            print(f"   Average Brightness: {avg_brightness:.2f} {'✅' if 200 < avg_brightness < 250 else '⚠️'}")
        if quality_metrics['centering_scores']:
            avg_centering = np.mean(quality_metrics['centering_scores'])
            print(f"   Average Centering: {avg_centering:.2f} {'✅' if avg_centering > 0.7 else '⚠️'}")
        print()
        
    def test_dataset_completeness(self):
        """Test 4: Dataset Completeness (Count files)"""
        print("📋 Test 4: Dataset Completeness")
        print("-" * 70)
        
        expected_count = 12 * 2 * 4 * 20  # alphabets × fonts × sizes × samples
        actual_count = 0
        count_by_font = defaultdict(int)
        count_by_size = defaultdict(int)
        count_by_alphabet = defaultdict(int)
        
        for font_dir in self.dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            font_name = font_dir.name.lower()
            
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                alphabet = alphabet_dir.name
                
                for size_dir in alphabet_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    size = size_dir.name
                    
                    png_count = len(list(size_dir.glob('*.png')))
                    actual_count += png_count
                    count_by_font[font_name] += png_count
                    count_by_size[size] += png_count
                    count_by_alphabet[alphabet] += png_count
                    
                    # Check if count matches expected
                    if png_count != 20:
                        self.results['issues'].append(
                            f"{font_name}/{alphabet}/{size}: Expected 20, found {png_count}"
                        )
        
        completeness_pct = (actual_count / expected_count * 100) if expected_count > 0 else 0
        
        self.results['dataset_completeness']['counts'] = {
            'expected': expected_count,
            'actual': actual_count,
            'completeness': f"{completeness_pct:.1f}%",
            'by_font': dict(count_by_font),
            'by_size': dict(count_by_size),
            'status': 'PASS' if actual_count == expected_count else 'FAIL'
        }
        
        print(f"   Expected: {expected_count} images")
        print(f"   Actual: {actual_count} images")
        print(f"   Completeness: {completeness_pct:.1f}% {'✅' if completeness_pct == 100 else '⚠️'}")
        print(f"   By font: {dict(count_by_font)}")
        print(f"   By size: {dict(count_by_size)}")
        print()
        
    def test_statistical_distribution(self):
        """Test 5: Statistical Distribution Analysis"""
        print("📈 Test 5: Statistical Distribution")
        print("-" * 70)
        
        if not self.metadata:
            print("   ⚠️  Skipped (no metadata)")
            print()
            return
        
        # Analyze distribution
        font_dist = defaultdict(int)
        size_dist = defaultdict(int)
        alphabet_dist = defaultdict(int)
        
        for entry in self.metadata:
            font_dist[entry.get('font', 'unknown')] += 1
            size_dist[entry.get('font_size_pt', 'unknown')] += 1
            alphabet_dist[entry.get('alphabet', 'unknown')] += 1
        
        # Check balance
        font_balance = all(count == font_dist[list(font_dist.keys())[0]] for count in font_dist.values())
        size_balance = all(count == size_dist[list(size_dist.keys())[0]] for count in size_dist.values())
        
        self.results['statistical_analysis'] = {
            'font_distribution': dict(font_dist),
            'size_distribution': dict(size_dist),
            'alphabet_distribution': dict(alphabet_dist),
            'font_balanced': font_balance,
            'size_balanced': size_balance
        }
        
        print(f"   Font distribution: {dict(font_dist)}")
        print(f"   Size distribution: {dict(size_dist)}")
        print(f"   Font balance: {'✅' if font_balance else '⚠️'}")
        print(f"   Size balance: {'✅' if size_balance else '⚠️'}")
        print()
        
    def test_visual_quality(self):
        """Test 6: Visual Quality (Readability, Artifacts)"""
        print("👁️  Test 6: Visual Quality Assessment")
        print("-" * 70)
        
        # Sample images and check for common issues
        issues_found = {
            'too_dark': 0,
            'too_bright': 0,
            'low_contrast': 0,
            'possible_artifacts': 0
        }
        
        sample_count = 0
        max_samples = 30
        
        for font_dir in self.dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                for size_dir in alphabet_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    for img_file in size_dir.glob('*.png'):
                        if sample_count >= max_samples:
                            break
                        
                        try:
                            img = Image.open(img_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            img_array = np.array(img)
                            
                            # Check brightness
                            mean_brightness = np.mean(img_array)
                            if mean_brightness < 150:
                                issues_found['too_dark'] += 1
                            elif mean_brightness > 250:
                                issues_found['too_bright'] += 1
                            
                            # Check contrast
                            std_contrast = np.std(img_array)
                            if std_contrast < 20:
                                issues_found['low_contrast'] += 1
                            
                            # Check for artifacts (unusual patterns)
                            # Simple check: look for extreme values
                            if np.any(img_array < 10) or np.any(img_array > 245):
                                issues_found['possible_artifacts'] += 1
                            
                            sample_count += 1
                            
                        except Exception:
                            pass
                        
                        if sample_count >= max_samples:
                            break
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        
        self.results['visual_quality'] = {
            'samples_checked': sample_count,
            'issues': issues_found,
            'quality_score': 'GOOD' if sum(issues_found.values()) < sample_count * 0.1 else 'NEEDS_REVIEW'
        }
        
        print(f"   Samples checked: {sample_count}")
        print(f"   Too dark: {issues_found['too_dark']} {'⚠️' if issues_found['too_dark'] > 0 else '✅'}")
        print(f"   Too bright: {issues_found['too_bright']} {'⚠️' if issues_found['too_bright'] > 0 else '✅'}")
        print(f"   Low contrast: {issues_found['low_contrast']} {'⚠️' if issues_found['low_contrast'] > 0 else '✅'}")
        print(f"   Possible artifacts: {issues_found['possible_artifacts']} {'⚠️' if issues_found['possible_artifacts'] > 0 else '✅'}")
        print()
        
    def test_consistency(self):
        """Test 7: Consistency Across Dataset"""
        print("🔄 Test 7: Consistency Check")
        print("-" * 70)
        
        file_sizes = []
        sample_count = 0
        max_samples = 100
        
        for font_dir in self.dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            for alphabet_dir in font_dir.iterdir():
                if not alphabet_dir.is_dir():
                    continue
                for size_dir in alphabet_dir.iterdir():
                    if not size_dir.is_dir():
                        continue
                    for img_file in size_dir.glob('*.png'):
                        if sample_count >= max_samples:
                            break
                        file_sizes.append(img_file.stat().st_size)
                        sample_count += 1
                        if sample_count >= max_samples:
                            break
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        
        if file_sizes:
            avg_size = np.mean(file_sizes)
            std_size = np.std(file_sizes)
            cv = (std_size / avg_size) * 100  # Coefficient of variation
            
            consistency = 'GOOD' if cv < 30 else 'MODERATE' if cv < 50 else 'POOR'
            
            self.results['statistical_analysis']['consistency'] = {
                'avg_file_size_kb': avg_size / 1024,
                'std_file_size_kb': std_size / 1024,
                'coefficient_of_variation': f"{cv:.1f}%",
                'status': consistency
            }
            
            print(f"   Average file size: {avg_size/1024:.1f} KB")
            print(f"   Size variation: {cv:.1f}% {'✅' if cv < 30 else '⚠️'}")
            print(f"   Consistency: {consistency}")
        else:
            print("   ⚠️  No files found")
        print()
        
    def generate_report(self):
        """Generate comprehensive quality report"""
        print("=" * 70)
        print("QUALITY REPORT SUMMARY")
        print("=" * 70)
        
        # Overall status
        total_issues = len(self.results['issues'])
        status = "✅ PRODUCTION READY" if total_issues == 0 else f"⚠️  {total_issues} ISSUES FOUND"
        
        print(f"\nOverall Status: {status}\n")
        
        # Summary
        if 'counts' in self.results['dataset_completeness']:
            counts = self.results['dataset_completeness']['counts']
            print(f"Dataset Completeness: {counts['completeness']}")
            print(f"Total Images: {counts['actual']} / {counts['expected']}")
        
        # Issues
        if self.results['issues']:
            print(f"\n⚠️  Issues Found ({len(self.results['issues'])}):")
            for issue in self.results['issues'][:10]:  # Show first 10
                print(f"   - {issue}")
            if len(self.results['issues']) > 10:
                print(f"   ... and {len(self.results['issues']) - 10} more")
        
        # Save detailed report
        report_file = self.dataset_dir / 'quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("=" * 70)

if __name__ == "__main__":
    tester = DatasetQualityTester('dataset')
    tester.run_all_tests()
