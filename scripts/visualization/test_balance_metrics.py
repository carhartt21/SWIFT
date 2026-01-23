#!/usr/bin/env python3
"""
Unit tests for Imbalance Ratio (IR) and Normalized Shannon Entropy (H_norm) metrics.

These tests verify the correctness of the dataset balance metrics added to 
WeatherDatasetVisualizer.
"""

import math
import sys
import os

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weather_dataset_visualizer import WeatherDatasetVisualizer


def test_imbalance_ratio():
    """Test the Imbalance Ratio (IR) calculation."""
    print("=" * 60)
    print("Testing Imbalance Ratio (IR)")
    print("=" * 60)
    
    # Create a visualizer instance (we just need access to the methods)
    # Using a dummy path since we won't actually load any datasets
    visualizer = WeatherDatasetVisualizer("/tmp")
    
    # Test 1: Perfectly balanced dataset (all 7 categories have same count)
    print("\nTest 1: Perfectly balanced dataset")
    balanced_counts = {cat: 100 for cat in visualizer.weather_categories}
    ir_balanced = visualizer.compute_imbalance_ratio(balanced_counts)
    print(f"  Counts: {balanced_counts}")
    print(f"  IR = {ir_balanced}")
    assert ir_balanced == 1.0, f"Expected IR=1.0 for balanced dataset, got {ir_balanced}"
    print("  ✓ PASSED: IR = 1.0 for perfectly balanced dataset")
    
    # Test 2: Imbalanced dataset (2:1 ratio)
    print("\nTest 2: Imbalanced dataset (2:1 ratio)")
    imbalanced_counts = {
        'clear_day': 200,
        'foggy': 100,
        'snowy': 150,
        'night': 150,
        'rainy': 150,
        'dawn_dusk': 150,
        'cloudy': 150
    }
    ir_imbalanced = visualizer.compute_imbalance_ratio(imbalanced_counts)
    print(f"  Counts: {imbalanced_counts}")
    print(f"  IR = {ir_imbalanced}")
    assert ir_imbalanced == 2.0, f"Expected IR=2.0, got {ir_imbalanced}"
    print("  ✓ PASSED: IR = 2.0 for 2:1 imbalance")
    
    # Test 3: Dataset with zero samples in one category
    print("\nTest 3: Dataset with zero samples in one category")
    zero_counts = {
        'clear_day': 100,
        'foggy': 0,  # Zero samples!
        'snowy': 100,
        'night': 100,
        'rainy': 100,
        'dawn_dusk': 100,
        'cloudy': 100
    }
    ir_zero = visualizer.compute_imbalance_ratio(zero_counts)
    print(f"  Counts: {zero_counts}")
    print(f"  IR = {ir_zero}")
    assert ir_zero == float('inf'), f"Expected IR=inf for zero samples, got {ir_zero}"
    print("  ✓ PASSED: IR = inf when any category has 0 samples")
    
    # Test 4: Dataset with missing category (treated as 0)
    print("\nTest 4: Dataset with missing category (not in dict)")
    missing_counts = {
        'clear_day': 100,
        'foggy': 100,
        'snowy': 100,
        # 'night' is missing - should be treated as 0
        'rainy': 100,
        'dawn_dusk': 100,
        'cloudy': 100
    }
    ir_missing = visualizer.compute_imbalance_ratio(missing_counts)
    print(f"  Counts: {missing_counts}")
    print(f"  IR = {ir_missing}")
    assert ir_missing == float('inf'), f"Expected IR=inf for missing category, got {ir_missing}"
    print("  ✓ PASSED: IR = inf when category is missing from dict")
    
    # Test 5: Empty dictionary
    print("\nTest 5: Empty category counts")
    ir_empty = visualizer.compute_imbalance_ratio({})
    print(f"  IR = {ir_empty}")
    assert ir_empty == float('inf'), f"Expected IR=inf for empty dict, got {ir_empty}"
    print("  ✓ PASSED: IR = inf for empty counts")
    
    # Test 6: Extreme imbalance (10:1)
    print("\nTest 6: Extreme imbalance (10:1)")
    extreme_counts = {
        'clear_day': 1000,
        'foggy': 100,
        'snowy': 500,
        'night': 500,
        'rainy': 500,
        'dawn_dusk': 500,
        'cloudy': 500
    }
    ir_extreme = visualizer.compute_imbalance_ratio(extreme_counts)
    print(f"  Counts: {extreme_counts}")
    print(f"  IR = {ir_extreme}")
    assert ir_extreme == 10.0, f"Expected IR=10.0, got {ir_extreme}"
    print("  ✓ PASSED: IR = 10.0 for 10:1 imbalance")
    
    print("\n" + "=" * 60)
    print("All Imbalance Ratio tests PASSED!")
    print("=" * 60)


def test_normalized_shannon_entropy():
    """Test the Normalized Shannon Entropy (H_norm) calculation."""
    print("\n" + "=" * 60)
    print("Testing Normalized Shannon Entropy (H_norm)")
    print("=" * 60)
    
    visualizer = WeatherDatasetVisualizer("/tmp")
    
    # Test 1: Perfectly uniform distribution (all 7 categories have same count)
    print("\nTest 1: Perfectly uniform distribution")
    uniform_counts = {cat: 100 for cat in visualizer.weather_categories}
    h_uniform = visualizer.compute_normalized_shannon_entropy(uniform_counts)
    print(f"  Counts: {uniform_counts}")
    print(f"  H_norm = {h_uniform}")
    assert abs(h_uniform - 1.0) < 1e-10, f"Expected H_norm=1.0 for uniform distribution, got {h_uniform}"
    print("  ✓ PASSED: H_norm ≈ 1.0 for perfectly uniform distribution")
    
    # Test 2: Slightly imbalanced distribution
    print("\nTest 2: Slightly imbalanced distribution")
    slight_imbalance = {
        'clear_day': 200,
        'foggy': 100,
        'snowy': 150,
        'night': 150,
        'rainy': 150,
        'dawn_dusk': 150,
        'cloudy': 150
    }
    h_slight = visualizer.compute_normalized_shannon_entropy(slight_imbalance)
    print(f"  Counts: {slight_imbalance}")
    print(f"  H_norm = {h_slight:.6f}")
    assert 0 < h_slight < 1.0, f"Expected 0 < H_norm < 1 for imbalanced data, got {h_slight}"
    print(f"  ✓ PASSED: 0 < H_norm < 1 (H_norm = {h_slight:.6f})")
    
    # Test 3: Extreme imbalance (one category has all data)
    print("\nTest 3: Extreme imbalance (one category has all data)")
    extreme_counts = {
        'clear_day': 1000,
        'foggy': 0,
        'snowy': 0,
        'night': 0,
        'rainy': 0,
        'dawn_dusk': 0,
        'cloudy': 0
    }
    h_extreme = visualizer.compute_normalized_shannon_entropy(extreme_counts)
    print(f"  Counts: {extreme_counts}")
    print(f"  H_norm = {h_extreme}")
    assert h_extreme == 0.0, f"Expected H_norm=0 for extreme imbalance, got {h_extreme}"
    print("  ✓ PASSED: H_norm = 0 when all data is in one category")
    
    # Test 4: Two categories with equal data (partial coverage)
    print("\nTest 4: Two categories with equal data")
    two_cat_counts = {
        'clear_day': 500,
        'foggy': 500,
        'snowy': 0,
        'night': 0,
        'rainy': 0,
        'dawn_dusk': 0,
        'cloudy': 0
    }
    h_two = visualizer.compute_normalized_shannon_entropy(two_cat_counts)
    # For 2 equal categories: H = log(2), H_max = log(7), H_norm = log(2)/log(7)
    expected_h_two = math.log(2) / math.log(7)
    print(f"  Counts: {two_cat_counts}")
    print(f"  H_norm = {h_two:.6f} (expected: {expected_h_two:.6f})")
    assert abs(h_two - expected_h_two) < 1e-10, f"Expected H_norm={expected_h_two}, got {h_two}"
    print(f"  ✓ PASSED: H_norm = log(2)/log(7) ≈ {expected_h_two:.6f}")
    
    # Test 5: Empty dataset (total = 0)
    print("\nTest 5: Empty dataset (total = 0)")
    empty_counts = {cat: 0 for cat in visualizer.weather_categories}
    h_empty = visualizer.compute_normalized_shannon_entropy(empty_counts)
    print(f"  Counts: {empty_counts}")
    print(f"  H_norm = {h_empty}")
    assert math.isnan(h_empty), f"Expected H_norm=NaN for empty dataset, got {h_empty}"
    print("  ✓ PASSED: H_norm = NaN for empty dataset")
    
    # Test 6: Empty dictionary
    print("\nTest 6: Empty dictionary")
    h_empty_dict = visualizer.compute_normalized_shannon_entropy({})
    print(f"  H_norm = {h_empty_dict}")
    assert math.isnan(h_empty_dict), f"Expected H_norm=NaN for empty dict, got {h_empty_dict}"
    print("  ✓ PASSED: H_norm = NaN for empty dict")
    
    # Test 7: Verify H_norm is always in [0, 1] range for valid data
    print("\nTest 7: H_norm is always in [0, 1] range")
    test_cases = [
        {cat: i * 100 + 1 for i, cat in enumerate(visualizer.weather_categories)},
        {cat: (i + 1) ** 2 for i, cat in enumerate(visualizer.weather_categories)},
        {'clear_day': 1, 'foggy': 1000, 'snowy': 1, 'night': 1, 'rainy': 1, 'dawn_dusk': 1, 'cloudy': 1},
    ]
    for counts in test_cases:
        h = visualizer.compute_normalized_shannon_entropy(counts)
        print(f"  Counts: {counts}")
        print(f"  H_norm = {h:.6f}")
        assert 0 <= h <= 1, f"H_norm out of range [0,1]: {h}"
    print("  ✓ PASSED: H_norm is always in [0, 1]")
    
    print("\n" + "=" * 60)
    print("All Normalized Shannon Entropy tests PASSED!")
    print("=" * 60)


def test_combined_validation():
    """Combined validation tests as specified in requirements."""
    print("\n" + "=" * 60)
    print("Validation Tests (Combined IR and H_norm)")
    print("=" * 60)
    
    visualizer = WeatherDatasetVisualizer("/tmp")
    
    # Validation 1: Perfectly balanced dataset
    print("\nValidation 1: Perfectly balanced dataset")
    print("  Expected: IR = 1, H_norm = 1")
    balanced = {cat: 100 for cat in visualizer.weather_categories}
    ir = visualizer.compute_imbalance_ratio(balanced)
    h_norm = visualizer.compute_normalized_shannon_entropy(balanced)
    print(f"  Result: IR = {ir}, H_norm = {h_norm}")
    assert ir == 1.0 and abs(h_norm - 1.0) < 1e-10
    print("  ✓ PASSED")
    
    # Validation 2: Dataset missing at least one domain
    print("\nValidation 2: Dataset missing at least one domain")
    print("  Expected: IR = inf, H_norm < 1")
    missing_domain = {
        'clear_day': 100,
        'foggy': 100,
        'snowy': 100,
        'night': 100,
        'rainy': 100,
        'dawn_dusk': 100,
        # 'cloudy' is missing
    }
    ir = visualizer.compute_imbalance_ratio(missing_domain)
    h_norm = visualizer.compute_normalized_shannon_entropy(missing_domain)
    print(f"  Result: IR = {ir}, H_norm = {h_norm:.6f}")
    assert ir == float('inf') and h_norm < 1.0
    print("  ✓ PASSED")
    
    # Validation 3: Extreme imbalance (one domain has all samples)
    print("\nValidation 3: Extreme imbalance (one domain has all samples)")
    print("  Expected: IR = inf (others have 0), H_norm ≈ 0")
    extreme = {
        'clear_day': 1000,
        'foggy': 0,
        'snowy': 0,
        'night': 0,
        'rainy': 0,
        'dawn_dusk': 0,
        'cloudy': 0
    }
    ir = visualizer.compute_imbalance_ratio(extreme)
    h_norm = visualizer.compute_normalized_shannon_entropy(extreme)
    print(f"  Result: IR = {ir}, H_norm = {h_norm}")
    assert ir == float('inf') and h_norm == 0.0
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All Validation Tests PASSED!")
    print("=" * 60)


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# RUNNING ALL TESTS FOR BALANCE METRICS")
    print("#" * 70)
    
    try:
        test_imbalance_ratio()
        test_normalized_shannon_entropy()
        test_combined_validation()
        
        print("\n" + "#" * 70)
        print("# ALL TESTS PASSED SUCCESSFULLY!")
        print("#" * 70)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
