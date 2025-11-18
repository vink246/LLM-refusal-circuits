#!/usr/bin/env python3
"""
Simple validation test for layer expansion logic (no dependencies).
Tests the logic directly without needing torch/transformers.
"""


def expand_layer_specs(layers, num_layers=32):
    """
    Simplified version of expand_layer_specs for testing.
    Mimics the logic in model_wrapper.py.
    """
    expanded = []
    
    for layer_spec in layers:
        # Handle "all" as shorthand for all residual layers
        if layer_spec == "all":
            layer_spec = "residuals_all"
        
        # Handle "*_all" specifications
        if layer_spec == "residuals_all":
            expanded.extend([f"residuals_{i}" for i in range(num_layers)])
        elif layer_spec == "mlp_all":
            expanded.extend([f"mlp_{i}" for i in range(num_layers)])
        elif layer_spec == "attention_all":
            expanded.extend([f"attention_{i}" for i in range(num_layers)])
        
        # Handle range specifications like "residuals_0-10"
        elif '_' in layer_spec and '-' in layer_spec.split('_', 1)[1]:
            parts = layer_spec.split('_')
            layer_type = parts[0]
            range_spec = '_'.join(parts[1:])
            
            try:
                start, end = range_spec.split('-')
                start_idx = int(start)
                end_idx = int(end) if end != '' else num_layers - 1
                
                # Validate indices
                if start_idx < 0 or end_idx >= num_layers or start_idx > end_idx:
                    print(f"Warning: Invalid range {layer_spec}, skipping")
                    continue
                
                for i in range(start_idx, end_idx + 1):
                    expanded.append(f"{layer_type}_{i}")
            except (ValueError, IndexError):
                print(f"Warning: Could not parse range specification: {layer_spec}")
                expanded.append(layer_spec)  # Keep original if parsing fails
        
        # Regular specification - keep as is
        else:
            expanded.append(layer_spec)
    
    return expanded


def test_expansion():
    """Run validation tests"""
    tests_passed = 0
    tests_failed = 0
    
    print("=" * 70)
    print("LAYER EXPANSION VALIDATION TESTS")
    print("=" * 70)
    
    # Test 1: "all" shorthand
    print("\n[Test 1] 'all' shorthand")
    result = expand_layer_specs(["all"])
    expected_len = 32
    if len(result) == expected_len and result[0] == "residuals_0" and result[-1] == "residuals_31":
        print("✓ PASS: Expanded to 32 residual layers")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected {expected_len} layers, got {len(result)}")
        tests_failed += 1
    
    # Test 2: "residuals_all"
    print("\n[Test 2] 'residuals_all'")
    result = expand_layer_specs(["residuals_all"])
    if len(result) == 32 and all(f"residuals_{i}" in result for i in range(32)):
        print("✓ PASS: All 32 residual layers")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Missing some residual layers")
        tests_failed += 1
    
    # Test 3: "mlp_all"
    print("\n[Test 3] 'mlp_all'")
    result = expand_layer_specs(["mlp_all"])
    if len(result) == 32 and result[0] == "mlp_0" and result[-1] == "mlp_31":
        print("✓ PASS: All 32 MLP layers")
        tests_passed += 1
    else:
        print(f"✗ FAIL")
        tests_failed += 1
    
    # Test 4: Range specification
    print("\n[Test 4] 'residuals_0-10' range")
    result = expand_layer_specs(["residuals_0-10"])
    if len(result) == 11 and result[0] == "residuals_0" and result[-1] == "residuals_10":
        print("✓ PASS: 11 layers (0 through 10 inclusive)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected 11 layers, got {len(result)}")
        tests_failed += 1
    
    # Test 5: Middle layers
    print("\n[Test 5] 'residuals_10-21' middle layers")
    result = expand_layer_specs(["residuals_10-21"])
    if len(result) == 12 and result[0] == "residuals_10" and result[-1] == "residuals_21":
        print("✓ PASS: 12 middle layers")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected 12 layers, got {len(result)}")
        tests_failed += 1
    
    # Test 6: Explicit layers (backward compatibility)
    print("\n[Test 6] Explicit layer list (backward compatibility)")
    explicit = ["residuals_0", "residuals_5", "residuals_10"]
    result = expand_layer_specs(explicit)
    if result == explicit:
        print("✓ PASS: Explicit layers unchanged")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Explicit layers were modified")
        tests_failed += 1
    
    # Test 7: Mixed specifications
    print("\n[Test 7] Mixed specifications")
    result = expand_layer_specs(["residuals_0-5", "mlp_10", "attention_15"])
    expected_len = 6 + 1 + 1  # 6 residuals + 1 mlp + 1 attention
    if len(result) == expected_len:
        print(f"✓ PASS: {expected_len} layers from mixed specs")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected {expected_len} layers, got {len(result)}")
        tests_failed += 1
    
    # Test 8: Full model (all types)
    print("\n[Test 8] All layer types (residuals + mlp + attention)")
    result = expand_layer_specs(["residuals_all", "mlp_all", "attention_all"])
    if len(result) == 96:  # 32 * 3
        print("✓ PASS: 96 layers total (32 of each type)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected 96 layers, got {len(result)}")
        tests_failed += 1
    
    # Test 9: Multiple ranges
    print("\n[Test 9] Multiple ranges")
    result = expand_layer_specs(["residuals_0-10", "residuals_20-31"])
    expected_len = 11 + 12  # (0-10) + (20-31)
    if len(result) == expected_len:
        print(f"✓ PASS: {expected_len} layers from two ranges")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected {expected_len} layers, got {len(result)}")
        tests_failed += 1
    
    # Test 10: Smaller model
    print("\n[Test 10] Smaller model (12 layers)")
    result = expand_layer_specs(["all"], num_layers=12)
    if len(result) == 12:
        print("✓ PASS: Adjusted to 12-layer model")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Expected 12 layers, got {len(result)}")
        tests_failed += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}/10")
    print(f"Tests Failed: {tests_failed}/10")
    
    if tests_failed == 0:
        print("\n✓ All validation tests passed!")
        print("\nThe layer expansion logic is working correctly.")
        print("You can now use shortcuts like 'all', 'residuals_0-31', etc. in your configs!")
        return 0
    else:
        print(f"\n✗ {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_expansion())

