#!/usr/bin/env python3
"""
Unit tests for layer specification expansion in ModelWrapper.

Tests the new layer shorthand notation: "all", "residuals_all", "residuals_0-31", etc.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockModel:
    """Mock model for testing without loading actual models"""
    def __init__(self, num_layers=32):
        self.model = type('obj', (object,), {
            'layers': [None] * num_layers
        })


class TestLayerExpansion(unittest.TestCase):
    """Test layer specification expansion logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.models.model_wrapper import ModelWrapper, ModelConfig
        
        # Create mock model wrapper
        config = ModelConfig(model_name="test/model")
        self.wrapper = ModelWrapper(config)
        self.wrapper.model = MockModel(num_layers=32)
    
    def test_all_shorthand(self):
        """Test 'all' expands to all residual layers"""
        result = self.wrapper.expand_layer_specs(["all"])
        expected = [f"residuals_{i}" for i in range(32)]
        self.assertEqual(result, expected)
    
    def test_residuals_all(self):
        """Test 'residuals_all' expands correctly"""
        result = self.wrapper.expand_layer_specs(["residuals_all"])
        expected = [f"residuals_{i}" for i in range(32)]
        self.assertEqual(result, expected)
    
    def test_mlp_all(self):
        """Test 'mlp_all' expands correctly"""
        result = self.wrapper.expand_layer_specs(["mlp_all"])
        expected = [f"mlp_{i}" for i in range(32)]
        self.assertEqual(result, expected)
    
    def test_attention_all(self):
        """Test 'attention_all' expands correctly"""
        result = self.wrapper.expand_layer_specs(["attention_all"])
        expected = [f"attention_{i}" for i in range(32)]
        self.assertEqual(result, expected)
    
    def test_range_specification(self):
        """Test range notation like 'residuals_0-10'"""
        result = self.wrapper.expand_layer_specs(["residuals_0-10"])
        expected = [f"residuals_{i}" for i in range(11)]  # 0 through 10 inclusive
        self.assertEqual(result, expected)
    
    def test_range_middle_layers(self):
        """Test middle layer range"""
        result = self.wrapper.expand_layer_specs(["residuals_10-21"])
        expected = [f"residuals_{i}" for i in range(10, 22)]  # 10 through 21 inclusive
        self.assertEqual(result, expected)
    
    def test_mlp_range(self):
        """Test MLP layer range"""
        result = self.wrapper.expand_layer_specs(["mlp_5-15"])
        expected = [f"mlp_{i}" for i in range(5, 16)]  # 5 through 15 inclusive
        self.assertEqual(result, expected)
    
    def test_explicit_layer(self):
        """Test that explicit layers pass through unchanged"""
        result = self.wrapper.expand_layer_specs(["residuals_5"])
        self.assertEqual(result, ["residuals_5"])
    
    def test_mixed_specifications(self):
        """Test mixing different specification types"""
        result = self.wrapper.expand_layer_specs([
            "residuals_0-5",
            "mlp_10",
            "attention_15"
        ])
        expected = (
            [f"residuals_{i}" for i in range(6)] +  # 0-5
            ["mlp_10"] +
            ["attention_15"]
        )
        self.assertEqual(result, expected)
    
    def test_multiple_ranges(self):
        """Test multiple range specifications"""
        result = self.wrapper.expand_layer_specs([
            "residuals_0-10",
            "residuals_20-31"
        ])
        expected = (
            [f"residuals_{i}" for i in range(11)] +  # 0-10
            [f"residuals_{i}" for i in range(20, 32)]  # 20-31
        )
        self.assertEqual(result, expected)
    
    def test_empty_list(self):
        """Test empty specification list"""
        result = self.wrapper.expand_layer_specs([])
        self.assertEqual(result, [])
    
    def test_full_model_32_layers(self):
        """Test all layer types for 32-layer model"""
        result = self.wrapper.expand_layer_specs([
            "residuals_all",
            "mlp_all",
            "attention_all"
        ])
        expected = (
            [f"residuals_{i}" for i in range(32)] +
            [f"mlp_{i}" for i in range(32)] +
            [f"attention_{i}" for i in range(32)]
        )
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 96)  # 32 * 3
    
    def test_get_num_layers(self):
        """Test get_num_layers method"""
        num_layers = self.wrapper.get_num_layers()
        self.assertEqual(num_layers, 32)
    
    def test_smaller_model(self):
        """Test with a smaller model (e.g., 12 layers)"""
        self.wrapper.model = MockModel(num_layers=12)
        result = self.wrapper.expand_layer_specs(["all"])
        expected = [f"residuals_{i}" for i in range(12)]
        self.assertEqual(result, expected)
    
    def test_backward_compatibility(self):
        """Test that old-style explicit lists still work"""
        layers = [
            "residuals_0",
            "residuals_5",
            "residuals_10",
            "residuals_15",
            "residuals_20",
            "residuals_25",
            "residuals_30"
        ]
        result = self.wrapper.expand_layer_specs(layers)
        self.assertEqual(result, layers)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayerExpansion)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("LAYER EXPANSION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())

