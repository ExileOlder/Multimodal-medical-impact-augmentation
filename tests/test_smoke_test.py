"""
Unit tests for the smoke test script.

This test verifies that the smoke test script is properly structured
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_smoke_test_script_exists():
    """Test that the smoke test script exists."""
    smoke_test_path = project_root / 'scripts' / 'smoke_test.py'
    assert smoke_test_path.exists(), f"Smoke test script not found at {smoke_test_path}"


def test_smoke_test_has_required_functions():
    """Test that the smoke test script has all required functions."""
    # Import the smoke test module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "smoke_test",
        project_root / 'scripts' / 'smoke_test.py'
    )
    smoke_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_test)  # Need to execute the module
    
    # Check for required functions
    required_functions = [
        'test_data_loading',
        'test_model_initialization',
        'test_forward_pass',
        'main'
    ]
    
    for func_name in required_functions:
        assert hasattr(smoke_test, func_name), f"Missing function: {func_name}"


def test_smoke_test_functions_are_callable():
    """Test that all smoke test functions are callable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "smoke_test",
        project_root / 'scripts' / 'smoke_test.py'
    )
    smoke_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_test)
    
    # Check that functions are callable
    assert callable(smoke_test.test_data_loading)
    assert callable(smoke_test.test_model_initialization)
    assert callable(smoke_test.test_forward_pass)
    assert callable(smoke_test.main)


if __name__ == '__main__':
    # Run tests manually
    print("Running smoke test verification...")
    
    try:
        test_smoke_test_script_exists()
        print("✓ Smoke test script exists")
        
        test_smoke_test_has_required_functions()
        print("✓ All required functions present")
        
        test_smoke_test_functions_are_callable()
        print("✓ All functions are callable")
        
        print("\n✓ All verification tests passed!")
        
    except AssertionError as e:
        print(f"\n✗ Verification failed: {e}")
        sys.exit(1)
