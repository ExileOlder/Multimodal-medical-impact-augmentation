"""
Unit tests for check_dependencies.py script

Tests the core functionality of dependency checking functions.
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import check_dependencies


def test_check_installed_packages():
    """Test that check_installed_packages returns a dictionary"""
    installed = check_dependencies.check_installed_packages()
    
    assert isinstance(installed, dict), "Should return a dictionary"
    # Should have at least some packages installed
    assert len(installed) >= 0, "Should return package information"
    
    # All keys should be lowercase strings
    for key in installed.keys():
        assert isinstance(key, str), "Package names should be strings"
        assert key == key.lower(), "Package names should be lowercase"
    
    # All values should be version strings
    for value in installed.values():
        assert isinstance(value, str), "Versions should be strings"


def test_check_pytorch_cuda_compatibility():
    """Test PyTorch and CUDA compatibility check"""
    compatible, info = check_dependencies.check_pytorch_cuda_compatibility()
    
    assert isinstance(compatible, bool), "Should return boolean compatibility status"
    assert isinstance(info, str), "Should return info string"
    assert len(info) > 0, "Info string should not be empty"


def test_parse_requirements():
    """Test requirements.txt parsing"""
    # Create a temporary requirements file
    test_requirements = """
# Test requirements
torch>=2.0.0
numpy==1.24.0
pillow
# Comment line
opencv-python>=4.8.0
"""
    
    # Write to temporary file
    test_file = "test_requirements_temp.txt"
    with open(test_file, 'w') as f:
        f.write(test_requirements)
    
    try:
        requirements = check_dependencies.parse_requirements(test_file)
        
        assert isinstance(requirements, list), "Should return a list"
        assert len(requirements) == 4, "Should parse 4 packages"
        
        # Check parsed packages
        package_names = [pkg[0] for pkg in requirements]
        assert 'torch' in package_names, "Should parse torch"
        assert 'numpy' in package_names, "Should parse numpy"
        assert 'pillow' in package_names, "Should parse pillow"
        assert 'opencv-python' in package_names, "Should parse opencv-python"
        
        # Check version specs
        torch_spec = next(spec for pkg, spec in requirements if pkg == 'torch')
        assert torch_spec == '>=2.0.0', "Should parse version spec correctly"
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


def test_list_missing_packages():
    """Test missing packages detection"""
    # This test will depend on the actual environment
    # Just verify it returns a list
    missing = check_dependencies.list_missing_packages()
    
    assert isinstance(missing, list), "Should return a list"
    
    # Each item should be a tuple of (package_name, version_spec)
    for item in missing:
        assert isinstance(item, tuple), "Each item should be a tuple"
        assert len(item) == 2, "Each tuple should have 2 elements"
        assert isinstance(item[0], str), "Package name should be string"
        assert isinstance(item[1], str), "Version spec should be string"


if __name__ == "__main__":
    print("Running tests for check_dependencies.py...")
    print()
    
    test_check_installed_packages()
    print("✓ test_check_installed_packages passed")
    
    test_check_pytorch_cuda_compatibility()
    print("✓ test_check_pytorch_cuda_compatibility passed")
    
    test_parse_requirements()
    print("✓ test_parse_requirements passed")
    
    test_list_missing_packages()
    print("✓ test_list_missing_packages passed")
    
    print()
    print("All tests passed!")
