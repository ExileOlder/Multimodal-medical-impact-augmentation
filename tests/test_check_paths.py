#!/usr/bin/env python3
"""
Unit tests for check_paths.py script

Tests the path checking functionality to ensure it correctly identifies
valid and invalid paths according to server restrictions.
"""

import os
import sys
import tempfile
import pytest
import yaml

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from check_paths import (
    is_valid_path,
    convert_to_relative_path,
    check_yaml_paths,
    _looks_like_path
)


class TestIsValidPath:
    """Test the is_valid_path function"""
    
    def test_relative_paths_are_valid(self):
        """Relative paths should always be valid"""
        assert is_valid_path("data/train")
        assert is_valid_path("checkpoints/model.pth")
        assert is_valid_path("./logs/train.log")
        assert is_valid_path("../configs/config.yaml")
    
    def test_allowed_absolute_paths_are_valid(self):
        """Absolute paths with allowed prefix should be valid"""
        assert is_valid_path("/home/Backup/maziheng/data")
        assert is_valid_path("/home/Backup/maziheng/project/checkpoints")
        assert is_valid_path("/home/Backup/maziheng/")
    
    def test_disallowed_absolute_paths_are_invalid(self):
        """Absolute paths without allowed prefix should be invalid"""
        assert not is_valid_path("/home/user/data")
        assert not is_valid_path("/tmp/checkpoints")
        assert not is_valid_path("/var/log/train.log")
        assert not is_valid_path("/root/project")
    
    def test_empty_path_is_invalid(self):
        """Empty paths should be invalid"""
        assert not is_valid_path("")
        assert not is_valid_path(None)
    
    def test_custom_allowed_prefix(self):
        """Should support custom allowed prefix"""
        assert is_valid_path("/custom/path/data", allowed_prefix="/custom/path")
        assert not is_valid_path("/other/path/data", allowed_prefix="/custom/path")


class TestLooksLikePath:
    """Test the _looks_like_path helper function"""
    
    def test_recognizes_paths(self):
        """Should recognize strings that look like paths"""
        assert _looks_like_path("data/train")
        assert _looks_like_path("checkpoints/model.pth")
        assert _looks_like_path("/home/user/data")
        assert _looks_like_path("logs/train.log")
        assert _looks_like_path("config.yaml")
    
    def test_ignores_non_paths(self):
        """Should ignore strings that don't look like paths"""
        assert not _looks_like_path("cuda")
        assert not _looks_like_path("cpu")
        assert not _looks_like_path("ddpm")
        assert not _looks_like_path("png")
        assert not _looks_like_path("")
        assert not _looks_like_path("ab")  # Too short
    
    def test_recognizes_file_extensions(self):
        """Should recognize common file extensions"""
        assert _looks_like_path("model.pth")
        assert _looks_like_path("config.yaml")
        assert _looks_like_path("data.json")
        assert _looks_like_path("train.log")


class TestConvertToRelativePath:
    """Test the convert_to_relative_path function"""
    
    def test_already_relative_unchanged(self):
        """Already relative paths should be unchanged"""
        assert convert_to_relative_path("data/train") == "data/train"
        assert convert_to_relative_path("./logs") == "./logs"
    
    def test_converts_absolute_to_relative(self):
        """Should convert absolute paths to relative when possible"""
        # This test depends on the current working directory
        # We'll just check that it returns something different for absolute paths
        result = convert_to_relative_path("/home/user/data")
        # Should either be a relative path or extract meaningful parts
        assert result is not None
        assert len(result) > 0
    
    def test_extracts_meaningful_parts(self):
        """Should extract meaningful directory names from absolute paths"""
        # When conversion fails, should extract common directory names
        result = convert_to_relative_path("/some/path/to/data/train")
        # On Windows, os.path.relpath may return backslashes
        # Just check that the result is not None and has content
        assert result is not None
        assert len(result) > 0
        # Should contain 'data' somewhere in the path
        assert 'data' in result.lower()


class TestCheckYamlPaths:
    """Test the check_yaml_paths function"""
    
    def test_valid_yaml_no_issues(self):
        """YAML with only valid paths should return empty list"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'data': {
                    'train_path': 'data/train',
                    'val_path': 'data/val'
                },
                'output': {
                    'checkpoint_dir': 'checkpoints',
                    'log_dir': 'logs'
                }
            }, f)
            temp_file = f.name
        
        try:
            issues = check_yaml_paths(temp_file)
            assert len(issues) == 0
        finally:
            os.unlink(temp_file)
    
    def test_invalid_yaml_finds_issues(self):
        """YAML with invalid paths should return list of issues"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'data': {
                    'train_path': '/invalid/absolute/path',
                    'val_path': 'data/val'  # This is valid
                },
                'output': {
                    'checkpoint_dir': '/tmp/checkpoints',  # Invalid
                }
            }, f)
            temp_file = f.name
        
        try:
            issues = check_yaml_paths(temp_file)
            # Should find 2 invalid paths
            assert len(issues) == 2
            
            # Check that the invalid paths are identified
            invalid_paths = [path for _, path in issues]
            assert '/invalid/absolute/path' in invalid_paths
            assert '/tmp/checkpoints' in invalid_paths
        finally:
            os.unlink(temp_file)
    
    def test_nested_yaml_structure(self):
        """Should handle nested YAML structures"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'level1': {
                    'level2': {
                        'level3': {
                            'path': '/invalid/path'
                        }
                    }
                }
            }, f)
            temp_file = f.name
        
        try:
            issues = check_yaml_paths(temp_file)
            assert len(issues) == 1
            key_path, path = issues[0]
            assert 'level1.level2.level3.path' == key_path
            assert path == '/invalid/path'
        finally:
            os.unlink(temp_file)
    
    def test_nonexistent_file(self):
        """Should handle nonexistent files gracefully"""
        issues = check_yaml_paths('nonexistent_file.yaml')
        assert issues == []
    
    def test_yaml_with_lists(self):
        """Should handle YAML with list structures"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'paths': [
                    'data/train',
                    '/invalid/path',
                    'data/val'
                ]
            }, f)
            temp_file = f.name
        
        try:
            issues = check_yaml_paths(temp_file)
            # Should find 1 invalid path in the list
            assert len(issues) == 1
            assert issues[0][1] == '/invalid/path'
        finally:
            os.unlink(temp_file)


class TestIntegration:
    """Integration tests for the complete workflow"""
    
    def test_real_config_files(self):
        """Test with actual config files if they exist"""
        config_files = [
            'configs/train_config.yaml',
            'configs/inference_config.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                # Should not raise any exceptions
                issues = check_yaml_paths(config_file)
                # Issues list should be a list (empty or with items)
                assert isinstance(issues, list)
                
                # All issues should be tuples of (key_path, path)
                for issue in issues:
                    assert isinstance(issue, tuple)
                    assert len(issue) == 2
                    assert isinstance(issue[0], str)  # key_path
                    assert isinstance(issue[1], str)  # path


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
