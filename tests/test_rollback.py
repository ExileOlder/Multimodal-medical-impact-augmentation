#!/usr/bin/env python3
"""
Unit tests for rollback.py

Tests the rollback script functionality including:
1. Reading cleanup logs
2. Parsing operations
3. Rollback operations
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rollback import (
    read_cleanup_log,
    parse_operation,
    rollback_operation,
    verify_project_structure
)


def test_parse_operation_rename():
    """测试解析重命名操作"""
    timestamp = "2024-01-01 12:00:00"
    description = "Renamed codes to codes_backup"
    
    operation = parse_operation(timestamp, description)
    
    assert operation['type'] == 'rename'
    assert operation['old_name'] == 'codes'
    assert operation['new_name'] == 'codes_backup'
    assert operation['timestamp'] == timestamp
    print("✓ test_parse_operation_rename passed")


def test_parse_operation_delete():
    """测试解析删除操作"""
    timestamp = "2024-01-01 12:05:00"
    description = "Deleted codes_backup"
    
    operation = parse_operation(timestamp, description)
    
    assert operation['type'] == 'delete'
    assert operation['path'] == 'codes_backup'
    assert operation['timestamp'] == timestamp
    print("✓ test_parse_operation_delete passed")


def test_read_cleanup_log():
    """测试读取清理日志"""
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log', encoding='utf-8') as f:
        f.write("2024-01-01 12:00:00: Renamed codes to codes_backup\n")
        f.write("2024-01-01 12:05:00: Deleted codes_backup\n")
        temp_log = f.name
    
    try:
        operations = read_cleanup_log(temp_log)
        
        assert len(operations) == 2, f"Expected 2 operations, got {len(operations)}"
        assert operations[0]['type'] == 'rename', f"Expected 'rename', got '{operations[0]['type']}'"
        assert operations[1]['type'] == 'delete', f"Expected 'delete', got '{operations[1]['type']}'"
        print("✓ test_read_cleanup_log passed")
    finally:
        os.unlink(temp_log)


def test_rollback_rename_operation():
    """测试回滚重命名操作"""
    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as tmpdir:
        old_dir = Path(tmpdir) / 'test_old'
        new_dir = Path(tmpdir) / 'test_new'
        
        # 创建原始目录并重命名
        old_dir.mkdir()
        old_dir.rename(new_dir)
        
        # 验证重命名成功
        assert not old_dir.exists()
        assert new_dir.exists()
        
        # 执行回滚
        operation = {
            'type': 'rename',
            'old_name': str(old_dir),
            'new_name': str(new_dir)
        }
        
        # 切换到临时目录
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = rollback_operation(operation)
            
            # 验证回滚成功
            assert result == True
            assert old_dir.exists()
            assert not new_dir.exists()
            print("✓ test_rollback_rename_operation passed")
        finally:
            os.chdir(original_cwd)


def test_rollback_delete_operation():
    """测试回滚删除操作（应该返回 False，因为无法恢复）"""
    operation = {
        'type': 'delete',
        'path': 'some_deleted_path'
    }
    
    result = rollback_operation(operation)
    
    # 删除操作无法回滚，应该返回 False
    assert result == False
    print("✓ test_rollback_delete_operation passed")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Rollback Script Unit Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_parse_operation_rename,
        test_parse_operation_delete,
        test_read_cleanup_log,
        test_rollback_rename_operation,
        test_rollback_delete_operation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
