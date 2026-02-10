#!/usr/bin/env python3
"""
回滚脚本 (Rollback Script)

本脚本用于在清理操作失败时恢复原状。
功能包括：
1. 读取清理操作日志
2. 回滚单个操作（重命名、删除）
3. 执行完整回滚流程
4. 验证项目结构完整性

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def read_cleanup_log(log_file: str = "logs/cleanup.log") -> List[Dict]:
    """
    读取清理操作日志
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        List[Dict]: 操作记录列表，每个记录包含 type, timestamp 等字段
    """
    operations = []
    
    try:
        log_path = Path(log_file)
        
        if not log_path.exists():
            print(f"⚠️ 日志文件不存在: {log_file}")
            return operations
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                # 解析日志行
                # 格式示例: "2024-01-01 12:00:00: Renamed codes to codes_backup"
                # 或: "2024-01-01 12:05:00: Deleted codes_backup"
                
                try:
                    # 分离时间戳和操作描述
                    # 格式: "2024-01-01 12:00:00: Renamed codes to codes_backup"
                    if ':' in line:
                        # 查找第三个冒号（时间戳中有两个冒号）
                        colon_positions = [i for i, c in enumerate(line) if c == ':']
                        
                        if len(colon_positions) >= 3:
                            # 第三个冒号后是操作描述
                            split_pos = colon_positions[2] + 1
                            timestamp_str = line[:split_pos-1].strip()
                            operation_desc = line[split_pos:].strip()
                        elif len(colon_positions) == 2:
                            # 可能是简化格式 "HH:MM: Description"
                            split_pos = colon_positions[1] + 1
                            timestamp_str = line[:split_pos-1].strip()
                            operation_desc = line[split_pos:].strip()
                        else:
                            continue
                        
                        # 解析操作类型
                        operation = parse_operation(timestamp_str, operation_desc)
                        if operation:
                            operations.append(operation)
                except Exception as e:
                    print(f"⚠️ 解析日志行失败: {line}")
                    print(f"   错误: {e}")
                    continue
        
        print(f"成功读取 {len(operations)} 条操作记录")
        
    except Exception as e:
        print(f"✗ 读取日志文件失败: {e}")
    
    return operations


def parse_operation(timestamp_str: str, operation_desc: str) -> Dict:
    """
    解析操作描述
    
    Args:
        timestamp_str: 时间戳字符串
        operation_desc: 操作描述
        
    Returns:
        Dict: 操作记录字典
    """
    operation = {
        'timestamp': timestamp_str,
        'description': operation_desc
    }
    
    # 清理操作描述（去除前后空格）
    operation_desc = operation_desc.strip()
    
    # 解析重命名操作
    # 格式: "Renamed codes to codes_backup"
    if 'renamed' in operation_desc.lower():
        parts = operation_desc.split()
        # 查找 'to' 关键字的位置
        try:
            to_index = next(i for i, word in enumerate(parts) if word.lower() == 'to')
            if to_index >= 2 and to_index + 1 < len(parts):
                operation['type'] = 'rename'
                operation['old_name'] = parts[to_index - 1]
                operation['new_name'] = parts[to_index + 1]
                return operation
        except StopIteration:
            pass
    
    # 解析删除操作
    # 格式: "Deleted codes_backup"
    if 'deleted' in operation_desc.lower():
        parts = operation_desc.split()
        if len(parts) >= 2:
            operation['type'] = 'delete'
            # 删除操作的路径是第二个词
            operation['path'] = parts[1]
            return operation
    
    # 未知操作类型
    operation['type'] = 'unknown'
    return operation


def rollback_operation(operation: Dict) -> bool:
    """
    回滚单个操作
    
    Args:
        operation: 操作记录字典
        
    Returns:
        bool: 是否成功回滚
    """
    try:
        op_type = operation.get('type')
        
        if op_type == 'rename':
            # 恢复重命名：将 new_name 改回 old_name
            old_name = operation.get('old_name')
            new_name = operation.get('new_name')
            
            if not old_name or not new_name:
                print(f"✗ 重命名操作信息不完整: {operation}")
                return False
            
            new_path = Path(new_name)
            old_path = Path(old_name)
            
            if not new_path.exists():
                print(f"⚠️ 目标路径不存在，无需回滚: {new_name}")
                return True
            
            if old_path.exists():
                print(f"⚠️ 原路径已存在，跳过回滚: {old_name}")
                return True
            
            print(f"回滚重命名: {new_name} -> {old_name}")
            os.rename(new_name, old_name)
            print(f"✓ 成功恢复: {old_name}")
            return True
            
        elif op_type == 'delete':
            # 删除操作无法恢复，仅警告
            path = operation.get('path')
            print(f"⚠️ 无法恢复已删除的文件/目录: {path}")
            print(f"   建议: 从备份或版本控制系统恢复")
            return False
            
        elif op_type == 'unknown':
            print(f"⚠️ 未知操作类型，跳过: {operation.get('description')}")
            return False
            
        else:
            print(f"⚠️ 不支持的操作类型: {op_type}")
            return False
            
    except Exception as e:
        print(f"✗ 回滚操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_project_structure() -> bool:
    """
    验证项目结构完整性
    
    Returns:
        bool: 项目结构是否完整
    """
    print("\n验证项目结构...")
    
    # 检查关键目录和文件
    required_paths = [
        'src/',
        'src/data/',
        'src/models/',
        'src/training/',
        'configs/',
        'configs/train_config.yaml',
        'requirements.txt',
        'train.py'
    ]
    
    missing = []
    for path_str in required_paths:
        path = Path(path_str)
        if not path.exists():
            missing.append(path_str)
    
    if missing:
        print("✗ 项目结构不完整，缺少以下路径:")
        for path in missing:
            print(f"  - {path}")
        return False
    else:
        print("✓ 项目结构完整")
        return True


def main():
    """主函数：执行完整回滚流程"""
    print("=" * 60)
    print("回滚操作开始")
    print("=" * 60)
    print()
    
    # 1. 读取清理操作日志
    print("1. 读取清理操作日志")
    print("-" * 60)
    operations = read_cleanup_log()
    
    if not operations:
        print("没有找到清理操作日志或日志为空")
        print("提示: 如果 codes_backup/ 存在，可以手动恢复:")
        print("  mv codes_backup codes")
        return 0
    
    print(f"找到 {len(operations)} 个操作记录")
    print()
    
    # 2. 显示操作记录
    print("2. 操作记录详情")
    print("-" * 60)
    for i, op in enumerate(operations, 1):
        print(f"{i}. [{op.get('timestamp')}] {op.get('description')}")
        print(f"   类型: {op.get('type')}")
    print()
    
    # 3. 确认回滚
    print("3. 确认回滚")
    print("-" * 60)
    print("⚠️ 警告: 回滚操作将恢复清理前的状态")
    print("   - 重命名操作将被撤销")
    print("   - 删除操作无法恢复（需要从备份恢复）")
    print()
    
    try:
        response = input("是否继续回滚? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("回滚操作已取消")
            return 0
    except (KeyboardInterrupt, EOFError):
        print("\n回滚操作已取消")
        return 0
    
    print()
    
    # 4. 执行回滚（逆序）
    print("4. 执行回滚")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    # 逆序回滚操作
    for op in reversed(operations):
        print(f"\n处理: {op.get('description')}")
        if rollback_operation(op):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("-" * 60)
    print(f"回滚结果: {success_count} 成功, {fail_count} 失败")
    print()
    
    # 5. 验证项目结构
    print("5. 验证项目结构")
    print("-" * 60)
    structure_ok = verify_project_structure()
    print()
    
    # 6. 总结
    print("=" * 60)
    print("回滚操作完成")
    print("=" * 60)
    
    if fail_count == 0 and structure_ok:
        print("✓ 回滚成功，项目结构已恢复")
        print()
        print("建议:")
        print("  1. 运行冒烟测试验证: python scripts/smoke_test.py")
        print("  2. 检查依赖问题: python scripts/check_dependencies.py")
        print("  3. 修复问题后重新执行清理流程")
        return 0
    else:
        print("⚠️ 回滚过程中遇到问题")
        if fail_count > 0:
            print(f"  - {fail_count} 个操作回滚失败")
        if not structure_ok:
            print("  - 项目结构不完整")
        print()
        print("建议:")
        print("  1. 检查上述错误信息")
        print("  2. 手动恢复缺失的文件/目录")
        print("  3. 从版本控制系统或备份恢复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
