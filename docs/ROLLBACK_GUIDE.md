# Rollback Script Usage Guide

## Overview

The rollback script (`scripts/rollback.py`) is designed to restore the project to its previous state if cleanup operations fail. It reads the cleanup log and reverses operations in the correct order.

## Features

1. **Read Cleanup Log**: Parses `logs/cleanup.log` to identify operations
2. **Rollback Operations**: Reverses rename operations automatically
3. **Verify Structure**: Checks project structure integrity after rollback
4. **Interactive Confirmation**: Asks for user confirmation before proceeding

## Usage

### Basic Usage

```bash
python scripts/rollback.py
```

The script will:
1. Read the cleanup log from `logs/cleanup.log`
2. Display all operations found
3. Ask for confirmation
4. Execute rollback in reverse order
5. Verify project structure

### Example Output

```
============================================================
回滚操作开始
============================================================

1. 读取清理操作日志
------------------------------------------------------------
成功读取 2 条操作记录
找到 2 个操作记录

2. 操作记录详情
------------------------------------------------------------
1. [2024-01-15 10:30:00] Renamed codes to codes_backup
   类型: rename
2. [2024-01-15 10:35:00] Deleted codes_backup
   类型: delete

3. 确认回滚
------------------------------------------------------------
⚠️ 警告: 回滚操作将恢复清理前的状态
   - 重命名操作将被撤销
   - 删除操作无法恢复（需要从备份恢复）

是否继续回滚? (yes/no): yes

4. 执行回滚
------------------------------------------------------------

处理: Deleted codes_backup
⚠️ 无法恢复已删除的文件/目录: codes_backup
   建议: 从备份或版本控制系统恢复

处理: Renamed codes to codes_backup
回滚重命名: codes_backup -> codes
✓ 成功恢复: codes

------------------------------------------------------------
回滚结果: 1 成功, 1 失败

5. 验证项目结构
------------------------------------------------------------

验证项目结构...
✓ 项目结构完整

============================================================
回滚操作完成
============================================================
✓ 回滚成功，项目结构已恢复

建议:
  1. 运行冒烟测试验证: python scripts/smoke_test.py
  2. 检查依赖问题: python scripts/check_dependencies.py
  3. 修复问题后重新执行清理流程
```

## Supported Operations

### Rename Operations

**Format in log**: `YYYY-MM-DD HH:MM:SS: Renamed <old_name> to <new_name>`

**Rollback behavior**: Renames `<new_name>` back to `<old_name>`

**Example**:
```
2024-01-15 10:30:00: Renamed codes to codes_backup
```

Rollback will execute: `mv codes_backup codes`

### Delete Operations

**Format in log**: `YYYY-MM-DD HH:MM:SS: Deleted <path>`

**Rollback behavior**: Cannot be automatically restored. The script will warn the user to restore from backup or version control.

**Example**:
```
2024-01-15 10:35:00: Deleted codes_backup
```

Rollback will display a warning and suggest manual restoration.

## Log File Format

The cleanup log (`logs/cleanup.log`) should follow this format:

```
YYYY-MM-DD HH:MM:SS: <Operation description>
```

Examples:
```
2024-01-15 10:30:00: Renamed codes to codes_backup
2024-01-15 10:35:00: Deleted codes_backup
```

## Error Handling

### Missing Log File

If `logs/cleanup.log` doesn't exist:
```
⚠️ 日志文件不存在: logs/cleanup.log
没有找到清理操作日志或日志为空
提示: 如果 codes_backup/ 存在，可以手动恢复:
  mv codes_backup codes
```

### Path Already Exists

If the target path already exists during rollback:
```
⚠️ 原路径已存在，跳过回滚: codes
```

### Path Doesn't Exist

If the source path doesn't exist during rollback:
```
⚠️ 目标路径不存在，无需回滚: codes_backup
```

## Manual Rollback

If the script fails or the log is missing, you can manually rollback:

### Restore codes/ directory

```bash
# If codes_backup/ exists
mv codes_backup codes

# Verify structure
ls -la codes/
```

### Verify project works

```bash
# Run smoke test
python scripts/smoke_test.py

# Check dependencies
python scripts/check_dependencies.py
```

## Integration with Cleanup Process

The rollback script is designed to work with the cleanup process:

1. **Before cleanup**: No action needed
2. **During cleanup**: Operations are logged to `logs/cleanup.log`
3. **If cleanup fails**: Run `python scripts/rollback.py`
4. **After rollback**: Fix issues and retry cleanup

## Requirements Satisfied

This script satisfies the following requirements:

- **9.1**: Creates operation log before cleanup
- **9.2**: Records all rename and delete operations
- **9.3**: Restores file structure based on log when requested
- **9.4**: Provides rollback.py script
- **9.5**: Verifies project structure integrity after rollback

## Testing

Unit tests are available in `tests/test_rollback.py`:

```bash
python tests/test_rollback.py
```

Tests cover:
- Parsing rename operations
- Parsing delete operations
- Reading cleanup logs
- Rollback rename operations
- Rollback delete operations (warning behavior)

## Troubleshooting

### Script doesn't find operations

**Problem**: "没有找到清理操作日志或日志为空"

**Solution**: 
- Check if `logs/cleanup.log` exists
- Verify log format matches expected format
- Check file encoding (should be UTF-8)

### Rollback fails with permission error

**Problem**: Permission denied when renaming

**Solution**:
- Check file permissions
- Ensure no process is using the directory
- Run with appropriate permissions

### Project structure incomplete after rollback

**Problem**: "项目结构不完整"

**Solution**:
- Check which paths are missing
- Restore from version control (git)
- Restore from backup if available

## Best Practices

1. **Always check the log before rollback**: Review operations to understand what will be reversed
2. **Backup important data**: Before any cleanup operation, ensure you have backups
3. **Use version control**: Git can help recover deleted files
4. **Test after rollback**: Always run smoke tests after rollback
5. **Fix root cause**: Identify why cleanup failed before retrying

## See Also

- [Cleanup Process Documentation](../CLEANUP_CHECKLIST.md)
- [Smoke Test Guide](../scripts/smoke_test.py)
- [Dependency Check Guide](../scripts/check_dependencies.py)
