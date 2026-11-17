import sys

# 文件路径
file_path = '0/OH'

# 需要修改的行范围（行号从1开始）
start_line = 23
end_line = 150022

print(f"正在处理文件: {file_path}")
print(f"将第 {start_line} 到第 {end_line} 行的数字替换为 0")

# 读取文件
with open(file_path, 'r') as f:
    lines = f.readlines()

print(f"文件总行数: {len(lines)}")

# 修改指定行范围的内容
modified_count = 0
for i in range(start_line - 1, min(end_line, len(lines))):
    original_line = lines[i]
    # 如果这行包含数字，替换为0
    stripped = original_line.strip()
    if stripped and not stripped.startswith('(') and not stripped.startswith(')'):
        try:
            # 尝试转换为浮点数，如果成功说明是数字行
            float(stripped)
            lines[i] = '0\n'
            modified_count += 1
        except ValueError:
            # 不是数字，跳过
            pass

print(f"共修改了 {modified_count} 行")

# 写回文件
with open(file_path, 'w') as f:
    f.writelines(lines)

print(f"文件修改完成: {file_path}")

