import pandas as pd
import numpy as np

if __name__ == "__main__":
    file_path = './../../data/Test_Scores.xlsx'
    sheet_names = pd.ExcelFile(file_path).sheet_names[:6]

    # 读取并处理所有表单
    processed_dfs = []
    for i, sheet_name in enumerate(sheet_names, 1):
        print(f"处理表 {i}: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"原始数据形状: {df.shape}")

        # 去除空行（全为NaN的行）
        df = df.dropna(how='all')
        print(f"去除空行后数据形状: {df.shape}")

        # 添加表单来源列（可选，便于追踪数据来源）
        df['_sheet_source'] = sheet_name

        # 处理绩点缺失（向量化，保留全缺失行）
        if '绩点' in df.columns:
            col_idx = df.columns.get_loc('绩点')
            prev_col = df.columns[col_idx - 1] if col_idx > 0 else None

            if prev_col:
                mask = df['绩点'].isnull() & df[prev_col].notnull()
                df.loc[mask, '绩点'] = (df[prev_col][mask] - 50) / 10

        # 处理总评成绩
        if '总评成绩' in df.columns:
            mapping = {
                '旷考': 1,
                '优秀': 95,
                '良好': 85,
                '中等': 75,
                '及格': 65,
                '不及格': 55
            }
            replaced = df['总评成绩'].replace(mapping, inplace=False)
            count = (replaced != df['总评成绩']).sum()
            if count:
                df['总评成绩'] = replaced

        # 处理折算成绩
        if '折算成绩' in df.columns:
            df['折算成绩'] = pd.to_numeric(df['折算成绩'], errors='coerce')
            low_mask = df['折算成绩'] < 60
            if low_mask.any():
                df.loc[low_mask, '绩点'] = 0

        processed_dfs.append(df)
        print(f"已处理表单 '{sheet_name}'")

    # 合并所有表单
    merged_df = pd.concat(processed_dfs, ignore_index=True, sort=False)
    print(f"\n合并完成！总数据形状: {merged_df.shape}")

    # 生成编号：原表名 + 姓名
    if '姓名' in merged_df.columns:
        # 生成编号：原表名_姓名
        merged_df['编号'] = merged_df['_sheet_source'] + '_' + merged_df['姓名'].astype(str)
        print(f"已生成编号列，格式为：原表名_姓名")
    else:
        print("警告：未找到'姓名'列，无法生成编号")
        # 如果没有姓名列，使用原表名_行号作为编号
        merged_df['编号'] = merged_df['_sheet_source'] + '_' + merged_df.index.astype(str)
        print(f"已生成编号列，格式为：原表名_行号")

    # 修正课程名称
    if '课程名称' in merged_df.columns:
        # 定义课程名称替换映射
        course_mapping = {
            '创新与创业管理B（在线）': '创新与创业管理B(在线)',
            '物理实验 （上）': '物理实验（上）',
            '马克思主义基本原理': '马克思主义基本原理概论',
            '电路': '电路分析基础B',
            '数据库技术与应用': '数据库应用'
        }

        # 统计替换前后的变化
        original_values = merged_df['课程名称'].value_counts()
        merged_df['课程名称'] = merged_df['课程名称'].replace(course_mapping)
        new_values = merged_df['课程名称'].value_counts()

        print(f"已修正课程名称:")
        for old_name, new_name in course_mapping.items():
            if old_name in original_values:
                count = original_values[old_name]
                print(f"  '{old_name}' -> '{new_name}' ({count} 个)")
    else:
        print("警告：未找到'课程名称'列，跳过课程名称修正")

    # 处理重复数据：相同编号和课程名称的记录，保留总评成绩更高的
    if '编号' in merged_df.columns and '课程名称' in merged_df.columns and '总评成绩' in merged_df.columns:
        print(f"\n开始处理重复数据...")

        # 将总评成绩转换为数值类型，便于比较
        merged_df['总评成绩'] = pd.to_numeric(merged_df['总评成绩'], errors='coerce')

        # 按编号和课程名称分组，保留总评成绩最高的记录
        # 首先按总评成绩降序排序，然后去重保留第一个（即总评成绩最高的）
        merged_df = merged_df.sort_values(['编号', '课程名称', '总评成绩'], ascending=[True, True, False])
        original_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['编号', '课程名称'], keep='first')
        removed_count = original_count - len(merged_df)

        print(f"已处理重复数据：删除了 {removed_count} 条重复记录，保留了 {len(merged_df)} 条唯一记录")
    else:
        print("警告：缺少必要的列（编号、课程名称、总评成绩），跳过重复数据处理")

    # 对特定课程进行标签化处理：数据库系统、概率论与数理统计
    if '课程名称' in merged_df.columns and '总评成绩' in merged_df.columns:
        print(f"\n开始对特定课程进行标签化处理...")

        # 确保总评成绩是数值类型
        merged_df['总评成绩'] = pd.to_numeric(merged_df['总评成绩'], errors='coerce')

        # 定义要处理的课程
        target_courses = ['数据库系统', '概率论与数理统计']

        for course in target_courses:
            # 找到该课程的记录
            course_mask = merged_df['课程名称'] == course
            if course_mask.any():
                # 提取该课程的总评成绩
                course_scores = merged_df.loc[course_mask, '总评成绩']

                # 创建标签列（初始为NaN）
                label_col = f'{course}_标签'
                merged_df[label_col] = np.nan

                # 应用标签规则：80以上为2，60-79为1，不及格为0
                merged_df.loc[course_mask & (course_scores >= 80), label_col] = 2
                merged_df.loc[course_mask & (course_scores >= 60) & (course_scores < 80), label_col] = 1
                merged_df.loc[course_mask & (course_scores < 60), label_col] = 0

                # 统计标签分布
                label_counts = merged_df.loc[course_mask, label_col].value_counts().sort_index()
                print(f"  {course} 标签分布:")
                for label, count in label_counts.items():
                    label_name = {2: '80分以上', 1: '60-79分', 0: '不及格'}[label]
                    print(f"    {label_name}({label}): {count} 个")
            else:
                print(f"  未找到课程: {course}")
    else:
        print("警告：缺少必要的列（课程名称、总评成绩），跳过标签化处理")

    # 删除临时的表单来源列（如果不需要的话）
    merged_df = merged_df.drop(columns=['_sheet_source'])

    # 保存合并后的数据
    output_file = 'merged_test_scores_final.xlsx'
    merged_df.to_excel(output_file, index=False)
    print(f"\n已保存合并后的数据到: {output_file}")

    # 显示合并后的基本信息
    print(f"\n合并后数据统计:")
    print(f"- 总行数: {len(merged_df)}")
    print(f"- 总列数: {len(merged_df.columns)}")
    print(f"- 列名: {list(merged_df.columns)}")

    # 显示前几行数据示例
    print(f"\n前5行数据示例:")
    print(merged_df.head())