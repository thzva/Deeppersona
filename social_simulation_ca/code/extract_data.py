import pandas as pd
import numpy as np
import os
import csv
from collections import defaultdict
import statistics
from typing import Dict, List, Any

# 定义问题列
QUESTION_COLUMNS = ['Q45', 'Q46', 'Q57', 'Q184', 'Q218', 'Q254']

def read_survey_data(file_path: str) -> pd.DataFrame:
    """
    读取调查数据，并筛选印度(IND)数据
    
    Args:
        file_path: 调查数据CSV文件路径
        
    Returns:
        筛选后的印度数据
    """
    print(f"Reading survey data from: {file_path}")
    
    # 读取CSV文件，指定编码和引号字符
    df = pd.read_csv(file_path, quotechar='"', low_memory=False)
    
    # 查看数据基本信息
    print(f"Total data shape: {df.shape}")
    
    # 筛选印度数据 (B_COUNTRY_ALPHA为IND且C_COW_NUM为750的数据)
    ind_data = df[(df['B_COUNTRY_ALPHA'] == 'CAN') & (df['C_COW_NUM'] == 20)]
    print(f"Russia data shape: {ind_data.shape}")
    
    # 检查问题列是否存在
    for col in QUESTION_COLUMNS:
        if col not in df.columns:
            # 尝试查找对应的列
            possible_columns = [c for c in df.columns if col in c]
            print(f"Column {col} not found. Possible matches: {possible_columns}")
    
    return ind_data

def map_question_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    映射问题列到正确的列名
    
    Args:
        df: 原始数据框
        
    Returns:
        包含映射后问题列的数据框，只包含所有值都>0的行
    """
    # 直接使用数据中的问题列
    # 在WVS数据中，问题列已经有相应的名称
    question_mapping = {
        'Q45': 'Q45',  # Income Equality
        'Q46': 'Q46',  # Private vs State Ownership
        'Q57': 'Q57',  # Government Responsibility
        'Q184': 'Q184',  # Happiness
        'Q218': 'Q218',  # National Prideƒ
        'Q254': 'Q254'  # Confidence in Government
    }
    
    # 创建新的数据框，只包含需要的列
    result_df = pd.DataFrame()
    
    # 添加用户ID列
    result_df['profile_id'] = df['S007'].astype(str)
    result_df['country_name'] = 'CAN'
    result_df['country_code'] = '20'
    
    # 添加问题列
    for new_col, original_col in question_mapping.items():
        if original_col in df.columns:
            result_df[new_col] = df[original_col]
            # 检查数据类型并转换为数值
            result_df[new_col] = pd.to_numeric(result_df[new_col], errors='coerce')
            # 检查是否有缺失值
            missing = result_df[new_col].isna().sum()
            if missing > 0:
                print(f"Column {new_col} ({original_col}) has {missing} missing values")
        else:
            print(f"Column {original_col} not found in the dataset")
            result_df[new_col] = np.nan
    
    # 删除有缺失值的行
    result_df_clean = result_df.dropna(subset=QUESTION_COLUMNS)
    print(f"Clean data shape after removing missing values: {result_df_clean.shape}")
    
    # 只保留所有值都>0的行
    rows_before = len(result_df_clean)
    # 创建一个布尔掩码，标识所有问题列中值都>0的行
    positive_values_mask = True
    for col in QUESTION_COLUMNS:
        positive_values_mask = positive_values_mask & (result_df_clean[col] > 0)
    
    # 应用掩码筛选数据
    result_df_clean = result_df_clean[positive_values_mask]
    
    print(f"Data shape after removing rows with values <= 0: {result_df_clean.shape}")
    print(f"Removed {rows_before - len(result_df_clean)} rows with values <= 0")
    
    return result_df_clean

def process_all_russia_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理所有俄罗斯数据
        
    Returns:
        处理后的数据框
    """
    print(f"Processing all {len(df)} Russia records...")
    
    # 创建结果数据框
    result_df = pd.DataFrame(columns=['profile_id', 'country_name', 'country_code'] + QUESTION_COLUMNS)
    
    # 填充结果
    for i, (_, row) in enumerate(df.iterrows()):
        sample_data = {
            'profile_id': str(i + 1),
            'country_name': 'CAN',
            'country_code': '20'
        }
        
        # 添加问题列的值
        for col in QUESTION_COLUMNS:
            sample_data[col] = f"{row[col]:.2f}"
        
        # 添加到数据框
        result_df = pd.concat([result_df, pd.DataFrame([sample_data])], ignore_index=True)
    
    return result_df

def write_csv_file(file_path: str, data: pd.DataFrame):
    """
    将数据框写入CSV文件
    
    Args:
        file_path: 输出CSV文件路径
        data: 要写入的数据框
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入CSV文件
    data.to_csv(file_path, index=False)
    print(f"Data written to: {file_path}")

def merge_responses(data: pd.DataFrame) -> pd.DataFrame:
    """
    合并所有响应，计算平均值
    
    Args:
        data: 包含所有响应的数据框
        
    Returns:
        合并后的数据框
    """
    # 创建合并结果数据框
    merged_df = pd.DataFrame(columns=['country_code', 'country_name', 'profile_count'] + QUESTION_COLUMNS)
    
    # 计算每个问题的平均值
    merged_data = {
        'country_code': '20',
        'country_name': 'CAN',
        'profile_count': str(len(data))
    }
    
    # 计算每个问题的平均值
    for col in QUESTION_COLUMNS:
        values = pd.to_numeric(data[col], errors='coerce')
        merged_data[col] = f"{values.mean():.2f}"
    
    # 添加到数据框
    merged_df = pd.concat([merged_df, pd.DataFrame([merged_data])], ignore_index=True)
    
    return merged_df

def main():
    # 文件路径
    survey_path = "/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/data/survey.csv"
    output_responses_path = "/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/all_Canda_survey_responses.csv"
    output_merged_path = "/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/merged_all_canda_survey_responses.csv"
    
    # 读取并处理调查数据
    india_data = read_survey_data(survey_path)
    
    # 映射问题列
    mapped_data = map_question_columns(india_data)
    
    # 如果没有足够的数据，打印警告并退出
    if len(mapped_data) == 0:
        print("No valid data found after mapping and cleaning. Exiting.")
        return
    
    # 处理所有俄罗斯数据（只保留所有值都大于0的行）
    processed_data = process_all_russia_data(mapped_data)
    
    # 保存处理结果
    write_csv_file(output_responses_path, processed_data)
    
    # 合并响应
    merged_data = merge_responses(processed_data)
    
    # 保存合并结果
    write_csv_file(output_merged_path, merged_data)
    
    # 打印结果摘要
    print("\nRussia Data Processing Results Summary:")
    print(f"Processed all {len(processed_data)} records from {len(mapped_data)} original data points")
    print(f"Individual responses saved to: {output_responses_path}")
    print(f"Merged response saved to: {output_merged_path}")
    
    # 打印合并结果
    print("\nMerged Response:")
    for col in QUESTION_COLUMNS:
        print(f"  {col}: {merged_data.iloc[0][col]}")

if __name__ == "__main__":
    main()
