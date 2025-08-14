#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    读取JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_character_responses_to_csv(input_json_path: str, output_csv_path: str, include_name: bool = False):
    """
    从角色回答JSON中提取数据并保存为CSV
    
    Args:
        input_json_path: 输入JSON文件路径
        output_csv_path: 输出CSV文件路径
    """
    # 读取角色回答数据
    print(f"Reading character responses from: {input_json_path}")
    character_responses = read_json_file(input_json_path)
    print(f"Successfully read responses for {len(character_responses)} characters")
    
    # 我们关心的问题ID
    target_question_ids = ["Q45", "Q46", "Q57", "Q184", "Q218", "Q254"]
    print(f"Target question IDs: {', '.join(target_question_ids)}")
    
    # 准备角色数据列表
    character_data_list = []
    
    # 处理每个角色的回答
    for index, (character_id, character_info) in enumerate(character_responses.items(), 1):
        # 提取角色基本信息
        name = character_info.get('Name', '')
        country_name = character_info.get('country_name', 'USA')
        country_code = character_info.get('country_code', '840')  # 美国的ISO国家代码
        
        # 创建角色数据字典
        character_data = {
            'character_id': str(index),  # 将character_id改为序号格式（1, 2, 3, 4...）
            'country_name': country_name,
            'country_code': country_code
        }
        
        # 如果需要包含名字，则添加name字段
        if include_name:
            character_data['name'] = name
        
        # 添加每个目标问题的回答
        for question_id in target_question_ids:
            if question_id in character_info:
                # 保留原始值（已经是字符串格式）
                character_data[question_id] = character_info[question_id]
            else:
                character_data[question_id] = "N/A"
        
        character_data_list.append(character_data)
        print(f"Processed character: {character_id} - {name} (Country: {country_name})")

    
    # 按character_id排序
    character_data_list.sort(key=lambda x: x['character_id'])
    
    # 准备CSV的列
    fieldnames = ['character_id', 'country_name', 'country_code']
    if include_name:
        fieldnames.insert(1, 'name')
    fieldnames.extend(target_question_ids)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入CSV文件
    print(f"Writing character data to CSV: {output_csv_path}")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(character_data_list)
    
    print(f"Successfully wrote character data to CSV. File contains {len(character_data_list)} characters and {len(fieldnames)} columns.")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将角色回答JSON转换为CSV格式')
    parser.add_argument('--input', '-i', type=str, 
                        default='/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/opencharacter_responses.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, 
                        default='/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/opencharacter_responses.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--include-name', '-n', action='store_true',
                        help='是否在CSV中包含name列')
    args = parser.parse_args()
    
    # 从角色回答提取数据并保存为CSV
    extract_character_responses_to_csv(args.input, args.output, args.include_name)


if __name__ == "__main__":
    main()
