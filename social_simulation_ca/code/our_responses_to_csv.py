
import json
import csv
import os
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


def extract_responses_to_csv(input_json_path: str, output_csv_path: str):
    """
    从用户回答JSON中提取数据并保存为CSV
    
    Args:
        input_json_path: 输入JSON文件路径
        output_csv_path: 输出CSV文件路径
    """
    # 读取用户回答数据
    print(f"Reading user responses from: {input_json_path}")
    user_responses = read_json_file(input_json_path)
    print(f"Successfully read responses for {len(user_responses)} users")
    
    # 我们关心的问题ID
    target_question_ids = ["Q45", "Q46", "Q57", "Q184", "Q218", "Q254"]
    print(f"Target question IDs: {', '.join(target_question_ids)}")
    
    # 准备用户数据列表
    user_data_list = []
    
    # 处理每个用户的回答
    for profile_id, user_info in user_responses.items():
        country_name = user_info.get('country_name', '')
        country_code = user_info.get('country_code', '999')
        
        if not country_name:  # 跳过没有国家名称的数据
            continue
        
        # 创建用户数据字典，使用country_country_code格式的profile_id
        user_data = {
            'profile_id': f"country_{country_code}",
            'country_name': country_name,
            'country_code': country_code
        }
        
        # 添加每个目标问题的回答
        for question_id in target_question_ids:
            if question_id in user_info:
                # 保留原始值（已经是字符串格式）
                user_data[question_id] = user_info[question_id]
            else:
                user_data[question_id] = "N/A"
        
        user_data_list.append(user_data)
        print(f"Processed user from: {country_name} ({country_code})")
    
    # 按国家名称排序
    user_data_list.sort(key=lambda x: x['country_name'])
    
    # 准备CSV的列
    fieldnames = ['profile_id', 'country_name', 'country_code'] + target_question_ids
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入CSV文件
    print(f"Writing user data to CSV: {output_csv_path}")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(user_data_list)
    
    print(f"Successfully wrote user data to CSV. File contains {len(user_data_list)} users and {len(fieldnames)} columns.")


def main():
    # 获取文件路径
    input_json_path = "/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/our_responses.json"
    output_csv_path = "/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/output/our_responses.csv"
    
    # 从用户回答提取数据并保存为CSV
    extract_responses_to_csv(str(input_json_path), str(output_csv_path))


if __name__ == "__main__":
    main()
