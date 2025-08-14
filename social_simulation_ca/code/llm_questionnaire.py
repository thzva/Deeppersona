#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全球问卷生成器 - 生成不同国家用户的问卷回答

此脚本从country_info.csv文件中读取国家信息，
让LLM扮演不同国家人的角色回答问卷问题，并将结果保存到JSON文件。
使用并行处理方法加速API调用，每次处理多个国家。
"""

import json
import os
import csv
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# 导入配置和OpenAI客户端
from config import client, GPT_MODEL, get_completion


def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    """
    读取CSV文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        解析后的CSV数据列表
    """
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append(row)
    return result


def extract_all_countries(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    从数据中提取所有国家的信息
    
    Args:
        data: 包含国家数据的列表
        
    Returns:
        所有国家的信息列表
    """
    all_countries = []
    
    for country_data in data:
        # 设置国家为CAN，国家代码为20
        country_name = "CAN"
        country_code = "20"
        
        country = {
            "country_name": country_name,
            "country_code": country_code
        }
        all_countries.append(country)
    
    return all_countries


def read_questions(file_path: str) -> List[Dict[str, str]]:
    """
    读取问卷问题CSV文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        问题列表
    """
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    
    return questions


def get_single_response_from_llm(country_name: str, question_prompt: str) -> str:
    """
    使用LLM生成用户对单个问题的回答
    
    Args:
        country_name: 用户所在国家名称
        question_prompt: 问题提示
        
    Returns:
        LLM生成的回答
    """
    # 构建提示
    system_content = f"You are an average human being born in {country_name} and living in {country_name} responding to the following survey question."
    
    messages = [
        {"role": "system", "content": system_content}, 
        {"role": "user", "content": question_prompt}
    ]
    
    # 调用API获取回答
    try:
        response = get_completion(messages)
        return response.strip()
    except Exception as e:
        print(f"Error getting response from LLM: {str(e)}")
        return "Error: Unable to generate response"


def get_multiple_responses(country_name: str, question_prompt: str, num_responses: int = 3) -> List[str]:
    """
    为同一个问题获取多次回答
    
    Args:
        country_name: 用户所在国家名称
        question_prompt: 问题提示
        num_responses: 获取回答的次数
        
    Returns:
        多个回答的列表
    """
    responses = []
    
    for i in range(num_responses):
        # 获取单个回答
        response = get_single_response_from_llm(country_name, question_prompt)
        responses.append(response)
        
        # 避免API限制
        time.sleep(0.01)
    
    return responses


def process_country_question(country_data: Dict[str, Any], question: Dict[str, str], 
                          num_responses: int) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个用户的单个问题
    
    Args:
        country_data: 用户数据
        question: 问题数据
        num_responses: 每个问题获取回答的次数
        
    Returns:
        用户ID和包含问题回答的字典
    """
    profile_id = country_data["profile_id"]
    country_name = country_data["country_name"]
    country_code = country_data["country_code"]
    question_id = question["ID"]
    question_prompt = question["Question prompt with response formatting instructions"]
    
    # 初始化用户结果
    user_result = {}
    
    # 使用新的函数获取多次回答，每次都是单独发送给GPT
    responses = get_multiple_responses(country_name, question_prompt, num_responses)
    
    try:
        # 计算平均回答，现在返回的是字符串
        average_response = calculate_average_response(responses, question_id)
        
        # 将问题ID和平均回答添加到用户结果中
        user_result[question_id] = average_response
        
        # 打印进度信息
        print(f"已处理用户 {profile_id} (国家: {country_name}) 的问题 {question_id}")
        print(f"平均回答: {average_response}")
        
        return profile_id, {question_id: average_response}
    except ValueError as e:
        # 如果无法计算有效的数值平均值，记录错误并跳过该问题
        print(f"处理问题 {question_id} 时出错，用户 {profile_id} (国家: {country_name}): {str(e)}")
        print("跳过此问题并继续其他问题...")
        return profile_id, {}


def calculate_average_response(responses: List[str], question_id: str) -> str:
    """
    计算多次回答的数值平均值，保留两位小数
    
    Args:
        responses: 多次回答的列表
        question_id: 问题ID
        
    Returns:
        平均值（字符串格式，保留两位小数）
        
    Raises:
        ValueError: 如果没有有效的数值回答
    """
    import re
    
    # 尝试将所有回答转换为数值并计算平均值
    valid_responses = []
    
    for resp in responses:
        try:
            # 首先尝试直接转换
            try:
                num_value = float(resp.strip())
                valid_responses.append(num_value)
                continue
            except (ValueError, TypeError):
                pass
            
            # 尝试匹配“评价...为数字”或“给出数字”的通用模式
            number_match = re.search(r'\b(rate|give|score|evaluate|assess|as a|as an)\b[^.]*?\b([1-9]|10)\b', resp.lower())
            if number_match:
                num_value = float(number_match.group(2))
                valid_responses.append(num_value)
                continue
            
            # 对于其他问题，尝试从文本中提取数字
            # 匹配单个数字或数字范围
            number_match = re.search(r'\b([1-9]|10)\b', resp)
            if number_match:
                num_value = float(number_match.group(1))
                valid_responses.append(num_value)
                continue
                
            # 尝试匹配数字单词
            number_words = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            for word, value in number_words.items():
                if re.search(r'\b' + word + r'\b', resp.lower()):
                    valid_responses.append(float(value))
                    continue
                
            print(f"Warning: Could not convert response '{resp}' to number for question {question_id}")
        except Exception as e:
            print(f"Error processing response '{resp}' for question {question_id}: {str(e)}")
            continue
    
    # 如果有有效的数值回答，计算平均值并保留两位小数
    if valid_responses:
        average = sum(valid_responses) / len(valid_responses)
        # 保留两位小数，确保即使是整数值也显示两位小数
        return f"{average:.2f}"
    
    # 如果没有有效的数值回答，直接报错
    raise ValueError(f"No valid numerical responses for question {question_id}")


def process_country(country: Dict[str, Any], questions: List[Dict[str, str]],
                   num_responses: int) -> Dict[str, Dict[str, Any]]:
    """
    处理单个用户的所有问题
    
    Args:
        country: 用户信息
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含用户回答的字典
    """
    profile_id = country["profile_id"]
    country_name = country["country_name"]
    country_code = country["country_code"]
    
    # 初始化用户结果
    user_result = {
        "profile_id": profile_id,
        "country_name": country_name,
        "country_code": country_code
    }
    
    # 处理每个问题
    for question in questions:
        try:
            # 调用处理单个用户单个问题的函数
            _, question_result = process_country_question(country, question, num_responses)
            
            # 将平均回答添加到用户结果中
            if question_result:
                for question_id, response in question_result.items():
                    user_result[question_id] = response
        except Exception as e:
            # 如果处理问题时出错，记录错误并跳过该问题
            print(f"处理用户 {profile_id} (国家: {country_name}) 的问题时出错: {str(e)}")
            print("跳过此问题并继续其他问题...")
    
    return {profile_id: user_result}


def generate_responses_for_users(users: List[Dict[str, Any]], questions: List[Dict[str, str]], num_responses: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    为每个用户生成问题的多次回答并计算平均值，使用并行处理方法
    每次处理多个用户，提高处理速度
    
    Args:
        users: 用户信息列表
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含用户信息和问题平均回答的字典
    """
    
    results = {}
    
    total_users = len(users)
    print(f"Generating responses for {total_users} users across {len(questions)} questions (each question will be answered {num_responses} times)...")
    print(f"Total number of questions to process: {total_users * len(questions)}")
    print(f"Using parallel processing to handle multiple users at once.")
    
    # 创建一个进度条
    progress_bar = tqdm(total=total_users, desc="Processing users")
    
    # 使用并行处理，每次处理5个用户
    batch_size = 5
    for i in range(0, total_users, batch_size):
        batch_users = users[i:i+batch_size]
        batch_results = {}
        
        # 使用线程池并行处理用户
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # 提交任务
            future_to_user = {executor.submit(process_country, user, questions, num_responses): user 
                             for user in batch_users}
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_user):
                user = future_to_user[future]
                try:
                    user_result = future.result()
                    batch_results.update(user_result)
                    # 更新进度条
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error processing user {user['profile_id']}: {str(e)}")
                    progress_bar.update(1)
        
        # 更新总结果
        results.update(batch_results)
        
        # 打印处理进度
        print(f"\nProcessed {len(results)}/{total_users} users.")
    
    # 关闭进度条
    progress_bar.close()
    
    return results


def save_to_json(data: Dict[str, Any], output_file: str):
    """
    将数据保存到JSON文件
    
    Args:
        data: 要保存的数据
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 获取文件路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    questions_path = data_dir / "questions.csv"
    output_path = output_dir / "llm_responses.json"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取问题数据
    print(f"Reading questions from: {questions_path}")
    questions = read_questions(str(questions_path))
    print(f"Successfully read {len(questions)} questions")
    
    # 创建100个用户
    people = []
    for i in range(1, 101):
        person = {
            "profile_id": str(i),
            "country_name": "CAN",
            "country_code": "20"
        }
        people.append(person)
    print(f"Created {len(people)} people profiles")
    
    # 设置每个问题回答的次数
    num_responses = 3
    
    # 生成问题的回答，使用并行处理
    results = generate_responses_for_users(people, questions, num_responses)
    print(f"Generated responses for {len(results)} people (each question answered {num_responses} times)")
    
    # 保存结果
    print(f"Saving all people's responses to: {output_path}")
    save_to_json(results, str(output_path))
    print("Response generation complete!")
    
    # 打印示例
    if results:
        print("\n示例用户及其回答:")
        sample_person_id = list(results.keys())[0]
        sample_person = results[sample_person_id]
        print(f"用户ID: {sample_person_id} (国家: {sample_person.get('country_name', 'CAN')}, 国家代码: {sample_person.get('country_code', '20')})")

        
        # 打印一个问题的回答示例
        if questions and len(questions) > 0:
            question_id = questions[0]["ID"]
            
            print(f"问题ID: {question_id}")
            print(f"平均回答: {sample_person.get('responses', {}).get(question_id, '无回答')}")
            print(f"(这是{num_responses}次单独回答的平均值)")
            
            # 显示所有问题的平均回答
            print("\n该用户的所有问题回答:")
            for q in questions:
                q_id = q["ID"]
                q_response = sample_person.get('responses', {}).get(q_id, "No response")
                print(f"  {q_id}: {q_response}")
    
    print(f"\nAll results have been saved to: {output_path}")

if __name__ == "__main__":
    main()
