#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全球问卷生成器 - 从所有国家的用户数据中提取用户并生成问卷

此脚本从country_summaries.json文件中提取所有国家的用户信息，
让LLM扮演用户角色回答问卷问题，并将结果保存到JSON文件。
使用并行处理方法加速API调用，每次处理多个用户。
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


def extract_all_users(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从数据中只提取用户摘要信息
    
    Args:
        data: 包含用户数据的字典
        
    Returns:
        所有用户的摘要信息列表
    """
    all_users = []
    
    for profile_id, profile_data in data.items():
        # 只提取profile_id和summary
        if profile_id == "metadata":
            continue
            
        user = {
            "profile_id": profile_id,
            "summary": profile_data.get("summary", "")
        }
        all_users.append(user)
    
    return all_users


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


def get_single_response_from_llm(user_summary: str, question_prompt: str) -> str:
    """
    使用LLM生成用户对单个问题的回答
    
    Args:
        user_summary: 用户摘要
        question_prompt: 问题提示
        
    Returns:
        LLM生成的回答
    """
    # 构建提示
    system_content = f"You're a person with the following background Person's profile.\n\nPerson's profile:\n{user_summary}\n\nYou must respond as if you are this person. Follow any formatting instructions in the question exactly."
    
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


def get_multiple_responses(user_summary: str, question_prompt: str, num_responses: int = 3) -> List[str]:
    """
    为同一个问题获取多次回答
    
    Args:
        user_summary: 用户摘要
        question_prompt: 问题提示
        num_responses: 获取回答的次数
        
    Returns:
        多个回答的列表
    """
    responses = []
    
    for i in range(num_responses):
        # 获取单个回答
        response = get_single_response_from_llm(user_summary, question_prompt)
        responses.append(response)
        
        # 避免API限制
        time.sleep(0.01)
    
    return responses


def process_user_question(user_data: Dict[str, Any], question: Dict[str, str], 
                           num_responses: int) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个用户的单个问题
    
    Args:
        user_data: 用户数据
        question: 问题数据
        num_responses: 每个问题获取回答的次数
        
    Returns:
        用户ID和包含问题回答的字典
    """
    profile_id = user_data["profile_id"]
    user_summary = user_data["summary"]
    question_id = question["ID"]
    question_prompt = question["Question prompt with response formatting instructions"]
    
    # 初始化用户结果，包含原始用户信息
    user_result = {}
    
    # 使用新的函数获取多次回答，每次都是单独发送给GPT
    responses = get_multiple_responses(user_summary, question_prompt, num_responses)
    
    try:
        # 计算平均回答，现在返回的是字符串
        average_response = calculate_average_response(responses, question_id)
        
        # 将问题ID和平均回答添加到用户结果中
        user_result[question_id] = average_response
        
        # 打印进度信息
        print(f"\nProcessed question {question_id} for user {profile_id}")
        print(f"Average response: {average_response}")
        
        return profile_id, {question_id: average_response}
    except ValueError as e:
        # 如果无法计算有效的数值平均值，记录错误并跳过该问题
        print(f"\nError processing question {question_id} for user {profile_id}: {str(e)}")
        print("Skipping this question and continuing with others...")
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
    # 尝试将所有回答转换为数值并计算平均值
    valid_responses = []
    for resp in responses:
        try:
            # 尝试提取数值
            num_value = float(resp.strip())
            valid_responses.append(num_value)
        except (ValueError, TypeError):
            # 如果无法转换为数值，则跳过
            print(f"Warning: Could not convert response '{resp}' to number for question {question_id}")
            continue
    
    # 如果有有效的数值回答，计算平均值并保留两位小数
    if valid_responses:
        average = sum(valid_responses) / len(valid_responses)
        # 保留两位小数，确保即使是整数值也显示两位小数
        return f"{average:.2f}"
    
    # 如果没有有效的数值回答，直接报错
    raise ValueError(f"No valid numerical responses for question {question_id}")


def process_user(user: Dict[str, Any], questions: List[Dict[str, str]], 
                 num_responses: int) -> Dict[str, Dict[str, Any]]:
    """
    处理单个用户的所有问题
    
    Args:
        user: 用户信息
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含用户回答的字典
    """
    profile_id = user["profile_id"]
    
    # 初始化用户结果，包含原始用户信息和默认国家信息
    user_result = {
        "profile_id": user["profile_id"],
        "country_name": "CAN",
        "country_code": "20",
        "summary": user["summary"]
    }
    
    # 遍历每个问题
    for question in questions:
        try:
            _, question_result = process_user_question(user, question, num_responses)
            # 更新用户结果
            user_result.update(question_result)
        except Exception as e:
            print(f"Error processing user {profile_id} question {question['ID']}: {str(e)}")
    
    print(f"\nCompleted all questions for user {profile_id}\n")
    return {profile_id: user_result}


def generate_responses_for_users(users: List[Dict[str, Any]], questions: List[Dict[str, str]], num_responses: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    为每个用户生成问题的多次回答并计算平均值，使用并行处理方法
    每次处理多个用户，提高处理速度
    
    Args:
        users: 用户信息列表
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含原始用户信息和问题平均回答的字典
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
            future_to_user = {executor.submit(process_user, user, questions, num_responses): user 
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
        
        # 显示处理进度
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
    """
    主函数
    """
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输入和输出文件路径
    input_file = Path("/home/zhou/persona/world_value_survey/gpt-4.1-mini_ca/data/extracted_summaries_CAN.json")
    questions_file = data_dir / "questions.csv"
    output_file = output_dir / "our_responses.json"
    
    # 读取数据
    print(f"Reading data from {input_file}...")
    data = read_json_file(str(input_file))
    
    # 提取所有用户
    print("Extracting users...")
    all_users = extract_all_users(data)
    print(f"Extracted {len(all_users)} users")
    
    # 读取问题
    print(f"Reading questions from {questions_file}...")
    questions = read_questions(str(questions_file))
    print(f"Read {len(questions)} questions")
    
    # 设置每个问题获取回答的次数
    num_responses = 3
    
    # 限制处理的用户数量（用于测试）
    # 如果要处理所有用户，请将此值设置为 None
    max_users = None
    
    if max_users:
        all_users = all_users[:max_users]
        print(f"Limited to {max_users} users for testing")
    
    # 生成回答
    print(f"Generating responses for {len(all_users)} users, {len(questions)} questions, {num_responses} responses per question...")
    results = generate_responses_for_users(all_users, questions, num_responses)
    
    # 保存结果
    print(f"Saving results to {output_file}...")
    save_to_json(results, str(output_file))
    
    print("Done!")
    # 打印示例
    if results:
        print("\nExample user with responses:")
        sample_user_id = list(results.keys())[0]
        sample_user = results[sample_user_id]
        print(f"User: {sample_user_id}")
        
        # 打印一个问题的回答示例
        if questions and len(questions) > 0:
            question_id = questions[0]["ID"]
            
            print(f"Question ID: {question_id}")
            print(f"Average Response: {sample_user.get(question_id, 'No response')}")
            print(f"(This is the average of {num_responses} individual responses)")
            
            # 显示所有问题的平均回答
            print("\nAll question responses for this user:")
            for q in questions:
                q_id = q["ID"]
                q_response = sample_user.get(q_id, "No response")
                print(f"  {q_id}: {q_response}")


if __name__ == "__main__":
    main()
