#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
角色问卷生成器 - 从美国角色数据中提取角色并生成问卷回答

此脚本从usa_characters_100.json文件中提取角色信息，
让LLM扮演角色回答问卷问题，并将结果保存到JSON文件。
使用并行处理方法加速API调用，每次处理多个角色。
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


def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSON文件，处理不同格式的角色数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的角色数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据格式并提取角色列表
    characters = []
    
    # 如果是列表格式，直接返回
    if isinstance(data, list):
        return data
    
    # 如果是字典格式，检查是否是OpenCharacter格式
    elif isinstance(data, dict):
        # 检查是否包含metadata和Profile_开头的键
        if "metadata" in data:
            # 这是all_profiles_India.json格式，提取所有Profile_开头的键
            for key, value in data.items():
                if key.startswith("Profile_") and isinstance(value, dict):
                    # 添加character_id字段
                    profile = value.copy()
                    profile["character_id"] = key
                    
                    # 如果有Summary字段，确保能够提取到
                    if "Summary" in profile:
                        profile["persona"] = profile["Summary"]
                    
                    characters.append(profile)
        else:
            # 其他字典格式，可能是每个键都是一个角色ID
            for key, value in data.items():
                if isinstance(value, dict):
                    character = value.copy()
                    character["character_id"] = key
                    characters.append(character)
    
    print(f"Extracted {len(characters)} characters from the input file")
    return characters


def extract_character_summary(character: Dict[str, Any]) -> str:
    """
    从角色数据中提取摘要信息
    
    Args:
        character: 角色数据
        
    Returns:
        角色摘要信息
    """
    # 提取从Name到Personality的所有字段
    summary_parts = []
    
    fields = ["Name", "Age", "Gender", "Race", "Born Place", "Appearance", "General Experience", "Personality"]
    
    for field in fields:
        if field in character:
            summary_parts.append(f"{field}: {character[field]}")
    
    return "\n\n".join(summary_parts)


def read_questions(file_path: str) -> List[Dict[str, str]]:
    """
    读取问卷问题CSV文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        问题列表
    """
    questions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 跳过空行
                if all(not v for v in row.values()):
                    continue
                    
                # 确保所有必要的字段都存在
                if 'ID' in row and 'Question prompt with response formatting instructions' in row:
                    questions.append({
                        'ID': row['ID'],
                        'prompt': row['Question prompt with response formatting instructions']
                    })
    except Exception as e:
        print(f"Error reading questions file: {e}")
    
    return questions


def get_single_response_from_llm(character_summary: str, question_prompt: str) -> str:
    """
    使用LLM生成角色对单个问题的回答
    
    Args:
        character_summary: 角色摘要
        question_prompt: 问题提示
        
    Returns:
        LLM生成的回答
    """
    try:
        # 构建提示
        system_content = f"You're a person with the following background Person's profile.\n\nPerson's profile:\n{character_summary}\n\nYou must respond as if you are this person. Follow any formatting instructions in the question exactly."
        
        messages = [
            {"role": "system", "content": system_content}, 
            {"role": "user", "content": question_prompt}
        ]
        
        # 调用LLM
        response = get_completion(
            messages=messages,
            model=GPT_MODEL,
            temperature=0.7
        )
        
        return response.strip()
    except Exception as e:
        print(f"Error getting response from LLM: {e}")
        return ""


def get_multiple_responses(character_summary: str, question_prompt: str, num_responses: int = 3) -> List[str]:
    """
    为同一个问题获取多次回答
    
    Args:
        character_summary: 角色摘要
        question_prompt: 问题提示
        num_responses: 获取回答的次数
        
    Returns:
        多个回答的列表
    """
    responses = []
    
    for _ in range(num_responses):
        response = get_single_response_from_llm(character_summary, question_prompt)
        if response:
            responses.append(response)
    
    return responses


def process_character_question(character_data: Dict[str, Any], question: Dict[str, str], 
                           num_responses: int) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个角色的单个问题
    
    Args:
        character_data: 角色数据
        question: 问题数据
        num_responses: 每个问题获取回答的次数
        
    Returns:
        角色ID和包含问题回答的字典
    """
    try:
        character_id = character_data["character_id"]
        question_id = question["ID"]
        question_prompt = question["prompt"]
        
        # 提取角色摘要
        character_summary = extract_character_summary(character_data)
        
        # 获取多次回答
        responses = get_multiple_responses(character_summary, question_prompt, num_responses)
        
        # 计算平均回答
        average_response = calculate_average_response(responses, question_id)
        
        # 返回结果
        result = {
            question_id: average_response,
            f"{question_id}_raw_responses": responses
        }
        
        return character_id, result
    except Exception as e:
        print(f"Error processing question for character {character_data.get('character_id', 'unknown')}: {e}")
        return character_data.get("character_id", "unknown"), {}


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
    if not responses:
        return "No response"
    
    # 尝试提取数值
    numeric_responses = []
    
    for response in responses:
        # 清理回答，只保留数字
        cleaned_response = response.strip()
        
        # 尝试提取数字
        try:
            # 首先尝试直接转换为浮点数
            numeric_value = float(cleaned_response)
            numeric_responses.append(numeric_value)
        except ValueError:
            # 如果直接转换失败，尝试从文本中提取数字
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_response)
            if numbers:
                try:
                    numeric_value = float(numbers[0])
                    numeric_responses.append(numeric_value)
                except ValueError:
                    pass
    
    # 如果有有效的数值回答，计算平均值
    if numeric_responses:
        average = sum(numeric_responses) / len(numeric_responses)
        return f"{average:.2f}"
    else:
        # 如果没有有效的数值回答，返回最常见的回答
        from collections import Counter
        most_common = Counter(responses).most_common(1)
        if most_common:
            return most_common[0][0]
        else:
            return "No valid response"


def process_character(character: Dict[str, Any], questions: List[Dict[str, str]], 
                 num_responses: int) -> Dict[str, Any]:
    """
    处理单个角色的所有问题
    
    Args:
        character: 角色信息
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含角色回答的字典
    """
    try:
        character_id = character["character_id"]
        print(f"Processing character: {character_id}")
        
        # 初始化结果字典，包含角色基本信息
        result = {
            "character_id": character_id,
            "persona": character.get("persona", ""),
            "Name": character.get("Name", ""),
            "country_name": character.get("country_name", "")
        }
        
        # 处理每个问题
        for question in questions:
            _, question_result = process_character_question(character, question, num_responses)
            result.update(question_result)
        
        return result
    except Exception as e:
        print(f"Error processing character {character.get('character_id', 'unknown')}: {e}")
        return {"character_id": character.get("character_id", "unknown"), "error": str(e)}


def generate_responses_for_characters(characters: List[Dict[str, Any]], questions: List[Dict[str, str]], num_responses: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    为每个角色生成问题的多次回答并计算平均值，使用并行处理方法
    每次处理多个角色，提高处理速度
    
    Args:
        characters: 角色信息列表
        questions: 问题列表
        num_responses: 每个问题获取回答的次数
        
    Returns:
        包含原始角色信息和问题平均回答的字典
    """
    results = {}
    total_characters = len(characters)
    
    print(f"Generating responses for {total_characters} characters...")
    
    # 设置进度条
    progress_bar = tqdm(total=total_characters, desc="Processing characters")
    
    # 使用线程池并行处理角色
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有角色处理任务
        future_to_character = {executor.submit(process_character, character, questions, num_responses): character for character in characters}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_character):
            character = future_to_character[future]
            try:
                character_result = future.result()
                if character_result and "character_id" in character_result:
                    character_id = character_result["character_id"]
                    results[character_id] = character_result
            except Exception as e:
                print(f"Error processing character {character.get('character_id', 'unknown')}: {e}")
            
            # 更新进度条
            progress_bar.update(1)
    
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
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


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
    input_file = data_dir / "canda_characters.json"
    questions_file = data_dir / "questions.csv"
    output_file = output_dir / "opencharacter_responses.json"
    
    # 读取数据
    print(f"Reading data from {input_file}...")
    characters = read_json_file(str(input_file))
    print(f"Read {len(characters)} characters")
    
    # 读取问题
    print(f"Reading questions from {questions_file}...")
    questions = read_questions(str(questions_file))
    print(f"Read {len(questions)} questions")
    
    # 设置每个问题获取回答的次数
    num_responses = 3
    
    # 限制处理的角色数量（用于测试）
    # 如果要处理所有角色，请将此值设置为 None
    max_characters = None
    
    if max_characters:
        characters = characters[:max_characters]
        print(f"Limited to {max_characters} characters for testing")
    
    # 生成回答
    print(f"Generating responses for {len(characters)} characters, {len(questions)} questions, {num_responses} responses per question...")
    results = generate_responses_for_characters(characters, questions, num_responses)
    
    # 保存结果
    print(f"Saving results to {output_file}...")
    save_to_json(results, str(output_file))
    
    print("Done!")
    # 打印示例
    if results:
        print("\nExample character with responses:")
        sample_character_id = list(results.keys())[0]
        sample_character = results[sample_character_id]
        print(f"Character: {sample_character.get('Name', sample_character_id)}")
        
        # 打印一个问题的回答示例
        if questions and len(questions) > 0:
            question_id = questions[0]["ID"]
            
            print(f"Question ID: {question_id}")
            print(f"Average Response: {sample_character.get(question_id, 'No response')}")
            print(f"(This is the average of {num_responses} individual responses)")
            
            # 显示所有问题的平均回答
            print("\nAll question responses for this character:")
            for q in questions:
                q_id = q["ID"]
                q_response = sample_character.get(q_id, "No response")
                print(f"  {q_id}: {q_response}")


if __name__ == "__main__":
    main()