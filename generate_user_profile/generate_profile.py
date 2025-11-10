#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import random
import sys
import time
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from config import get_completion
import subprocess
# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_str(value):
    """
    Ensure a string is returned.
      - If value is already a str, return it.
      - Otherwise (dict, list, or other), return the JSON serialization with multi-line formatting
    """
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)

def get_project_root() -> str:
    """获取项目根目录的路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    return project_root


def copy_files_from_source_to_target():
    """复制文件从源位置到目标位置"""
    # 源路径 - 现在已经不需要复制，因为我们直接保存到正确的目录
    # 但为了兼容性，我们保留这个函数
    correct_output_dir = os.path.join(get_project_root(), "output")
    
    # 确保目标目录存在
    os.makedirs(correct_output_dir, exist_ok=True)
    
    print(f"输出目录已设置为: {correct_output_dir}")
    return True


def get_timestamped_filename(base_path: str) -> str:
    """为文件路径添加时间戳
    
    Args:
        base_path: 基础文件路径
        
    Returns:
        str: 带时间戳的文件路径
    """
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(directory, timestamped_filename)


def save_json_file(file_path: str, data: Dict, use_timestamp: bool = True) -> str:
    """保存JSON文件
    
    Args:
        file_path: 目标文件路径
        data: 要保存的数据
        use_timestamp: 是否使用时间戳，默认为True
        
    Returns:
        str: 实际保存的文件路径
    """
    try:
        if use_timestamp:
            actual_path = get_timestamped_filename(file_path)
        else:
            actual_path = file_path
            
        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return actual_path
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
        return file_path


def extract_paths(obj: Dict, prefix: str = "") -> List[str]:
    """从嵌套的JSON对象中提取所有属性路径
    
    Args:
        obj: 嵌套的JSON对象
        prefix: 当前路径前缀
        
    Returns:
        List[str]: 属性路径列表
    """
    paths = []
    for key, value in obj.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if not value:  # 空字典表示叶子节点
                paths.append(new_prefix)
            else:
                paths.extend(extract_paths(value, new_prefix))
    return paths



def generate_category_attributes(category_paths: Dict, custom_prompt: str, category_name: str) -> Dict:
    """一次性生成一个一级大类下的所有属性值。
    
    参数:
        category_paths: 一级大类下的所有属性路径及其结构。
        custom_prompt: 自定义的完整prompt，包含具体的生成指令。
        category_name: 一级大类名称。
        
    返回:
        Dict: 生成的所有属性值。
    """
    # 收集该类别下的所有叶子节点路径
    leaf_paths = []
    
    def collect_leaf_paths(obj, current_path):
        for key, value in obj.items():
            path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, dict):
                if not value:  # 叶子节点
                    leaf_paths.append(path)
                else:
                    collect_leaf_paths(value, path)
    
    collect_leaf_paths(category_paths, "")
    
    # 如果没有叶子节点，直接返回空字典
    if not leaf_paths:
        return {}
    
    # 简化的系统提示，只负责JSON格式
    system_prompt = """Format your response as a JSON object where each key is the attribute path and each value is the generated attribute value (not exceeding 100 characters)."""
    
    # 使用自定义prompt + 属性路径列表
    user_prompt = f"{custom_prompt}\n\nAttribute Paths to generate values for:\n"
    for path in leaf_paths:
        user_prompt += f"- {path}\n"
    user_prompt += "\nGenerate suitable values for all these attributes in JSON format."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        print(f"  正在一次性生成 {category_name} 下的 {len(leaf_paths)} 个属性值...")
        response = get_completion(messages)
        if not response:
            print(f"  生成 {category_name} 属性值失败: 空响应")
            return {}
            
        # 尝试解析JSON响应
        try:
            import json
            # 清理响应，移除可能的markdown代码块标记
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            generated_values = json.loads(cleaned_response)
            print(f"  成功生成 {len(generated_values)} 个属性值")
            return generated_values
        except json.JSONDecodeError as e:
            print(f"  解析 {category_name} 属性值JSON失败: {e}")
            print(f"  响应内容: {response[:100]}..." if len(response) > 100 else f"响应内容: {response}")
            return {}
    except Exception as e:
        print(f"  生成 {category_name} 属性值时出错: {e}")
        return {}


def generate_final_summary(profile: Dict, base_info: Dict = None) -> str:
    """为用户档案生成最终摘要。
    
    参数:
        profile: 完整的用户档案数据。
        base_info: 基础信息，包含life_story等内容。
    返回:
        str: 最终的摘要文本。
    """
    system_prompt = """
Your task: Based solely on the provided user attributes and personal story, create an objective and factual personal profile, strictly between 150–400 words.

Content Requirements:
	•	The profile must be written entirely in the first-person perspective.
	•	The output should be a coherent, logically structured narrative, not a list of points. The order may vary: it does not need to follow the fixed “background → challenge → conclusion” pattern, and may instead begin with daily life or interests.
 	•	The opening must explicitly state my country or region, ensuring that geographic location is clearly highlighted at the very start.
	•	Must include:
	1.	Basic background (e.g., location, identity)
	2.	Daily life or work routines
	3.	Personal interests and hobbies (explicitly highlighted)
	4.	Behavioral tendencies or values (positive or negative)
	•	Interests and hobbies must be integrated naturally, not superficially. Add small, ordinary details (e.g., food preferences, leisure activities, quirks) that make the character feel real.
	•	If there are negative traits, imperfections, or contradictions, they must be represented faithfully without softening. Do not reframe them as “growth” or “lessons learned.”
	•	No declarative or reflective endings. Avoid abstract statements like “I’ve learned…,” “This shows…,” or “Success means….” The ending should remain grounded in daily routines or interests.
	•	Only include information explicitly provided in the attributes and story. No invention, speculation, or interpretation.
	•	Prohibit the use of words such as' balance 'and' balance '
"""
    user_prompt = f"Complete Profile (in JSON format):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
    
    user_prompt +="""Generate a first-person narrative of 100-400 words from the provided profile. Your primary goal is to make the person feel real, believable, and authentic.

To achieve this, strictly follow the 'Show, Don't Tell' principle:
1.  **Illustrate, Don't Declare:** Show values and traits through specific actions, stories, and decisions, rather than stating them directly.
2.  **Connect Actions to Motivation:** Briefly explain the 'why' behind key life choices and habits to reveal the person's inner logic and create narrative depth.
3.  **Maintain a Natural Voice:** The tone must be sincere and grounded—thoughtful but not overly abstract or dramatic.

Weave all elements into a cohesive story, not a simple list of facts."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = get_completion(messages)
        summary = response.strip() if response else ""
        # Check if word count is within acceptable range (100-400 words)
        word_count = len(summary.split())
        if word_count < 100:
            print(f"Warning: Summary is only {word_count} words (minimum 100)")
        elif word_count > 400:
            summary = enforce_word_limit(summary, 400)
            print(f"Summary was adjusted to 400 words (from {word_count})")
        else:
            print(f"Summary generated with {word_count} words")
        return summary
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return ""


def print_section(section: Dict, indent: int = 0) -> None:
    """打印配置部分的内容
    
    参数:
        section: 要打印的配置部分
        indent: 缩进级别
    """
    indent_str = "  " * indent
    for key, value in section.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_section(value, indent + 1)
        else:
            print(f"{indent_str}{key}: {value}")


def generate_section(template_section: Dict, base_info: str, section_name: str, indent: int = 0) -> Dict:
    """生成配置文件的一个部分。
    
    参数:
        template_section: 模板中的对应部分。
        base_info: 基础信息文本。
        section_name: 部分名称。
        indent: 缩进级别。
        
    返回:
        Dict: 生成的配置部分。
    """
    section_result = {}
    indent_str = "  " * indent
    
    print(f"{indent_str}正在生成 {section_name} 部分...")
    
    # 如果是一级大类，一次性生成所有属性
    if indent == 0:  # 一级大类
        # 使用新函数一次性生成所有属性值
        all_attributes = generate_category_attributes(template_section, base_info, section_name)
        
        # 如果成功生成了属性值，将其添加到结果中
        if all_attributes:
            # 构建结果字典
            for path, value in all_attributes.items():
                # 分解路径
                parts = path.split('.')
                # 跳过第一部分（大类名称）
                if len(parts) > 1 and parts[0] == section_name:
                    parts = parts[1:]
                
                # 递归构建嵌套字典
                current = section_result
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:  # 最后一个部分，设置值
                        current[part] = value
                        print(f"{indent_str}  - {'.'.join(parts)}: {value}")
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            
            return section_result
    
    # 如果不是一级大类或者一次性生成失败，则使用原来的递归方式
    for key, value in template_section.items():
        current_path = f"{section_name}.{key}" if section_name else key
        
        if isinstance(value, dict):
            if not value:  # 叶子节点
                generated_value = generate_attribute_value(current_path, base_info)
                section_result[key] = generated_value
                print(f"{indent_str}  - {key}: {generated_value}")
            else:  # 嵌套节点
                section_result[key] = generate_section(value, base_info, current_path, indent + 1)
    
    return section_result


def enforce_word_limit(text: str, limit: int = 300) -> str:
    """将文本修剪为最多`limit`个单词。"""
    words = text.split()
    if len(words) > limit:
        return ' '.join(words[:limit])
    return text


def append_profile_to_json(file_path: str, profile: Dict, use_timestamp: bool = True) -> str:
    """追加个人资料到 JSON 文件
    
    参数:
        file_path: 目标文件路径
        profile: 要追加的个人资料
        use_timestamp: 是否使用时间戳，默认为True
        
    返回:
        str: 实际保存的文件路径
    """
    try:
        if use_timestamp:
            actual_path = get_timestamped_filename(file_path)
            profiles = [profile]  # 新文件，只包含当前profile
        else:
            actual_path = file_path
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
            else:
                profiles = []
            profiles.append(profile)
        
        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        return actual_path
    except Exception as e:
        print(f"追加个人资料到 JSON 文件时出错: {e}")
        return file_path


def generate_single_profile(template: Dict = None, profile_index: int = 0, attribute_count: int = 200) -> Dict:
    """根据给定的模板生成完整的用户档案。
    
    参数:
        template: 可选的用于生成的模板。
        profile_index: 要生成的档案索引。
        attribute_count: 要包含的属性数量。
        
    返回:
        Dict: 生成的用户档案。
    """

    
    # First, run select_attributes.py to update base files (user_profile.json and selected_paths.json)
    print(f'Running select_attributes.py to update base files with {attribute_count} attributes...')
    try:
        # 直接导入select_attributes模块的函数，而不是通过subprocess运行
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from select_attributes import generate_user_profile as gen_profile
        from select_attributes import get_selected_attributes, save_results
        
        # 生成用户配置文件
        user_profile = gen_profile()
        # 获取指定数量的属性
        selected_paths = get_selected_attributes(user_profile, attribute_count)
        # 保存结果
        correct_output_dir = os.path.join(get_project_root(), "output")
        save_results(user_profile, selected_paths, correct_output_dir)
        
        # 复制文件从源位置到目标位置
        copy_files_from_source_to_target()
    except Exception as e:
        print(f"Error executing select_attributes functions: {e}")
        return {}

    # Load basic profile information and selected paths (base info is only a reference for GPT generation)
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "output")
    base_info_path = os.path.join(output_dir, 'user_profile.json')
    with open(base_info_path, 'r', encoding='utf-8') as f:
        base_info = json.load(f)
    if 'Occupations' not in base_info:
        print("Warning: 'Occupations' key is missing in the user profile. Setting it to an empty list.")
        base_info['Occupations'] = []

    selected_paths_path = os.path.join(output_dir, 'selected_paths.json')
    with open(selected_paths_path, 'r', encoding='utf-8') as f:
        selected_paths = json.load(f)

    # Ensure these fields are strings
    for k in ("life_attitude", "interests"):
        base_info[k] = safe_str(base_info.get(k, ""))

    # Example assertion: ensure the profile includes an 'Occupations' field
    assert 'Occupations' in base_info, "The 'Occupations' key is missing in the user profile."
    
    # 初始化个人资料字典
    profile = {
        "Base Info": base_info,
        "Generated At": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Profile Index": profile_index + 1
    }
    
    # 步骤1：生成 Demographic Information
    life_story = base_info.get("personal_story", {}).get("personal_story", "")
    demographic_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Instructions: Based on the `base_info` and `life_story` provided, **develop and elaborate on** the 'Demographic Information' section in English. Your task is to **appropriately expand upon and enrich** the existing information from `base_info` and incorporate relevant insights from the `life_story`. Focus on elaborating on the given data points, adding further relevant details, or providing context to make the demographic profile more comprehensive and insightful. While you should avoid simply repeating the `base_info` verbatim, ensure that all generated content is **directly built upon and logically extends** the information available in `base_info` and `life_story`, rather than introducing entirely new, unrelated demographic facts. The goal is a coherent, more descriptive, and enhanced version of the original data that reflects the person's life experiences."
    )
    demographic_template = selected_paths.get("Demographic Information")
    if demographic_template and demographic_template != "":
        print('Generating Demographic Information...')
        demographic_section = generate_category_attributes(demographic_template, demographic_input, "Demographic Information")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in demographic_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Demographic Information":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Demographic Information"] = nested_result
    else:
        print('No valid "Demographic Information" template found in selected_paths, skipping Demographic Information.')
    
    # 步骤2：生成职业信息
    career_template = selected_paths.get("Career and Work Identity")
    if career_template and career_template != "":
        print('Generating Career and Work Identity...')
        # Construct input for Career and Work Identity, including Demographic Information
        career_input = (
            "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
            "Life Story (for reference):\n" + str(life_story) + "\n\n"
            "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
            "Instructions: Based on the `base_info`, `life_story`, and `Demographic Information` provided above, **develop and elaborate on** the 'Career and Work Identity' section in English. "
            "Your aim is to distill and articulate the career identity, professional journey, and work-related aspirations that are **evident or can be reasonably inferred from the combined `base_info`, `life_story`, and `Demographic Information`**. "
            "Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the provided data** that illuminate these aspects. "
            "Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. "
            "**Do not introduce new career details or aspirations that are not grounded in or clearly supported by the source material.** "
            "The section should be an insightful and coherent expansion of what can be understood from the source material."
        )
        career_info_section = generate_category_attributes(career_template, career_input, "Career and Work Identity")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in career_info_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Career and Work Identity":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Career and Work Identity"] = nested_result
    else:
        print('No valid "Career and Work Identity" template found in selected_paths, skipping.')
    
    # 步骤3：生成 Core Values, Beliefs, and Philosophy
    pv_orientation = base_info.get("personal_values", {}).get("values_orientation", "")
    if not isinstance(pv_orientation, str):
        pv_orientation = json.dumps(pv_orientation, ensure_ascii=False)
    core_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Personal Values (for reference):\n" + pv_orientation + "\n\n"
        "Instructions: Based on the `life_story` and other information provided above, **develop and elaborate on** the 'Core Values, Beliefs, and Philosophy' section in English. Your aim is to distill and articulate the core values, beliefs, and philosophical outlook that are **evident or can be reasonably inferred from the `life_story` and other provided information**. Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the provided data** that illuminate these guiding principles. Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. **Do not introduce new values, beliefs, or philosophies that are not grounded in or clearly supported by the source material.** The section should be an insightful and coherent expansion of what can be understood from the source material.IMPORTANT: Avoid including anything related to community-building activities.Prohibit the use of words such as' balance 'and' balance '"
    )
    core_template = selected_paths.get("Core Values, Beliefs, and Philosophy")
    if core_template and core_template != "":
        print('Generating Core Values, Beliefs, and Philosophy...')
        core_values_section = generate_category_attributes(core_template, core_input, "Core Values, Beliefs, and Philosophy")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in core_values_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Core Values, Beliefs, and Philosophy":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Core Values, Beliefs, and Philosophy"] = nested_result
    else:
        print('No valid "Core Values, Beliefs, and Philosophy" template found in selected_paths, skipping.')
    
    # 步骤4：生成 Lifestyle and Daily Routine
    life_attitude = base_info["life_attitude"]
    lifestyle_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Life Attitude (for reference):\n" + life_attitude + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story`, `life_attitude`, and other information provided above, generate detailed Lifestyle and Daily Routine section in English. Use the life story to inform realistic daily routines that align with the person's experiences and background.Prohibit the use of words such as' balance 'and' balance '"
    )
    lifestyle_template = selected_paths.get("Lifestyle and Daily Routine")
    if lifestyle_template and lifestyle_template != "":
        print('Generating Lifestyle and Daily Routine...')
        lifestyle_section = generate_category_attributes(lifestyle_template, lifestyle_input, "Lifestyle and Daily Routine")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in lifestyle_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Lifestyle and Daily Routine":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Lifestyle and Daily Routine"] = nested_result
    else:
        print('No valid "Lifestyle and Daily Routine" template found in selected_paths, skipping.')
    
    # 步骤5：生成 Cultural and Social Context
    cultural_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Life Attitude (for reference):\n" + life_attitude + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Lifestyle (for reference):\n" + json.dumps(profile.get("Lifestyle and Daily Routine", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story`, `life_attitude`, and other information provided above, generate detailed Cultural and Social Context section in English. Use the life story to inform realistic cultural contexts that align with the person's experiences and background.Prohibit the use of words such as' balance 'and' balance '"
    )
    cultural_template = selected_paths.get("Cultural and Social Context")
    if cultural_template and cultural_template != "":
        print('Generating Cultural and Social Context...')
        cultural_section = generate_category_attributes(cultural_template, cultural_input, "Cultural and Social Context")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in cultural_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Cultural and Social Context":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Cultural and Social Context"] = nested_result
    else:
        print('No valid "Cultural and Social Context" template found in selected_paths, skipping.')
    
    # 步骤6：生成 Hobbies, Interests, and Lifestyle
    interests = base_info["interests"]
    hobbies_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values, Beliefs, and Philosophy (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Lifestyle and Daily Routine (for reference):\n" + json.dumps(profile.get("Lifestyle and Daily Routine", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Cultural and Social Context (for reference):\n" + json.dumps(profile.get("Cultural and Social Context", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Ensure that all hobbies, interests, and lifestyle choices presented are:1.  **Firmly anchored to and primarily derived from the hobbies indicated in `base_info` and experiences from `life_story`.**2.  Logically consistent with all provided information.3.  Enriched by supplementary information where appropriate, without overshadowing the core hobbies from `base_info`.**Do not introduce new primary hobbies or interests that are not clearly supported by or cannot be reasonably inferred from the `base_info` and `life_story` themselves.** Any lifestyle elements should logically flow from or align with these established hobbies and the overall profile.Prohibit the use of words such as' balance 'and' balance '"
    )
    hobbies_template = selected_paths.get("Hobbies, Interests, and Lifestyle")
    if hobbies_template and hobbies_template != "":
        print('Generating Hobbies, Interests, and Lifestyle...')
        hobbies_section = generate_category_attributes(hobbies_template, hobbies_input, "Hobbies, Interests, and Lifestyle")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in hobbies_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Hobbies, Interests, and Lifestyle":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Hobbies, Interests, and Lifestyle"] = nested_result
    else:
        print('No valid "Hobbies, Interests, and Lifestyle" template found in selected_paths, skipping.')
    
    # 步骤7：生成 Other Attributes
    other_attributes_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Complete Profile (for reference):\n" + json.dumps(profile, ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story` and complete profile, generate the remaining attributes for the user profile in English with refined details. Ensure that all attributes are consistent with the person's life experiences as described in the life story."
    )
    other_template = selected_paths.get("Other Attributes")
    if other_template and other_template != "":
        print('Generating Other Attributes...')
        other_attributes_section = generate_category_attributes(other_template, other_attributes_input, "Other Attributes")
        # 构建嵌套字典结构
        nested_result = {}
        for path, value in other_attributes_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Other Attributes":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Other Attributes"] = nested_result
    else:
        print('No valid "Other Attributes" template found in selected_paths, skipping.')

    # Prepare a copy of profile for summary generation by removing unwanted keys
    profile_for_summary = profile.copy()
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile_for_summary.pop(key, None)
    
    # Generate the final summary using the filtered profile and base_info
    final_summary_text = generate_final_summary(profile_for_summary, base_info)
    profile["Summary"] = final_summary_text
    
    # Remove unwanted keys from the final profile
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile.pop(key, None)

    return profile


def generate_multiple_profiles(num_rounds: int = 8) -> None:
    """生成多轮完整的用户档案，每轮包含不同数量的属性，并将它们保存到一个合并的 JSON 文件中。
    
    参数:
        num_rounds: 要生成的轮数，默认为8轮，每轮会生成8种不同属性数量的档案。
    """
    start_time = time.time()
    print(f"开始生成 {num_rounds} 轮个人资料，每轮包含8种不同数量的属性...")
    
    # 获取项目根目录
    project_root = get_project_root()
    
    # 创建输出目录
    output_dir = "/home/zhou/deeppersona/generate_user_profile_final/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义每个档案的属性数量
    attribute_counts = [100, 150, 200, 250, 300, 350]
    total_profiles = num_rounds * len(attribute_counts)
    
    # 初始化存储所有配置文件的字典
    all_profiles = {
        "metadata": {
            "profiles_completed": 0,
            "total_profiles": total_profiles,
            "total_rounds": num_rounds,
            "description": "包含多轮不同属性数量的用户档案集合"
        }
    }
    
    # 设置合并文件路径（不使用时间戳）
    base_all_profiles_path = os.path.join(output_dir, f"profile_ind.json")
    all_profiles_path = base_all_profiles_path
    
    # 初始化保存合并文件
    actual_path = save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
    print(f"初始化合并文件: {actual_path}")
    all_profiles_path = actual_path  # 使用实际保存的路径
    
    # 计数器，用于跟踪总共生成的档案数量
    profile_count = 0
    
    # 逐轮生成配置文件
    for round_num in range(num_rounds):
        print(f"\n===== 开始生成第 {round_num+1}/{num_rounds} 轮用户资料 =====\n")
        
        # 在每轮中生成所有不同属性数量的档案
        for attr_index, current_attribute_count in enumerate(attribute_counts):
            profile_count += 1
            
            print(f"\n----- 开始生成第 {round_num+1}.{attr_index+1} 个用户资料 (属性数量: {current_attribute_count}) -----\n")
            
            try:
                # 生成单个配置文件，传入属性数量
                profile = generate_single_profile(None, profile_count-1, current_attribute_count)
                
                if not profile:
                    print(f"第 {round_num+1}.{attr_index+1} 个资料生成失败，跳过")
                    continue
                
                # 添加到总字典并保存
                profile_key = f"Profile_R{round_num+1}_A{attr_index+1}_Count_{current_attribute_count}"
                all_profiles[profile_key] = profile
                all_profiles["metadata"]["profiles_completed"] = profile_count
                
                # 保存更新后的合并文件
                save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
                print(f"\n总进度更新: {profile_count}/{total_profiles} 个资料已完成 (第 {round_num+1}/{num_rounds} 轮)")
                print(f"已将第 {round_num+1}.{attr_index+1} 个用户资料 (属性数量: {current_attribute_count}) 添加到合并文件: {all_profiles_path}")
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"生成第 {round_num+1}.{attr_index+1} 个个人资料时出错: {e}")
                continue
        
        print(f"\n===== 第 {round_num+1}/{num_rounds} 轮用户资料生成完成 =====\n")
        print("\n" + "="*50 + "\n")
    
    # 添加生成完成状态
    all_profiles["metadata"]["status"] = "completed"
    save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n所有 {all_profiles['metadata']['profiles_completed']} 个个人资料已成功生成并保存到: {all_profiles_path}")
    print(f"生成完成，耗时 {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    generate_multiple_profiles(10)