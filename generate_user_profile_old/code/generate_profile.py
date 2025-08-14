#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import random
import sys
import time
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
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_root


def save_json_file(file_path: str, data: Dict) -> None:
    """保存JSON文件
    
    Args:
        file_path: 目标文件路径
        data: 要保存的数据
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")


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


def generate_attribute_value(path: str, base_summary: str) -> str:
    """为给定的属性路径生成值。
    
    参数:
        path: 属性路径。
        base_summary: 基础摘要文本。
        
    返回:
        str: 生成的属性值。
    """
    system_prompt = """You are an AI assistant specialized in generating attribute values for personal profiles. Based on the provided base summary and attribute path, generate a logically consistent value that:
    1.Be factually consistent with the information present in the base summary.
    2.Maintain strict logical consistency with all previously generated attribute values for this specific profile, ensuring no contradictions or logical flaws are introduced.
    3.Be semantically relevant to the specified attribute path.
    4.Be plausible, realistic, and contain appropriate detail while upholding overall credibility of the profile.
    IMPORTANT: Avoid including anything related to community-building activities.
The output should be a concise value, not exceeding 50 characters."""
    user_prompt = f"Base Summary:\n{base_summary}\n\nAttribute Path: {path}\n\nPlease generate a suitable value for this attribute that is consistent with the base summary and fits the semantic context of the attribute path."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = get_completion(messages)
        return response.strip() if response else ""
    except Exception as e:
        print(f"生成属性值时出错 ({path}): {e}")
        return ""


def generate_category_attributes(category_paths: Dict, base_summary: str, category_name: str) -> Dict:
    """一次性生成一个一级大类下的所有属性值。
    
    参数:
        category_paths: 一级大类下的所有属性路径及其结构。
        base_summary: 基础摘要文本。
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
    
    collect_leaf_paths(category_paths, category_name)
    
    # 如果没有叶子节点，直接返回空字典
    if not leaf_paths:
        return {}
    
    # 构建提示，一次性生成所有属性值
    system_prompt = """You are an AI assistant specialized in generating attribute values for personal profiles. Based on the provided base summary and multiple attribute paths, generate logically consistent values for each attribute path that:
    1. Are factually consistent with the information present in the base summary.
    2. Maintain strict logical consistency with each other, ensuring no contradictions or logical flaws are introduced.
    3. Are semantically relevant to their respective attribute paths.
    4. Are plausible, realistic, and contain appropriate detail while upholding overall credibility of the profile.
    IMPORTANT: Avoid including anything related to community-building activities.
    
    Format your response as a JSON object where each key is the attribute path and each value is the generated attribute value (not exceeding 50 characters).
    Example format:
    {
        "Category.Subcategory.AttributePath1": "Generated value 1",
        "Category.Subcategory.AttributePath2": "Generated value 2"
    }
    """
    
    user_prompt = f"Base Summary:\n{base_summary}\n\nCategory: {category_name}\n\nAttribute Paths to generate values for:\n"
    for path in leaf_paths:
        user_prompt += f"- {path}\n"
    user_prompt += "\nPlease generate suitable values for all these attributes that are consistent with the base summary and fit the semantic context of each attribute path."
    
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
            generated_values = json.loads(response)
            print(f"  成功生成 {len(generated_values)} 个属性值")
            return generated_values
        except json.JSONDecodeError as e:
            print(f"  解析 {category_name} 属性值JSON失败: {e}")
            print(f"  响应内容: {response[:100]}..." if len(response) > 100 else f"响应内容: {response}")
            return {}
    except Exception as e:
        print(f"  生成 {category_name} 属性值时出错: {e}")
        return {}


def generate_final_summary(profile: Dict) -> str:
    """为用户档案生成最终摘要。
    
    参数:
        profile: 完整的用户档案数据。
    返回:
        str: 最终的摘要文本。
    """
    system_prompt = """
Your Task: Create an objective and factual personal profile, strictly 150-400 words, based exclusively on the provided text.

Content Requirements:

Adopt a first-person perspective to present a coherent portrayal of my character and background by accurately integrating all stated attributes.

The output should be a seamless narrative written from my point of view. While avoiding a purely mechanical, sentence-by-sentence list, ensure any connections drawn between pieces of information are based only on explicit facts from the profile, not on your interpretations or assumptions.

!! CRUCIAL INSTRUCTIONS - ADHERE STRICTLY !!
- The entire output must be in the first person ("I", "my", "me").


Strict Adherence to Source (No Speculation):

Include ONLY information explicitly stated in the provided profile.
DO NOT invent, infer beyond direct statements, or add any embellishments or speculative details. All content must be directly and unequivocally supported by the source text.
Objective Statement (NO INTERPRETATION):

Describe attributes and experiences directly and factually.
YOU MUST STRICTLY AVOID all interpretive or evaluative language. For example, DO NOT USE phrases like: "this demonstrates...", "which reflects...", "this embodies...", "this highlights...", "this suggests...", or any similar phrases that explain significance, draw conclusions, or offer analysis not explicitly present in the source.    
    
"""
    user_prompt = f"Complete Profile (in JSON format):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\nPlease generate an objective and factual summary that covers all core information from the profile in clear, coherent paragraphs. The summary should be between 150-400 words."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = get_completion(messages)
        summary = response.strip() if response else ""
        # Enforce the word limit between 300-400 words
        word_count = len(summary.split())
        if word_count < 150:
            print(f"Warning: Summary is only {word_count} words, less than the target minimum of 300 words")
        elif word_count > 400:
            summary = enforce_word_limit(summary, 400)
            print(f"Summary was trimmed to 400 words (from {word_count})")
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


def append_profile_to_json(file_path: str, profile: Dict) -> None:
    """追加个人资料到 JSON 文件
    
    参数:
        file_path: 目标文件路径
        profile: 要追加的个人资料
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
        else:
            profiles = []
        
        profiles.append(profile)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"追加个人资料到 JSON 文件时出错: {e}")


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
        save_results(user_profile, selected_paths)
    except Exception as e:
        print(f"Error executing select_attributes functions: {e}")
        return {}

    # Load basic profile information and selected paths (base info is only a reference for GPT generation)
    base_info_path = os.path.join(os.path.dirname(__file__), 'output', 'user_profile.json')
    with open(base_info_path, 'r', encoding='utf-8') as f:
        base_info = json.load(f)
    if 'Occupations' not in base_info:
        print("Warning: 'Occupations' key is missing in the user profile. Setting it to an empty list.")
        base_info['Occupations'] = []

    selected_paths_path = os.path.join(os.path.dirname(__file__), 'output', 'selected_paths.json')
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
    demographic_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `base_info` provided, **develop and elaborate on** the 'Demographic Information' section in English. Your task is to **appropriately expand upon and enrich** the existing information from `base_info`. Focus on elaborating on the given data points, adding further relevant details, or providing context to make the demographic profile more comprehensive and insightful. While you should avoid simply repeating the `base_info` verbatim, ensure that all generated content is **directly built upon and logically extends** the information available in `base_info`, rather than introducing entirely new, unrelated demographic facts. The goal is a coherent, more descriptive, and enhanced version of the original data."
    )
    demographic_template = selected_paths.get("Demographic Information")
    if isinstance(demographic_template, dict):
        demographic_template = json.dumps(demographic_template, ensure_ascii=False)
    if demographic_template and demographic_template != "":
        print('Generating Demographic Information...')
        demographic_section = generate_section(json.loads(demographic_template), demographic_input, "Demographic Information")
        profile["Demographic Information"] = demographic_section
    else:
        print('No valid "Demographic Information" template found in selected_paths, skipping Demographic Information.')
        
    # 步骤2：生成职业信息
    career_template = selected_paths.get("Career and Work Identity")
    if isinstance(career_template, dict):
        career_template = json.dumps(career_template, ensure_ascii=False)
    if career_template and career_template != "":
        print('Generating Career and Work Identity...')
        # Construct input for Career and Work Identity, including Demographic Information
        career_input = (
            "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
            "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
            "Instructions: Based on the `base_info` and `Demographic Information` provided above, **develop and elaborate on** the 'Career and Work Identity' section in English. "
            "Your aim is to distill and articulate the career identity, professional journey, and work-related aspirations that are **evident or can be reasonably inferred from the combined `base_info` and `Demographic Information`**. "
            "Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the provided data** that illuminate these aspects. "
            "Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. "
            "**Do not introduce new career details or aspirations that are not grounded in or clearly supported by the source material.** "
            "The section should be an insightful and coherent expansion of what can be understood from the source material."
        )
        career_info_section = generate_section(json.loads(career_template), career_input, "Career and Work Identity")
        profile["Career and Work Identity"] = career_info_section
    else:
        print('No valid "Career and Work Identity" template found in selected_paths, skipping.')
        # Optionally, to stop overall generation if career is mandatory, you could return profile here; else just continue.
        # return profile
    
    # 步骤3：生成 Core Values, Beliefs, and Philosophy
    pv_orientation = base_info.get("personal_values", {}).get("values_orientation", "")
    if not isinstance(pv_orientation, str):
        pv_orientation = json.dumps(pv_orientation, ensure_ascii=False)
    core_input = (
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Personal Values (for reference):\n" + pv_orientation + "\n\n"
        "Instructions: Based on the `base_info` provided above, **develop and elaborate on** the 'Core Values, Beliefs, and Philosophy' section in English. Your aim is to distill and articulate the core values, beliefs, and philosophical outlook that are **evident or can be reasonably inferred from the `base_info`**. Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the `base_info`** that illuminate these guiding principles. Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. **Do not introduce new values, beliefs, or philosophies that are not grounded in or clearly supported by the `base_info`.** The section should be an insightful and coherent expansion of what can be understood from the source material.IMPORTANT: Avoid including anything related to community-building activities."
    )
    core_template = selected_paths.get("Core Values, Beliefs, and Philosophy")
    if isinstance(core_template, dict):
        core_template = json.dumps(core_template, ensure_ascii=False)
    if core_template and core_template != "":
        print('Generating Core Values, Beliefs, and Philosophy...')
        core_values_section = generate_section(json.loads(core_template), core_input, "Core Values, Beliefs, and Philosophy")
        profile["Core Values, Beliefs, and Philosophy"] = core_values_section
    else:
        print('No valid "Core Values, Beliefs, and Philosophy" template found in selected_paths, skipping.')
    
    # 步骤4：生成 Lifestyle and Daily Routine 及 Cultural and Social Context
    life_attitude = base_info["life_attitude"]
    lifestyle_input = (
        "Life Attitude (for reference):\n" + life_attitude + "\n\n"
        "Instructions: Based on the above, generate detailed Lifestyle and Daily Routine and Cultural and Social Context sections in English."
    )
    lifestyle_template = selected_paths.get("Lifestyle and Daily Routine")
    if isinstance(lifestyle_template, dict):
        lifestyle_template = json.dumps(lifestyle_template, ensure_ascii=False)
    if lifestyle_template and lifestyle_template != "":
        print('Generating Lifestyle and Daily Routine...')
        lifestyle_section = generate_section(json.loads(lifestyle_template), lifestyle_input, "Lifestyle and Daily Routine")
        profile["Lifestyle and Daily Routine"] = lifestyle_section
    else:
        print('No valid "Lifestyle and Daily Routine" template found in selected_paths, skipping.')
    cultural_template = selected_paths.get("Cultural and Social Context")
    if isinstance(cultural_template, dict):
        cultural_template = json.dumps(cultural_template, ensure_ascii=False)
    if cultural_template and cultural_template != "":
        print('Generating Cultural and Social Context...')
        cultural_section = generate_section(json.loads(cultural_template), lifestyle_input, "Cultural and Social Context")
        profile["Cultural and Social Context"] = cultural_section
    else:
        print('No valid "Cultural and Social Context" template found in selected_paths, skipping.')
    
    # 步骤5：生成 Hobbies, Interests, and Lifestyle
    interests = base_info["interests"]
    hobbies_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values, Beliefs, and Philosophy (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Lifestyle and Daily Routine (for reference):\n" + json.dumps(profile.get("Lifestyle and Daily Routine", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Cultural and Social Context (for reference):\n" + json.dumps(profile.get("Cultural and Social Context", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Ensure that all hobbies, interests, and lifestyle choices presented are:1.  **Firmly anchored to and primarily derived from the hobbies indicated in `base_info`.**2.  Logically consistent with all provided information.3.  Enriched by supplementary information where appropriate, without overshadowing the core hobbies from `base_info`.**Do not introduce new primary hobbies or interests that are not clearly supported by or cannot be reasonably inferred from the `base_info` itself.** Any lifestyle elements should logically flow from or align with these established hobbies and the overall profile."
    )
    hobbies_template = selected_paths.get("Hobbies, Interests, and Lifestyle")
    if isinstance(hobbies_template, dict):
        hobbies_template = json.dumps(hobbies_template, ensure_ascii=False)
    if hobbies_template and hobbies_template != "":
        print('Generating Hobbies, Interests, and Lifestyle...')
        hobbies_section = generate_section(json.loads(hobbies_template), hobbies_input, "Hobbies, Interests, and Lifestyle")
        profile["Hobbies, Interests, and Lifestyle"] = hobbies_section
    else:
        print('No valid "Hobbies, Interests, and Lifestyle" template found in selected_paths, skipping.')
    
    # 步骤6：生成 Other Attributes
    other_attributes_input = (
        "Complete Profile (for reference):\n" + json.dumps(profile, ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the complete profile, generate the remaining attributes for the user profile in English with refined details."
    )
    other_template = selected_paths.get("Other Attributes")
    if isinstance(other_template, dict):
        other_template = json.dumps(other_template, ensure_ascii=False)
    if other_template and other_template != "":
        print('Generating Other Attributes...')
        other_attributes_section = generate_section(json.loads(other_template), other_attributes_input, "Other Attributes")
        profile["Other Attributes"] = other_attributes_section
    else:
        print('No valid "Other Attributes" template found in selected_paths, skipping.')

    # Prepare a copy of profile for summary generation by removing unwanted keys
    profile_for_summary = profile.copy()
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile_for_summary.pop(key, None)
    
    # Generate the final summary using the filtered profile
    final_summary_text = generate_final_summary(profile_for_summary)
    profile["Summary"] = final_summary_text
    
    # Remove unwanted keys from the final profile
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile.pop(key, None)

    # 将生成的profile追加到一个新的JSON文件中，支持存储多个user profile
    new_profiles_file = os.path.join(os.path.dirname(__file__), 'output', 'new_user_profiles.json')
    append_profile_to_json(new_profiles_file, profile)

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
    output_dir = os.path.join(project_root, "generate_user_profile", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义每个档案的属性数量
    attribute_counts = [50, 100, 150, 200, 250, 300, 350, 400]
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
    
    # 设置合并文件路径
    all_profiles_path = os.path.join(output_dir, f"all_profiles_varied_attributes.json")
    
    # 初始化保存合并文件
    save_json_file(all_profiles_path, all_profiles)
    print(f"初始化合并文件: {all_profiles_path}")
    
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
                save_json_file(all_profiles_path, all_profiles)
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
    save_json_file(all_profiles_path, all_profiles)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n所有 {all_profiles['metadata']['profiles_completed']} 个个人资料已成功生成并保存到: {all_profiles_path}")
    print(f"生成完成，耗时 {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    generate_multiple_profiles(10)