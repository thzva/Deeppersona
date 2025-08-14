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


# 已删除不再需要的generate_attribute_value函数


def generate_category_attributes(category_paths: Dict, base_summary: str, category_name: str) -> Dict:
    """按类别批量生成属性值。
    
    参数:
        category_paths: 类别下的所有属性路径及其结构。
        base_summary: 基础摘要文本。
        category_name: 类别名称。
        
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
    
    # 如果叶子节点过多，分批处理
    MAX_PATHS_PER_BATCH = 50  # 每批处理的最大路径数
    all_generated_values = {}
    
    # 将叶子节点分批
    batches = [leaf_paths[i:i + MAX_PATHS_PER_BATCH] for i in range(0, len(leaf_paths), MAX_PATHS_PER_BATCH)]
    
    print(f"  正在生成 {category_name} 下的 {len(leaf_paths)} 个属性值，分 {len(batches)} 批处理...")
    
    for batch_index, batch_paths in enumerate(batches):
        user_prompt = f"Base Summary:\n{base_summary}\n\nCategory: {category_name}\n\nAttribute Paths to generate values for (Batch {batch_index+1}/{len(batches)}):\n"
        for path in batch_paths:
            user_prompt += f"- {path}\n"
        user_prompt += "\nPlease generate suitable values for all these attributes that are consistent with the base summary and fit the semantic context of each attribute path."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print(f"  正在生成第 {batch_index+1}/{len(batches)} 批，包含 {len(batch_paths)} 个属性...")
            response = get_completion(messages)
            if not response:
                print(f"  生成第 {batch_index+1} 批属性值失败: 空响应")
                continue
                
            # 尝试解析JSON响应
            try:
                import json
                import re
                
                # 检查响应是否包含在Markdown代码块中
                json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response)
                if json_match:
                    # 提取代码块中的JSON内容
                    json_content = json_match.group(1).strip()
                    batch_values = json.loads(json_content)
                else:
                    # 尝试直接解析整个响应
                    batch_values = json.loads(response)
                    
                print(f"  成功生成第 {batch_index+1} 批的 {len(batch_values)} 个属性值")
                all_generated_values.update(batch_values)
            except json.JSONDecodeError as e:
                print(f"  解析第 {batch_index+1} 批属性值JSON失败: {e}")
                print(f"  响应内容: {response[:100]}..." if len(response) > 100 else f"响应内容: {response}")
        except Exception as e:
            print(f"  生成第 {batch_index+1} 批属性值时出错: {e}")
    
    print(f"  所有批次处理完成，共生成 {len(all_generated_values)} 个属性值")
    return all_generated_values


def generate_final_summary(profile: Dict) -> str:
    """为用户档案生成最终摘要。
    
    参数:
        profile: 完整的用户档案数据。
    返回:
        str: 最终的摘要文本。
    """
    system_prompt = """
Your Task: Create a believable and engaging personal profile, 150-400 words, based on the provided text. Your ultimate goal is a narrative that feels like it comes from a real, self-aware person.

Content Requirements:

Adopt a first-person perspective ("I," "my," "me"). The output should be a seamless narrative, not just a list of facts.

!! GUIDELINES FOR A HUMAN-LIKE AND COHERENT NARRATIVE !!

1.  **Ensure an Authentic Voice:**
    Your narrative should generally reflect the tone and sophistication suggested by the profile's data. 
    However, if the language in the source text seems inconsistent with the character's stated age or background, your primary duty is to adopt a voice that is truly authentic to the character's context. You must **translate** the *meaning* of the data into a believable voice, rather than copying the source text's style verbatim.

2.  **Build a Plausible Narrative from the Facts:**
    Your narrative must be grounded in the source text. Do not invent new core facts. You can and should add minor, plausible details that bridge gaps between facts to make the narrative more vivid and realistic.
    Additionally, if a core fact in the source data appears highly implausible or contradictory in a real-world context, you should creatively reframe it to make the story believable. Instead of stating the implausible fact directly, you can frame it as part of a unique circumstance, a powerful memory, a personal aspiration, or a fantasy. Your goal is to produce a believable story, even if it requires creatively interpreting the source data to resolve inconsistencies.
    
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
    
    # 无论是一级大类还是子类，都一次性生成所有属性
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
    
    # 如果一次性生成失败，返回空结果
    print(f"{indent_str}一次性生成失败，返回空结果")
    # 不再使用递归方式生成属性值
    
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


def generate_single_profile(template=None, profile_index=None) -> Dict:
    """生成单个用户档案。
    
    参数:
        template: 可选的用于生成的模板。
        profile_index: 要生成的档案索引。
        
    返回:
        Dict: 生成的用户档案。
    """

    # 第一步：运行 select_attributes.py 选择属性
    print(f'\n第一步：运行 select_attributes.py 选择属性...')
    try:
        # 使用subprocess运行select_attributes.py
        subprocess_result = subprocess.run(
            ["python", os.path.join(os.path.dirname(__file__), "select_attributes.py")],
            capture_output=False,  # 直接输出到终端，而不是捕获
            text=True,
            check=True
        )
        print("\nselect_attributes.py 运行完成，属性选择完毕")
        print("\n第二步：按一级属性批量生成内容...")
    except Exception as e:
        print(f"Error executing select_attributes functions: {e}")
        return {}

    # Load basic profile information and selected paths (base info is only a reference for GPT generation)
    base_info_path = '/home/zhou/persona/generate_user_profile/output/user_profile.json'
    with open(base_info_path, 'r', encoding='utf-8') as f:
        base_info = json.load(f)
    if 'Occupations' not in base_info:
        print("Warning: 'Occupations' key is missing in the user profile. Setting it to an empty list.")
        base_info['Occupations'] = []

    selected_paths_path = '/home/zhou/persona/generate_user_profile/output/selected_paths.json'
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
        "Instructions: Based on the full context provided (including base_info, Demographic Information, and Career Information), develop and elaborate on the 'Core Values, Beliefs, and Philosophy' section in English. Your aim is to distill and articulate the core values, beliefs, and philosophical outlook that are evident or can be reasonably inferred from the provided information. Offer fresh insights by providing a deeper, more nuanced interpretation or by highlighting connections between all data points that illuminate these guiding principles. Pay special attention to the person's location and background, infusing the philosophical outlook with relevant cultural nuances from their region. Ensure that this elaboration is logically consistent with and directly stems from the provided information. Do not introduce new values, beliefs, or philosophies that are not grounded in or clearly supported by the source material. The section should be an insightful and coherent expansion of what can be understood from the source material.IMPORTANT: Avoid including anything related to community-building activities."
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
    
    # 将各部分转换为字符串以便在提示中使用
    base_info_str = json.dumps(base_info, ensure_ascii=False, indent=2)
    demographic_info_str = json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2)
    career_info_str = json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2)
    core_values_str = json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2)
    
    lifestyle_input = (
    "## Full Profile Context (for reference)\n\n"
    f"### Base Information:\n{base_info_str}\n\n"
    f"### Demographic Information:\n{demographic_info_str}\n\n"
    f"### Career and Work Identity:\n{career_info_str}\n\n"
    f"### Core Values, Beliefs, and Philosophy:\n{core_values_str}\n\n"
    
    "## Instructions:\n"
    "Based on the **complete profile context provided above**, your task is to generate content for the section you are asked to create. Follow the specific guidance below for the relevant section.\n\n"
    
    "**Guidance for 'Lifestyle and Daily Routine':**\n"
    "When generating this section, focus on translating the person's values, career, age, and philosophy into **concrete daily actions and habits**. How do their beliefs manifest in their **work schedule, sleep patterns, physical activity, vacation style, and travel preferences**? Ensure the lifestyle is a logical and practical extension of their established character.\n\n"
    
    "**Guidance for 'Cultural and Social Context':**\n"
    "When generating this section, focus on the person's **relationship with society and their environment**. Based on their location, values, and life story, describe their **living environment, preferred social circle, communication style, parenting style (if applicable), and relationship with tradition**. How do their internal beliefs shape their external social world?"
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
        
        "## Instructions:\n"
        "Based on the complete profile context provided above, generate the 'Hobbies, Interests, and Lifestyle' section.\n\n"
        
        "1.  **Use Base Hobbies as a Starting Point:** Begin with the interests listed in `base_info`, but treat them as seeds for deeper exploration, not as rigid boundaries.\n\n"
        
        "2.  **Embrace Imagination and Psychological Depth:** This is the key instruction. You are encouraged to use **creative imagination**. Based on the person's complete profile, the interpretation of their interests can be **positive, deeply negative, or even 'toxic'**. Explore the hidden psychological dimensions of their hobbies. How could a seemingly simple interest become a mechanism for **coping, obsession, self-destruction, control, or escapism**? Be bold in your interpretation.\n\n"
        
        "3.  **Synthesize with Full Context:** Ensure your imaginative elaborations are still **psychologically consistent** with the person's established character. The 'how' and 'why' of their interests should be deeply connected to their career, demographics, and worldview.\n\n"
        
        "4.  **Detail Related Lifestyle Choices:** Describe the lifestyle choices that support these interests. This includes the types of products they might buy, the media they consume, or other related activities, all of which should reflect the **positive or negative nature** of their engagement with the hobby."
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
    "## Instructions:\n"
    "Based on the complete profile provided, your task is to synthesize and generate a set of 'Other Attributes' in English that add final depth and nuance to the character. Instead of random facts, derive these attributes by analyzing the entire profile as a whole. Specifically, generate details for the following categories:\n\n"
    
    "1.  **Communication Style:**\n"
    "Describe their typical manner of speaking and writing. Is it direct, quiet, formal, verbose, witty, or something else? How do their core values and social context shape how they communicate with others?\n\n"
    
    "2.  **Decision-Making Style:**\n"
    "Analyze how they generally make choices. Are they impulsive, analytical, cautious, risk-averse, or guided by emotion? Connect this style to their career, financial status, and key life events.\n\n"
    
    "3.  **Defining Quirks or Habits:**\n"
    "Imagine and describe one or two small, defining habits or personal quirks that make the character feel real and unique. This could be a daily ritual, a nervous gesture, or an unusual habit related to their hobbies. Ensure it is psychologically consistent with their established personality.\n\n"
    
    "4.  **Core Internal Conflict:**\n"
    "Identify and articulate the central psychological tension or conflict that defines the character's inner world. What two opposing forces, desires, or beliefs are constantly at odds within them? This should serve as a concise summary of their core psychological drama."
    
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


def generate_multiple_profiles(num_profiles: int = 50) -> None:
    """生成多个完整的用户档案，并将它们保存到一个合并的 JSON 文件中。
    
    参数:
        num_profiles: 要生成的档案数量，默认为50个。
    """
    start_time = time.time()
    print(f"开始生成 {num_profiles} 个个人资料...")
    
    # 获取项目根目录
    project_root = get_project_root()
    
    # 创建输出目录
    output_dir = os.path.join(project_root, "generate_user_profile", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 总配置文件数量
    total_profiles = num_profiles
    
    # 初始化存储所有配置文件的字典
    all_profiles = {
        "metadata": {
            "profiles_completed": 0,
            "total_profiles": total_profiles,
            "description": f"包含{total_profiles}个用户档案的集合"
        }
    }
    
    # 设置合并文件路径
    all_profiles_path = os.path.join(output_dir, f"all_profile_{num_profiles}.json")
    
    # 初始化保存合并文件
    save_json_file(all_profiles_path, all_profiles)
    print(f"初始化合并文件: {all_profiles_path}")
    
    # 计数器，用于跟踪总共生成的档案数量
    profile_count = 0
    
    # 逐个生成配置文件
    for profile_index in range(num_profiles):
        profile_count += 1
        
        print(f"\n----- 开始生成第 {profile_index+1}/{num_profiles} 个用户资料 -----\n")
        
        try:
            # 生成单个配置文件
            profile = generate_single_profile(None, profile_index)
            
            if not profile:
                print(f"第 {profile_index+1} 个资料生成失败，跳过")
                continue
            
            # 添加到总字典并保存
            profile_key = f"Profile_{profile_index+1}"
            all_profiles[profile_key] = profile
            all_profiles["metadata"]["profiles_completed"] = profile_count
            
            # 保存更新后的合并文件
            save_json_file(all_profiles_path, all_profiles)
            print(f"\n总进度更新: {profile_count}/{total_profiles} 个资料已完成")
            print(f"已将第 {profile_index+1} 个用户资料添加到合并文件: {all_profiles_path}")
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"生成第 {profile_index+1} 个个人资料时出错: {e}")
            continue
    
    # 添加生成完成状态
    all_profiles["metadata"]["status"] = "completed"
    save_json_file(all_profiles_path, all_profiles)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n所有 {all_profiles['metadata']['profiles_completed']} 个个人资料已成功生成并保存到: {all_profiles_path}")
    print(f"生成完成，耗时 {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    generate_multiple_profiles(40)