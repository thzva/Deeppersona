"""数据源模块，负责从各种数据源获取用户配置信息。

此模块包含以下主要功能：
- 从本地文件获取职业数据
- 生成真实的地理位置信息
- 生成年龄信息
- 生成性别信息
"""

import json
import os
import random
from typing import Dict, List, Optional, Union, Any
from geonamescache import GeonamesCache
from config import get_completion, parse_gpt_response, parse_json_response, extract_json_from_markdown, parse_nested_json_response

_occupations_cache = None

def get_occupations() -> List[str]:
    """从本地文件获取职业数据。

    从预定义的JSON文件中读取职业列表。使用缓存机制避免重复读取文件。
    如果文件读取失败，将返回空列表。

    Returns:
        List[str]: 职业列表。如果获取失败则返回空列表
    """
    global _occupations_cache

    if _occupations_cache is not None:
        return _occupations_cache

    try:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'occupations_english.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            _occupations_cache = json.load(f)
        return _occupations_cache
    except Exception as e:
        print(f"Error loading occupations data: {e}")
        return []

def generate_age_info() -> Dict[str, Union[int, str]]:
    """生成年龄信息。

    首先在3-95岁范围内随机生成一个年龄，然后根据年龄确定对应的年龄组。
    年龄组包括：幼儿、儿童、青少年、青年、成年、中年、老年。

    Returns:
        Dict[str, Union[int, str]]: 包含以下字段的字典：
            - age: 具体年龄（整数）
            - age_group: 年龄组类别
    """
    # 首先随机生成年龄
    age = random.randint(7, 85)

    # 根据年龄确定年龄组
    if age <= 6:
        age_group = "toddler"  # 幼儿
    elif age <= 12:
        age_group = "child"    # 儿童
    elif age <= 19:
        age_group = "adolescent"  # 青少年
    elif age <= 29:
        age_group = "young_adult"  # 青年
    elif age <= 45:
        age_group = "adult"  # 成年
    elif age <= 65:
        age_group = "middle_aged"  # 中年
    else:
        age_group = "senior"  # 老年

    return {
        "age": age,
        "age_group": age_group
    }

def generate_career_info(age: int) -> Dict[str, str]:
    """生成职业相关信息。

    根据年龄决定职业生成方式：
    - 对18岁以下和65岁以上的人，职业状态将由GPT根据年龄生成
    - 其他年龄段的人，从本地职业数据库中随机选择职业

    Args:
        age: 年龄

    Returns:
        Dict[str, str]: 职业相关信息，包含职业状态
    """
    if age < 18 or age > 65:
        # 对18岁以下和65岁以上的人，职业状态由GPT生成
        prompt = f"Generate an appropriate occupation or status for a {age} year old person. "
        if age < 18:
            prompt += "Consider that the individuals are likely in school or engaged in youth activities. They may not have any formal occupation. If appropriate, you can mention their student status or indicate they have no occupation yet. Only in some cases, consider potential interest in early employment opportunities, internships, or non-traditional educational paths."
        else:  # age > 65
            prompt += "Consider they might be retired but could still be active in various ways."

        messages = [
            {"role": "system", "content": "You are an AI that generates realistic occupation statuses based on age. Respond with just the status, no explanation."},
            {"role": "user", "content": prompt}
        ]

        status = get_completion(messages)
        if not status:
            return {"status": ""}
        return {"status": status}

    # 其他年龄段从职业数据库选择
    occupations = get_occupations()
    if not occupations:
        return {"status": ""}

    career_status = random.choice(occupations)
    return {"status": career_status}

def generate_location() -> Dict[str, str]:
    """生成真实的地理位置信息。

    使用 GeoNames 数据库随机选择一个国家和城市。

    Returns:
        Dict[str, str]: 包含以下字段的字典：
            - country: 国家名称
            - city: 城市名称
    """
    gc = GeonamesCache()

    # 获取所有国家
    countries = gc.get_countries()
    country_code = random.choice(list(countries.keys()))
    country = countries[country_code]

    # 获取选国家的所有城市
    cities = gc.get_cities()
    country_cities = [city for city in cities.values() if city['countrycode'] == country_code]

    if not country_cities:
        return {
            "country": country['name'],
            "city": "Unknown City"
        }

    # 随机选择一个城市
    city_data = random.choice(country_cities)

    return {
        "country": country['name'],
        "city": city_data['name']
    }


# def generate_location() -> Dict[str, str]:
#     """
#     生成加拿大随机城市的地理位置信息。

#     Returns
#     -------
#     Dict[str, str]
#         - country : 始终为 "Canada"
#         - city    : 随机城市名
#     """
#     gc = GeonamesCache()

#     # ---- 固定国家为加拿大 ----
#     country_code = "CA"
#     countries = gc.get_countries()
#     country     = countries[country_code]

#     # ---- 过滤出加拿大的所有城市 ----
#     cities         = gc.get_cities()
#     ca_cities_list = [
#         city for city in cities.values()
#         if city["countrycode"] == country_code
#     ]

#     # ---- 若意外无数据，兼底处理 ----
#     if not ca_cities_list:
#         return {"country": country["name"], "city": "Toronto"}

#     # ---- 随机选择一个城市 ----
#     city_data = random.choice(ca_cities_list)
#     return {"country": country["name"], "city": city_data["name"]}


def generate_gender() -> str:
    """随机生成性别（男/女）"""
    return random.choice(['male', 'female'])


def generate_personal_values(age: int, gender: str, occupation: str, location: Dict[str, str]) -> Dict[str, str]:
    """使用GPT根据人口统计信息生成个人价值观。

    首先从 positive, negative, neutral 三种类型中随机选择一种价值观类型，
    然后将这个类型放入prompt中，让GPT生成具体的价值观细节。

    Args:
        age: 人物年龄
        gender: 人物性别
        occupation: 人物职业
        location: 包含'city'和'country'键的字典

    Returns:
        包含'values_orientation'键的字典，其中包含价值观描述
    """
    # 随机选择价值观类型
    value_type = random.choice(['positive', 'negative', 'neutral'])
    
    prompt = f"""
    Generate a concise description of a person's core values and belief system based on:
    Age: {age}, Gender: {gender}, Occupation: {occupation}, Location: {location['city']}, {location['country']}

    IMPORTANT: This person has a {value_type.upper()} value system. Their values may be entirely consistent with their personal background or may conflict with it. Avoid introducing unnecessary contrasts or contradictions in their beliefs. Try to avoid being related to the community as much as possible.

    Please generate a short phrase that clearly captures the essence of this person's core values and beliefs without adding conflicting ideas or turnarounds.

    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{
        "values_orientation": "short phrase describing their values"
    }}

    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that generates realistic human value systems in ONE SENTENCE, including both positive and negative values. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        # 使用config.py中的解析函数
        result = parse_gpt_response(
            response, 
            expected_fields=["values_orientation"], 
            field_defaults={"values_orientation": ""}
        )
        
        return {
            "values_orientation": result.get("values_orientation") or response.strip()
        }
    except Exception as e:
        print(f"\nError in generate_personal_values: {e}")
        raise




def generate_life_attitude(age: int = None, gender: str = None, occupation: str = None, 
                        location: Dict[str, str] = None, values_orientation: str = None) -> Dict[str, Union[str, Dict, bool]]:
    """使用GPT根据人口统计信息和个人价值观生成生活态度。

    此函数使用GPT根据年龄、性别、职业、地区和个人价值观生成一个人的生活态度。
    生成的态度可能是积极的、中性的或消极的。

    Args:
        age: 人物年龄
        gender: 人物性别
        occupation: 人物职业
        location: 包含'city'和'country'键的字典
        values_orientation: 个人价值观描述

    Returns:
        Dict: 包含以下字段的字典：
            - attitude: 表示用户生活态度的字符串
            - attitude_details: 关于这种态度如何表现的详细信息
            - coping_mechanism: 人物如何应对生活挑战
    """
    
    # 创建提示词
    prompt = f"""
    Generate specific attributes about a person's life attitude based on the following information:
    
    Age: {age}
    Gender: {gender}
    Occupation: {occupation}
    Location: {location['city']}, {location['country']}
    Core Values: {values_orientation}
    
    IMPORTANT: This person's attitude toward life can be positive, neutral, or negative. In a negative state, they may hold a pessimistic, cynical, or even nihilistic view of life. Avoid involving concepts such as community or balance.
    
    I need you to generate ONLY the following specific attributes, each expressed as a single sentence:
    
    1. attitude: A single, concise sentence (5-10 words) describing their overall life attitude
    2. attitude_details: A single sentence (15-20 words) explaining how this attitude manifests in their daily life
    3. coping_mechanism: A single sentence (5-10 words) describing how they deal with challenges
    
    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{"attitude": "single sentence", "attitude_details": "single sentence", "coping_mechanism": "single sentence"}}
    
    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that generates realistic human life attitudes in ONE SENTENCE, including positive, neutral, and negative outlooks. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        # 使用config.py中的解析函数
        result = parse_gpt_response(
            response,
            expected_fields=["attitude", "attitude_details", "coping_mechanism"],
            field_defaults={
                "attitude": "",
                "attitude_details": "",
                "coping_mechanism": ""
            }
        )
        
        # 检查是否有空值
        for field in ["attitude", "attitude_details", "coping_mechanism"]:
            if not result[field]:
                raise ValueError(f"Missing required field: {field}")
        
        attitude = result["attitude"]
        attitude_details = result["attitude_details"]
        coping_mechanism = result["coping_mechanism"]
    except Exception as e:
        print(f"\nError in generate_life_attitude: {e}")
        raise
    
    # 返回结果
    return {
        "attitude": attitude,
        "attitude_details": attitude_details,
        "coping_mechanism": coping_mechanism
    }

def generate_personal_story(age: int, gender: str, occupation: str, location: Dict[str, str], 
                                    values_orientation: str, life_attitude: Dict[str, str]) -> Dict[str, Union[str, List[str]]]:
    """根据人物的基本信息生成个人故事。

    使用GPT根据年龄、性别、职业、地区、价值观和生活态度生成一个人物的详细故事和关键生活事件。
    故事可以包含各种正面或负面的经历，包括极端情况。

    Args:
        age: 人物年龄
        gender: 人物性别
        occupation: 人物职业
        location: 包含'city'和'country'键的字典
        values_orientation: 个人价值观描述
        life_attitude: 生活态度信息字典

    Returns:
        Dict: 包含以下字段的字典：
            - personal_story: 个人故事描述
            - key_life_events: 关键生活事件列表
    """
    # 提取生活态度信息
    attitude = life_attitude.get("attitude", "")
    attitude_category = life_attitude.get("attitude_category", "neutral")
    
    # 创建提示词
    # 随机决定生成几个故事
    num_stories = random.randint(1, 3)
    
    prompt = f"""
    Generate {num_stories} concise personal stories for a person with the following characteristics:
    
    Age: {age}
    Gender: {gender}
    Occupation: {occupation}
    Location: {location['city']}, {location['country']}
    Core Values: {values_orientation}
    Life Attitude: {attitude} ({attitude_category})
    
    IMPORTANT: The story can be positive, negative, or a mix of both. Please do not avoid including life experiences that may be controversial or have negative consequences. The narration should be as specific as possible, objective, and free from any subjective comments or value judgments. This could be a random event unrelated to the background, or a significant turning point in their life. Please avoid including anything related to community building.
    
    Please provide:
    1. {num_stories} brief personal stories (each 100-150 words) that explain different aspects of how this person became who they are today
    2. 1-2 key life events for each story (with approximate ages when they occurred)
    
    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{"personal_stories": ["story 1", "story 2", ...], "key_life_events": [["event 1 for story 1", "event 2 for story 1"], ["event 1 for story 2", ...], ...]}}
    
    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    messages = [
        {"role": "system", "content": "You are an assistant that generates concise realistic personal stories, including both positive and negative life experiences. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        # 解析JSON响应，只使用新的多故事格式
        result = parse_gpt_response(
            response,
            expected_fields=["personal_stories", "key_life_events"],
            field_defaults={
                "personal_stories": [],
                "key_life_events": [[]]
            }
        )
        
        # 处理多故事格式
        stories = result["personal_stories"]
        events = result["key_life_events"]
        
        # 确保有数据
        if not stories or not events:
            raise ValueError("Failed to generate personal stories or key life events")
        
        # 将多故事格式转换为统一的格式
        # 将所有故事连接起来，用分隔符隔开
        combined_story = "\n\n".join([f"Story {i+1}: {story}" for i, story in enumerate(stories)])
        
        # 将所有事件展平并标记每个故事的事件
        all_events = []
        for i, event_group in enumerate(events):
            for event in event_group:
                all_events.append(f"Story {i+1}: {event}")
        
        # 如果没有事件，添加一个默认事件
        if not all_events:
            all_events = ["Story 1: Significant life event"]
        
        return {
            "personal_story": combined_story,
            "key_life_events": all_events
        }
    except Exception as e:
        print(f"\nError in generate_personal_story_and_interests: {e}")
        raise
        
    return {
        "personal_story": combined_story,
        "key_life_events": all_events
    }

def generate_interests_and_hobbies(personal_story: Dict[str, Any]) -> Dict[str, Any]:
    """根据人物的故事生成兴趣爱好列表。

    基于人物的个人故事和关键生活事件，生成兴趣爱好列表，可以包含好的或坏的习惯。
    严格基于故事内容生成，生成3-4个兴趣爱好。

    Args:
        personal_story: 人物的个人故事字典，包含"personal_story"和"key_life_events"字段

    Returns:
        Dict: 包含以下字段的字典：
            - interests: 兴趣爱好列表(3-4个)
    """
    # 确保我们有个人故事数据
    if not personal_story or not isinstance(personal_story, dict):
        raise ValueError("必须提供个人故事数据才能生成兴趣爱好")
    
    personal_story_data = personal_story
    
    # 提取故事和关键事件
    story_text = personal_story_data.get("personal_story", "")
    
    # 创建提示词
    prompt = f"""
   Based on the following personal story and key life events, infer two to three hobbies or interests this person might use to relax. These activities can be positive or negative and may include non-traditional, controversial, or unexpected ones, such as various sports, traveling, or even smoking, drinking, or using marijuana. Please make inferences about the person’s possible interests based on the story, rather than simply extracting them directly from the story. 
   
     Personal Story: {story_text}

     IMPORTANT: Avoid including anything related to community-building activities.
     
    Please extract 2 hobbies or interests with 3-4 words each based on these reflections and format your response as a JSON object:

     {{
        "interests": ["interest1", "interest2"]
     }}

    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that extracts realistic interests and hobbies from a person's life story, including both positive activities and negative habits. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.2)
        
        # 使用config.py中的解析函数
        from config import parse_gpt_response
        result = parse_gpt_response(
            response, 
            expected_fields=["interests"], 
            field_defaults={"interests": []}
        )
        
        # 获取兴趣爱好列表
        interests = result.get("interests", [])
    except Exception as e:
        # 记录错误并重新抛出
        print(f"\nError in generate_interests_and_hobbies: {e}")
        raise
    
    # 输出生成的兴趣爱好数量信息
    print(f"\nInfo: 生成的兴趣爱好数量: {len(interests)}")
    
    # 返回结果
    return {
        "interests": interests
    }