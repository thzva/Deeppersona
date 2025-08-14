"""配置模块，管理API密钥、代理设置和OpenAI客户端配置。

此模块包含以下主要功能：
- OpenAI API配置
- 代理服务器设置
- OpenAI客户端初始化
- API调用工具函数
- GPT响应解析函数
"""

import os
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union, Tuple

# API配置
# 注意：在实际生产环境中，应该从环境变量或配置文件中读取API密钥

# GeoNames API配置
GEONAMES_USERNAME = "demo"  # 替换为你的GeoNames用户名
GEONAMES_API_BASE = "http://api.geonames.org"

# OpenAI API配置
OPENAI_API_KEY = "OPENAI-API-KEY"

# 使用的GPT模型版本
GPT_MODEL = ""

# Proxy Configuration
os.environ['https_proxy'] = ''
os.environ['http_proxy'] = ''

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_completion(messages: List[Dict[str, str]], model: str = GPT_MODEL, temperature: float = 0.5) -> Optional[str]:
    """调用OpenAI API获取回复。

    使用配置的OpenAI客户端发送请求并获取回复。支持自定义模型和temperature参数。

    Args:
        messages (List[Dict[str, str]]): 消息列表，每个消息包含role和content
        model (str, optional): 使用的GPT模型名称。默认为配置中的GPT_MODEL
        temperature (float, optional): 采样温度，控制输出的随机性。默认为0.2

    Returns:
        Optional[str]: API返回的文本回复。如果请求失败则返回None
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {e}")
        return None


def extract_json_from_markdown(response: str) -> str:
    """从Markdown代码块中提取JSON内容。

    Args:
        response (str): 可能包含Markdown代码块的响应文本

    Returns:
        str: 提取的JSON内容或原始响应
    """
    if response and response.strip().startswith('```') and '```' in response:
        # 提取代码块内容
        code_content = response.split('```', 2)[1]
        if code_content.startswith('json'):
            code_content = code_content[4:].strip()
        response = code_content.strip()
    return response


def parse_json_response(response: str, default_value: Any = None) -> Any:
    """解析JSON响应，处理可能的解析错误。

    Args:
        response (str): JSON格式的响应文本
        default_value (Any, optional): 解析失败时返回的默认值

    Returns:
        Any: 解析后的JSON对象或默认值
    """
    if not response:
        return default_value
    
    # 首先尝试从Markdown中提取JSON
    response = extract_json_from_markdown(response)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        print(f"Response was: {response[:100]}..." if len(response) > 100 else f"Response was: {response}")
        return default_value


def parse_gpt_response(response: str, expected_fields: List[str] = None, field_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """解析GPT响应，提取预期字段并应用默认值。

    Args:
        response (str): GPT响应文本
        expected_fields (List[str], optional): 预期的字段列表
        field_defaults (Dict[str, Any], optional): 字段默认值字典

    Returns:
        Dict[str, Any]: 包含所有预期字段的字典
    """
    if field_defaults is None:
        field_defaults = {}
    
    # 解析JSON响应
    result = parse_json_response(response, {})
    
    # 如果没有指定预期字段，直接返回解析结果
    if not expected_fields:
        return result
    
    # 确保返回所有预期字段
    output = {}
    for field in expected_fields:
        output[field] = result.get(field, field_defaults.get(field))
    
    return output


def parse_nested_json_response(response: str) -> Tuple[Dict[str, Any], bool]:
    """解析可能嵌套的JSON响应。

    处理可能嵌套在Markdown代码块中的JSON字符串表示的JSON对象。

    Args:
        response (str): 可能包含嵌套JSON的响应文本

    Returns:
        Tuple[Dict[str, Any], bool]: 解析后的JSON对象和是否成功解析的标志
    """
    # 首先从Markdown中提取内容
    extracted = extract_json_from_markdown(response)
    
    # 尝试解析JSON
    try:
        result = json.loads(extracted)
        
        # 检查是否是JSON字符串表示的JSON对象
        if isinstance(result, dict) and len(result) == 1 and next(iter(result.values())).startswith('{'):
            key = next(iter(result.keys()))
            try:
                nested_json = json.loads(result[key])
                return nested_json, True
            except json.JSONDecodeError:
                pass
        
        return result, True
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return {}, False
