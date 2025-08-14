#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 属性选择器脚本
# 此脚本用于分析用户配置文件并选择最适合的属性

import json
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import argparse
from pathlib import Path
import random
import numpy as np
import pickle
from tqdm import tqdm
import time
ATTRIBUTE_SELECTION_CACHE = None

# 导入项目配置
from config import client, GPT_MODEL, parse_json_response

# 定义get_completion函数
def get_completion(messages, model=GPT_MODEL, temperature=0.7):
    """使用OpenAI API生成文本完成"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None

# 导入based_data模块中的函数
from based_data import (
    generate_age_info,
    generate_gender,
    generate_career_info,
    generate_location,
    generate_personal_values,
    generate_life_attitude,
    generate_personal_story,
    generate_interests_and_hobbies
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 属性数据集路径
ATTRIBUTES_PATH = "/home/zhou/persona/generate_user_profile/code/data/large_attributes.json"  # 属性数据集路径

# 向量数据库路径
EMBEDDINGS_PATH = "/home/zhou/persona/generate_user_profile/code/data/attribute_embeddings.pkl"  # 属性嵌入向量路径

# 默认模型来自配置
DEFAULT_MODEL = GPT_MODEL

# 向量搜索参数
NEAR_NEIGHBOR_COUNT = 1  # 近邻数量
MID_NEIGHBOR_COUNT = 1   # 中距离邻居数量
FAR_NEIGHBOR_COUNT = 1   # 远距离邻居数量
DIVERSITY_THRESHOLD = 0.7  # 多样性阈值（余弦相似度）

class AttributeSelector:
    """
    属性选择器类
    用于分析用户配置文件并使用GPT-4o选择适当的属性
    """
    
    def __init__(self, model: str = DEFAULT_MODEL, user_profile: Dict = None):
        """
        初始化属性选择器
        
        参数：
            model: 要使用的GPT模型
            user_profile: 用户配置文件数据（可选）
        """
        self.model = model
        
        # OpenAI客户端已在config.py中设置
        
        # 加载属性数据
        self.attributes = self._load_json(ATTRIBUTES_PATH)  # 加载属性数据
        
        # 设置用户配置文件
        self.user_profile = user_profile
        
        # 验证数据
        self._validate_data()
        
        # 加载向量数据库
        self.embeddings_data = self._load_embeddings()
        
        # 初始化属性路径和向量映射
        self.path_to_embedding = {}
        self.paths = []
        self.embeddings = []
        
        if self.embeddings_data:
            self.paths = self.embeddings_data.get('paths', [])
            self.embeddings = self.embeddings_data.get('embeddings', [])
            
            # 创建路径到向量的映射
            for i, path in enumerate(self.paths):
                if i < len(self.embeddings):
                    self.path_to_embedding[path] = self.embeddings[i]
            
            logger.info(f"已加载 {len(self.paths)} 条属性路径和对应的向量嵌入")
        
        logger.info(f"已加载属性，包含 {len(self.attributes.keys())} 个顶级类别")
    
    def _load_json(self, file_path: str) -> Dict:
        """从文件加载JSON数据"""
        try:
            return json.loads(Path(file_path).read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"从 {file_path} 加载JSON时出错: {e}")
            raise
            
    def _load_embeddings(self) -> Dict:
        """加载属性嵌入向量数据库"""
        try:
            if not os.path.exists(EMBEDDINGS_PATH):
                logger.warning(f"嵌入向量文件 {EMBEDDINGS_PATH} 不存在")
                return None
                
            with open(EMBEDDINGS_PATH, 'rb') as f:
                embeddings_data = pickle.load(f)
                
            # 检查数据结构并标准化键名
            paths_key = 'attribute_paths' if 'attribute_paths' in embeddings_data else 'paths'
            embeddings_key = 'embeddings'
            
            # 获取路径和嵌入向量
            paths = embeddings_data.get(paths_key, [])
            embeddings = embeddings_data.get(embeddings_key, [])
            
            # 如果数据无效，返回空值
            if not isinstance(embeddings_data, dict) or not paths or not isinstance(embeddings, np.ndarray):
                logger.warning("嵌入向量数据格式无效")
                return None
            
            # 标准化返回的数据字典
            standardized_data = {
                'paths': paths,
                'embeddings': embeddings
            }
                
            logger.info(f"从 {EMBEDDINGS_PATH} 加载了 {len(paths)} 条属性嵌入向量")
            return standardized_data
            
        except Exception as e:
            logger.error(f"加载嵌入向量时出错: {e}")
            return None
    
    def _validate_data(self):
        """验证加载的数据并转换格式（如需要）"""
        # 检查属性格式
        if isinstance(self.attributes, dict):
            # 如果是嵌套字典格式（如large_attributes.json），转换为路径格式
            if "paths" not in self.attributes:
                logger.info("正在将属性从嵌套字典格式转换为路径格式")
                self.attributes = {"paths": self._flatten_attributes(self.attributes)}
        else:
            raise ValueError("无效的属性格式：不是字典")
            
    def _create_profile_embedding(self, profile: Dict) -> np.ndarray:
        """
        为用户配置文件创建嵌入向量
        
        参数：
            profile: 用户配置文件字典
            
        返回：
            配置文件的嵌入向量
        """
        try:
            # 提取配置文件摘要
            profile_summary = self._extract_profile_summary(profile)
            
            # 使用OpenAI API生成嵌入向量
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=profile_summary
            )
            
            # 提取嵌入向量
            embedding = np.array(response.data[0].embedding)
            logger.info(f"成功为用户配置文件创建了嵌入向量")
            return embedding
            
        except Exception as e:
            logger.error(f"创建配置文件嵌入向量时出错: {e}")
            return None
            
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量之间的余弦相似度
        
        参数：
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回：
            余弦相似度值，范围为[-1, 1]
        """
        # 处理空向量或None值
        if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
            return 0.0
            
        # 确保向量维度匹配
        if len(vec1) != len(vec2):
            logger.warning(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")
            return 0.0
            
        # 计算向量范数
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 处理零向量
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # 计算余弦相似度
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def get_profile(self) -> Dict:
        """获取用户配置文件"""
        return self.user_profile
        
    def _check_if_career_needed(self, profile: Dict, profile_summary: str) -> bool:
        """
        使用GPT判断是否需要"Career and Work Identity"属性
        
        参数：
            profile: 用户配置文件字典
            profile_summary: 用户配置文件摘要
            
        返回：
            布尔值，表示是否需要职业相关属性
        """
        # 创建专门用于判断职业属性必要性的提示词
        prompt = f"""
        # Career Attribute Necessity Assessment
        
        ## User Profile Summary:
        {profile_summary}
        
        ## Task Description:
        You are an AI assistant tasked with determining whether the "Career and Work Identity" attribute category is relevant and necessary for this specific individual based on their profile.
        
        Consider the following factors:
        1. The person's age and life stage
        2. Current employment or educational status
        3. How central career and work identity appears to be to this person's life
        4. Whether career information would be essential to create a complete picture of this person
        
        ## Output Format:
        Provide a JSON response with these keys:
        1. "is_career_needed": Boolean (true/false) indicating whether Career and Work Identity attributes are needed
        2. "reasoning": String explaining your rationale
        
        Your response must be valid JSON that can be parsed by Python's json.loads() function.
        """
        
        try:
            # 调用GPT API，使用config.py中的get_completion函数
            messages = [
                {"role": "system", "content": "You are an AI assistant that analyzes user profiles and determines whether career attributes are necessary. You always respond with valid JSON."}, 
                {"role": "user", "content": prompt}
            ]
            content = get_completion(messages, model=self.model, temperature=0.3)
            
            # Use the project's JSON parsing function
            result = parse_json_response(content, {"is_career_needed": True, "reasoning": "默认需要职业属性"})
            
            # 确保结果包含必要的键
            if "is_career_needed" not in result:
                logger.warning("Missing 'is_career_needed' key in GPT response, defaulting to True")
                result["is_career_needed"] = True
                
            logger.info(f"Career attributes needed: {result['is_career_needed']}, Reason: {result.get('reasoning', 'No reasoning provided')}")
            
            return result["is_career_needed"]
                
        except Exception as e:
            logger.error(f"Error calling GPT API for career assessment: {e}")
            # 出错时默认需要职业属性
            return True
    
    def _extract_profile_summary(self, profile: Dict) -> str:
        """
        提取配置文件摘要，用于GPT分析
        
        参数：
            profile: 用户配置文件字典
            
        返回：
            配置文件摘要字符串
        """
        summary_parts = []
        
        # Add age information
        if "age_info" in profile:
            age_info = profile["age_info"]
            if "age" in age_info:
                summary_parts.append(f"Age: {age_info['age']}")
            if "age_group" in age_info:
                summary_parts.append(f"Age Group: {age_info['age_group']}")
        
        # Add gender information
        if "gender" in profile:
            gender = profile["gender"]
            # Convert Chinese characters to English if needed
            if gender == "男":
                gender = "Male"
            elif gender == "女":
                gender = "Female"
            summary_parts.append(f"Gender: {gender}")
        
        # Add location information
        if "location" in profile:
            location = profile["location"]
            location_str = f"{location.get('city', '')}, {location.get('country', '')}"
            summary_parts.append(f"Location: {location_str}")
        
        # Add career information
        if "career_info" in profile and "status" in profile["career_info"]:
            summary_parts.append(f"Career Status: {profile['career_info']['status']}")
        
        # Add personal values
        if "personal_values" in profile and "values_orientation" in profile["personal_values"]:
            summary_parts.append(f"Values: {profile['personal_values']['values_orientation']}")
        
        # Add life attitude
        if "life_attitude" in profile:
            attitude = profile["life_attitude"]
            if "attitude" in attitude:
                summary_parts.append(f"Life Attitude: {attitude['attitude']}")
            if "attitude_details" in attitude:
                summary_parts.append(f"Attitude Details: {attitude['attitude_details']}")
            if "coping_mechanism" in attitude:
                summary_parts.append(f"Coping Mechanism: {attitude['coping_mechanism']}")
        
        # Add interests
        if "interests" in profile and "interests" in profile["interests"]:
            interests = profile["interests"]["interests"]
            if interests and isinstance(interests, list):
                summary_parts.append(f"Interests: {', '.join(interests)}")
        
        # Add summary if available (but not personal story)
        if "summary" in profile:
            summary_parts.append(f"Profile Summary: {profile['summary']}")
            
        # 注意：不提取personal_story部分的内容
        
        return "\n".join(summary_parts)
    
    def _flatten_attributes(self, attributes: Dict, prefix: str = "") -> List[str]:
        """
        将属性字典扁平化为属性路径列表
        
        参数：
            attributes: 属性字典或子字典
            prefix: 当前路径前缀
        """
        result = []
        
        def _flatten(attr_dict, curr_prefix):
            for k, v in attr_dict.items():
                path = f"{curr_prefix}.{k}" if curr_prefix else k
                # 只有叶子节点（空字典）才添加到结果中
                if isinstance(v, dict):
                    if not v:  # 空字典，这是叶子节点
                        result.append(path)
                    else:  # 非空字典，继续递归
                        _flatten(v, path)
                else:  # 非字典值，直接添加
                    result.append(path)
        
        _flatten(attributes, prefix)
        return result
    
    def _get_attribute_categories(self) -> List[str]:
        """获取顶级属性类别"""
        # 从路径中提取唯一的顶级类别
        return sorted({path.split('.')[0] for path in self.attributes["paths"]})
    
    def _format_attributes_tree(self, attributes_dict: Dict, prefix: str = "", depth: int = 0) -> List[str]:
        """
        将属性字典格式化为文本格式的树结构
        
        参数：
            attributes_dict: 属性字典
            prefix: 当前路径前缀
            depth: 树中的当前深度
            
        返回：
            表示树的格式化行列表
        """
        lines = []
        
        for key, value in sorted(attributes_dict.items()):
            current_path = f"{prefix}.{key}" if prefix else key
            indent = "  " * depth
            
            if depth > 0:
                prefix_char = "│ " * (depth - 1) + "- "
            else:
                prefix_char = "- "
                
            lines.append(f"{indent}{prefix_char}{key}")
            
            if isinstance(value, dict) and value:
                sub_lines = self._format_attributes_tree(value, current_path, depth + 1)
                lines.extend(sub_lines)
                
        return lines
    
    def analyze_profile_for_attributes(self, profile: Dict) -> Dict[str, List[str]]:
        """
        分析用户配置文件并确定应该生成哪些属性
        
        参数：
            profile: 用户配置文件字典
            
        返回：
            包含'recommended'和'not_recommended'属性类别的字典
        """
        # 提取配置文件摘要
        profile_summary = self._extract_profile_summary(profile)
        
        # 获取属性类别
        categories = self._get_attribute_categories()
        
        # 第一步：使用GPT判断是否需要"Career and Work Identity"
        career_needed = self._check_if_career_needed(profile, profile_summary)
        
        # 第二步：保留所有一级属性（除了可能被排除的"Career and Work Identity"）
        recommended_categories = []
        not_recommended_categories = []
        
        for category in categories:
            if category == "Career and Work Identity" and not career_needed:
                not_recommended_categories.append(category)
                logger.info(f"根据分析，不需要职业和工作身份属性")
            else:
                recommended_categories.append(category)
        
        # 生成结果
        result = {
            "recommended": recommended_categories,
            "not_recommended": not_recommended_categories,
            "reasoning": f"根据用户背景分析，{'需要' if career_needed else '不需要'}职业和工作身份属性。保留所有其他一级属性，从每个属性中选择最符合用户背景的特征。"
        }
        
        return result
    
    def process_profile(self) -> Dict:
        """
        处理用户配置文件并生成属性推荐
        
        返回：
            包含配置文件和属性推荐的字典
        """
        # 如果未提供用户配置文件，生成一个
        if not self.user_profile:
            self.user_profile = generate_user_profile()
        
        # 提取配置文件摘要
        profile_summary = self._extract_profile_summary(self.user_profile)
        
        # 分析配置文件并获取属性推荐
        attribute_recommendations = self.analyze_profile_for_attributes(self.user_profile)
        
        # 返回结果
        return {
            "profile_summary": profile_summary,
            "attribute_recommendations": attribute_recommendations
        }
    
    def _get_nested_attributes(self, category: str) -> Dict:
        """
        通过从路径重建获取特定类别的嵌套属性
        
        参数：
            category: 类别名称
        """
        result = {}
        
        # 筛选以该类别开头的路径并重建嵌套结构
        for path in [p for p in self.attributes["paths"] if p.startswith(f"{category}.")]:
            parts = path.split('.')[1:]  # 跳过第一部分（类别）
            current = result
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = {}  # 叶节点
                else:
                    current.setdefault(part, {})  # 如果键不存在则创建
                    current = current[part]
        
        return result
    

    
    def select_top_attributes(self, path_list, target_count=200):
        """
        从路径列表中选择最重要和最具代表性的顶级属性。
        
        参数：
            path_list: 属性路径列表
            target_count: 要选择的目标属性数量
            
        返回：
            选定的属性路径列表
        """
        if not path_list:
            return []
            
        # If we already have fewer paths than target, return all of them
        if len(path_list) <= target_count:
            return path_list
        
        # Extract unique top-level categories
        categories = {}
        for path in path_list:
            parts = path.split('.')
            if len(parts) > 0:
                category = parts[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(path)
        
        # Calculate how many attributes to select from each category
        # proportional to their representation in the original list
        total_paths = len(path_list)
        category_counts = {}
        for category, paths in categories.items():
            # Calculate proportional count but ensure at least 1 attribute per category
            count = max(1, int(len(paths) / total_paths * target_count))
            category_counts[category] = count
        
        # Adjust counts to match target_count as closely as possible
        total_selected = sum(category_counts.values())
        if total_selected < target_count:
            # Distribute remaining slots to largest categories
            remaining = target_count - total_selected
            sorted_categories = sorted(categories.keys(), 
                                      key=lambda c: len(categories[c]), 
                                      reverse=True)
            for i in range(remaining):
                if i < len(sorted_categories):
                    category_counts[sorted_categories[i]] += 1
        elif total_selected > target_count:
            # Remove from largest categories until we hit target
            excess = total_selected - target_count
            sorted_categories = sorted(categories.keys(), 
                                      key=lambda c: category_counts[c], 
                                      reverse=True)
            for i in range(excess):
                if i < len(sorted_categories) and category_counts[sorted_categories[i]] > 1:
                    category_counts[sorted_categories[i]] -= 1
        
        # Select paths from each category
        selected_paths = []
        for category, count in category_counts.items():
            category_paths = categories[category]
            
            # Prioritize paths with fewer segments (more general attributes)
            # and paths that represent common attributes across domains
            scored_paths = []
            for path in category_paths:
                parts = path.split('.')
                # Score is based on path depth (shorter is better) and presence of common terms
                common_terms = ['general', 'common', 'basic', 'core', 'essential', 'fundamental', 'key']
                common_term_bonus = any(term in path.lower() for term in common_terms)
                score = (10 - len(parts)) + (5 if common_term_bonus else 0)
                scored_paths.append((path, score))
            
            # Sort by score (higher is better) and select top paths
            sorted_paths = sorted(scored_paths, key=lambda x: x[1], reverse=True)
            selected_category_paths = [p[0] for p in sorted_paths[:count]]
            selected_paths.extend(selected_category_paths)
        
        return selected_paths
    
    # 注意：删除了_select_best_matching_attributes方法，因为它已经被新的随机选择和GPT筛选方法替代
        
    def _find_interesting_neighbors(self, profile_embedding: np.ndarray, category_paths: List[str], 
    target_count: int = 300) -> List[str]:
        """
        实现"有趣的邻居"向量搜索方案
        
        参数：
            profile_embedding: 用户配置文件的嵌入向量
            category_paths: 特定类别的属性路径列表
            target_count: 要选择的目标属性数量
            
        返回：
            选定的属性路径列表或空列表（如果出现读取问题）
        """
        # 如果没有向量数据库或路径为空，返回空列表
        if not self.embeddings_data or not category_paths:
            logger.warning("没有可用的向量数据库或类别路径为空，返回空列表")
            return []
        
        # 过滤出在向量数据库中有嵌入向量的路径
        valid_paths = [path for path in category_paths if path in self.path_to_embedding]
        
        if not valid_paths:
            logger.warning("没有有效的属性路径匹配向量数据库，返回空列表")
            return []
        
        # 计算每个路径与配置文件的相似度
        path_similarities = []
        for path in valid_paths:
            embedding = self.path_to_embedding[path]
            similarity = self._compute_cosine_similarity(profile_embedding, embedding)
            path_similarities.append((path, similarity))
        
        # 按相似度排序
        path_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 计算要选择的数量
        total_paths = len(path_similarities)
        
        # 如果路径数量少于目标数量，返回所有路径
        if total_paths <= target_count:
            return [p[0] for p in path_similarities]
        
        selected_paths = []
        used_indices = set()
        
        # 1. 选择近邻（相似度最高的属性）
        near_count = min(int(target_count * 0.4), total_paths // 3)  # 大约40%的目标数量
        near_indices = list(range(near_count))
        for i in random.sample(near_indices, near_count):
            if i not in used_indices:
                selected_paths.append(path_similarities[i][0])
                used_indices.add(i)
        
        # 2. 选择中距离邻居
        mid_start = total_paths // 3
        mid_end = 2 * total_paths // 3
        mid_count = min(int(target_count * 0.4), (mid_end - mid_start))  # 大约40%的目标数量
        mid_indices = list(range(mid_start, mid_end))
        for i in random.sample(mid_indices, mid_count):
            if i not in used_indices:
                selected_paths.append(path_similarities[i][0])
                used_indices.add(i)
        
        # 3. 选择远距离邻居（相似度最低的属性）
        far_start = 2 * total_paths // 3
        far_count = min(int(target_count * 0.2), (total_paths - far_start))  # 大约20%的目标数量
        far_indices = list(range(far_start, total_paths))
        for i in random.sample(far_indices, far_count):
            if i not in used_indices:
                selected_paths.append(path_similarities[i][0])
                used_indices.add(i)
        
        # 如果还需要更多属性来达到目标数量，从未使用的索引中随机选择
        remaining_count = target_count - len(selected_paths)
        if remaining_count > 0:
            remaining_indices = [i for i in range(total_paths) if i not in used_indices]
            if remaining_indices:
                additional_count = min(remaining_count, len(remaining_indices))
                for i in random.sample(remaining_indices, additional_count):
                    selected_paths.append(path_similarities[i][0])
        
        logger.info(f"使用向量搜索选择了 {len(selected_paths)} 条属性（近邻: {near_count}, 中距离: {mid_count}, 远距离: {far_count}）")
        return selected_paths
    
    def get_top_attributes(self, result: Dict) -> List[str]:
        """
        获取属性列表
        
        参数：
            result: 来自analyze_profile_for_attributes的结果
            
        返回：
            属性路径列表
        """
        try:
            # 收集所有推荐的类别
            all_recommended = set()
            
            if "recommended" in result:
                all_recommended.update(result["recommended"])
            
            # 提取所有路径
            all_paths = []
            category_paths = {}
            
            if "paths" in self.attributes:
                # 如果我们使用带有路径的新格式
                for path in self.attributes["paths"]:
                    # 只包括来自推荐类别的路径
                    category = path.split('.')[0] if '.' in path else path
                    if category in all_recommended:
                        all_paths.append(path)
                        # 按类别分组路径
                        if category not in category_paths:
                            category_paths[category] = []
                        category_paths[category].append(path)
            else:
                # 如果我们使用旧的嵌套格式，将其扁平化
                for category in all_recommended:
                    paths = self._flatten_attributes(self._get_nested_attributes(category), category)
                    all_paths.extend(paths)
                    category_paths[category] = paths
            
            # 随机选择属性数量
            attribute_count_options = [250, 300, 350]
            target_count = random.choice(attribute_count_options)
            
            # 如果有向量数据库，使用向量搜索
            if self.embeddings_data and self.user_profile:
                # 创建用户配置文件的嵌入向量
                profile_embedding = self._create_profile_embedding(self.user_profile)
                
                if profile_embedding is not None:
                    # 为每个类别选择属性
                    final_paths = []
                    for category, paths in category_paths.items():
                        # 根据类别大小按比例分配目标数量
                        category_ratio = len(paths) / len(all_paths)
                        category_target = max(10, int(target_count * category_ratio))
                        
                        # 使用向量搜索选择该类别的属性
                        category_selected = self._find_interesting_neighbors(
                            profile_embedding, paths, category_target
                        )
                        final_paths.extend(category_selected)
                    
                    # 如果选择的属性超过目标数量，随机减少
                    if len(final_paths) > target_count:
                        final_paths = random.sample(final_paths, target_count)
                    
                    logger.info(f"使用向量搜索从 {len(all_paths)} 条属性中选择了 {len(final_paths)} 条属性")
                    return final_paths
            
            # 如果没有向量数据库或向量搜索失败，直接返回空列表
            logger.warning(f"没有可用的向量数据库或向量搜索失败，返回空列表")
            return []
            
        except Exception as e:
            logger.error(f"Error generating attribute list: {e}")
            # 如果出错，直接返回空列表
            logger.error(f"属性选择过程出错，返回空列表")
            return []
            
    # 注意：删除了_random_select_from_top_categories方法，因为现在直接在get_top_attributes中随机选择属性
    


def generate_user_profile() -> Dict:
    """生成用户基础信息配置文件"""
    # 生成并存储直接函数返回值
    age_info = generate_age_info()
    gender = generate_gender()
    location = generate_location()
    career_info = generate_career_info(age_info["age"])
    
    # 生成个人价值观
    values = generate_personal_values(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location
    )
    
    # 生成生活态度
    life_attitude = generate_life_attitude(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location,
        values_orientation=values.get("values_orientation", "")
    )
    
    # 生成个人故事
    personal_story = generate_personal_story(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location,
        values_orientation=values.get("values_orientation", ""),
        life_attitude=life_attitude
    )
    
    # 生成兴趣爱好
    interests = generate_interests_and_hobbies(personal_story)
    
    # 存储函数返回值
    user_profile = {
        "age_info": age_info,
        "gender": gender,
        "location": location,
        "career_info": career_info,
        "personal_values": values,
        "life_attitude": life_attitude,
        "personal_story": personal_story,
        "interests": interests
    }
    
    return user_profile

def get_selected_attributes(user_profile=None):
    global ATTRIBUTE_SELECTION_CACHE
    if ATTRIBUTE_SELECTION_CACHE is not None:
        return ATTRIBUTE_SELECTION_CACHE

    try:
        # 如果没有提供用户配置文件，生成一个
        if user_profile is None:
            user_profile = generate_user_profile()
        
        # 创建选择器并传入用户配置文件
        selector = AttributeSelector(user_profile=user_profile)
        
        # 处理配置文件
        result = selector.process_profile()
        
        # 获取属性推荐
        attribute_recommendations = result.get("attribute_recommendations", {})
        
        # 获取属性列表
        top_paths = selector.get_top_attributes(attribute_recommendations)
        
        # 返回属性列表
        ATTRIBUTE_SELECTION_CACHE = top_paths
        return top_paths
        
    except Exception as e:
        logger.error(f"Error getting selected attributes: {e}")
        return []

def build_nested_dict(paths: List[str]) -> Dict:
    result = {}
    for path in paths:
        parts = path.split('.')
        current = result
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return result

def save_results(user_profile: Dict, selected_paths: List[str], output_dir: str = '/home/zhou/persona/generate_user_profile/output') -> None:
    """
    保存用户配置文件和选定的属性路径到文件
    参数：
        user_profile: 用户配置文件
        selected_paths: 选定的属性路径 (列表形式)
        output_dir: 输出目录（默认为 '/home/zhou/persona/generate_user_profile/output'）
    """
    try:
        from pathlib import Path
        import json
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存用户配置文件
        profile_path = output_path / "user_profile.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(user_profile, f, ensure_ascii=False, indent=2)
        logger.info(f"用户配置文件已保存到 {profile_path}")
        
        # 将 selected_paths 转换为嵌套字典结构
        nested_selected_paths = build_nested_dict(selected_paths)
        
        # 保存属性路径（嵌套字典格式）
        paths_path = output_path / "selected_paths.json"
        with open(paths_path, 'w', encoding='utf-8') as f:
            json.dump(nested_selected_paths, f, ensure_ascii=False, indent=2)
        logger.info(f"属性路径已保存到 {paths_path}")
    except Exception as e:
        logger.error(f"保存结果时出错: {e}")
        raise

# 示例：在生成用户基本信息和属性列表的函数中自动调用保存（请根据实际情况将此调用添加到合适位置）
user_profile = generate_user_profile()
selected_paths = get_selected_attributes(user_profile)
save_results(user_profile, selected_paths)

# 此文件只供其他文件导入使用