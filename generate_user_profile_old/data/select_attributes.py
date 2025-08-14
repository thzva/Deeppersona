#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 属性选择器脚本
# 此脚本用于分析用户配置文件并选择最适合的属性

import json
import os
import sys
import logging
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
import random
import numpy as np
ATTRIBUTE_SELECTION_CACHE = None

# 导入项目配置
from config import client, GPT_MODEL, parse_json_response, get_completion

# 导入属性嵌入器
import sys
sys.path.append('/home/zhou/persona/test_evaluation/misc')
from embed_attributes import AttributeEmbedder

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

# 默认模型来自配置
DEFAULT_MODEL = GPT_MODEL

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
        
        logger.info(f"已加载属性，包含 {len(self.attributes.keys())} 个顶级类别")
    
    def _load_json(self, file_path: str) -> Dict:
        """从文件加载JSON数据"""
        try:
            return json.loads(Path(file_path).read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"从 {file_path} 加载JSON时出错: {e}")
            raise
    
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
    
    def _select_attributes_with_gpt(self, category, paths, profile_summary, count):
        """
        使用GPT为特定类别选择最匹配的属性
        
        参数：
            category: 类别名称
            paths: 该类别下的所有属性路径
            profile_summary: 用户配置文件摘要
            count: 要选择的属性数量
            
        返回：
            选定的属性路径列表
        """
        # 如果路径数量少于或等于目标数量，直接返回所有路径
        if len(paths) <= count:
            return paths
            
        # 准备GPT提示词
        prompt = f"""
        # Attribute Selection for Category: {category}
        
        ## User Profile Summary:
        {profile_summary}
        
        ## Available Attributes in {category} Category:
        {json.dumps(paths, ensure_ascii=False, indent=2)}
        
        ## Task Description:
        You are an AI assistant tasked with selecting the {count} most relevant attributes from the {category} category that best match this user's background and profile.
        
        ## Selection Criteria:
        1. Select attributes that are most relevant to the user's age, gender, location, occupation, values, and life attitude
        2. Choose attributes that form a coherent and consistent picture of the person
        3. Prioritize attributes that are more specific and descriptive over generic ones
        4. Consider the cultural and demographic context of the user
        
        ## Output Format:
        Provide a JSON response with this key:
        "selected_paths": Array of exactly {count} attribute paths that you've selected as most relevant
        
        Your response must be valid JSON that can be parsed by Python's json.loads() function.
        """
        
        try:
            # 调用GPT API，使用config.py中的get_completion函数
            messages = [
                {"role": "system", "content": "You are an AI assistant that selects the most relevant attributes for a user profile. You always respond with valid JSON."}, 
                {"role": "user", "content": prompt}
            ]
            content = get_completion(messages, model=self.model, temperature=0.3)
            
            # 使用项目的JSON解析函数
            result = parse_json_response(content, {"selected_paths": []})
            
            # 验证结果
            if "selected_paths" not in result or not isinstance(result["selected_paths"], list):
                logger.warning(f"Invalid GPT response format for {category} category selection")
                # 如果响应无效，随机选择路径

                return random.sample(paths, min(count, len(paths)))
                
            selected_paths = result["selected_paths"]
            
            # 验证所有路径都在原始路径列表中
            valid_paths = [p for p in selected_paths if p in paths]
            
            # 如果有效路径数量不足，补充其他路径
            if len(valid_paths) < count:
                remaining_paths = [p for p in paths if p not in valid_paths]
                # 随机选择剩余路径
                additional_paths = random.sample(remaining_paths, min(count - len(valid_paths), len(remaining_paths)))
                valid_paths.extend(additional_paths)
            
            # 确保不超过请求的数量
            return valid_paths[:count]
                
        except Exception as e:
            logger.error(f"Error calling GPT API for {category} attribute selection: {e}")
            # 出错时随机选择路径
            return random.sample(paths, min(count, len(paths)))
    
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
        
    def _filter_similar_attributes(self, embedder, attributes: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """
        过滤相似度高的属性，只保留相似度低于阈值的属性
        
        参数：
            embedder: AttributeEmbedder实例
            attributes: 属性列表
            similarity_threshold: 相似度阈值，低于此值的属性将被保留
            
        返回：
            过滤后的属性列表
        """
        if not attributes or len(attributes) <= 1:
            return attributes
            
        logger.info(f"开始过滤相似属性，原始属性数量: {len(attributes)}")
        
        # 获取属性的嵌入向量
        attribute_indices = [embedder.attribute_paths.index(path) for path in attributes if path in embedder.attribute_paths]
        if not attribute_indices:
            return attributes
            
        attribute_embeddings = embedder.embeddings[attribute_indices]
        
        # 初始化结果列表，从第一个属性开始
        filtered_attributes = [attributes[0]]
        filtered_indices = [0]
        
        # 遍历其余属性
        for i in range(1, len(attributes)):
            # 检查当前属性与已选属性的相似度
            current_embedding = attribute_embeddings[i]
            
            # 如果没有已选属性，直接添加当前属性
            if not filtered_indices:
                filtered_attributes.append(attributes[i])
                filtered_indices.append(i)
                continue
                
            # 获取已选属性的嵌入向量
            selected_embeddings = attribute_embeddings[filtered_indices]
            
            # 计算当前属性与所有已选属性的相似度
            # 归一化当前嵌入向量
            current_norm = np.linalg.norm(current_embedding)
            if current_norm > 0:
                current_embedding_normalized = current_embedding / current_norm
            else:
                current_embedding_normalized = current_embedding
                
            # 归一化已选属性的嵌入向量
            selected_norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
            selected_norms[selected_norms == 0] = 1  # 避免除以0
            selected_embeddings_normalized = selected_embeddings / selected_norms
            
            # 计算余弦相似度
            similarities = np.dot(selected_embeddings_normalized, current_embedding_normalized)
            
            # 如果与所有已选属性的相似度都低于阈值，则添加该属性
            if np.all(similarities < similarity_threshold):
                filtered_attributes.append(attributes[i])
                filtered_indices.append(i)
        
        logger.info(f"过滤完成，保留了 {len(filtered_attributes)} 个属性")
        return filtered_attributes

    def get_top_attributes(self, result: Dict, target_count: int = 450) -> List[str]:
        """
        获取属性列表 - 从每个大类中随机选择10-25个属性
        
        参数：
            result: 来自analyze_profile_for_attributes的结果，包含推荐的类别
            target_count: 要选择的属性数量，默认为450
            
        返回：
            属性路径列表
        """
        # 检查是否有用户配置文件
        if not self.user_profile:
            logger.error("没有用户配置文件，无法选择属性")
            return []
        
        # 获取推荐的类别
        recommended_categories = result.get('recommended', [])
        not_recommended_categories = result.get('not_recommended', [])
        
        logger.info(f"推荐的类别: {recommended_categories}")
        logger.info(f"不推荐的类别: {not_recommended_categories}")
        
        # 提取用户配置文件摘要
        profile_summary = self._extract_profile_summary(self.user_profile)
        
        # 获取所有可用的属性路径
        all_paths = []
        
        if "paths" in self.attributes:
            # 如果属性已经是扁平化的路径列表
            all_paths = self.attributes["paths"]
        else:
            # 如果属性是嵌套结构，需要扁平化
            all_paths = self._flatten_attributes(self.attributes)
        
        try:
            # 创建AttributeEmbedder实例
            embedder = AttributeEmbedder()
            
            # 获取所有大类
            categories = self._get_attribute_categories()
            logger.info(f"发现了 {len(categories)} 个属性大类")
            
            # 最终选择的属性
            final_selected_paths = []
            
            # 遍历每个大类，使用RAG选择属性
            for category in categories:
                # 只处理推荐的类别
                if category not in recommended_categories:
                    logger.info(f"类别 '{category}' 不在推荐列表中，跳过")
                    continue
                    
                logger.info(f"处理类别 '{category}'")
                
                # 检查该大类下的属性数量
                category_paths = [path for path in all_paths if path.split('.')[0] == category]
                if len(category_paths) < 10:  # 如果大类内属性条数<10条则跳过该大类
                    logger.info(f"大类 '{category}' 下的属性数量少于10，跳过")
                    continue
                
                # 创建一个临时的简化用户配置文件字典
                temp_profile = {"summary": profile_summary}
                
                # 随机决定从该大类选择的属性数量（10-25之间）
                num_to_select = random.randint(10, 25)
                
                # 使用RAG从每个大类选取最相关的属性
                top_k = 100
                if len(category_paths) < top_k:
                    # 如果属性数量不足100个，则选择50%
                    top_k = max(10, len(category_paths) // 2)
                    logger.info(f"大类 '{category}' 属性数量不足100，将选择 {top_k} 个属性")
                
                category_selected = embedder.find_attributes_by_category(temp_profile, category, top_k=top_k)
                
                if category_selected:
                    logger.info(f"RAG为大类 '{category}' 选择了 {len(category_selected)} 个属性")
                    
                    # 计算属性之间的相似度并过滤高相似度的属性
                    filtered_attributes = self._filter_similar_attributes(embedder, category_selected, 0.93)  # 相似度低于93%的留下
                    logger.info(f"大类 '{category}' 过滤后保留了 {len(filtered_attributes)} 个属性")
                    
                    # 如果过滤后的属性数量大于要选择的数量，随机选择指定数量
                    if len(filtered_attributes) > num_to_select:
                        selected_attributes = random.sample(filtered_attributes, num_to_select)
                        logger.info(f"从大类 '{category}' 随机选择了 {num_to_select} 个属性")
                    else:
                        selected_attributes = filtered_attributes
                        logger.info(f"大类 '{category}' 过滤后的属性数量不足 {num_to_select}，保留所有 {len(filtered_attributes)} 个属性")
                    
                    final_selected_paths.extend(selected_attributes)
            
            logger.info(f"总共选择了 {len(final_selected_paths)} 个属性")
            
            # 如果属性数量超过300个，随机选择[50,100,150,200,250,300]条属性
            if len(final_selected_paths) > 300:
                target_count = random.choice([50, 100, 150, 200, 250, 300])
                final_selected_paths = random.sample(final_selected_paths, target_count)
                logger.info(f"属性数量超过300，随机选择了 {target_count} 个属性")
            else:
                logger.info(f"属性数量不超过300，保留所有 {len(final_selected_paths)} 个属性")
            
            return final_selected_paths
            
        except Exception as e:
            logger.error(f"Error calling RAG API for attribute selection: {e}")
            logger.error(f"RAG选择属性时出错: {e}", exc_info=True)
            # 直接抛出异常，不再使用备用方法
            raise
            
    def _select_attributes_with_gpt_from_all(self, all_paths: List[str], profile: Dict, target_count: int) -> List[str]:
        """
        使用GPT从所有属性路径中选择指定数量的最终属性
        
        参数：
            all_paths: 所有可用的属性路径列表
            profile: 用户配置文件
            target_count: 要选择的属性数量
            
        返回：
            最终选择的属性路径列表
        """
        if not all_paths:
            return []
            
        # 提取配置文件摘要
        profile_summary = self._extract_profile_summary(profile)
        
        # 准备GPT提示词
        prompt = f"""
        # Final Attribute Selection
        
        ## User Profile Summary:
        {profile_summary}
        
        ## Task Description:
        You are an AI assistant tasked with selecting {target_count} attributes from a large pool of available attributes that best match this user's background and profile. You need to select exactly {target_count} attributes that are most relevant to the user. These will be the most important attributes that define this user's profile.
        
        ## Available Attributes:
        {json.dumps(all_paths, ensure_ascii=False, indent=2)}
        
        ## Selection Criteria:
        1. Select attributes that are most relevant to the user's age, gender, location, occupation, values, and life attitude
        2. Choose attributes that form a coherent and consistent picture of the person
        3. Prioritize attributes that are more specific and descriptive over generic ones
        4. Consider the cultural and demographic context of the user
        5. Select a diverse set of attributes that cover different aspects of the person's identity and life
        
        ## Output Format:
        Provide a JSON response with this key:
        "selected_paths": Array of exactly {target_count} attribute paths that you've selected as most relevant
        
        Your response must be valid JSON that can be parsed by Python's json.loads() function.
        """
        
        try:
            # 调用GPT API，使用config.py中的get_completion函数
            messages = [
                {"role": "system", "content": "You are an AI assistant that selects the most relevant attributes for a user profile. You always respond with valid JSON."}, 
                {"role": "user", "content": prompt}
            ]
            content = get_completion(messages, model=self.model, temperature=0.3)
            
            # 使用项目的JSON解析函数
            result = parse_json_response(content, {"selected_paths": []})
            
            # 验证结果
            if "selected_paths" not in result or not isinstance(result["selected_paths"], list):
                logger.warning("Invalid GPT response format for final attribute selection")
                # 如果响应无效，随机选择属性
                if len(all_paths) > target_count:
                    return random.sample(all_paths, target_count)
                else:
                    return all_paths
                
            selected_paths = result["selected_paths"]
            
            # 验证所有路径都在原始路径列表中
            valid_paths = [p for p in selected_paths if p in all_paths]
            
            # 如果有效路径数量不足，补充其他路径
            if len(valid_paths) < target_count:
                remaining_paths = [p for p in all_paths if p not in valid_paths]
                # 随机选择剩余路径
                additional_paths = random.sample(remaining_paths, min(target_count - len(valid_paths), len(remaining_paths)))
                valid_paths.extend(additional_paths)
            
            # 确保不超过请求的数量
            return valid_paths[:target_count]
                
        except Exception as e:
            logger.error(f"Error calling GPT API for final attribute selection: {e}")
            # 出错时随机选择属性
            if len(all_paths) > target_count:
                return random.sample(all_paths, target_count)
            else:
                return all_paths
                
    def _select_attributes_with_gpt_from_random(self, random_paths: List[str], profile: Dict) -> List[str]:
        """
        使用GPT从随机选择的属性子集中选择最终属性
        此方法保留用于向后兼容，但不再使用
        
        参数：
            random_paths: 随机选择的属性路径列表
            profile: 用户配置文件
            
        返回：
            最终选择的属性路径列表
        """
        # 调用新方法，传入相同数量的目标属性数
        return self._select_attributes_with_gpt_from_all(random_paths, profile, len(random_paths))

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

def save_results(user_profile: Dict, selected_paths: List[str], output_dir: str = '/home/zhou/persona/generate_user_profile/code/output') -> None:
    """
    保存用户配置文件和选定的属性路径到文件
    参数：
        user_profile: 用户配置文件
        selected_paths: 选定的属性路径 (列表形式)
        output_dir: 输出目录（默认为 '/home/zhou/persona/generate_user_profile/code/output'）
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