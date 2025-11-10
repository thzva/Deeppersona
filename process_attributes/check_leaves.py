#!/usr/bin/env python3
import os
import json
from tqdm import tqdm
from typing import Tuple, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 初始化sentence transformer模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 设置OpenAI客户端
OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_MODEL = "gpt-4o"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def get_sibling_paths(data: dict, current_path: str) -> List[str]:
    """获取同级路径"""
    if not current_path:
        return []
        
    # 分解路径
    parts = current_path.split('.')
    parent_path = '.'.join(parts[:-1])  # 父路径
    
    # 如果是顶级路径
    if not parent_path:
        return [key for key in data.keys() if key != parts[0]]
    
    # 获取父节点
    current_dict = data
    for part in parent_path.split('.'):
        current_dict = current_dict.get(part, {})
    
    # 返回同级路径
    siblings = [f"{parent_path}.{key}" for key in current_dict.keys() if key != parts[-1]]
    return siblings


def convert_tree_to_paths(data: dict, current_path: str = "") -> list:
    """将树形结构转换为路径列表"""
    paths = []
    for key, value in data.items():
        new_path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            if not value:  # 叶子节点
                paths.append(new_path)
            else:
                paths.extend(convert_tree_to_paths(value, new_path))
    return paths

def convert_paths_to_tree(paths: list) -> dict:
    """将路径列表转换回树形结构"""
    tree = {}
    for path in paths:
        current = tree
        parts = path.split('.')
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = {}  # 叶子节点
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
    return tree

def remove_duplicates(paths: list) -> list:
    """移除重复的路径，保持顺序"""
    if not paths:
        return []
    
    result = [paths[0]]
    for i in range(1, len(paths)):
        if paths[i] != paths[i-1]:
            result.append(paths[i])
    return result

def check_path_similarity(path1: str, path2: str) -> bool:
    """使用sentence-transformers检查两条路径是否过于相似
    
    Args:
        path1: 第一个路径
        path2: 第二个路径
        
    Returns:
        bool: 如果两条路径的相似度超过阈值，返回true，否则返回false
    """
    # 将路径转换为向量
    embeddings = model.encode([path1, path2])
    
    # 计算余弦相似度
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # 如果相似度超过阈值，则认为路径过于相似
    threshold = 0.85  # 可以根据需要调整这个阈值
    if similarity > threshold:
        print(f"发现相似路径(相似度: {similarity:.2f}):\n- {path1}\n- {path2}")
    return similarity > threshold

def check_level_compatibility(current_level: str, parent_level: str) -> bool:
    """Check if the current level is compatible with its parent level using ChatGPT.
    
    Args:
        current_level: The current level to check
        parent_level: The parent level to check against
        
    Returns:
        bool: True if the levels are compatible, False otherwise
    """
    prompt = f"""Analyze if the current level '{current_level}' is compatible with its parent level '{parent_level}'.

Rules:
1. The current level must be a logical subdivision, attribute, or subcategory of the parent level.
2. Both levels should be general categories, not specific instances.
3. The current level should represent a more specific subset of the parent level.
4. The relationship between levels must make logical sense in a hierarchical structure.

Please respond with ONLY 'true' or 'false'.
- 'true' means the levels are compatible and form a valid hierarchy
- 'false' means the levels are incompatible or illogical
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a path hierarchy analyzer. Your task is to strictly judge if two levels are compatible. Only return 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.lower() == 'true'
        if result:
            print(f"Level compatibility check (GPT):\n- Current level: {current_level}\n- Parent level: {parent_level}")
        else:
            print(f"Level incompatibility found (GPT):\n- Current level: {current_level}\n- Parent level: {parent_level}")
        return result
        
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return False

def check_attribute_quality(attribute: str, full_path: str) -> bool:
    """Check if an attribute meets the quality requirements for personalization.
    
    Args:
        attribute: The leaf node attribute to check
        full_path: The full path of the attribute for context
        
    Returns:
        bool: True if the attribute meets all quality requirements
    """
    prompt = f"""Analyze if the attribute '{attribute}' (from path: {full_path}) meets these requirements:

1. User-Centric Focus:
   - Must describe personal characteristics/attributes
   - Should be general enough to apply to many individuals
   - Should enable rich content generation about a person

2. Category Requirements:
   - Must be a general category (no specific instances, behaviors, or values)

Please respond with ONLY 'true' or 'false'.
- 'true' means the attribute meets ALL requirements
- 'false' means it fails one or more requirements
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict attribute quality checker for persona generation. Your task is to ensure attributes meet specific quality standards. Only return 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.lower() == 'true'
        
        if result:
            print(f"Retained: {full_path}")
        
        return result
        
    except Exception as e:
        print(f"Error checking attribute quality: {e}")
        return False

def validate_path_levels(path: str) -> bool:
    """Validate each node and level structure of the path.
    
    Args:
        path: The path to validate
        
    Returns:
        bool: True if all levels in the path are valid
    """
    levels = path.split('.')
    
    # First check if the leaf node (last level) meets quality requirements
    if not check_attribute_quality(levels[-1], path):
        return False
    
    # Then check each level's compatibility with its parent
    for i in range(len(levels)-1, 0, -1):
        current = levels[i]  # Current node (starting from leaf)
        parent = levels[i-1]  # Parent node
        if not check_level_compatibility(current, parent):
            print(f"Level incompatibility in path '{path}': '{current}' is not compatible with parent '{parent}'")
            return False
            
    return True

class PathFilter:
    def __init__(self):
        self.retained_paths = []  # 使用列表而不是集合，以保持顺序
    
    def check_node_quality(self, node: str, path: str) -> bool:
        """Check if a node meets quality requirements.
        
        Args:
            node: The node to check
            path: Full path for context
            
        Returns:
            bool: True if the node meets quality requirements
        """
        return check_attribute_quality(node, path)
    
    def check_node_compatibility(self, current: str, parent: str) -> bool:
        """Check if a node is compatible with its parent.
        
        Args:
            current: Current node
            parent: Parent node
            
        Returns:
            bool: True if the nodes are compatible
        """
        return check_level_compatibility(current, parent)
    
    def filter_tree(self, data: dict, current_path: str = "", pbar=None) -> dict:
        """Process the tree structure in two phases:
        1. First check path similarity within first-level groups
        2. Then process leaf nodes with quality and compatibility checks
        """
        # Phase 1: Group paths by first level and check similarity
        print("\nPhase 1: Checking path similarity...")
        paths = convert_tree_to_paths(data)
        paths.sort()
        paths = remove_duplicates(paths)
        
        # Group by first level
        paths_by_first_level = {}
        first_level_paths = []  # Store first level paths
        
        for path in paths:
            parts = path.split('.')
            if len(parts) <= 1:  # If it's a first level path
                first_level_paths.append(path)
                continue
            
            first_level = parts[0]
            if first_level not in paths_by_first_level:
                paths_by_first_level[first_level] = []
            paths_by_first_level[first_level].append(path)
        
        # Check similarity within each first level group
        filtered_paths = first_level_paths.copy()
        
        for first_level, group_paths in paths_by_first_level.items():
            group_filtered_paths = []
            for path in group_paths:
                if pbar:
                    pbar.update(1)
                
                # Check similarity only within the same first level group
                is_similar = False
                for retained_path in group_filtered_paths:
                    if check_path_similarity(path, retained_path):
                        is_similar = True
                        print(f"Found similar paths:\n- {path}\n- {retained_path}")
                        break
                
                if not is_similar:
                    group_filtered_paths.append(path)
                    print(f"Passed similarity check: {path}")
            
            filtered_paths.extend(group_filtered_paths)
        
        # Phase 2: Process leaf nodes
        print("\nPhase 2: Processing leaf nodes...")
        def process_leaf_nodes(tree_data: dict, path: str = "") -> dict:
            filtered = {}
            
            for key, value in list(tree_data.items()):
                new_path = f"{path}.{key}" if path else key
                
                # Skip if path was filtered out in Phase 1
                if new_path not in filtered_paths and new_path not in first_level_paths:
                    continue
                
                levels = new_path.split('.')
                
                # For non-first level nodes, check quality and compatibility
                if len(levels) > 1:
                    # Check current node's quality
                    if not self.check_node_quality(levels[-1], new_path):
                        continue
                    
                    # Check compatibility with parent
                    if not self.check_node_compatibility(levels[-1], levels[-2]):
                        continue
                
                # For leaf nodes
                if isinstance(value, dict) and not value:
                    filtered[key] = {}
                    self.retained_paths.append(new_path)
                    print(f"Retained leaf node: {new_path}")
                
                # For non-leaf nodes
                elif isinstance(value, dict):
                    filtered_children = process_leaf_nodes(value, new_path)
                    if filtered_children or len(levels) == 1:  # Keep if has children or is first level
                        filtered[key] = filtered_children
            
            return filtered
        
        return process_leaf_nodes(data)

def count_leaves(d: dict) -> int:
    """统计叶子节点数量"""
    count = 0
    for value in d.values():
        if isinstance(value, dict):
            if not value:  # 空字典表示叶子节点
                count += 1
            else:
                count += count_leaves(value)
    return count

def get_all_paths(d: dict, current_path: str = "") -> List[str]:
    """获取字典中所有的路径"""
    paths = []
    for key, value in d.items():
        new_path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            if not value:  # 叶子节点
                paths.append(new_path)
            else:
                paths.extend(get_all_paths(value, new_path))
    return paths

def main():
    print("开始处理...")
    input_file = "/home/zhou/persona/src/process_attributes_test/2.24/outputs/run_20250326_125810/attributes_merged.json"
    output_file = os.path.join(os.path.dirname(input_file), "filtered_attributes1.json")
    log_file = os.path.join(os.path.dirname(input_file), "filter_log1.txt")
    
    try:
        print("读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    original_leaves = count_leaves(data)
    all_paths = get_all_paths(data)
    print(f"开始过滤 {original_leaves} 个叶子节点...")
    
    # 创建日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"开始时间: {os.path.basename(input_file)}\n")
        f.write(f"原始叶子节点数: {original_leaves}\n\n")
    
    # 创建 PathFilter 实例并进行过滤
    path_filter = PathFilter()
    with tqdm(total=original_leaves, desc="Filtering nodes") as pbar:
        filtered_data = path_filter.filter_tree(data, pbar=pbar)
    
    filtered_leaves = count_leaves(filtered_data)
    filtered_paths = get_all_paths(filtered_data)
    removed_paths = set(all_paths) - set(filtered_paths)
    similar_paths = set(all_paths) - set(filtered_paths) - set(path_filter.retained_paths)

    try:
        print("\n保存结果...")
        # 保存过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
        # 追加日志信息
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n保留的路径:\n")
            for path in sorted(path_filter.retained_paths):
                f.write(f"+ {path}\n")
                
            f.write("\n删除的路径:\n")
            for path in sorted(removed_paths):
                if path in similar_paths:
                    f.write(f"- {path} (与其他路径相似)\n")
                else:
                    f.write(f"- {path} (不符合要求)\n")
                    
            f.write(f"\n统计信息:\n")
            f.write(f"- 原始叶子节点数: {original_leaves}\n")
            f.write(f"- 过滤后叶子节点数: {filtered_leaves}\n")
            f.write(f"- 删除的节点数: {original_leaves - filtered_leaves}\n")
            f.write(f"- 其中相似路径数: {len(similar_paths)}\n")
            
        print(f"完成! 结果已保存到: {output_file}")
        print(f"日志已保存到: {log_file}")
        print(f"\n统计信息:")
        print(f"- 原始叶子节点数: {original_leaves}")
        print(f"- 过滤后叶子节点数: {filtered_leaves}")
        print(f"- 删除的节点数: {original_leaves - filtered_leaves}")
        print(f"- 其中相似路径数: {len(similar_paths)}")
        print(f"- 总删除率: {((original_leaves - filtered_leaves) / original_leaves * 100):.2f}%")
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    main()