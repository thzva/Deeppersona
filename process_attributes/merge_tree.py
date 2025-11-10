import json
import os
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_MODEL = "gpt-4o"



class OutputManager:
    def __init__(self):
        """Initialize output manager with outputs directory in the current script's directory"""
        # 获取当前脚本文件的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建outputs目录
        self.outputs_dir = os.path.join(script_dir, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # 创建时间戳目录
        self.output_dir = os.path.join(self.outputs_dir, f"run_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
    
    def get_output_path(self, filename: str) -> str:
        """Get full path for output file"""
        return os.path.join(self.output_dir, filename)

@dataclass
class TreeNode:
    value: str
    children: Dict[str, 'TreeNode']
    level: int
    original_path: str
    
    def __init__(self, value: str, level: int, original_path: str = ""):
        self.value = value
        self.children = {}
        self.level = level
        self.original_path = original_path
        
    def __hash__(self):
        return hash(self.original_path)
        
    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.original_path == other.original_path

def json_to_tree(json_data: Dict, current_key: str = "root", level: int = 0, path: str = "") -> TreeNode:
    """Convert JSON data to TreeNode object"""
    node = TreeNode(value=current_key, level=level, original_path=path)
    
    if isinstance(json_data, dict):
        for key, value in sorted(json_data.items()):
            new_path = f"{path}.{key}" if path else key
            child_node = json_to_tree(value, key, level + 1, new_path)
            node.children[key] = child_node
    
    return node

def tree_to_json(node: TreeNode) -> Dict:
    """Convert TreeNode back to JSON format with consistent structure"""
    result = {}
    
    # Sort children by key to maintain consistent order
    for key, child in sorted(node.children.items()):
        result[key] = tree_to_json(child)
        
    return result

def get_nodes_at_level(root: TreeNode, target_level: int) -> List[TreeNode]:
    """Get all nodes at a specific level"""
    if target_level == root.level:
        return [root]
    nodes = []
    for child in root.children.values():
        nodes.extend(get_nodes_at_level(child, target_level))
    return nodes

def validate_gpt_response(response_text: str) -> Dict[str, str]:
    """Validate and clean up GPT response to ensure it's valid JSON"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse GPT response: {response_text}")
            print(f"Error: {str(e)}")
            return {}

def process_merge_response(mapping: Dict[str, str], nodes_to_merge: List[TreeNode]) -> Dict[str, TreeNode]:
    """Process merge mapping and create merged nodes"""
    if not mapping:
        return {node.value: node for node in nodes_to_merge}
        
    print(f"Processing mapping: {mapping}")
    
    new_nodes = {}
    for old_node in nodes_to_merge:
        new_value = mapping.get(old_node.value, old_node.value)
        
        if new_value not in new_nodes:
            if new_value == old_node.value:
                new_nodes[new_value] = old_node
            else:
                new_node = TreeNode(
                    new_value,
                    old_node.level,
                    new_value
                )
                new_nodes[new_value] = new_node
        
        if new_value != old_node.value:
            new_nodes[new_value].children.update(old_node.children)
    
    return new_nodes

def merge_level_nodes(nodes: List[TreeNode], level: int, client: OpenAI) -> Dict[str, TreeNode]:
    """Merge nodes at the same level with personalization and abstraction requirements"""
    if len(nodes) <= 1:
        return {node.value: node for node in nodes}

    try:
        prompt = f"""You are an expert in analyzing and organizing hierarchical data structures.
Your task is to analyze nodes at the same level and suggest merges based on semantic similarity.
Return ONLY a JSON dictionary mapping current node names to new names, nothing else.

Current nodes at level {level}: {[n.value for n in nodes]}

Merging Strategy:
1. Primary Goal: Merge semantically similar attributes

2 Similarity Thresholds:
   - If nodes share core concept/purpose (>80% similar): Directly merge
   - If completely different (<80% similar): Keep separate


STRICT REQUIREMENTS:
1. User-Centric Focus:
   - Must be user personalization attributes that reflect individual characteristics/attributes
  
2. Must be general category (no specific instances, behaviors, or values)

3. Must logically refine parent level
  
4. Attributes must be highly general, enabling GPT to generate rich content for that attribute


"""

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-only response bot. Return valid JSON dictionaries only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        mapping = validate_gpt_response(response.choices[0].message.content.strip())
        return process_merge_response(mapping, nodes)
        
    except Exception as e:
        print(f"Error in merge_level_nodes at level {level}: {str(e)}")
        return {node.value: node for node in nodes}

def validate_parent_attribute(node: TreeNode, client: OpenAI) -> str:
    """Validate and potentially update parent attribute name to ensure it meets requirements"""
    if not node.original_path or '.' not in node.original_path:
        return node.value
        
    parent_path = '.'.join(node.original_path.split('.')[:-1])
    
    prompt = f"""Analyze this attribute name and determine if it meets these requirements:
1. User-Centric Focus:
   - Must be user personalization attributes that reflect individual characteristics/attributes
  
2. Check each level:
  - Must be general category (no specific instances, behaviors, or values)
  - Must logically refine parent level

3. Clarity and Meaningfulness:
   - MUST be clear and meaningful in describing personality traits


Attribute to analyze: {parent_path}

If the attribute meets ALL requirements, respond with: KEEP:{parent_path}
If it doesn't meet requirements, suggest a new name that does meet ALL requirements with: CHANGE:new_name

Provide ONLY the response in the format above, no other text."""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a direct response bot. Respond only with KEEP: or CHANGE: followed by the attribute name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.startswith('KEEP:'):
            return parent_path
        elif result.startswith('CHANGE:'):
            new_name = result.split(':', 1)[1].strip()
            return new_name
        else:
            print(f"Unexpected response format: {result}")
            return parent_path
            
    except Exception as e:
        print(f"Error in validate_parent_attribute: {str(e)}")
        return parent_path

def process_tree_level_by_level(root: TreeNode, client: OpenAI, output_manager: OutputManager) -> None:
    """Process the tree level by level while preserving first level attributes"""
    max_level = 4
    
    # First, restructure the tree to separate content before and after first dot
    for first_level_key, first_level_node in list(root.children.items()):
        # Process each first level node's children
        new_children = {}
        for child_key, child_node in first_level_node.children.items():
            if '.' in child_key:
                # Split at first dot and restructure
                prefix, rest = child_key.split('.', 1)
                if prefix == first_level_key:
                    # If prefix matches parent, just use the rest
                    child_node.value = rest
                    child_node.original_path = f"{first_level_key}.{rest}"
                    new_children[rest] = child_node
            else:
                # Keep nodes without dots unchanged
                new_children[child_key] = child_node
        first_level_node.children = new_children
    
    # Save the initial state after restructuring
    save_intermediate_results(root, 1, output_manager)
    
    # 处理Y层（第2层）
    print("\n处理Y层...")
    for first_level_key, first_level_node in root.children.items():
        if first_level_node.children:
            print(f"\n处理 '{first_level_key}' 下的Y层节点...")
            new_nodes = merge_level_nodes(list(first_level_node.children.values()), 2, client)
            first_level_node.children = new_nodes
            
            # Validate parent attributes for merged nodes
            for node in new_nodes.values():
                new_parent = validate_parent_attribute(node, client)
                if new_parent != first_level_key:
                    node.original_path = f"{new_parent}.{node.value}"
    
    save_intermediate_results(root, 2, output_manager)
    
    # 处理Z层（第3层）
    print("\n处理Z层...")
    for x_node in root.children.values():
        for y_key, y_node in list(x_node.children.items()):
            if y_node.children:
                print(f"\n处理 '{x_node.value}.{y_key}' 下的Z层节点...")
                new_nodes = merge_level_nodes(list(y_node.children.values()), 3, client)
                y_node.children = new_nodes
                
                # Validate parent attributes for merged nodes
                parent_path = f"{x_node.value}.{y_key}"
                for node in new_nodes.values():
                    new_parent = validate_parent_attribute(node, client)
                    if new_parent != parent_path:
                        node.original_path = f"{new_parent}.{node.value}"
    
    save_intermediate_results(root, 3, output_manager)
    
    # 检查并处理第4层
    print("\n检查第4层...")
    has_fourth_level = False
    for x_node in root.children.values():
        for y_node in x_node.children.values():
            for z_key, z_node in list(y_node.children.items()):
                if z_node.children:
                    has_fourth_level = True
                    break
            if has_fourth_level:
                break
        if has_fourth_level:
            break
    
    if has_fourth_level:
        print("发现第4层节点，开始处理...")
        for x_node in root.children.values():
            for y_node in x_node.children.values():
                for z_key, z_node in list(y_node.children.items()):
                    if z_node.children:
                        print(f"\n处理 '{x_node.value}.{y_node.value}.{z_key}' 下的节点...")
                        new_nodes = merge_level_nodes(list(z_node.children.values()), 4, client)
                        z_node.children = new_nodes
        save_intermediate_results(root, 4, output_manager)
        print("第4层处理完成")
    else:
        print("未发现第4层节点，跳过处理")
    
    print("\n全部处理完成")

def save_intermediate_results(root: TreeNode, level: int, output_manager: OutputManager) -> None:
    """Save intermediate results in the same format as input JSON"""
    output_path = output_manager.get_output_path(f"attributes_level_{level}.json")
    json_data = tree_to_json(root)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved level {level} results to {output_path}")

def build_simple_tree_structure(attributes: list) -> dict:
    """
    将点分隔的属性列表转换为简单的嵌套树结构
    
    Args:
        attributes: 点分隔的属性列表，如 ["X.Y.Z", "X.Y2.Z2"]
        
    Returns:
        dict: 嵌套的树结构字典
    """
    # 构建树结构
    tree = {}
    for attr in attributes:
        parts = attr.split('.')
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    
    return tree

def main():
    input_file = "PATH.json"
    output_manager = OutputManager()
    
    try:
        # 读取属性列表并构建树结构
        print("Building and optimizing tree structure...")
        with open(input_file, 'r', encoding='utf-8') as f:
            attributes = json.load(f)
        
        # 构建初始树结构
        tree = build_simple_tree_structure(attributes)
        
        # 保存初始树结构
        initial_tree_path = output_manager.get_output_path("initial_tree.json")
        with open(initial_tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        print(f"Saved initial tree structure to {initial_tree_path}")
        
        # 转换为TreeNode结构
        print("Converting to TreeNode structure...")
        root = TreeNode(value="root", level=0)
        for key, value in sorted(tree.items()):
            child_node = json_to_tree(value, key, level=1, path=key)
            root.children[key] = child_node
        print(f"Successfully loaded tree with {len(root.children)} top-level nodes")
        
    except Exception as e:
        print(f"Error building tree structure: {str(e)}")
        return
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 开始合并过程
    print("\nStarting tree merge process...")
    process_tree_level_by_level(root, client, output_manager)
    
    # 保存最终结果
    print("Saving final results...")
    final_json = tree_to_json(root)
    output_path = output_manager.get_output_path("attributes_merged.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"Final merged attributes saved to {output_path}")


if __name__ == "__main__":
    main()