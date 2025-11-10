import json
from openai import OpenAI
import os
from typing import Dict, List, Tuple

# OpenAI API Settings
OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_MODEL = "gpt-4o"

def check_last_segment(client: OpenAI, segment: str) -> bool:
    """检查属性的最后一段是否符合描述要求"""
    try:
        prompt = f"""Determine if this segment represents a general category or aspect rather than a specific instance.

Consider it VALID (true) if it describes:
1. A general category or classification (e.g., 'Role', 'Type', 'Level', 'Category')
2. A broad aspect or dimension (e.g., 'Style', 'Pattern', 'Approach')
3. A general capability or trait (e.g., 'Skills', 'Knowledge', 'Experience')
4. A characteristic or attribute (e.g., 'Status', 'Background', 'Identity')
5. An area or domain

Consider it INVALID (false) if it is:
1. A specific instance or example (e.g., 'Python', 'Manager', 'Sales')
2. A concrete value or measurement (e.g., '5 years', 'Level 3')
3. A specific organization or location (e.g., 'Google', 'New York')
4. A proper noun or named entity

Segment: {segment}
Return only: true or false"""

        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing attribute descriptions."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = completion.choices[0].message.content.strip().lower() == 'true'
        print(f"最后一段 '{segment}' - {'符合' if result else '不符合'}描述要求")
        return result
        
    except Exception as e:
        print(f"检查最后一段时出错: {str(e)}")
        return False

class PersonalizedAttributeAnalyzer:
    def __init__(self):
        # 读取模板文件中的一级属性
        with open('/home/zhou/persona/dataset/2.7/template.json', 'r') as f:
            template = json.load(f)
            self.valid_categories = list(template['persona_categories'])
    
    def check_top_level_category(self, client: OpenAI, attribute: str) -> str:
        """使用GPT检查并修正一级属性的归类"""
        segments = attribute.split('.')
        top_level = segments[0]
        rest = segments[1:]
        
        if top_level not in self.valid_categories and len(rest) > 0:
            # 如果一级属性不在模板中，使用GPT判断
            prompt = f"""Given an attribute path and a list of valid top-level categories, determine which category this attribute should belong to.

Attribute path: {attribute}

Valid categories:
{chr(10).join('- ' + cat for cat in self.valid_categories)}

Consider the meaning and context of the attribute. For example:
- Business-related attributes usually belong to 'Career and Work Identity'
- Communication-related attributes usually belong to 'Relationships and Social Networks'
- Learning-related attributes usually belong to 'Education and Learning'

Return only the category name, exactly as shown in the list above."""

            completion = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in categorizing personal attributes."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            new_top = completion.choices[0].message.content.strip()
            if new_top in self.valid_categories and new_top != top_level:
                new_attr = f"{new_top}.{'.'.join(rest)}"
                print(f"属性重分类: {attribute} -> {new_attr}")
                return new_attr
        
        return attribute

def check_if_personalized(client: OpenAI, attribute: str) -> bool:
    """检查属性是否为中性化属性"""
    prompt = f"""Determine if the following attribute meets the personalization criteria.
Standards for evaluation:

1. User-Centric Focus:
  - Must describe personal characteristics/attributes
  - Remove business/marketing terms
  - Remove metrics/objectives/adjective
2. Check each level:
  - Must be general category (no specific instances, behaviors, or values)
  - Must logically refine parent level
3. Attributes must be highly general, enabling GPT to generate rich content for that attribute

Attribute: {attribute}
Return only: true or false"""

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in determining if attributes reflect personal characteristics."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip().lower() == 'true'

def check_category(client: OpenAI, attribute: str) -> str:
    """检查并修正属性的一级分类"""
    prompt = f"""Given an attribute path and a list of valid categories, determine the most appropriate category for this attribute.

Attribute: {attribute}

Valid categories:
- Demographic Information (for attributes about age, gender, family, ethnicity)
- Physical and Health Characteristics (for attributes about physical features, health)
- Psychological and Cognitive Aspects (for attributes about thinking, personality)
- Cultural and Social Context (for attributes about cultural background)
- Relationships and Social Networks (for attributes about relationships)
- Career and Work Identity (for attributes about work, profession)
- Education and Learning (for attributes about education)
- Hobbies, Interests, and Lifestyle (for attributes about interests)
- Lifestyle and Daily Routine (for attributes about daily habits)
- Core Values, Beliefs, and Philosophy (for attributes about values)
- Emotional and Relational Skills (for attributes about emotional intelligence)
- Media Consumption and Engagement (for attributes about media habits)

Analyze the attribute's meaning and return the most appropriate category name from the list above. Return ONLY the category name, exactly as shown."""

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in categorizing personal attributes."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def analyze_single_attribute(client: OpenAI, attribute: str) -> Tuple[str, bool]:
    """分析单个属性，返回处理后的属性和是否为个性化属性"""
    try:
        segments = attribute.split('.')
        original_attribute = attribute
        print(f"\n开始分析属性: {attribute}")
        
        while len(segments) > 1:
            # 检查并修正一级分类
            suggested_category = check_category(client, attribute)
            if suggested_category != segments[0]:
                segments[0] = suggested_category
                attribute = '.'.join(segments)
                print(f"修正属性分类: {original_attribute} -> {attribute}")
                original_attribute = attribute
            
            # 先检查最后一段是否符合要求
            last_segment = segments[-1]
            is_valid_segment = check_last_segment(client, last_segment)
            print(f"最后一段 '{last_segment}' - {'符合' if is_valid_segment else '不符合'}描述要求")
            
            if not is_valid_segment:
                # 如果最后一段不符合要求，删除它
                segments = segments[:-1]
                attribute = '.'.join(segments)
                print(f"删除不符合要求的最后一段，新属性: {attribute}")
                continue
            
            # 然后检查是否为个性化属性
            is_personalized = check_if_personalized(client, attribute)
            print(f"属性 '{attribute}' - {'个性化' if is_personalized else '非个性化'}属性")
            if not is_personalized:
                # 如果最后一段符合要求但整个属性不是个性化的，直接返回非个性化
                return attribute, False
            
            # 如果是个性化属性，返回结果
            return attribute, True
        
        # 如果只剩下一级属性，再检查一次
        is_personalized = check_if_personalized(client, attribute)
        print(f"属性 '{attribute}' - {'个性化' if is_personalized else '非个性化'}属性")
        return attribute, is_personalized
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return attribute, False

        # 然后检查是否为个性化属性
        prompt = f"""Determine if this attribute path describes an individual's characteristics.

Consider it PERSONAL if it's about:
1. Demographics and identity:
   - Gender, age, family status
   - Cultural background
   - Personal identity aspects

2. Individual characteristics:
   - Skills and capabilities
   - Preferences and interests
   - Experiences and background
   - Communication and learning styles
   - Decision-making patterns

3. Personal context:
   - Family composition
   - Professional background
   - Educational history

Consider it NOT PERSONAL only if it's about:
1. External systems or organizations
2. Historical or cultural events
3. General facts or concepts that don't vary by individual

Attribute: {attribute}
Return only: true or false"""


        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing personalized attributes."},
                {"role": "user", "content": prompt}
            ]
        )
        
        is_personalized = completion.choices[0].message.content.strip().lower() == 'true'
        print(f"属性 '{attribute}' - {'个性化' if is_personalized else '非个性化'}属性")
        
        if attribute != original_attribute:
            print(f"属性被修改: {original_attribute} -> {attribute}")
            
        return attribute, is_personalized
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return attribute, False

def save_results(result: Dict[str, List[str]], output_file: str):
    """保存结果到文件"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"分类结果: 个性化({len(result['personalized_attributes'])}), 非个性化({len(result['impersonalized_attributes'])})")
    except Exception as e:
        print(f"错误: {str(e)}")



def process_attributes(input_file: str, output_file: str):
    """处理所有属性"""
    try:
        # 读取属性列表
        with open(input_file, 'r', encoding='utf-8') as f:
            attributes = json.load(f)
        print(f"开始处理 {len(attributes)} 个属性")
        
        # 初始化分析器和 OpenAI 客户端
        analyzer = PersonalizedAttributeAnalyzer()
        client = OpenAI(api_key=OPENAI_API_KEY)
        result = {
            "personalized_attributes": [],
            "impersonalized_attributes": []
        }
        
        # 逐个处理属性
        for i, attr in enumerate(attributes, 1):
            print(f"[{i}/{len(attributes)}]")
            # 首先检查并修正一级属性
            attr = analyzer.check_top_level_category(client, attr)
            processed_attr, is_personalized = analyze_single_attribute(client, attr)
            if is_personalized:
                result["personalized_attributes"].append(processed_attr)
            else:
                result["impersonalized_attributes"].append(processed_attr)
            save_results(result, output_file)
            
        print("处理完成")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

def main():
    input_file = "PATH.json"
    output_file = "PATH.json"
    
    process_attributes(input_file, output_file)

if __name__ == "__main__":
    main()