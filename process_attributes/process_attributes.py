import json

# 读取JSON文件
with open('PATH.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取personalized_attributes并去重
personalized_attributes = list(set(data['personalized_attributes']))

# 按字母顺序排序
personalized_attributes.sort()

# 打印结果
print(f"去重前的属性数量: {len(data['personalized_attributes'])}")
print(f"去重后的属性数量: {len(personalized_attributes)}")
print("\n去重后的属性列表:")
for attr in personalized_attributes:
    print(attr)
