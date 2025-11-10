# Attribute Processing Pipeline

A comprehensive toolkit for processing, filtering, and organizing user attributes in a hierarchical structure using GPT and semantic similarity analysis.

## 📋 Overview

This pipeline processes user attributes through multiple stages:
1. **Extraction**: Extract personalized attributes from questions and reasons
2. **Filtering**: Validate and filter attributes based on quality criteria
3. **Merging**: Merge multiple attribute trees into a unified structure
4. **Quality Check**: Validate leaf nodes using semantic similarity and GPT
5. **Conversion**: Convert hierarchical structures to path notation (X.Y.Z format)

## 🚀 Features

- **Intelligent Extraction**: GPT powered attribute extraction from natural language
- **Quality Validation**: Multi-level validation using both rule-based and AI-based checks
- **Semantic Similarity**: Detect and remove similar attributes using sentence transformers
- **Hierarchical Processing**: Maintain and validate parent-child relationships
- **Tree Merging**: Combine multiple attribute sources into a unified taxonomy
- **Flexible Output**: Support both nested JSON and flat path formats

## 📁 Project Structure

```
process_attributes/
├── extract_personalized_attributes.py   # Extract attributes from questions
├── filter_personalized_attributes.py    # Filter and validate attributes
├── merge_tree.py                        # Merge multiple attribute trees
├── check_leaves.py                      # Validate leaf nodes quality
├── convert_to_X.Y.Z.py                  # Convert to path notation
├── process_attributes.py                # Utility for deduplication
└── README.md                            # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/thzva/Deeppersona.git
cd process_attributes
```

2. **Install dependencies**
```bash
pip install openai sentence-transformers scikit-learn numpy tqdm
```

3. **Configure API keys**

Update the `OPENAI_API_KEY` in each script:
```python
OPENAI_API_KEY = "your-api-key-here"
```

## 💻 Usage

### 1. Extract Personalized Attributes

Extract attributes from questions and personalization reasons:

```python
from extract_personalized_attributes import PersonalizedAttributeExtractor

extractor = PersonalizedAttributeExtractor()
result = extractor.extract_attributes(
    question="What are some good restaurants nearby?",
    reason="User's location and food preferences affect recommendations"
)
print(result['attributes'])
# Output: ['Location.Current Location.City', 'Preferences.Food.Cuisine Type', ...]
```

**Key Features:**
- Extracts attributes in X.Y.Z format (3-level hierarchy)
- Validates against predefined top-level categories
- Handles markdown-formatted JSON responses

### 2. Filter and Validate Attributes

Filter attributes based on quality criteria:

```python
from filter_personalized_attributes import PersonalizedAttributeAnalyzer

analyzer = PersonalizedAttributeAnalyzer()
# Validates top-level categories
# Checks if last segment is a general category (not specific instance)
# Ensures attributes meet quality standards
```

**Validation Rules:**
- ✅ General categories (e.g., 'Skills', 'Preferences', 'Background')
- ✅ Broad aspects (e.g., 'Style', 'Pattern', 'Approach')
- ❌ Specific instances (e.g., 'Python', 'Google', 'New York')
- ❌ Concrete values (e.g., '5 years', 'Level 3')

### 3. Merge Attribute Trees

Combine multiple attribute sources:

```python
from merge_tree import merge_trees

# Merge multiple JSON files
merged_tree = merge_trees([tree1, tree2, tree3])
```

**Features:**
- Preserves hierarchical structure
- Handles conflicts intelligently
- Maintains consistent ordering
- Generates timestamped outputs

### 4. Check Leaf Node Quality

Validate leaf nodes using semantic similarity and GPT-4:

```python
from check_leaves import PathFilter

filter = PathFilter()
filtered_data = filter.filter_tree(data)
```

**Two-Phase Filtering:**

**Phase 1: Similarity Check**
- Groups paths by first-level category
- Uses sentence transformers (all-MiniLM-L6-v2)
- Removes similar paths within same category (threshold: 0.85)

**Phase 2: Quality Validation**
- Validates leaf node quality using GPT
- Checks parent-child compatibility
- Ensures attributes are user-centric and general

### 5. Convert to Path Notation

Convert hierarchical structure to flat path list:

```python
from convert_to_X.Y.Z import extract_paths, generate_tree_text

# Extract all paths
paths = extract_paths(data)
# Output: ['user_preferences.food.cuisine_type', 'location.current.city', ...]

# Generate tree visualization
tree_text = generate_tree_text(parent_child_map)
```

**Output Formats:**
- JSON: List of paths in X.Y.Z format
- TXT: Tree visualization with indentation

### 6. Deduplicate Attributes

Remove duplicate attributes and sort:

```python
# Using process_attributes.py
python process_attributes.py
```

## 📊 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. Extract Attributes                                       │
│     - Parse questions and reasons                           │
│     - Generate X.Y.Z format attributes                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Filter Attributes                                        │
│     - Validate top-level categories                         │
│     - Check last segment quality                            │
│     - Remove invalid attributes                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Merge Trees                                              │
│     - Combine multiple sources                              │
│     - Resolve conflicts                                     │
│     - Build unified taxonomy                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Check Leaf Quality                                       │
│     - Phase 1: Semantic similarity check                    │
│     - Phase 2: GPT-4 quality validation                     │
│     - Remove low-quality nodes                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Convert & Export                                         │
│     - Generate path notation                                │
│     - Create tree visualization                             │
│     - Export to JSON/TXT                                    │
└─────────────────────────────────────────────────────────────┘
```