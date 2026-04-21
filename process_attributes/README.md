<h1 align="center">🌳 Taxonomy Processing Pipeline</h1>

<p align="center">
  <em>Stage 1 of DeepPersona — mine real user–ChatGPT dialogues into a hierarchical human-attribute taxonomy with 8,000+ attributes.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.8-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/taxonomy-8,000+_attributes-06B6D4" alt="Taxonomy">
  <img src="https://img.shields.io/badge/similarity_threshold-0.85-8B5CF6" alt="Threshold">
  <img src="https://img.shields.io/badge/output-JSON_|_TXT-F59E0B" alt="Output">
</p>

This module is the **taxonomy construction pipeline**. It extracts personalized attributes from Q&A pairs, validates and merges them, runs dual-phase quality checks, and exports both hierarchical JSON and flat `X.Y.Z` paths.

---

## 📁 Layout

```
process_attributes/
├── extract_personalized_attributes.py   # 1️⃣ Extract attrs from Q&A
├── filter_personalized_attributes.py    # 2️⃣ Validate top-level + leaf quality
├── merge_tree.py                        # 3️⃣ Merge multi-source trees
├── check_leaves.py                      # 4️⃣ Semantic + GPT leaf validation
├── convert_to_X.Y.Z.py                  # 5️⃣ Flat path / tree visualization
├── process_attributes.py                # 🧹 Deduplication utility
└── template.json                        # 📐 Reference taxonomy skeleton
```

---

## 🔄 Pipeline

```
┌────────────────────────────────────────────┐
│  1. Extract Attributes                      │
│     Q&A + reason → X.Y.Z paths              │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│  2. Filter Attributes                       │
│     Top-level + last-segment validation     │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│  3. Merge Trees                             │
│     Combine + resolve conflicts             │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│  4. Leaf Quality Check                      │
│     Phase A: semantic similarity            │
│     Phase B: GPT-4 validation               │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│  5. Convert & Export                        │
│     Hierarchical JSON · X.Y.Z · tree TXT    │
└────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Install

```bash
pip install openai sentence-transformers scikit-learn numpy tqdm
```

### Configure

Set your API key in each script (or centralize via env var):

```python
OPENAI_API_KEY = "sk-..."
```

---

## 🧩 Stages

### 1️⃣ Extract Personalized Attributes

```python
from extract_personalized_attributes import PersonalizedAttributeExtractor

extractor = PersonalizedAttributeExtractor()
result = extractor.extract_attributes(
    question="What are some good restaurants nearby?",
    reason="User's location and food preferences affect recommendations",
)
print(result["attributes"])
# ['Location.Current Location.City', 'Preferences.Food.Cuisine Type', ...]
```

- `X.Y.Z` format (3-level hierarchy)
- Validates against predefined top-level categories
- Tolerant to markdown-wrapped JSON responses

### 2️⃣ Filter & Validate

```python
from filter_personalized_attributes import PersonalizedAttributeAnalyzer

analyzer = PersonalizedAttributeAnalyzer()
```

| Rule | Keep | Drop |
|------|------|------|
| General categories | ✅ `Skills`, `Preferences`, `Background` | ❌ — |
| Broad aspects | ✅ `Style`, `Pattern`, `Approach` | ❌ — |
| Specific instances | ❌ | `Python`, `Google`, `New York` |
| Concrete values | ❌ | `5 years`, `Level 3` |

### 3️⃣ Merge Trees

```python
from merge_tree import merge_trees

merged = merge_trees([tree1, tree2, tree3])
```

Preserves hierarchy, resolves overlaps, maintains stable ordering, writes timestamped outputs.

### 4️⃣ Leaf Quality Check

```python
from check_leaves import PathFilter

pf = PathFilter()
clean = pf.filter_tree(data)
```

**Phase A — Similarity**
- Group paths by top-level category
- Embed with `all-MiniLM-L6-v2`
- Drop intra-category duplicates (cosine ≥ 0.85)

**Phase B — Quality**
- GPT-4 validates leaf node semantics
- Checks parent-child compatibility
- Enforces user-centric, general phrasing

### 5️⃣ Convert to Path Notation

```python
from convert_to_X.Y.Z import extract_paths, generate_tree_text

paths = extract_paths(data)
# ['user_preferences.food.cuisine_type', 'location.current.city', ...]

tree = generate_tree_text(parent_child_map)   # ASCII tree
```

**Output formats**
- `paths.json` — flat `X.Y.Z` strings
- `tree.txt` — indented tree visualization

### 🧹 Deduplicate

```bash
python process_attributes.py
```

---

## 🔗 Related

- 🧬 Downstream persona generation — [`../generate_user_profile/`](../generate_user_profile/)
- 📦 Dataset — [🤗 `THzva/deeppersona_dataset`](https://huggingface.co/datasets/THzva/deeppersona_dataset)
- 🌐 Persona simulator — [deeppersona-sim.zhou-yufan.com/interaction](https://deeppersona-sim.zhou-yufan.com/interaction/)
