<h1 align="center">🧬 Persona Generation Engine</h1>

<p align="center">
  <em>Stage 2 of DeepPersona — progressively sample the taxonomy to produce deep, coherent personas.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.8-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/attributes-200+_per_profile-10B981" alt="Attributes">
  <img src="https://img.shields.io/badge/selector-vector_search-8B5CF6" alt="Selector">
  <img src="https://img.shields.io/badge/output-JSON-F59E0B" alt="Output">
</p>

This module is the **persona generation engine**. Given the 4,676-node taxonomy from [`process_attributes/`](../process_attributes/), it anchors a demographic and psychological core, then progressively samples attributes while the LLM fills each node conditioned on the evolving profile.

---

## ✨ What It Does

| Stage | Description |
|-------|-------------|
| **1. Demographic anchor** | Age, gender, geographic location (GeoNames), occupation |
| **2. Psychological anchor** | Personal values, life attitude, coping mechanisms |
| **3. Narrative anchor** | Life story coherent with demographic + psychological core |
| **4. Attribute selection** | Vector-based search over the taxonomy with multi-stage filtering |
| **5. Value generation** | GPT fills each selected node conditioned on the evolving profile |

The anchor-first design avoids majority-culture defaults; the **stochastic breadth-first selector** biases toward long-tail branches for diversity.

---

## 📁 Layout

```
generate_user_profile/
├── config.py              # ⚙️  API client, keys, proxy, JSON helpers
├── based_data.py          # 🧱  Demographic / psychological / narrative core
├── select_attributes.py   # 🔍  Vector-search attribute selector
├── generate_profile.py    # 🚀  Batch orchestrator + CLI
└── output/                # 📂  Generated profiles
```

---

## 🚀 Quick Start

### 1. Install

```bash
pip install openai sentence-transformers scikit-learn numpy tqdm geonamescache
```

### 2. Configure

Edit [`config.py`](config.py):

```python
OPENAI_API_KEY = "sk-..."
```

Data files expected (relative to this directory):

| Path | Purpose |
|------|---------|
| `../data/occupations_english.json` | Occupation anchor list |
| `../data/attributes_merged.json` | Taxonomy tree |
| `../data/attribute_embeddings.pkl` | Sentence-transformer embeddings |

### 3. Generate One Persona

```python
from select_attributes import generate_user_profile, get_selected_attributes

user_profile = generate_user_profile()                         # anchor core
selected     = get_selected_attributes(user_profile,
                                       attribute_count=200)    # progressive sampling

print(user_profile)
print(f"Selected {len(selected)} attributes")
```

### 4. Batch via CLI

```bash
python generate_profile.py --num-profiles 50 --attribute-count 150
```

---

## 🧩 Modules

### `config.py`
API client initialization, OpenAI key / proxy setup, robust JSON response parser.

### `based_data.py`
Anchor core generation:

- Age & demographic sampling
- Occupation selection (from `occupations_english.json`)
- Geographic location (GeoNames-backed)
- Personal values, life attitude, coping mechanisms
- Narrative life story conditioned on the above
- Interests & hobbies inference

### `select_attributes.py`
Attribute selection pipeline:

- Sentence-transformer embeddings for semantic similarity
- GPT-powered filtering for relevance
- **Multi-stage neighborhood sampling** — near / mid / far to balance focus and diversity
- Final diversity-aware filtering to trim redundant nodes

### `generate_profile.py`
Batch orchestration:

- Parallel profile generation
- Output file naming & incremental saving
- Profile-level quality checks
- Run summary

---

## 🔗 Related

- 🌳 Upstream taxonomy construction — [`../process_attributes/`](../process_attributes/)
- 📦 Dataset — [🤗 `THzva/deeppersona_dataset`](https://huggingface.co/datasets/THzva/deeppersona_dataset)
- 🌐 Persona simulator — [deeppersona-sim.zhou-yufan.com/interaction](https://deeppersona-sim.zhou-yufan.com/interaction/)
