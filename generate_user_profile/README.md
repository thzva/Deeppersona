# User Profile Generator

A comprehensive Python-based system for generating realistic, diverse user profiles with rich attributes using GPT and vector-based attribute selection.

## 📋 Overview

This system generates detailed user profiles by:
1. Creating demographic information (age, gender, location, occupation)
2. Generating personal values, life attitudes, and stories
3. Selecting relevant attributes from a large attribute database using semantic similarity
4. Producing complete user profiles with personalized content

## 🚀 Features

- **Demographic Generation**: Realistic age, gender, location, and career information
- **Psychological Profiling**: Personal values, life attitudes, and coping mechanisms
- **Story Generation**: Contextual personal stories based on demographic and psychological attributes
- **Intelligent Attribute Selection**: Vector-based semantic search for relevant attributes
- **Scalable**: Batch generation of multiple profiles
- **Customizable**: Configurable attribute counts and generation parameters

## 📁 Project Structure

```
generate_user_profile/
├── config.py              # API configuration and utility functions
├── based_data.py          # Core data generation functions
├── select_attributes.py   # Attribute selection using vector search
├── generate_profile.py    # Main profile generation orchestrator
├── output/                # Generated profiles output directory
└── README.md             # This file
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
cd generate_user_profile
```

2. **Install dependencies**
```bash
pip install openai sentence-transformers scikit-learn numpy tqdm geonamescache
```

3. **Configure API keys**

Edit `config.py` and add your OpenAI API key:
```python
OPENAI_API_KEY = "your-api-key-here"
```

4. **Prepare data files**

Ensure the following data files are in the correct locations:
- `../data/occupations_english.json` - List of occupations
- Attribute database (JSON format)
- Vector embeddings database (pickle format)

## 💻 Usage

### Basic Usage

Generate a single user profile:

```python
from select_attributes import generate_user_profile, get_selected_attributes

# Generate basic user information
user_profile = generate_user_profile()

# Select relevant attributes
selected_attributes = get_selected_attributes(user_profile, attribute_count=200)

print(f"Generated profile: {user_profile}")
print(f"Selected {len(selected_attributes)} attributes")
```

### Batch Generation

Generate multiple profiles:

```python
from generate_profile import generate_profiles_batch

# Generate 10 profiles
generate_profiles_batch(num_profiles=10, attribute_count=200)
```

### Command Line Usage

```bash
# Generate profiles with custom parameters
python generate_profile.py --num-profiles 50 --attribute-count 150
```

## 📝 Module Descriptions

### `config.py`
Configuration module managing API keys, proxy settings, and OpenAI client initialization. Includes utility functions for API calls and JSON response parsing.

### `based_data.py`
Core data generation module containing functions for:
- Age and demographic information generation
- Career and occupation selection
- Geographic location generation
- Personal values and life attitude generation
- Personal story creation
- Interests and hobbies inference

### `select_attributes.py`
Attribute selection system using:
- Vector embeddings for semantic similarity
- GPT for intelligent attribute filtering
- Multi-stage selection process (near, mid, far neighbors)
- Diversity-based filtering

### `generate_profile.py`
Main orchestration module for:
- Batch profile generation
- File management and output organization
- Profile validation and quality checks
- Summary generation
