# DeepPersona

A comprehensive system for generating realistic, diverse user profiles with rich personalized attributes using GPT and semantic similarity analysis.

🌐 **[Project Homepage](https://thzva.github.io/deeppersona.github.io/)** | 📊 **[Dataset](https://huggingface.co/datasets/THzva/deeppersona_dataset)** | 🚀 **[Demo](https://huggingface.co/spaces/THzva/deeppersona-experience)**

## 📋 Overview

DeepPersona consists of two main components:

1. **User Profile Generator** - Creates detailed user profiles with demographic, psychological, and behavioral attributes
2. **Attribute Processing Pipeline** - Processes, filters, and organizes user attributes in a hierarchical structure

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/thzva/Deeppersona.git
cd Deeppersona

# Install dependencies
pip install openai sentence-transformers scikit-learn numpy tqdm geonamescache

# Configure your OpenAI API key in config files
```

## 📁 Project Structure

```
Deeppersona/
├── generate_user_profile/     # User profile generation system
│   ├── config.py              # API configuration
│   ├── based_data.py          # Core data generation
│   ├── select_attributes.py   # Attribute selection
│   ├── generate_profile.py    # Profile orchestrator
│   └── README.md              # Detailed documentation
│
├── process_attributes/        # Attribute processing pipeline
│   ├── extract_personalized_attributes.py
│   ├── filter_personalized_attributes.py
│   ├── merge_tree.py
│   ├── check_leaves.py
│   ├── convert_to_X.Y.Z.py
│   └── README.md              # Detailed documentation
│
└── data/                      # Data files
    ├── attributes_merged.json
    ├── attribute_embeddings.pkl
    └── occupations_english.json
```

## 🎯 Features

### User Profile Generator
- Generate realistic demographic information (age, gender, location, occupation)
- Create psychological profiles (values, attitudes, life stories)
- Select relevant attributes using vector-based semantic search
- Batch generation with customizable parameters

### Attribute Processing Pipeline
- Extract personalized attributes from natural language
- Validate and filter attributes using GPT-4
- Merge multiple attribute sources
- Check quality using semantic similarity
- Convert between hierarchical and flat formats

## 🌟 Resources

- **Project Homepage**: [https://thzva.github.io/deeppersona.github.io/](https://thzva.github.io/deeppersona.github.io/)
- **Dataset**: [DeepPersona Dataset on Hugging Face](https://huggingface.co/datasets/THzva/deeppersona_dataset)
- **Interactive Demo**: [Try DeepPersona on Hugging Face Spaces](https://huggingface.co/spaces/THzva/deeppersona-experience)

## 💻 Usage

### Generate User Profiles

```python
from generate_user_profile.select_attributes import generate_user_profile, get_selected_attributes

# Generate a user profile
user_profile = generate_user_profile()
selected_attributes = get_selected_attributes(user_profile, attribute_count=200)
```

### Process Attributes

```python
from process_attributes.extract_personalized_attributes import PersonalizedAttributeExtractor

# Extract attributes from a question
extractor = PersonalizedAttributeExtractor()
result = extractor.extract_attributes(
    question="What are some good restaurants nearby?",
    reason="User's location and food preferences affect recommendations"
)
```

## 📚 Documentation

- [User Profile Generator Documentation](./generate_user_profile/README.md)
- [Attribute Processing Pipeline Documentation](./process_attributes/README.md)

## 🎮 Try It Out

Want to see DeepPersona in action? Visit our **[Interactive Demo](https://huggingface.co/spaces/THzva/deeppersona-experience)** on Hugging Face Spaces to generate personalized user profiles instantly!

## 📊 Dataset

The **[DeepPersona Dataset](https://huggingface.co/datasets/THzva/deeppersona_dataset)** is available on Hugging Face, containing thousands of diverse, realistic user profiles with rich attributes.

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Dependencies: `openai`, `sentence-transformers`, `scikit-learn`, `numpy`, `tqdm`, `geonamescache`

## 📧 Contact

zhouyufan365@gmail.com

## 🙏 Acknowledgments

- OpenAI for GPT API
- Sentence Transformers for embedding models
- GeoNames for geographic data

---

For detailed usage instructions, please refer to the README files in each component directory.
