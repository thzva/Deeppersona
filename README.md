
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

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Dependencies: `openai`, `sentence-transformers`, `scikit-learn`, `numpy`, `tqdm`, `geonamescache`

## 🙏 Acknowledgments

- OpenAI for GPT API
- Sentence Transformers for embedding models
- GeoNames for geographic data
