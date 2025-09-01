# E2H Medical Agent - Project Structure

## Directory Organization

```
cmrl/                          # Root project directory
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables
│
├── backend/                   # Backend Python code
│   ├── __init__.py
│   ├── medical_agent_app.py   # Main Flask application
│   ├── app.py                # Legacy Flask apps
│   ├── app_ollama.py         # (can be removed)
│   ├── app_with_memory.py    # (can be removed)
│   │
│   ├── agents/               # Medical AI agents
│   │   ├── __init__.py
│   │   ├── e2h_medical_agent.py      # Main E2H agent
│   │   └── medical_agent_core.py     # Core classes & enums
│   │
│   ├── retrieval/            # RAG and data loading
│   │   ├── __init__.py
│   │   ├── medrag_system.py          # MMed-RAG implementation
│   │   └── medical_data_loader.py    # Data loading utilities
│   │
│   ├── training/             # Training and optimization
│   │   ├── __init__.py
│   │   └── grpo_trainer.py           # GRPO training system
│   │
│   └── models/               # Model definitions (empty for now)
│       └── __init__.py
│
├── frontend/                 # Web interface
│   ├── templates/            # HTML templates
│   │   ├── index.html        # Basic E2H demo
│   │   ├── medical_agent.html # Medical agent interface
│   │   └── ollama_index.html  # Ollama interface (empty)
│   │
│   ├── static/               # JavaScript and CSS
│   │   ├── app.js            # Basic app JavaScript
│   │   ├── memory_app.js     # Memory-enabled app
│   │   ├── ollama_app.js     # Ollama app JavaScript
│   │   └── script.js         # E2H demo script
│   │
│   └── assets/               # Images, CSS, etc. (empty)
│
├── data/                     # Medical datasets
│   ├── medical_corpus/       # Text-based medical knowledge
│   │   ├── cardiology.json   # Cardiology documents
│   │   ├── radiology.json    # Radiology documents
│   │   ├── neurology.json    # Neurology documents
│   │   ├── pathology.json    # Pathology documents
│   │   └── general.json      # General medical knowledge
│   │
│   └── medvlm_data/          # Vision-Language medical data
│       ├── annotations.json  # Image question-answer pairs
│       └── images/           # Medical images
│           ├── radiology/    # X-rays, CT, MRI scans
│           ├── pathology/    # Tissue samples, microscopy
│           └── general/      # Other medical images
│
├── config/                   # Configuration and setup
│   └── setup_medical_data.py # Data setup script
│
└── docs/                     # Documentation
    ├── README.md             # Project documentation
    └── medical_data_requirements.md # Data format specs
```

## Key Components

### Backend Architecture
- **Agents**: Core medical AI logic with E2H curriculum learning
- **Retrieval**: MMed-RAG system for medical knowledge retrieval
- **Training**: GRPO and curriculum learning training systems
- **Models**: Future location for custom medical models

### Frontend Architecture
- **Templates**: HTML interfaces for different use cases
- **Static**: JavaScript, CSS, and client-side logic
- **Assets**: Static resources like images and styles

### Data Architecture
- **Medical Corpus**: Domain-specific medical text knowledge
- **MedVLM Data**: Medical images with questions and answers
- **Structured Format**: JSON-based for easy integration

## Usage

1. **Setup Data**: Run `python config/setup_medical_data.py`
2. **Add Medical Data**: Replace example files in `data/` with real medical content
3. **Run System**: `python main.py`
4. **Access Interface**: http://localhost:5000

## Integration Points

- Flask app loads from `backend/medical_agent_app.py`
- Data loaded from `data/medical_corpus/` and `data/medvlm_data/`
- Frontend templates in `frontend/templates/`
- All imports use relative paths within backend structure
