# LexBel 🇧🇪

> **Intelligent Legal Search for Belgian Law** — Production RAG system that makes Belgian legal code accessible through natural language queries.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## What is LexBel?

LexBel allows citizens to ask juridic questions with the insurance that the answers are not hallucinated by the LLM.

### Demo


https://github.com/user-attachments/assets/b52fa5de-ecf9-4210-a7a2-564d15aa63ae


## Covered Legal Codes

the corpus of articles support the RAG system comprise 32 Belgian codes, collected in 2022


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional but recommended)
- Mistral API key ([get one here](https://console.mistral.ai/))

### Installation


```bash
# Clone the repository
git clone https://github.com/P-mir/LexBel.git
cd LexBel

# Set up your environment
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

# Launch with Docker Compose
docker-compose up
```

Visit `http://localhost:8501`


---

## Architecture

LexBel follows a modular, clean architecture designed for maintainability and extensibility:


### Tech Stack

**Core ML/AI:**
- `sentence-transformers` — Multilingual embeddings (paraphrase-multilingual-mpnet-base-v2 for local run on cpu)
- `FAISS` — High performance Vector similarity search with
- **MMR (Maximal Marginal Relevance)** for diversity-aware retrieval
  - **Hybrid search** combining dense vectors + BM25 lexical matching
- `LangChain` & `LlamaIndex` — RAG orchestration
- `Mistral AI` — LLM for answer generation (mistral-small-latest)

**Application Layer:**
- `Streamlit` — Interactive web interface
- `Pandas`
- `Plotly`

**Infrastructure & devops:**
- `Docker`
- `uv` — Fast (er than poetry) dependency management
- `pytest`

**Code Quality & Security:**

- pre-commit hooks
- Type hints with mypy
- Logging (`logs/ingestion.log`)
- Modularity & separations of concerns
- Bandit for vulnerability scanning


##  Usage Example

### Basic Question Answering

```python
from src.chains.langchain_qa import LangChainQA
from src.embeddings.cloud_embedder import CloudEmbedder
from src.vector_store.faiss_store import FAISSVectorStore

# Initialize components
embedder = CloudEmbedder()
vector_store = FAISSVectorStore.load("data/vector_store")
qa_chain = LangChainQA(vector_store, embedder)

# Ask a question
result = qa_chain.query(
    "Quelle est la durée de validité d'un certificat PEB ?",
    top_k=5
)

print(result["answer"])
print(f"Sources: {result['sources']}")
```



## Performance Metrics

The system tracks and displays:
- **Query processing time** (embedding + retrieval + generation)
- **Retrieval confidence scores** (cosine similarity)
- **Token usage** (input/output)
- **Source diversity** (across legal codes)

Analytics are saved to `data/metrics/` for continuous monitoring.



##  Development


### Adding New Legal Codes

1. Add CSV file to `data/` with columns: `id`, `article_number`, `article_text`, `code_name`
2. Run ingestion: `python scripts/ingest.py --input data/new_code.csv`
3. Vector store automatically updates

### Dataset Citation

If you use this dataset in your research, please cite:

> **Louis, A., & Spanakis, G.** (2022). *A Statutory Article Retrieval Dataset in French*.
> In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022),
> Dublin, Ireland (pp. 6789–6803). Association for Computational Linguistics.
> https://doi.org/10.18653/v1/2022.acl-long.468

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{louis2022statutory,
  title = {A Statutory Article Retrieval Dataset in French},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  month = may,
  year = {2022},
  address = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2022.acl-long.468/},
  doi = {10.18653/v1/2022.acl-long.468},
  pages = {6789–6803},
}
```

</details>

