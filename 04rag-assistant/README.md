# 04 — RAG Assistant: Zombie Survival Guide

A Retrieval-Augmented Generation system that answers questions about a zombie survival guide using local LLM inference. The pipeline loads a document, splits it into chunks, embeds them into a vector store, retrieves relevant context, and generates answers with TinyLlama.

---

## Architecture

```
Document (.txt)
       |
       v
+------------+    +------------+    +------------+    +------------+    +------------+
| Load       |--->| Split      |--->| Embed      |--->| Retrieve   |--->| Generate   |
| TextLoader |    | Recursive  |    | MiniLM     |    | Chroma     |    | TinyLlama  |
+------------+    +------------+    +------------+    +------------+    +------------+
```

---

## Tech Stack

| Component | Library |
|---|---|
| **Orchestration** | LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **LLM** | TinyLlama 1.1B Chat (local, CPU) |
| **Document Loading** | LangChain `TextLoader` |

---

## Project Structure

```
04rag-assistant/
├── main.py              # Entry point — run full pipeline
├── README.md
├── data/
│   └── sample.txt       # Zombie survival guide document
└── src/
    ├── __init__.py
    ├── step1_load.py     # Document loading
    ├── step2_split.py    # Text splitting
    ├── step3_embed.py    # Embedding & vector storage
    ├── step4_retrieve.py # Similarity search retrieval
    └── step5_generate.py # LLM answer generation
```

---

## How to Run

```bash
# Create and activate environment
conda create -n rag_env python=3.11 -y
conda activate rag_env

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python 04rag-assistant/main.py
```

> **Note:** The first run downloads the embedding model (~80 MB) and TinyLlama (~2 GB). Subsequent runs use cached models.

---

## Sample Output

```
============================================================
  RAG Assistant: Zombie Survival Guide
============================================================

Loading documents...
  Document loaded (1 document)

Splitting into chunks...
  1 document split into 6 chunks

Creating vector store...
  Stored 6 vectors

Loading LLM...
  Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
  Model loaded.

------------------------------------------------------------
--- Query 1/4 ---
Q: What types of zombies are there?
A: Walkers are slow undead with strong bite force. Runners are
   recently turned with speed and aggression. Screamers attract
   nearby threats with piercing shrieks. Brutes can smash weak
   barricades in seconds.
Sources: 3 chunks

------------------------------------------------------------
--- Query 2/4 ---
Q: What is the best weapon for close combat?
A: A sturdy crowbar is excellent — quiet, nearly unbreakable,
   good for prying and striking. Hatchets are compact and effective
   for controlled strikes.
Sources: 3 chunks

============================================================
Done. Total execution time: 42.3s
============================================================
```

---

## Lessons Learned

- **RAG architecture** — Separating retrieval from generation lets a small LLM answer domain-specific questions it was never trained on
- **Chunking strategy** — Chunk size and overlap directly affect retrieval quality; too small loses context, too large dilutes relevance
- **Embeddings** — Sentence-transformers map semantically similar text to nearby vectors, enabling meaning-based search rather than keyword matching
- **Prompt engineering** — The chat template format (`<|system|>`, `<|user|>`, `<|assistant|>`) and explicit grounding instructions ("answer based ONLY on reference materials") significantly reduce hallucination
- **Local LLM inference** — TinyLlama runs on CPU without API keys, making the pipeline fully self-contained and reproducible

---

## References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
