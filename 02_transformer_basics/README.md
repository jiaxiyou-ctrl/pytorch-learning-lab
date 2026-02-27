# 02 -- Transformer Basics

From-scratch implementations of scaled dot-product self-attention and a full transformer encoder block in PyTorch, plus two inference scripts that use pre-trained HuggingFace models (DistilBERT for sentiment analysis, GPT-2 for text generation).

---

## Architecture

SimpleSelfAttention -- single-head, no masking:

```
+-------+     +---------+     +---------+     +--------+
| Input |---->| Q, K, V |---->| Scaled  |---->| Output |
| x     |     | Linear  |     | Softmax |     |        |
+-------+     +---------+     +---------+     +--------+
```

TransformerBlock -- uses `nn.MultiheadAttention` under the hood:

```
+-------+     +-----------+     +-------+     +-----------+     +--------+
| Input |---->| Attention |---->| Add + |---->| FFN       |---->| Add +  |
|       |     |           |     | Norm  |     | (2-layer) |     | Norm   |
+-------+     +-----------+     +-------+     +-----------+     +--------+
                    ^               |               ^               |
                    +-- residual ---+               +-- residual ---+
```

---

## Quick Start

```bash
cd 02_transformer_basics

# run the from-scratch attention smoke test
python transformer_block.py

# sentiment analysis with DistilBERT
python sentiment_analysis.py

# text generation with GPT-2
python text_generation.py
```

---

## Files

```
02_transformer_basics/
  transformer_block.py    # SimpleSelfAttention + TransformerBlock from scratch
  sentiment_analysis.py   # DistilBERT sentiment classification via HF pipeline
  text_generation.py      # GPT-2 text generation via HF pipeline
```
