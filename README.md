## Novelty & Innovation

- Innovation 1 (Tier-1 Fast Reject): SBERT semantic cache layer with sub-10ms cosine-similarity matching against vector database
- Innovation 2 (Sliding-Window Inference): Single fine-tuned DistilBERT model with overlapping chunk slicing for long prompts
- Innovation 3 (Decoupled Explainability): Asynchronous SHAP, LIME, and attention attribution running as background workers
- Innovation 4 (Stratified Dataset): Mathematically balanced dataset with rare edge cases like Base64 obfuscation and URL indirecting

## System Architecture

```
Tier 1: Fast Reject Layer (Regex + SBERT Semantic Cache)
Tier 2: Tokenization & Chunking (Sliding Window Tokenizer)
Tier 3: Inference Engine (DistilBERT evaluates chunks)
Tier 4: Decision Router (Block if ANY chunk malicious)
Tier 5: Async Logging & Explainability (background workers)
```

## Technical Specification

| Model           | Description                                  |
|----------------|----------------------------------------------|
| DistilBERT     | inference                                   |
| SBERT          | caching                                     |

## Dataset Description

- Change structure from CSV to JSON schema format  
- Total: 12,032 rows (60% Benign: 7,175 rows, 40% Attacks: 4,858 rows)  
- Attacks perfectly balanced 1:1 between Jailbreaks (2,429) and Prompt Injections (2,429)  
- Data sources: databricks/databricks-dolly-15k (Benign), JailbreakBench, lmsys/toxic-chat, PKU-SafeRLHF-QA (Jailbreaks), and deepset, neuralchemy, wambosec (Prompt Injections)  

## Core Features

- Tier-1 Fast Reject  
- Sliding-Window Inference  
- Decoupled Explainability  
