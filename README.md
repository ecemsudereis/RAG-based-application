# RAG-based Application — SWE015 Final Project

A multi-turn Retrieval-Augmented Generation (RAG) assistant built over 13 lecture slide PDFs from the Introduction to Large Language Models course (SWE015) at Istinye University.

## Overview

The system ingests ~600 pages of lecture slides, splits them into 1,429 overlapping chunks, encodes them with all-MiniLM-L6-v2 (a 22M-parameter distilled Transformer), and indexes them in FAISS. At query time, the top-5 most semantically similar chunks are retrieved and passed as grounding context to Llama 3.3 70B via the Groq API. A Streamlit chat interface with conversation memory allows students to ask follow-up questions that reference prior turns.

## Tech Stack

- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Vector store: FAISS IndexFlatL2 (1,429 vectors)
- LLM: Llama 3.3 70B via Groq API
- UI: Streamlit (st.chat_message API) with last-4-turns memory
- Deployment: Google Colab + ngrok tunnel

## Results

- Recall@1: 1.000 on 17 hand-crafted evaluation questions
- MRR: 1.000
- Average retrieval latency: 21.7 ms
- End-to-end query latency: ~0.37 s
- Ablation vs multi-qa-MiniLM-L6-cos-v1: general-purpose MiniLM wins (1.000 vs 0.941 Recall@1)

## Files

- RAG_Assistant.ipynb — Main Colab notebook (full pipeline, evaluation, ablation)
- app.py — Streamlit multi-turn chat interface
- retrieval_eval.json — Recall@k and MRR detailed results
- evaluation_results.json — Per-question answers and sources
- model_comparison.csv — Embedding model comparison metrics
- recall_curve.png — Recall@k evaluation figure
- model_comparison.png — Embedding model comparison figure

## Author

Ecem Sude Reis — 210901004  
Istinye University, Department of Computer Engineering  
Instructor: Alper Öner  
