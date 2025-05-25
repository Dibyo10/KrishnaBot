# KrishnaBot Chatbot Backend

A backend API for **KrishnaBot** — an AI chatbot embodying the divine voice of Śrī Krishna, built with FastAPI and powered by OpenAI's GPT-4o and LangChain vector search (FAISS). This bot answers questions based on Bhagavad Gita and Śrīmad Bhāgavatam teachings, with tone-aware, playful, and poetic responses.

---

## Table of Contents

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Architecture Overview](#architecture-overview)  
- [Setup & Installation](#setup--installation)  
- [Environment Variables](#environment-variables)  
- [Running the Server](#running-the-server)  
- [API Endpoints](#api-endpoints)  
- [Core Components](#core-components)  
- [Tone Detection](#tone-detection)  
- [Vector Store (FAISS)](#vector-store-faiss)  
- [LangChain & LLM Usage](#langchain--llm-usage)  
- [Logging & Middleware](#logging--middleware)  
- [Extensibility](#extensibility)  
- [Troubleshooting](#troubleshooting)  

---

## Features

- **FastAPI** backend providing a RESTful `/ask` endpoint for chatting with KrishnaBot.
- **Tone detection** using TextBlob to detect user's sentiment and dynamically adjust response tone.
- **Retrieval-Augmented Generation (RAG)** with FAISS vector search on Bhagavad Gita indexed documents.
- **Fallback mechanism** to pure GPT-4o answer generation when no relevant context is found.
- **Persona-driven system prompt** that shapes KrishnaBot’s divine voice and playful style.
- Lightweight and stateless — no authentication or persistent sessions, focusing purely on chat logic.
- CORS enabled to allow frontend integration from any origin.

---

## Tech Stack

- **Python 3.10+**  
- **FastAPI** — Web API framework  
- **Uvicorn** — ASGI server for running FastAPI  
- **LangChain** — For vector store (FAISS) integration, prompt templates, and memory management  
- **OpenAI GPT-4o** — LLM for generating chatbot responses  
- **FAISS** — Vector similarity search over Bhagavad Gita embeddings  
- **TextBlob** — Sentiment analysis for tone detection  
- **python-dotenv** — Environment variable management  
- **CORS Middleware** — For cross-origin resource sharing  

---

## Architecture Overview

```plaintext
[User Request]
       ↓
FastAPI /ask endpoint
       ↓
Tone detection (TextBlob)
       ↓
Build system prompt with tone instructions
       ↓
Vector search in FAISS for relevant Bhagavad Gita docs
       ↓
If relevant docs found (score > threshold):
    → RAG with LangChain + GPT-4o (context + prompt)
Else fallback:
    → Pure GPT-4o answer generation with full chat history
       ↓
Response sent to user
