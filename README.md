#TÜRKÇE
# 💬 RAG Chatbot — AKBANK GenAI Bootcamp Projesi

![Chatbot Arayüzü](assets/chatbot2.png)

Bu proje, **Retrieval-Augmented Generation (RAG)** yaklaşımını kullanan bir yapay zekâ sohbet botudur.  
**Streamlit**, **LangChain**, **Pinecone** ve **Hugging Face** teknolojileri ile geliştirilmiştir.  
Chatbot, yüklenen şirket raporlarından (örneğin Tesla, NVIDIA, American Express, Apple) bilgi çekerek sorulara bağlama uygun, kısa ve net yanıtlar üretir.

---

## 🧠 Genel Bakış

Chatbot, kullanıcıdan gelen soruyu vektör formatına dönüştürüp **Pinecone** veritabanındaki benzer bölümleri bulur.  
Ardından bu bölümleri bir **LLM** (örnek: *Mistral-7B-Instruct*) ile birleştirerek anlamlı ve bağlama uygun yanıt üretir.

---

## 🧩 Kullanılan Teknolojiler

| Bileşen | Açıklama |
|----------|-----------|
| **LangChain** | Bilgi getirme (retrieval) ve yanıt üretim zinciri oluşturur |
| **Pinecone** | Vektör tabanlı veritabanı; embedding sorgularını hızla yürütür |
| **Hugging Face Hub** | Embedding modeli (BAAI/bge-m3) ve dil modeli (Mistral-7B) |
| **Streamlit** | Web arayüzü — chatbot etkileşimini sağlar |
| **Python-dotenv** | `.env` dosyasından gizli anahtarları yükler |
| **PyPDF / Text Splitters** | PDF belgelerini parçalara böler ve işler |

---

## ⚙️ Kurulum ve Çalıştırma

### 1️⃣ Depoyu klonla
```bash
git clone https://github.com/Talyaakuvvet/rag-chatbot.git
cd rag-chatbot

#ENGLISH
# 💬 RAG Chatbot — AKBANK GenAI Bootcamp Project

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, **LangChain**, **Pinecone**, and **Hugging Face**.  
It can answer questions based on the content of financial and sustainability reports (e.g., Tesla, NVIDIA, American Express, Apple).

---

## 🧠 Overview

The chatbot retrieves the most relevant document chunks from company reports stored in **Pinecone**, then uses a **Hugging Face LLM** (Mistral-7B-Instruct) to generate concise, context-aware answers.

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **LangChain** | Framework for chaining retrieval + generation logic |
| **Pinecone** | Vector database for storing and querying embeddings |
| **Hugging Face Hub** | Provides embeddings (BAAI/bge-m3) and LLM (Mistral-7B) |
| **Streamlit** | Web interface for interactive chatting |
| **Python-dotenv** | Loads environment variables from `.env` |
| **PyPDF / Text Splitters** | Extracts and processes document chunks |

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Talyaakuvvet/rag-chatbot.git
cd rag-chatbot

##📊 Example Questions
#Try these in your chatbot:
“What are Tesla’s sustainability goals for 2024?”
“Summarize NVIDIA’s 2024 financial highlights.”
“How does Apple report on carbon neutrality?”
“Which ESG initiatives are mentioned in American Express’s 2024 report?”


