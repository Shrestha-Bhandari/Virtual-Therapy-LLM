# Virtual Therapy Chatbot

## Overview
The **Virtual Therapy Chatbot** is a project aimed at creating an empathetic and responsive conversational AI tailored for mental health support. It combines modern machine learning techniques, state-of-the-art large language models (LLMs), and efficient document retrieval to provide meaningful and contextually aware interactions.

This project serves as a practical example for students and developers interested in exploring conversational AI, natural language processing, and neural network-based applications.

[**GitHub Repository**](https://github.com/Shrestha-Bhandari/Virtual-Therapy-LLM)

---

## Features
- **Empathy-Driven Conversations**: The chatbot generates warm, understanding, and contextually relevant responses.
- **Document Retrieval**: Leverages a vector database to incorporate additional knowledge for user queries.
- **Model Flexibility**: Supports integration of multiple LLMs, such as LLaMA-2 and Mistral-7B.
- **Easy Deployment**: Built using accessible frameworks like LangChain and Chainlit for real-time interaction.

---

## Why Use This Project?
This project is ideal for:
- **Students**: Learn the end-to-end process of building conversational AI, from preprocessing data to deploying an interactive bot.
- **Researchers**: Experiment with different language models and retrieval techniques for mental health or general conversational systems.


---

## How It Works
1. **Preprocessing**:
   - Text data from PDFs is split into manageable chunks for effective vectorization.
   - FAISS is used to build a searchable vector database.

2. **Model Integration**:
   - Includes configurations for LLaMA-2 and Mistral-7B models.
   - Fine-tuned settings ensure conversational quality and response diversity.

3. **Conversational Logic**:
   - A custom prompt ensures responses are empathetic and human-like.
   - User queries are processed through a RetrievalQA chain for contextual accuracy.

4. **Interface**:
   - Chainlit powers the user-friendly real-time chat interface.

---

## Getting Started

### Prerequisites
- **Python**: Version 3.8 or above.
- **Libraries**: Install dependencies listed in `requirements.txt`.
- **Hardware**: 
  - Minimum: CPU for basic functionality.
  - Recommended: GPU for faster response times.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shrestha-Bhandari/Virtual-Therapy-LLM.git
   cd Virtual-Therapy-LLM



