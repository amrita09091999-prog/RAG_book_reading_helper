📚 **RAG Book Reading Assistant**

A Retrieval-Augmented Generation (RAG) system for question answering on novels, designed to help readers recall information as they read.

🔍**Overview**

Book_QA_RAG is an end-to-end Retrieval-Augmented Generation (RAG) system built to help avid readers recall characters, events, and contextual details while reading long novels.
Readers often lose track of information across chapters. This system allows users to upload a book (PDF) and ask natural-language questions, receiving:

Context-grounded answers
High-quality retrieved passages
Transparent evaluation of retrieval and generation quality

**Goal -** To build a general-purpose QA system for novels that:

1. Improves reading continuity
2. Helps users recall forgotten details
3. Provides transparent evaluation of answer quality

🧠 Orchestration Framework - LangChain

🏗️ **System Architecture** -  

      Book PDF
         ↓
      Chunking & Embeddings
         ↓
      Vector Store (FAISS - HNSW)
         ↓
      Hybrid Retrieval (BM25 + Semantic)
         ↓
      Cross-Encoder Re-ranking
         ↓
      LLM Answer Generation
         ↓
      LLM-based Evaluation



🔄 **RAG Pipeline Details**

1️⃣ **Chunking & Embeddings**
      Uses recursive chunking to break books into meaningful text segments
      Each chunk is embedded using HuggingFace sentence embeddings
      Embeddings are stored in FAISS (HNSW) vector store

**Metadata added:**
      Page number
      Chapter number

2️⃣ **Retrieval**

A **hybrid retrieval** strategy is used:

a) **Initial Retrieval**
      Keyword-based search: BM25
      Semantic search: embedding similarity
      Retrieves ~50 candidate documents

b) **Re-ranking**
      Uses HuggingFace Cross-Encoder
      Re-ranks query–document pairs based on semantic relevance
      Selects top 10 most relevant chunks

3️⃣ **Generation**

Uses an automated prompt template

**Inputs:** 
      User query
      Top-ranked retrieved documents
**Output:** Context-aware generated answer
**LLM used**: Qwen 8B Instruct

📊 **Evaluation**

Evaluation is implemented using the LLM-as-a-Judge paradigm.
LLM used for evaluation: Qwen 14B Instruct
Prompts are crafted using rubric-based prompting, instructing the LLM to evaluate outputs based on predefined logic.

**Evaluation Metrics**

a) **Answer Relevance**
      Inputs: Question + retrieved context
      Judges whether the context is sufficient to answer the question

b) **Faithfulness**
      Inputs: Context + generated answer
      Measures factual grounding of the answer

c) **Retrieval Relevance**
      Inputs: Query + retrieved documents
      Evaluates relevance of retrieved chunks


⚙️ **Backend**

      Built using FastAPI, exposing the following endpoints:
      Book Upload – Upload book PDFs for ingestion
      Question Answering – Submit queries and receive generated answers
      Evaluation – Returns structured evaluation feedback:
      Answer relevance, Faithfulness, Retrieval relevance

🖥️ **Frontend**

Built using Streamlit
      Features
      Upload book PDFs
      Ask questions interactively
      View generated answers
      Inspect retrieval quality and evaluation results
